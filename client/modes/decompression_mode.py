"""
Decompression mode - Animated 3D eyeball with face-tracking pupil.

Renders a 3D eyeball model where the pupil follows detected faces in camera input,
with intermittent blinking animation. Each element (sclera, iris, pupil) is masked
with different colors.
"""

from .base_mode import BaseMode
from .website_mode import WebsiteMode
from PIL import Image
import time
import numpy as np
from threading import Thread, Lock
import cv2
import math
import random
import mediapipe as mp
import os
from piper import PiperVoice
import sounddevice as sd


class DecompressionMode(WebsiteMode):
    """Animated 3D eyeball mode with face-tracking pupil and blinking
    
    Renders a 3D eyeball where:
    - The pupil follows detected faces in camera input
    - The eye blinks intermittently
    - Each element (sclera, iris, pupil, highlights) has different colors
    
    Status System:
    The decompression mode supports multiple statuses with different visual behaviors:
    - 'eye': Default status showing the 3D eyeball with face-tracking pupil
    - 'people': Shows people outlines from camera as masks for Hydra visuals
    
    To switch statuses, use:
        mode.set_status('eye')    # Switch to eye status
        mode.set_status('people') # Switch to people status
    
    To get current status:
        current_status = mode.get_status()
    
    Future statuses can be added by implementing new _render_*_status() methods
    and updating the update() method to route to them.
    """
    
    def __init__(self, width=40, height=30):
        super().__init__(width, height)
        
        # Camera settings
        self.camera_enabled = True
        self.camera = None
        self.camera_thread = None
        self.camera_lock = Lock()
        self.camera_running = False
        
        # Face detection settings
        self.current_frame = None
        self.face_detection_lock = Lock()
        self.target_face_position = None  # Normalized position [0-1, 0-1] relative to camera frame
        self.detected_faces = []  # List of all detected faces
        self.current_face_index = 0  # Index of currently tracked face
        self.last_face_switch_time = time.time()
        self.face_switch_interval = 1.5  # Seconds between face switches when multiple faces detected
        
        # Mediapipe face detection
        self.face_detector = None
        self._mp_face_module = mp.solutions.face_detection
        
        # Mediapipe hand detection
        self.hand_detector = None
        self._mp_hands_module = mp.solutions.hands
        self.hand_wave_history = []  # Track hand x positions for wave detection
        self.wave_detection_threshold = 0.15  # Minimum movement to detect wave
        
        # Mediapipe selfie segmentation for people outlines
        self.selfie_segmenter = None
        self._mp_selfie_module = mp.solutions.selfie_segmentation
        self.people_mask = None  # Current people mask from camera
        self.people_mask_lock = Lock()
        
        # Robust state management system
        self.current_status = 'eye'  # Current status: 'eye', 'people', 'wave'
        self.status_lock = Lock()
        self.status_start_time = time.time()  # When current status started
        self.status_duration = None  # Duration for timed statuses (None = indefinite)
        self.status_substate = None  # Sub-state for complex statuses (e.g., 'hand_waving', 'hai_text' for wave)
        self.previous_status = None  # Track previous status for transitions
        self.status_transition_callbacks = {}  # Callbacks for status entry/exit
        self._wave_detected_flag = False  # Flag for wave detection
        
        # Eye 3D model parameters
        self.eye_center_x = width / 2
        self.eye_center_y = height / 2
        self.eye_radius = min(width, height) * 0.4  # Eye sphere radius
        
        # Pupil tracking (normalized to [-1, 1] range)
        self.pupil_offset_x = 0.0  # -1 to 1, left to right
        self.pupil_offset_y = 0.0  # -1 to 1, top to bottom
        self.pupil_smoothing = 0.4  # Smoothing factor for pupil movement (increased for more responsiveness)
        
        # Pupil animation (dilation/constriction)
        self.pupil_base_radius = 0.12  # Base normalized pupil radius (smaller - iris takes more space)
        self.pupil_current_radius = 0.12  # Current animated radius
        self.pupil_animation_phase = 0.0  # Animation phase for breathing/dilation
        self.pupil_animation_speed = 0.05  # Speed of pupil animation
        self.pupil_dilation_range = 0.05  # How much the pupil can dilate (reduced from 0.15)
        self.multiple_faces_dilation = 0.15  # Additional dilation when multiple faces detected (reduced from 0.35)
        
        # Blinking animation
        self.is_blinking = False
        self.blink_progress = 0.0  # 0.0 to 1.0
        self.blink_speed = 0.01  # Blink animation speed (much slower)
        self.blink_duration = 5.0  # Maximum duration of a blink in seconds
        self.blink_start_time = None  # When the current blink started
        self.last_blink_time = time.time()
        self.blink_interval_min = 20.0  # Minimum seconds between blinks (average ~30s)
        self.blink_interval_max = 40.0  # Maximum seconds between blinks
        self.next_blink_time = time.time() + random.uniform(self.blink_interval_min, self.blink_interval_max)
        
        # Eye element colors (RGB)
        self.sclera_color = (255, 255, 255)  # White sclera
        self.iris_color = (100, 150, 200)  # Blue iris
        self.pupil_color = (0, 0, 0)  # Black pupil
        self.highlight_color = (255, 255, 255)  # White highlight
        
        # 3D sphere parameters
        self.sphere_center = np.array([0, 0, 0])  # Center of sphere in 3D
        self.view_distance = 2.0  # Distance from viewer to sphere center
        
        # Hydra visual settings - single instance used for both iris and background
        self.hydra_url = "http://localhost:5173"
        
        # Single Hydra visual (shared for iris and background)
        self.hydra_frame = None
        self.hydra_frame_lock = Lock()
        
        # Expanding circles effect parameters (glitter effect)
        self.circle_count = 32  # Number of small circles (glitter particles, increased by 10)
        self.circle_speed = 0.05  # Speed of circle expansion (units per second)
        # Circle diameter: 3-5 pixels. eye_radius is ~12 pixels, so radius is 1.5-2.5px = 0.125-0.208 normalized
        self.circle_max_radius = 0.2  # Radius of each circle (normalized, ~2.4px diameter ~4.8px)
        self.circle_max_distance = 2.5  # Maximum distance circles can travel before respawning
        self.circles = []  # List of circle objects with position and spawn time
        self._initialize_circles()
        
        # Timing
        self.start_time = time.time()
        
        # TTS (Text-to-Speech) settings
        self.tts_voice = None
        self.tts_message = "welcome to decompression"
        self.last_tts_time = time.time()
        self.tts_interval_min = 30.0  # Minimum seconds between TTS messages
        self.tts_interval_max = 90.0  # Maximum seconds between TTS messages
        self.next_tts_time = time.time() + random.uniform(self.tts_interval_min, self.tts_interval_max)
        self.tts_playing = False
        self.tts_lock = Lock()
    
    def _initialize_circles(self):
        """Initialize glitter circles with random positions and spawn times"""
        self.circles = []
        for i in range(self.circle_count):
            # Random angle for each circle
            angle = random.uniform(0, 2 * math.pi)
            # Random spawn delay (staggered spawning)
            spawn_delay = random.uniform(0, self.circle_max_distance / self.circle_speed)
            self.circles.append({
                'angle': angle,
                'spawn_time': -spawn_delay,  # Negative so they spawn at different times
                'speed': self.circle_speed * random.uniform(0.8, 1.2)  # Slight speed variation
            })
    
    def setup(self, **kwargs):
        """Set up Hydra URL"""
        self.url = kwargs.get('url', self.hydra_url)
        super().setup(**kwargs)
    
    def _load_sketches(self):
        """Load sketches from sketches.txt and return the last one"""
        script_dir = os.path.dirname(os.path.abspath(__file__))  # client/modes/
        sketches_file = os.path.join(script_dir, "sketches.txt")
        
        if not os.path.exists(sketches_file):
            return None
        
        try:
            with open(sketches_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if lines:
                # Return the last sketch (most recent)
                return lines[-1]
            return None
        except Exception as e:
            print(f"Error loading sketches: {e}")
            return None
    
    def _build_hydra_url(self):
        """Build Hydra URL with sketch parameter if available"""
        url = self.hydra_url
        
        # Load sketch from sketches.txt
        sketch = self._load_sketches()
        
        if sketch:
            # Append sketch parameter to URL
            if '?' in url:
                url += f"&sketch={urllib.parse.quote(sketch)}"
            else:
                url += f"?sketch={urllib.parse.quote(sketch)}"
        
        return url
    
    def init(self):
        """Initialize camera, face detection, and Hydra visuals"""
        print("Initializing decompression mode (3D eyeball with Hydra visuals)...")
        
        # Set URL for WebsiteMode parent - load Hydra with sketch parameter if available
        self.url = self._build_hydra_url()
        
        # Initialize parent (WebsiteMode) for Hydra rendering
        # This will start the browser and screenshot thread
        super().init()
        
        # Hide UI elements after page loads
        if self.driver:
            try:
                time.sleep(2)  # Wait for Hydra to load
                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
            except Exception as e:
                print(f"Warning: Could not hide UI elements: {e}")
        
        # Initialize camera
        if self.camera_enabled:
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    print("Warning: Could not open camera")
                    self.camera_enabled = False
                else:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera_running = True
                    self.camera_thread = Thread(target=self._camera_loop, daemon=True)
                    self.camera_thread.start()
                    print("Camera initialized")
            except Exception as e:
                print(f"Warning: Could not initialize camera: {e}")
                self.camera_enabled = False
        
        # Initialize Mediapipe face detection
        try:
            self.face_detector = self._mp_face_module.FaceDetection(
                model_selection=0,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )
            print("Mediapipe face detector initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Mediapipe face detection: {e}")
            self.face_detector = None
        
        # Initialize Mediapipe hand detection
        try:
            self.hand_detector = self._mp_hands_module.Hands(
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("Mediapipe hand detector initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Mediapipe hand detection: {e}")
            self.hand_detector = None
        
        # Initialize Mediapipe selfie segmentation for people outlines
        try:
            self.selfie_segmenter = self._mp_selfie_module.SelfieSegmentation(
                model_selection=1  # 0 for general, 1 for landscape (better for full body)
            )
            print("Mediapipe selfie segmentation initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Mediapipe selfie segmentation: {e}")
            self.selfie_segmenter = None
        
        # Initialize Piper TTS
        self._init_tts()
        
        print("Decompression mode (3D eyeball) initialized")
    
    def _init_tts(self):
        """Initialize Piper TTS voice"""
        # Search for voice models in common locations
        search_dirs = [
            os.path.expanduser("~/.local/share/piper/voices"),
            os.path.join(os.path.dirname(__file__), "..", "..", "voices"),
        ]
        
        model_path = None
        config_path = None
        
        # Search for .onnx files recursively
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            # Walk through directory tree to find .onnx files
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.onnx'):
                        model_path = os.path.join(root, file)
                        # Look for corresponding .json config file
                        # Try same name with .json extension
                        config_path = model_path + ".json"
                        if not os.path.exists(config_path):
                            # Try without .onnx extension
                            config_path = model_path.replace(".onnx", ".json")
                        if not os.path.exists(config_path):
                            # Try looking in same directory for any .json file with similar name
                            base_name = os.path.splitext(file)[0]
                            for json_file in files:
                                if json_file.endswith('.json') and base_name in json_file:
                                    config_path = os.path.join(root, json_file)
                                    break
                        break
                if model_path:
                    break
            if model_path:
                break
        
        if model_path and os.path.exists(model_path):
            try:
                if config_path and os.path.exists(config_path):
                    self.tts_voice = PiperVoice.load(model_path, config_path)
                    print(f"Piper TTS initialized with model: {model_path} and config: {config_path}")
                else:
                    # Try loading without explicit config (Piper may auto-detect)
                    self.tts_voice = PiperVoice.load(model_path)
                    print(f"Piper TTS initialized with model: {model_path} (auto-detected config)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Piper TTS model at {model_path}: {e}\n"
                    "Make sure you have installed piper-tts: pip install piper-tts"
                )
        else:
            # Provide helpful error message
            voices_dir = os.path.expanduser("~/.local/share/piper/voices")
            raise FileNotFoundError(
                f"Piper TTS model not found in {voices_dir}.\n"
                "Please download a voice model from https://github.com/rhasspy/piper-voices\n"
                f"and place the .onnx file (and .json config if available) in {voices_dir}/\n"
                "You can place it directly in the voices directory or in any subdirectory."
            )
    
    def _play_tts(self, text):
        """Generate and play TTS audio in a separate thread"""
        def play_audio():
            try:
                with self.tts_lock:
                    if self.tts_playing:
                        return  # Already playing
                    self.tts_playing = True
                
                # Synthesize speech - returns a generator of AudioChunk objects
                audio_generator = self.tts_voice.synthesize(text)
                
                # Consume the generator to get audio bytes
                audio_chunks = []
                for audio_chunk in audio_generator:
                    # AudioChunk objects have an audio_int16_bytes attribute
                    if hasattr(audio_chunk, 'audio_int16_bytes'):
                        audio_chunks.append(audio_chunk.audio_int16_bytes)
                    elif hasattr(audio_chunk, 'audio_bytes'):
                        audio_chunks.append(audio_chunk.audio_bytes)
                    elif isinstance(audio_chunk, bytes):
                        audio_chunks.append(audio_chunk)
                    else:
                        # Try to convert to bytes
                        audio_chunks.append(bytes(audio_chunk))
                
                # Combine all chunks into a single bytes object
                audio_data = b''.join(audio_chunks)
                
                # Get sample rate from voice config
                sample_rate = self.tts_voice.config.sample_rate if hasattr(self.tts_voice.config, 'sample_rate') else 22050
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Normalize to float32 [-1, 1] for sounddevice
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Play audio
                sd.play(audio_float, samplerate=sample_rate)
                sd.wait()  # Wait until playback is finished
                
            except Exception as e:
                print(f"Error playing TTS audio: {e}")
            finally:
                with self.tts_lock:
                    self.tts_playing = False
        
        # Play in a separate thread to avoid blocking
        tts_thread = Thread(target=play_audio, daemon=True)
        tts_thread.start()
    
    def _check_and_play_tts(self):
        """Check if it's time to play TTS and play it"""
        current_time = time.time()
        
        # Check if it's time to play TTS
        if current_time >= self.next_tts_time:
            if not self.tts_playing:
                self._play_tts(self.tts_message)
                # Schedule next TTS
                self.next_tts_time = current_time + random.uniform(
                    self.tts_interval_min, self.tts_interval_max
                )
    
    def _camera_loop(self):
        """Background thread for camera capture and face detection"""
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self.camera_lock:
                        self.current_frame = frame_rgb
                    
                    # Detect faces
                    if self.face_detector:
                        self._detect_faces(frame_rgb)
                    
                    # Detect people outlines for people status
                    if self.selfie_segmenter:
                        self._detect_people_mask(frame_rgb)
                    
                    # Detect hands for wave gesture (only if not already in wave status)
                    if self.hand_detector:
                        current_status = self.get_status()
                        if current_status != 'wave':
                            self._detect_hand_wave(frame_rgb)
            except Exception as e:
                print(f"Camera error: {e}")
                break
    
    def _detect_faces(self, frame):
        """Detect faces in frame and update target position"""
        try:
            results = self.face_detector.process(frame)
            
            if results.detections:
                # Store all detected faces with their positions and sizes
                faces = []
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    face_center_x = bbox.xmin + bbox.width / 2
                    face_center_y = bbox.ymin + bbox.height / 2
                    face_size = bbox.width * bbox.height  # Area as size metric
                    
                    # Convert to normalized position relative to center
                    # Invert horizontal motion: multiply x by -1
                    normalized_x = -(face_center_x - 0.5) * 2.0  # -1 to 1, INVERTED
                    normalized_y = (face_center_y - 0.5) * 2.0  # -1 to 1
                    
                    faces.append({
                        'position': (normalized_x, normalized_y),
                        'size': face_size,
                        'bbox': bbox
                    })
                
                # Sort by size (largest first)
                faces.sort(key=lambda f: f['size'], reverse=True)
                
                with self.face_detection_lock:
                    self.detected_faces = faces
                    
                    # Ensure current_face_index is valid
                    if len(faces) > 0:
                        # Clamp current_face_index to valid range
                        if self.current_face_index >= len(faces):
                            self.current_face_index = 0
                        
                        # If multiple faces, switch between them intermittently
                        if len(faces) > 1:
                            current_time = time.time()
                            if current_time - self.last_face_switch_time >= self.face_switch_interval:
                                # Switch to next face
                                self.current_face_index = (self.current_face_index + 1) % len(faces)
                                self.last_face_switch_time = current_time
                        
                        # Use current face index
                        self.target_face_position = faces[self.current_face_index]['position']
                    else:
                        self.target_face_position = None
                        self.current_face_index = 0
            else:
                # No face detected, slowly return to center
                with self.face_detection_lock:
                    self.detected_faces = []
                    self.current_face_index = 0
                    if self.target_face_position:
                        # Gradually move toward center
                        self.target_face_position = (
                            self.target_face_position[0] * 0.95,
                            self.target_face_position[1] * 0.95
                        )
        except Exception as e:
            print(f"Face detection error: {e}")
    
    def _detect_people_mask(self, frame):
        """Detect people outlines in frame and create mask"""
        try:
            results = self.selfie_segmenter.process(frame)
            
            if results.segmentation_mask is not None:
                # Get segmentation mask (values 0.0 to 1.0)
                mask = results.segmentation_mask
                
                # Resize mask to match display dimensions
                mask_resized = cv2.resize(mask, (self.width, self.height))
                
                # Store the mask
                with self.people_mask_lock:
                    self.people_mask = mask_resized
        except Exception as e:
            print(f"People mask detection error: {e}")
    
    def set_status(self, status, duration=None, substate=None):
        """Set the current status of decompression mode with robust state management
        
        Args:
            status: String status name ('eye', 'people', 'wave', etc.)
            duration: Optional duration in seconds (None = indefinite)
            substate: Optional sub-state for complex statuses (e.g., 'hand_waving', 'hai_text' for wave)
        """
        with self.status_lock:
            # Exit previous status
            if self.current_status != status:
                self._exit_status(self.current_status)
                self.previous_status = self.current_status
            
            # Set new status
            self.current_status = status
            self.status_start_time = time.time()
            self.status_duration = duration
            self.status_substate = substate
            
            # Enter new status
            self._enter_status(status, substate)
            
            print(f"Decompression mode status changed to: {status}" + 
                  (f" (substate: {substate})" if substate else "") +
                  (f" (duration: {duration}s)" if duration else ""))
    
    def get_status(self):
        """Get the current status of decompression mode"""
        with self.status_lock:
            return self.current_status
    
    def get_status_info(self):
        """Get detailed status information"""
        with self.status_lock:
            elapsed = time.time() - self.status_start_time
            remaining = None
            if self.status_duration:
                remaining = max(0, self.status_duration - elapsed)
            return {
                'status': self.current_status,
                'substate': self.status_substate,
                'elapsed': elapsed,
                'remaining': remaining,
                'previous': self.previous_status
            }
    
    def _enter_status(self, status, substate=None):
        """Handle status entry logic"""
        if status == 'wave':
            # Initialize wave animation
            self.status_substate = substate or 'hand_waving'
            self.status_start_time = time.time()
            # Wave has two phases: hand_waving (2s) then hai_text (3s)
            self.status_duration = 2.0 if self.status_substate == 'hand_waving' else 3.0
        elif status == 'eye':
            # Reset any wave-specific state
            self.status_substate = None
        elif status == 'people':
            # Reset any wave-specific state
            self.status_substate = None
    
    def _exit_status(self, status):
        """Handle status exit logic"""
        if status == 'wave':
            # Clean up wave animation state
            self.status_substate = None
    
    def _update_status_transitions(self):
        """Update status transitions and timeouts"""
        current_time = time.time()
        wave_detected = False
        next_status = None
        
        # Check status transitions (need to release lock before calling set_status)
        with self.status_lock:
            elapsed = current_time - self.status_start_time
            
            # Handle timed statuses
            if self.status_duration and elapsed >= self.status_duration:
                if self.current_status == 'wave':
                    # Transition wave sub-states
                    if self.status_substate == 'hand_waving':
                        # Move to hai_text phase
                        self.status_substate = 'hai_text'
                        self.status_start_time = current_time
                        self.status_duration = 3.0
                        print("Wave animation: transitioning to 'hai' text")
                    elif self.status_substate == 'hai_text':
                        # Wave animation complete, return to previous status or default to eye
                        next_status = self.previous_status if self.previous_status else 'eye'
                        print(f"Wave animation complete, returning to: {next_status}")
                else:
                    # Other timed statuses - return to default
                    next_status = 'eye'
            
            # Handle automatic status transitions based on conditions
            # Wave detection triggers wave status
            if self.current_status != 'wave' and self.hand_detector:
                # Check for wave in hand detection (will be set by _detect_hand_wave)
                if self._wave_detected_flag:
                    self._wave_detected_flag = False
                    wave_detected = True
        
        # Apply status changes outside the lock to avoid deadlock
        if next_status:
            self.set_status(next_status)
        elif wave_detected:
            self.set_status('wave', duration=5.0, substate='hand_waving')
    
    def _detect_hand_wave(self, frame):
        """Detect hand waving gesture"""
        try:
            # Convert to RGB if needed (Mediapipe needs RGB)
            results = self.hand_detector.process(frame)
            
            if results.multi_hand_landmarks:
                # Get first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get wrist x position (landmark 0)
                wrist_x = hand_landmarks.landmark[0].x
                
                # Add to history
                self.hand_wave_history.append(wrist_x)
                
                # Keep only last 15 frames (~0.5 seconds at 30fps)
                if len(self.hand_wave_history) > 15:
                    self.hand_wave_history.pop(0)
                
                # Detect wave: check for left-right-left or right-left-right motion
                if len(self.hand_wave_history) >= 15:
                    # Calculate movement range
                    min_x = min(self.hand_wave_history)
                    max_x = max(self.hand_wave_history)
                    movement_range = max_x - min_x
                    
                    # Check for significant movement (wave)
                    if movement_range > self.wave_detection_threshold:
                        # Detect oscillation (wave pattern)
                        # Count direction changes
                        direction_changes = 0
                        for i in range(1, len(self.hand_wave_history) - 1):
                            # Check if direction changed
                            prev_diff = self.hand_wave_history[i] - self.hand_wave_history[i-1]
                            next_diff = self.hand_wave_history[i+1] - self.hand_wave_history[i]
                            if (prev_diff > 0 and next_diff < 0) or (prev_diff < 0 and next_diff > 0):
                                direction_changes += 1
                        
                        # If at least 2 direction changes, it's a wave
                        if direction_changes >= 2:
                            print("Wave detected!")
                            # Set flag for status system to pick up
                            self._wave_detected_flag = True
                            self.hand_wave_history = []  # Reset to avoid re-triggering
            else:
                # No hand detected - clear history
                self.hand_wave_history = []
                
        except Exception as e:
            print(f"Hand detection error: {e}")
    
    def _render_wave_status(self):
        """Render the wave status (hand wave and 'hai' text animation)"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        substate = status_info['substate']
        
        if substate == 'hand_waving':
            return self._render_waving_hand(elapsed)
        elif substate == 'hai_text':
            return self._render_hai_text(elapsed)
        else:
            # Default to hand_waving if substate not set
            return self._render_waving_hand(elapsed)
    
    def _render_waving_hand(self, elapsed):
        """Render clear waving hand for 40x30 screen"""
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Hand color (skin tone)
        hand_color = (255, 220, 177)
        
        # Wave animation: hand rocks left and right
        wave_phase = math.sin(elapsed * 5)  # Oscillate
        
        # Center position
        cx, cy = self.width // 2, self.height // 2
        
        # Draw a clearer hand shape (side view, palm forward)
        # Wrist/arm (bottom)
        for y in range(4):
            for x in range(-2, 3):
                px, py = cx + x, cy + 8 + y
                if 0 <= px < self.width and 0 <= py < self.height:
                    pixels[py, px] = hand_color
        
        # Palm (middle, wider)
        for y in range(8):
            for x in range(-3, 4):
                px, py = cx + x, cy + y
                if 0 <= px < self.width and 0 <= py < self.height:
                    pixels[py, px] = hand_color
        
        # Fingers (top) - 4 fingers with wave motion
        finger_positions = [-2, -1, 1, 2]  # Skip middle for spacing
        for i, fx in enumerate(finger_positions):
            # Each finger waves with phase offset
            finger_offset = int(wave_phase * 3 + math.sin(i * 1.5) * 2)
            # Finger length
            for fy in range(5):
                px = cx + fx + finger_offset
                py = cy - 1 - fy
                if 0 <= px < self.width and 0 <= py < self.height:
                    pixels[py, px] = hand_color
        
        # Thumb (side, shorter)
        thumb_offset = int(wave_phase * 2)
        for ty in range(3):
            px = cx - 4 + thumb_offset
            py = cy + 2 - ty
            if 0 <= px < self.width and 0 <= py < self.height:
                pixels[px, py] = hand_color
        
        return Image.fromarray(pixels)
    
    def _render_hai_text(self, elapsed):
        """Render wavy 'hai' text"""
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Text color (bright, friendly)
        text_color = (100, 200, 255)  # Cyan
        
        # Simple pixel font for "hai"
        # Center text
        cx, cy = self.width // 2, self.height // 2
        
        # Wave effect: vertical offset based on time and x position
        wave_frequency = 2.0
        wave_amplitude = 2.0
        
        # Letter patterns (5x7 simple pixel font)
        # H
        h_pattern = [
            [1,0,1],
            [1,0,1],
            [1,1,1],
            [1,0,1],
            [1,0,1]
        ]
        
        # A
        a_pattern = [
            [0,1,0],
            [1,0,1],
            [1,1,1],
            [1,0,1],
            [1,0,1]
        ]
        
        # I
        i_pattern = [
            [1,1,1],
            [0,1,0],
            [0,1,0],
            [0,1,0],
            [1,1,1]
        ]
        
        letters = [h_pattern, a_pattern, i_pattern]
        letter_spacing = 4
        
        # Calculate total width
        total_width = len(letters) * 3 + (len(letters) - 1) * letter_spacing
        start_x = cx - total_width // 2
        
        # Draw each letter
        for letter_idx, letter in enumerate(letters):
            letter_x = start_x + letter_idx * (3 + letter_spacing)
            
            for row_idx, row in enumerate(letter):
                for col_idx, pixel in enumerate(row):
                    if pixel == 1:
                        px = letter_x + col_idx
                        # Apply wave effect
                        wave_offset = int(wave_amplitude * math.sin(elapsed * wave_frequency + px * 0.5))
                        py = cy - 2 + row_idx + wave_offset
                        
                        if 0 <= px < self.width and 0 <= py < self.height:
                            pixels[py, px] = text_color
        
        return Image.fromarray(pixels)
    
    def _update_hydra_frame(self):
        """Update the cached frame from Hydra"""
        if self.driver:
            try:
                # Get frame from parent class (WebsiteMode)
                frame = self.get_frame()
                if frame:
                    with self.hydra_frame_lock:
                        self.hydra_frame = frame
            except Exception as e:
                print(f"Error updating Hydra frame: {e}")
    
    def _get_hydra_texture_color(self, nx, ny, normalize_brightness=False, target_brightness=0.5):
        """Get color from Hydra texture at normalized coordinates
        
        Args:
            nx, ny: Normalized coordinates (-1 to 1)
            normalize_brightness: If True, normalize brightness to target_brightness
            target_brightness: Target brightness level (0.0 to 1.0) when normalizing
        """
        with self.hydra_frame_lock:
            if self.hydra_frame is None:
                return None
            
            try:
                # Convert normalized coordinates (-1 to 1) to pixel coordinates
                hydra_width, hydra_height = self.hydra_frame.size
                
                # Map normalized coords to Hydra coords
                # Center is at (0, 0), so we map to center of frame
                hydra_x = int((nx + 1.0) / 2.0 * hydra_width)
                hydra_y = int((ny + 1.0) / 2.0 * hydra_height)
                
                # Clamp to valid range
                hydra_x = max(0, min(hydra_width - 1, hydra_x))
                hydra_y = max(0, min(hydra_height - 1, hydra_y))
                
                # Get pixel color
                pixel = self.hydra_frame.getpixel((hydra_x, hydra_y))
                
                # Handle both RGB and RGBA
                if len(pixel) == 4:
                    color = np.array(pixel[:3])  # RGB, ignore alpha
                else:
                    color = np.array(pixel)
                
                if normalize_brightness:
                    # Calculate current brightness (luminance)
                    # Using standard luminance formula: 0.299*R + 0.587*G + 0.114*B
                    current_brightness = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255.0
                    
                    if current_brightness > 0.001:  # Avoid division by zero
                        # Scale color to target brightness while preserving hue/saturation
                        scale_factor = target_brightness / current_brightness
                        color = color * scale_factor
                        color = np.clip(color, 0, 255)
                
                return tuple(color.astype(np.uint8))
                
            except Exception as e:
                return None
    
    def _get_background_gradient(self, nx, ny, dist_from_center):
        """Calculate gradient mask for background visual"""
        # Create gradient that fades out near the eye
        # dist_from_center is normalized (0 to 1+)
        eye_edge = 1.0  # Edge of eye shape
        
        if dist_from_center <= eye_edge:
            # Inside eye - no background
            return 0.0
        
        # Outside eye - gradient from 0 at edge to maximum at far away
        # Add some space between eye and background
        gradient_start = eye_edge + 0.2  # Start gradient 0.2 units outside eye
        gradient_end = eye_edge + 1.5    # Full intensity 1.5 units outside
        gradient_max = 0.8  # Maximum gradient intensity (80% max brightness)
        
        if dist_from_center < gradient_start:
            return 0.0
        elif dist_from_center > gradient_end:
            return gradient_max
        else:
            # Linear gradient up to max
            gradient = (dist_from_center - gradient_start) / (gradient_end - gradient_start)
            return gradient * gradient_max
    
    def _get_circle_mask_value(self, nx, ny, dist_from_center, current_time):
        """Check if pixel is inside any translating circle and return mask value
        
        Args:
            nx, ny: Normalized coordinates (-1 to 1)
            dist_from_center: Distance from eye center (normalized)
            current_time: Current time for animation
            
        Returns:
            float: Maximum mask value (0.0 to 1.0) if inside any circle, None otherwise
        """
        eye_edge = 1.0
        
        # Only show circles outside the eye
        if dist_from_center <= eye_edge:
            return None
        
        # Calculate angle from center
        angle = math.atan2(ny, nx)
        
        max_mask_value = 0.0
        
        # Check each circle
        for circle in self.circles:
            # Calculate time since spawn
            time_since_spawn = current_time - circle['spawn_time']
            
            # Respawn if circle has traveled too far
            if time_since_spawn * circle['speed'] > self.circle_max_distance:
                # Respawn at center with new random angle
                circle['angle'] = random.uniform(0, 2 * math.pi)
                circle['spawn_time'] = current_time
                time_since_spawn = 0
            
            # Calculate circle's current distance from eye edge (translation distance)
            circle_distance = time_since_spawn * circle['speed']
            
            # Skip if circle hasn't spawned yet or is too far
            if circle_distance < 0 or circle_distance > self.circle_max_distance:
                continue
            
            # Calculate circle's current position (center of circle)
            # Circle translates outward along its angle from the eye center
            # Start from eye edge and move outward
            circle_center_distance = eye_edge + circle_distance
            circle_center_x = math.cos(circle['angle']) * circle_center_distance
            circle_center_y = math.sin(circle['angle']) * circle_center_distance
            
            # Calculate distance from pixel to circle center
            # Use the same coordinate system (normalized coordinates)
            pixel_x = nx
            pixel_y = ny
            dx = pixel_x - circle_center_x
            dy = pixel_y - circle_center_y
            dist_from_circle_center = math.sqrt(dx * dx + dy * dy)
            
            # Check if pixel is within circle radius
            if dist_from_circle_center < self.circle_max_radius:
                # Calculate mask value (1.0 at center, fading at edges)
                mask_value = 1.0 - (dist_from_circle_center / self.circle_max_radius)
                # Smooth falloff
                mask_value = mask_value * mask_value  # Quadratic falloff
                max_mask_value = max(max_mask_value, mask_value)
        
        if max_mask_value > 0.0:
            return max_mask_value
        
        return None
    
    def _get_sunbeam_mask_value(self, nx, ny, dist_from_center, current_time):
        """Check if pixel is inside any rotating sunbeam and return mask value
        
        Args:
            nx, ny: Normalized coordinates (-1 to 1)
            dist_from_center: Distance from eye center (normalized)
            current_time: Current time for animation
            
        Returns:
            float: Maximum mask value (0.0 to 1.0) if inside a sunbeam, None otherwise
        """
        eye_edge = 1.0
        
        # Only show sunbeams outside the eye
        if dist_from_center <= eye_edge:
            return None
        
        # Don't show sunbeams too far away
        if dist_from_center > eye_edge + self.sunbeam_length:
            return None
        
        # Calculate angle from center
        angle = math.atan2(ny, nx)
        
        max_mask_value = 0.0
        
        # Check each sunbeam
        for i in range(self.sunbeam_count):
            # Calculate sunbeam angle (rotating)
            beam_base_angle = (2.0 * math.pi / self.sunbeam_count) * i
            beam_angle = beam_base_angle + current_time * self.sunbeam_rotation_speed
            
            # Normalize angle difference
            angle_diff = abs(angle - beam_angle)
            if angle_diff > math.pi:
                angle_diff = 2.0 * math.pi - angle_diff
            
            # Check if pixel is within sunbeam width
            if angle_diff < self.sunbeam_width / 2.0:
                # Calculate distance from eye edge
                distance_from_edge = dist_from_center - eye_edge
                
                # Check if pixel is within sunbeam length
                if distance_from_edge <= self.sunbeam_length:
                    # Calculate mask value based on distance along beam
                    # Fade out along the length of the beam
                    normalized_distance = distance_from_edge / self.sunbeam_length
                    length_falloff = 1.0 - (normalized_distance * self.sunbeam_falloff)
                    
                    # Also fade based on angular distance from center of beam
                    angular_falloff = 1.0 - (angle_diff / (self.sunbeam_width / 2.0))
                    
                    # Combine both falloffs
                    mask_value = length_falloff * angular_falloff
                    max_mask_value = max(max_mask_value, mask_value)
        
        if max_mask_value > 0.0:
            return max_mask_value
        
        return None
    
    def update(self):
        """Update and return current frame based on current status"""
        # Check if it's time to play TTS
        self._check_and_play_tts()
        
        # Update status transitions and timeouts
        self._update_status_transitions()
        
        # Get current status
        status = self.get_status()
        
        # Route to appropriate render method based on status
        if status == 'wave':
            return self._render_wave_status()
        elif status == 'people':
            return self._render_people_status()
        else:  # Default to 'eye' status
            return self._render_eye_status()
    
    def _render_eye_status(self):
        """Render the eye status (original 3D eyeball)"""
        # Update Hydra frame (raw visual from website)
        self._update_hydra_frame()
        
        # Update pupil position based on face detection
        self._update_pupil_position()
        
        # Update pupil animation (dilation/constriction)
        self._update_pupil_animation()
        
        # Update blinking animation
        self._update_blinking()
        
        # Render 3D eyeball
        img = self._render_eyeball()
        
        return img
    
    def _render_people_status(self):
        """Render the people status using people masks as Hydra mask source"""
        # Update Hydra frame (raw visual from website)
        self._update_hydra_frame()
        
        # Create image with black background
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Get people mask
        with self.people_mask_lock:
            people_mask = self.people_mask
        
        if people_mask is None:
            # No mask available yet, return black
            return img
        
        # Render Hydra visual masked by people outlines
        for y in range(self.height):
            for x in range(self.width):
                # Get mask value at this pixel (0.0 to 1.0)
                # Horizontally flip the mask (mirror effect)
                flipped_x = self.width - 1 - x
                mask_value = people_mask[y, flipped_x]
                
                if mask_value > 0.01:  # Only render where people are detected
                    # Convert pixel coordinates to normalized coordinates (-1 to 1)
                    nx = (x - self.width / 2) / (self.width / 2)
                    ny = (y - self.height / 2) / (self.height / 2)
                    
                    # Get Hydra color
                    hydra_color = self._get_hydra_texture_color(nx, ny, normalize_brightness=False)
                    
                    if hydra_color:
                        # Apply mask value to Hydra color
                        # Use mask value directly for brightness
                        effect_color = np.array(hydra_color, dtype=float) * mask_value
                        pixels[y, x] = tuple(np.clip(effect_color, 0, 255).astype(np.uint8))
                    else:
                        pixels[y, x] = (0, 0, 0)
                else:
                    # No person detected at this pixel
                    pixels[y, x] = (0, 0, 0)
        
        return Image.fromarray(pixels)
    
    def _update_pupil_position(self):
        """Update pupil position based on detected face"""
        with self.face_detection_lock:
            if self.target_face_position:
                target_x, target_y = self.target_face_position
                
                # Smooth interpolation toward target
                self.pupil_offset_x += (target_x - self.pupil_offset_x) * self.pupil_smoothing
                self.pupil_offset_y += (target_y - self.pupil_offset_y) * self.pupil_smoothing
            else:
                # No face detected - smoothly return to center
                self.pupil_offset_x += (0.0 - self.pupil_offset_x) * self.pupil_smoothing
                self.pupil_offset_y += (0.0 - self.pupil_offset_y) * self.pupil_smoothing
            
            # Clamp to valid range
            self.pupil_offset_x = max(-1.0, min(1.0, self.pupil_offset_x))
            self.pupil_offset_y = max(-1.0, min(1.0, self.pupil_offset_y))
    
    def _update_pupil_animation(self):
        """Update pupil dilation/constriction animation"""
        # Update animation phase
        self.pupil_animation_phase += self.pupil_animation_speed
        
        # Check if multiple faces are detected
        with self.face_detection_lock:
            multiple_faces = len(self.detected_faces) > 1
        
        # Calculate base radius with breathing animation
        # Use sine wave for smooth dilation/constriction
        breathing_factor = (math.sin(self.pupil_animation_phase) + 1) / 2  # 0 to 1
        breathing_variation = breathing_factor * self.pupil_dilation_range
        
        # Add extra dilation when multiple faces detected
        dilation_bonus = self.multiple_faces_dilation if multiple_faces else 0.0
        
        # Calculate final radius
        self.pupil_current_radius = self.pupil_base_radius + breathing_variation + dilation_bonus
        
        # Clamp to reasonable range (adjusted for smaller base size)
        self.pupil_current_radius = max(0.12, min(0.45, self.pupil_current_radius))
    
    def _update_blinking(self):
        """Update blinking animation - slower and can last up to 5 seconds"""
        current_time = time.time()
        
        if self.is_blinking:
            # Continue blink animation - slower progression
            self.blink_progress += self.blink_speed
            
            # Blink can last up to blink_duration seconds
            # Calculate progress based on time elapsed
            if self.blink_start_time is None:
                self.blink_start_time = current_time
            
            elapsed_time = current_time - self.blink_start_time
            time_based_progress = elapsed_time / self.blink_duration
            
            # Use the slower of the two progressions
            self.blink_progress = max(self.blink_progress, time_based_progress)
            
            if self.blink_progress >= 1.0:
                # Blink complete
                self.is_blinking = False
                self.blink_progress = 0.0
                self.blink_start_time = None
                # Schedule next blink
                self.next_blink_time = current_time + random.uniform(
                    self.blink_interval_min, self.blink_interval_max
                )
        else:
            # Check if it's time to blink
            if current_time >= self.next_blink_time:
                self.is_blinking = True
                self.blink_progress = 0.0
                self.blink_start_time = current_time
    
    def _render_eyeball(self):
        """Render the 3D eyeball model as a true 3D sphere"""
        # Create image with black background
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Calculate blink factor (0.0 = fully open, 1.0 = fully closed)
        blink_factor = 0.0
        if self.is_blinking:
            # Smooth blink: fast close, slow open
            if self.blink_progress < 0.3:
                # Closing phase
                blink_factor = self.blink_progress / 0.3
            else:
                # Opening phase
                blink_factor = 1.0 - ((self.blink_progress - 0.3) / 0.7)
        
        # Render eye for each pixel using true 3D sphere projection
        for y in range(self.height):
            for x in range(self.width):
                # Convert to normalized coordinates centered at eye center
                nx = (x - self.eye_center_x) / self.eye_radius
                ny = (y - self.eye_center_y) / self.eye_radius
                
                # Check if pixel is within eye bounds using almond/pointed shape
                eye_dist = self._get_eye_shape_distance(nx, ny)
                
                # Get eye color
                if eye_dist <= 1.0:
                    # Project 2D screen coordinates to 3D sphere surface
                    # This is a true 3D render - we're calculating the actual 3D position
                    color = self._get_eye_pixel_color_3d(nx, ny, blink_factor, eye_dist)
                    pixels[y, x] = color
                else:
                    # Outside eye - show Hydra visual inside circles with gradient mask
                    current_time = time.time() - self.start_time
                    
                    # Get circle mask (check for circles outside the eye)
                    circle_mask = self._get_circle_mask_value(nx, ny, eye_dist, current_time)
                    
                    # Only show visual where there are circles
                    if circle_mask is not None and circle_mask > 0.01:
                        # Get background gradient to apply on top
                        bg_gradient = self._get_background_gradient(nx, ny, eye_dist)
                        
                        # Get Hydra color WITHOUT brightness normalization
                        hydra_color = self._get_hydra_texture_color(nx, ny, normalize_brightness=False)
                        if hydra_color:
                            # Apply BOTH masks: circle mask AND gradient mask
                            # This should result in max 20% brightness (0.2 gradient max)
                            final_brightness = circle_mask * bg_gradient
                            effect_color = np.array(hydra_color, dtype=float) * final_brightness
                            pixels[y, x] = tuple(np.clip(effect_color, 0, 255).astype(np.uint8))
                        else:
                            pixels[y, x] = (0, 0, 0)
                    else:
                        # Not in a circle - show black
                        pixels[y, x] = (0, 0, 0)
        
        return Image.fromarray(pixels)
    
    def _get_eye_shape_distance(self, nx, ny):
        """Calculate distance from center using almond/pointed eye shape
        
        Creates an eye shape with pointed corners (almond shape) instead of a circle.
        The shape is wider horizontally and has sharp pointed ends at left and right.
        """
        # Create almond/pointed eye shape
        # Horizontal stretch factor (wider eye)
        horizontal_stretch = 1.4
        
        # Calculate angle from center (0 = right, pi = left)
        angle = math.atan2(ny, nx)
        
        # Create pointed ends at left and right (angles pi and 0)
        # Use a sharper function to create more pronounced points
        # abs(cos(angle)) is 1 at left/right (pi, 0) and 0 at top/bottom (pi/2, -pi/2)
        corner_sharpness = 0.5  # How sharp the corners are (0 = circle, higher = sharper)
        # Use cosine squared for sharper points
        corner_factor = 1.0 + corner_sharpness * (math.cos(angle) ** 2)
        
        # Apply horizontal stretch
        stretched_x = nx * horizontal_stretch
        stretched_y = ny
        
        # Calculate base distance on stretched ellipse
        base_dist = math.sqrt(stretched_x * stretched_x + stretched_y * stretched_y)
        
        # Apply corner sharpening (makes ends more pointed)
        # The corner_factor reduces the effective distance at the sides, creating points
        shaped_dist = base_dist / corner_factor
        
        return shaped_dist
    
    def _get_eye_pixel_color_3d(self, nx, ny, blink_factor, eye_shape_dist):
        """Get color for a pixel using true 3D sphere rendering
        
        Projects 2D screen coordinates to 3D sphere surface and calculates
        proper 3D distances for pupil/iris positioning.
        """
        # If eye is fully closed, show eyelid
        if blink_factor > 0.95:
            # Fully closed - show eyelid color with slight variation
            eyelid_color = np.array([50, 40, 30])
            # Add slight texture
            texture = math.sin(nx * 20) * math.sin(ny * 20) * 0.1
            color = eyelid_color * (1.0 + texture)
            return tuple(np.clip(color, 0, 255).astype(np.uint8))
        
        # Apply blink (squash vertically in 2D, then project to 3D)
        ny_3d = ny
        if blink_factor > 0:
            ny_blink = ny / (1.0 - blink_factor * 0.9)  # Squash vertically when blinking
            eye_dist_blink = self._get_eye_shape_distance(nx, ny_blink)
            if eye_dist_blink > 1.0:
                return (50, 40, 30)  # Eyelid color
            ny_3d = ny_blink
        
        # Project 2D screen coordinates to 3D sphere surface
        # First, normalize the 2D coordinates to account for eye shape
        # We need to map the almond shape to a sphere
        
        # Calculate the 2D distance from center (before shape distortion)
        dist_2d = math.sqrt(nx * nx + ny * ny)
        
        # Clamp to prevent domain errors
        dist_2d = min(0.999, max(0.0, dist_2d))
        
        # Calculate z coordinate on unit sphere
        # This gives us the actual 3D position
        z = math.sqrt(1.0 - dist_2d * dist_2d)
        
        # Create 3D point on sphere surface
        # Normalize nx, ny to get proper 3D coordinates
        if dist_2d > 0.001:
            scale = math.sqrt(1.0 - z * z) / dist_2d
            x_3d = nx * scale
            y_3d = ny_3d * scale
        else:
            x_3d = 0.0
            y_3d = 0.0
            z = 1.0
        
        # 3D point on sphere: (x_3d, y_3d, z)
        point_3d = np.array([x_3d, y_3d, z])
        
        # Calculate 3D normal vector (same as point on unit sphere)
        normal = point_3d / np.linalg.norm(point_3d)
        
        # Light direction (from top-left-front)
        light_dir = np.array([-0.5, -0.5, 0.7])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Calculate lighting intensity using 3D normal
        light_intensity = max(0.3, np.dot(normal, light_dir))
        
        # Calculate pupil position in 3D space
        # The pupil offset is in screen space, so we need to project it onto the sphere
        pupil_offset_x_3d = self.pupil_offset_x * 0.7
        pupil_offset_y_3d = self.pupil_offset_y * 0.7
        
        # Project pupil offset onto sphere surface
        # Create a 3D vector pointing in the direction of the offset
        pupil_dir_2d = np.array([pupil_offset_x_3d, pupil_offset_y_3d])
        pupil_dir_2d_len = np.linalg.norm(pupil_dir_2d)
        
        if pupil_dir_2d_len > 0.001:
            # Project onto sphere: create a 3D direction vector
            # We'll use spherical coordinates to place the pupil center
            pupil_center_2d = np.array([pupil_offset_x_3d, pupil_offset_y_3d])
            pupil_center_2d_len = min(0.8, np.linalg.norm(pupil_center_2d))  # Limit range
            
            # Calculate angle and distance
            pupil_angle = math.atan2(pupil_center_2d[1], pupil_center_2d[0])
            pupil_dist_2d = pupil_center_2d_len
            
            # Project to 3D sphere surface
            pupil_z = math.sqrt(1.0 - pupil_dist_2d * pupil_dist_2d)
            pupil_scale = math.sqrt(1.0 - pupil_z * pupil_z) / pupil_dist_2d if pupil_dist_2d > 0.001 else 0
            pupil_center_3d = np.array([
                math.cos(pupil_angle) * pupil_dist_2d * pupil_scale,
                math.sin(pupil_angle) * pupil_dist_2d * pupil_scale,
                pupil_z
            ])
        else:
            pupil_center_3d = np.array([0.0, 0.0, 1.0])  # Center of sphere (front)
        
        # Calculate 3D distance from current point to pupil center on sphere surface
        # Use great circle distance (arc length on sphere)
        dot_product = np.dot(point_3d, pupil_center_3d)
        dot_product = max(-1.0, min(1.0, dot_product))  # Clamp for acos
        sphere_distance = math.acos(dot_product)  # Angle in radians
        
        # Convert angle to normalized distance (0 to 1)
        # Maximum angle is pi (180 degrees), so normalize by pi
        normalized_sphere_dist = sphere_distance / math.pi
        
        # Determine which part of the eye this pixel belongs to
        # Use animated pupil radius
        pupil_radius = self.pupil_current_radius  # Animated normalized sphere distance
        iris_radius = 0.35  # Normalized sphere distance (reduced to show more sclera)
        
        # Determine pixel type based on 3D sphere distance
        if normalized_sphere_dist < pupil_radius:
            # Pupil - keep it black, don't apply lighting (or minimal)
            # Pupil should be truly black, not affected by lighting
            color = np.array(self.pupil_color)  # Pure black (0, 0, 0)
            # Add very subtle highlight at center only
            highlight = 0.05 if normalized_sphere_dist < pupil_radius * 0.3 else 0.0
            color = color + np.array([255, 255, 255]) * highlight
        elif normalized_sphere_dist < iris_radius:
            # Iris - use Hydra texture if available
            # Normalize brightness so all iris pixels have consistent brightness
            hydra_color = self._get_hydra_texture_color(nx, ny, normalize_brightness=True, target_brightness=0.4)
            
            if hydra_color:
                # Use Hydra texture color (already normalized to same brightness)
                base_color = np.array(hydra_color)
                
                # Transform ONLY precisely black pixels to white (only inside iris)
                # Check for exactly black pixels
                if base_color[0] == 0 and base_color[1] == 0 and base_color[2] == 0:
                    # Convert to white (will be normalized with others)
                    base_color = np.array([255, 255, 255])
                
                # Apply lighting to texture
                # The LED brightness will be applied later at the hardware level
                color = base_color * light_intensity
            else:
                # Fallback to default iris color
                base_color = np.array(self.iris_color)
                # Add radial pattern using angle
                angle = math.atan2(y_3d - pupil_center_3d[1], x_3d - pupil_center_3d[0])
                radial_pattern = (math.sin(angle * 8) + 1) / 2 * 0.2
                color = base_color * (light_intensity + radial_pattern)
            
            # Add highlight (reflection) - positioned in 3D space
            highlight_pos_3d = np.array([-0.25, -0.2, 0.9])
            highlight_dist_3d = math.acos(max(-1.0, min(1.0, np.dot(point_3d, highlight_pos_3d / np.linalg.norm(highlight_pos_3d)))))
            highlight_normalized = highlight_dist_3d / math.pi
            if highlight_normalized < 0.08:
                highlight_strength = (1.0 - highlight_normalized / 0.08) * 0.6
                color = color + np.array(self.highlight_color) * highlight_strength
        else:
            # Sclera (white part) - make it much brighter/whiter
            base_color = np.array(self.sclera_color)
            # Add veins pattern using 3D coordinates (very subtle)
            vein_pattern = math.sin(x_3d * 15) * math.sin(y_3d * 15) * 0.01
            # Much brighter - ensure minimum brightness even in shadows
            min_brightness = 0.7  # Minimum 70% brightness even in darkest areas
            effective_light = max(light_intensity, min_brightness)
            color = base_color * (effective_light * 1.3 + vein_pattern)
            # Ensure it stays very bright white (minimum 220, maximum 255)
            color = np.clip(color, 220, 255)
        
        # Clamp color values
        color = np.clip(color, 0, 255).astype(np.uint8)
        
        return tuple(color)
    
    def cleanup(self):
        """Clean up camera resources"""
        print("Cleaning up decompression mode...")
        
        # Stop camera thread
        self.camera_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Release mediapipe face detector
        if self.face_detector:
            try:
                self.face_detector.close()
            except AttributeError:
                pass
            self.face_detector = None
        
        # Release mediapipe hand detector
        if self.hand_detector:
            try:
                self.hand_detector.close()
            except AttributeError:
                pass
            self.hand_detector = None
        
        # Release mediapipe selfie segmenter
        if self.selfie_segmenter:
            try:
                self.selfie_segmenter.close()
            except AttributeError:
                pass
            self.selfie_segmenter = None
        
        print("Decompression mode cleaned up")
