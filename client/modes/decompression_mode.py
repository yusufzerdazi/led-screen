"""
Decompression mode - Animated 3D eyeball with face-tracking pupil.

Renders a 3D eyeball model where the pupil follows detected faces in camera input,
with intermittent blinking animation. Each element (sclera, iris, pupil) is masked
with different colors.
"""

from .base_mode import BaseMode
from .website_mode import WebsiteMode
# Import detectors and services using sys.path manipulation for cross-package imports
import sys
import os
# Add parent directory to path if not already there
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from detectors import GestureDetector, FaceDetector, MaskDetector
from services import VideoManager, AudioService
from text_scroller import TextScroller
from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np
from threading import Thread, Lock
from queue import Queue, Empty
import os
try:
    # Try to set CPU affinity for better performance on multi-core systems
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import cv2
import math
import random
import urllib.parse
import warnings

# Suppress OpenCV/FFmpeg H.264 decoder warnings
# These are harmless warnings that occur during frame seeking
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    # Older OpenCV versions might not have setLogLevel
    pass

# Context manager to suppress stderr (for FFmpeg warnings)
import contextlib
import sys

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr to hide FFmpeg warnings"""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr



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
        
        # Detection modules (initialized in init())
        self.face_detector_module: FaceDetector = None
        self.gesture_detector_module: GestureDetector = None
        self.mask_detector_module: MaskDetector = None
        self.video_manager: VideoManager = None
        
        # Console UI reference for service control (set by client.py)
        self._console_ui_ref = None
        
        # Background gesture detection queue and thread
        self.gesture_queue = Queue(maxsize=1)  # Single frame queue - always process latest frame only
        self.gesture_thread = None
        self.gesture_thread_running = False
        
        # Video configuration
        self.videos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'videos')
        self.video_config = {
            'hand_waving': {'filename': 'hand_wave.mp4', 'max_frames': 154},
            'thumbs_up': {'filename': 'thumbs_up.mp4'},
            'smile': {'filename': 'smile.mp4'},
        }
        # Store max_frames per video for frame limiting
        self.video_max_frames = {}
        for action_name, config in self.video_config.items():
            if isinstance(config, dict) and 'max_frames' in config:
                self.video_max_frames[action_name] = config['max_frames']
        
        # State management system
        self.current_status = 'eye'  # Current status: 'eye', 'people', 'wave', 'thumbs_up', 'smile'
        self.status_lock = Lock()
        self.status_start_time = time.time()
        self.status_duration = None
        self.status_substate = None
        self.previous_status = None
        self.status_transition_callbacks = {}
        
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
        
        # Sketch rotation settings
        self.sketch_rotation_interval = 600.0  # 10 minutes in seconds
        self.last_sketch_change_time = time.time()
        self.current_sketch = None
        self.available_sketches = []
        
        # Font settings - use KiwiSoda.ttf from client folder
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # client/
        font_path = os.path.join(script_dir, "KiwiSoda.ttf")
        
        # Fallback to slkscr.ttf if KiwiSoda not found, or system font as last resort
        if not os.path.exists(font_path):
            font_path = os.path.join(script_dir, "slkscr.ttf")
        if not os.path.exists(font_path):
            # System fallback
            try:
                font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
            except:
                font_path = None
        
        self.font_path = font_path
        self.font_size = 20  # Base font size for rendering (will be scaled down)
        
        # Initialize text scroller
        self.text_scroller = TextScroller(self.width, self.height, self.font_path, self.font_size)
        
        # Audio service (initialized in init())
        self.audio_service: AudioService = None
        self.tts_message = "welcome to decompression"
        self.tts_interval_min = 30.0
        self.tts_interval_max = 90.0
        self.next_tts_time = time.time() + random.uniform(self.tts_interval_min, self.tts_interval_max)
    
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
        """Set up Hydra URL and optional speech-to-text device"""
        self.url = kwargs.get('url', self.hydra_url)
        # Allow overriding microphone device index (defaults to 0)
        # Audio device index can be set via audio_service after initialization
        super().setup(**kwargs)
    
    def _load_all_sketches(self):
        """Load all sketches from sketches.txt"""
        script_dir = os.path.dirname(os.path.abspath(__file__))  # client/modes/
        sketches_file = os.path.join(script_dir, "sketches.txt")
        
        if not os.path.exists(sketches_file):
            return []
        
        try:
            with open(sketches_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            return lines
        except Exception as e:
            print(f"Error loading sketches: {e}")
            return []
    
    def _load_sketches(self):
        """Load sketches from sketches.txt and return a random one"""
        # Load all sketches if not already loaded
        if not self.available_sketches:
            self.available_sketches = self._load_all_sketches()
        
        if not self.available_sketches:
            return None
        
        # Return a random sketch
        return random.choice(self.available_sketches)
    
    def _build_hydra_url(self, sketch=None):
        """Build Hydra URL with sketch_id parameter if available
        
        Args:
            sketch: Optional sketch to use (if None, loads a random one)
        """
        url = self.hydra_url
        
        # Use provided sketch or load a random one
        if sketch is None:
            sketch = self._load_sketches()
        
        if sketch:
            # Append sketch_id parameter to URL
            if '?' in url:
                url += f"&sketch_id={urllib.parse.quote(sketch)}"
            else:
                url += f"?sketch_id={urllib.parse.quote(sketch)}"
        
        return url
    
    def _check_and_rotate_sketch(self):
        """Check if it's time to rotate to a new sketch and reload Hydra if needed"""
        current_time = time.time()
        time_since_change = current_time - self.last_sketch_change_time
        
        if time_since_change >= self.sketch_rotation_interval:
            # Time to rotate sketch
            if self.available_sketches:
                # Reload sketches in case new ones were added
                self.available_sketches = self._load_all_sketches()
                
                if self.available_sketches:
                    # Select a new random sketch (different from current)
                    new_sketch = random.choice(self.available_sketches)
                    # If same sketch selected, try again (max 10 attempts)
                    attempts = 0
                    while new_sketch == self.current_sketch and len(self.available_sketches) > 1 and attempts < 10:
                        new_sketch = random.choice(self.available_sketches)
                        attempts += 1
                    
                    if new_sketch != self.current_sketch:
                        print(f"[SKETCH] Rotating to new sketch: {new_sketch[:50]}...")
                        self.current_sketch = new_sketch
                        
                        # Reload Hydra with new sketch
                        if self.driver:
                            try:
                                new_url = self._build_hydra_url(sketch=new_sketch)
                                self.driver.get(new_url)
                                time.sleep(2)  # Wait for Hydra to load
                                
                                # Hide UI elements
                                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                                
                                print(f"[SKETCH] Hydra reloaded with new sketch")
                            except Exception as e:
                                print(f"[SKETCH] Error reloading Hydra with new sketch: {e}")
                    
                    self.last_sketch_change_time = current_time
                else:
                    # No sketches available, just update time
                    self.last_sketch_change_time = current_time
            else:
                # No sketches available, just update time
                self.last_sketch_change_time = current_time
    
    def init(self):
        """Initialize camera, face detection, and Hydra visuals"""
        print("Initializing decompression mode (3D eyeball with Hydra visuals)...")
        
        # Load all available sketches
        self.available_sketches = self._load_all_sketches()
        if self.available_sketches:
            print(f"[SKETCH] Loaded {len(self.available_sketches)} sketches")
            # Select initial random sketch
            self.current_sketch = random.choice(self.available_sketches)
            print(f"[SKETCH] Initial sketch: {self.current_sketch[:50]}...")
        else:
            print("[SKETCH] No sketches found in sketches.txt")
            self.current_sketch = None
        
        # Set URL for WebsiteMode parent - load Hydra with sketch parameter if available
        self.url = self._build_hydra_url(sketch=self.current_sketch)
        
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
                    # Set CPU affinity for camera thread (use core 1 if available)
                    if PSUTIL_AVAILABLE:
                        try:
                            p = psutil.Process(self.camera_thread.ident if self.camera_thread.ident else os.getpid())
                            # Pin camera thread to CPU core 1 (0-indexed, so core 1 = second core)
                            cpu_count = psutil.cpu_count()
                            if cpu_count > 1:
                                p.cpu_affinity([1 % cpu_count])
                        except (AttributeError, psutil.NoSuchProcess):
                            pass
                    print("Camera initialized")
            except Exception as e:
                print(f"Warning: Could not initialize camera: {e}")
                self.camera_enabled = False
        
        # Initialize detection modules
        try:
            self.face_detector_module = FaceDetector()
            print("Face detector initialized")
        except Exception as e:
            print(f"Warning: Could not initialize face detector: {e}")
            self.face_detector_module = None
        
        try:
            self.gesture_detector_module = GestureDetector()
            # Initialize AI model if available
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models',
                'gesture_recognizer.task'
            )
            self.gesture_detector_module.initialize_ai_model(model_path)
            print("Gesture detector initialized")
            
            # Start background gesture detection thread
            self.gesture_thread_running = True
            self.gesture_thread = Thread(target=self._gesture_detection_loop, daemon=True)
            self.gesture_thread.start()
            print("Gesture detection thread started")
        except Exception as e:
            print(f"Warning: Could not initialize gesture detector: {e}")
            self.gesture_detector_module = None
        
        # Initialize face landmarker for smile detection
        try:
            face_landmarker_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models',
                'face_landmarker.task'
            )
            if os.path.exists(face_landmarker_model_path):
                self.face_detector_module.initialize_face_landmarker(face_landmarker_model_path)
                print("Face landmarker initialized for smile detection")
            else:
                print(f"Warning: Face landmarker model not found at {face_landmarker_model_path}")
        except Exception as e:
            print(f"Warning: Could not initialize face landmarker: {e}")
        
        try:
            self.mask_detector_module = MaskDetector(self.width, self.height)
            print("Mask detector initialized")
        except Exception as e:
            print(f"Warning: Could not initialize mask detector: {e}")
            self.mask_detector_module = None
        
        # Initialize video manager
        try:
            print("Initializing video manager...")
            self.video_manager = VideoManager(self.videos_dir)
            # Extract filenames and max_frames from config (support both string and dict formats)
            video_filenames = {}
            max_frames_config = {}
            for action_name, config in self.video_config.items():
                if isinstance(config, dict):
                    video_filenames[action_name] = config['filename']
                    # Print max_frames if configured
                    if 'max_frames' in config:
                        max_frames_config[action_name] = config['max_frames']
                        print(f"  Video '{action_name}': {config['filename']} (max_frames: {config['max_frames']})")
                    else:
                        print(f"  Video '{action_name}': {config['filename']}")
                else:
                    video_filenames[action_name] = config
                    print(f"  Video '{action_name}': {config}")
            self.video_manager.load_videos(video_filenames, max_frames_config=max_frames_config)
            print("✓ Video manager initialized")
        except Exception as e:
            print(f"Warning: Could not initialize video manager: {e}")
            self.video_manager = None
        
        # Videos are loaded by VideoManager in init()
        # Detection modules are initialized above (face_detector_module, gesture_detector_module, mask_detector_module)
        
        # Initialize audio service
        try:
            self.audio_service = AudioService()
            self.audio_service.initialize_tts()
            self.audio_service.initialize_stt(
                input_device_index=None,
                voice_command_callback=self._handle_voice_command
            )
            print("Audio service initialized")
        except Exception as e:
            print(f"Warning: Could not initialize audio service: {e}")
            self.audio_service = None
        
        print("Decompression mode (3D eyeball) initialized")
    
    def _handle_voice_command(self, text):
        """Handle voice commands and change states accordingly"""
        if not text:
            return
        
        text_lower = text.lower().strip()
        print(f"[STT] Voice command received: {text}")
        
        # Parse commands and change states
        if "eye" in text_lower or "look" in text_lower:
            print("[STT] Switching to eye status")
            self.set_status('eye')
        elif "people" in text_lower or "person" in text_lower or "outline" in text_lower:
            print("[STT] Switching to people status")
            self.set_status('people')
        elif "wave" in text_lower or "hello" in text_lower or "hi" in text_lower:
            print("[STT] Triggering wave animation")
            self.set_status('wave', duration=5.0, substate='hand_waving')
    
    def _check_and_play_tts(self):
        """Check if it's time to play TTS and play it"""
        if not self.audio_service:
            return
        
        current_time = time.time()
        if current_time >= self.next_tts_time:
            self.audio_service.speak(self.tts_message)
            self.next_tts_time = current_time + random.uniform(
                self.tts_interval_min, self.tts_interval_max
            )
    
    def _camera_loop(self):
        """Background thread for camera capture and face detection"""
        import time
        
        # Performance optimizations
        frame_skip_counter = 0
        frame_skip_interval = 2  # Process every 2nd frame (30fps -> 15fps processing)
        last_detection_time = {}
        detection_intervals = {
            'face': 0.1,      # Detect faces every 100ms (~10fps)
            'people': 0.033,  # Detect people every 33ms (~30fps) - smooth updates
            'gesture': 0.2,   # Detect gestures every 200ms (~5fps) - reduced frequency to save resources
            'smile': 0.1,     # Detect smiles every 100ms (~10fps)
        }
        frame_timestamp_ms = 0  # For face landmarker video mode
        
        # Downscale factor for faster processing (Mediapipe works well with smaller images)
        process_width = 320  # Process at 320px width instead of full resolution
        process_height = None  # Will calculate to maintain aspect ratio
        
        # Even smaller frame for gesture detection (further reduces processing time)
        gesture_process_width = 160  # Process gestures at 160px width (4x smaller = much faster)
        gesture_process_height = None  # Will calculate to maintain aspect ratio
        
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Frame skipping for performance
                frame_skip_counter += 1
                if frame_skip_counter < frame_skip_interval:
                    continue
                frame_skip_counter = 0
                
                current_time = time.time()
                
                # Get current status once per frame
                current_status = self.get_status()
                
                # Convert BGR to RGB (needed for Mediapipe)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Store full resolution frame for display
                with self.camera_lock:
                    self.current_frame = frame_rgb
                    
                # Downscale for processing (much faster, Mediapipe works well with smaller images)
                original_height, original_width = frame_rgb.shape[:2]
                if process_height is None:
                    # Calculate height to maintain aspect ratio
                    process_height = int(original_height * process_width / original_width)
                
                # Only downscale if significantly larger than process size
                if original_width > process_width:
                    frame_small = cv2.resize(frame_rgb, (process_width, process_height), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_small = frame_rgb
                
                # Check service enable flags (from console UI if available)
                # Get enabled status from console UI if it exists
                # Default to enabled, but always check console UI if available
                # This ensures console UI control is respected
                face_enabled = True
                gesture_enabled = True
                segmentation_enabled = True
                
                # Check if console UI has service control flags - ALWAYS check if available
                # Console UI is the source of truth for service enabled/disabled state
                if hasattr(self, '_console_ui_ref') and self._console_ui_ref:
                    console_ui = self._console_ui_ref
                    if hasattr(console_ui, 'services') and console_ui.services:
                        # Get enabled status from console UI - this is the source of truth
                        # Always use the value from console UI, don't default
                        face_service = console_ui.services.get('Face Detection')
                        if face_service is not None:
                            face_enabled = face_service.enabled
                            # face_enabled is now set from console UI
                        
                        gesture_service = console_ui.services.get('Gesture Detection')
                        if gesture_service is not None:
                            gesture_enabled = gesture_service.enabled
                        
                        segmentation_service = console_ui.services.get('People Segmentation')
                        if segmentation_service is not None:
                            segmentation_enabled = segmentation_service.enabled
                
                # Smile uses face detector, so use face_enabled
                smile_enabled = face_enabled
                
                # Detect faces (only if enabled, in eye status, and throttle)
                if (face_enabled and self.face_detector_module and
                    current_status == 'eye' and
                    current_time - last_detection_time.get('face', 0) >= detection_intervals['face']):
                    self.face_detector_module.detect(frame_small)
                    last_detection_time['face'] = current_time
                
                # Detect people outlines (only if enabled, in people status, and throttle)
                if (segmentation_enabled and self.mask_detector_module and
                    current_status == 'people' and
                    current_time - last_detection_time.get('people', 0) >= detection_intervals['people']):
                    self.mask_detector_module.detect(frame_small)
                    last_detection_time['people'] = current_time
                
                # Queue frame for background gesture detection (non-blocking)
                # This prevents gesture detection from blocking the camera loop
                if (gesture_enabled and self.gesture_detector_module and
                    current_status not in ['wave', 'thumbs_up', 'smile']):
                    if (current_time - last_detection_time.get('gesture', 0) >= detection_intervals['gesture']):
                        # Downscale frame further for gesture detection (much faster processing)
                        if gesture_process_height is None:
                            gesture_process_height = int(original_height * gesture_process_width / original_width)
                        
                        if original_width > gesture_process_width:
                            frame_tiny = cv2.resize(frame_small, (gesture_process_width, gesture_process_height), 
                                                   interpolation=cv2.INTER_LINEAR)
                        else:
                            frame_tiny = frame_small
                        
                        # Queue frame for background processing (non-blocking)
                        # Use put_nowait with maxsize=1 to always keep only the latest frame
                        # This avoids expensive frame copying and memory overhead
                        try:
                            # Try to get and discard old frame if queue is full (keep only latest)
                            try:
                                self.gesture_queue.get_nowait()
                            except Empty:
                                pass
                            # Add new frame (non-blocking) - use tiny frame for faster processing
                            self.gesture_queue.put_nowait(frame_tiny)
                        except:
                            # Queue full - skip this frame (shouldn't happen with maxsize=1 and get_nowait)
                            pass
                        last_detection_time['gesture'] = current_time
                
                # Detect smiles (only if enabled, not already in a gesture status, and throttle)
                if (smile_enabled and self.face_detector_module and
                    current_status not in ['wave', 'thumbs_up', 'smile']):
                    if (current_time - last_detection_time.get('smile', 0) >= detection_intervals['smile']):
                        self.face_detector_module.detect_smile(frame_small, frame_timestamp_ms)
                        frame_timestamp_ms += int(detection_intervals['smile'] * 1000)  # Convert to ms
                        last_detection_time['smile'] = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                print(f"Camera error: {e}")
                break
    
    def _gesture_detection_loop(self):
        """Background thread for gesture detection - runs independently to avoid blocking camera loop"""
        import time
        
        while self.gesture_thread_running and self.gesture_detector_module:
            try:
                # Get frame from queue (blocking with timeout to allow checking thread_running)
                try:
                    frame = self.gesture_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Check if gesture detection is still enabled
                gesture_enabled = True
                if hasattr(self, '_console_ui_ref') and self._console_ui_ref:
                    console_ui = self._console_ui_ref
                    if hasattr(console_ui, 'services') and console_ui.services:
                        gesture_service = console_ui.services.get('Gesture Detection')
                        if gesture_service is not None:
                            gesture_enabled = gesture_service.enabled
                
                # Only process if enabled
                if gesture_enabled:
                    # Process gesture detection (this is CPU-intensive, runs in background)
                    # Prefer AI detection if available (more accurate), fallback to manual if needed
                    if self.gesture_detector_module.gesture_recognizer_available:
                        # Use AI gesture recognizer (covers both wave and thumbs up)
                        self.gesture_detector_module.detect_ai(frame)
                    else:
                        # Fallback to manual detection if AI not available
                        self.gesture_detector_module.detect_wave(frame)
                        self.gesture_detector_module.detect_thumbs_up(frame)
                
                # Mark task as done
                self.gesture_queue.task_done()
                
            except Exception as e:
                print(f"Gesture detection error: {e}")
                # Continue processing - don't break on errors
    
    def _update_face_position(self):
        """Update target face position from face detector module."""
        if not self.face_detector_module:
            return
        
        # Check if face detection is enabled via console UI
        face_enabled = True  # Default to enabled
        if hasattr(self, '_console_ui_ref') and self._console_ui_ref:
            console_ui = self._console_ui_ref
            if hasattr(console_ui, 'services'):
                face_service = console_ui.services.get('Face Detection')
                if face_service:
                    face_enabled = face_service.enabled
        
        # If face detection is disabled, clear position and return
        if not face_enabled:
            with self.face_detection_lock:
                self.target_face_position = None
                self.detected_faces = []
            return
        
        face_pos = self.face_detector_module.get_target_position()
        if face_pos:
            # Convert from [0-1, 0-1] to normalized [-1 to 1] with inverted x
            normalized_x = -(face_pos.x - 0.5) * 2.0  # Invert horizontal
            normalized_y = (face_pos.y - 0.5) * 2.0
            
            with self.face_detection_lock:
                self.target_face_position = (normalized_x, normalized_y)
                self.detected_faces = [{'position': (normalized_x, normalized_y)}] * self.face_detector_module.get_face_count()
        else:
            with self.face_detection_lock:
                if self.target_face_position:
                    # Gradually move toward center
                    self.target_face_position = (
                        self.target_face_position[0] * 0.95,
                        self.target_face_position[1] * 0.95
                    )
                self.detected_faces = []
    
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
            
            # Set duration based on substate
            if self.status_substate == 'hand_waving':
                # Use video duration if available, otherwise default to 2s
                if self.video_manager and self.video_manager.has_video('hand_waving'):
                    video_duration = self.video_manager.get_duration('hand_waving')
                    # Add a small buffer to ensure full playback
                    self.status_duration = video_duration + 0.5 if video_duration > 0 else 2.0
                else:
                    self.status_duration = 2.0
            else:
                # hai_text phase
                self.status_duration = 3.0
        elif status == 'thumbs_up':
            # Initialize thumbs up animation
            self.status_substate = None
            self.status_start_time = time.time()
            # Use video duration if available
            if self.video_manager and self.video_manager.has_video('thumbs_up'):
                video_duration = self.video_manager.get_duration('thumbs_up')
                # Add small buffer to ensure video plays fully
                self.status_duration = video_duration + 0.2 if video_duration > 0 else 2.0
            else:
                self.status_duration = 2.0
        elif status == 'smile':
            # Initialize smile animation
            self.status_substate = None
            self.status_start_time = time.time()
            # Use video duration if available, add time for scrolling text
            if self.video_manager and self.video_manager.has_video('smile'):
                video_duration = self.video_manager.get_duration('smile')
                # Calculate scroll time based on text width
                # Estimate: text width ~200px, scroll speed 20px/s = 10s, but we'll use 8s for safety
                scroll_time = 8.0
                self.status_duration = video_duration + scroll_time + 1.0 if video_duration > 0 else scroll_time + 1.0  # +1s buffer for full exit
            else:
                self.status_duration = 10.0  # Default: 2s video placeholder + 8s text
            # Play TTS
            if self.audio_service:
                self.audio_service.speak("i see you smiling")
        elif status == 'eye':
            # Reset any gesture-specific state
            self.status_substate = None
        elif status == 'people':
            # Reset any gesture-specific state
            self.status_substate = None
    
    def _exit_status(self, status):
        """Handle status exit logic"""
        if status == 'wave':
            # Clean up wave animation state
            self.status_substate = None
    
    def _update_status_transitions(self):
        """Update status transitions and timeouts"""
        current_time = time.time()
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
                        # Ensure enough time for text to fully scroll off (scroll_time ~2.5s + buffer)
                        if elapsed < 3.5:  # Give extra time for text to fully exit
                            self.status_duration = 3.5
                            next_status = None  # Don't transition yet
                        else:
                            next_status = self.previous_status if self.previous_status else 'eye'
                            print(f"Wave animation complete, returning to: {next_status}")
                else:
                    # Other timed statuses - return to default
                    next_status = 'eye'
            
            # Handle automatic status transitions based on conditions
            # (Detection flags are set by _detect_gestures_ai or _detect_hand_wave/_detect_thumbs_up)
        
        # Apply status changes outside the lock to avoid deadlock
        if next_status:
            self.set_status(next_status)
        else:
            # Check for detected gestures via gesture detector module
            gesture_triggered = False
            
            if self.gesture_detector_module:
                # Check thumbs up
                if self.gesture_detector_module.get_thumbs_up_detected():
                    duration = None
                    if self.video_manager and self.video_manager.has_video('thumbs_up'):
                        video_duration = self.video_manager.get_duration('thumbs_up')
                        if video_duration > 0:
                            duration = video_duration + 0.2
                    self.set_status('thumbs_up', duration=duration)
                    gesture_triggered = True
                
                # Check for wave gesture
                if self.gesture_detector_module.get_wave_detected():
                    duration = None
                    if self.video_manager and self.video_manager.has_video('hand_waving'):
                        video_duration = self.video_manager.get_duration('hand_waving')
                        if video_duration > 0:
                            # Calculate duration for hai_text scrolling
                            # Estimate: "HAI" text width ~100px, scroll speed 20px/s = 5s, add buffer
                            hai_scroll_time = 5.0 + 1.0  # Scroll time + buffer for full exit
                            duration = video_duration + 0.5 + hai_scroll_time
                    self.set_status('wave', duration=duration, substate='hand_waving')
                    gesture_triggered = True
            
            # Check for smile detection
            if self.face_detector_module and self.face_detector_module.get_smile_detected():
                duration = None
                if self.video_manager and self.video_manager.has_video('smile'):
                    video_duration = self.video_manager.get_duration('smile')
                    if video_duration > 0:
                        scroll_time = 8.0  # Time for text to scroll
                        duration = video_duration + scroll_time + 1.0  # +1s buffer for full exit
                self.set_status('smile', duration=duration)
                self.face_detector_module.reset_smile_detected()
                gesture_triggered = True
    
    def add_action_video(self, action_name, video_filename):
        """Add a new action video dynamically
        
        Args:
            action_name: Name of the action (e.g., 'clapping', 'pointing')
            video_filename: Filename of the video in the videos directory (e.g., 'clap.mp4')
        
        Returns:
            True if video was loaded successfully, False otherwise
        """
        # Add to config
        # Update video config (support both string and dict formats)
        self.video_config[action_name] = {'filename': video_filename}
        # Clear max_frames if it was set (new video might not need it)
        if action_name in self.video_max_frames:
            del self.video_max_frames[action_name]
        
        # Load via video manager
        if self.video_manager:
            return self.video_manager.load_video(action_name, video_filename)
        return False
    
    def _render_action_video(self, action_name, elapsed):
        """Render Hydra visual masked by video intensity
        
        Uses the video's brightness/intensity to control the brightness of the Hydra visual.
        Each pixel's brightness matches the corresponding pixel in the video.
        
        Args:
            action_name: Name of the action (must be in video_config and loaded_videos)
            elapsed: Elapsed time since the action started (for frame calculation)
        
        Returns:
            PIL Image of the masked visual, or black screen if video/visual not available
        """
        # Check if video manager is available
        if not self.video_manager or not self.video_manager.has_video(action_name):
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Get Hydra visual (already updated in main update() loop)
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        if hydra_frame is None:
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Limit video to max_frames if specified in config
        if action_name in self.video_max_frames:
            max_frames = self.video_max_frames[action_name]
            # Get fps from video manager (accessing internal structure, but necessary for frame limit)
            if action_name in self.video_manager._videos:
                fps = self.video_manager._videos[action_name].get('fps', 30.0)
                max_elapsed = max_frames / fps if fps > 0 else elapsed
                elapsed = min(elapsed, max_elapsed)
        
        # Get video frame from video manager
        video_frame_rgb = self.video_manager.get_frame(action_name, elapsed)
        if video_frame_rgb is None:
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Get video dimensions
        vid_height, vid_width = video_frame_rgb.shape[:2]
        target_width, target_height = self.width, self.height
        
        # Use "fit" mode (letterbox/pillarbox) for smile video, "fill" mode (crop) for others
        if action_name == 'smile':
            # Fit mode: fit entire video, add letterbox/pillarbox
            scale_x = target_width / vid_width
            scale_y = target_height / vid_height
            scale = min(scale_x, scale_y)  # Use min to fit entire video (no cropping)
            
            # Resize video frame maintaining aspect ratio
            new_width = int(vid_width * scale)
            new_height = int(vid_height * scale)
            video_resized = cv2.resize(video_frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Create black background and center the resized video (letterbox/pillarbox)
            video_cropped = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            offset_x = (target_width - new_width) // 2
            offset_y = (target_height - new_height) // 2
            video_cropped[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = video_resized
        else:
            # Fill mode: crop edges to fill screen (original behavior)
            scale_x = target_width / vid_width
            scale_y = target_height / vid_height
            scale = max(scale_x, scale_y)  # Use max to fill screen (crop edges)
            
            # Resize video frame maintaining aspect ratio
            new_width = int(vid_width * scale)
            new_height = int(vid_height * scale)
            video_resized = cv2.resize(video_frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Crop to target size (center crop)
            crop_x = (new_width - target_width) // 2
            crop_y = (new_height - target_height) // 2
            video_cropped = video_resized[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
        
        # Get intensity mask from video (grayscale brightness)
        video_gray = cv2.cvtColor(video_cropped, cv2.COLOR_RGB2GRAY)
        intensity_mask = video_gray.astype(np.float32) / 255.0  # 0-1 range
        
        # Set pixels below 10% brightness to 0 (completely black)
        intensity_mask[intensity_mask < 0.1] = 0.0
        
        # Resize Hydra frame to match target size
        hydra_resized = hydra_frame.resize((target_width, target_height), Image.LANCZOS)
        hydra_array = np.array(hydra_resized).astype(np.float32)
        
        # Apply intensity mask: multiply Hydra colors by video intensity
        # This makes the brightness of each pixel match the video's brightness
        # Pixels below 10% brightness will be completely black (0)
        intensity_mask_3d = np.stack([intensity_mask] * 3, axis=-1)
        masked_visual = (hydra_array * intensity_mask_3d).astype(np.uint8)
        
        # Convert to PIL Image
        return Image.fromarray(masked_visual)
    
    def _render_wave_status(self):
        """Render the wave status (hand wave and 'hai' text animation)"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        substate = status_info['substate']
        
        if substate == 'hand_waving':
            return self._render_waving_hand(elapsed)
        elif substate == 'hai_text':
            # Use scrolling text for "hai" - elapsed is time since hai_text substate started
            # The substate starts after the video, so elapsed is already the correct time for scrolling
            return self._render_scrolling_text("hai", elapsed, wobble_amount=0.05)
        else:
            # Default to hand_waving if substate not set
            return self._render_waving_hand(elapsed)
    
    def _render_waving_hand(self, elapsed):
        """Render hand wave video with intensity masking for 40x30 screen"""
        return self._render_action_video('hand_waving', elapsed)
    
    def _render_thumbs_up_status(self):
        """Render the thumbs up status"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        return self._render_action_video('thumbs_up', elapsed)
    
    def _render_smile_status(self):
        """Render the smile status with video and scrolling text"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        
        # Get video duration to determine when to show text
        video_duration = 0
        if self.video_manager and self.video_manager.has_video('smile'):
            video_duration = self.video_manager.get_duration('smile')
        
        # Show video for first part, then scrolling text
        if elapsed < video_duration:
            # Show video with intensity masking
            return self._render_action_video('smile', elapsed)
        else:
            # Show scrolling text after video
            text_elapsed = elapsed - video_duration
            return self._render_scrolling_text("i see you smiling", text_elapsed, wobble_amount=0.05)
    
    def _render_scrolling_text(self, text: str, elapsed: float, wobble_amount=1.0):
        """Render scrolling text that moves across the screen with wobbly effects.
        
        Args:
            text: Text to display
            elapsed: Elapsed time since text started (seconds)
            wobble_amount: Amount of wobbling effect (default: 1.0)
        """
        return self.text_scroller.render_scrolling_text(
            text=text,
            elapsed=elapsed,
            scroll_speed=20.0,
            wobble_amount=wobble_amount,
            font_size_scale=1.0,
            base_color=(255, 255, 255)  # White text
        )
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
        # Update Hydra frame once per update cycle (shared by all render methods)
        self._update_hydra_frame()
        
        # Update face position from detector module (for eye tracking)
        self._update_face_position()
        
        # Check if it's time to play TTS
        self._check_and_play_tts()
        
        # Check if it's time to rotate sketch
        self._check_and_rotate_sketch()
        
        # Update status transitions and timeouts
        self._update_status_transitions()
        
        # Get current status
        status = self.get_status()
        
        # Route to appropriate render method based on status
        if status == 'wave':
            return self._render_wave_status()
        elif status == 'thumbs_up':
            return self._render_thumbs_up_status()
        elif status == 'smile':
            return self._render_smile_status()
        elif status == 'people':
            return self._render_people_status()
        else:  # Default to 'eye' status
            return self._render_eye_status()
    
    def _render_eye_status(self):
        """Render the eye status (original 3D eyeball)"""
        # Hydra frame already updated in update() method
        
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
        # Hydra frame already updated in update() method
        
        # Create image with black background
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Get people mask from mask detector (thread-safe double buffering)
        if not self.mask_detector_module:
            return img
        
        people_mask = self.mask_detector_module.get_mask()
        if people_mask is None:
            return img
        
        # Render Hydra visual masked by people outlines
        # Use vectorized operations for better performance
        for y in range(self.height):
            for x in range(self.width):
                # Get mask value at this pixel (0.0 to 1.0)
                # Horizontally flip the mask (mirror effect)
                flipped_x = self.width - 1 - x
                mask_value = people_mask[y, flipped_x]
                
                # Convert pixel coordinates to normalized coordinates (-1 to 1)
                nx = (x - self.width / 2) / (self.width / 2)
                ny = (y - self.height / 2) / (self.height / 2)
                
                # Get Hydra color
                hydra_color = self._get_hydra_texture_color(nx, ny, normalize_brightness=False)
                
                # Apply mask value to Hydra color (use full mask range, no threshold)
                effect_color = np.array(hydra_color, dtype=float) * mask_value
                pixels[y, x] = tuple(np.clip(effect_color, 0, 255).astype(np.uint8))
        
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
        
        # Clean up detection modules
        if self.face_detector_module:
            self.face_detector_module.cleanup()
        # Stop gesture detection thread
        if self.gesture_thread_running:
            self.gesture_thread_running = False
            # Wait for thread to finish (with timeout)
            if self.gesture_thread and self.gesture_thread.is_alive():
                self.gesture_thread.join(timeout=1.0)
        
        if self.gesture_detector_module:
            self.gesture_detector_module.cleanup()
        if self.mask_detector_module:
            self.mask_detector_module.cleanup()
        if self.video_manager:
            self.video_manager.cleanup()
        if self.audio_service:
            self.audio_service.cleanup()
        
        # Clean up parent
        super().cleanup()
        
        print("Decompression mode cleaned up")
