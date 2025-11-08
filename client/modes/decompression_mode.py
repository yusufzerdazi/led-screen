"""
Decompression mode - Animated 3D eyeball with face-tracking pupil.

Renders a 3D eyeball model where the pupil follows detected faces in camera input,
with intermittent blinking animation. Each element (sclera, iris, pupil) is masked
with different colors.
"""

from .base_mode import BaseMode
from .website_mode import WebsiteMode
from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np
from threading import Thread, Lock
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
import mediapipe as mp
import urllib.parse
from piper import PiperVoice
import sounddevice as sd
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

# Speech-to-text import
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from speech_to_text import SpeechToText
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("Warning: speech_to_text not available - voice commands will be disabled")


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
        
        # Mediapipe hand detection and gesture recognition
        self.hand_detector = None
        self._mp_hands_module = mp.solutions.hands
        self.gesture_recognizer = None
        self._mp_gesture_module = None
        self._mp_image_class = None
        try:
            # Try to import Mediapipe Gesture Recognizer (newer API)
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self._mp_gesture_module = vision
            self._mp_gesture_base = python
            
            # Image class is in mediapipe.python._framework_bindings.image
            # ImageFormat is in mediapipe.python._framework_bindings.image_frame
            try:
                import mediapipe.python._framework_bindings.image as mp_image_module
                from mediapipe.python._framework_bindings import image_frame
                self._mp_image_class = mp_image_module.Image
                self._mp_image_format = image_frame.ImageFormat
            except (ImportError, AttributeError):
                # Will handle in detection method with fallbacks
                self._mp_image_class = None
                self._mp_image_format = None
            
            self.gesture_recognizer_available = True
        except (ImportError, AttributeError) as e:
            # Fallback to manual detection if gesture recognizer not available
            self.gesture_recognizer_available = False
            print(f"Mediapipe Gesture Recognizer not available: {e}")
            print("  Using manual landmark-based detection")
        
        # Fallback manual detection (if gesture recognizer not available)
        self.hand_wave_history = []  # Track hand x positions for wave detection
        self.wave_detection_threshold = 0.15  # Minimum movement to detect wave
        self._thumbs_up_detected_flag = False  # Flag for thumbs up detection
        self.thumbs_up_frames = 0  # Count consecutive frames with thumbs up
        self.thumbs_up_threshold = 10  # Frames needed to confirm thumbs up
        
        # Gesture detection flags
        self._wave_detected_flag = False
        self._gesture_frames = {}  # Track consecutive frames for each gesture
        self.gesture_threshold = 3  # Frames needed to confirm gesture (reduced for faster detection)
        self.gesture_decay_rate = 0.5  # Decay rate when gesture disappears (0.5 = lose 1 frame every 2 frames)
        
        # Gesture to status mapping configuration
        # Maps Mediapipe gesture names to status names and detection flags
        # To add a new gesture:
        #   1. Add video to video_config
        #   2. Add entry here: 'Gesture_Name': {'status': 'status_name', 'flag': '_flag_name'}
        #   3. Add flag initialization in __init__
        #   4. Add status rendering method if needed
        self.gesture_to_status_map = {
            'Thumb_Up': {
                'status': 'thumbs_up',
                'flag': '_thumbs_up_detected_flag',
                'requires_movement': False  # Simple gesture, no movement needed
            },
            'Open_Palm': {
                'status': 'wave',
                'flag': '_wave_detected_flag',
                'requires_movement': False  # Just use Open_Palm gesture, no movement needed
            },
            # Add more gestures here:
            # 'Victory': {
            #     'status': 'peace_sign',
            #     'flag': '_peace_detected_flag',
            #     'requires_movement': False
            # },
        }
        
        # Video loading system for action animations
        # This system allows you to easily add multiple videos for different actions.
        # 
        # Usage:
        # 1. Add videos to client/videos/ directory
        # 2. Add entries to video_config below: 'action_name': 'video_filename.mp4'
        # 3. Use _render_action_video('action_name', elapsed) to render any video
        # 4. Or call add_action_video('action_name', 'video.mp4') dynamically
        #
        # Example:
        #   self.video_config = {
        #       'hand_waving': 'hand_wave.mp4',
        #       'clapping': 'clap.mp4',
        #       'pointing': 'point.mp4',
        #   }
        #   Then render with: self._render_action_video('clapping', elapsed)
        #
        self.videos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'videos')
        
        # Map action names to video filenames
        # Add new actions by adding entries here: 'action_name': 'video_filename.mp4'
        self.video_config = {
            'hand_waving': 'hand_wave.mp4',
            'thumbs_up': 'thumbs_up.mp4',
            # Add more actions here as needed:
            # 'clapping': 'clap.mp4',
            # 'pointing': 'point.mp4',
        }
        
        # Dictionary to store loaded video objects and metadata
        # Structure: {action_name: {'capture': cv2.VideoCapture, 'fps': float, 'frame_count': int, 'duration': float, 'current_frame': int, 'last_elapsed': float, 'last_frame_time': float, 'cached_frame': numpy array, 'cached_frame_num': int}}
        self.loaded_videos = {}
        self.video_lock = Lock()  # Lock for video operations
        
        # Mediapipe selfie segmentation for people outlines
        self.selfie_segmenter = None
        self._mp_selfie_module = mp.solutions.selfie_segmentation
        self.people_mask = None  # Current people mask from camera
        self.people_mask_lock = Lock()
        
        # Robust state management system
        self.current_status = 'eye'  # Current status: 'eye', 'people', 'wave', 'thumbs_up'
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
        self._font_cache = None  # Cache the loaded font
        
        # TTS (Text-to-Speech) settings
        self.tts_voice = None
        self.tts_message = "welcome to decompression"
        self.last_tts_time = time.time()
        self.tts_interval_min = 30.0  # Minimum seconds between TTS messages
        self.tts_interval_max = 90.0  # Maximum seconds between TTS messages
        self.next_tts_time = time.time() + random.uniform(self.tts_interval_min, self.tts_interval_max)
        self.tts_playing = False
        self.tts_lock = Lock()
        
        # Speech-to-text settings
        self.speech_to_text = None
        self.stt_enabled = STT_AVAILABLE
        self.stt_input_device_index = 0  # Default to device 0
    
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
        if 'stt_device_index' in kwargs:
            self.stt_input_device_index = kwargs['stt_device_index']
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
        
        # Initialize Mediapipe Gesture Recognizer if available
        if self.gesture_recognizer_available and self._mp_gesture_module:
            try:
                # Path to gesture recognizer model file
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'models',
                    'gesture_recognizer.task'
                )
                
                if os.path.exists(model_path):
                    # Create base options with model path
                    base_options = self._mp_gesture_base.BaseOptions(
                        model_asset_path=model_path
                    )
                    
                    # Create gesture recognizer options
                    options = self._mp_gesture_module.GestureRecognizerOptions(
                        base_options=base_options,
                        running_mode=self._mp_gesture_module.RunningMode.IMAGE,
                        num_hands=1
                    )
                    
                    # Create the recognizer
                    self.gesture_recognizer = self._mp_gesture_module.GestureRecognizer.create_from_options(options)
                    
                    # Import Image class and ImageFormat for frame conversion
                    try:
                        import mediapipe.python._framework_bindings.image as mp_image_module
                        from mediapipe.python._framework_bindings import image_frame
                        self._mp_image_class = mp_image_module.Image
                        self._mp_image_format = image_frame.ImageFormat
                    except (ImportError, AttributeError):
                        # Will use fallback methods in detection
                        self._mp_image_class = None
                        self._mp_image_format = None
                    
                    print(f"✓ Mediapipe Gesture Recognizer initialized (model: {model_path})")
                else:
                    print(f"⚠ Gesture Recognizer model not found at: {model_path}")
                    print("  Falling back to manual landmark-based detection")
                    print("  See GESTURE_SETUP.md for instructions to download the model")
                    self.gesture_recognizer_available = False
            except Exception as e:
                print(f"Warning: Could not initialize Gesture Recognizer: {e}")
                print("  Falling back to manual landmark-based detection")
                self.gesture_recognizer_available = False
        
        # Initialize all action videos
        self._load_action_videos()
        
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
        
        # Initialize speech-to-text for voice commands
        self._init_speech_to_text(input_device_index=self.stt_input_device_index)
        
        print("Decompression mode (3D eyeball) initialized")
    
    def _init_speech_to_text(self, input_device_index=None):
        """Initialize speech-to-text for voice commands
        
        Args:
            input_device_index: Optional microphone device index (None = use default)
        """
        if not self.stt_enabled:
            print("[STT] Speech-to-text not available, voice commands disabled")
            print(f"[STT] STT_AVAILABLE={STT_AVAILABLE}")
            return
        
        try:
            print("[STT] Initializing speech-to-text...")
            # Use tiny model for fastest response and lowest CPU usage
            self.speech_to_text = SpeechToText(
                model_size="tiny", 
                language="en", 
                use_whisper=True,
                input_device_index=input_device_index
            )
            
            # Start listening with callback to handle voice commands
            success = self.speech_to_text.start_listening(on_text_callback=self._handle_voice_command)
            if success:
                print("[STT] Speech-to-text enabled - listening for voice commands")
            else:
                print("[STT] Failed to start speech-to-text")
                self.stt_enabled = False
                self.speech_to_text = None
        except Exception as e:
            print(f"[STT] Error initializing speech-to-text: {e}")
            import traceback
            traceback.print_exc()
            self.stt_enabled = False
            self.speech_to_text = None
    
    def _handle_voice_command(self, text):
        """Handle voice commands and change states accordingly"""
        if not text:
            return
        
        text_lower = text.lower().strip()
        print(f"[STT] Voice command received: {text}")
        
        # Parse commands and change states
        # Status commands
        if "eye" in text_lower or "look" in text_lower:
            print("[STT] Switching to eye status")
            self.set_status('eye')
        elif "people" in text_lower or "person" in text_lower or "outline" in text_lower:
            print("[STT] Switching to people status")
            self.set_status('people')
        elif "wave" in text_lower or "hello" in text_lower or "hi" in text_lower:
            print("[STT] Triggering wave animation")
            self.set_status('wave', duration=5.0, substate='hand_waving')
        
        # You can add more command parsing here
        # Examples:
        # - "blink" -> trigger blink animation
        # - "dilate" -> change pupil size
        # - "color" -> change colors
    
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
        import time
        
        # Performance optimizations
        frame_skip_counter = 0
        frame_skip_interval = 2  # Process every 2nd frame (30fps -> 15fps processing)
        last_detection_time = {}
        detection_intervals = {
            'face': 0.1,      # Detect faces every 100ms (~10fps)
            'people': 0.033,  # Detect people every 33ms (~30fps) - smooth updates
            'gesture': 0.05,  # Detect gestures every 50ms (~20fps) - needs to be responsive
        }
        
        # Downscale factor for faster processing (Mediapipe works well with smaller images)
        process_width = 320  # Process at 320px width instead of full resolution
        process_height = None  # Will calculate to maintain aspect ratio
        
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
                
                # Detect faces (only if in eye status, and throttle)
                if (current_status == 'eye' and
                    current_time - last_detection_time.get('face', 0) >= detection_intervals['face']):
                    self._detect_faces(frame_small)
                    last_detection_time['face'] = current_time
                
                # Detect people outlines (only if in people status, and throttle)
                if (current_status == 'people' and
                    current_time - last_detection_time.get('people', 0) >= detection_intervals['people']):
                    self._detect_people_mask(frame_small)
                    last_detection_time['people'] = current_time
                
                # Detect hands for gestures (only if not already in a gesture status, and throttle)
                if current_status not in ['wave', 'thumbs_up']:
                    if (current_time - last_detection_time.get('gesture', 0) >= detection_intervals['gesture']):
                        # Use AI gesture recognizer
                        self._detect_gestures_ai(frame_small)
                        # Also run manual detection in parallel
                        self._detect_hand_wave(frame_small)
                        self._detect_thumbs_up(frame_small)
                        last_detection_time['gesture'] = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)  # 1ms sleep
                
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
        results = self.selfie_segmenter.process(frame)
        
        if results.segmentation_mask is not None:
            # Get segmentation mask (values 0.0 to 1.0)
            mask = results.segmentation_mask
            
            # Resize mask to match display dimensions
            mask_resized = cv2.resize(mask, (self.width, self.height))
            
            # Store the mask (copy to avoid race conditions)
            with self.people_mask_lock:
                self.people_mask = mask_resized.copy()
    
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
                if 'hand_waving' in self.loaded_videos:
                    video_duration = self.loaded_videos['hand_waving']['duration']
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
            if 'thumbs_up' in self.loaded_videos:
                video_duration = self.loaded_videos['thumbs_up']['duration']
                # Add small buffer to ensure video plays fully
                self.status_duration = video_duration + 0.2 if video_duration > 0 else 2.0
            else:
                self.status_duration = 2.0
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
            # Unified gesture handling system - checks all configured gestures
            # This makes it easy to add new gestures - just add to gesture_to_status_map
            gesture_triggered = False
            
            # First, check AI-detected gestures (if AI recognizer is available)
            if self.gesture_recognizer_available:
                for gesture_name, gesture_config in self.gesture_to_status_map.items():
                    status_name = gesture_config['status']
                    flag_name = gesture_config['flag']
                    
                    # Check if this gesture was detected and we're not already in that status
                    if (self.current_status != status_name and 
                        hasattr(self, flag_name) and 
                        getattr(self, flag_name)):
                        # Reset flag
                        setattr(self, flag_name, False)
                        
                        # Calculate duration based on video length if available
                        duration = None
                        if status_name in self.loaded_videos:
                            video_duration = self.loaded_videos[status_name]['duration']
                            if video_duration > 0:
                                # Add small buffer to ensure video plays fully
                                duration = video_duration + 0.2  # Small buffer (reduced from 0.5)
                        
                        # Special handling for wave status (has substates)
                        if status_name == 'wave':
                            # Add extra time for hai_text phase if video duration available
                            if duration and 'hand_waving' in self.loaded_videos:
                                duration = duration + 3.0  # Add time for hai_text phase
                            self.set_status('wave', duration=duration, substate='hand_waving')
                            print(f"✓ Wave gesture detected - triggering wave status")
                        else:
                            self.set_status(status_name, duration=duration)
                        gesture_triggered = True
                        break  # Only trigger one gesture per frame
            
            # Fallback: Check manual detection flags (for when AI recognizer not available)
            if not gesture_triggered and not self.gesture_recognizer_available:
                # Thumbs up (manual detection)
                if (self.current_status != 'thumbs_up' and 
                    hasattr(self, '_thumbs_up_detected_flag') and 
                    self._thumbs_up_detected_flag):
                    self._thumbs_up_detected_flag = False
                    duration = None
                    if 'thumbs_up' in self.loaded_videos:
                        video_duration = self.loaded_videos['thumbs_up']['duration']
                        if video_duration > 0:
                            # Add small buffer to ensure video plays fully
                            duration = video_duration + 0.2  # Small buffer (reduced from 0.5)
                    self.set_status('thumbs_up', duration=duration)
                    gesture_triggered = True
                
                # Wave (manual detection)
                elif (self.current_status != 'wave' and 
                      hasattr(self, '_wave_detected_flag') and 
                      self._wave_detected_flag):
                    self._wave_detected_flag = False
                    duration = None
                    if 'hand_waving' in self.loaded_videos:
                        video_duration = self.loaded_videos['hand_waving']['duration']
                        if video_duration > 0:
                            duration = video_duration + 0.5 + 3.0  # +3s for hai_text phase
                    self.set_status('wave', duration=duration, substate='hand_waving')
                    gesture_triggered = True
    
    def _detect_gestures_ai(self, frame):
        """Detect gestures using Mediapipe Gesture Recognizer AI model"""
        # Convert frame to Mediapipe Image format
        mp_image = self._mp_image_class(self._mp_image_format.SRGB, frame)
        
        # Recognize gestures
        recognition_result = self.gesture_recognizer.recognize(mp_image)
        
        if recognition_result.gestures:
            # Get the top gesture (highest confidence)
            top_gesture = recognition_result.gestures[0][0]
            gesture_name = top_gesture.category_name
            confidence = top_gesture.score
            
            # Only trigger if confidence is high enough
            if confidence > 0.7:
                # Track consecutive frames for each gesture
                if gesture_name not in self._gesture_frames:
                    self._gesture_frames[gesture_name] = 0.0  # Use float for gradual decay
                
                # Increment counter (with bonus for high confidence)
                confidence_bonus = (confidence - 0.7) * 2.0  # Extra boost for high confidence
                self._gesture_frames[gesture_name] += 1.0 + confidence_bonus
                
                # Reset other gesture counters (but more gradually)
                for other_gesture in self._gesture_frames:
                    if other_gesture != gesture_name:
                        # Decay other gestures gradually instead of instant reset
                        self._gesture_frames[other_gesture] *= self.gesture_decay_rate
                
                # Check if we have enough consecutive frames
                if self._gesture_frames[gesture_name] >= self.gesture_threshold:
                    # Look up gesture in mapping configuration
                    if gesture_name in self.gesture_to_status_map:
                        gesture_config = self.gesture_to_status_map[gesture_name]
                        status_name = gesture_config['status']
                        flag_name = gesture_config['flag']
                        requires_movement = gesture_config.get('requires_movement', False)
                        
                        # Check if movement is required (for gestures like wave)
                        if requires_movement:
                            # Track palm position for movement detection
                            attr_name = f'{gesture_name.lower()}_position_history'
                            if not hasattr(self, attr_name):
                                setattr(self, attr_name, [])
                            
                            position_history = getattr(self, attr_name)
                            
                            # Get hand landmarks for position tracking
                            if recognition_result.hand_landmarks:
                                hand_landmarks = recognition_result.hand_landmarks[0]
                                wrist_x = hand_landmarks[0].x
                                position_history.append(wrist_x)
                                
                                # Keep only last 15 frames
                                if len(position_history) > 15:
                                    position_history.pop(0)
                                
                                # Check for wave motion (horizontal oscillation)
                                if len(position_history) >= 10:
                                    min_x = min(position_history)
                                    max_x = max(position_history)
                                    movement_range = max_x - min_x
                                    
                                    if movement_range > 0.1:  # Significant horizontal movement
                                        # Count direction changes
                                        direction_changes = 0
                                        for i in range(1, len(position_history) - 1):
                                            prev_diff = position_history[i] - position_history[i-1]
                                            next_diff = position_history[i+1] - position_history[i]
                                            if (prev_diff > 0 and next_diff < 0) or (prev_diff < 0 and next_diff > 0):
                                                direction_changes += 1
                                        
                                        if direction_changes >= 2:
                                            # Movement confirmed - trigger gesture
                                            print(f"{gesture_name} detected via movement! (confidence: {confidence:.2f})")
                                            setattr(self, flag_name, True)
                                            self._gesture_frames[gesture_name] = 0
                                            position_history.clear()
                        else:
                            # Simple gesture - no movement required
                            print(f"{gesture_name} detected! (confidence: {confidence:.2f}, frames: {self._gesture_frames[gesture_name]:.1f})")
                            setattr(self, flag_name, True)
                            self._gesture_frames[gesture_name] = 0  # Reset to avoid re-triggering
                    else:
                        # Gesture not mapped - log for debugging
                        if gesture_name not in ['None', '']:
                            print(f"Unmapped gesture detected: {gesture_name} (confidence: {confidence:.2f})")
                            print(f"  Add to gesture_to_status_map to enable this gesture")
            else:
                # Low confidence - gradually decay counters (slower decay)
                for gesture_name in self._gesture_frames:
                    if self._gesture_frames[gesture_name] > 0:
                        self._gesture_frames[gesture_name] *= self.gesture_decay_rate
        else:
            # No gesture detected - gradually decay counters (slower decay to handle brief disappearances)
            for gesture_name in self._gesture_frames:
                if self._gesture_frames[gesture_name] > 0:
                    # Slower decay when gesture disappears briefly
                    self._gesture_frames[gesture_name] *= self.gesture_decay_rate
    
    def _detect_hand_wave(self, frame):
        """Detect hand waving gesture using improved landmark-based detection"""
        results = self.hand_detector.process(frame)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            
            # Use multiple points for more robust detection
            # Track wrist, middle finger MCP, and index finger MCP
            wrist_x = landmarks[0].x
            middle_mcp_x = landmarks[9].x
            index_mcp_x = landmarks[5].x
            
            # Average x position for more stable tracking
            avg_x = (wrist_x + middle_mcp_x + index_mcp_x) / 3.0
            
            # Add to history
            self.hand_wave_history.append(avg_x)
            
            # Keep only last 20 frames (~0.67 seconds at 30fps)
            if len(self.hand_wave_history) > 20:
                self.hand_wave_history.pop(0)
            
            # Detect wave: check for left-right-left or right-left-right motion
            if len(self.hand_wave_history) >= 15:
                # Calculate movement range
                min_x = min(self.hand_wave_history)
                max_x = max(self.hand_wave_history)
                movement_range = max_x - min_x
                
                # Check for significant horizontal movement (wave)
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
            # No hand detected - clear history gradually (don't reset immediately)
            if len(self.hand_wave_history) > 0:
                self.hand_wave_history.pop(0)  # Remove oldest entry
    
    def _detect_thumbs_up(self, frame):
        """Detect thumbs up gesture using improved Mediapipe hand landmarks"""
        results = self.hand_detector.process(frame)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            
            # Mediapipe hand landmarks:
            # 0: WRIST
            # 1-4: THUMB (CMC, MCP, IP, TIP)
            # 5-8: INDEX (MCP, PIP, DIP, TIP)
            # 9-12: MIDDLE (MCP, PIP, DIP, TIP)
            # 13-16: RING (MCP, PIP, DIP, TIP)
            # 17-20: PINKY (MCP, PIP, DIP, TIP)
            
            # Check if thumb is extended upward
            thumb_tip = landmarks[4]  # Thumb tip
            thumb_ip = landmarks[3]    # Thumb IP joint
            thumb_mcp = landmarks[2]   # Thumb MCP joint
            
            # Thumb should be extended (tip above IP joint)
            # Also check that thumb is significantly above MCP (more robust)
            thumb_extended = (thumb_tip.y < thumb_ip.y) and (thumb_tip.y < thumb_mcp.y - 0.05)
            
            # Check if other fingers are closed/curled
            # For each finger, check if tip is below PIP joint (curled)
            # Use a threshold to allow for slight variations
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            index_closed = index_tip.y > index_pip.y + 0.02
            
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            middle_closed = middle_tip.y > middle_pip.y + 0.02
            
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            ring_closed = ring_tip.y > ring_pip.y + 0.02
            
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            pinky_closed = pinky_tip.y > pinky_pip.y + 0.02
            
            # Thumbs up: thumb extended and other fingers closed
            # Require at least 3 out of 4 fingers closed (more forgiving)
            closed_fingers = sum([index_closed, middle_closed, ring_closed, pinky_closed])
            if thumb_extended and closed_fingers >= 3:
                self.thumbs_up_frames += 1
                # Require multiple consecutive frames to avoid false positives
                if self.thumbs_up_frames >= self.thumbs_up_threshold:
                    print("Thumbs up detected!")
                    self._thumbs_up_detected_flag = True
                    self.thumbs_up_frames = 0  # Reset counter
            else:
                # Reset counter if gesture breaks
                if self.thumbs_up_frames > 0:
                    self.thumbs_up_frames = max(0, self.thumbs_up_frames - 1)  # Gradual decay
        else:
            # No hand detected - gradually reset counter
            if self.thumbs_up_frames > 0:
                self.thumbs_up_frames = max(0, self.thumbs_up_frames - 2)
    
    def _load_action_videos(self):
        """Load all action videos defined in video_config"""
        print(f"Loading action videos from: {self.videos_dir}")
        
        for action_name, video_filename in self.video_config.items():
            video_path = os.path.join(self.videos_dir, video_filename)
            
            try:
                if os.path.exists(video_path):
                    video_capture = cv2.VideoCapture(video_path)
                    if video_capture.isOpened():
                        fps = video_capture.get(cv2.CAP_PROP_FPS)
                        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        self.loaded_videos[action_name] = {
                            'capture': video_capture,
                            'fps': fps,
                            'frame_count': frame_count,
                            'duration': duration,
                            'current_frame': 0,
                            'last_elapsed': 0.0,
                            'last_frame_time': 0.0
                        }
                        
                        print(f"  ✓ Loaded '{action_name}': {video_filename} "
                              f"({frame_count} frames, {fps:.2f} fps, {duration:.2f}s)")
                    else:
                        print(f"  ✗ Could not open video: {video_path}")
                else:
                    print(f"  ✗ Video not found: {video_path}")
            except Exception as e:
                print(f"  ✗ Error loading video '{action_name}' ({video_filename}): {e}")
        
        if not self.loaded_videos:
            print("  Warning: No action videos loaded")
        else:
            print(f"Successfully loaded {len(self.loaded_videos)} action video(s)")
    
    def add_action_video(self, action_name, video_filename):
        """Add a new action video dynamically
        
        Args:
            action_name: Name of the action (e.g., 'clapping', 'pointing')
            video_filename: Filename of the video in the videos directory (e.g., 'clap.mp4')
        
        Returns:
            True if video was loaded successfully, False otherwise
        """
        # Add to config for persistence
        self.video_config[action_name] = video_filename
        
        # Load the video
        video_path = os.path.join(self.videos_dir, video_filename)
        
        try:
            if os.path.exists(video_path):
                video_capture = cv2.VideoCapture(video_path)
                if video_capture.isOpened():
                    fps = video_capture.get(cv2.CAP_PROP_FPS)
                    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    self.loaded_videos[action_name] = {
                        'capture': video_capture,
                        'fps': fps,
                        'frame_count': frame_count,
                        'duration': duration,
                        'current_frame': 0,
                        'last_elapsed': 0.0,
                        'last_frame_time': 0.0
                    }
                    
                    print(f"Added action video '{action_name}': {video_filename} "
                          f"({frame_count} frames, {fps:.2f} fps, {duration:.2f}s)")
                    return True
                else:
                    print(f"Warning: Could not open video: {video_path}")
                    return False
            else:
                print(f"Warning: Video not found: {video_path}")
                return False
        except Exception as e:
            print(f"Error loading video '{action_name}' ({video_filename}): {e}")
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
        # Check if video is loaded
        if action_name not in self.loaded_videos:
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Get Hydra visual (already updated in main update() loop)
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        if hydra_frame is None:
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        video_info = self.loaded_videos[action_name]
        video_capture = video_info['capture']
        fps = video_info['fps']
        frame_count = video_info['frame_count']
        duration = video_info['duration']
        
        # Check if video capture is still valid, try to reopen if needed
        if video_capture is None or not video_capture.isOpened():
            # Try to reopen the video
            try:
                video_path = os.path.join(self.videos_dir, self.video_config[action_name])
                if os.path.exists(video_path):
                    video_capture = cv2.VideoCapture(video_path)
                    if video_capture.isOpened():
                        video_info['capture'] = video_capture
                        print(f"Reopened video: {action_name}")
                    else:
                        return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
                else:
                    return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            except Exception as e:
                print(f"Error reopening video {action_name}: {e}")
                return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Calculate which frame to show based on elapsed time
        # Don't loop - stop at the end and hold the last frame
        if duration > 0:
            # Cap elapsed time at video duration to prevent looping
            # Use duration exactly (no offset) to ensure we can reach the last frame
            video_time = min(elapsed, duration)
            target_frame_number = int(video_time * fps)
            # Ensure we don't exceed frame count, but allow reaching the last frame
            target_frame_number = min(target_frame_number, frame_count - 1)
            # Ensure we don't go negative
            target_frame_number = max(0, target_frame_number)
        else:
            target_frame_number = 0
        
        # Check if we need to reset (new status started)
        current_frame = video_info['current_frame']
        last_elapsed = video_info['last_elapsed']
        
        # Reset if elapsed went backwards (new status started)
        if elapsed < last_elapsed:
            with suppress_stderr():
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_info['current_frame'] = 0
            video_info['last_elapsed'] = elapsed
            video_info['last_frame_time'] = elapsed
            current_frame = 0
        
        # Sequential frame reading for smooth playback (no seeking during playback)
        frame_diff = target_frame_number - current_frame
        
        # Suppress FFmpeg warnings during frame reading
        with suppress_stderr():
            if frame_diff < 0:
                # We're past the end of video, hold the last frame
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = video_capture.read()
                if not ret or frame is None:
                    return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
                current_frame = frame_count - 1
            elif frame_diff > 10:
                # Very large jump (video restarted or jumped ahead significantly)
                # Only seek for large jumps to avoid choppiness
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)
                ret, frame = video_capture.read()
                if not ret or frame is None:
                    return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
                current_frame = target_frame_number
            elif frame_diff > 0:
                # Small to medium jump - use sequential reading for smooth playback
                # This is the normal case during smooth playback
                for _ in range(frame_diff):
                    ret, frame = video_capture.read()
                    if not ret or frame is None:
                        # End of video or read error
                        if current_frame > 0:
                            # Try to seek to last frame
                            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                            ret, frame = video_capture.read()
                            if ret and frame is not None:
                                current_frame = frame_count - 1
                                break
                        return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
                    current_frame += 1
            else:
                # frame_diff == 0, we're at the right frame
                # Don't read again - this happens when update() is called multiple times
                # for the same frame (common when rendering is faster than video FPS)
                # Just reuse the current position without reading
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = video_capture.read()
                if not ret or frame is None:
                    return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        if not ret or frame is None:
            # Return black frame instead of crashing
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Update tracking
        video_info['current_frame'] = current_frame
        video_info['last_elapsed'] = elapsed
        video_info['last_frame_time'] = elapsed
        
        # Convert BGR to RGB
        video_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get video dimensions
        vid_height, vid_width = video_frame_rgb.shape[:2]
        target_width, target_height = self.width, self.height
        
        # Calculate scaling for "fit" mode (crop edges to fit)
        scale_x = target_width / vid_width
        scale_y = target_height / vid_height
        scale = max(scale_x, scale_y)
        
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
            return self._render_hai_text(elapsed)
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
    
    def _get_font(self):
        """Get or load the TTF font"""
        if self._font_cache is None and self.font_path:
            try:
                self._font_cache = ImageFont.truetype(self.font_path, self.font_size)
            except Exception as e:
                print(f"Warning: Could not load font {self.font_path}: {e}")
                self._font_cache = None
        return self._font_cache
    
    def _text_to_pixel_map(self, text, font_size_scale=1.0):
        """Render text using TTF font and convert to pixel map
        
        Returns a list of (x, y) tuples representing pixel positions relative to text origin
        """
        font = self._get_font()
        if font is None:
            # Fallback: return empty map
            return []
        
        # Render text at high resolution for better quality
        scale_factor = 4  # Render at 4x resolution for smoother edges
        render_size = int(self.font_size * font_size_scale * scale_factor)
        
        try:
            # Create temporary font at scaled size
            temp_font = ImageFont.truetype(self.font_path, render_size) if self.font_path else None
            if temp_font is None:
                return []
            
            # Get text bounding box to determine canvas size
            # Use a temporary image to measure
            temp_img = Image.new('RGB', (render_size * len(text) * 2, render_size * 2), (0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=temp_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Create image for rendering
            img_width = int(text_width + render_size)
            img_height = int(text_height + render_size)
            img = Image.new('RGB', (img_width, img_height), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw text in white
            draw.text((render_size // 2, render_size // 2), text, font=temp_font, fill=(255, 255, 255))
            
            # Convert to numpy array and extract pixel positions
            pixels = np.array(img)
            # Get all white pixels (where text is)
            text_pixels = np.where((pixels[:, :, 0] > 128) & 
                                   (pixels[:, :, 1] > 128) & 
                                   (pixels[:, :, 2] > 128))
            
            # Convert to list of (x, y) coordinates relative to text origin
            pixel_map = []
            for y, x in zip(text_pixels[0], text_pixels[1]):
                # Scale down to final resolution and adjust for origin
                final_x = (x - render_size // 2) / scale_factor
                final_y = (y - render_size // 2) / scale_factor
                pixel_map.append((final_x, final_y))
            
            return pixel_map
            
        except Exception as e:
            print(f"Error rendering text to pixel map: {e}")
            return []
    
    def _get_pixel_font_char(self, char):
        """DEPRECATED: Kept for compatibility, but now uses TTF font"""
        # This method is no longer used but kept to avoid breaking code
        return []
    
    def _apply_bubbly_effect(self, px, py, elapsed, text_center_x, text_center_y):
        """Apply bubbly/wavy distortion to pixel positions
        
        Args:
            px, py: Pixel position relative to text origin
            elapsed: Elapsed time for animation
            text_center_x, text_center_y: Center of the text for reference
        
        Returns (offset_x, offset_y) for the pixel position
        """
        # Base wave effect - per-pixel wave based on position
        wave_freq_x = 2.5
        wave_freq_y = 3.0
        wave_amp = 0.8
        
        # Per-pixel wave based on position
        wave_x = math.sin(elapsed * wave_freq_x + px * 0.3) * wave_amp
        wave_y = math.sin(elapsed * wave_freq_y + py * 0.2) * wave_amp
        
        # Bubbly effect - distance from text center creates rounded distortion
        dist_from_center = math.sqrt((px - text_center_x)**2 + (py - text_center_y)**2)
        max_dist = math.sqrt((self.width)**2 + (self.height)**2) * 0.5
        
        # Normalize distance (0-1)
        norm_dist = min(dist_from_center / max_dist, 1.0) if max_dist > 0 else 0
        
        # Apply slight inward curve at edges (bubbly effect)
        bubble_factor = math.sin(norm_dist * math.pi) * 0.3
        bubble_x = math.cos(math.atan2(py - text_center_y, px - text_center_x)) * bubble_factor if dist_from_center > 0 else 0
        bubble_y = math.sin(math.atan2(py - text_center_y, px - text_center_x)) * bubble_factor if dist_from_center > 0 else 0
        
        return (wave_x + bubble_x, wave_y + bubble_y)
    
    def _render_text_with_effects(self, text, elapsed, center_x=None, center_y=None, 
                                   base_color=None, use_hydra_mask=True, 
                                   wobble_amount=1.5, font_size_scale=1.0):
        """Render text with TTF font, bubbly/wavy effects, dynamic wobbling, and hydra masking
        
        Args:
            text: String to render
            elapsed: Elapsed time for animations
            center_x, center_y: Center position (defaults to screen center)
            base_color: Base text color (RGB tuple, defaults to cyan)
            use_hydra_mask: If True, mask text with hydra visual colors
            wobble_amount: Amount of wobbling (rotation/translation)
            font_size_scale: Scale factor for font size (1.0 = default)
        """
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        if center_x is None:
            center_x = self.width // 2
        if center_y is None:
            center_y = self.height // 2
        if base_color is None:
            base_color = (100, 200, 255)  # Cyan
        
        # Update hydra frame for masking
        if use_hydra_mask:
            self._update_hydra_frame()
        
        # Get pixel map from TTF font rendering
        pixel_map = self._text_to_pixel_map(text, font_size_scale)
        
        if not pixel_map:
            # Fallback: return black image if font rendering failed
            return img
        
        # Calculate text bounds to center it
        if pixel_map:
            min_x = min(px for px, py in pixel_map)
            max_x = max(px for px, py in pixel_map)
            min_y = min(py for px, py in pixel_map)
            max_y = max(py for px, py in pixel_map)
            text_width = max_x - min_x
            text_height = max_y - min_y
            text_center_x_rel = (min_x + max_x) / 2
            text_center_y_rel = (min_y + max_y) / 2
        else:
            text_width = 0
            text_height = 0
            text_center_x_rel = 0
            text_center_y_rel = 0
        
        # Dynamic wobbling - overall rotation and translation
        wobble_rot = math.sin(elapsed * 1.2) * wobble_amount * 0.1  # Rotation in radians
        wobble_x = math.cos(elapsed * 0.8) * wobble_amount
        wobble_y = math.sin(elapsed * 1.0) * wobble_amount * 0.7
        
        # Render each pixel from the pixel map
        for px_rel, py_rel in pixel_map:
            # Apply bubbly/wavy effect
            bubble_offset_x, bubble_offset_y = self._apply_bubbly_effect(
                px_rel, py_rel, elapsed, text_center_x_rel, text_center_y_rel
            )
            
            # Apply wobbling (rotation around text center)
            wobble_cos = math.cos(wobble_rot)
            wobble_sin = math.sin(wobble_rot)
            
            # Rotate around text center
            px_rotated = (px_rel - text_center_x_rel) * wobble_cos - (py_rel - text_center_y_rel) * wobble_sin
            py_rotated = (px_rel - text_center_x_rel) * wobble_sin + (py_rel - text_center_y_rel) * wobble_cos
            
            # Add bubble offset and wobble translation
            px_final = px_rotated + text_center_x_rel + bubble_offset_x + wobble_x
            py_final = py_rotated + text_center_y_rel + bubble_offset_y + wobble_y
            
            # Final pixel position (centered on screen)
            px = int(center_x + px_final - text_center_x_rel)
            py = int(center_y + py_final - text_center_y_rel)
            
            # Check bounds
            if 0 <= px < self.width and 0 <= py < self.height:
                # Get color - use hydra mask if enabled
                if use_hydra_mask:
                    # Convert pixel coordinates to normalized coordinates
                    nx = (px - self.width / 2) / (self.width / 2)
                    ny = (py - self.height / 2) / (self.height / 2)
                    
                    # Get hydra color at this position
                    hydra_color = self._get_hydra_texture_color(nx, ny, normalize_brightness=False)
                    
                    if hydra_color:
                        # Blend base color with hydra color (70% hydra, 30% base)
                        color = tuple(np.array(hydra_color) * 0.7 + np.array(base_color) * 0.3)
                        color = tuple(np.clip(color, 0, 255).astype(np.uint8))
                    else:
                        color = base_color
                else:
                    color = base_color
                
                pixels[py, px] = color
        
        return Image.fromarray(pixels)
    
    def _render_hai_text(self, elapsed):
        """Render wavy 'hai' text with TTF font and dynamic effects"""
        return self._render_text_with_effects(
            text="HAI",
            elapsed=elapsed,
            center_x=self.width // 2,
            center_y=self.height // 2,
            base_color=(100, 200, 255),  # Cyan
            use_hydra_mask=True,
            wobble_amount=1.5,
            font_size_scale=0.8  # Slightly smaller to fit better
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
        
        # Get people mask (copy while holding lock to avoid race conditions)
        with self.people_mask_lock:
            people_mask = self.people_mask.copy() if self.people_mask is not None else None
        
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
        
        # Stop speech-to-text
        if self.speech_to_text:
            self.speech_to_text.stop_listening()
            self.speech_to_text = None
        
        # Stop camera thread
        self.camera_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Release all action videos
        for action_name, video_info in self.loaded_videos.items():
            if video_info.get('capture'):
                try:
                    video_info['capture'].release()
                    print(f"Released video: {action_name}")
                except Exception as e:
                    print(f"Error releasing video '{action_name}': {e}")
        self.loaded_videos.clear()
        
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
