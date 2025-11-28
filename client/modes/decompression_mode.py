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
from state_manager import StateManager
from intent_command_handler import IntentCommandHandler
from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np
from threading import Thread, Lock, RLock
from queue import Queue, Empty
from typing import List, Dict, Optional, Tuple
from logger import get_logger

# Enable multi-threading for NumPy and OpenCV to utilize all CPU cores
os.environ.setdefault('OPENCV_NUM_THREADS', '0')  # 0 = use all available threads
os.environ.setdefault('OMP_NUM_THREADS', '0')  # OpenMP threads for NumPy (0 = all cores)
os.environ.setdefault('MKL_NUM_THREADS', '0')  # Intel MKL threads (0 = all cores)
os.environ.setdefault('NUMEXPR_NUM_THREADS', '0')  # NumExpr threads (0 = all cores)
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '0')  # Accelerate framework (macOS)

try:
    # Try to set CPU affinity for better performance on multi-core systems
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import cv2

# Set OpenCV to use all threads
try:
    cv2.setNumThreads(0)  # 0 = use all available threads
except AttributeError:
    pass  # Older OpenCV versions might not have this

# Enable GPU acceleration for OpenCV if available
try:
    from gpu_utils import get_gpu_detector
    gpu_detector = get_gpu_detector()
    gpu_detector.enable_opencv_gpu()
except ImportError:
    pass  # GPU utils not available
except Exception:
    pass  # GPU detection failed, continue with CPU
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


def enumerate_cameras(max_cameras=10):
    """Enumerate available cameras by testing each index.
    
    Args:
        max_cameras: Maximum number of camera indices to test
        
    Returns:
        List of tuples (index, backend) for available cameras
    """
    available_cameras = []
    
    # Try V4L2 backend first (Linux)
    for i in range(max_cameras):
        try:
            with suppress_stderr():
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append((i, cv2.CAP_V4L2))
                    cap.release()
        except:
            pass
    
    # If no V4L2 cameras found, try default backend
    if not available_cameras:
        for i in range(max_cameras):
            try:
                with suppress_stderr():
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        # Test if we can read a frame
                        ret, _ = cap.read()
                        if ret:
                            available_cameras.append((i, None))  # None means default backend
                        cap.release()
            except:
                pass
    
    return available_cameras



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
    - 'wave', 'smile', 'thumbs_up': Interactive statuses triggered by user gestures
    
    State Management:
    The mode includes a robust state transition system that supports:
    - Time-based transitions: Switch between statuses after a set interval
    - Random transitions: Randomly select from a pool of available statuses
    - User interaction: Gesture detection triggers interactive statuses
    - Cooldown integration: All transitions respect cooldown timers
    
    Example usage:
        # Switch between eye and people every 10 minutes
        mode.add_transition_rule('time', interval=600, from_status='eye', to_status='people')
        mode.add_transition_rule('time', interval=600, from_status='people', to_status='eye')
        
        # Randomly choose from main statuses every 5 minutes
        mode.add_transition_rule('random', interval=300, status_pool=['eye', 'people'])
        
        # Get state information
        info = mode.get_state_info()
    
    Future statuses can be added by implementing new _render_*_status() methods
    and updating the update() method to route to them.
    """
    
    def __init__(self, width=40, height=30):
        super().__init__(width, height)
        
        # Set up logger
        self.logger = get_logger("Decompression Mode")
        
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
            'hand_waving': {'filename': 'hand_wave.mp4', 'max_frames': 162},
            'thumbs_up': {'filename': 'thumbs_up.mp4'},
            'smile': {'filename': 'smile.mp4'},
            'think': {'filename': 'think.mp4'},
            'peace': {'filename': 'peace.mp4'},
        }
        # Store max_frames per video for frame limiting
        self.video_max_frames = {}
        for action_name, config in self.video_config.items():
            if isinstance(config, dict) and 'max_frames' in config:
                self.video_max_frames[action_name] = config['max_frames']
        
        # Speech-to-text control
        self.enable_stt = True
        self._stt_thread = None
        
        # State management system
        # Separate "status" (persistent) from "animation" (temporary overlay)
        # Base statuses: eye, people, raw, people_kaleidoscope (persistent)
        # Animation statuses: wave, thumbs_up, smile, peace, think, etc. (temporary overlays)
        self.base_status = 'people_kaleidoscope'  # Persistent status
        self.animation_status = None  # Current animation playing on top (None if no animation)
        self.current_status = 'people_kaleidoscope'  # Combined: animation_status if set, else base_status
        self.status_lock = RLock()  # Use RLock to allow reentrant locking (needed when _enter_status is called from set_status)
        self.status_start_time = time.time()
        # Set initial duration for people_kaleidoscope (10 minutes)
        self.status_duration = 600.0
        self.status_substate = None
        self.previous_status = None
        self.status_transition_callbacks = {}
        
        # Main statuses (persistent backgrounds)
        self.main_statuses = ['eye', 'people', 'raw', 'people_kaleidoscope']
        # Animation statuses (temporary overlays)
        self.animation_statuses = ['wave', 'thumbs_up', 'smile', 'peace', 'think', 'text_scroller_intent']
        
        # Text scroller for voice intents
        self.intent_text_scroller_text = None
        self.intent_text_scroller_wobble = 0.0
        self.text_scroller_lock = Lock()  # Lock for text scroller updates
        
        # Animation text context (shared across gesture animations)
        self.animation_text = None
        self.animation_text_wobble = 0.0
        self.pending_animation_config = None
        self.pending_animation_config_status = None
        
        # Gesture processing lock - prevents multiple gestures from being processed simultaneously
        self.gesture_processing_lock = Lock()
        self.gesture_processing = False
        
        # Action queue - items to process after animations complete
        self.action_queue = []  # List of action dicts: {'type': 'text_scroller', 'text': str, 'wobble': float, ...}
        self.action_queue_lock = Lock()  # Lock for queue operations
        
        # Cached copy of eye Hydra frame for rendering (updated once per frame to prevent flickering)
        self.eye_hydra_frame_cached = None
        
        # Raw mode state - sparse sketches
        self.available_sparse_sketches = []
        self.current_sparse_sketch = None
        self.sparse_sketch_switch_interval = 60.0  # Switch sketch every 1 minute
        self.last_sparse_sketch_switch_time = None
        
        # Intent handler processes commands instantly, no pending tracking needed
        self.think_return_status = None  # Main status to return to after think completes
        
        # People detection timeout tracking
        self.people_mode_entered_time = None
        self.last_people_segment_time = None
        self.people_timeout_seconds = 5.0  # Switch to another status if no segments for 5 seconds
        # Config: what status to switch to on people timeout
        # Can be: 'random' (randomly select from main_statuses excluding 'people'), 
        #         a specific status name (e.g., 'eye', 'raw'), or None (use random)
        self.people_timeout_target_status = 'random'  # Default: randomly select from main statuses
        
        # Cooldown times: eye/people/raw = 10 min (600s), gestures = 1 min (60s)
        status_cooldowns = {
            'eye': 600.0,      # 10 minutes
            'people': 600.0,   # 10 minutes
            'raw': 600.0,     # 10 minutes
            'people_kaleidoscope': 600.0,  # 10 minutes
            'wave': 60.0,      # 1 minute
            'smile': 60.0,     # 1 minute
            'thumbs_up': 60.0, # 1 minute
            'think': 60.0,     # 1 minute
        }
        
        # Dynamically add gesture cooldowns from config manager
        # This will be populated after config_manager is initialized
        main_statuses = ['eye', 'people', 'raw', 'people_kaleidoscope']
        interactive_statuses = ['wave', 'smile', 'thumbs_up', 'think']
        
        # Initialize state manager
        self.state_manager = StateManager(
            status_cooldowns=status_cooldowns,
            main_statuses=main_statuses,
            interactive_statuses=interactive_statuses
        )
        
        # Status configuration - defines behavior for each status
        self.status_config = {
            'wave': {
                'video_name': 'hand_waving',
                'has_substates': True,
                'default_substate': 'hand_waving',
                'has_text_scroller': True,
                'render_method': self._render_wave_status,
            },
            'thumbs_up': {
                'video_name': 'thumbs_up',
                'has_substates': False,
                'has_text_scroller': True,
                'render_method': self._render_thumbs_up_status,
            },
            'smile': {
                'video_name': 'smile',
                'has_substates': False,
                'has_text_scroller': True,
                'render_method': self._render_smile_status,
            },
            'think': {
                'video_name': 'think',
                'has_substates': False,
                'has_text_scroller': False,  # Uses custom thinking text
                'thinking_text_duration': 10.0,
                'render_method': self._render_think_status,
            },
            'people': {
                'video_name': None,
                'has_substates': False,
                'render_method': self._render_people_status,
            },
            'people_kaleidoscope': {
                'video_name': None,
                'has_substates': False,
                'render_method': self._render_people_kaleidoscope_status,
            },
            'eye': {
                'video_name': None,
                'has_substates': False,
                'render_method': self._render_eye_status,
            },
            'raw': {
                'video_name': None,
                'has_substates': False,
                'render_method': self._render_raw_status,
            },
            'text_scroller_intent': {
                'video_name': None,
                'has_substates': False,
                'render_method': self._render_text_scroller_intent_status,
            },
            # New Burning Man gestures
            'peace': {
                'video_name': 'peace',
                'has_substates': False,
                'has_text_scroller': True,
                'render_method': self._render_gesture_status,
            },
            'heart': {
                'video_name': 'heart',
                'has_substates': False,
                'has_text_scroller': True,
                'render_method': self._render_gesture_status,
            },
            'rock_on': {
                'video_name': 'rock_on',
                'has_substates': False,
                'has_text_scroller': True,
                'render_method': self._render_gesture_status,
            },
            # point and fist_pump removed - not supported by AI gesture model
            # clap, heart, rock_on also removed - only AI model gestures (wave, thumbs_up, peace) supported
        }
        
        # Initialize intent-based command handler
        # Resolve CSV path relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "..", "intent_mappings.csv")
        csv_path = os.path.normpath(csv_path)
        
        self.intent_command_handler = IntentCommandHandler(
            intent_csv_path=csv_path,
            use_intent_model=False  # Use keyword matching instead of ML model
        )
        self.intent_command_handler.set_status_change_callback(self._handle_voice_status_change)
        self.intent_command_handler.set_tts_callback(self._handle_voice_tts)
        self.intent_command_handler.set_text_scroller_callback(self._handle_voice_text_scroller)
        
        # Preload model at startup to avoid dynamic installation
        try:
            self.intent_command_handler.preload_model()
            self.logger.info("Intent classification model preloaded at startup")
        except Exception as e:
            self.logger.warning(f"Could not preload intent model: {e}")
        
        # Supported gestures: wave, thumbs_up, peace (AI model), smile (face-based)
        # point, fist_pump, heart, rock_on, clap removed - not supported by AI model
        supported_gestures = ['wave', 'thumbs_up', 'peace', 'smile']
        for gesture_name in supported_gestures:
            if gesture_name not in ['wave', 'smile', 'thumbs_up']:  # Already added above
                status_cooldowns[gesture_name] = 60.0  # 1 minute cooldown
                if gesture_name not in interactive_statuses:
                    interactive_statuses.append(gesture_name)
            
            # Dynamically add status configs for gestures
            if gesture_name not in self.status_config:  # Skip if already configured
                self.status_config[gesture_name] = {
                    'video_name': gesture_name,
                    'has_substates': False,
                    'video_buffer': 0.2,
                    'text_buffer': 0.5,
                    'has_text_scroller': True,
                    'render_method': self._render_gesture_status,
                }
        
        # Update state manager with dynamic cooldowns
        self.state_manager.status_cooldowns.update(status_cooldowns)
        self.state_manager.interactive_statuses = interactive_statuses
        
        # Store gesture trigger configs for easy access (loaded lazily) - DEPRECATED, use config_manager
        self.gesture_configs = {}  # Maps gesture name -> config dict
        self.tts_triggers = []  # Will be loaded lazily on first use
        
        # Register state manager callbacks
        self.state_manager.register_status_change_callback(self._on_state_manager_status_change)
        
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
        
        # Second Hydra instance for eye mode outside area (uses sketches_sparse.txt)
        self.eye_hydra_driver = None
        self.eye_hydra_frame = None
        self.eye_hydra_frame_lock = Lock()
        self.eye_hydra_driver_lock = Lock()  # Lock to prevent driver contention between URL updates and screenshots
        self.eye_hydra_url = "http://localhost:5173"
        self.eye_current_sparse_sketch = None
        self.eye_sparse_sketch_switch_interval = 60.0  # Switch sketch every 1 minute
        self.eye_last_sparse_sketch_switch_time = None
        
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
        
        # Font settings - use slkscr.ttf (Silkscreen font) from client folder
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # client/
        font_path = os.path.join(script_dir, "slkscr.ttf")
        
        # Fallback to KiwiSoda.ttf if slkscr not found, or system font as last resort
        if not os.path.exists(font_path):
            font_path = os.path.join(script_dir, "KiwiSoda.ttf")
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
        # TTS triggers will be loaded from config
        self.tts_triggers = []
        self.next_tts_time = None
        # Periodic text scroller (5 minutes)
        self.next_text_scroller_time = None
        self.text_scroller_quips = [
            "dusty hugs rolling by",
            "welcome to decompression lane",
            "riding the glow waves",
            "kaleidoscope dreams",
            "london decompression vibes",
            "burning man energy",
            "playa magic in motion",
            "decompress and flow",
            "mirror mirror on the wall",
            "infinite reflections",
            "kaleidoscopic wonder",
            "decompression mode activated",
            "welcome wanderer",
            "dust and dreams",
            "playa dust never settles",
            "decompress your mind",
            "kaleidoscope of souls",
            "london meets black rock",
            "infinite patterns",
            "decompression in progress"
        ]
    
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
        if 'enable_stt' in kwargs:
            self.enable_stt = kwargs['enable_stt']
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
    
    def _load_all_sparse_sketches(self):
        """Load all sparse sketches from sketches_sparse.txt"""
        script_dir = os.path.dirname(os.path.abspath(__file__))  # client/modes/
        sketches_file = os.path.join(script_dir, "sketches_sparse.txt")
        
        if not os.path.exists(sketches_file):
            return []
        
        try:
            with open(sketches_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            return lines
        except Exception as e:
            self.logger.error(f"Error loading sparse sketches: {e}")
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
        
        # Initialize second Hydra instance for eye mode outside area
        self._init_eye_hydra_instance()
        
        # Initialize camera
        if self.camera_enabled:
            self.camera = None
            try:
                # Enumerate available cameras first
                available_cameras = enumerate_cameras(max_cameras=10)
                
                if available_cameras:
                    # Use the first available camera
                    camera_index, backend = available_cameras[0]
                    backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "default"
                    self.logger.info(f"Found {len(available_cameras)} available camera(s), using camera index {camera_index} with {backend_name} backend")
                    
                    # Suppress OpenCV camera errors by temporarily redirecting stderr
                    with suppress_stderr():
                        if backend == cv2.CAP_V4L2:
                            self.camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
                        else:
                            self.camera = cv2.VideoCapture(camera_index)
                else:
                    self.logger.warning("No available cameras found")
                    self.camera = None
                
                # Check if camera opened successfully
                if not self.camera or not self.camera.isOpened():
                    self.logger.warning("Camera not available - face detection will be disabled")
                    self.camera_enabled = False
                    if self.camera:
                        try:
                            self.camera.release()
                        except:
                            pass
                    self.camera = None
                else:
                    # Test if we can actually read a frame
                    ret, _ = self.camera.read()
                    if not ret:
                        self.logger.warning("Camera opened but cannot read frames - disabling")
                        try:
                            self.camera.release()
                        except:
                            pass
                        self.camera = None
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
                        self.logger.info("Camera initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize camera: {e}")
                self.camera_enabled = False
                if self.camera:
                    try:
                        self.camera.release()
                    except:
                        pass
                self.camera = None
        
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
            # Get TTS output device from intent handler
            tts_output_device = self.intent_command_handler.tts_output_device_index
            
            self.audio_service = AudioService(tts_output_device_index=tts_output_device)
            tts_initialized = self.audio_service.initialize_tts()
            if not tts_initialized:
                print("Warning: TTS initialization failed - TTS will not be available")
                print("  Check that Piper TTS model is installed in ~/.local/share/piper/voices/")
            else:
                print("TTS initialized successfully")
            
            # Start speech-to-text asynchronously to avoid blocking startup
            if self.enable_stt:
                self._start_stt_background()
            else:
                print("Speech-to-text disabled (use --stt to enable)")
            
            print("Audio service initialized")
        except Exception as e:
            print(f"Warning: Could not initialize audio service: {e}")
            import traceback
            traceback.print_exc()
            self.audio_service = None
        
        # Initialize default transition rules
        self._setup_default_transition_rules()
        
        # Preload gesture configs from CSV to avoid missing configs during runtime
        self._preload_gesture_configs()
        
        # Load gesture and TTS configs lazily (on first use) to avoid blocking startup
        # They will be loaded when first needed
        
        # Initialize the initial status properly (set duration and people mode tracking)
        self._enter_status(self.current_status, None)
        # Ensure duration is set (in case _enter_status didn't set it)
        if self.status_duration is None and self.current_status in ['eye', 'people', 'people_kaleidoscope']:
            self.status_duration = 600.0
        elif self.status_duration is None and self.current_status == 'raw':
            self.status_duration = 60.0
        
        print("Decompression mode (3D eyeball) initialized")
    
    def _setup_default_transition_rules(self):
        """Set up default transition rules for the state management system
        
        Example: Switch between eye and people every 10 minutes
        """
        # Example: Switch between eye and people every 10 minutes (600 seconds)
        # Uncomment to enable:
        # self.add_transition_rule('time', interval=600, from_status='eye', to_status='people')
        # self.add_transition_rule('time', interval=600, from_status='people', to_status='eye')
        
        # Example: Randomly choose from main statuses every 5 minutes
        # Uncomment to enable:
        # self.add_transition_rule('random', interval=300, status_pool=self.state_manager.main_statuses)
        
        pass  # No default rules - user can add their own
    
    def _preload_gesture_configs(self):
        """Verify gesture configs are available (but don't cache selected responses)
        
        We don't cache the selected responses here - we want random selection each time.
        This just verifies that the gestures can be loaded from the CSV.
        """
        supported_gestures = ['wave', 'thumbs_up', 'peace', 'smile']
        for gesture_name in supported_gestures:
            try:
                # Just verify it can be loaded (don't cache - we want variety)
                trigger = self.intent_command_handler.get_gesture_trigger(gesture_name)
                if trigger:
                    self.logger.info(f"Gesture config available for: {gesture_name}")
                else:
                    self.logger.debug(f"Gesture config for {gesture_name} not yet available")
            except Exception as e:
                self.logger.error(f"Error checking gesture config for {gesture_name}: {e}")
    
    def _get_gesture_config(self, gesture_name: str, retry: bool = True):
        """Get gesture config from CSV intent mappings via intent handler
        
        Always selects a random response - does NOT cache the selected response
        to ensure variety. Individual TTS audio responses are cached at the audio_service level.
        
        Args:
            gesture_name: Name of gesture (e.g., 'wave', 'smile', 'thumbs_up', 'peace', etc.)
            retry: If True, attempt to load if not found (default: True)
            
        Returns:
            Gesture config dict or None if not available
        """
        if not retry:
            return None
        
        # Try intent handler (loads from CSV)
        # NOTE: We do NOT cache the selected response here - always get a fresh random selection
        # This ensures variety. Individual TTS audio is cached by audio_service based on the text.
        try:
            # Ensure intent mappings are loaded
            if not self.intent_command_handler.intent_mappings:
                self.logger.warning(f"Intent mappings not loaded, attempting reload...")
                self.intent_command_handler._load_intent_mappings()
            
            trigger = self.intent_command_handler.get_gesture_trigger(gesture_name)
            if trigger:
                # Only log once per gesture status entry, not on every render call
                # This prevents spam in logs
                return trigger
            else:
                # Only log warning if gesture should exist
                if gesture_name in self.intent_command_handler.available_gestures:
                    # Check if intent mappings are actually loaded
                    intent_count = len(self.intent_command_handler.intent_mappings)
                    if intent_count == 0:
                        self.logger.error(
                            f"Gesture '{gesture_name}' not found - intent mappings appear to be empty! "
                            f"CSV path: {self.intent_command_handler.intent_csv_path}"
                        )
                    else:
                        self.logger.warning(
                            f"Gesture '{gesture_name}' is in available_gestures but not found in CSV. "
                            f"Loaded {intent_count} intents from CSV."
                        )
        except Exception as e:
            self.logger.error(f"Error loading gesture config for {gesture_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _get_wave_config(self):
        """Get wave gesture config with guaranteed fallback
        
        Always selects a random response - does NOT cache to ensure variety.
        
        Returns:
            Gesture config dict, or None if not available (caller should handle gracefully)
        """
        config = self._get_gesture_config('wave', retry=True)
        if not config:
            # Try one more time without rate limiting if this is critical
            try:
                config = self.intent_command_handler.get_gesture_trigger('wave')
                # Don't cache - we want variety each time
                return config
            except Exception:
                pass
        return config
    
    def _get_wave_text_config(self, gesture_config=None):
        """Get wave text_scroller config from CSV intent mappings
        
        Returns:
            Tuple of (text: str, wobble: float) or (None, 0.0) if no text_scroller configured
            Note: wobble is always 0.0 (no wobble effect)
        """
        if gesture_config is None:
            gesture_config = self._get_wave_config()
        if gesture_config and gesture_config.get('text_scroller'):
            text = gesture_config['text_scroller'].get('text')
            # Always return wobble=0.0 (no wobble effect)
            return text, 0.0
        return None, 0.0
    
    def _calculate_wave_duration(self, gesture_config=None):
        """Calculate duration for wave status (video only)
        
        Note: Text scrolling is now queued separately after the video completes,
        so this only returns the video duration.
        
        Args:
            gesture_config: Optional gesture config (will be fetched if None)
            
        Returns:
            Video duration in seconds (no buffers, no text scroll time)
        """
        video_duration = 0.0
        if self.video_manager.has_video('hand_waving'):
            video_duration = self.video_manager.get_duration('hand_waving')
        
        # Only return video duration - text scrolling is queued separately
        return video_duration
    
    def _calculate_scroll_time(self, text: str, scroll_speed: float = 20.0) -> float:
        """Calculate scroll time based on when rightmost pixel exits screen (has negative x)
        
        Args:
            text: Text to calculate scroll time for
            scroll_speed: Pixels per second (default: 20.0)
            
        Returns:
            Scroll time in seconds (no buffer - calculated exactly when rightmost pixel exits)
        """
        if not text:
            return 0.0
        
        # Get text pixel map to calculate width
        pixel_map = self.text_scroller._text_to_pixel_map(text, font_size_scale=1.0)
        if not pixel_map:
            return 0.0
        
        # Calculate text bounds
        min_x = min(px for px, py in pixel_map)
        max_x = max(px for px, py in pixel_map)
        
        # Calculate start position (same as text_scroller)
        start_x = self.width + 1 - min_x
        
        # The rightmost pixel is at position: current_x + max_x
        # We need it to scroll until: current_x + max_x < 0
        # So: current_x < -max_x
        # Since current_x = start_x - (progress * scroll_distance)
        # We need: start_x - (progress * scroll_distance) < -max_x
        # So: scroll_distance * progress > start_x + max_x
        # When progress = 1.0 (complete scroll), we need scroll_distance >= start_x + max_x
        
        # Calculate the distance the rightmost pixel needs to travel
        # Rightmost pixel starts at: start_x + max_x
        # Rightmost pixel needs to be at: < 0
        # So it needs to travel: (start_x + max_x) pixels
        rightmost_exit_distance = start_x + max_x
        
        # Calculate scroll time based on rightmost pixel exit
        if rightmost_exit_distance <= 0:
            return 0.0
        
        scroll_time = rightmost_exit_distance / scroll_speed
        return scroll_time
    
    def _calculate_status_duration(self, status: str, gesture_config = None, 
                                   video_name = None, 
                                   is_text_scroller_substate: bool = False,
                                   video_buffer: float = 0.0,  # No buffer - video plays exactly
                                   text_buffer: float = 0.0,  # No buffer - text exits exactly when rightmost pixel < 0
                                   default_duration: float = 0.0) -> float:
        """Calculate duration for a status based on video and text scrolling
        Duration is calculated exactly: video plays -> text scrolls until rightmost pixel exits -> end
        No buffers - duration is based on actual completion.
        
        Args:
            status: Status name (e.g., 'wave', 'smile', 'thumbs_up')
            gesture_config: Optional gesture configuration dict
            video_name: Optional video name (if different from status, e.g., 'hand_waving' for 'wave')
            is_text_scroller_substate: If True, only calculate text scroll time (no video)
            video_buffer: Buffer time to add after video (default: 0.0 - no buffer)
            text_buffer: Buffer time to add after text scroll (default: 0.0 - no buffer)
            default_duration: Default duration if no video/text (default: 0.0 - end immediately)
            
        Returns:
            Calculated duration in seconds (video + text, no buffers)
        """
        # Determine video name
        if video_name is None:
            video_name = status
        
        # Calculate scroll time from text_scroller config if available
        scroll_time = 0.0
        if gesture_config and gesture_config.get('text_scroller'):
            text = gesture_config['text_scroller'].get('text', '')
            if text:
                scroll_time = self._calculate_scroll_time(text)
        
        # If this is a text_scroller substate, only return text scroll time
        if is_text_scroller_substate:
            return scroll_time  # No buffer
        
        # Get video duration
        video_duration = 0.0
        if self.video_manager.has_video(video_name):
            video_duration = self.video_manager.get_duration(video_name)
        
        # Combine video and text durations (no buffers)
        if video_duration > 0:
            # Video exists: video + text (if available)
            return video_duration + scroll_time
        elif scroll_time > 0:
            # No video but has text: text scroll only
            return scroll_time
        else:
            # No video and no text: end immediately
            return default_duration
    
    def _load_gesture_configs(self):
        """Verify gesture trigger configurations are available (but don't cache)
        
        We don't cache the selected responses - we want random selection each time
        to ensure variety. Individual TTS audio is cached by audio_service.
        """
        triggers = self.intent_command_handler.get_gesture_triggers()
        self.logger.debug(f"Verified {len(triggers)} gesture triggers are available")
    
    def _load_tts_triggers(self):
        """Load TTS trigger configurations from intent handler (lazy load on first use)"""
        if not self.tts_triggers or len(self.tts_triggers) == 0:
            try:
                self.tts_triggers = self.intent_command_handler.get_tts_triggers()
                self.logger.info(f"Loaded {len(self.tts_triggers)} TTS trigger(s)")
            except Exception as e:
                self.logger.error(f"Failed to load TTS triggers: {e}")
                self.tts_triggers = []
                return
        
        # Initialize next TTS time if triggers exist - use 1 minute intervals
        if self.tts_triggers:
            trigger = self.tts_triggers[0]  # Use first trigger for now
            trigger_type = trigger.get('type')
            self.logger.info(f"Using TTS trigger type: {trigger_type}")
            
            # Override intervals to ~1 minute (50-70 seconds)
            interval_min = 50.0
            interval_max = 70.0
            initial_delay = random.uniform(interval_min, interval_max)
            self.next_tts_time = time.time() + initial_delay
            # Store trigger config for later use (with updated intervals)
            self.tts_trigger_config = trigger.copy()
            self.tts_trigger_config['interval_min'] = interval_min
            self.tts_trigger_config['interval_max'] = interval_max
            self.logger.info(f"TTS trigger configured: ~1 minute intervals, first TTS in {initial_delay:.1f}s")
        else:
            self.logger.warning("No TTS triggers found")
            self.next_tts_time = None
    
    def enable_transitions(self):
        """Enable automatic state transitions"""
        self.state_manager.enable_transitions()
    
    def disable_transitions(self):
        """Disable automatic state transitions"""
        self.state_manager.disable_transitions()
    
    def clear_transition_rules(self):
        """Clear all transition rules"""
        self.state_manager.clear_transition_rules()
    
    def get_transition_rules(self):
        """Get all current transition rules
        
        Returns:
            List of transition rule dictionaries
        """
        return self.state_manager.get_transition_rules()
    
    def remove_transition_rule(self, index):
        """Remove a transition rule by index
        
        Args:
            index: Index of rule to remove
        """
        return self.state_manager.remove_transition_rule(index)
    
    def add_main_status(self, status):
        """Add a status to the main statuses pool (for random selection)
        
        Args:
            status: Status name to add
        """
        self.state_manager.add_main_status(status)
    
    def remove_main_status(self, status):
        """Remove a status from the main statuses pool
        
        Args:
            status: Status name to remove
        """
        self.state_manager.remove_main_status(status)
    
    def add_interactive_status(self, status):
        """Add a status to the interactive statuses list
        
        Interactive statuses are temporary and don't participate in automatic transitions.
        
        Args:
            status: Status name to add
        """
        self.state_manager.add_interactive_status(status)
    
    def get_state_info(self):
        """Get comprehensive state information for debugging/monitoring
        
        Returns:
            dict with current state, available transitions, cooldowns, and rules
        """
        with self.status_lock:
            return self.state_manager.get_state_info(
                current_status=self.current_status,
                previous_status=self.previous_status,
                status_start_time=self.status_start_time,
                status_duration=self.status_duration,
                status_substate=self.status_substate
            )
    
    def reload_voice_commands(self):
        """Reload voice command configuration"""
        self.intent_command_handler.reload_config()
        self._load_gesture_configs()
        self._load_tts_triggers()
        self.logger.info("Voice commands reloaded")
    
    def _start_stt_background(self):
        """Initialize speech-to-text in a background thread to avoid blocking startup"""
        if not self.audio_service:
            return
        
        if not self.enable_stt:
            self.logger.info("Speech-to-text disabled via configuration")
            return
        
        if self._stt_thread and self._stt_thread.is_alive():
            return
        
        def _init_stt():
            try:
                print("Initializing speech-to-text (Whisper)… this may download model files.")
                success = self.audio_service.initialize_stt(
                    input_device_index=None,
                    voice_command_callback=self._handle_voice_command
                )
                if success:
                    print("Speech-to-text is ready")
                else:
                    print("Speech-to-text initialization failed")
            except Exception as stt_error:
                print(f"Warning: Speech-to-text error: {stt_error}")
                import traceback
                traceback.print_exc()
        
        self._stt_thread = Thread(target=_init_stt, daemon=True)
        self._stt_thread.start()
    
    def _handle_voice_command(self, text):
        """Handle voice commands using intent classification
        
        Args:
            text: Text from speech-to-text
        """
        if not text:
            return
        
        self.logger.info(f"Voice command received: {text}")
        
        # Process command through intent handler (instant, no async needed)
        success, is_unknown = self.intent_command_handler.process_voice_command(text)
        
        if not success:
            # Command not recognized
            self.logger.warning(f"Command not recognized: {text}")
    
    
    def _check_and_play_tts(self):
        """Check if it's time to play TTS and play it based on config triggers (1 minute intervals)"""
        if not self.audio_service:
            return
        
        if not self.audio_service.tts_voice:
            return
        
        if self.next_tts_time is None:
            return
        
        current_time = time.time()
        if current_time >= self.next_tts_time:
            # Get response from trigger config
            if hasattr(self, 'tts_trigger_config') and self.tts_trigger_config:
                tts_message = self.tts_trigger_config.get('response')
                if tts_message:
                    self.logger.info(f"Playing scheduled TTS: {tts_message}")
                    self.audio_service.speak(tts_message)
                    
                    # Schedule next TTS - approximately 1 minute (60 seconds)
                    interval_min = 50.0  # ~1 minute
                    interval_max = 70.0
                    next_interval = random.uniform(interval_min, interval_max)
                    self.next_tts_time = current_time + next_interval
                    self.logger.info(f"Next TTS scheduled in {next_interval:.1f}s")
                else:
                    self.logger.warning("No responses in trigger config")
                    self.next_tts_time = None
            else:
                self.logger.warning("No trigger config available")
                self.next_tts_time = None
    
    def _check_and_play_text_scroller(self):
        """Check if it's time to show periodic text scroller (5 minute intervals)"""
        # Only show if not already in an animation status
        with self.status_lock:
            if self.animation_status and self.animation_status != 'text_scroller_intent':
                return  # Don't interrupt other animations
        
        if self.next_text_scroller_time is None:
            # Initialize first text scroller time (5 minutes)
            self.next_text_scroller_time = time.time() + random.uniform(280.0, 320.0)  # ~5 minutes
            return
        
        current_time = time.time()
        if current_time >= self.next_text_scroller_time:
            # Select a random quip
            quip = random.choice(self.text_scroller_quips)
            
            # Calculate scroll duration
            scroll_time = self._calculate_scroll_time(quip)
            duration = scroll_time  # No buffer - exact calculation
            
            # Set up text scroller
            with self.text_scroller_lock:
                self.intent_text_scroller_text = quip
                self.intent_text_scroller_wobble = 0.0  # No wobble
            
            # Switch to text_scroller_intent status temporarily
            with self.status_lock:
                if self.animation_status != 'text_scroller_intent':
                    self.previous_status = self.base_status
                self.animation_status = 'text_scroller_intent'
                self.status_duration = duration
                self.status_start_time = current_time
            
            self.logger.info(f"Showing periodic text scroller: {quip}")
            
            # Schedule next text scroller (5 minutes)
            next_interval = random.uniform(280.0, 320.0)  # ~5 minutes
            self.next_text_scroller_time = current_time + next_interval
            self.logger.info(f"Next text scroller scheduled in {next_interval:.1f}s")
    
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
                
                # Check service enable flags from console UI
                # Console UI is the source of truth for service enabled/disabled state
                face_enabled = True
                gesture_enabled = True
                segmentation_enabled = True
                
                if self._console_ui_ref:
                    console_ui = self._console_ui_ref
                    face_service = console_ui.services.get('Face Detection')
                    if face_service:
                        face_enabled = face_service.enabled
                    
                    gesture_service = console_ui.services.get('Gesture Detection')
                    if gesture_service:
                        gesture_enabled = gesture_service.enabled
                    
                    segmentation_service = console_ui.services.get('People Segmentation')
                    if segmentation_service:
                        segmentation_enabled = segmentation_service.enabled
                
                # Smile uses face detector, so use face_enabled
                smile_enabled = face_enabled
                
                # Detect faces (only if enabled, in eye status, and throttle)
                if (face_enabled and current_status == 'eye' and
                    current_time - last_detection_time.get('face', 0) >= detection_intervals['face']):
                    self.face_detector_module.detect(frame_small)
                    last_detection_time['face'] = current_time
                
                # Detect people outlines (only if enabled, in people or people_kaleidoscope status, and throttle)
                if (segmentation_enabled and current_status in ['people', 'people_kaleidoscope'] and
                    current_time - last_detection_time.get('people', 0) >= detection_intervals['people']):
                    self.mask_detector_module.detect(frame_small)
                    last_detection_time['people'] = current_time
                    
                    # Check if people segments were detected (excluding background segments)
                    if self.mask_detector_module.has_mask():
                        mask = self.mask_detector_module.get_mask()
                        if mask is not None:
                            # Check if mask has any non-zero pixels (people detected)
                            # Filter out background segments (typically large segments covering most of screen)
                            if self._has_valid_people_segments(mask):
                                self.last_people_segment_time = current_time
                                # Note: people_mode_entered_time is set when entering people status,
                                # so we don't need to set it here
                
                # Queue frame for background gesture detection (non-blocking)
                # This prevents gesture detection from blocking the camera loop
                if (gesture_enabled and current_status not in ['wave', 'thumbs_up', 'smile']):
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
                if (smile_enabled and current_status not in ['wave', 'thumbs_up', 'smile']):
                    if (current_time - last_detection_time.get('smile', 0) >= detection_intervals['smile']):
                        self.face_detector_module.detect_smile(frame_small, frame_timestamp_ms)
                        frame_timestamp_ms += int(detection_intervals['smile'] * 1000)  # Convert to ms
                        last_detection_time['smile'] = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                self.logger.error(f"Camera error: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                # Continue instead of breaking to keep face detection running
                time.sleep(0.1)  # Brief pause before retrying
                continue
    
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
                if self._console_ui_ref:
                    console_ui = self._console_ui_ref
                    gesture_service = console_ui.services.get('Gesture Detection')
                    if gesture_service:
                        gesture_enabled = gesture_service.enabled
                
                # Only process if enabled
                if gesture_enabled:
                    # Process gesture detection using AI model only
                    if self.gesture_detector_module.gesture_recognizer_available:
                        # Use AI gesture recognizer (covers wave, thumbs_up, peace)
                        self.gesture_detector_module.detect_ai(frame)
                    else:
                        self.logger.warning("AI gesture recognizer not available - gestures disabled")
                
                # Mark task as done
                self.gesture_queue.task_done()
                
            except Exception as e:
                print(f"Gesture detection error: {e}")
                # Continue processing - don't break on errors
    
    def _update_face_position(self):
        """Update target face position from face detector module."""
        # Check if face detection is enabled via console UI
        face_enabled = True  # Default to enabled
        if self._console_ui_ref:
            console_ui = self._console_ui_ref
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
    
    def _on_state_manager_status_change(self, old_status: str, new_status: str):
        """Callback when state manager detects a status change
        
        Args:
            old_status: Previous status
            new_status: New status
        """
        print(f"[StateManager] Status change detected: {old_status} -> {new_status}")
    
    def _handle_voice_status_change(self, status: str) -> bool:
        """Handle status change from intent handler - sets base_status for main statuses, animation for others
        
        Args:
            status: Status name to change to
            
        Returns:
            bool: True if status was set successfully
        """
        # Determine if this is a base status or animation
        is_base = status in self.main_statuses
        
        # Calculate duration for interactive statuses (animations)
        duration = None
        substate = None
        
        if not is_base:
            # For animations, calculate duration including text
            gesture_config = self._get_gesture_config(status)
            self._set_pending_animation_config(status, gesture_config)
        else:
            gesture_config = None
        
        # Use status configuration to determine duration and substate
        config = self.status_config.get(status)
        if config:
            # Determine substate
            if config.get('has_substates', False):
                substate = config.get('default_substate')
                if gesture_config and gesture_config.get('action'):
                    substate = gesture_config['action'].get('substate', substate)
            
            # Calculate duration using configuration (includes text length)
            duration = self._calculate_status_duration(
                status=status,
                gesture_config=gesture_config,
                video_name=config.get('video_name'),
                video_buffer=0.0,  # No buffer - video plays exactly
                text_buffer=0.0  # No buffer - text exits exactly when rightmost pixel < 0
            )
        
        return self.set_status(status, duration=duration, substate=substate, is_base_status=is_base)
    
    def _handle_voice_tts(self, text: str):
        """Handle TTS response from voice command
        
        Args:
            text: Text to speak
        """
        if self.audio_service and self.audio_service.tts_voice:
            print(f"[Voice Command TTS] Speaking: {text}")
            self.audio_service.speak(text)
        else:
            print(f"[Voice Command TTS] TTS not available, would have spoken: {text}")
    
    def _handle_voice_text_scroller(self, text: str, wobble_amount: float = 0.0):
        """Handle text scroller display from voice intent
        
        Args:
            text: Text to display in scroller
            wobble_amount: Wobble effect amount (ignored - always 0.0, no wobble)
        """
        if not text:
            return
        
        # Check if we're currently in an animation
        with self.status_lock:
            animation_status = self.animation_status
        
        # If we're in an animation, queue the text scroller to play after
        if animation_status and animation_status != 'text_scroller_intent':
            self.logger.info(f"[Voice Text Scroller] Animation '{animation_status}' active, queuing text scroller")
            self.queue_action('text_scroller', text=text, wobble=0.0)
            return
        
        # If text_scroller_intent is already active, update it with new text
        if animation_status == 'text_scroller_intent':
            self.logger.info(f"[Voice Text Scroller] Updating existing text scroller with new text: '{text}'")
            # Update text and recalculate duration (wobble always 0.0)
            with self.text_scroller_lock:
                self.intent_text_scroller_text = text
                self.intent_text_scroller_wobble = 0.0
            scroll_time = self._calculate_scroll_time(text)
            duration = scroll_time  # No buffer - exact calculation
            with self.status_lock:
                self.status_duration = duration
                self.status_start_time = time.time()  # Reset timer for new text
            return
        
        # No animation active, show text scroller immediately (wobble always 0.0)
        self._execute_text_scroller(text, wobble_amount=0.0)
    
    def _set_pending_animation_config(self, status: str, gesture_config: Optional[Dict]):
        """Store gesture config so _enter_status can reuse it without re-randomizing"""
        with self.status_lock:
            self.pending_animation_config_status = status
            self.pending_animation_config = gesture_config
    
    def _consume_pending_animation_config(self, status: str) -> Optional[Dict]:
        """Retrieve and clear pending animation config for the given status"""
        with self.status_lock:
            if self.pending_animation_config_status == status:
                config = self.pending_animation_config
                self.pending_animation_config_status = None
                self.pending_animation_config = None
                return config
        return None
    
    def _get_text_from_gesture_config(self, gesture_config: Optional[Dict]) -> Tuple[Optional[str], float]:
        """Extract text scroller info from a gesture config
        
        Returns:
            Tuple of (text: str, wobble: float)
            Note: wobble is always 0.0 (no wobble effect)
        """
        text = None
        if gesture_config and gesture_config.get('text_scroller'):
            text = gesture_config['text_scroller'].get('text')
        # Always return wobble=0.0 (no wobble effect)
        return text, 0.0
    
    def queue_action(self, action_type: str, **kwargs):
        """Queue an action to be processed after current animation completes
        
        Args:
            action_type: Type of action ('text_scroller', 'tts', 'status', etc.)
            **kwargs: Action-specific parameters
                For 'text_scroller': text (str), wobble (float)
                For 'tts': message (str)
                For 'status': status (str), duration (float), etc.
        """
        with self.action_queue_lock:
            action = {'type': action_type, **kwargs}
            self.action_queue.append(action)
            self.logger.info(f"[Queue] Queued action: {action_type} (queue size: {len(self.action_queue)})")
    
    def _process_action_queue(self) -> Optional[Dict]:
        """Process the next item in the action queue
        
        Returns:
            The processed action dict if one was processed, None if queue is empty
        """
        with self.action_queue_lock:
            if not self.action_queue:
                return None
            
            action = self.action_queue.pop(0)
            self.logger.info(f"[Queue] Processing queued action: {action['type']} (remaining: {len(self.action_queue)})")
        
        # Process the action based on type
        if action['type'] == 'text_scroller':
            text = action.get('text')
            if text:
                # Always use wobble=0.0 (no wobble effect)
                self._execute_text_scroller(text, wobble_amount=0.0)
                return action
        elif action['type'] == 'tts':
            message = action.get('message')
            if message and self.audio_service and self.audio_service.tts_voice:
                self.audio_service.speak(message)
                return action
        elif action['type'] == 'status':
            status = action.get('status')
            duration = action.get('duration')
            substate = action.get('substate')
            if status:
                self.set_status(status, duration=duration, substate=substate)
                return action
        else:
            self.logger.warning(f"[Queue] Unknown action type: {action['type']}")
        
        return action
    
    def _execute_text_scroller(self, text: str, wobble_amount: float = 0.0):
        """Execute a text scroller action (called from queue or directly)
        
        Args:
            text: Text to display
            wobble_amount: Wobble amount for text (ignored - always 0.0, no wobble)
        """
        # Calculate scroll duration
        scroll_time = self._calculate_scroll_time(text)
        
        # Store text for rendering (wobble always 0.0)
        with self.text_scroller_lock:
            self.intent_text_scroller_text = text
            self.intent_text_scroller_wobble = 0.0
        
        # Store previous status if not already stored
        with self.status_lock:
            if self.previous_status is None:
                self.previous_status = self.base_status
        
        # Switch to text_scroller_intent status
        self.set_status('text_scroller_intent', duration=scroll_time, force=True)
    
    def is_status_on_cooldown(self, status):
        """Check if a status is currently on cooldown
        
        Args:
            status: Status name to check
            
        Returns:
            tuple: (is_on_cooldown: bool, remaining_time: float)
                   If not on cooldown, remaining_time is 0.0
        """
        return self.state_manager.is_status_on_cooldown(status)
    
    def get_cooldown_info(self):
        """Get cooldown information for all statuses
        
        Returns:
            dict: Maps status -> {'remaining': float, 'cooldown': float, 'progress': float}
                  progress is 0.0 (just started) to 1.0 (cooldown complete)
        """
        return self.state_manager.get_cooldown_info()
    
    def add_transition_rule(self, rule_type, **kwargs):
        """Add a transition rule to the state management system
        
        Args:
            rule_type: Type of rule - 'time', 'random', or 'interaction'
            **kwargs: Rule-specific parameters (see StateManager.add_transition_rule)
        """
        self.state_manager.add_transition_rule(rule_type, **kwargs)
    
    def get_available_statuses(self, status_pool=None):
        """Get list of statuses available for transition (not on cooldown)
        
        Args:
            status_pool: Optional list of statuses to filter from. If None, uses all statuses.
            
        Returns:
            List of status names that are not on cooldown
        """
        return self.state_manager.get_available_statuses(status_pool)
    
    def set_status(self, status, duration=None, substate=None, force=False, is_base_status=False):
        """Set the current status of decompression mode with robust state management
        
        Args:
            status: String status name ('eye', 'people', 'wave', etc.)
            duration: Optional duration in seconds (None = indefinite)
            substate: Optional sub-state for complex statuses (e.g., 'hand_waving', 'text_scroller' for wave)
            force: If True, bypass cooldown check (default: False)
            is_base_status: If True, set as base_status (persistent). If False, auto-detect based on status type.
            
        Returns:
            bool: True if status was set, False if blocked by cooldown
        """
        with self.status_lock:
            # Determine if this is a base status or animation
            if is_base_status or status in self.main_statuses:
                # This is a base status change - clear any animation and set base
                if self.animation_status:
                    self._exit_status(self.animation_status)
                    self.animation_status = None
                    self.animation_text = None
                    self.animation_text_wobble = 0.0
                
                # Check cooldown for base status
                if not force and self.base_status != status:
                    is_on_cooldown, remaining = self.is_status_on_cooldown(status)
                    if is_on_cooldown:
                        print(f"Base status '{status}' is on cooldown. {remaining:.1f}s remaining.")
                        return False
                
                # Exit previous base status
                if self.base_status != status:
                    self._exit_status(self.base_status)
                    self.previous_status = self.base_status
                
                # Set new base status
                self.base_status = status
                self.current_status = status
                self.status_start_time = time.time()
                self.status_substate = substate
                
                # Record execution time for cooldown tracking
                self.state_manager.record_status_execution(status)
                
                # Enter new status (this will set status_duration based on status type)
                self._enter_status(status, substate)
                
                # If duration was explicitly provided, use it (overrides _enter_status)
                if duration is not None:
                    self.status_duration = duration
                
                print(f"Base status changed to: {status}" + 
                      (f" (substate: {substate})" if substate else "") +
                      (f" (duration: {duration}s)" if duration else ""))
                
                return True
            else:
                # This is an animation - play on top of current base status
                # Prevent overlapping animations
                if self.animation_status and self.animation_status != status:
                    self.logger.warning(f"Animation '{status}' requested but '{self.animation_status}' is already playing. Ignoring.")
                    return False
                
                # Check cooldown for animation
                if not force and self.animation_status != status:
                    is_on_cooldown, remaining = self.is_status_on_cooldown(status)
                    if is_on_cooldown:
                        print(f"Animation '{status}' is on cooldown. {remaining:.1f}s remaining.")
                        return False
                
                # Exit previous animation if different
                if self.animation_status and self.animation_status != status:
                    self._exit_status(self.animation_status)
                
                # Set new animation
                self.animation_status = status
                self.current_status = status  # Animation takes precedence for rendering
                self.status_start_time = time.time()
                self.status_substate = substate
                
                # Record execution time for cooldown tracking
                self.state_manager.record_status_execution(status)
                
                # Enter new animation (this will set status_duration based on animation type)
                self._enter_status(status, substate)
                
                # If duration was explicitly provided, use it (overrides _enter_status)
                if duration is not None:
                    self.status_duration = duration
                
                print(f"Animation '{status}' started on base '{self.base_status}'" + 
                  (f" (substate: {substate})" if substate else "") +
                  (f" (duration: {duration}s)" if duration else ""))
            
            return True
    
    def get_status(self):
        """Get the current status of decompression mode"""
        with self.status_lock:
            # Return animation if active, otherwise base status
            return self.animation_status if self.animation_status else self.base_status
    
    def get_base_status(self):
        """Get the current base status (persistent background)"""
        with self.status_lock:
            return self.base_status
    
    def get_animation_status(self):
        """Get the current animation status (temporary overlay)"""
        with self.status_lock:
            return self.animation_status
    
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
        """Handle status entry logic - uses status configuration"""
        config = self.status_config.get(status)
        if not config:
            self.logger.warning(f"Unknown status: {status}")
            return
        
        # Initialize common state
        self.status_start_time = time.time()
        
        # Set substate based on config
        if config.get('has_substates', False):
            self.status_substate = substate or config.get('default_substate')
        else:
            self.status_substate = None
        
        # Special handling for people statuses
        if status in ['people', 'people_kaleidoscope']:
            self.people_mode_entered_time = time.time()
            self.last_people_segment_time = None
        
        animation_config = None
        if status in self.animation_statuses:
            animation_config = self._consume_pending_animation_config(status)
            text = None
            wobble = 0.0
            if config.get('has_text_scroller'):
                if animation_config is None:
                    if status == 'wave':
                        animation_config = self._get_wave_config()
                    else:
                        animation_config = self._get_gesture_config(status)
                if status == 'wave':
                    text, wobble = self._get_wave_text_config(animation_config)
                else:
                    text, wobble = self._get_text_from_gesture_config(animation_config)
            with self.status_lock:
                self.animation_text = text
                self.animation_text_wobble = wobble
        else:
            with self.status_lock:
                self.animation_text = None
                self.animation_text_wobble = 0.0
        
        # Special handling for raw status
        if status == 'raw':
            # Navigate to root page (no sketch_id) so Hydra can iterate through all sketches
            self.current_sparse_sketch = None  # No specific sketch selected
            self.last_sparse_sketch_switch_time = time.time()
            # Update Hydra URL to root page without sketch_id
            self._update_raw_hydra_url()
            # Set duration to 1 minute (60 seconds) for raw status
            self.status_duration = 60.0
        elif status in ['eye', 'people', 'people_kaleidoscope']:
            # Set duration to 10 minutes (600 seconds) for main statuses
            self.status_duration = 600.0
        elif status == 'wave':
            # Wave status - only video duration (text is queued separately)
            gesture_config = animation_config or self._get_wave_config()
            # Wave only has hand_waving substate - text scrolling is queued separately
            self.status_duration = self._calculate_wave_duration(gesture_config)
        elif status == 'think':
            # Think status has custom duration calculation
            video_duration = 0
            if config['video_name'] and self.video_manager.has_video(config['video_name']):
                video_duration = self.video_manager.get_duration(config['video_name'])
            thinking_text_duration = config.get('thinking_text_duration', 10.0)
            self.status_duration = video_duration + thinking_text_duration  # No buffer
        else:
            # Other gesture statuses - get config and calculate duration
            # Get gesture config for all gesture statuses that might have text_scroller
            if status in ['smile', 'thumbs_up', 'peace']:
                gesture_config = self._get_gesture_config(status)
            else:
                gesture_config = None
            # Use general duration calculation
            self.status_duration = self._calculate_status_duration(
                status=status,
                gesture_config=gesture_config,
                video_name=config.get('video_name'),
                video_buffer=0.0,  # No buffer - video plays exactly
                text_buffer=0.0  # No buffer - text exits exactly when rightmost pixel < 0
            )
        
        # Play TTS from config if available
        if status == 'wave':
            gesture_config = self._get_wave_config()
        elif status in ['smile', 'thumbs_up', 'peace']:
            gesture_config = self._get_gesture_config(status)
        else:
            gesture_config = None
            
        if gesture_config:
            tts_response = gesture_config.get('tts_response')
            if tts_response and self.audio_service and self.audio_service.tts_voice:
                self.audio_service.speak(tts_response)
    
    def _exit_status(self, status):
        """Handle status exit logic"""
        # Reset gesture processing flag when exiting gesture statuses
        if status in ['wave', 'thumbs_up', 'smile', 'peace', 'think']:
            with self.gesture_processing_lock:
                self.gesture_processing = False
        
        if status == 'wave':
            # Clean up wave animation state
            self.status_substate = None
            with self.status_lock:
                self.animation_text = None
                self.animation_text_wobble = 0.0
    
    def _has_valid_people_segments(self, mask):
        """Check if mask contains valid people segments (excluding background)
        
        Args:
            mask: Mask array (float 0-1 or uint8 0-255)
            
        Returns:
            True if valid people segments detected, False otherwise
        """
        if mask is None:
            return False
        
        # Convert to binary if needed
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            binary_mask = (mask > 128).astype(np.uint8) * 255
        
        # If no pixels, no segments
        if not np.any(binary_mask > 0):
            return False
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # If no segments found, return False
        if num_labels <= 1:
            return False
        
        total_pixels = mask.shape[0] * mask.shape[1]
        background_threshold = 0.8  # Segments covering >80% are considered background
        
        # Check each component (excluding background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            coverage = area / total_pixels
            
            # Valid people segments should be:
            # 1. At least 20 pixels (small noise filter)
            # 2. Not covering most of the screen (<80% coverage - filters out background)
            if area >= 20 and coverage < background_threshold:
                return True
        
        # No valid people segments found
        return False
    
    def _get_people_timeout_target_status(self):
        """Get the target status for people timeout transition
        
        Returns:
            Status name to switch to (from main_statuses, excluding 'people')
        """
        target_config = self.people_timeout_target_status
        
        # Get available main statuses (excluding 'people')
        available_main = [s for s in self.state_manager.main_statuses if s != 'people']
        
        if not available_main:
            # Fallback: use 'eye' if no main statuses available
            self.logger.warning("No main statuses available for people timeout, defaulting to 'eye'")
            return 'eye'
        
        # If config is 'random' or None, randomly select from available main statuses
        if target_config in ('random', None):
            return random.choice(available_main)
        
        # If config is a specific status name, use it if it's available
        if target_config in available_main:
            return target_config
        
        # If configured status is not available, fall back to random selection
        self.logger.warning(
            f"Configured people_timeout_target_status '{target_config}' not available in main_statuses. "
            f"Available: {available_main}. Using random selection."
        )
        return random.choice(available_main)
    
    def _update_status_transitions(self):
        """Update status transitions and timeouts using the state management system"""
        current_time = time.time()
        next_status = None
        is_people_timeout = False  # Track if this is a people timeout transition
        
        # Check status transitions (need to release lock before calling set_status)
        with self.status_lock:
            elapsed = current_time - self.status_start_time
            
            # Check people detection timeout - switch to configured status if no segments detected for 5 seconds
            # Only check base_status, not animations
            # Skip timeout check if an animation is playing (don't interrupt animations)
            if self.base_status in ['people', 'people_kaleidoscope'] and self.animation_status is None:
                if self.last_people_segment_time is None:
                    # No segments detected yet, check if timeout exceeded
                    if self.people_mode_entered_time is not None:
                        time_since_entered = current_time - self.people_mode_entered_time
                        if time_since_entered >= self.people_timeout_seconds:
                            # Timeout exceeded, determine target status
                            target_status = self._get_people_timeout_target_status()
                            self.logger.info(f"People mode timeout: no segments detected for {time_since_entered:.1f}s, switching to {target_status}")
                            next_status = target_status
                            is_people_timeout = True
                else:
                    # Segments were detected, check if it's been too long since last detection
                    time_since_last_segment = current_time - self.last_people_segment_time
                    if time_since_last_segment >= self.people_timeout_seconds:
                        # Timeout exceeded, determine target status
                        target_status = self._get_people_timeout_target_status()
                        self.logger.info(f"People mode timeout: no segments detected for {time_since_last_segment:.1f}s, switching to {target_status}")
                        next_status = target_status
                        is_people_timeout = True
            
            # Handle timed statuses (animations that have durations)
            # Use get_status() to get the actual current status (handles animation_status properly)
            actual_current_status = self.animation_status if self.animation_status else self.base_status
            if self.status_duration and elapsed >= self.status_duration:
                if actual_current_status == 'think':
                    # Think animation complete - return to previous status
                    with self.status_lock:
                        self.animation_status = None
                        # Use think_return_status if set, otherwise previous_status, otherwise base_status
                        return_status = self.think_return_status if self.think_return_status else (self.previous_status if (self.previous_status and self.previous_status in self.main_statuses) else self.base_status)
                        self.current_status = return_status
                        self.base_status = return_status  # Update base_status to match
                        self.status_start_time = time.time()
                        self.animation_text = None
                        self.animation_text_wobble = 0.0
                    
                    self.think_return_status = None  # Clear return status
                    self.logger.info(f"Think animation complete, returning to previous status: {return_status}")
                    
                    # Process action queue before returning
                    queued_action = self._process_action_queue()
                    if not queued_action:
                        self.status_duration = None  # Clear duration
                    next_status = None  # No transition needed - already on return_status
                elif actual_current_status == 'wave':
                    # Transition wave sub-states (only if text_scroller is configured in CSV)
                    if self.status_substate == 'hand_waving':
                        # Get video duration to determine when to transition
                        video_duration = 0.0
                        if self.video_manager.has_video('hand_waving'):
                            video_duration = self.video_manager.get_duration('hand_waving')
                        
                        # Transition immediately when video ends (no buffer delay)
                        # Use a small tolerance to ensure transition happens even with timing precision issues
                        if video_duration > 0 and elapsed >= (video_duration - 0.05):
                            # Check if text_scroller is configured in CSV
                            with self.status_lock:
                                text = self.animation_text
                                wobble = self.animation_text_wobble
                            if not text:
                                gesture_config = self._get_wave_config()
                                text, wobble = self._get_wave_text_config(gesture_config)
                                with self.status_lock:
                                    self.animation_text = text
                                    self.animation_text_wobble = wobble
                            if text:
                                # Queue text scroller to play after wave animation completes
                                self.queue_action('text_scroller', text=text, wobble=0.0)
                                self.logger.info(f"Wave: queued text scroller '{text[:30]}...' to play after animation")
                                # Wave animation complete - return to previous status
                                with self.status_lock:
                                    self.animation_status = None
                                    # Return to previous_status if valid, otherwise base_status
                                    return_status = self.previous_status if (self.previous_status and self.previous_status in self.main_statuses) else self.base_status
                                    self.current_status = return_status
                                    self.base_status = return_status  # Update base_status to match
                                    self.status_start_time = time.time()
                                    self.animation_text = None
                                    self.animation_text_wobble = 0.0
                                
                                # Reset gesture processing flag
                                with self.gesture_processing_lock:
                                    self.gesture_processing = False
                                
                                # Process action queue (will show text scroller)
                                queued_action = self._process_action_queue()
                                if not queued_action:
                                    self.status_duration = None
                                next_status = None  # No transition needed - already on return_status
                                return  # Exit early, don't continue to else block
                            else:
                                # No text configured in CSV, wave animation complete - return to previous status
                                with self.status_lock:
                                    self.animation_status = None
                                    # Return to previous_status if valid, otherwise base_status
                                    return_status = self.previous_status if (self.previous_status and self.previous_status in self.main_statuses) else self.base_status
                                    self.current_status = return_status
                                    self.base_status = return_status  # Update base_status to match
                                    self.status_start_time = time.time()
                                    self.animation_text = None
                                    self.animation_text_wobble = 0.0
                                
                                # Reset gesture processing flag
                                with self.gesture_processing_lock:
                                    self.gesture_processing = False
                                
                                self.logger.info(f"Wave animation complete (no text_scroller), returning to previous status: {return_status}")
                                
                                # Process action queue before returning
                                queued_action = self._process_action_queue()
                                if not queued_action:
                                    self.status_duration = None
                                next_status = None  # No transition needed - already on return_status
                    # Note: text_scroller substate is no longer used - text is queued separately
                    # This elif block is kept for backwards compatibility but should not be reached
                elif self.base_status in ['eye', 'people', 'raw', 'people_kaleidoscope']:
                    # Main statuses timeout: eye/people/people_kaleidoscope after 10 minutes, raw after 1 minute
                    # Note: people_kaleidoscope also respects people detection timeout (checked above)
                    # Use state management to determine next status
                    # This allows random selection from main_statuses including 'raw'
                    transition_status = self.state_manager.evaluate_transitions(
                        self.current_status, self.status_duration, self.status_start_time
                    )
                    if transition_status:
                        next_status = transition_status
                        self.logger.info(f"{self.current_status} timeout complete, transitioning to: {next_status}")
                    else:
                        # Fallback: randomly choose from other main statuses (excluding current)
                        available_main = [s for s in self.state_manager.main_statuses if s != self.current_status]
                        if available_main:
                            next_status = random.choice(available_main)
                            self.logger.info(f"{self.current_status} timeout complete, randomly selecting: {next_status}")
                        else:
                            # Last resort: cycle to next main status or default to eye
                            next_status = self.previous_status if self.previous_status in ['eye', 'people', 'raw', 'people_kaleidoscope'] else 'eye'
                            self.logger.info(f"{self.current_status} timeout complete, falling back to: {next_status}")
                    # Clear status_duration so the new status can have automatic transitions
                    self.status_duration = None
                elif actual_current_status == 'text_scroller_intent':
                    # Text scroller animation complete - return to previous status
                    with self.status_lock:
                        self.animation_status = None
                        # Return to previous_status if valid, otherwise base_status
                        return_status = self.previous_status if (self.previous_status and self.previous_status in self.main_statuses) else self.base_status
                        self.current_status = return_status
                        self.base_status = return_status  # Update base_status to match
                        self.status_start_time = time.time()
                        self.animation_text = None
                        self.animation_text_wobble = 0.0
                    
                    # Clear stored text
                    self.intent_text_scroller_text = None
                    self.intent_text_scroller_wobble = 0.0
                    self.logger.info(f"Text scroller intent complete, returning to previous status: {return_status}")
                    
                    # Process action queue before returning
                    queued_action = self._process_action_queue()
                    if not queued_action:
                        # Clear status_duration so the new status can have automatic transitions
                        self.status_duration = None
                    next_status = None  # No transition needed - already on return_status
                else:
                    # Animation complete - return to previous status
                    # Clear animation and return to previous status if valid, otherwise base_status
                    with self.status_lock:
                        animation_name = self.animation_status
                        self.animation_status = None
                        # Return to previous_status if valid, otherwise base_status
                        return_status = self.previous_status if (self.previous_status and self.previous_status in self.main_statuses) else self.base_status
                        self.current_status = return_status
                        self.base_status = return_status  # Update base_status to match
                        # Reset status_start_time for return status
                        self.status_start_time = time.time()
                        self.animation_text = None
                        self.animation_text_wobble = 0.0
                    
                    # Reset gesture processing flag for gesture animations
                    if animation_name in ['wave', 'thumbs_up', 'smile', 'peace']:
                        with self.gesture_processing_lock:
                            self.gesture_processing = False
                    
                    self.logger.info(f"{animation_name} animation complete, returning to previous status: {return_status}")
                    
                    # Process action queue before returning to previous status
                    queued_action = self._process_action_queue()
                    if queued_action:
                        # Queue item was processed, don't return to previous status yet
                        # The queued action will handle status transitions
                        next_status = None
                    else:
                        # No queued actions, return to previous status
                        # Don't set next_status - we've already cleared the animation
                        # The return_status is already active, just need to clear duration
                        self.status_duration = None
                        next_status = None  # No transition needed - already on return_status
            
            # Evaluate automatic transitions (time-based, random) if not in a timed status
            if not next_status and not self.status_duration:
                # Log available statuses for debugging
                with self.status_lock:
                    current_base = self.base_status
                    current_animation = self.animation_status
                    all_statuses = list(self.status_config.keys())
                    available_statuses = self.get_available_statuses(self.main_statuses)
                    cooldown_info = self.get_cooldown_info()
                
                # Reduced logging - only log when transitions actually happen
                # Removed verbose DEBUG logs for cooldowns and available statuses
                
                # Only evaluate transitions if we're actually on a base status (not stuck in animation)
                current_status_for_transition = self.get_status()  # Use get_status() to get correct current status
                if current_status_for_transition in self.main_statuses:
                    next_status = self.state_manager.evaluate_transitions(
                        current_status_for_transition, self.status_duration, self.status_start_time
                    )
                    
                    if next_status:
                        self.logger.info(f"[Status Transition] Auto-transition selected: {next_status}")
                    # Removed debug log for "No transition triggered" - too verbose
        
        # Apply status changes outside the lock to avoid deadlock
        if next_status:
            # When returning to a previous status after animation, bypass cooldown
            # Also bypass cooldown for people timeout transitions (automatic timeout, not user-initiated)
            # But respect cooldowns for new transitions
            force = (next_status == self.previous_status) or is_people_timeout
            # Don't pass duration=None - let _enter_status set the correct duration for main statuses
            # For main statuses: eye/people duration = 600s (10 min), raw duration = 60s (1 min)
            # For interactive statuses, duration will be calculated based on video/text
            self.set_status(next_status, duration=None, force=force)
        else:
            # Check for detected gestures via gesture detector module
            # Only check if no animation is currently playing
            with self.status_lock:
                if self.animation_status:
                    # Animation already playing (including text_scroller_intent), skip gesture detection
                    return
            
            # Use lock to prevent multiple gestures from being processed simultaneously
            with self.gesture_processing_lock:
                if self.gesture_processing:
                    # Another gesture is already being processed, skip detection
                    return
                
            gesture_triggered = False
            
            # Check thumbs up
            if self.gesture_detector_module.get_thumbs_up_detected():
                    # Get gesture config to calculate proper duration including text
                    gesture_config = self._get_gesture_config('thumbs_up')
                    if gesture_config:
                        duration = self._calculate_status_duration(
                            status='thumbs_up',
                            gesture_config=gesture_config,
                            video_name='thumbs_up',
                            video_buffer=0.0,  # No buffer - video plays exactly
                            text_buffer=0.0  # No buffer - text exits exactly when rightmost pixel < 0
                        )
                    else:
                        # Fallback: calculate duration without config (will use video duration only)
                        video_duration = 0.0
                        if self.video_manager.has_video('thumbs_up'):
                            video_duration = self.video_manager.get_duration('thumbs_up')
                        duration = video_duration  # No buffer - exact video duration
                    
                    # Check cooldown before triggering (silently skip if on cooldown)
                    is_on_cooldown, _ = self.is_status_on_cooldown('thumbs_up')
                    if not is_on_cooldown:
                        self._set_pending_animation_config('thumbs_up', gesture_config)
                        self.gesture_processing = True
                        success = self.set_status('thumbs_up', duration=duration)
                        if success:
                            gesture_triggered = True
                        else:
                            self.gesture_processing = False
            
            # Check for wave gesture (only if not already in wave status)
            # Check cooldown first before consuming the detection flag
            is_on_cooldown, remaining_cooldown = self.is_status_on_cooldown('wave')
            if self.current_status != 'wave' and not is_on_cooldown and self.gesture_detector_module.get_wave_detected():
                # Get wave config
                gesture_config = self._get_wave_config()
                
                # Calculate duration using helper method
                duration = self._calculate_wave_duration(gesture_config)
                
                # Determine substate from config
                substate = 'hand_waving'
                if gesture_config and gesture_config.get('action'):
                    substate = gesture_config['action'].get('substate', 'hand_waving')
                
                with self.gesture_processing_lock:
                    if not self.gesture_processing:
                        self._set_pending_animation_config('wave', gesture_config)
                        self.gesture_processing = True
                        success = self.set_status('wave', duration=duration, substate=substate)
                        if success:
                            gesture_triggered = True
                            self.logger.info(f"Wave gesture triggered, transitioning to wave status (duration: {duration:.1f}s)")
                        else:
                            self.gesture_processing = False
                            self.logger.warning(f"Wave gesture detected but set_status failed")
                    else:
                        self.logger.debug("Wave gesture detected but another gesture is processing, skipping")
            elif is_on_cooldown and self.current_status != 'wave':
                # Check if wave was detected but we're on cooldown (don't consume the flag)
                # We can't check the flag without consuming it, so we'll just skip
                pass
            
            # Check for smile detection (handled separately via face detector, not gesture registry)
            if self.face_detector_module.get_smile_detected():
                # Get gesture config from CSV intent mappings
                gesture_config = self._get_gesture_config('smile')
                if not gesture_config:
                    self.logger.warning("Smile gesture config not found in CSV - skipping")
                    self.face_detector_module.reset_smile_detected()
                else:
                    # Calculate duration using CSV config
                    config = self.status_config.get('smile', {})
                    video_name = config.get('video_name')
                    duration = self._calculate_status_duration(
                        status='smile',
                        gesture_config=gesture_config,
                        video_name=video_name,
                        video_buffer=0.0,  # No buffer - video plays exactly
                        text_buffer=0.0  # No buffer - text exits exactly when rightmost pixel < 0
                    )
                    
                    # Check cooldown before triggering
                    is_on_cooldown, _ = self.is_status_on_cooldown('smile')
                    if not is_on_cooldown:
                        self._set_pending_animation_config('smile', gesture_config)
                        self.gesture_processing = True
                        success = self.set_status('smile', duration=duration)
                        if success:
                            # Trigger TTS from gesture config
                            tts_response = gesture_config.get('tts_response')
                            if tts_response and self.audio_service and self.audio_service.tts_voice:
                                self.audio_service.speak(tts_response)
                            self.face_detector_module.reset_smile_detected()
                            gesture_triggered = True
                        else:
                            self.gesture_processing = False
                    else:
                        # Reset smile detected flag even if on cooldown to prevent spam
                        self.face_detector_module.reset_smile_detected()
            
            # Check for new gesture detections
            self._handle_gesture_detections()
    
    def _handle_gesture_detections(self):
        """Handle all gesture detections dynamically using gesture registry and CSV intents
        
        Note: Smile is handled separately via face detector, not through gesture registry
        """
        if not self.gesture_detector_module:
            return
        
        # Get registered gestures from detector (only AI-based gestures: wave, thumbs_up, peace)
        gesture_registry = getattr(self.gesture_detector_module, 'gesture_registry', {})
        
        # Only process gestures that are in the registry (smile is handled separately)
        for gesture_name, gesture_info in gesture_registry.items():
            # Skip if already in this status
            if self.current_status == gesture_name:
                continue
            
            # Get detection method from registry
            get_method = gesture_info.get('get_method')
            if not get_method:
                continue
            
            # Check if gesture was detected
            detected = False
            try:
                detected = get_method()
            except Exception as e:
                self.logger.warning(f"Error checking {gesture_name} detection: {e}")
                continue
            
            if detected:
                # Get gesture config from CSV intent mappings
                gesture_config = self._get_gesture_config(gesture_name)
                if not gesture_config:
                    self.logger.warning(f"Gesture config not found for {gesture_name} - skipping")
                    continue
                
                # Calculate duration using CSV config
                config = self.status_config.get(gesture_name, {})
                video_name = config.get('video_name')
                duration = self._calculate_status_duration(
                    status=gesture_name,
                    gesture_config=gesture_config,
                    video_name=video_name,
                    video_buffer=config.get('video_buffer', 0.2),
                    text_buffer=config.get('text_buffer', 2.0)  # Increased default to ensure text fully exits
                )
                
                # Check cooldown
                is_on_cooldown, remaining_cooldown = self.is_status_on_cooldown(gesture_name)
                if not is_on_cooldown:
                    # Get substate from gesture config if available
                    substate = None
                    if gesture_config.get('action', {}).get('substate'):
                        substate = gesture_config['action']['substate']
                    
                    self._set_pending_animation_config(gesture_name, gesture_config)
                    success = self.set_status(gesture_name, duration=duration, substate=substate)
                    if success:
                        # Trigger TTS from gesture config if available
                        tts_response = gesture_config.get('tts_response')
                        if tts_response and self.audio_service and self.audio_service.tts_voice:
                            self.audio_service.speak(tts_response)
                        self.logger.info(f"{gesture_name} gesture triggered (duration: {duration:.1f}s)")
                else:
                    self.logger.debug(f"{gesture_name} gesture detected but on cooldown ({remaining_cooldown:.1f}s remaining)")
    
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
        return self.video_manager.load_video(action_name, video_filename)
    
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
        # Check if video exists
        if not self.video_manager.has_video(action_name):
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
        """Render the wave status (hand wave video only - text is queued separately)"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        substate = status_info['substate']
        
        # Wave only has hand_waving substate - text scrolling is queued separately
        # Always render the hand waving video
        return self._render_waving_hand(elapsed)
    
    def _render_waving_hand(self, elapsed):
        """Render hand wave video with intensity masking for 40x30 screen"""
        # Clamp elapsed to video duration to prevent cutting off
        video_duration = 0.0
        if self.video_manager.has_video('hand_waving'):
            video_duration = self.video_manager.get_duration('hand_waving')
        if video_duration > 0:
            elapsed = min(elapsed, video_duration)
        return self._render_action_video('hand_waving', elapsed)
    
    def _render_thumbs_up_status(self):
        """Render the thumbs up status with video and optional scrolling text (from CSV intent mappings)"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        
        # Get video duration to determine when to show text
        video_duration = 0.0
        if self.video_manager.has_video('thumbs_up'):
            video_duration = self.video_manager.get_duration('thumbs_up')
        
        # Show video for first part, then scrolling text if configured
        if video_duration > 0 and elapsed < video_duration:
            # Show video with intensity masking
            return self._render_action_video('thumbs_up', elapsed)
        else:
            # Video has finished, show scrolling text after video
            # Compute elapsed time relative to text phase start (when video ended)
            text_elapsed = max(0.0, elapsed - video_duration) if video_duration > 0 else elapsed
            
            # Get text from stored animation_text (set when entering status)
            with self.status_lock:
                text = self.animation_text
                wobble = self.animation_text_wobble
            
            # Fallback: try to load text from config if not stored
            if not text:
                gesture_config = self._get_gesture_config('thumbs_up')
                text, wobble = self._get_text_from_gesture_config(gesture_config)
                with self.status_lock:
                    if not self.animation_text:
                        self.animation_text = text
                        self.animation_text_wobble = wobble
            
            # Render text scroller if text is available
            if text:
                return self._render_scrolling_text(text, text_elapsed, wobble_amount=0.0)
            
            # No text configured in CSV, return black frame (don't hold last video frame)
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
    
    def _render_gesture_status(self):
        """Generic render method for gesture statuses (peace, heart, rock_on, etc.)"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        status = status_info['status']
        
        # Try to render video if available
        config = self.status_config.get(status, {})
        video_name = config.get('video_name')
        video_duration = 0.0
        if video_name and self.video_manager and self.video_manager.has_video(video_name):
            video_duration = self.video_manager.get_duration(video_name)
            # Only show video if we're still within video duration
            if video_duration > 0 and elapsed < video_duration:
                video_frame = self._render_action_video(video_name, elapsed)
                if video_frame:
                    return video_frame
        
        # Video has finished (or doesn't exist), now render text scroller if available
        # Compute elapsed time relative to text phase start (when video ended)
        text_elapsed = max(0.0, elapsed - video_duration) if video_duration > 0 else elapsed
        
        # Get text from stored animation_text (set when entering status)
        with self.status_lock:
            text = self.animation_text
            wobble = self.animation_text_wobble
        
        # Fallback: try to load text from config if not stored
        if not text:
            gesture_config = self._get_gesture_config(status)
            text, wobble = self._get_text_from_gesture_config(gesture_config)
            with self.status_lock:
                if not self.animation_text:
                    self.animation_text = text
                    self.animation_text_wobble = wobble
        
        # Render text scroller if text is available
        if text:
            return self._render_scrolling_text(text, text_elapsed, wobble_amount=wobble)
        
        # No text configured - return black frame
        return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
    
    def _render_think_status(self):
        """Render the think status with video and thinking text"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        
        # Get video duration
        video_duration = 0
        if self.video_manager.has_video('think'):
            video_duration = self.video_manager.get_duration('think')
        
        # Show video for first part, then thinking text
        if elapsed < video_duration:
            # Show video with intensity masking
            return self._render_action_video('think', elapsed)
        else:
            # Show thinking text after video - decompression/burning man themed
            text_elapsed = elapsed - video_duration
            # Rotate through different thinking messages
            thinking_texts = [
                "processing...",
                "thinking...",
                "decompressing...",
                "the playa provides...",
                "considering...",
                "reflecting...",
                "wandering...",
                "exploring...",
            ]
            # Cycle through texts based on elapsed time
            text_index = int(text_elapsed / 1.5) % len(thinking_texts)
            text = thinking_texts[text_index]
            return self._render_scrolling_text(text, text_elapsed, wobble_amount=0.0)
    
    def _render_smile_status(self):
        """Render the smile status with video and scrolling text (from CSV intent mappings)"""
        status_info = self.get_status_info()
        elapsed = status_info['elapsed']
        
        # Get video duration to determine when to show text
        video_duration = 0
        if self.video_manager.has_video('smile'):
            video_duration = self.video_manager.get_duration('smile')
        
        # Show video for first part, then scrolling text
        if video_duration > 0 and elapsed < video_duration:
            # Show video with intensity masking
            return self._render_action_video('smile', elapsed)
        else:
            # Video has finished, show scrolling text after video
            # Compute elapsed time relative to text phase start (when video ended)
            text_elapsed = max(0.0, elapsed - video_duration) if video_duration > 0 else elapsed
            
            # Get text from stored animation_text (set when entering status)
            with self.status_lock:
                text = self.animation_text
                wobble = self.animation_text_wobble
            
            # Fallback: try to load text from config if not stored
            if not text:
                gesture_config = self._get_gesture_config('smile')
                text, wobble = self._get_text_from_gesture_config(gesture_config)
                with self.status_lock:
                    if not self.animation_text:
                        self.animation_text = text
                        self.animation_text_wobble = wobble
            
            # Render text scroller if text is available
            if text:
                return self._render_scrolling_text(text, text_elapsed, wobble_amount=0.0)
            
            # No text configured in CSV, return black frame (don't hold last video frame)
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
    
    def _render_scrolling_text(self, text: str, elapsed: float, wobble_amount=0.0):
        """Render scrolling text that moves across the screen with wobbly effects.
        
        Uses the Hydra visual as a color source, so text pixels sample colors from the visual.
        
        Args:
            text: Text to display
            elapsed: Elapsed time since text started (seconds)
            wobble_amount: Amount of wobbling effect (default: 1.0)
        """
        # Get Hydra visual to use as color source
        hydra_frame = None
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        return self.text_scroller.render_scrolling_text(
            text=text,
            elapsed=elapsed,
            scroll_speed=20.0,
            wobble_amount=wobble_amount,
            font_size_scale=1.0,
            base_color=(255, 255, 255),  # Fallback to white if no hydra visual
            color_source=hydra_frame  # Use hydra visual as color source
        )
    
    def _render_text_scroller_intent_status(self):
        """Render text scroller for voice intents
        
        This status is triggered when a voice intent has text_scroller_text configured.
        It displays the text and then returns to the previous status.
        """
        try:
            status_info = self.get_status_info()
            elapsed = status_info['elapsed']
            
            # Get stored text (thread-safe access)
            with self.status_lock:
                text = self.intent_text_scroller_text
            
            if text:
                return self._render_scrolling_text(text, elapsed, wobble_amount=0.0)
            else:
                # Fallback: return black frame if no text
                return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        except Exception as e:
            self.logger.error(f"Error rendering text_scroller_intent: {e}")
            import traceback
            traceback.print_exc()
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
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
        
        if dist_from_center < gradient_start:
            return 0.0
        elif dist_from_center > gradient_end:
            return 1.0  # Full brightness
        else:
            # Linear gradient up to full brightness
            gradient = (dist_from_center - gradient_start) / (gradient_end - gradient_start)
            return gradient
    
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
        
        # Check if it's time to show periodic text scroller
        self._check_and_play_text_scroller()
        
        # Check if it's time to rotate sketch
        self._check_and_rotate_sketch()
        
        # Update status transitions and timeouts
        self._update_status_transitions()
        
        # Get current status - animation takes precedence over base status
        with self.status_lock:
            status = self.animation_status if self.animation_status else self.base_status
        
        # Route to appropriate render method based on status configuration
        config = self.status_config.get(status)
        if config and config.get('render_method'):
            try:
                return config['render_method']()
            except Exception as e:
                self.logger.error(f"Error rendering status '{status}': {e}")
                import traceback
                traceback.print_exc()
                # Fallback to base status render
                base_config = self.status_config.get(self.base_status)
                if base_config and base_config.get('render_method'):
                    return base_config['render_method']()
                return self._render_eye_status()
        else:
            # Default to 'eye' status
            return self._render_eye_status()
    
    def _render_eye_status(self):
        """Render the eye status (original 3D eyeball)"""
        # Hydra frame already updated in update() method
        
        # Update eye Hydra frame (second instance for outside area)
        self._update_eye_hydra_frame()
        
        # Cache a copy of the eye Hydra frame once per render cycle to prevent flickering
        # This ensures all pixel lookups use the same frame snapshot
        with self.eye_hydra_frame_lock:
            if self.eye_hydra_frame is not None:
                self.eye_hydra_frame_cached = self.eye_hydra_frame.copy()
            else:
                self.eye_hydra_frame_cached = None
        
        # Update pupil position based on face detection
        self._update_pupil_position()
        
        # Update pupil animation (dilation/constriction)
        self._update_pupil_animation()
        
        # Update blinking animation
        self._update_blinking()
        
        # Render 3D eyeball
        img = self._render_eyeball()
        
        return img
    
    def _init_eye_hydra_instance(self):
        """Initialize second Hydra instance for eye mode outside area"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            import shutil
            import os
            
            # Configure ChromeOptions
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--use-gl=egl")
            chrome_options.add_argument("--enable-webgl")
            chrome_options.add_argument("--ignore-gpu-blocklist")
            chrome_options.add_argument("--window-size=240,160")
            
            # Use webdriver-manager to automatically handle chromedriver
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                self.eye_hydra_driver = webdriver.Chrome(service=service, options=chrome_options)
            except ImportError:
                chromedriver_path = shutil.which('chromedriver')
                if not chromedriver_path:
                    for path in ['/usr/bin/chromedriver', '/usr/local/bin/chromedriver']:
                        if os.path.exists(path):
                            chromedriver_path = path
                            break
                
                if chromedriver_path:
                    service = Service(chromedriver_path)
                    self.eye_hydra_driver = webdriver.Chrome(service=service, options=chrome_options)
                else:
                    self.eye_hydra_driver = webdriver.Chrome(options=chrome_options)
            
            self.eye_hydra_driver.set_window_size(240, 160)
            
            # Load sparse sketches and set initial sketch
            if not self.available_sparse_sketches:
                self.available_sparse_sketches = self._load_all_sparse_sketches()
            
            if self.available_sparse_sketches:
                self.eye_current_sparse_sketch = random.choice(self.available_sparse_sketches)
                self.eye_last_sparse_sketch_switch_time = time.time()
                self._update_eye_hydra_url()
            else:
                # Load default URL if no sketches available
                self.eye_hydra_driver.get(self.eye_hydra_url)
            
            # Hide UI elements
            try:
                time.sleep(2)  # Wait for Hydra to load
                self.eye_hydra_driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.eye_hydra_driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.eye_hydra_driver.execute_script("document.getElementById('info-container').style.display = 'none';")
            except Exception as e:
                self.logger.warning(f"Could not hide UI elements in eye Hydra instance: {e}")
            
            # Start screenshot thread for eye Hydra instance
            from threading import Thread
            self._eye_hydra_screenshot_active = True
            self._eye_hydra_screenshot_thread = Thread(target=self._eye_hydra_screenshot_loop)
            self._eye_hydra_screenshot_thread.daemon = True
            self._eye_hydra_screenshot_thread.start()
            
            self.logger.info("Second Hydra instance initialized for eye mode outside area")
        except Exception as e:
            self.logger.error(f"Error initializing eye Hydra instance: {e}")
            self.eye_hydra_driver = None
    
    def _eye_hydra_screenshot_loop(self):
        """Background thread for capturing screenshots from eye Hydra instance"""
        from PIL import Image
        from io import BytesIO
        import base64
        
        while self._eye_hydra_screenshot_active and self.eye_hydra_driver:
            try:
                # Capture screenshot (blocking operation)
                # Use driver lock to prevent contention with URL updates
                # This will capture the current frame from the Hydra visual, which should be animating
                with self.eye_hydra_driver_lock:
                    if not self.eye_hydra_driver:
                        break
                    image_data = self.eye_hydra_driver.get_screenshot_as_base64()
                
                frame = Image.open(BytesIO(base64.b64decode(image_data)))
                
                # Update cached frame (only if successful)
                # Keep previous frame if capture fails to prevent flickering
                if frame is not None:
                    with self.eye_hydra_frame_lock:
                        self.eye_hydra_frame = frame
                    
            except Exception as e:
                # Log error but continue capturing - don't break the loop
                self.logger.debug(f"Error capturing eye Hydra screenshot: {e}")
                # Brief pause before retrying to avoid rapid error loops
                time.sleep(0.1)
                continue
            
            # Rate limit screenshot capture (same as main instance)
            # This ensures we capture at ~60 FPS (0.016s interval)
            time.sleep(self.screenshot_interval)
    
    def _update_eye_hydra_url(self):
        """Update eye Hydra URL with current sparse sketch (non-blocking)"""
        if not self.eye_current_sparse_sketch:
            return
        
        url = self.eye_hydra_url
        if '?' in url:
            # Remove existing sketch_id if present, then add new one
            base_url = url.split('?')[0]
            params = {}
            if '?' in url:
                query_string = url.split('?', 1)[1]
                for param in query_string.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        if key != 'sketch_id':
                            params[key] = value
            params['sketch_id'] = self.eye_current_sparse_sketch
            url = base_url + '?' + '&'.join([f"{k}={urllib.parse.quote(v)}" for k, v in params.items()])
        else:
            url += f"?sketch_id={urllib.parse.quote(self.eye_current_sparse_sketch)}"
        
        # Update the URL in a background thread to avoid blocking frame capture
        if self.eye_hydra_driver:
            def update_url_async():
                try:
                    # Use driver lock to prevent contention with screenshot capture
                    with self.eye_hydra_driver_lock:
                        if not self.eye_hydra_driver:
                            return
                        self.eye_hydra_driver.get(url)
                    
                    # Wait for page to load (in background thread, won't block screenshot loop)
                    time.sleep(2.0)
                    
                    # Hide UI elements after page loads (also use lock)
                    with self.eye_hydra_driver_lock:
                        if self.eye_hydra_driver:
                            try:
                                self.eye_hydra_driver.execute_script("document.getElementById('modal').style.display = 'none';")
                                self.eye_hydra_driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                                self.eye_hydra_driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                            except Exception as e:
                                self.logger.debug(f"Could not hide UI elements in eye Hydra: {e}")
                except Exception as e:
                    self.logger.error(f"Error updating eye Hydra URL: {e}")
            
            # Start URL update in background thread
            from threading import Thread
            update_thread = Thread(target=update_url_async, daemon=True)
            update_thread.start()
    
    def _update_eye_hydra_frame(self):
        """Update the cached frame from eye Hydra instance and switch sketches periodically"""
        if not self.eye_hydra_driver:
            return
        
        # Check if it's time to switch sketches
        current_time = time.time()
        if (self.eye_last_sparse_sketch_switch_time is None or 
            current_time - self.eye_last_sparse_sketch_switch_time >= self.eye_sparse_sketch_switch_interval):
            
            if self.available_sparse_sketches:
                # Select a new sketch (different from current)
                attempts = 0
                new_sketch = random.choice(self.available_sparse_sketches)
                while (new_sketch == self.eye_current_sparse_sketch and 
                       len(self.available_sparse_sketches) > 1 and 
                       attempts < 10):
                    new_sketch = random.choice(self.available_sparse_sketches)
                    attempts += 1
                
                self.eye_current_sparse_sketch = new_sketch
                self.eye_last_sparse_sketch_switch_time = current_time
                self._update_eye_hydra_url()
                self.logger.debug(f"Switched eye Hydra to new sparse sketch: {self.eye_current_sparse_sketch}")
    
    def _get_eye_hydra_texture_color(self, nx, ny, normalize_brightness=False, target_brightness=0.5):
        """Get color from eye Hydra texture at normalized coordinates
        
        Args:
            nx, ny: Normalized coordinates (-1 to 1)
            normalize_brightness: If True, normalize brightness to target_brightness
            target_brightness: Target brightness level (0.0 to 1.0) when normalizing
        """
        # Use the cached frame copy (updated once per render cycle) to prevent flickering
        # This avoids copying the frame on every pixel lookup and ensures consistency
        frame_copy = self.eye_hydra_frame_cached
        if frame_copy is None:
            return None
        
        try:
            # Convert normalized coordinates (-1 to 1) to pixel coordinates
            hydra_width, hydra_height = frame_copy.size
            
            # Map normalized coords to Hydra coords
            # Center is at (0, 0), so we map to center of frame
            hydra_x = int((nx + 1.0) / 2.0 * hydra_width)
            hydra_y = int((ny + 1.0) / 2.0 * hydra_height)
            
            # Clamp to valid range
            hydra_x = max(0, min(hydra_width - 1, hydra_x))
            hydra_y = max(0, min(hydra_height - 1, hydra_y))
            
            # Get pixel color from the copy
            pixel = frame_copy.getpixel((hydra_x, hydra_y))
            
            # Handle both RGB and RGBA
            if len(pixel) == 4:
                color = np.array(pixel[:3])  # RGB, ignore alpha
            else:
                color = np.array(pixel)
            
            if normalize_brightness:
                # Calculate current brightness (luminance)
                # Using standard luminance formula: 0.299*R + 0.587*G + 0.114*B
                current_brightness = np.dot(color, [0.299, 0.587, 0.114]) / 255.0
                
                if current_brightness > 0.01:  # Avoid division by zero
                    # Scale to target brightness
                    scale_factor = target_brightness / current_brightness
                    color = color * scale_factor
                    color = np.clip(color, 0, 255)
            
            return tuple(color.astype(np.uint8))
        except Exception as e:
            self.logger.debug(f"Error getting eye Hydra texture color: {e}")
            return None
    
    def _update_raw_hydra_url(self):
        """Update Hydra URL to root page (no sketch_id) to iterate through all sketches (non-blocking)"""
        # Navigate to root page without sketch_id so Hydra can iterate through all sketches
        url = self.hydra_url
        # Remove any existing sketch_id parameter
        if '?' in url:
            base_url = url.split('?')[0]
            params = {}
            query_string = url.split('?', 1)[1]
            for param in query_string.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    if key != 'sketch_id':
                        params[key] = value
            if params:
                url = base_url + '?' + '&'.join([f"{k}={urllib.parse.quote(v)}" for k, v in params.items()])
            else:
                url = base_url
        # If no query params, url is already the root page
        
        # Update the URL in a background thread to avoid blocking frame capture
        if self.driver:
            def update_url_async():
                try:
                    self.logger.info(f"Updating raw Hydra URL to root page (no sketch_id): {url}")
                    self.driver.get(url)
                    # Wait for page to load (in background thread, won't block rendering)
                    time.sleep(2.5)
                    # Hide UI elements after page loads
                    try:
                        self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                        self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                        self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                    except Exception as e:
                        self.logger.debug(f"Could not hide UI elements: {e}")
                except Exception as e:
                    self.logger.error(f"Error updating raw Hydra URL: {e}")
            
            # Start URL update in background thread
            from threading import Thread
            update_thread = Thread(target=update_url_async, daemon=True)
            update_thread.start()
    
    def _render_raw_status(self):
        """Render the raw status - displays sketches from root page (no sketch_id), letting Hydra iterate through all sketches"""
        # Hydra will automatically iterate through all sketches when on root page (no sketch_id)
        # No need to manually switch sketches
        
        # Get frame from Hydra (already updated in update() method)
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        if hydra_frame:
            # Resize to match display dimensions
            return hydra_frame.resize((self.width, self.height), Image.LANCZOS)
        else:
            # Return black frame if no Hydra frame available
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
    
    def _render_people_status(self):
        """Render the people status using people masks as Hydra mask source"""
        # Hydra frame already updated in update() method
        
        # Create image with black background
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Get people mask from mask detector (thread-safe double buffering)
        people_mask = self.mask_detector_module.get_mask()
        if people_mask is None:
            return img
        
        # Render Hydra visual masked by people outlines
        # Use vectorized operations for better performance (utilizes all CPU cores)
        
        # Flip mask horizontally (mirror effect) using vectorized operation
        flipped_mask = np.fliplr(people_mask)
        
        # Get Hydra frame as numpy array for faster access
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        if hydra_frame is None:
            return img
        
        # Resize Hydra frame to match target size and convert to numpy array
        hydra_resized = hydra_frame.resize((self.width, self.height), Image.LANCZOS)
        hydra_array = np.array(hydra_resized, dtype=np.float32)
        
        # Apply mask to all pixels at once (vectorized - uses all cores via NumPy)
        # Expand mask to 3 channels for RGB multiplication
        mask_3d = np.stack([flipped_mask] * 3, axis=-1)
        
        # Vectorized multiplication (NumPy will use multiple cores for large arrays)
        masked_result = (hydra_array * mask_3d).astype(np.uint8)
        
        return Image.fromarray(masked_result)
    
    def _render_people_kaleidoscope_status(self):
        """Render people kaleidoscope status - transforms camera input with bottom row as center
        
        Transformation:
        - Bottom row of camera input maps to center of output screen
        - Rest of screen radially rotates around center
        - Distance from center in output = distance from bottom row in input
        - Angle around center determines which column of input to sample
        """
        # Hydra frame already updated in update() method
        
        # Create image with black background
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Get camera frame (thread-safe)
        with self.camera_lock:
            camera_frame = self.current_frame
        
        if camera_frame is None:
            # Fallback to Hydra if no camera
            with self.hydra_frame_lock:
                hydra_frame = self.hydra_frame
            if hydra_frame:
                return hydra_frame.resize((self.width, self.height), Image.LANCZOS)
            return img
        
        # Resize camera frame to match target size
        camera_height, camera_width = camera_frame.shape[:2]
        camera_resized = cv2.resize(camera_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        camera_array = np.array(camera_resized, dtype=np.float32)
        
        # Get people mask from mask detector (thread-safe double buffering)
        people_mask = self.mask_detector_module.get_mask()
        if people_mask is None:
            # No mask available, return Hydra only (no camera pass-through)
            with self.hydra_frame_lock:
                hydra_frame = self.hydra_frame
            if hydra_frame:
                return hydra_frame.resize((self.width, self.height), Image.LANCZOS)
            return img  # Return black if no Hydra
        
        # Filter out background segments (same validation as regular people mode)
        if not self._has_valid_people_segments(people_mask):
            # No valid people segments, return Hydra only (no camera pass-through)
            with self.hydra_frame_lock:
                hydra_frame = self.hydra_frame
            if hydra_frame:
                return hydra_frame.resize((self.width, self.height), Image.LANCZOS)
            return img  # Return black if no Hydra
        
        # Resize mask to match screen size
        mask_resized = cv2.resize(people_mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        # Convert mask to float if needed
        if mask_resized.dtype != np.float32 and mask_resized.dtype != np.float64:
            mask_float = mask_resized.astype(np.float32) / 255.0
        else:
            mask_float = mask_resized.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float = mask_float / 255.0
        
        # Get Hydra frame as color source
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        if hydra_frame is None:
            # No Hydra, use camera directly with mask
            mask_3d = np.stack([mask_float] * 3, axis=-1)
            masked_result = (camera_array * mask_3d).astype(np.uint8)
            return Image.fromarray(masked_result)
        
        # Resize Hydra frame to match target size
        hydra_resized = hydra_frame.resize((self.width, self.height), Image.LANCZOS)
        hydra_array = np.array(hydra_resized, dtype=np.float32)
        
        # Center of output screen
        center_x = self.width / 2.0
        center_y = self.height / 2.0
        
        # Transform to central 30x30 pixels to ensure circle remains inside screen
        kaleidoscope_size = 30
        half_size = kaleidoscope_size / 2.0
        max_distance = half_size  # Maximum distance from center (15 pixels)
        
        # Create transformed output
        transformed_output = np.zeros((self.height, self.width, 3), dtype=np.float32)
        transformed_mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # For each pixel in output, calculate where to sample from input
        for out_y in range(self.height):
            for out_x in range(self.width):
                # Calculate position relative to center
                dx = out_x - center_x
                dy = out_y - center_y
                
                # Calculate distance from center
                distance = np.sqrt(dx**2 + dy**2)
                
                # Only transform pixels within the 30x30 circle
                if distance <= max_distance:
                    # Normalize distance to 0-1 within the circle
                    distance_normalized = distance / max_distance if max_distance > 0 else 0
                    
                    # Calculate angle around center (0 = right, increasing counter-clockwise)
                    angle = np.arctan2(dy, dx)
                    
                    # Map distance to input row:
                    # - Center (distance=0) → bottom row of input (y = height-1)
                    # - Edge (distance=max) → top row of input (y = 0)
                    input_y = int((1.0 - distance_normalized) * (self.height - 1))
                    input_y = np.clip(input_y, 0, self.height - 1)
                    
                    # Map angle to input column:
                    # - Angle determines which column to sample
                    # - Wrap angle to 0-2π, then map to 0-width
                    angle_normalized = (angle + np.pi) / (2 * np.pi)  # 0 to 1
                    input_x = int(angle_normalized * self.width)
                    input_x = np.clip(input_x, 0, self.width - 1)
                    
                    # Sample from input camera frame
                    camera_color = camera_array[input_y, input_x]
                    mask_value = mask_float[input_y, input_x]
                    
                    # Apply mask and add to output
                    if mask_value > 0:
                        # Use Hydra color modulated by camera mask
                        hydra_color = hydra_array[out_y, out_x]
                        # Blend: camera color * mask intensity, with Hydra as base
                        blended_color = camera_color * mask_value + hydra_color * (1.0 - mask_value * 0.5)
                        transformed_output[out_y, out_x] = blended_color
                        transformed_mask[out_y, out_x] = mask_value
                else:
                    # Outside the circle - use Hydra with gradient fade towards edges
                    # Calculate distance from circle edge
                    distance_from_circle = distance - max_distance
                    # Calculate distance to screen edge (normalized)
                    edge_x = min(out_x, self.width - out_x) / (self.width / 2.0)
                    edge_y = min(out_y, self.height - out_y) / (self.height / 2.0)
                    distance_to_edge = min(edge_x, edge_y)  # Closest edge distance (0 to 1)
                    
                    # Gradient: fade from circle edge (1.0) to screen edges (0.0)
                    # Start fade immediately outside circle, reach 0 at screen edges
                    if distance_to_edge < 0.3:  # Near screen edges
                        gradient = distance_to_edge / 0.3  # Fade to 0 at edges
                    else:  # Away from edges
                        gradient = 1.0  # Full brightness
                    
                    # Apply gradient to Hydra color
                    hydra_color = hydra_array[out_y, out_x] * gradient
                    transformed_output[out_y, out_x] = hydra_color
        
        # Convert to uint8
        result = transformed_output.astype(np.uint8)
        
        return Image.fromarray(result)
    
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
                    # Outside eye - show circles emanating from center on original Hydra visual
                    # Get circle mask value (circles emanating from center)
                    current_time = time.time()
                    circle_mask = self._get_circle_mask_value(nx, ny, eye_dist, current_time)
                    
                    # Get background gradient to apply
                    bg_gradient = self._get_background_gradient(nx, ny, eye_dist)
                    
                    # Get color from main Hydra instance (original visual)
                    with self.hydra_frame_lock:
                        hydra_frame = self.hydra_frame
                    
                    if hydra_frame and circle_mask is not None:
                        # Get color from main Hydra visual
                        hydra_color = self._get_hydra_texture_color(nx, ny, normalize_brightness=False)
                        if hydra_color:
                            # Apply both circle mask and gradient mask
                            combined_mask = circle_mask * bg_gradient
                            effect_color = np.array(hydra_color, dtype=float) * combined_mask
                            pixels[y, x] = tuple(np.clip(effect_color, 0, 255).astype(np.uint8))
                        else:
                            pixels[y, x] = (0, 0, 0)
                    else:
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
        # Cleanup second Hydra instance for eye mode
        if hasattr(self, '_eye_hydra_screenshot_active'):
            self._eye_hydra_screenshot_active = False
        if hasattr(self, '_eye_hydra_screenshot_thread'):
            self._eye_hydra_screenshot_thread.join(timeout=1.0)
        if self.eye_hydra_driver:
            try:
                self.eye_hydra_driver.quit()
            except:
                pass
        
        super().cleanup()
        
        print("Decompression mode cleaned up")
