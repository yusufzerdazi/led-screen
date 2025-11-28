"""
Mischief mode - Combines overlay masking with people detection and eye rendering.

Includes:
- people: Shows people outlines from camera as masks for Hydra visuals
- video_mask: Shows video masked by people outlines
- people_inverted: Same as people but apply the mask OUTSIDE the detected segments
- eye: 3D eyeball with face-tracking pupil from decompression mode
"""

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
from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np
from threading import Thread, Lock, RLock
from queue import Queue, Empty
from typing import List, Dict, Optional, Tuple
from logger import get_logger
from io import BytesIO
import cv2
import math
import random
import urllib.parse
import colorsys

# Enable multi-threading for NumPy and OpenCV
os.environ.setdefault('OPENCV_NUM_THREADS', '0')
os.environ.setdefault('OMP_NUM_THREADS', '0')
os.environ.setdefault('MKL_NUM_THREADS', '0')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '0')

try:
    cv2.setNumThreads(0)
except AttributeError:
    pass


class ColorTransform:
    """Color transformation system for mischief mode"""
    
    # Event color palette (from user specifications)
    PALETTE = {
        'pink': {'rgb': (255, 0, 202), 'hex': '#FF00CA', 'hsv': (312/360, 1.0, 1.0)},  # Hue: 312, Sat: 100, Luma: 100
        'purple': {'rgb': (144, 0, 166), 'hex': '#9000A6', 'hsv': (292/360, 1.0, 0.65)},  # Hue: 292, Sat: 100, Luma: 65
        'cyan': {'rgb': (89, 224, 255), 'hex': '#59E0FF', 'hsv': (191/360, 0.65, 1.0)},  # Hue: 191, Sat: 65, Luma: 100
        'blue': {'rgb': (0, 85, 230), 'hex': '#0055E6', 'hsv': (218/360, 1.0, 0.90)},  # Hue: 218, Sat: 100, Luma: 90
    }
    
    # Pre-compute RGB arrays for faster processing
    PALETTE_RGB = np.array([
        PALETTE['pink']['rgb'],
        PALETTE['purple']['rgb'],
        PALETTE['cyan']['rgb'],
        PALETTE['blue']['rgb'],
    ], dtype=np.float32)
    
    def __init__(self, mode=1, color_x='pink', color_y='cyan', color_z='blue', two_color_x='pink', two_color_y='cyan', enabled=True, logger=None):
        """
        Initialize color transform
        
        Args:
            mode: Transformation mode (1, 2, 3, or 4)
            color_x: Color for R channel (mode 1)
            color_y: Color for G channel (mode 1)
            color_z: Color for B channel (mode 1)
            two_color_x: First color for mode 3
            two_color_y: Second color for mode 3
            enabled: Whether color transformation is enabled (default: True)
        """
        self.mode = mode
        self.color_x = color_x
        self.color_y = color_y
        self.color_z = color_z
        self.two_color_x = two_color_x
        self.two_color_y = two_color_y
        self.enabled = enabled
        self.logger = logger  # Logger for Rich console
        
        # Pre-compute color mappings
        self._setup_color_mappings()
    
    def _setup_color_mappings(self):
        """Pre-compute color mappings for faster processing"""
        # Store as (3,) arrays for broadcasting
        self.color_x_rgb = np.array(self.PALETTE[self.color_x]['rgb'], dtype=np.float32)
        self.color_y_rgb = np.array(self.PALETTE[self.color_y]['rgb'], dtype=np.float32)
        self.color_z_rgb = np.array(self.PALETTE[self.color_z]['rgb'], dtype=np.float32)
        self.two_color_x_rgb = np.array(self.PALETTE[self.two_color_x]['rgb'], dtype=np.float32)
        self.two_color_y_rgb = np.array(self.PALETTE[self.two_color_y]['rgb'], dtype=np.float32)
    
    def transform_image(self, img_array):
        """
        Transform image array to event color palette
        
        Args:
            img_array: numpy array of shape (height, width, 3) with RGB values 0-255
            
        Returns:
            Transformed image array
        """
        # If disabled, return original
        if not self.enabled:
            return img_array
        
        # Handle empty or invalid arrays
        if img_array is None or img_array.size == 0:
            if self.logger:
                self.logger.warning("Empty or None image array")
            return img_array
        
        # Ensure array has correct shape
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            if self.logger:
                self.logger.warning(f"Invalid shape {img_array.shape}, expected (H, W, 3)")
            return img_array
        
        # Convert to uint8 if needed
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Check if image is all black (skip transformation to avoid issues)
        if np.all(img_array == 0):
            return img_array
        
        try:
            if self.mode == 1:
                result = self._mode1_rgb_interpolation(img_array)
            elif self.mode == 2:
                result = self._mode2_closest_color(img_array)
            elif self.mode == 3:
                result = self._mode3_two_colors(img_array)
            elif self.mode == 4:
                result = self._mode4_tbd(img_array)
            else:
                result = img_array
            
            # Ensure result is valid
            if result is None or result.size == 0:
                if self.logger:
                    self.logger.error("Result is None or empty, returning original")
                return img_array
            
            # Ensure result is uint8
            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Check if result is all black (this would be a problem)
            if np.all(result == 0):
                if self.logger:
                    self.logger.warning("Result is all black! Returning original image.")
                return img_array
            
            return result
        except Exception as e:
            # If transformation fails, return original
            if self.logger:
                self.logger.error(f"Color transformation error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            return img_array
    
    def _mode1_rgb_interpolation(self, img_array):
        """Mode 1: RGB channels map to different colors with interpolation"""
        # Normalize input to 0-1 (img_array is 0-255 uint8)
        img_float = img_array.astype(np.float32) / 255.0
        
        # Extract RGB channels
        r = img_float[:, :, 0:1]  # Keep dimension for broadcasting
        g = img_float[:, :, 1:2]
        b = img_float[:, :, 2:3]
        
        # Calculate brightness (luminance) of original pixel
        # Using standard luminance formula: 0.299*R + 0.587*G + 0.114*B
        brightness = (0.299 * r + 0.587 * g + 0.114 * b)
        
        # Interpolate between colors based on RGB values
        # Each channel contributes to the final color
        # Normalize contributions so they sum to 1
        total = r + g + b
        total = np.where(total > 0, total, 1.0)  # Avoid division by zero
        
        r_weight = r / total
        g_weight = g / total
        b_weight = b / total
        
        # Blend the three colors (reshape color arrays for proper broadcasting)
        # color_x_rgb is (3,), r_weight is (H, W, 1), so we need to reshape color to (1, 1, 3)
        color_x_3d = self.color_x_rgb.reshape(1, 1, 3)
        color_y_3d = self.color_y_rgb.reshape(1, 1, 3)
        color_z_3d = self.color_z_rgb.reshape(1, 1, 3)
        
        # Blend colors based on RGB weights
        # Note: color_x_3d etc are in 0-255 range, weights are 0-1, so blended is in 0-255 range
        blended = (r_weight * color_x_3d + 
                  g_weight * color_y_3d + 
                  b_weight * color_z_3d)
        
        # Preserve original brightness
        # Calculate brightness of blended color (blended is in 0-255 range, normalize to 0-1 for brightness calc)
        blended_normalized = blended / 255.0
        blended_brightness = (0.299 * blended_normalized[:, :, 0:1] + 
                             0.587 * blended_normalized[:, :, 1:2] + 
                             0.114 * blended_normalized[:, :, 2:3])
        blended_brightness = np.where(blended_brightness > 0.001, blended_brightness, 0.001)
        
        # Scale to match original brightness
        # brightness is in 0-1 range, blended_brightness is in 0-1 range
        brightness_ratio = brightness / blended_brightness
        
        # Apply brightness ratio to blended (which is in 0-255 range)
        result = blended * brightness_ratio
        
        # Clip to valid range and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Ensure result has correct shape
        if result.shape != img_array.shape:
            if self.logger:
                self.logger.error(f"Mode 1: Shape mismatch! Expected {img_array.shape}, got {result.shape}")
            return img_array
        
        return result
    
    def _mode2_closest_color(self, img_array):
        """Mode 2: Snap to closest color in palette, preserving brightness"""
        # Convert to HSV for better color distance calculation
        img_float = img_array.astype(np.float32) / 255.0
        height, width = img_array.shape[:2]
        
        # Calculate brightness (luminance) of original
        brightness = (0.299 * img_float[:, :, 0] + 
                     0.587 * img_float[:, :, 1] + 
                     0.114 * img_float[:, :, 2])
        
        # Convert to HSV for color distance
        hsv_array = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                r, g, b = img_float[y, x]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv_array[y, x] = [h, s, v]
        
        # Find closest color in palette for each pixel
        result = np.zeros_like(img_array, dtype=np.float32)
        for y in range(height):
            for x in range(width):
                pixel_hsv = hsv_array[y, x]
                pixel_brightness = brightness[y, x]
                
                # Calculate distance to each palette color
                min_dist = float('inf')
                closest_color = self.PALETTE['pink']['rgb']
                
                for color_name, color_info in self.PALETTE.items():
                    target_hsv = color_info['hsv']
                    # Calculate HSV distance (wrapping hue)
                    h_diff = min(abs(pixel_hsv[0] - target_hsv[0]), 
                               1.0 - abs(pixel_hsv[0] - target_hsv[0]))
                    s_diff = abs(pixel_hsv[1] - target_hsv[1])
                    v_diff = abs(pixel_hsv[2] - target_hsv[2])
                    
                    # Weighted distance (hue is most important)
                    dist = (h_diff * 2.0 + s_diff + v_diff) / 4.0
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_color = color_info['rgb']
                
                # Use closest color but adjust brightness
                closest_rgb = np.array(closest_color, dtype=np.float32) / 255.0
                closest_brightness = (0.299 * closest_rgb[0] + 
                                    0.587 * closest_rgb[1] + 
                                    0.114 * closest_rgb[2])
                
                if closest_brightness > 0:
                    brightness_ratio = pixel_brightness / closest_brightness
                    adjusted_color = closest_rgb * brightness_ratio
                    adjusted_color = np.clip(adjusted_color, 0, 1)
                    result[y, x] = adjusted_color * 255.0
                else:
                    result[y, x] = closest_rgb * 255.0
        
        return result.astype(np.uint8)
    
    def _mode3_two_colors(self, img_array):
        """Mode 3: Map to just 2 colors based on brightness threshold"""
        # Calculate brightness
        img_float = img_array.astype(np.float32) / 255.0
        brightness = (0.299 * img_float[:, :, 0] + 
                     0.587 * img_float[:, :, 1] + 
                     0.114 * img_float[:, :, 2])
        
        # Threshold at 50% brightness
        threshold = 0.5
        mask = brightness >= threshold
        
        # Create result array
        result = np.zeros_like(img_array, dtype=np.float32)
        
        # Map bright pixels to color_x, dark pixels to color_y
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                if mask[y, x]:
                    # Bright -> color_x
                    target_rgb = self.two_color_x_rgb / 255.0
                else:
                    # Dark -> color_y
                    target_rgb = self.two_color_y_rgb / 255.0
                
                # Preserve original brightness
                target_brightness = (0.299 * target_rgb[0] + 
                                   0.587 * target_rgb[1] + 
                                   0.114 * target_rgb[2])
                
                if target_brightness > 0:
                    brightness_ratio = brightness[y, x] / target_brightness
                    adjusted_color = target_rgb * brightness_ratio
                    adjusted_color = np.clip(adjusted_color, 0, 1)
                    result[y, x] = adjusted_color * 255.0
                else:
                    result[y, x] = target_rgb * 255.0
        
        return result.astype(np.uint8)
    
    def _mode4_tbd(self, img_array):
        """Mode 4: Placeholder for future transformation"""
        # For now, just return original
        return img_array
    
    def set_mode(self, mode, **kwargs):
        """Change transformation mode and parameters"""
        self.mode = mode
        if 'color_x' in kwargs:
            self.color_x = kwargs['color_x']
        if 'color_y' in kwargs:
            self.color_y = kwargs['color_y']
        if 'color_z' in kwargs:
            self.color_z = kwargs['color_z']
        if 'two_color_x' in kwargs:
            self.two_color_x = kwargs['two_color_x']
        if 'two_color_y' in kwargs:
            self.two_color_y = kwargs['two_color_y']
        self._setup_color_mappings()


class MischiefMode(WebsiteMode):
    """Mischief mode with people detection, video masking, and eye rendering"""
    
    def __init__(self, width=40, height=30, color_mode=1):
        super().__init__(width, height)
        
        # Set up logger
        self.logger = get_logger("Mischief Mode")
        
        self.hydra_url = "http://localhost:5173"
        
        # Overlay mode (from tush mode changes)
        self.overlay_enabled = False
        self.overlay_image = None
        self.overlay_mask = None
        self._load_overlay()
        
        # Current status and transiti
        # ons
        self.current_status = 'people'  # Default to people mode
        self.status_lock = Lock()
        self.status_start_time = time.time()
        
        # Status durations (in seconds)
        self.status_durations = {
            'people': 600.0,        # 10 minutes
            'eye': 300.0,           # 5 minutes
            'people_inverted': 600.0,  # 10 minutes
            'video_mask': 300.0,    # 5 minutes
        }
        
        # Status cycle order
        self.status_cycle = ['people', 'eye', 'people_inverted', 'video_mask']
        
        # Camera settings
        self.camera_enabled = True
        self.camera = None
        self.camera_thread = None
        self.camera_lock = Lock()
        self.camera_running = False
        self.current_frame = None
        
        # Face detection settings
        self.face_detection_lock = Lock()
        self.target_face_position = None
        self.detected_faces = []
        
        # Detection modules (initialized in init())
        self.face_detector_module: FaceDetector = None
        self.mask_detector_module: MaskDetector = None
        self.gesture_detector_module: GestureDetector = None  # Not used but needed for console_ui compatibility
        
        # Video manager for mischief.mp4
        self.video_manager: VideoManager = None
        self.videos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'videos')
        self.video_mask_start_time = None
        
        # Video configuration (for console_ui compatibility)
        self.video_config = {
            'mischief': {'filename': 'mischief.mp4'},
        }
        self.video_max_frames = {}
        
        # Audio/TTS/STT disabled for mischief mode
        # Create a dummy audio_service object for console_ui compatibility
        class DummyAudioService:
            def __init__(self):
                self.stt_enabled = False
                self.speech_to_text = None
            
            def initialize_stt(self, *args, **kwargs):
                return False
        
        self.audio_service = DummyAudioService()
        self.enable_stt = False
        
        # Status cooldowns (for console_ui compatibility)
        # Mischief mode doesn't use cooldowns, but console_ui expects this
        self.status_cooldowns = {}
        
        # Previous status tracking
        self.previous_status = None
        
        # Color transformation system
        # If color_mode is 0, disable transformation
        if color_mode == 0:
            self.color_transform = ColorTransform(mode=1, enabled=False, logger=self.logger)
            self.color_mode = 0
        else:
            self.color_transform = ColorTransform(mode=color_mode, enabled=True, logger=self.logger)
            self.color_mode = color_mode
        
        # Color mode switching (every 1 minute)
        self.color_mode_start_time = time.time()
        self.color_mode_switch_interval = 60.0  # 1 minute in seconds
        self.available_color_modes = [1, 2, 3, 4]  # Available color modes to cycle through
        
        # Hydra frame caching
        self.hydra_frame = None
        self.hydra_frame_lock = Lock()
        
        # Sketch loading for Hydra
        self.available_sketches = []
        self.current_sketch = None
        self.sketch_rotation_interval = 60.0  # Rotate sketch every 60 seconds
        self.last_sketch_change_time = time.time()
        
        # Eye rendering settings (from decompression mode)
        self.eye_center_x = width / 2
        self.eye_center_y = height / 2
        self.eye_radius = min(width, height) * 0.4
        
        # Pupil tracking
        self.pupil_offset_x = 0.0
        self.pupil_offset_y = 0.0
        self.pupil_smoothing = 0.4
        self.pupil_base_radius = 0.12
        self.pupil_current_radius = 0.12
        self.pupil_animation_phase = 0.0
        self.pupil_animation_speed = 0.05
        self.pupil_dilation_range = 0.05
        self.multiple_faces_dilation = 0.15
        
        # Blinking animation
        self.is_blinking = False
        self.blink_progress = 0.0
        self.blink_speed = 0.01
        self.blink_duration = 5.0
        self.blink_start_time = None
        self.last_blink_time = time.time()
        self.blink_interval_min = 20.0
        self.blink_interval_max = 40.0
        self.next_blink_time = time.time() + random.uniform(self.blink_interval_min, self.blink_interval_max)
        
        # Eye element colors
        self.sclera_color = (255, 255, 255)
        self.iris_color = (100, 150, 200)
        self.pupil_color = (0, 0, 0)
        self.highlight_color = (255, 255, 255)
        
        # Eye Hydra instance (for outside area)
        self.eye_hydra_driver = None
        self.eye_hydra_frame = None
        self.eye_hydra_frame_lock = Lock()
        self.eye_hydra_frame_cached = None
        self.eye_hydra_url = self.hydra_url
        self._eye_hydra_screenshot_active = False
        self._eye_hydra_screenshot_thread = None
        
        # Sparse sketches for eye mode
        self.available_sparse_sketches = []
        self.eye_current_sparse_sketch = None
        self.eye_sparse_sketch_switch_interval = 60.0
        self.eye_last_sparse_sketch_switch_time = None
        
        # Circles for eye background (expanding circles effect)
        self.circle_count = 8  # Number of circles
        self.circle_speed = 0.05  # Speed of circle expansion (units per second)
        self.circle_max_radius = 0.2  # Radius of each circle (normalized)
        self.circle_max_distance = 2.5  # Maximum distance circles can travel before respawning
        self.circles = []  # List of circle objects with position and spawn time
        self._initialize_circles()
    
    def _load_overlay(self):
        """Load the overlay image (mischief.png) and prepare it as a mask"""
        try:
            client_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            overlay_path = os.path.join(client_dir, 'mischief.png')
            
            if os.path.exists(overlay_path):
                self.overlay_image = Image.open(overlay_path)
                self.logger.info(f"Loaded overlay image from: {overlay_path}")
            else:
                self.logger.warning(f"Overlay image not found at: {overlay_path}")
                self.overlay_image = None
        except Exception as e:
            self.logger.error(f"Error loading overlay image: {e}")
            self.overlay_image = None
    
    def _prepare_overlay_mask(self, target_width, target_height):
        """Prepare overlay mask scaled to fill screen while maintaining aspect ratio, 10% smaller"""
        if self.overlay_image is None:
            return None
        
        try:
            overlay_width, overlay_height = self.overlay_image.size
            
            # Calculate scale to fill screen (cover mode), then make it 10% smaller
            scale_w = target_width / overlay_width
            scale_h = target_height / overlay_height
            scale = max(scale_w, scale_h) * 0.9
            
            # Calculate new dimensions
            new_width = int(overlay_width * scale)
            new_height = int(overlay_height * scale)
            
            # Resize overlay maintaining aspect ratio
            resized_overlay = self.overlay_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a black canvas of target size with alpha channel
            canvas = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
            
            # Center the resized overlay on the canvas
            left = (target_width - new_width) // 2
            top = (target_height - new_height) // 2
            
            # Ensure resized_overlay has alpha channel
            if resized_overlay.mode != 'RGBA':
                resized_overlay = resized_overlay.convert('RGBA')
            
            # Paste with alpha channel as mask
            canvas.paste(resized_overlay, (left, top), resized_overlay)
            
            # Convert to grayscale mask (alpha channel if available, otherwise convert to grayscale)
            if canvas.mode == 'RGBA':
                mask_array = np.array(canvas.split()[3])
            else:
                mask_array = np.array(canvas.convert('L'))
            
            # Normalize to 0-1 range
            mask_array = mask_array.astype(np.float32) / 255.0
            
            return mask_array
        except Exception as e:
            self.logger.error(f"Error preparing overlay mask: {e}")
            return None
    
    def setup(self, **kwargs):
        """Set up mode configuration"""
        # Load sketches before setting URL
        if not self.available_sketches:
            self.available_sketches = self._load_all_sketches()
            if self.available_sketches:
                self.current_sketch = random.choice(self.available_sketches)
                self.logger.info(f"Loaded {len(self.available_sketches)} sketches, using: {self.current_sketch[:50]}...")
        
        # Build Hydra URL with sketch
        hydra_url = self._build_hydra_url(sketch=self.current_sketch)
        self.url = kwargs.get('url', hydra_url)
        
        if 'status' in kwargs:
            with self.status_lock:
                self.current_status = kwargs['status']
                self.status_start_time = time.time()  # Reset timer when status is manually set
        
        # Update color transform mode if specified
        if 'color_mode' in kwargs:
            self.set_color_mode(kwargs['color_mode'], **kwargs)
    
    def set_color_mode(self, mode, **kwargs):
        """Change color transformation mode
        
        Args:
            mode: Transformation mode (1, 2, 3, or 4)
            **kwargs: Optional color parameters (color_x, color_y, color_z, two_color_x, two_color_y)
        """
        self.color_mode = mode
        self.color_transform.set_mode(mode, **kwargs)
        # Reset timer when color mode is manually changed
        self.color_mode_start_time = time.time()
        self.logger.info(f"Color transform mode set to {mode}")
    
    def switch_visual(self, sketch=None):
        """Manually switch to a different Hydra visual/sketch
        
        Args:
            sketch: Optional sketch name to switch to. If None, chooses a random one.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Reload sketches in case new ones were added
            self.available_sketches = self._load_all_sketches()
            
            if not self.available_sketches:
                self.logger.warning("No sketches available to switch to")
                return False
            
            # Select sketch
            if sketch:
                # Try to find the specified sketch
                matching_sketches = [s for s in self.available_sketches if sketch.lower() in s.lower()]
                if matching_sketches:
                    new_sketch = matching_sketches[0]
                else:
                    self.logger.warning(f"Sketch '{sketch}' not found, choosing random")
                    new_sketch = random.choice(self.available_sketches)
            else:
                # Choose random sketch (different from current if possible)
                if len(self.available_sketches) > 1:
                    new_sketch = random.choice([s for s in self.available_sketches if s != self.current_sketch])
                else:
                    new_sketch = self.available_sketches[0]
            
            # Update sketch
            self.current_sketch = new_sketch
            self.last_sketch_change_time = time.time()
            
            # Update Hydra URL with new sketch
            new_url = self._build_hydra_url(sketch=self.current_sketch)
            if self.driver and new_url != self.url:
                self.url = new_url
                self.driver.get(new_url)
                time.sleep(2)
                # Hide UI elements
                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                self.logger.info(f"Switched to sketch: {self.current_sketch[:50]}...")
                return True
            else:
                self.logger.warning("Driver not available or URL unchanged")
                return False
        except Exception as e:
            self.logger.error(f"Error switching visual: {e}")
            return False
    
    def init(self):
        """Initialize mode and start detection modules"""
        # Load sketches if not already loaded
        if not self.available_sketches:
            self.available_sketches = self._load_all_sketches()
            if self.available_sketches:
                self.current_sketch = random.choice(self.available_sketches)
                self.logger.info(f"Loaded {len(self.available_sketches)} sketches, using: {self.current_sketch[:50]}...")
        
        # Update URL with sketch if available
        if self.current_sketch:
            self.url = self._build_hydra_url(sketch=self.current_sketch)
        
        super().init()
        
        # Hide UI elements after page loads (like decompression mode)
        if self.driver:
            try:
                time.sleep(2)  # Wait for Hydra to load
                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                self.logger.info("Hidden Hydra UI elements")
            except Exception as e:
                self.logger.warning(f"Could not hide UI elements: {e}")
        
        # Initialize detection modules
        try:
            self.mask_detector_module = MaskDetector(width=self.width, height=self.height)
            self.logger.info("Mask detector initialized")
        except Exception as e:
            self.logger.error(f"Error initializing mask detector: {e}")
            self.mask_detector_module = None
        
        try:
            self.face_detector_module = FaceDetector()
            self.logger.info("Face detector initialized")
        except Exception as e:
            self.logger.error(f"Error initializing face detector: {e}")
            self.face_detector_module = None
        
        # Initialize video manager for mischief.mp4
        try:
            self.logger.info("Initializing video manager...")
            self.video_manager = VideoManager(self.videos_dir)
            
            # Check if mischief.mp4 exists in client directory
            client_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mischief_video_path = os.path.join(client_dir, 'mischief.mp4')
            
            # Try loading from videos directory first
            if self.video_manager.load_video('mischief', 'mischief.mp4'):
                self.logger.info("Loaded mischief.mp4 from videos directory")
            elif os.path.exists(mischief_video_path):
                # Video exists in client directory - VideoManager will use fallback method
                self.logger.info("mischief.mp4 found in client directory, will use direct loading")
            else:
                self.logger.warning("mischief.mp4 not found in videos or client directory")
            
            self.logger.info("✓ Video manager initialized")
        except Exception as e:
            self.logger.error(f"Error initializing video manager: {e}")
            self.video_manager = None
        
        # Start camera capture
        if self.camera_enabled:
            self._start_camera()
        
        # Initialize eye Hydra instance if in eye mode
        if self.current_status == 'eye':
            self._init_eye_hydra_instance()
    
    def _start_camera(self):
        """Start camera capture thread"""
        if self.camera_running:
            return
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.logger.warning("Could not open camera")
                return
            
            self.camera_running = True
            self.camera_thread = Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            self.logger.info("Camera started")
        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
    
    def _camera_loop(self):
        """Camera capture loop"""
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    with self.camera_lock:
                        self.current_frame = frame
                    
                    # Feed frame to mask detector for people segmentation
                    if self.mask_detector_module:
                        # Resize frame for faster processing (mask detector expects smaller frames)
                        frame_small = cv2.resize(frame, (160, 120))
                        # Convert BGR to RGB (OpenCV uses BGR, MediaPipe expects RGB)
                        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                        self.mask_detector_module.detect(frame_rgb)
                    
                    # Feed frame to face detector
                    if self.face_detector_module:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.face_detector_module.detect(frame_rgb)
            except Exception as e:
                self.logger.error(f"Error in camera loop: {e}")
                break
    
    def _update_hydra_frame(self):
        """Update the cached frame from Hydra"""
        if self.driver:
            try:
                frame = self.get_frame()
                if frame:
                    with self.hydra_frame_lock:
                        self.hydra_frame = frame
            except Exception as e:
                self.logger.error(f"Error updating Hydra frame: {e}")
    
    def _update_face_position(self):
        """Update target face position from face detector module"""
        if not self.face_detector_module:
            return
        
        face_pos = self.face_detector_module.get_target_position()
        if face_pos:
            # Convert from [0-1, 0-1] to normalized [-1 to 1] with inverted x
            normalized_x = -(face_pos.x - 0.5) * 2.0
            normalized_y = (face_pos.y - 0.5) * 2.0
            
            with self.face_detection_lock:
                self.target_face_position = (normalized_x, normalized_y)
                self.detected_faces = [{'position': (normalized_x, normalized_y)}] * self.face_detector_module.get_face_count()
        else:
            with self.face_detection_lock:
                if self.target_face_position:
                    self.target_face_position = (
                        self.target_face_position[0] * 0.95,
                        self.target_face_position[1] * 0.95
                    )
                self.detected_faces = []
    
    def _has_valid_people_segments(self, mask):
        """Check if mask contains valid people segments"""
        if mask is None:
            return False
        
        # Convert to binary if needed
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            binary_mask = (mask > 128).astype(np.uint8) * 255
        
        if not np.any(binary_mask > 0):
            return False
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        if num_labels <= 1:
            return False
        
        total_pixels = mask.shape[0] * mask.shape[1]
        background_threshold = 0.8
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            coverage = area / total_pixels
            
            if area >= 20 and coverage < background_threshold:
                return True
        
        return False
    
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
    
    def _filter_background_segments(self, mask):
        """Filter out background segments (largest segment if it covers too much of the screen)
        
        Args:
            mask: Mask array (float 0-1 or uint8 0-255)
            
        Returns:
            Filtered mask with background segments removed
        """
        if mask is None:
            return mask
        
        # Convert to binary if needed
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            binary_mask = (mask > 128).astype(np.uint8) * 255
        
        # If no pixels, return original
        if not np.any(binary_mask > 0):
            return mask
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # If no segments found or only background, return original
        if num_labels <= 1:
            return mask
        
        total_pixels = mask.shape[0] * mask.shape[1]
        background_threshold = 0.8  # Segments covering >80% are considered background
        
        # Find the largest segment
        component_areas = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            coverage = area / total_pixels
            component_areas.append((i, area, coverage))
        
        if not component_areas:
            return mask
        
        # Sort by area (largest first)
        component_areas.sort(key=lambda x: x[1], reverse=True)
        
        # Create filtered mask - exclude largest segment if it's too large (background)
        filtered_mask = np.zeros_like(binary_mask)
        
        for i, area, coverage in component_areas:
            # Skip the largest segment if it covers too much (background)
            if i == component_areas[0][0] and coverage >= background_threshold:
                continue
            # Keep all other segments
            filtered_mask[labels == i] = 255
        
        # Convert back to original format
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            filtered_mask = filtered_mask.astype(mask.dtype) / 255.0
        else:
            # Keep as uint8
            filtered_mask = filtered_mask.astype(mask.dtype)
        
        return filtered_mask
    
    def _filter_small_segments(self, mask, min_coverage=0.2):
        """Filter out segments smaller than a minimum coverage threshold
        
        Args:
            mask: Mask array (float 0-1 or uint8 0-255)
            min_coverage: Minimum coverage threshold (0.0 to 1.0). Default 0.2 (1/5 of screen)
            
        Returns:
            Filtered mask with small segments removed
        """
        if mask is None:
            return mask
        
        # Convert to binary if needed
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            binary_mask = (mask > 128).astype(np.uint8) * 255
        
        # If no pixels, return original
        if not np.any(binary_mask > 0):
            return mask
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # If no segments found or only background, return original
        if num_labels <= 1:
            return mask
        
        total_pixels = mask.shape[0] * mask.shape[1]
        
        # Create filtered mask - only keep segments larger than min_coverage
        filtered_mask = np.zeros_like(binary_mask)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            coverage = area / total_pixels
            
            # Only keep segments that are at least min_coverage of the screen
            if coverage >= min_coverage:
                filtered_mask[labels == i] = 255
        
        # Convert back to original format
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            filtered_mask = filtered_mask.astype(mask.dtype) / 255.0
        else:
            # Keep as uint8
            filtered_mask = filtered_mask.astype(mask.dtype)
        
        return filtered_mask
    
    def update(self):
        """Update and return current frame based on current status"""
        # Check and rotate sketch periodically
        self._check_and_rotate_sketch()
        
        # Check and switch color mode periodically (every 5 minutes)
        self._check_and_switch_color_mode()
        
        # Update Hydra frame
        self._update_hydra_frame()
        
        # Update face position (for eye mode)
        self._update_face_position()
        
        # Check and update status transitions
        self._update_status_transitions()
        
        # Route to appropriate render method
        with self.status_lock:
            status = self.current_status
        
        # Track video start time for looping (reset when entering video_mask mode)
        if status == 'video_mask':
            if self.video_mask_start_time is None:
                self.video_mask_start_time = time.time()
        else:
            # Reset video timing when not in video_mask mode
            if self.video_mask_start_time is not None:
                self.video_mask_start_time = None
        
        if status == 'people':
            return self._render_people_status()
        elif status == 'video_mask':
            return self._render_video_mask_status()
        elif status == 'people_inverted':
            return self._render_people_inverted_status()
        elif status == 'eye':
            return self._render_eye_status()
        else:
            return self._render_people_status()  # Default
    
    def _load_all_sketches(self):
        """Load all sketches from sketches.txt"""
        script_dir = os.path.dirname(os.path.abspath(__file__))  # client/modes/
        sketches_file = os.path.join(script_dir, "sketches.txt")
        
        if not os.path.exists(sketches_file):
            self.logger.warning(f"Sketches file not found: {sketches_file}")
            return []
        
        try:
            with open(sketches_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            return lines
        except Exception as e:
            self.logger.error(f"Error loading sketches: {e}")
            return []
    
    def _build_hydra_url(self, sketch=None):
        """Build Hydra URL with sketch_id parameter if available
        
        Args:
            sketch: Optional sketch to use (if None, uses current_sketch or loads a random one)
        """
        url = self.hydra_url
        
        # Use provided sketch or current sketch or load a random one
        if sketch is None:
            if self.current_sketch:
                sketch = self.current_sketch
            elif self.available_sketches:
                sketch = random.choice(self.available_sketches)
                self.current_sketch = sketch
        
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
                    # Select a new sketch (different from current)
                    attempts = 0
                    new_sketch = random.choice(self.available_sketches)
                    while (new_sketch == self.current_sketch and 
                           len(self.available_sketches) > 1 and 
                           attempts < 10):
                        new_sketch = random.choice(self.available_sketches)
                        attempts += 1
                    
                    if new_sketch != self.current_sketch:
                        self.current_sketch = new_sketch
                        self.last_sketch_change_time = current_time
                        
                        # Update Hydra URL with new sketch
                        new_url = self._build_hydra_url(sketch=self.current_sketch)
                        if self.driver and new_url != self.url:
                            try:
                                self.url = new_url
                                self.driver.get(new_url)
                                time.sleep(2)
                                # Hide UI elements
                                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                                self.logger.info(f"Rotated to new sketch: {self.current_sketch[:50]}...")
                            except Exception as e:
                                self.logger.error(f"Error rotating sketch: {e}")
                        else:
                            self.logger.info(f"Rotated to new sketch: {self.current_sketch[:50]}...")
            else:
                # No sketches available, just update time
                self.last_sketch_change_time = current_time
    
    def _check_and_switch_color_mode(self):
        """Check if it's time to switch color mode (every 1 minute)"""
        # Skip if color mode is disabled (mode 0)
        if self.color_mode == 0:
            return
        
        current_time = time.time()
        time_since_change = current_time - self.color_mode_start_time
        
        if time_since_change >= self.color_mode_switch_interval:
            # Time to switch color mode
            try:
                # Find current mode in available modes
                old_mode = self.color_mode
                current_index = self.available_color_modes.index(old_mode)
                # Move to next mode in cycle
                next_index = (current_index + 1) % len(self.available_color_modes)
                next_mode = self.available_color_modes[next_index]
                
                # Switch to next color mode
                self.set_color_mode(next_mode)
                self.color_mode_start_time = current_time
                self.logger.info(f"Color mode switched: {old_mode} -> {next_mode} (after {time_since_change:.1f}s)")
            except ValueError:
                # Current mode not in available modes, reset to first mode
                next_mode = self.available_color_modes[0]
                self.set_color_mode(next_mode)
                self.color_mode_start_time = current_time
                self.logger.warning(f"Current color mode {self.color_mode} not in available modes, resetting to {next_mode}")
    
    def _update_status_transitions(self):
        """Update status transitions based on elapsed time"""
        current_time = time.time()
        
        with self.status_lock:
            elapsed = current_time - self.status_start_time
            current_status = self.current_status
            
            # Get duration for current status
            duration = self.status_durations.get(current_status, 600.0)
            
            # Check if it's time to transition
            if elapsed >= duration:
                # Find current status in cycle
                try:
                    current_index = self.status_cycle.index(current_status)
                    # Move to next status in cycle
                    next_index = (current_index + 1) % len(self.status_cycle)
                    next_status = self.status_cycle[next_index]
                    
                    # Update previous status
                    self.previous_status = current_status
                    
                    # Transition to next status
                    self.current_status = next_status
                    self.status_start_time = current_time
                    
                    # Reset video timing when switching to video_mask
                    if next_status == 'video_mask':
                        self.video_mask_start_time = None
                    
                    self.logger.info(f"Status transition: {current_status} -> {next_status} (after {elapsed:.1f}s)")
                except ValueError:
                    # Current status not in cycle, default to first in cycle
                    self.current_status = self.status_cycle[0]
                    self.status_start_time = current_time
                    self.logger.warning(f"Current status '{current_status}' not in cycle, resetting to '{self.status_cycle[0]}'")
    
    def get_status(self):
        """Get the current status of mischief mode"""
        with self.status_lock:
            return self.current_status
    
    def get_status_info(self):
        """Get detailed status information"""
        with self.status_lock:
            elapsed = time.time() - self.status_start_time
            duration = self.status_durations.get(self.current_status, 600.0)
            remaining = max(0, duration - elapsed)
            return {
                'status': self.current_status,
                'elapsed': elapsed,
                'remaining': remaining,
                'previous': self.previous_status
            }
    
    def get_state_info(self):
        """Get comprehensive state information for console_ui"""
        with self.status_lock:
            elapsed = time.time() - self.status_start_time
            duration = self.status_durations.get(self.current_status, 600.0)
            remaining = max(0, duration - elapsed)
            return {
                'current_status': self.current_status,
                'previous_status': self.previous_status,
                'available_statuses': self.status_cycle,
                'main_statuses': self.status_cycle,
                'interactive_statuses': [],
                'elapsed': elapsed,
                'remaining': remaining,
                'status_start_time': self.status_start_time,
                'status_duration': duration,
            }
    
    def get_cooldown_info(self):
        """Get cooldown information (mischief mode doesn't use cooldowns, return empty)"""
        # Mischief mode doesn't use cooldowns, but console_ui expects this method
        return {}
    
    def _render_people_status(self):
        """Render the people status using people masks as Hydra mask source"""
        # Get Hydra frame first (required)
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        # If no Hydra frame, try to get it directly from parent
        if hydra_frame is None:
            hydra_frame = self.get_frame()
            if hydra_frame:
                with self.hydra_frame_lock:
                    self.hydra_frame = hydra_frame
        
        # If still no Hydra frame, return black
        if hydra_frame is None:
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Resize Hydra frame to match target size
        hydra_resized = hydra_frame.resize((self.width, self.height), Image.LANCZOS)
        hydra_array = np.array(hydra_resized, dtype=np.float32)
        
        # Get people mask from mask detector (optional - if not available, show full Hydra)
        if not self.mask_detector_module:
            transformed = self.color_transform.transform_image(hydra_array.astype(np.uint8))
            return Image.fromarray(transformed)
        
        people_mask = self.mask_detector_module.get_mask()
        if people_mask is None:
            # No mask available, show full Hydra
            transformed = self.color_transform.transform_image(hydra_array.astype(np.uint8))
            return Image.fromarray(transformed)
        
        # Filter out background segments (exclude largest if it's too large)
        people_mask = self._filter_background_segments(people_mask)
        
        # Filter out segments smaller than 1/5 of the screen
        people_mask = self._filter_small_segments(people_mask, min_coverage=0.2)
        
        # Flip mask horizontally (mirror effect)
        flipped_mask = np.fliplr(people_mask)
        
        # Resize mask to match screen size
        mask_resized = cv2.resize(flipped_mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if mask_resized.dtype != np.float32 and mask_resized.dtype != np.float64:
            mask_float = mask_resized.astype(np.float32) / 255.0
        else:
            mask_float = mask_resized.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float = mask_float / 255.0
        
        # Apply mask to all pixels at once (vectorized)
        mask_3d = np.stack([mask_float] * 3, axis=-1)
        masked_result = (hydra_array * mask_3d).astype(np.uint8)
        
        # Apply color transformation
        transformed = self.color_transform.transform_image(masked_result)
        return Image.fromarray(transformed)
    
    def _render_video_mask_status(self):
        """Render mischief.mp4 video masked by people outlines (looped), blended with Hydra visuals"""
        # Get Hydra frame first (for blending)
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        # If no Hydra frame, try to get it directly from parent
        if hydra_frame is None:
            hydra_frame = self.get_frame()
            if hydra_frame:
                with self.hydra_frame_lock:
                    self.hydra_frame = hydra_frame
        
        # Get video frame
        video_frame = None
        if self.video_manager:
            if self.video_mask_start_time is None:
                self.video_mask_start_time = time.time()
            
            elapsed = time.time() - self.video_mask_start_time
            
            # Get video duration for looping
            if self.video_manager.has_video('mischief'):
                duration = self.video_manager.get_duration('mischief')
                if duration > 0:
                    # Loop the video using modulo
                    elapsed_looped = elapsed % duration
                    video_frame = self.video_manager.get_frame('mischief', elapsed_looped)
                else:
                    video_frame = self.video_manager.get_frame('mischief', elapsed)
            else:
                # Fallback: try to load video directly if not in VideoManager
                video_frame = self._get_mischief_video_frame(elapsed)
        
        # If no video frame, fall back to Hydra only
        if video_frame is None:
            if hydra_frame:
                hydra_resized = hydra_frame.resize((self.width, self.height), Image.LANCZOS)
                hydra_array = np.array(hydra_resized, dtype=np.float32)
                transformed = self.color_transform.transform_image(hydra_array.astype(np.uint8))
                return Image.fromarray(transformed)
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Convert video frame to RGB if needed (OpenCV uses BGR)
        if len(video_frame.shape) == 3 and video_frame.shape[2] == 3:
            # Assume it's BGR, convert to RGB
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        
        # Resize video frame to match target size
        video_height, video_width = video_frame.shape[:2]
        video_resized = cv2.resize(video_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        video_array = np.array(video_resized, dtype=np.float32)
        
        # Get Hydra frame for blending
        if hydra_frame:
            hydra_resized = hydra_frame.resize((self.width, self.height), Image.LANCZOS)
            hydra_array = np.array(hydra_resized, dtype=np.float32)
        else:
            # No Hydra, just use video
            hydra_array = None
        
        # Get people mask (optional - if not available, blend video with Hydra or show full video)
        if not self.mask_detector_module:
            # Multiply video with Hydra (white pixels show full Hydra, black shows black)
            if hydra_array is not None:
                # Normalize video to 0-1 range for multiplication
                video_normalized = video_array / 255.0
                # Multiply: white (1.0) becomes full Hydra, black (0.0) becomes black
                blended = (hydra_array * video_normalized).astype(np.uint8)
            else:
                blended = video_array.astype(np.uint8)
            transformed = self.color_transform.transform_image(blended)
            return Image.fromarray(transformed)
        
        people_mask = self.mask_detector_module.get_mask()
        if people_mask is None:
            # No mask available, multiply video with Hydra or show full video
            if hydra_array is not None:
                # Normalize video to 0-1 range for multiplication
                video_normalized = video_array / 255.0
                # Multiply: white (1.0) becomes full Hydra, black (0.0) becomes black
                blended = (hydra_array * video_normalized).astype(np.uint8)
            else:
                blended = video_array.astype(np.uint8)
            transformed = self.color_transform.transform_image(blended)
            return Image.fromarray(transformed)
        
        # Flip mask horizontally (mirror effect)
        flipped_mask = np.fliplr(people_mask)
        
        # Resize mask to match screen size
        mask_resized = cv2.resize(flipped_mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if mask_resized.dtype != np.float32 and mask_resized.dtype != np.float64:
            mask_float = mask_resized.astype(np.float32) / 255.0
        else:
            mask_float = mask_resized.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float = mask_float / 255.0
        
        # Multiply video with Hydra (if available), then apply mask
        if hydra_array is not None:
            # Normalize video to 0-1 range for multiplication
            video_normalized = video_array / 255.0
            # Multiply: white (1.0) becomes full Hydra, black (0.0) becomes black
            blended_base = hydra_array * video_normalized
        else:
            # Just use video
            blended_base = video_array
        
        # Apply mask to blended result (show blended content only where people are detected)
        mask_3d = np.stack([mask_float] * 3, axis=-1)
        masked_result = (blended_base * mask_3d).astype(np.uint8)
        
        # Apply color transformation
        transformed = self.color_transform.transform_image(masked_result)
        
        return Image.fromarray(transformed)
    
    def _get_mischief_video_frame(self, elapsed):
        """Fallback method to load mischief.mp4 directly if not in VideoManager"""
        try:
            mischief_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mischief.mp4')
            if not os.path.exists(mischief_path):
                return None
            
            # Use OpenCV to read video frame
            cap = cv2.VideoCapture(mischief_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            if duration > 0:
                elapsed_looped = elapsed % duration
                frame_number = int(elapsed_looped * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
        except Exception as e:
            self.logger.error(f"Error loading mischief.mp4 directly: {e}")
        
        return None
    
    def _render_people_inverted_status(self):
        """Render people status with mask applied OUTSIDE the detected segments"""
        # Get Hydra frame first (required)
        with self.hydra_frame_lock:
            hydra_frame = self.hydra_frame
        
        # If no Hydra frame, try to get it directly from parent
        if hydra_frame is None:
            hydra_frame = self.get_frame()
            if hydra_frame:
                with self.hydra_frame_lock:
                    self.hydra_frame = hydra_frame
        
        # If still no Hydra frame, return black
        if hydra_frame is None:
            return Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Resize Hydra frame to match target size
        hydra_resized = hydra_frame.resize((self.width, self.height), Image.LANCZOS)
        hydra_array = np.array(hydra_resized, dtype=np.float32)
        
        # Get people mask (optional - if not available, show full Hydra)
        if not self.mask_detector_module:
            transformed = self.color_transform.transform_image(hydra_array.astype(np.uint8))
            return Image.fromarray(transformed)
        
        people_mask = self.mask_detector_module.get_mask()
        if people_mask is None:
            # No mask available, show full Hydra
            transformed = self.color_transform.transform_image(hydra_array.astype(np.uint8))
            return Image.fromarray(transformed)
        
        # Filter out background segments (exclude largest if it's too large)
        filtered_mask = self._filter_background_segments(people_mask)
        
        # Filter out segments smaller than 1/5 of the screen
        filtered_mask = self._filter_small_segments(filtered_mask, min_coverage=0.2)
        
        # Check if filtering removed everything
        if filtered_mask is not None:
            # Check if mask has any non-zero pixels
            if np.any(filtered_mask > 0):
                people_mask = filtered_mask
        
        # Flip mask horizontally (mirror effect)
        flipped_mask = np.fliplr(people_mask)
        
        # Resize Hydra frame to match target size
        hydra_resized = hydra_frame.resize((self.width, self.height), Image.LANCZOS)
        hydra_array = np.array(hydra_resized, dtype=np.float32)
        
        # Resize mask to match screen size
        mask_resized = cv2.resize(flipped_mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if mask_resized.dtype != np.float32 and mask_resized.dtype != np.float64:
            mask_float = mask_resized.astype(np.float32) / 255.0
        else:
            mask_float = mask_resized.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float = mask_float / 255.0
        
        # Invert mask: apply mask OUTSIDE detected segments
        inverted_mask = 1.0 - mask_float
        
        # Apply inverted mask to all pixels at once (vectorized)
        mask_3d = np.stack([inverted_mask] * 3, axis=-1)
        masked_result = (hydra_array * mask_3d).astype(np.uint8)
        
        # Apply color transformation
        transformed = self.color_transform.transform_image(masked_result)
        
        return Image.fromarray(transformed)
    
    def _render_eye_status(self):
        """Render the eye status (3D eyeball with face-tracking pupil)"""
        # Update eye Hydra frame
        self._update_eye_hydra_frame()
        
        # Cache eye Hydra frame
        with self.eye_hydra_frame_lock:
            if self.eye_hydra_frame is not None:
                self.eye_hydra_frame_cached = self.eye_hydra_frame.copy()
            else:
                self.eye_hydra_frame_cached = None
        
        # Update pupil position
        self._update_pupil_position()
        
        # Update pupil animation
        self._update_pupil_animation()
        
        # Update blinking
        self._update_blinking()
        
        # Render 3D eyeball
        return self._render_eyeball()
    
    def _update_pupil_position(self):
        """Update pupil position based on detected face"""
        with self.face_detection_lock:
            if self.target_face_position:
                target_x, target_y = self.target_face_position
                self.pupil_offset_x += (target_x - self.pupil_offset_x) * self.pupil_smoothing
                self.pupil_offset_y += (target_y - self.pupil_offset_y) * self.pupil_smoothing
            else:
                self.pupil_offset_x += (0.0 - self.pupil_offset_x) * self.pupil_smoothing
                self.pupil_offset_y += (0.0 - self.pupil_offset_y) * self.pupil_smoothing
            
            self.pupil_offset_x = max(-1.0, min(1.0, self.pupil_offset_x))
            self.pupil_offset_y = max(-1.0, min(1.0, self.pupil_offset_y))
    
    def _update_pupil_animation(self):
        """Update pupil dilation/constriction animation"""
        self.pupil_animation_phase += self.pupil_animation_speed
        
        with self.face_detection_lock:
            multiple_faces = len(self.detected_faces) > 1
        
        breathing_factor = (math.sin(self.pupil_animation_phase) + 1) / 2
        breathing_variation = breathing_factor * self.pupil_dilation_range
        dilation_bonus = self.multiple_faces_dilation if multiple_faces else 0.0
        
        self.pupil_current_radius = self.pupil_base_radius + breathing_variation + dilation_bonus
        self.pupil_current_radius = max(0.12, min(0.45, self.pupil_current_radius))
    
    def _update_blinking(self):
        """Update blinking animation"""
        current_time = time.time()
        
        if self.is_blinking:
            self.blink_progress += self.blink_speed
            
            if self.blink_start_time is None:
                self.blink_start_time = current_time
            
            elapsed_time = current_time - self.blink_start_time
            time_based_progress = elapsed_time / self.blink_duration
            self.blink_progress = max(self.blink_progress, time_based_progress)
            
            if self.blink_progress >= 1.0:
                self.is_blinking = False
                self.blink_progress = 0.0
                self.blink_start_time = None
                self.next_blink_time = current_time + random.uniform(
                    self.blink_interval_min, self.blink_interval_max
                )
        else:
            if current_time >= self.next_blink_time:
                self.is_blinking = True
                self.blink_progress = 0.0
                self.blink_start_time = current_time
    
    def _render_eyeball(self):
        """Render the 3D eyeball model"""
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Calculate blink factor
        blink_factor = 0.0
        if self.is_blinking:
            if self.blink_progress < 0.3:
                blink_factor = self.blink_progress / 0.3
            else:
                blink_factor = 1.0 - ((self.blink_progress - 0.3) / 0.7)
        
        # Render eye for each pixel
        for y in range(self.height):
            for x in range(self.width):
                nx = (x - self.eye_center_x) / self.eye_radius
                ny = (y - self.eye_center_y) / self.eye_radius
                
                eye_dist = self._get_eye_shape_distance(nx, ny)
                
                if eye_dist <= 1.0:
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
        
        # Apply color transformation to eye rendering
        transformed = self.color_transform.transform_image(pixels)
        return Image.fromarray(transformed)
    
    def _get_eye_shape_distance(self, nx, ny):
        """Calculate distance from center using almond/pointed eye shape"""
        horizontal_stretch = 1.4
        angle = math.atan2(ny, nx)
        corner_sharpness = 0.5
        corner_factor = 1.0 + corner_sharpness * (math.cos(angle) ** 2)
        stretched_x = nx * horizontal_stretch
        stretched_y = ny
        base_dist = math.sqrt(stretched_x * stretched_x + stretched_y * stretched_y)
        shaped_dist = base_dist / corner_factor
        return shaped_dist
    
    def _get_eye_pixel_color_3d(self, nx, ny, blink_factor, eye_shape_dist):
        """Get color for a pixel using true 3D sphere rendering"""
        if blink_factor > 0.95:
            eyelid_color = np.array([50, 40, 30])
            texture = math.sin(nx * 20) * math.sin(ny * 20) * 0.1
            color = eyelid_color * (1.0 + texture)
            return tuple(np.clip(color, 0, 255).astype(np.uint8))
        
        # Apply blink
        ny_3d = ny
        if blink_factor > 0:
            ny_blink = ny / (1.0 - blink_factor * 0.9)
            eye_dist_blink = self._get_eye_shape_distance(nx, ny_blink)
            if eye_dist_blink > 1.0:
                return (50, 40, 30)
            ny_3d = ny_blink
        
        # Project to 3D sphere
        dist_2d = math.sqrt(nx * nx + ny * ny)
        dist_2d = min(0.999, max(0.0, dist_2d))
        z = math.sqrt(1.0 - dist_2d * dist_2d)
        
        if dist_2d > 0.001:
            scale = math.sqrt(1.0 - z * z) / dist_2d
            x_3d = nx * scale
            y_3d = ny_3d * scale
        else:
            x_3d = 0.0
            y_3d = 0.0
            z = 1.0
        
        point_3d = np.array([x_3d, y_3d, z])
        normal = point_3d / np.linalg.norm(point_3d)
        
        # Lighting
        light_dir = np.array([-0.5, -0.5, 0.7])
        light_dir = light_dir / np.linalg.norm(light_dir)
        light_intensity = max(0.3, np.dot(normal, light_dir))
        
        # Pupil position in 3D
        pupil_offset_x_3d = self.pupil_offset_x * 0.7
        pupil_offset_y_3d = self.pupil_offset_y * 0.7
        
        pupil_dir_2d = np.array([pupil_offset_x_3d, pupil_offset_y_3d])
        pupil_dir_2d_len = np.linalg.norm(pupil_dir_2d)
        
        if pupil_dir_2d_len > 0.001:
            pupil_center_2d = np.array([pupil_offset_x_3d, pupil_offset_y_3d])
            pupil_center_2d_len = min(0.8, np.linalg.norm(pupil_center_2d))
            pupil_angle = math.atan2(pupil_center_2d[1], pupil_center_2d[0])
            pupil_dist_2d = pupil_center_2d_len
            pupil_z = math.sqrt(1.0 - pupil_dist_2d * pupil_dist_2d)
            pupil_scale = math.sqrt(1.0 - pupil_z * pupil_z) / pupil_dist_2d if pupil_dist_2d > 0.001 else 0
            pupil_center_3d = np.array([
                math.cos(pupil_angle) * pupil_dist_2d * pupil_scale,
                math.sin(pupil_angle) * pupil_dist_2d * pupil_scale,
                pupil_z
            ])
        else:
            pupil_center_3d = np.array([0.0, 0.0, 1.0])
        
        # Calculate 3D distance from point to pupil center
        dot_product = np.dot(point_3d, pupil_center_3d)
        dot_product = max(-1.0, min(1.0, dot_product))
        sphere_distance = math.acos(dot_product)
        normalized_sphere_dist = sphere_distance / math.pi
        
        # Determine eye part
        pupil_radius = self.pupil_current_radius
        iris_radius = 0.5
        
        if normalized_sphere_dist < pupil_radius:
            # Pupil
            color = np.array(self.pupil_color)
        elif normalized_sphere_dist < iris_radius:
            # Iris
            iris_color = np.array(self.iris_color)
            # Add some texture
            texture = math.sin(normalized_sphere_dist * 20) * 0.1
            color = iris_color * (1.0 + texture) * light_intensity
        else:
            # Sclera
            sclera_color = np.array(self.sclera_color)
            color = sclera_color * light_intensity
        
        # Add highlight
        if normalized_sphere_dist < iris_radius:
            highlight_pos = np.array([-0.3, -0.3, 0.9])
            highlight_dist = np.linalg.norm(point_3d - highlight_pos)
            if highlight_dist < 0.15:
                highlight_intensity = (1.0 - highlight_dist / 0.15) * 0.8
                highlight_color = np.array(self.highlight_color)
                color = color + highlight_color * highlight_intensity
        
        return tuple(np.clip(color, 0, 255).astype(np.uint8))
    
    def _init_eye_hydra_instance(self):
        """Initialize second Hydra instance for eye mode outside area"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            import shutil
            
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--use-gl=egl")
            chrome_options.add_argument("--enable-webgl")
            chrome_options.add_argument("--ignore-gpu-blocklist")
            chrome_options.add_argument("--window-size=240,160")
            
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
            self.eye_hydra_driver.get(self.eye_hydra_url)
            
            try:
                time.sleep(2)
                self.eye_hydra_driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.eye_hydra_driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.eye_hydra_driver.execute_script("document.getElementById('info-container').style.display = 'none';")
            except Exception as e:
                self.logger.warning(f"Could not hide UI elements in eye Hydra instance: {e}")
            
            self._eye_hydra_screenshot_active = True
            self._eye_hydra_screenshot_thread = Thread(target=self._eye_hydra_screenshot_loop)
            self._eye_hydra_screenshot_thread.daemon = True
            self._eye_hydra_screenshot_thread.start()
            
            self.logger.info("Eye Hydra instance initialized")
        except Exception as e:
            self.logger.error(f"Error initializing eye Hydra instance: {e}")
            self.eye_hydra_driver = None
    
    def _eye_hydra_screenshot_loop(self):
        """Screenshot loop for eye Hydra instance"""
        while self._eye_hydra_screenshot_active and self.eye_hydra_driver:
            try:
                screenshot = self.eye_hydra_driver.get_screenshot_as_png()
                frame = Image.open(BytesIO(screenshot))
                with self.eye_hydra_frame_lock:
                    self.eye_hydra_frame = frame
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                self.logger.error(f"Error in eye Hydra screenshot loop: {e}")
                break
    
    def _update_eye_hydra_frame(self):
        """Update the cached frame from eye Hydra instance"""
        # Screenshot loop handles updates
        pass
    
    def cleanup(self):
        """Clean up resources"""
        self.camera_running = False
        if self.camera:
            self.camera.release()
        
        self._eye_hydra_screenshot_active = False
        if self.eye_hydra_driver:
            self.eye_hydra_driver.quit()
        
        if self.mask_detector_module:
            self.mask_detector_module.stop()
        
        if self.face_detector_module:
            self.face_detector_module.stop()
        
        super().cleanup()

