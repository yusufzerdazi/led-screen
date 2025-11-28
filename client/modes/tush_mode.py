"""
Tush/Music visualizer mode.

Displays Hydra visualizations synchronized to music with shape masking and BPM detection.
"""

import time
import json
import base64
import urllib.parse
import numpy as np
from PIL import Image
from threading import Thread, Lock
from io import BytesIO
import random
import math
import os

from .website_mode import WebsiteMode

# Optional dependencies for audio processing
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    print("Warning: pyaudio not installed - audio capture disabled")
    PYAUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not installed - advanced audio processing disabled")
    LIBROSA_AVAILABLE = False


class TushMode(WebsiteMode):
    """Music visualizer mode using Hydra with audio-reactive shape masking"""
    
    def __init__(self, width=40, height=30, debug_overlay=False):
        super().__init__(width, height)
        self.hydra_url = "http://localhost:5173"
        
        # Overlay mode
        self.debug_overlay = debug_overlay
        self.overlay_enabled = debug_overlay
        self.overlay_image = None
        self.overlay_mask = None
        print(f"[TushMode] Initialized with debug_overlay={debug_overlay}, overlay_enabled={self.overlay_enabled}")
        self._load_overlay()
        
        # Animation speed multiplier (1/5 = 0.2 for 5x slower)
        self.animation_speed_multiplier = 0.2
        
        # FPS limiting for frame capture (configured by client)
        self.last_frame_time = 0
        
        # Threading
        self.monitoring_thread = None
        self.visualization_thread = None
        self.monitoring_active = False
        
        # Audio configuration
        self.audio_enabled = PYAUDIO_AVAILABLE
        self.audio_stream = None
        self.audio_thread = None
        self.p = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None
        self.audio_lock = Lock()
        
        # Audio processing
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Audio buffer and processing
        self.audio_buffer = []
        self.buffer_size = 88200  # 2 seconds at 44.1kHz
        self.analysis_interval = 0.5
        self.last_bpm_update = time.time()
        
        # BPM detection
        self.bpm = 135.0  # Default techno BPM
        self.min_bpm = 120
        self.max_bpm = 150
        self.bpm_window_size = 4
        self.bpm_history = []
        self.pulse_speed = 2.25
        self.pulse_phase = 0
        self.last_beat_time = time.time()
        self.bpm_indicator_on = False
        
        # Audio levels
        self.audio_levels = [0.0] * 8
        self.last_audio_update = time.time()
        
        # Visualization settings
        self.visualization_interval = 30
        self.last_visualization_time = time.time()
        self.current_visualization = None
        self.last_visual_load_time = time.time()
        
        # Fade effects
        self.is_fading = False
        self.is_fading_in = False
        self.fade_start_time = None
        self.fade_in_start_time = None
        self.fade_duration = 2.0
        
        # Shape animation
        self.current_shape = 0
        self.target_shape = 0
        self.shape_transition_progress = 0.0
        self.shape_transition_speed = 0.05
        self.shape_size = 6  # Scaled for 40x30 display
        self.shape_positions = []
        self.shape_sizes = []
        self.num_shapes = 3
        self.shape_switch_interval = 15
        self.last_shape_switch = time.time()
        self.shape_movement_offset = 0
        
        # BPM effects
        self.bpm_effect_type = 0
        self.bpm_effect_types = ['size', 'opacity', 'movement']
        self.last_bpm_effect_switch = time.time()
        self.bpm_effect_interval = 20
        
        # Strobe mode
        self.strobe_mode = False
        self.strobe_phase = 0
        self.strobe_frequency = 8.0
        self.last_strobe_toggle = time.time()
        self.strobe_interval = 30
        
        # Speed changes
        self.speed_change_interval = 10
        self.last_speed_change = time.time()
        self.tempo_modes = ['normal', 'double', 'half']
        self.current_tempo_mode = 'normal'
        
        # Rectangle scrolling
        self.rectangle_scroll_offset = 0
        self.rectangle_scroll_speed = 1.0
        
        # Sparse visual detection
        self.sparse_threshold = 0.4
        
        # Hydra documentation cache
        self.hydra_docs_cache = None
        self.hydra_docs_path = "/home/tush/hydra"
        
        # Initialize shape positions
        self._initialize_shapes()
    
    def _load_overlay(self):
        """Load the overlay image (tush.png) and prepare it as a mask"""
        try:
            # Get the client folder path (assuming we're in client/modes/)
            client_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            overlay_path = os.path.join(client_dir, 'tush.png')
            
            if os.path.exists(overlay_path):
                self.overlay_image = Image.open(overlay_path)
                print(f"Loaded overlay image from: {overlay_path}")
                print(f"Overlay image size: {self.overlay_image.size}")
            else:
                print(f"Warning: Overlay image not found at: {overlay_path}")
                self.overlay_image = None
        except Exception as e:
            print(f"Error loading overlay image: {e}")
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
            scale = max(scale_w, scale_h) * 0.9  # 10% smaller (90% of fill size)
            
            # Calculate new dimensions
            new_width = int(overlay_width * scale)
            new_height = int(overlay_height * scale)
            
            # Resize overlay maintaining aspect ratio
            resized_overlay = self.overlay_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a black canvas of target size with alpha channel
            canvas = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))  # Transparent black
            
            # Center the resized overlay on the canvas
            left = (target_width - new_width) // 2
            top = (target_height - new_height) // 2
            
            # Ensure resized_overlay has alpha channel
            if resized_overlay.mode != 'RGBA':
                resized_overlay = resized_overlay.convert('RGBA')
            
            # Paste with alpha channel as mask
            canvas.paste(resized_overlay, (left, top), resized_overlay)
            
            cropped_overlay = canvas
            
            # Convert to grayscale mask (alpha channel if available, otherwise convert to grayscale)
            if cropped_overlay.mode == 'RGBA':
                # Use alpha channel as mask
                mask_array = np.array(cropped_overlay.split()[3])  # Get alpha channel
                print(f"[TushMode] Overlay mask prepared: shape={mask_array.shape}, min={mask_array.min()}, max={mask_array.max()}, mean={mask_array.mean():.2f}")
            else:
                # Convert to grayscale and use as mask
                mask_array = np.array(cropped_overlay.convert('L'))
                print(f"[TushMode] Overlay mask prepared (grayscale): shape={mask_array.shape}, min={mask_array.min()}, max={mask_array.max()}, mean={mask_array.mean():.2f}")
            
            # Normalize to 0-1 range
            mask_array = mask_array.astype(np.float32) / 255.0
            
            return mask_array
        except Exception as e:
            print(f"Error preparing overlay mask: {e}")
            return None
    
    def _initialize_shapes(self):
        """Initialize shape positions and sizes"""
        for i in range(self.num_shapes):
            # Random positions within bounds (scaled for 40x30 display)
            pos = [
                random.randint(self.width // 4, 3 * self.width // 4),
                random.randint(self.height // 4, 3 * self.height // 4)
            ]
            size = random.randint(4, 8)  # Scaled for 40x30 display (was 20-40 for 240x160)
            self.shape_positions.append(pos)
            self.shape_sizes.append(size)
    
    def setup(self, **kwargs):
        """Set up Hydra URL and overlay mode"""
        self.url = kwargs.get('url', self.hydra_url)
        # Allow debug_overlay to be set via setup
        if 'debug_overlay' in kwargs:
            self.debug_overlay = kwargs['debug_overlay']
            self.overlay_enabled = kwargs['debug_overlay']
    
    def init(self):
        """Initialize Hydra and start audio/visualization threads"""
        super().init()
        
        if self.driver:
            try:
                # Hide Hydra UI elements
                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                
                # Auto-run any existing code
                self.driver.execute_script("""
                    if (typeof hydraSynth !== 'undefined' && hydraSynth) {
                        try {
                            eval(hydraSynth.getCode());
                        } catch(e) {
                            console.log('Auto-run failed:', e);
                        }
                    }
                """)
                print("Hydra visualizer initialized")
            except Exception as e:
                print(f"Could not configure Hydra: {e}")
        
        # Start audio capture
        if self.audio_enabled:
            self.start_audio_capture()
        
        # Start monitoring threads
        self.monitoring_active = True
        self.monitoring_thread = Thread(target=self.monitor_visualizations)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.visualization_thread = Thread(target=self.generate_visualizations)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
    
    def update(self):
        """Generate and return the next frame with shape masking"""
        # FPS limiting - don't process frames too frequently
        current_time = time.time()
        if self.frame_interval:
            if current_time - self.last_frame_time < self.frame_interval:
                return None  # Skip this frame to maintain target FPS
        self.last_frame_time = current_time
        
        # Get the frame from WebsiteMode
        frame = super().get_frame()
        
        if frame:
            # Resize to LED dimensions
            frame = frame.resize((self.width, self.height), Image.LANCZOS)
            
            # Apply overlay mask if enabled - skip shape masking in overlay mode
            if self.overlay_enabled:
                if self.overlay_image is None:
                    print("[TushMode] Warning: overlay_enabled=True but overlay_image is None")
                # In overlay mode, ONLY show the visual masked by the overlay, no shape masks
                frame = self.apply_overlay_mask(frame)
            else:
                # Normal mode: Apply shape masking with audio-reactive animations
                frame = self.apply_shape_mask(frame)
        
        return frame
    
    def start_audio_capture(self):
        """Start audio capture thread"""
        if not self.audio_enabled:
            print("Audio capture disabled")
            return
            
        try:
            self.audio_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            # Start audio processing thread
            self.audio_thread = Thread(target=self.audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("Audio capture started successfully")
            
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            self.audio_enabled = False
    
    def audio_capture_loop(self):
        """Main audio capture and processing loop"""
        print("Audio capture loop started")
        
        while self.audio_enabled and self.audio_stream:
            try:
                # Read audio data
                data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Process audio for frequency analysis
                self.process_audio_data(audio_array)
                
            except Exception as e:
                print(f"Error in audio capture loop: {e}")
                break
    
    def process_audio_data(self, audio_data):
        """Process audio data using librosa for BPM detection"""
        try:
            current_time = time.time()
            
            # Add to audio buffer
            self.audio_buffer.extend(audio_data)
            
            # Keep buffer size manageable
            if len(self.audio_buffer) > self.buffer_size * 2:
                self.audio_buffer = self.audio_buffer[-self.buffer_size:]
            
            # Analyze every analysis_interval seconds
            if current_time - self.last_bpm_update >= self.analysis_interval:
                if len(self.audio_buffer) >= self.buffer_size and LIBROSA_AVAILABLE:
                    # Convert to float32 for librosa
                    audio_float = np.array(self.audio_buffer[-self.buffer_size:], dtype=np.float32)
                    audio_float = audio_float / 32768.0  # Normalize to [-1, 1]
                    
                    # Use librosa for BPM detection
                    try:
                        tempo, beats = librosa.beat.beat_track(
                            y=audio_float, 
                            sr=self.RATE,
                            units='time',
                            hop_length=512,
                            start_bpm=135
                        )
                        
                        # Extract scalar value from tempo
                        if hasattr(tempo, '__len__') and len(tempo) > 0:
                            tempo_value = float(tempo[0])
                        else:
                            tempo_value = float(tempo)
                        
                        # Validate BPM is in expected range
                        if self.min_bpm <= tempo_value <= self.max_bpm:
                            self.bpm_history.append(tempo_value)
                            if len(self.bpm_history) > self.bpm_window_size:
                                self.bpm_history.pop(0)
                            
                            # Calculate smoothed BPM
                            smoothed_bpm = np.mean(self.bpm_history)
                            self.bpm = smoothed_bpm
                            self.pulse_speed = self.bpm / 60.0
                            
                    except Exception as e:
                        print(f"Librosa analysis error: {e}")
                
                # Get frequency bands using librosa
                if LIBROSA_AVAILABLE:
                    try:
                        # Convert buffer to librosa format
                        audio_float = np.array(self.audio_buffer[-self.buffer_size:], dtype=np.float32)
                        audio_float = audio_float / 32768.0
                        
                        # Get spectral features
                        mfccs = librosa.feature.mfcc(y=audio_float, sr=self.RATE, n_mfcc=8)
                        
                        # Create 8 frequency bands from spectral features
                        bands = []
                        for i in range(8):
                            if i < len(mfccs):
                                bands.append(np.mean(mfccs[i]))
                            else:
                                bands.append(0.0)
                        
                        # Normalize bands
                        max_band = max(bands) if max(bands) > 0 else 1
                        normalized_bands = [max(0, b / max_band) for b in bands]
                        
                        # Update audio levels
                        with self.audio_lock:
                            self.audio_levels = normalized_bands
                            self.last_audio_update = time.time()
                        
                        # Log high bass
                        bass_level = sum(normalized_bands[:2]) / 2
                        if bass_level > 0.5:
                            print(f"High bass: {bass_level:.2f}")
                            
                    except Exception as e:
                        print(f"Librosa spectral analysis error: {e}")
                
                self.last_bpm_update = current_time
            
        except Exception as e:
            print(f"Error processing audio data: {e}")
    
    def apply_shape_mask(self, im):
        """Apply shape masks that animate with BPM"""
        if im is None:
            return None
            
        # Update shape animation
        self.update_shape_animation()
        
        # Convert to numpy array for faster processing
        img_array = np.array(im)
        # numpy array from PIL has shape (height, width, channels)
        img_height, img_width, img_channels = img_array.shape
        
        # Apply fade effects
        fade_multiplier = 1.0
        current_time = time.time()
        
        if self.is_fading and self.fade_start_time:
            fade_progress = (current_time - self.fade_start_time) / self.fade_duration
            fade_multiplier = max(0.0, 1.0 - fade_progress)
        elif self.is_fading_in and self.fade_in_start_time:
            fade_progress = (current_time - self.fade_in_start_time) / self.fade_duration
            fade_multiplier = min(1.0, fade_progress)
        
        # Create mask array (faster than pixel-by-pixel)
        # Match image array dimensions: (height, width)
        mask = np.zeros((img_height, img_width), dtype=np.float32)
        brightness_mask = np.ones((img_height, img_width), dtype=np.float32)
        
        # Pre-calculate brightness multipliers for each shape
        brightness_mults = []
        for i in range(self.num_shapes):
            pulse_intensity = (math.sin(self.pulse_phase + i * 0.5) + 1) / 2
            if self.bpm_effect_type == 1:  # Opacity effect
                brightness_mult = 0.7 + pulse_intensity * 0.3
            else:
                brightness_mult = 0.8 + pulse_intensity * 0.2
            
            if self.strobe_mode:
                strobe_intensity = (math.sin(self.strobe_phase + i * 0.3) + 1) / 2
                strobe_mult = 0.1 + strobe_intensity * 0.9
                brightness_mult *= strobe_mult
            
            brightness_mults.append(brightness_mult)
        
        # Build mask by checking each shape
        for i in range(self.num_shapes):
            pos = self.shape_positions[i].copy()
            size = self.shape_sizes[i]
            
            # Apply BPM movement offset if in movement mode
            if self.bpm_effect_type == 2:
                pos[0] = min(max(0, pos[0] + self.shape_movement_offset), img_width - 1)
                pos[1] = min(max(0, pos[1] + self.shape_movement_offset), img_height - 1)
            
            # Calculate shape strength for all pixels at once
            for y in range(img_height):
                for x in range(img_width):
                    # Get shape strength with smooth transition
                    if self.current_shape != self.target_shape:
                        current_strength = self.get_shape_strength(x, y, self.current_shape, pos, size)
                        target_strength = self.get_shape_strength(x, y, self.target_shape, pos, size)
                        shape_strength = current_strength * (1 - self.shape_transition_progress) + \
                                       target_strength * self.shape_transition_progress
                    else:
                        shape_strength = self.get_shape_strength(x, y, self.current_shape, pos, size)
                    
                    if shape_strength > 0:
                        # Update mask (take max strength)
                        if shape_strength > mask[y, x]:
                            mask[y, x] = shape_strength
                            brightness_mask[y, x] = brightness_mults[i]
        
        # Apply mask to image using vectorized operations
        mask_3d = np.stack([mask, mask, mask], axis=2)
        brightness_3d = np.stack([brightness_mask, brightness_mask, brightness_mask], axis=2)
        
        # Apply combined effect: inside shapes get brightness, outside get black
        result_array = img_array.astype(np.float32)
        result_array = result_array * mask_3d * brightness_3d * fade_multiplier
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        result = Image.fromarray(result_array)
        
        return result
    
    def apply_overlay_mask(self, im):
        """Apply overlay mask from tush.png to the image - ONLY show visual where mask allows"""
        if im is None:
            return im
        if self.overlay_image is None:
            print("[TushMode] Warning: overlay_image is None, cannot apply overlay mask")
            return im
        
        try:
            # Prepare overlay mask
            mask = self._prepare_overlay_mask(self.width, self.height)
            if mask is None:
                print("[TushMode] Warning: Failed to prepare overlay mask")
                return im
            
            print(f"[TushMode] Applying overlay mask: mask shape={mask.shape}, mask min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")
            
            # Convert image to numpy array
            img_array = np.array(im)
            img_height, img_width, img_channels = img_array.shape
            
            # Ensure mask matches image dimensions
            if mask.shape != (img_height, img_width):
                # Resize mask if needed
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((img_width, img_height), Image.LANCZOS)
                mask = np.array(mask_pil).astype(np.float32) / 255.0
            
            # Start with black background
            result_array = np.zeros((img_height, img_width, img_channels), dtype=np.uint8)
            
            # Apply mask to image - only show visual where mask is non-zero
            mask_3d = np.stack([mask, mask, mask], axis=2)
            masked_visual = img_array.astype(np.float32) * mask_3d
            
            # Composite masked visual onto black background
            result_array = np.clip(masked_visual, 0, 255).astype(np.uint8)
            
            # Convert back to PIL Image
            result = Image.fromarray(result_array)
            return result
        except Exception as e:
            print(f"Error applying overlay mask: {e}")
            return im
    
    def get_shape_strength(self, x, y, shape_type, pos, size):
        """Get the strength (0.0 to 1.0) for a pixel in a given shape with edge gradient"""
        
        if shape_type == 0:  # Circle
            center_x = int(pos[0])
            center_y = int(pos[1])
            radius = int(size)
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            if distance <= radius:
                # Inside circle - gradient from center to edge
                normalized_dist = distance / radius if radius > 0 else 0
                return 1.0 - (normalized_dist * 0.7)  # 0.3 to 1.0
            elif distance <= radius + 3:  # Edge gradient
                # Edge falloff
                edge_dist = distance - radius
                edge_strength = 1.0 - (edge_dist / 3.0)
                return max(0, edge_strength * 0.3)  # Fade to 0 over 3 pixels
            else:
                return 0.0
                
        elif shape_type == 1:  # Square
            center_x = int(pos[0])
            center_y = int(pos[1])
            half_size = int(size)
            
            # Distance from square edge
            dist_x = abs(x - center_x)
            dist_y = abs(y - center_y)
            edge_dist = max(dist_x, dist_y)
            
            if edge_dist <= half_size:
                # Inside square - gradient from center
                normalized_dist = edge_dist / half_size if half_size > 0 else 0
                return 1.0 - (normalized_dist * 0.7)
            elif edge_dist <= half_size + 3:  # Edge gradient
                edge_falloff = edge_dist - half_size
                edge_strength = 1.0 - (edge_falloff / 3.0)
                return max(0, edge_strength * 0.3)
            else:
                return 0.0
                
        elif shape_type == 2:  # Triangle
            center_x = int(pos[0])
            center_y = int(pos[1])
            triangle_size = int(size)
            
            dx = x - center_x
            dy = y - center_y
            
            # Triangle distance calculation
            if abs(dx) <= triangle_size and dy >= -triangle_size and dy <= triangle_size:
                triangle_dist = abs(dx) - (triangle_size - abs(dy))
                if triangle_dist <= 0:  # Inside triangle
                    # Distance from center
                    distance = math.sqrt(dx*dx + dy*dy)
                    normalized_dist = distance / triangle_size if triangle_size > 0 else 0
                    return 1.0 - (normalized_dist * 0.7)
                elif triangle_dist <= 3:  # Edge gradient
                    edge_strength = 1.0 - (triangle_dist / 3.0)
                    return max(0, edge_strength * 0.3)
            
            return 0.0
            
        elif shape_type == 3:  # Rectangle (scrolling)
            rect_width = 3  # Much thinner rectangle
            rect_x = int(self.rectangle_scroll_offset)
            
            if x >= rect_x and x < rect_x + rect_width and y >= 0 and y < self.height:
                # Inside rectangle - gradient from left to right
                local_x = x - rect_x
                return 1.0 - (local_x / rect_width * 0.5)
            elif x >= rect_x - 3 and x < rect_x + rect_width + 3 and y >= 0 and y < self.height:
                # Edge gradient for rectangle
                if x < rect_x:  # Left edge
                    edge_dist = rect_x - x
                    edge_strength = 1.0 - (edge_dist / 3.0)
                else:  # Right edge
                    edge_dist = x - (rect_x + rect_width)
                    edge_strength = 1.0 - (edge_dist / 3.0)
                return max(0, edge_strength * 0.3)
            else:
                return 0.0
                
        elif shape_type == 4:  # Stars - multiple small circles
            # Create multiple small circles scattered around
            star_positions = [
                [pos[0] - size, pos[1] - size],
                [pos[0] + size, pos[1] - size], 
                [pos[0] - size, pos[1] + size],
                [pos[0] + size, pos[1] + size],
                [pos[0], pos[1] - size * 1.5],
                [pos[0], pos[1] + size * 1.5],
                [pos[0] - size * 1.5, pos[1]],
                [pos[0] + size * 1.5, pos[1]]
            ]
            
            max_star_strength = 0.0
            for star_pos in star_positions:
                star_dx = x - star_pos[0]
                star_dy = y - star_pos[1]
                star_distance = math.sqrt(star_dx*star_dx + star_dy*star_dy)
                star_radius = size * 0.3  # Small stars
                
                if star_distance <= star_radius:
                    star_strength = 1.0 - (star_distance / star_radius * 0.5)
                    max_star_strength = max(max_star_strength, star_strength)
            
            return max_star_strength
            
        elif shape_type == 5:  # Heart shape
            center_x = int(pos[0])
            center_y = int(pos[1])
            heart_size = int(size)
            
            # Heart equation: (x^2 + y^2 - 1)^3 - x^2*y^3 = 0
            dx = (x - center_x) / heart_size
            dy = (y - center_y) / heart_size
            
            # Heart shape approximation
            heart_eq = (dx*dx + dy*dy - 1)**3 - dx*dx * dy*dy*dy
            if heart_eq <= 0.1:  # Inside heart
                distance = math.sqrt(dx*dx + dy*dy)
                return 1.0 - (distance * 0.3)
            elif heart_eq <= 0.3:  # Edge gradient
                edge_strength = 1.0 - ((heart_eq - 0.1) / 0.2)
                return max(0, edge_strength * 0.4)
            else:
                return 0.0
                
        elif shape_type == 6:  # Spiral
            center_x = int(pos[0])
            center_y = int(pos[1])
            spiral_size = int(size)
            
            dx = x - center_x
            dy = y - center_y
            distance = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)
            
            # Spiral equation: r = a * θ
            spiral_radius = abs(angle) * spiral_size * 0.3
            spiral_width = spiral_size * 0.2
            
            if abs(distance - spiral_radius) <= spiral_width:
                spiral_strength = 1.0 - (abs(distance - spiral_radius) / spiral_width * 0.5)
                return max(0, spiral_strength)
            else:
                return 0.0
                
        elif shape_type == 7:  # Wave
            center_x = int(pos[0])
            center_y = int(pos[1])
            wave_size = int(size)
            
            # Wave pattern: y = sin(x) with some amplitude
            wave_x = (x - center_x) / wave_size * 4  # Scale for wave frequency
            wave_y = math.sin(wave_x) * wave_size * 0.3
            wave_center_y = center_y + wave_y
            
            # Check if pixel is near the wave
            wave_distance = abs(y - wave_center_y)
            wave_width = wave_size * 0.15
            
            if wave_distance <= wave_width:
                wave_strength = 1.0 - (wave_distance / wave_width * 0.5)
                return max(0, wave_strength)
            else:
                return 0.0
                
        elif shape_type == 8:  # Reserved for future overlay
            # Overlay functionality can be added later
            return 0.0
        
        return 0.0
    
    def update_shape_animation(self):
        """Update shape animations with BPM sync"""
        current_time = time.time()
        
        # Update pulse phase based on BPM
        tempo_multiplier = 1.0
        if self.current_tempo_mode == 'double':
            tempo_multiplier = 2.0
        elif self.current_tempo_mode == 'half':
            tempo_multiplier = 0.5
        
        self.pulse_phase += self.pulse_speed * tempo_multiplier * 0.1 * self.animation_speed_multiplier
        
        # Update strobe phase
        if self.strobe_mode:
            self.strobe_phase += self.strobe_frequency * 0.1 * self.animation_speed_multiplier
        
        # Update rectangle scrolling
        if self.current_shape == 3:  # Rectangle shape
            self.rectangle_scroll_offset += self.rectangle_scroll_speed * self.animation_speed_multiplier
            if self.rectangle_scroll_offset > self.width:
                self.rectangle_scroll_offset = 0
        
        # Shape switching
        if current_time - self.last_shape_switch > self.shape_switch_interval:
            self.target_shape = random.randint(0, 7)  # 8 shapes: 0-7
            self.shape_transition_progress = 0.0
            self.last_shape_switch = current_time
            shape_names = ["circle", "square", "triangle", "rectangle", "stars", "heart", "spiral", "wave"]
            print(f"Switching to shape: {shape_names[self.target_shape]}")
        
        # Smooth transition between shapes
        if self.current_shape != self.target_shape:
            self.shape_transition_progress = min(1.0, self.shape_transition_progress + self.shape_transition_speed * self.animation_speed_multiplier)
            if self.shape_transition_progress >= 1.0:
                self.current_shape = self.target_shape
        
        # BPM effect switching
        if current_time - self.last_bpm_effect_switch > self.bpm_effect_interval:
            self.bpm_effect_type = (self.bpm_effect_type + 1) % len(self.bpm_effect_types)
            self.last_bpm_effect_switch = current_time
            print(f"BPM effect: {self.bpm_effect_types[self.bpm_effect_type]}")
        
        # Update shape positions and sizes based on BPM
        for i in range(self.num_shapes):
            # Pulsing size (scaled for 40x30 display)
            if self.bpm_effect_type == 0:  # Size effect
                pulse_intensity = (math.sin(self.pulse_phase + i * math.pi / self.num_shapes) + 1) / 2
                self.shape_sizes[i] = 3 + pulse_intensity * 7  # 3-10 pixels for 40x30 display
            
            # Movement effect
            if self.bpm_effect_type == 2:  # Movement
                movement_intensity = math.sin(self.pulse_phase * 0.5)
                self.shape_movement_offset = movement_intensity * 10
            
            # Path following for certain shapes
            if self.current_shape in [0, 1, 2]:
                self.follow_path(i)
        
        # Strobe mode toggling
        if current_time - self.last_strobe_toggle > self.strobe_interval:
            self.strobe_mode = not self.strobe_mode
            self.last_strobe_toggle = current_time
            print(f"Strobe mode: {self.strobe_mode}")
        
        # Speed/tempo changes
        if current_time - self.last_speed_change > self.speed_change_interval:
            self.current_tempo_mode = random.choice(self.tempo_modes)
            self.last_speed_change = current_time
            print(f"Tempo mode: {self.current_tempo_mode}")
    
    def follow_path(self, shape_index):
        """Make shapes follow circular or figure-8 paths"""
        # Different path patterns
        path_type = shape_index % 3
        
        if path_type == 0:  # Circular path
            angle = self.pulse_phase + shape_index * 2 * math.pi / self.num_shapes
            radius = min(self.width, self.height) / 4
            center_x = self.width / 2
            center_y = self.height / 2
            self.shape_positions[shape_index][0] = center_x + radius * math.cos(angle)
            self.shape_positions[shape_index][1] = center_y + radius * math.sin(angle)
        elif path_type == 1:  # Figure-8 path
            t = self.pulse_phase * 2 + shape_index * 2 * math.pi / self.num_shapes
            scale = min(self.width, self.height) / 6
            center_x = self.width / 2
            center_y = self.height / 2
            self.shape_positions[shape_index][0] = center_x + scale * math.sin(t)
            self.shape_positions[shape_index][1] = center_y + scale * math.sin(2 * t) / 2
        else:  # Lissajous curve
            t = self.pulse_phase + shape_index * 2 * math.pi / self.num_shapes
            scale = min(self.width, self.height) / 5
            center_x = self.width / 2
            center_y = self.height / 2
            self.shape_positions[shape_index][0] = center_x + scale * math.sin(3 * t)
            self.shape_positions[shape_index][1] = center_y + scale * math.sin(4 * t)
    
    def monitor_visualizations(self):
        """Thread to monitor visualizations"""
        while self.monitoring_active:
            # Monitoring logic can be added here if needed
            time.sleep(0.1)
    
    def generate_visualizations(self):
        """Thread to refresh Hydra page periodically with fade effects"""
        while self.monitoring_active:
            current_time = time.time()
            if current_time - self.last_visualization_time >= self.visualization_interval:
                # Start fade to black
                if not self.is_fading:
                    self.is_fading = True
                    self.fade_start_time = current_time
                    print("Starting fade to black before refresh...")
                
                # Check if fade is complete
                if self.is_fading and current_time - self.fade_start_time >= self.fade_duration:
                    # Fade complete, navigate to refresh visual
                    if self.driver:
                        try:
                            print("Refreshing Hydra visual...")
                            self.driver.get(self.hydra_url)  # Use the correct URL
                            time.sleep(2)
                            # Hide UI elements
                            self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                            self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                            self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                            print("New visual loaded")
                            
                            # Start fade in
                            self.is_fading_in = True
                            self.fade_in_start_time = current_time
                            self.last_visual_load_time = current_time
                            print("Starting fade in...")
                        except Exception as e:
                            print(f"Error loading new visual: {e}")
                    
                    # Reset fade out state
                    self.is_fading = False
                    self.fade_start_time = None
                    self.last_visualization_time = current_time
            
            # Check if fade in is complete
            if self.is_fading_in and self.fade_in_start_time:
                if current_time - self.fade_in_start_time >= self.fade_duration:
                    self.is_fading_in = False
                    self.fade_in_start_time = None
                    print("Fade in complete")
            
            time.sleep(0.5)
    
    def update_audio_levels(self, frequencies):
        """Update audio frequency levels (for MQTT fallback)"""
        self.audio_levels = frequencies[:8]
        self.last_audio_update = time.time()
    
    def update_hydra_code(self, code):
        """Update the Hydra code via URL parameter"""
        if code and self.driver:
            try:
                print(f"Updating Hydra code: {code}")
                new_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
                url = "http://localhost:5173?code=" + urllib.parse.quote_plus(new_code)
                
                self.driver.get(url)
                time.sleep(2)
                # Hide UI elements
                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                print("Updated Hydra code")
            except Exception as e:
                print(f"Error updating Hydra code: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.monitoring_active = False
        
        # Stop audio capture
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.p:
            self.p.terminate()
        
        # Join threads
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        if self.visualization_thread:
            self.visualization_thread.join(timeout=1.0)
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        
        # Clean up browser
        super().cleanup()