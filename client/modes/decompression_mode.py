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


class DecompressionMode(WebsiteMode):
    """Animated 3D eyeball mode with face-tracking pupil and blinking
    
    Renders a 3D eyeball where:
    - The pupil follows detected faces in camera input
    - The eye blinks intermittently
    - Each element (sclera, iris, pupil, highlights) has different colors
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
        
        # Eye 3D model parameters
        self.eye_center_x = width / 2
        self.eye_center_y = height / 2
        self.eye_radius = min(width, height) * 0.4  # Eye sphere radius
        
        # Pupil tracking (normalized to [-1, 1] range)
        self.pupil_offset_x = 0.0  # -1 to 1, left to right
        self.pupil_offset_y = 0.0  # -1 to 1, top to bottom
        self.pupil_smoothing = 0.15  # Smoothing factor for pupil movement
        
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
        self.circle_count = 20  # Number of small circles (glitter particles)
        self.circle_speed = 0.05  # Speed of circle expansion (units per second)
        # Circle diameter: 3-5 pixels. eye_radius is ~12 pixels, so radius is 1.5-2.5px = 0.125-0.208 normalized
        self.circle_max_radius = 0.2  # Radius of each circle (normalized, ~2.4px diameter ~4.8px)
        self.circle_max_distance = 2.5  # Maximum distance circles can travel before respawning
        self.circles = []  # List of circle objects with position and spawn time
        self._initialize_circles()
        
        # Timing
        self.start_time = time.time()
    
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
    
    def init(self):
        """Initialize camera, face detection, and Hydra visuals"""
        print("Initializing decompression mode (3D eyeball with Hydra visuals)...")
        
        # Set URL for WebsiteMode parent - just load the raw Hydra website
        self.url = self.hydra_url
        
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
        
        print("Decompression mode (3D eyeball) initialized")
    
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
        
        # Outside eye - gradient from 0 at edge to 1.0 far away
        # Add some space between eye and background
        gradient_start = eye_edge + 0.15  # Start gradient 0.15 units outside eye
        gradient_end = eye_edge + 0.5    # Full intensity 0.5 units outside
        
        if dist_from_center < gradient_start:
            return 0.0
        elif dist_from_center > gradient_end:
            return 1.0
        else:
            # Linear gradient
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
    
    def update(self):
        """Update and return current frame with 3D eyeball"""
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
                    
                    # Get background gradient
                    bg_gradient = self._get_background_gradient(nx, ny, eye_dist)
                    
                    # Get circle mask (check for circles outside the eye)
                    circle_mask = self._get_circle_mask_value(nx, ny, eye_dist, current_time)
                    
                    if circle_mask is not None and circle_mask > 0.0:
                        # Pixel is inside a circle - show Hydra visual masked by circle AND gradient
                        hydra_color = self._get_hydra_texture_color(nx, ny)
                        if hydra_color:
                            # Apply both circle mask and gradient mask
                            circle_color = np.array(hydra_color) * circle_mask * bg_gradient
                            pixels[y, x] = tuple(np.clip(circle_color, 0, 255).astype(np.uint8))
                        else:
                            # No Hydra visual - show black
                            pixels[y, x] = (0, 0, 0)
                    else:
                        # Not in a circle - show black (no background)
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
                
                # Transform black pixels to white (only inside iris)
                # Check if pixel is black or very dark (brightness < threshold)
                brightness = np.mean(base_color) / 255.0
                if brightness == 0: # black pixesls
                    # Convert to bright white, but respect overall brightness
                    # Use a bright but not maximum white to respect LED brightness settings
                    base_color = np.array([240, 240, 240])  # Bright white (not pure 255)
                
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
        
        print("Decompression mode cleaned up")
