"""
Decompression mode - Emotion-driven kaleidoscopic visual synthesizer.

Captures video from a camera, detects faces and expressions, and transforms
them into evolving kaleidoscopic patterns that respond to emotions, gestures,
and group interactions. Multiple participants can collaborate to merge their
visuals into a collective bloom on the LED canvas.
"""

from .base_mode import BaseMode
from PIL import Image
import time
import numpy as np
from threading import Thread, Lock

# Optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed - camera capture will be disabled")
    print("Install with: pip install opencv-python")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Info: face_recognition not installed - using OpenCV face detection")
    print("For better results, install with: pip install face-recognition")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Info: mediapipe not installed - gesture and advanced tracking disabled")
    print("Install with: pip install mediapipe")


class DecompressionMode(BaseMode):
    """Emotion-aware interactive kaleidoscope mode
    
    Detects facial expressions, hand gestures, and group positioning to drive
    immersive kaleidoscopic visuals on the LED matrix.
    """
    
    def __init__(self, width=40, height=30):
        super().__init__(width, height)
        
        # Camera settings
        self.camera_enabled = CV2_AVAILABLE
        self.camera = None
        self.camera_thread = None
        self.camera_lock = Lock()
        self.camera_running = False
        
        # Face detection settings
        self.use_face_recognition = FACE_RECOGNITION_AVAILABLE
        self.face_cascade = None
        self.current_frame = None
        
        # Face tracking - now supports multiple faces
        self.detected_faces = []  # List of {center, size, expression, intensity}
        
        # Visual synthesis configuration
        self.kaleidoscope_config = {
            'symmetry_order': 6,
            'base_speed': 0.4,
            'noise_scale': 0.8,
            'collective_bloom_strength': 0.6,
            'emotion_decay': 0.85
        }
        self._emotion_memory = []  # Rolling buffer for smoothing
        self._grid_cache = None
        self._rng = np.random.default_rng()

        # Mediapipe components
        self.hands_detector = None

        # Gesture and interaction state
        self.gesture_state = {
            'rotation_velocity': 0.0,
            'hue_shift': 0.0,
            'distortion': 0.0,
            'shatter_active': False,
            'shatter_end': 0.0
        }
        self._last_hand_distance = None
        self.collective_state = {
            'cohesion': 0.0,
            'centroid': (0.5, 0.5),
            'participant_count': 0
        }

        # Timing
        self.start_time = time.time()
        self._rotation_angle = 0.0
        
        # Performance
        self.frame_skip = 2  # Process every Nth frame
        self.frame_count = 0
        
    def init(self):
        """Initialize camera and face detection"""
        if not self.camera_enabled:
            print("ERROR: Camera functionality requires OpenCV (cv2)")
            print("Install with: pip install opencv-python")
            return
        
        print("Initializing decompression mode with camera and facial recognition...")
        
        # Initialize camera
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("ERROR: Could not open camera")
                self.camera_enabled = False
                return
            
            # Set camera resolution (lower for better performance)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize camera: {e}")
            self.camera_enabled = False
            return
        
        # Initialize face detection
        if not self.use_face_recognition:
            # Use OpenCV's Haar Cascade face detector as fallback
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("Using OpenCV Haar Cascade face detection")
            except Exception as e:
                print(f"Warning: Could not load face cascade: {e}")
        else:
            print("Using face_recognition library for face detection")

        # Initialize gesture detection
        if MEDIAPIPE_AVAILABLE:
            try:
                self._mp_hands_module = mp.solutions.hands
                self.hands_detector = self._mp_hands_module.Hands(
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("Mediapipe hands detector initialized")
            except Exception as e:
                self.hands_detector = None
                print(f"Warning: Could not initialize mediapipe hands detector: {e}")
        else:
            self.hands_detector = None
        
        # Start camera capture thread
        self.camera_running = True
        self.camera_thread = Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        print("Decompression mode initialized - camera is running")
    
    def _camera_loop(self):
        """Background thread for camera capture and face detection"""
        while self.camera_running and self.camera:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Warning: Failed to read camera frame")
                    time.sleep(0.1)
                    continue
                
                # Skip frames for performance
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_height, frame_width = frame_rgb.shape[:2]
                
                # Detect multiple faces
                detected_faces = []
                
                if self.use_face_recognition:
                    # Use face_recognition library
                    face_locations = face_recognition.face_locations(frame_rgb)
                    
                    if face_locations:
                        # Get facial landmarks for expression detection
                        face_landmarks_list = face_recognition.face_landmarks(frame_rgb, face_locations)
                        
                        # Process each face
                        for i, (top, right, bottom, left) in enumerate(face_locations):
                            # Analyze expression from landmarks
                            expression = 'neutral'
                            intensity = 0.2
                            
                            if i < len(face_landmarks_list):
                                expression, intensity = self._analyze_expression(face_landmarks_list[i])
                            
                            # Calculate face center and size
                            face_center_x = (left + right) // 2
                            face_center_y = (top + bottom) // 2
                            face_size = max(right - left, bottom - top)
                            
                            # Convert to display coordinates (0-1 range)
                            center_x_norm = face_center_x / frame_width
                            center_y_norm = face_center_y / frame_height
                            size_norm = face_size / min(frame_width, frame_height)
                            
                            detected_faces.append({
                                'center': (center_x_norm, center_y_norm),
                                'size': size_norm,
                                'expression': expression,
                                'intensity': intensity
                            })
                else:
                    # Use OpenCV Haar Cascade
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.face_cascade is not None:
                        faces = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                        )
                    else:
                        faces = []
                    
                    # Process each face
                    for (x, y, w, h) in faces:
                        # Calculate face center and size
                        face_center_x = x + w // 2
                        face_center_y = y + h // 2
                        face_size = max(w, h)
                        
                        # Convert to display coordinates (0-1 range)
                        center_x_norm = face_center_x / frame_width
                        center_y_norm = face_center_y / frame_height
                        size_norm = face_size / min(frame_width, frame_height)
                        
                        # Without landmarks, assume neutral expression
                        detected_faces.append({
                            'center': (center_x_norm, center_y_norm),
                            'size': size_norm,
                            'expression': 'neutral',
                            'intensity': 0.2
                        })
                
                # Process gestures
                gesture_update = self._extract_gestures(frame_rgb)
                
                # Update detected faces
                with self.camera_lock:
                    self.current_frame = frame_rgb
                    self.detected_faces = detected_faces
                    if gesture_update:
                        self._update_gesture_state(gesture_update)
                    self.collective_state = self._compute_collective_state(detected_faces)
                
            except Exception as e:
                print(f"Error in camera loop: {e}")
                time.sleep(0.1)
    
    def _extract_gestures(self, frame_rgb):
        """Extract gesture signals from the frame using mediapipe hands"""
        if not self.hands_detector:
            return None
        try:
            image = frame_rgb.copy()
            image.flags.writeable = False
            results = self.hands_detector.process(image)
            image.flags.writeable = True
        except Exception:
            return None
        if not results or not results.multi_hand_landmarks:
            self._last_hand_distance = None
            return {
                'rotation_signal': 0.0,
                'hue_signal': 0.0,
                'distortion_signal': 0.0,
                'shatter_triggered': False,
                'timestamp': time.time()
            }
        centers = []
        spreads = []
        for hand_landmarks in results.multi_hand_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            center = coords.mean(axis=0)
            centers.append(center)
            # Spread: maximum distance from center in 2D plane
            diffs = coords[:, :2] - center[:2]
            dist = np.linalg.norm(diffs, axis=1)
            spreads.append(dist.max())
        rotation_signal = 0.0
        if centers:
            rotation_signal = max(0.0, min(1.0, 1.0 - min(c[1] for c in centers)))
        hue_signal = 0.0
        if centers:
            hue_signal = float(np.mean([c[0] for c in centers]) - 0.5)
        distortion_signal = 0.0
        if spreads:
            distortion_signal = min(1.0, max(spreads) * 4.0)
        shatter_triggered = False
        if len(centers) >= 2:
            a = centers[0]
            b = centers[1]
            distance = np.linalg.norm(a[:2] - b[:2])
            print(distance)
            if self._last_hand_distance is not None:
                # Detect clap or sudden proximity change
                if self._last_hand_distance - distance > 0.12 or distance < 0.08:
                    shatter_triggered = True
            self._last_hand_distance = distance
        else:
            self._last_hand_distance = None
        return {
            'rotation_signal': rotation_signal,
            'hue_signal': hue_signal,
            'distortion_signal': distortion_signal,
            'shatter_triggered': shatter_triggered,
            'timestamp': time.time()
        }

    def _update_gesture_state(self, gesture_update):
        """Smoothly update gesture state from detector signals"""
        alpha = 0.2
        self.gesture_state['rotation_velocity'] = self._smooth_value(
            self.gesture_state['rotation_velocity'],
            gesture_update.get('rotation_signal', 0.0),
            alpha
        )
        self.gesture_state['hue_shift'] = self._smooth_value(
            self.gesture_state['hue_shift'],
            gesture_update.get('hue_signal', 0.0),
            alpha
        )
        self.gesture_state['distortion'] = self._smooth_value(
            self.gesture_state['distortion'],
            gesture_update.get('distortion_signal', 0.0),
            alpha
        )
        if gesture_update.get('shatter_triggered'):
            self.gesture_state['shatter_active'] = True
            self.gesture_state['shatter_end'] = time.time() + 1.2
        elif self.gesture_state['shatter_active'] and time.time() > self.gesture_state['shatter_end']:
            self.gesture_state['shatter_active'] = False

    def _compute_collective_state(self, detected_faces):
        """Compute collective bloom metrics from participant positions"""
        if not detected_faces:
            return {
                'cohesion': 0.0,
                'centroid': (0.5, 0.5),
                'participant_count': 0
            }
        centers = np.array([face['center'] for face in detected_faces])
        centroid = centers.mean(axis=0)
        # Cohesion: inverse of average pairwise distance
        if len(centers) > 1:
            distances = []
            for i in range(len(centers)):
                diffs = centers[i+1:] - centers[i]
                if diffs.size:
                    pair_dists = np.linalg.norm(diffs, axis=1)
                    distances.extend(pair_dists.tolist())
            if distances:
                avg_distance = float(np.mean(distances))
                cohesion = max(0.0, min(1.0, 1.0 - avg_distance))
            else:
                cohesion = 0.0
        else:
            cohesion = 0.2
        return {
            'cohesion': cohesion,
            'centroid': (float(centroid[0]), float(centroid[1])),
            'participant_count': len(centers)
        }

    def _smooth_value(self, previous, target, alpha):
        return previous * (1 - alpha) + target * alpha
    
    def _analyze_expression(self, landmarks):
        """Analyze facial expression from landmarks and return (expression, intensity)"""
        try:
            # Calculate smile intensity from mouth shape
            top_lip = landmarks['top_lip']
            bottom_lip = landmarks['bottom_lip']
            
            # Get mouth corners and center points
            left_corner = top_lip[0]
            right_corner = top_lip[6]
            top_center = top_lip[3]
            bottom_center = bottom_lip[3]
            
            # Calculate mouth width and height
            mouth_width = np.sqrt((right_corner[0] - left_corner[0])**2 + 
                                 (right_corner[1] - left_corner[1])**2)
            mouth_height = np.sqrt((top_center[0] - bottom_center[0])**2 + 
                                  (top_center[1] - bottom_center[1])**2)
            
            # Mouth aspect ratio (higher = more open, smiling)
            mouth_ratio = mouth_width / (mouth_height + 1)
            
            # Calculate smile curvature (corners higher than center)
            corner_avg_y = (left_corner[1] + right_corner[1]) / 2
            center_avg_y = (top_center[1] + bottom_center[1]) / 2
            smile_curve = (center_avg_y - corner_avg_y) / (mouth_height + 1)
            
            # Smile intensity (0.0 to 1.0)
            smile_intensity = max(0.0, min(1.0, (mouth_ratio - 2.0) / 2.0 + smile_curve / 10))
            
            # Detect expression based on measurements
            if smile_intensity > 0.4:
                expression = 'happy'
                intensity = smile_intensity
            elif smile_curve < -0.15:
                expression = 'sad'
                intensity = min(1.0, abs(smile_curve) * 2.0)
            elif mouth_ratio > 3.5:
                expression = 'surprised'
                intensity = min(1.0, (mouth_ratio - 3.5) / 2.0)
            else:
                expression = 'neutral'
                intensity = 0.2
            
            return expression, intensity
                
        except Exception as e:
            # If expression analysis fails, return neutral
            return 'neutral', 0.2
    
    def _blend_with_memory(self, faces):
        """Blend new detections with previous state to avoid jitter"""
        if not self._emotion_memory:
            self._emotion_memory = [face.copy() for face in faces]
            return faces
        blended = []
        used_indices = set()
        for face in faces:
            blended_face = face.copy()
            best_idx = None
            best_dist = None
            current_center = np.array(face['center'])
            for idx, prev in enumerate(self._emotion_memory):
                if idx in used_indices:
                    continue
                prev_center = np.array(prev['center'])
                dist = np.linalg.norm(current_center - prev_center)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist is not None and best_dist < 0.2:
                prev = self._emotion_memory[best_idx]
                used_indices.add(best_idx)
                blended_face['intensity'] = self._smooth_value(prev['intensity'], face['intensity'], 0.4)
                blended_face['center'] = tuple(
                    self._smooth_value(prev['center'][i], face['center'][i], 0.3)
                    for i in range(2)
                )
                blended_face['size'] = self._smooth_value(prev['size'], face['size'], 0.3)
            blended.append(blended_face)
        self._emotion_memory = [face.copy() for face in blended]
        return blended

    def _get_coordinate_grid(self):
        if not self._grid_cache or self._grid_cache.get('size') != (self.width, self.height):
            xs = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)
            ys = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(xs, ys)
            radius = np.sqrt(grid_x ** 2 + grid_y ** 2)
            angle = np.arctan2(grid_y, grid_x)
            self._grid_cache = {
                'size': (self.width, self.height),
                'x': grid_x,
                'y': grid_y,
                'radius': radius,
                'angle': angle
            }
        return self._grid_cache

    def _render_emotion_kaleidoscope(self, faces, gestures, collective, timestamp):
        grid = self._get_coordinate_grid()
        angle = grid['angle']
        radius = grid['radius']
        symmetry = max(2, self.kaleidoscope_config['symmetry_order'])
        self._rotation_angle += 0.02 + gestures.get('rotation_velocity', 0.0) * 0.15
        rotated_angle = angle + self._rotation_angle
        kaleido_angle = np.mod(rotated_angle, (2 * np.pi) / symmetry) * symmetry
        distortion = 1.0 + gestures.get('distortion', 0.0) * 0.7
        time_phase = timestamp - self.start_time
        base_speed = self.kaleidoscope_config['base_speed']
        base_wave = np.sin(
            kaleido_angle * distortion +
            radius * (2.5 + gestures.get('distortion', 0.0) * 1.5) +
            time_phase * (base_speed + gestures.get('distortion', 0.0) * 0.5)
        )
        secondary_wave = np.cos(kaleido_angle * 0.5 + time_phase * 0.3)
        pattern = (base_wave + secondary_wave) * 0.5
        base_layer = (pattern + 1.0) * 0.15
        base_colors = np.stack([
            base_layer,
            np.roll(base_layer, 1, axis=1),
            np.roll(base_layer, -1, axis=0)
        ], axis=2)
        canvas = base_colors.astype(np.float32)
        for face in faces:
            center_x = face['center'][0] * 2.0 - 1.0
            center_y = face['center'][1] * 2.0 - 1.0
            local_radius = np.sqrt((grid['x'] - center_x) ** 2 + (grid['y'] - center_y) ** 2)
            spread = np.clip(face['size'] * 1.8, 0.25, 1.2)
            mask = np.exp(-(local_radius ** 2) / (0.15 * spread ** 2))
            layer = self._emotion_layer(
                face['expression'],
                face['intensity'],
                pattern,
                kaleido_angle,
                local_radius,
                time_phase
            )
            canvas += layer * mask[..., None]
        participant_count = collective.get('participant_count', 0)
        if participant_count >= 2:
            cohesion = collective.get('cohesion', 0.0)
            if cohesion > 0.01:
                centroid_x = collective['centroid'][0] * 2.0 - 1.0
                centroid_y = collective['centroid'][1] * 2.0 - 1.0
                centroid_radius = np.sqrt((grid['x'] - centroid_x) ** 2 + (grid['y'] - centroid_y) ** 2)
                bloom_mask = np.exp(-centroid_radius ** 2 / (0.45 + 0.25 * participant_count))
                bloom_strength = cohesion * self.kaleidoscope_config['collective_bloom_strength']
                bloom_color = np.array([0.9, 0.6, 1.0], dtype=np.float32)
                canvas += bloom_mask[..., None] * bloom_color * bloom_strength
        if gestures.get('shatter_active'):  # add glass-shard effect
            shard_mask = (self._rng.random((self.height, self.width)) > 0.7).astype(np.float32)
            shard_mask = shard_mask[..., None]
            shard_colors = self._rng.random((self.height, self.width, 3)).astype(np.float32)
            canvas = canvas * (1.0 - shard_mask) + shard_colors * shard_mask
        hue_shift = (gestures.get('hue_shift', 0.0) * 0.5) % 1.0
        canvas = np.clip(canvas, 0.0, 1.0)
        canvas = self._apply_hue_shift(canvas, hue_shift)
        canvas = np.clip(canvas, 0.0, 1.0)
        return Image.fromarray((canvas * 255).astype(np.uint8))

    def _emotion_layer(self, expression, intensity, pattern, kaleido_angle, local_radius, time_phase):
        expression = expression or 'neutral'
        intensity = np.clip(intensity, 0.0, 1.0)
        if expression == 'happy':
            mandala = 0.5 + 0.5 * np.sin(kaleido_angle * (2.0 + intensity) + time_phase * (1.2 + intensity))
            radial = 0.5 + 0.5 * np.cos(local_radius * (6.0 + intensity * 3.0) - time_phase * 0.8)
            mix = np.clip(0.6 * mandala + 0.4 * radial, 0.0, 1.0)
            colors = self._color_mix(mix, np.array([1.0, 0.75, 0.2]), np.array([1.0, 0.4, 0.0]))
        elif expression == 'sad':
            ripple = 0.5 + 0.5 * np.cos(local_radius * (4.0 + intensity * 2.0) + time_phase * 0.4)
            wave = 0.5 + 0.5 * np.sin(pattern * 3.0 + time_phase * 0.3)
            mix = np.clip(0.7 * ripple + 0.3 * wave, 0.0, 1.0)
            colors = self._color_mix(mix, np.array([0.2, 0.35, 0.8]), np.array([0.0, 0.1, 0.3]))
        elif expression == 'surprised':
            bursts = np.abs(np.sin(kaleido_angle * (4.0 + intensity * 4.0) + time_phase * 2.4))
            spikes = (np.mod(kaleido_angle * (2.0 + intensity * 2.0), np.pi) / np.pi)
            mix = np.clip(0.8 * bursts + 0.2 * (1.0 - spikes), 0.0, 1.0)
            colors = self._color_mix(mix, np.array([1.0, 0.1, 0.6]), np.array([0.2, 1.0, 0.5]))
        else:
            # Neutral baseline
            gentle = 0.5 + 0.5 * np.sin(local_radius * 3.0 + time_phase * 0.5)
            gradient = 0.5 + 0.5 * np.cos((pattern + kaleido_angle) * 0.8)
            mix = np.clip(0.5 * gentle + 0.5 * gradient, 0.0, 1.0)
            colors = self._color_mix(mix, np.array([0.8, 0.75, 0.9]), np.array([0.6, 0.8, 0.85]))
        strength = 0.35 + intensity * 0.65
        return colors * strength

    def _generate_idle_pattern(self, timestamp, gestures):
        grid = self._get_coordinate_grid()
        angle = grid['angle']
        radius = grid['radius']
        time_phase = timestamp - self.start_time
        wave = 0.5 + 0.5 * np.sin(angle * 1.5 + radius * 2.2 + time_phase * 0.2)
        colors = self._color_mix(wave, np.array([0.1, 0.1, 0.2]), np.array([0.2, 0.3, 0.4]))
        colors = self._apply_hue_shift(colors, (gestures.get('hue_shift', 0.0) * 0.2) % 1.0)
        colors = np.clip(colors, 0.0, 1.0)
        return Image.fromarray((colors * 255).astype(np.uint8))

    def _color_mix(self, factor, color_a, color_b):
        factor = np.clip(factor, 0.0, 1.0)[..., None]
        color_a = color_a.reshape(1, 1, 3)
        color_b = color_b.reshape(1, 1, 3)
        return color_a * (1.0 - factor) + color_b * factor

    def _apply_hue_shift(self, rgb_array, shift):
        if abs(shift) < 1e-4:
            return rgb_array
        shift = shift % 1.0
        r = rgb_array[..., 0]
        g = rgb_array[..., 1]
        b = rgb_array[..., 2]
        maxc = np.max(rgb_array, axis=2)
        minc = np.min(rgb_array, axis=2)
        delta = maxc - minc
        hue = np.zeros_like(maxc)
        mask = delta > 1e-6
        # Hue calculation
        r_mask = (maxc == r) & mask
        g_mask = (maxc == g) & mask
        b_mask = (maxc == b) & mask
        hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
        hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
        hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
        hue = (hue / 6.0 + shift) % 1.0
        saturation = np.zeros_like(maxc)
        saturation[mask] = delta[mask] / (maxc[mask] + 1e-6)
        value = maxc
        c = value * saturation
        m = value - c
        h_prime = hue * 6.0
        x = c * (1 - np.abs(h_prime % 2 - 1))
        r1 = np.zeros_like(h_prime)
        g1 = np.zeros_like(h_prime)
        b1 = np.zeros_like(h_prime)
        condlist = [
            (h_prime >= 0) & (h_prime < 1),
            (h_prime >= 1) & (h_prime < 2),
            (h_prime >= 2) & (h_prime < 3),
            (h_prime >= 3) & (h_prime < 4),
            (h_prime >= 4) & (h_prime < 5),
            (h_prime >= 5) & (h_prime < 6)
        ]
        r1[condlist[0]] = c[condlist[0]]
        g1[condlist[0]] = x[condlist[0]]
        r1[condlist[1]] = x[condlist[1]]
        g1[condlist[1]] = c[condlist[1]]
        g1[condlist[2]] = c[condlist[2]]
        b1[condlist[2]] = x[condlist[2]]
        g1[condlist[3]] = x[condlist[3]]
        b1[condlist[3]] = c[condlist[3]]
        r1[condlist[4]] = x[condlist[4]]
        b1[condlist[4]] = c[condlist[4]]
        r1[condlist[5]] = c[condlist[5]]
        b1[condlist[5]] = x[condlist[5]]
        rgb = np.stack([r1 + m, g1 + m, b1 + m], axis=2)
        rgb[~mask[..., None]] = rgb_array[~mask[..., None]]
        return rgb

    def update(self):
        """Generate and return the next kaleidoscopic frame"""
        timestamp = time.time()
        with self.camera_lock:
            detected_faces = [face.copy() for face in self.detected_faces]
            gestures = self.gesture_state.copy()
            collective = self.collective_state.copy()
        if detected_faces:
            faces = self._blend_with_memory(detected_faces)
            frame = self._render_emotion_kaleidoscope(faces, gestures, collective, timestamp)
            return frame
        # No faces – fade memory
        if self._emotion_memory:
            faded = []
            decay = self.kaleidoscope_config['emotion_decay']
            for face in self._emotion_memory:
                faded_face = face.copy()
                faded_face['intensity'] *= decay
                if faded_face['intensity'] > 0.05:
                    faded.append(faded_face)
            self._emotion_memory = faded
            if faded:
                frame = self._render_emotion_kaleidoscope(faded, gestures, collective, timestamp)
                return frame
        return self._generate_idle_pattern(timestamp, gestures)
    
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
        if self.hands_detector:
            try:
                self.hands_detector.close()
            except AttributeError:
                pass
            self.hands_detector = None
        
        print("Decompression mode cleaned up")

