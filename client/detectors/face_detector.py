"""
Face detection using Mediapipe.

Tracks faces and provides position data for pupil tracking.
Also detects smiles using Mediapipe Face Landmarker with blendshapes.
"""

import mediapipe as mp
import numpy as np
from threading import Lock
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FacePosition:
    """Normalized face position [0-1, 0-1] relative to camera frame."""
    x: float
    y: float


class FaceDetector:
    """Detects and tracks faces using Mediapipe."""
    
    def __init__(self):
        """Initialize face detector."""
        self._mp_face_module = mp.solutions.face_detection
        self.face_detector = self._mp_face_module.FaceDetection(
            model_selection=0,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Face Landmarker for smile detection (uses blendshapes)
        self._mp_face_landmarker_module = None
        self._mp_image_class = None
        self._mp_image_format = None
        self.face_landmarker = None
        self.face_landmarker_available = False
        self._init_face_landmarker()
        
        self._lock = Lock()
        self._target_position: Optional[FacePosition] = None
        self._detected_faces: List[FacePosition] = []
        self._current_face_index = 0
        self._last_face_switch_time = 0.0
        self.face_switch_interval = 1.5  # Seconds between face switches
        
        # Smile detection
        self._smile_detected = False
        self._smile_frames = 0
        self.smile_threshold = 8  # Frames needed to confirm smile
        self.smile_decay_rate = 0.7  # Decay rate when smile disappears
        self.smile_blendshape_threshold = 0.3  # Minimum blendshape value for smile
    
    def _init_face_landmarker(self) -> None:
        """Initialize Mediapipe Face Landmarker for smile detection."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self._mp_face_landmarker_module = vision
            
            # Import Image class and ImageFormat
            import mediapipe.python._framework_bindings.image as mp_image_module
            from mediapipe.python._framework_bindings import image_frame
            self._mp_image_class = mp_image_module.Image
            self._mp_image_format = image_frame.ImageFormat
            
            self.face_landmarker_available = True
        except (ImportError, AttributeError) as e:
            self.face_landmarker_available = False
            print(f"Mediapipe Face Landmarker not available: {e}")
    
    def initialize_face_landmarker(self, model_path: str) -> bool:
        """Initialize Face Landmarker with model file.
        
        Args:
            model_path: Path to face_landmarker.task model file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.face_landmarker_available:
            return False
        
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as base_options_module
            
            base_options = base_options_module.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,  # Enable blendshapes for expression detection
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            print(f"Face Landmarker initialized with model: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to initialize Face Landmarker: {e}")
            import traceback
            traceback.print_exc()
            self.face_landmarker = None
            return False
    
    def detect(self, frame: np.ndarray) -> None:
        """Detect faces in frame and update target position.
        
        Args:
            frame: RGB frame as numpy array
        """
        import time
        
        results = self.face_detector.process(frame)
        
        with self._lock:
            self._detected_faces = []
            
            if results.detections:
                for detection in results.detections:
                    # Get bounding box center
                    bbox = detection.location_data.relative_bounding_box
                    center_x = bbox.xmin + bbox.width / 2.0
                    center_y = bbox.ymin + bbox.height / 2.0
                    
                    self._detected_faces.append(FacePosition(x=center_x, y=center_y))
                
                # Switch between multiple faces periodically
                current_time = time.time()
                if len(self._detected_faces) > 1:
                    if current_time - self._last_face_switch_time >= self.face_switch_interval:
                        self._current_face_index = (self._current_face_index + 1) % len(self._detected_faces)
                        self._last_face_switch_time = current_time
                
                # Update target position
                if self._detected_faces:
                    target_face = self._detected_faces[self._current_face_index]
                    self._target_position = FacePosition(x=target_face.x, y=target_face.y)
            else:
                self._target_position = None
    
    def detect_smile(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """Detect smile using Face Landmarker blendshapes.
        
        Args:
            frame: RGB frame as numpy array
            timestamp_ms: Timestamp in milliseconds for video mode
        """
        if not self.face_landmarker_available or self.face_landmarker is None:
            return
        
        # Create Mediapipe Image
        mp_image = self._mp_image_class(self._mp_image_format.SRGB, frame)
        
        # Detect face landmarks and blendshapes
        detection_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        with self._lock:
            smile_detected_this_frame = False
            
            if detection_result.face_landmarks and detection_result.face_blendshapes:
                # Check blendshapes for smile indicators
                # Mediapipe blendshapes include: mouthSmileLeft, mouthSmileRight
                for blendshape in detection_result.face_blendshapes[0]:
                    if blendshape.category_name in ['mouthSmileLeft', 'mouthSmileRight']:
                        if blendshape.score >= self.smile_blendshape_threshold:
                            smile_detected_this_frame = True
                            break
                
                # Also check for overall smile expression
                for blendshape in detection_result.face_blendshapes[0]:
                    if blendshape.category_name == 'mouthSmile' and blendshape.score >= self.smile_blendshape_threshold:
                        smile_detected_this_frame = True
                        break
            
            # Update smile detection state
            if smile_detected_this_frame:
                self._smile_frames += 1
                if self._smile_frames >= self.smile_threshold:
                    self._smile_detected = True
                    self._smile_frames = 0  # Reset counter
            else:
                # Decay smile detection
                self._smile_frames = max(0, self._smile_frames - 1)
                if self._smile_frames == 0:
                    self._smile_detected = False
    
    def get_target_position(self) -> Optional[FacePosition]:
        """Get current target face position.
        
        Returns:
            FacePosition or None if no face detected
        """
        with self._lock:
            return self._target_position
    
    def has_faces(self) -> bool:
        """Check if any faces are detected.
        
        Returns:
            True if faces detected
        """
        with self._lock:
            return len(self._detected_faces) > 0
    
    def get_face_count(self) -> int:
        """Get number of detected faces.
        
        Returns:
            Number of faces
        """
        with self._lock:
            return len(self._detected_faces)
    
    def get_smile_detected(self) -> bool:
        """Check if smile is currently detected.
        
        Returns:
            True if smile detected
        """
        with self._lock:
            return self._smile_detected
    
    def reset_smile_detected(self) -> None:
        """Reset smile detection flag (call after handling smile)."""
        with self._lock:
            self._smile_detected = False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.face_detector:
            self.face_detector.close()
        if self.face_landmarker:
            self.face_landmarker.close()

