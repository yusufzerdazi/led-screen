"""
Gesture detection using Mediapipe AI and manual landmark detection.

Provides both AI-based gesture recognition and fallback manual detection
for robust gesture recognition.
"""

import numpy as np
import mediapipe as mp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os

# Add parent directory to path for logger import
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from logger import get_logger


@dataclass
class GestureConfig:
    """Configuration for a gesture mapping."""
    status: str
    flag: str
    requires_movement: bool = False


class GestureDetector:
    """Detects hand gestures using Mediapipe."""
    
    def __init__(self):
        """Initialize gesture detection systems."""
        # Set up logger
        self.logger = get_logger("Gesture Detection")
        
        # Mediapipe modules
        self._mp_hands_module = mp.solutions.hands
        self._mp_gesture_module = None
        self._mp_gesture_base = None
        self._mp_image_class = None
        self._mp_image_format = None
        
        # Initialize AI gesture recognizer
        self.gesture_recognizer = None
        self.gesture_recognizer_available = False
        self._init_ai_recognizer()
        
        # Manual landmark-based detection removed - using AI model only
        
        # Thread safety for gesture detection (runs in background thread)
        from threading import Lock
        self._lock = Lock()
        
        # Gesture tracking
        self._gesture_frames: Dict[str, float] = {}
        self.gesture_threshold = 3  # Frames needed to confirm gesture
        self.gesture_decay_rate = 0.5  # Decay rate when gesture disappears
        
        # Landmark-based detection removed - using AI model only
        
        # Dynamic gesture registry - maps gesture name to detection state
        # Format: {gesture_name: {'frames': 0, 'flag': False, 'detect_method': callable, 'get_method': callable}}
        self.gesture_registry = {}
        
        # Gesture flags managed by AI model detection
        
        # Register all gestures dynamically
        self._register_gestures()
        
        # Gesture to status mapping (Mediapipe AI gestures)
        self.gesture_to_status_map: Dict[str, GestureConfig] = {
            'Thumb_Up': GestureConfig(
                status='thumbs_up',
                flag='_thumbs_up_detected_flag',
                requires_movement=False
            ),
            'Open_Palm': GestureConfig(
                status='wave',
                flag='_wave_detected_flag',
                requires_movement=False
            ),
            'Victory': GestureConfig(  # V sign / Peace sign
                status='peace',
                flag='_peace_detected_flag',
                requires_movement=False
            ),
        }
    
    def _register_gestures(self):
        """Register all gestures dynamically - using AI model only"""
        # AI model handles: Thumb_Up, Open_Palm (wave), Victory (peace)
        # No manual landmark-based gestures registered
        # Initialize flags for AI-detected gestures
        self._wave_detected_flag = False
        self._thumbs_up_detected_flag = False
        self._peace_detected_flag = False
        
        # Register gesture detection methods in registry for dynamic handling
        self.gesture_registry = {
            'wave': {
                'get_method': self.get_wave_detected,
                'flag': '_wave_detected_flag'
            },
            'thumbs_up': {
                'get_method': self.get_thumbs_up_detected,
                'flag': '_thumbs_up_detected_flag'
            },
            'peace': {
                'get_method': self.get_peace_detected,
                'flag': '_peace_detected_flag'
            }
        }
    
    def get_registered_gestures(self):
        """Get list of all registered gesture names"""
        return list(self.gesture_registry.keys())
    
    def _init_ai_recognizer(self) -> None:
        """Initialize Mediapipe Gesture Recognizer AI model."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self._mp_gesture_module = vision
            self._mp_gesture_base = python
            
            # Import Image class and ImageFormat
            import mediapipe.python._framework_bindings.image as mp_image_module
            from mediapipe.python._framework_bindings import image_frame
            self._mp_image_class = mp_image_module.Image
            self._mp_image_format = image_frame.ImageFormat
            
            self.gesture_recognizer_available = True
        except (ImportError, AttributeError) as e:
            self.gesture_recognizer_available = False
            self.logger.warning(f"Mediapipe Gesture Recognizer not available: {e}")
    
    
    def initialize_ai_model(self, model_path: str) -> bool:
        """Initialize AI gesture recognizer with model file.
        
        Args:
            model_path: Path to gesture_recognizer.task model file
            
        Returns:
            True if initialized successfully
        """
        if not self.gesture_recognizer_available:
            return False
        
        try:
            import os
            if not os.path.exists(model_path):
                self.logger.warning(f"Gesture Recognizer model not found at: {model_path}")
                return False
            
            # Try to use GPU delegate if available
            # MediaPipe GPU delegate is typically available on mobile/desktop GPUs
            # For Raspberry Pi, CPU is usually the best option
            base_options = self._mp_gesture_base.BaseOptions(
                model_asset_path=model_path,
                # Uncomment to try GPU delegate (may not work on all systems):
                # delegate=self._mp_gesture_base.BaseOptions.Delegate.GPU
            )
            
            options = self._mp_gesture_module.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=self._mp_gesture_module.RunningMode.IMAGE,
                num_hands=1
            )
            
            self.gesture_recognizer = self._mp_gesture_module.GestureRecognizer.create_from_options(options)
            self.logger.info(f"Mediapipe Gesture Recognizer initialized (model: {model_path})")
            return True
        except Exception as e:
            self.logger.error(f"Could not initialize Gesture Recognizer: {e}")
            return False
    
    def detect_ai(self, frame: np.ndarray) -> None:
        """Detect gestures using AI model.
        
        Args:
            frame: RGB frame as numpy array
        """
        if not self.gesture_recognizer_available or self.gesture_recognizer is None:
            return
        
        # Create Mediapipe Image
        mp_image = self._mp_image_class(self._mp_image_format.SRGB, frame)
        
        # Recognize gestures
        recognition_result = self.gesture_recognizer.recognize(mp_image)
        
        # Thread-safe update of gesture state
        with self._lock:
            if recognition_result.gestures:
                top_gesture = recognition_result.gestures[0][0]
                gesture_name = top_gesture.category_name
                confidence = top_gesture.score
                
                if confidence > 0.7:
                    # Track consecutive frames
                    if gesture_name not in self._gesture_frames:
                        self._gesture_frames[gesture_name] = 0.0
                    
                    confidence_bonus = (confidence - 0.7) * 2.0
                    self._gesture_frames[gesture_name] += 1.0 + confidence_bonus
                    
                    # Decay other gestures
                    for other_gesture in self._gesture_frames:
                        if other_gesture != gesture_name:
                            self._gesture_frames[other_gesture] *= self.gesture_decay_rate
                    
                    # Check if threshold reached
                    if self._gesture_frames[gesture_name] >= self.gesture_threshold:
                        if gesture_name in self.gesture_to_status_map:
                            config = self.gesture_to_status_map[gesture_name]
                            setattr(self, config.flag, True)
                            self._gesture_frames[gesture_name] = 0
                            self.logger.info(f"{gesture_name} detected! (confidence: {confidence:.2f})")
                else:
                    # Low confidence - decay
                    for gesture_name in self._gesture_frames:
                        if self._gesture_frames[gesture_name] > 0:
                            self._gesture_frames[gesture_name] *= self.gesture_decay_rate
            else:
                # No gesture - decay all
                for gesture_name in self._gesture_frames:
                    if self._gesture_frames[gesture_name] > 0:
                        self._gesture_frames[gesture_name] *= self.gesture_decay_rate
    
    # Landmark-based detection methods removed - using AI model only
    # AI model handles: Thumb_Up, Open_Palm (wave), Victory (peace)
    
    
    def get_wave_detected(self) -> bool:
        """Get wave detection flag from AI model and reset it."""
        with self._lock:
            if hasattr(self, '_wave_detected_flag') and self._wave_detected_flag:
                self._wave_detected_flag = False
                return True
            return False
    
    def get_thumbs_up_detected(self) -> bool:
        """Get thumbs up detection flag from AI model and reset it."""
        with self._lock:
            if hasattr(self, '_thumbs_up_detected_flag') and self._thumbs_up_detected_flag:
                self._thumbs_up_detected_flag = False
                return True
            return False
    
    def get_peace_detected(self) -> bool:
        """Get peace detection flag from AI model and reset it."""
        with self._lock:
            if hasattr(self, '_peace_detected_flag') and self._peace_detected_flag:
                self._peace_detected_flag = False
                return True
            return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'gesture_recognizer') and self.gesture_recognizer:
            self.gesture_recognizer.close()

