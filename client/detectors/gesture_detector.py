"""
Gesture detection using Mediapipe AI and manual landmark detection.

Provides both AI-based gesture recognition and fallback manual detection
for robust gesture recognition.
"""

import numpy as np
import mediapipe as mp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


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
        
        # Manual detection (fallback)
        self.hand_detector = None
        self._init_manual_detector()
        
        # Gesture tracking
        self._gesture_frames: Dict[str, float] = {}
        self.gesture_threshold = 3  # Frames needed to confirm gesture
        self.gesture_decay_rate = 0.5  # Decay rate when gesture disappears
        
        # Wave detection
        self.hand_wave_history: list = []
        self.wave_detection_threshold = 0.15
        
        # Thumbs up detection
        self.thumbs_up_frames = 0
        self.thumbs_up_threshold = 10
        
        # Gesture flags
        self._wave_detected_flag = False
        self._thumbs_up_detected_flag = False
        
        # Gesture to status mapping
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
        }
    
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
            print(f"Mediapipe Gesture Recognizer not available: {e}")
    
    def _init_manual_detector(self) -> None:
        """Initialize manual landmark-based detector."""
        self.hand_detector = self._mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
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
                print(f"⚠ Gesture Recognizer model not found at: {model_path}")
                return False
            
            base_options = self._mp_gesture_base.BaseOptions(
                model_asset_path=model_path
            )
            
            options = self._mp_gesture_module.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=self._mp_gesture_module.RunningMode.IMAGE,
                num_hands=1
            )
            
            self.gesture_recognizer = self._mp_gesture_module.GestureRecognizer.create_from_options(options)
            print(f"✓ Mediapipe Gesture Recognizer initialized (model: {model_path})")
            return True
        except Exception as e:
            print(f"Warning: Could not initialize Gesture Recognizer: {e}")
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
                        print(f"{gesture_name} detected! (confidence: {confidence:.2f})")
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
    
    def detect_wave(self, frame: np.ndarray) -> None:
        """Detect hand wave using manual landmark detection.
        
        Args:
            frame: RGB frame as numpy array
        """
        results = self.hand_detector.process(frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            
            # Track average x position
            wrist_x = landmarks[0].x
            middle_mcp_x = landmarks[9].x
            index_mcp_x = landmarks[5].x
            avg_x = (wrist_x + middle_mcp_x + index_mcp_x) / 3.0
            
            self.hand_wave_history.append(avg_x)
            if len(self.hand_wave_history) > 20:
                self.hand_wave_history.pop(0)
            
            # Detect wave pattern
            if len(self.hand_wave_history) >= 15:
                min_x = min(self.hand_wave_history)
                max_x = max(self.hand_wave_history)
                movement_range = max_x - min_x
                
                if movement_range > self.wave_detection_threshold:
                    direction_changes = 0
                    for i in range(1, len(self.hand_wave_history) - 1):
                        prev_diff = self.hand_wave_history[i] - self.hand_wave_history[i-1]
                        next_diff = self.hand_wave_history[i+1] - self.hand_wave_history[i]
                        if (prev_diff > 0 and next_diff < 0) or (prev_diff < 0 and next_diff > 0):
                            direction_changes += 1
                    
                    if direction_changes >= 2:
                        print("Wave detected!")
                        self._wave_detected_flag = True
                        self.hand_wave_history = []
        else:
            if len(self.hand_wave_history) > 0:
                self.hand_wave_history.pop(0)
    
    def detect_thumbs_up(self, frame: np.ndarray) -> None:
        """Detect thumbs up using manual landmark detection.
        
        Args:
            frame: RGB frame as numpy array
        """
        results = self.hand_detector.process(frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            
            # Check thumb extended
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_mcp = landmarks[2]
            thumb_extended = (thumb_tip.y < thumb_ip.y) and (thumb_tip.y < thumb_mcp.y - 0.05)
            
            # Check other fingers closed
            index_closed = landmarks[8].y > landmarks[6].y + 0.02
            middle_closed = landmarks[12].y > landmarks[10].y + 0.02
            ring_closed = landmarks[16].y > landmarks[14].y + 0.02
            pinky_closed = landmarks[20].y > landmarks[18].y + 0.02
            
            closed_fingers = sum([index_closed, middle_closed, ring_closed, pinky_closed])
            
            if thumb_extended and closed_fingers >= 3:
                self.thumbs_up_frames += 1
                if self.thumbs_up_frames >= self.thumbs_up_threshold:
                    print("Thumbs up detected!")
                    self._thumbs_up_detected_flag = True
                    self.thumbs_up_frames = 0
            else:
                if self.thumbs_up_frames > 0:
                    self.thumbs_up_frames = max(0, self.thumbs_up_frames - 1)
        else:
            if self.thumbs_up_frames > 0:
                self.thumbs_up_frames = max(0, self.thumbs_up_frames - 2)
    
    def get_wave_detected(self) -> bool:
        """Get wave detection flag and reset it."""
        if self._wave_detected_flag:
            self._wave_detected_flag = False
            return True
        return False
    
    def get_thumbs_up_detected(self) -> bool:
        """Get thumbs up detection flag and reset it."""
        if self._thumbs_up_detected_flag:
            self._thumbs_up_detected_flag = False
            return True
        return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.hand_detector:
            self.hand_detector.close()
        if self.gesture_recognizer:
            self.gesture_recognizer.close()

