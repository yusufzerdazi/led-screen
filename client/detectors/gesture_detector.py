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
        
        # Manual detection (fallback)
        self.hand_detector = None
        self._init_manual_detector()
        
        # Thread safety for gesture detection (runs in background thread)
        from threading import Lock
        self._lock = Lock()
        
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
        
        # Gesture detection thresholds
        self.gesture_frame_threshold = 10
        
        # Dynamic gesture registry - maps gesture name to detection state
        # Format: {gesture_name: {'frames': 0, 'flag': False, 'detect_method': callable, 'get_method': callable}}
        self.gesture_registry = {}
        
        # Initialize gesture flags and detection state
        self._wave_detected_flag = False
        self._thumbs_up_detected_flag = False
        
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
            'Pointing_Up': GestureConfig(
                status='point',
                flag='_point_detected_flag',
                requires_movement=False
            ),
        }
    
    def _register_gestures(self):
        """Register all gestures dynamically - add new gestures here"""
        gestures = [
            {
                'name': 'peace',
                'detect_method': self.detect_peace,
                'get_method': self.get_peace_detected,
                'flag_attr': '_peace_detected_flag',
                'frames_attr': 'peace_frames',
            },
            {
                'name': 'heart',
                'detect_method': self.detect_heart,
                'get_method': self.get_heart_detected,
                'flag_attr': '_heart_detected_flag',
                'frames_attr': 'heart_frames',
            },
            {
                'name': 'rock_on',
                'detect_method': self.detect_rock_on,
                'get_method': self.get_rock_on_detected,
                'flag_attr': '_rock_on_detected_flag',
                'frames_attr': 'rock_on_frames',
            },
            {
                'name': 'point',
                'detect_method': self.detect_point,
                'get_method': self.get_point_detected,
                'flag_attr': '_point_detected_flag',
                'frames_attr': 'point_frames',
            },
            {
                'name': 'clap',
                'detect_method': self.detect_clap,
                'get_method': self.get_clap_detected,
                'flag_attr': '_clap_detected_flag',
                'frames_attr': 'clap_frames',
            },
            {
                'name': 'fist_pump',
                'detect_method': self.detect_fist_pump,
                'get_method': self.get_fist_pump_detected,
                'flag_attr': '_fist_pump_detected_flag',
                'frames_attr': 'fist_pump_frames',
            },
        ]
        
        # Initialize frames and flags for each gesture
        for gesture in gestures:
            setattr(self, gesture['frames_attr'], 0)
            setattr(self, gesture['flag_attr'], False)
            self.gesture_registry[gesture['name']] = {
                'detect_method': gesture['detect_method'],
                'get_method': gesture['get_method'],
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
    
    def detect_wave(self, frame: np.ndarray) -> None:
        """Detect hand wave using manual landmark detection.
        
        Args:
            frame: RGB frame as numpy array
        """
        # Process Mediapipe outside the lock to avoid blocking
        results = self.hand_detector.process(frame)
        
        # Only lock when updating shared state
        with self._lock:
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
                            self.logger.info("Wave detected!")
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
        # Process Mediapipe outside the lock to avoid blocking
        results = self.hand_detector.process(frame)
        
        # Only lock when updating shared state
        with self._lock:
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
                        self.logger.info("Thumbs up detected!")
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
        with self._lock:
            if self._wave_detected_flag:
                self._wave_detected_flag = False
                return True
            return False
    
    def get_thumbs_up_detected(self) -> bool:
        """Get thumbs up detection flag and reset it."""
        with self._lock:
            if self._thumbs_up_detected_flag:
                self._thumbs_up_detected_flag = False
                return True
            return False
    
    def detect_peace(self, frame: np.ndarray) -> None:
        """Detect peace sign (V sign) using manual landmark detection."""
        results = self.hand_detector.process(frame)
        with self._lock:
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark
                
                # Peace sign: index and middle finger extended, others closed
                index_extended = landmarks[8].y < landmarks[6].y - 0.02
                middle_extended = landmarks[12].y < landmarks[10].y - 0.02
                ring_closed = landmarks[16].y > landmarks[14].y + 0.02
                pinky_closed = landmarks[20].y > landmarks[18].y + 0.02
                thumb_closed = landmarks[4].y > landmarks[3].y + 0.02
                
                if index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed:
                    self.peace_frames += 1
                    if self.peace_frames >= self.gesture_frame_threshold:
                        self.logger.info("Peace sign detected!")
                        self._peace_detected_flag = True
                        self.peace_frames = 0
                else:
                    self.peace_frames = max(0, self.peace_frames - 1)
            else:
                self.peace_frames = max(0, self.peace_frames - 2)
    
    def detect_heart(self, frame: np.ndarray) -> None:
        """Detect heart hands gesture (two hands forming heart shape)."""
        results = self.hand_detector.process(frame)
        with self._lock:
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
                # Heart: two hands, thumbs and index fingers touching at tips
                hand1 = results.multi_hand_landmarks[0]
                hand2 = results.multi_hand_landmarks[1]
                
                # Check if thumbs and index fingers are close together
                thumb1_tip = hand1.landmark[4]
                thumb2_tip = hand2.landmark[4]
                index1_tip = hand1.landmark[8]
                index2_tip = hand2.landmark[8]
                
                thumb_distance = ((thumb1_tip.x - thumb2_tip.x)**2 + (thumb1_tip.y - thumb2_tip.y)**2)**0.5
                index_distance = ((index1_tip.x - index2_tip.x)**2 + (index1_tip.y - index2_tip.y)**2)**0.5
                
                if thumb_distance < 0.1 and index_distance < 0.1:
                    self.heart_frames += 1
                    if self.heart_frames >= self.gesture_frame_threshold:
                        self.logger.info("Heart hands detected!")
                        self._heart_detected_flag = True
                        self.heart_frames = 0
                else:
                    self.heart_frames = max(0, self.heart_frames - 1)
            else:
                self.heart_frames = max(0, self.heart_frames - 2)
    
    def detect_rock_on(self, frame: np.ndarray) -> None:
        """Detect rock on / devil horns gesture (index and pinky extended)."""
        results = self.hand_detector.process(frame)
        with self._lock:
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark
                
                # Rock on: index and pinky extended, middle and ring closed
                index_extended = landmarks[8].y < landmarks[6].y - 0.02
                middle_closed = landmarks[12].y > landmarks[10].y + 0.02
                ring_closed = landmarks[16].y > landmarks[14].y + 0.02
                pinky_extended = landmarks[20].y < landmarks[18].y - 0.02
                
                if index_extended and middle_closed and ring_closed and pinky_extended:
                    self.rock_on_frames += 1
                    if self.rock_on_frames >= self.gesture_frame_threshold:
                        self.logger.info("Rock on detected!")
                        self._rock_on_detected_flag = True
                        self.rock_on_frames = 0
                else:
                    self.rock_on_frames = max(0, self.rock_on_frames - 1)
            else:
                self.rock_on_frames = max(0, self.rock_on_frames - 2)
    
    def detect_point(self, frame: np.ndarray) -> None:
        """Detect pointing gesture (index finger extended, others closed)."""
        results = self.hand_detector.process(frame)
        with self._lock:
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark
                
                # Point: index extended, others closed
                index_extended = landmarks[8].y < landmarks[6].y - 0.02
                middle_closed = landmarks[12].y > landmarks[10].y + 0.02
                ring_closed = landmarks[16].y > landmarks[14].y + 0.02
                pinky_closed = landmarks[20].y > landmarks[18].y + 0.02
                
                if index_extended and middle_closed and ring_closed and pinky_closed:
                    self.point_frames += 1
                    if self.point_frames >= self.gesture_frame_threshold:
                        self.logger.info("Pointing detected!")
                        self._point_detected_flag = True
                        self.point_frames = 0
                else:
                    self.point_frames = max(0, self.point_frames - 1)
            else:
                self.point_frames = max(0, self.point_frames - 2)
    
    def detect_clap(self, frame: np.ndarray) -> None:
        """Detect clapping (two hands coming together)."""
        results = self.hand_detector.process(frame)
        with self._lock:
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
                hand1 = results.multi_hand_landmarks[0]
                hand2 = results.multi_hand_landmarks[1]
                
                # Check if palms are close together
                palm1 = hand1.landmark[0]  # Wrist
                palm2 = hand2.landmark[0]
                distance = ((palm1.x - palm2.x)**2 + (palm1.y - palm2.y)**2)**0.5
                
                if distance < 0.15:
                    self.clap_frames += 1
                    if self.clap_frames >= self.gesture_frame_threshold:
                        self.logger.info("Clapping detected!")
                        self._clap_detected_flag = True
                        self.clap_frames = 0
                else:
                    self.clap_frames = max(0, self.clap_frames - 1)
            else:
                self.clap_frames = max(0, self.clap_frames - 2)
    
    def detect_fist_pump(self, frame: np.ndarray) -> None:
        """Detect fist pump (closed fist moving up)."""
        results = self.hand_detector.process(frame)
        with self._lock:
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark
                
                # Fist: all fingers closed
                index_closed = landmarks[8].y > landmarks[6].y + 0.02
                middle_closed = landmarks[12].y > landmarks[10].y + 0.02
                ring_closed = landmarks[16].y > landmarks[14].y + 0.02
                pinky_closed = landmarks[20].y > landmarks[18].y + 0.02
                thumb_closed = landmarks[4].y > landmarks[3].y + 0.02
                
                if all([index_closed, middle_closed, ring_closed, pinky_closed, thumb_closed]):
                    self.fist_pump_frames += 1
                    if self.fist_pump_frames >= self.gesture_frame_threshold:
                        self.logger.info("Fist pump detected!")
                        self._fist_pump_detected_flag = True
                        self.fist_pump_frames = 0
                else:
                    self.fist_pump_frames = max(0, self.fist_pump_frames - 1)
            else:
                self.fist_pump_frames = max(0, self.fist_pump_frames - 2)
    
    # Getter methods for gestures (dynamically registered)
    def get_peace_detected(self) -> bool:
        with self._lock:
            if self._peace_detected_flag:
                self._peace_detected_flag = False
                return True
            return False
    
    def get_heart_detected(self) -> bool:
        with self._lock:
            if self._heart_detected_flag:
                self._heart_detected_flag = False
                return True
            return False
    
    def get_rock_on_detected(self) -> bool:
        with self._lock:
            if self._rock_on_detected_flag:
                self._rock_on_detected_flag = False
                return True
            return False
    
    def get_point_detected(self) -> bool:
        with self._lock:
            if self._point_detected_flag:
                self._point_detected_flag = False
                return True
            return False
    
    def get_clap_detected(self) -> bool:
        with self._lock:
            if self._clap_detected_flag:
                self._clap_detected_flag = False
                return True
            return False
    
    def get_fist_pump_detected(self) -> bool:
        with self._lock:
            if self._fist_pump_detected_flag:
                self._fist_pump_detected_flag = False
                return True
            return False
    
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.hand_detector:
            self.hand_detector.close()
        if self.gesture_recognizer:
            self.gesture_recognizer.close()

