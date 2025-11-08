"""
People mask detection using Mediapipe selfie segmentation.

Detects people outlines and provides mask data via double buffering.
"""

import mediapipe as mp
import cv2
import numpy as np
from .mask_buffer import MaskBuffer
from typing import Tuple


class MaskDetector:
    """Detects people masks using selfie segmentation."""
    
    def __init__(self, width: int, height: int):
        """Initialize mask detector.
        
        Args:
            width: Display width
            height: Display height
        """
        self.width = width
        self.height = height
        self._mp_selfie_module = mp.solutions.selfie_segmentation
        self.selfie_segmenter = self._mp_selfie_module.SelfieSegmentation(
            model_selection=1  # 0 for general, 1 for landscape
        )
        self.mask_buffer = MaskBuffer()
    
    def detect(self, frame: np.ndarray) -> None:
        """Detect people mask in frame and update buffer.
        
        Args:
            frame: RGB frame as numpy array
        """
        results = self.selfie_segmenter.process(frame)
        
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask
            
            # Resize mask to match display dimensions
            mask_resized = cv2.resize(mask, (self.width, self.height))
            
            # Write to buffer (handles double buffering internally)
            self.mask_buffer.write(mask_resized)
    
    def get_mask(self) -> np.ndarray:
        """Get current mask from buffer.
        
        Returns:
            Mask array or None if not available
        """
        return self.mask_buffer.read()
    
    def has_mask(self) -> bool:
        """Check if mask data is available.
        
        Returns:
            True if mask available
        """
        return self.mask_buffer.has_data()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.selfie_segmenter:
            self.selfie_segmenter.close()
        self.mask_buffer.clear()

