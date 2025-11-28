"""
People mask detection using Mediapipe selfie segmentation.

Detects people outlines and provides mask data via double buffering.
"""

import mediapipe as mp
import cv2
import numpy as np
import os
import sys
from .mask_buffer import MaskBuffer
from typing import Tuple

# Add parent directory for logger
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from logger import get_logger


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
        self.logger = get_logger("People Segmentation")
    
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
            
            # Filter to only keep the 3 largest segments
            mask_filtered = self._filter_largest_segments(mask_resized, num_segments=3)
            
            # Write to buffer (handles double buffering internally)
            self.mask_buffer.write(mask_filtered)
    
    def _filter_largest_segments(self, mask: np.ndarray, num_segments: int = 3) -> np.ndarray:
        """Filter mask to only keep the N largest connected segments.
        
        Args:
            mask: Binary mask (0-1 float or 0-255 uint8)
            num_segments: Number of largest segments to keep (default: 3)
            
        Returns:
            Filtered mask with only the largest segments
        """
        # Convert mask to binary (0 or 255) if needed
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            # Threshold float mask (typically 0.0-1.0)
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            # Already uint8, threshold at 128
            binary_mask = (mask > 128).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # If no segments found or only background, return original mask
        if num_labels <= 1:
            return mask
        
        # Calculate area for each component (excluding background label 0)
        component_areas = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            # Only keep components larger than 20 pixels
            if area >= 20:
                component_areas.append((i, area))
        
        # If no segments meet the size requirement, return empty mask
        if not component_areas:
            filtered_mask = np.zeros_like(binary_mask)
            if mask.dtype == np.float32 or mask.dtype == np.float64:
                filtered_mask = filtered_mask.astype(mask.dtype)
            return filtered_mask
        
        # Sort by area (largest first)
        component_areas.sort(key=lambda x: x[1], reverse=True)
        
        # Log segment information (only occasionally to avoid spam)
        if len(component_areas) > num_segments:
            total_segments = len(component_areas)
            kept_areas = [area for _, area in component_areas[:num_segments]]
            self.logger.debug(f"Found {total_segments} segments (>=20px), keeping {num_segments} largest (areas: {kept_areas})")
        
        # Keep only the N largest FOREGROUND segments (exclude background label 0)
        keep_labels = set()
        for i, _ in component_areas[:num_segments]:
            keep_labels.add(i)
        
        # Create filtered mask (background remains 0)
        filtered_mask = np.zeros_like(binary_mask)
        for label in keep_labels:
            filtered_mask[labels == label] = 255
        
        # Convert back to original format
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            filtered_mask = filtered_mask.astype(mask.dtype) / 255.0
        
        return filtered_mask
    
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

