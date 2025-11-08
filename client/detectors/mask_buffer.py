"""
Thread-safe double buffering for mask data.

Uses atomic buffer swapping to allow lock-free reads while ensuring
writes don't block the render thread.
"""

import numpy as np
from threading import Lock
from typing import Optional


class MaskBuffer:
    """Thread-safe double buffer for mask data with atomic swapping."""
    
    def __init__(self):
        """Initialize empty double buffer."""
        self._buffer_0: Optional[np.ndarray] = None
        self._buffer_1: Optional[np.ndarray] = None
        self._active_index: int = 0  # 0 or 1
        self._lock = Lock()
    
    def write(self, mask: np.ndarray) -> None:
        """Write new mask data to inactive buffer and atomically swap.
        
        Args:
            mask: New mask data (will be copied)
        """
        # Copy the mask outside the lock to minimize lock time
        mask_copy = mask.copy()
        
        with self._lock:
            # Write to inactive buffer
            inactive_index = 1 - self._active_index
            if inactive_index == 0:
                self._buffer_0 = mask_copy
            else:
                self._buffer_1 = mask_copy
            
            # Atomically swap active buffer
            self._active_index = inactive_index
    
    def read(self) -> Optional[np.ndarray]:
        """Read current active buffer (lock-free after getting reference).
        
        Returns:
            Copy of current active buffer, or None if no buffer available
        """
        # Get buffer reference atomically
        with self._lock:
            active_index = self._active_index
            if active_index == 0:
                buffer_ref = self._buffer_0
            else:
                buffer_ref = self._buffer_1
        
        # Copy outside the lock (lock-free copy)
        if buffer_ref is not None:
            return buffer_ref.copy()
        return None
    
    def has_data(self) -> bool:
        """Check if buffer has data available.
        
        Returns:
            True if at least one buffer has data
        """
        with self._lock:
            return (self._buffer_0 is not None) or (self._buffer_1 is not None)
    
    def clear(self) -> None:
        """Clear both buffers."""
        with self._lock:
            self._buffer_0 = None
            self._buffer_1 = None

