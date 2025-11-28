"""
Light test mode - displays a 3x3 square of pixels at 100% brightness in the bottom left corner.

This mode is useful for testing LED functionality and verifying pixel positions.
"""

from .base_mode import BaseMode
from PIL import Image
import numpy as np


class LightTestMode(BaseMode):
    """Light test mode that displays a 3x3 square in the bottom left corner at 100% brightness"""
    
    def __init__(self, width=40, height=30):
        super().__init__(width, height)
    
    def init(self):
        """Initialize the mode"""
        print("Light test mode initialized - showing 3x3 square in bottom left at 100% brightness")
    
    def update(self):
        """
        Generate and return the next frame with a 3x3 square in the bottom left.
        
        Returns:
            PIL.Image: The frame with white 3x3 square in bottom left corner
        """
        # Create black image
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        pixels = np.array(img)
        
        # Draw 3x3 square in bottom left corner
        # Bottom left in PIL coordinates: x=0-2, y=height-3 to height-1
        # (This will appear at bottom left on the LED display)
        for x in range(3):
            for y in range(self.height - 3, self.height):
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Set to white (255, 255, 255) for 100% brightness
                    pixels[y, x] = (255, 255, 255)
        
        return Image.fromarray(pixels)
    
    def cleanup(self):
        """Clean up the mode"""
        print("Light test mode cleaned up")

