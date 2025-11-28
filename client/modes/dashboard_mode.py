"""
Dashboard display mode.

Displays information from various sources (sensors, APIs, etc.)
"""

from .base_mode import BaseMode
import requests
from PIL import Image, ImageDraw, ImageFont

class DashboardMode(BaseMode):
    """Display dashboard information"""
    
    def __init__(self, width=256, height=144):
        super().__init__(width, height)
        self.last_update = 0
        self.update_interval = 5.0  # Update every 5 seconds
    
    def update(self):
        """Generate and return dashboard frame"""
        import time
        
        now = time.time()
        if now - self.last_update < self.update_interval:
            return None  # No update needed yet
        self.last_update = now
        
        try:
            # Example: Get sensor data
            # This is a placeholder - customize based on your needs
            sensor_response = requests.get(
                "http://192.168.0.46/api/45F3isezBAfXK82b401E9MfiyFgAMCIs7nIGtoUV/sensors/12",
                timeout=2
            ).json()
            
            temp = sensor_response['state']['temperature'] / 100
            
            # Create dashboard image
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw temperature
            text = f"{temp:.1f}°C"
            # Simple text rendering (you may want to use a proper font)
            draw.text((2, 2), text, fill=(255, 255, 255))
            
            print(f"Dashboard updated: {temp:.1f}°C")
            return img
        except Exception as e:
            print(f"Error updating dashboard: {e}")
            return None

