"""
Website display mode.

Displays a website using Selenium/Chrome and captures screenshots
to show on the LED display.
"""

from .base_mode import BaseMode
import time

class WebsiteMode(BaseMode):
    """Display a website on the LED screen"""
    
    def __init__(self, width=40, height=30):
        super().__init__(width, height)
        self.url = None
        self.driver = None
        self.screenshot_interval = 0.04  # ~25 FPS max capture
        self.last_screenshot_time = 0
        self._display_announced = False
    
    def setup(self, **kwargs):
        """Set up with URL"""
        # No default URL - must be provided
        self.url = kwargs.get('url', None)
    
    def init(self):
        """Load the website"""
        if not self.url:
            print("Warning: No URL provided for WebsiteMode")
            return
        self.load_website(self.url)
    
    def load_website(self, url):
        """Load a website using Selenium"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            import shutil
            import os
            
            # Configure ChromeOptions
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--use-gl=egl")
            chrome_options.add_argument("--enable-webgl")
            chrome_options.add_argument("--ignore-gpu-blocklist")
            chrome_options.add_argument("--window-size=240,160")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            print(f"ChromeDriver loaded, attempting to load path: {url}")
            
            self.driver.set_window_size(240, 160)
            self.driver.get(url)
            
            print(f"Loaded website: {url}")
        except Exception as e:
            print(f"Error loading website: {e}")
            self.driver = None
    
    def get_frame(self):
        """Get current frame from the browser"""
        if not self.driver:
            return None
        
        try:
            from PIL import Image
            from io import BytesIO
            import base64
            
            image_data = self.driver.get_screenshot_as_base64()
            frame = Image.open(BytesIO(base64.b64decode(image_data)))
            return frame
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def analyze_screenshot(self, frame):
        """Analyze screenshot to provide feedback (useful for subclasses)"""
        try:
            from PIL import Image
            
            # Resize to small size for analysis
            pixels = list(frame.getdata())
            
            # Calculate brightness and coverage
            total_brightness = 0
            bright_pixels = 0
            
            for pixel in pixels:
                brightness = sum(pixel) / 3  # Average RGB
                total_brightness += brightness
                
                if brightness > 100:  # Bright pixel threshold
                    bright_pixels += 1
            
            avg_brightness = total_brightness / len(pixels)
            coverage = bright_pixels / len(pixels)
            
            return {
                'avg_brightness': avg_brightness,
                'coverage': coverage,
                'is_sparse': coverage < 0.4
            }
        except Exception as e:
            print(f"Error analyzing screenshot: {e}")
            return None
    
    def update(self):
        """Capture and return website screenshot frame"""
        if not self.driver:
            return None
        
        try:
            # FPS cap for screenshots
            now = time.time()
            if now - self.last_screenshot_time < self.screenshot_interval:
                return None
            self.last_screenshot_time = now
            
            # Get frame
            frame = self.get_frame()
            
            if frame and not self._display_announced:
                print(f"Displaying website: {self.url}")
                self._display_announced = True
            
            return frame
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def cleanup(self):
        """Close the browser"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

