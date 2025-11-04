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
        
        # Frame caching to prevent blocking LED updates
        self._cached_frame = None
        self._frame_lock = None
    
    def setup(self, **kwargs):
        """Set up with URL"""
        # No default URL - must be provided
        self.url = kwargs.get('url', None)
    
    def init(self):
        """Load the website"""
        if not self.url:
            print("Warning: No URL provided for WebsiteMode")
            return
        
        # Initialize frame lock
        from threading import Lock, Thread
        self._frame_lock = Lock()
        
        self.load_website(self.url)
        
        # Start background screenshot thread
        self._screenshot_active = True
        self._screenshot_thread = Thread(target=self._screenshot_loop)
        self._screenshot_thread.daemon = True
        self._screenshot_thread.start()
    
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
            
            # Use webdriver-manager to automatically handle chromedriver
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                print("ChromeDriver loaded via webdriver-manager")
            except ImportError:
                # Fallback if webdriver-manager is not installed
                chromedriver_path = shutil.which('chromedriver')
                if not chromedriver_path:
                    for path in ['/usr/bin/chromedriver', '/usr/local/bin/chromedriver']:
                        if os.path.exists(path):
                            chromedriver_path = path
                            break
                
                if chromedriver_path:
                    service = Service(chromedriver_path)
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                else:
                    self.driver = webdriver.Chrome(options=chrome_options)
            print(f"ChromeDriver loaded, attempting to load path: {url}")
            
            self.driver.set_window_size(240, 160)
            self.driver.get(url)
            
            print(f"Loaded website: {url}")
        except Exception as e:
            print(f"Error loading website: {e}")
            self.driver = None
    
    def _screenshot_loop(self):
        """Background thread for capturing screenshots"""
        import time
        from PIL import Image
        from io import BytesIO
        import base64
        
        while self._screenshot_active and self.driver:
            try:
                # Capture screenshot (blocking operation)
                image_data = self.driver.get_screenshot_as_base64()
                frame = Image.open(BytesIO(base64.b64decode(image_data)))
                
                # Update cached frame
                with self._frame_lock:
                    self._cached_frame = frame
                    
            except Exception as e:
                print(f"Error capturing screenshot: {e}")
            
            # Rate limit screenshot capture
            time.sleep(self.screenshot_interval)
    
    def get_frame(self):
        """Get current frame from cache (non-blocking)"""
        if not self._frame_lock:
            return None
        
        with self._frame_lock:
            return self._cached_frame
    
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
        # Stop screenshot thread
        self._screenshot_active = False
        if hasattr(self, '_screenshot_thread'):
            self._screenshot_thread.join(timeout=1.0)
        
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

