"""
Hydra Mask Generation Mode.

Refreshes Hydra page periodically and automatically saves sketch parameters.
Analyzes frames for dark pixels and saves sketches that are not too dark (>20% dark pixels).
"""

from .website_mode import WebsiteMode
from PIL import Image
import numpy as np
import time
import urllib.parse
import threading
import sys
import os


class HydraMaskMode(WebsiteMode):
    """Mode for refreshing Hydra and saving sketch parameters"""
    
    def __init__(self, width=40, height=30):
        super().__init__(width, height)
        self.hydra_url = "http://localhost:5173"
        
        # Increase screenshot rate for better video quality
        self.screenshot_interval = 0.016  # 60 FPS (overrides parent's 25 FPS)
        
        # State management
        self.refresh_active = True
        
        # Sketch saving - save to sketches_sparse.txt in same folder as this file
        script_dir = os.path.dirname(os.path.abspath(__file__))  # client/modes/
        self.sketches_file = os.path.join(script_dir, "sketches_sparse.txt")
        
        # Frame analysis settings
        self.analysis_duration = 3.0  # Analyze frames for 3 seconds
        self.analysis_interval = 0.1  # Sample every 100ms
        self.dark_threshold = 50  # Percentage threshold for "dark" pixels
        self.dark_pixel_brightness = 20  # Pixels below this brightness are considered "dark"
        
        print(f"Sketches will be saved to: {os.path.abspath(self.sketches_file)}")
        print(f"Auto-judging sketches: saving if <{self.dark_threshold}% dark pixels")
        
    def setup(self, **kwargs):
        """Set up Hydra URL"""
        self.url = kwargs.get('url', self.hydra_url)
    
    def init(self):
        """Initialize Hydra and start refresh loop"""
        super().init()
        
        if self.driver:
            try:
                # Hide Hydra UI elements
                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                print("Hydra visualizer initialized")
            except Exception as e:
                print(f"Could not configure Hydra: {e}")
        
        # Start refresh and analysis thread
        self.refresh_active = True
        self.refresh_thread = threading.Thread(target=self._refresh_and_analyze_loop, daemon=True)
        self.refresh_thread.start()
        
        print("Hydra Mask Mode initialized. Auto-analyzing and saving sketches periodically.")
    
    def _refresh_hydra(self):
        """Reload the base Hydra URL to get a new sketch"""
        if not self.driver:
            return False
        
        try:
            # Navigate to base URL (without sketch_id) to get a new sketch
            self.driver.get(self.hydra_url)
            time.sleep(2)  # Wait for Hydra to load
            
            # Hide UI elements
            self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
            self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
            self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
            
            return True
        except Exception as e:
            print(f"Error reloading Hydra: {e}")
            return False
    
    def _get_current_sketch(self, wait_for_sketch=True, max_wait=10.0):
        """Get the sketch parameter from the current URL
        
        Args:
            wait_for_sketch: If True, poll until sketch appears in URL (up to max_wait seconds)
            max_wait: Maximum time to wait for sketch to appear in URL
        """
        if not self.driver:
            return None
        
        start_time = time.time()
        
        while True:
            try:
                current_url = self.driver.current_url
                parsed_url = urllib.parse.urlparse(current_url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                
                # Get sketch_id parameter
                if 'sketch_id' in query_params:
                    sketch_value = query_params['sketch_id'][0]
                    return sketch_value
                
                # If sketch not found and we're not waiting, return None
                if not wait_for_sketch:
                    return None
                
                # If we've waited too long, return None
                if time.time() - start_time > max_wait:
                    return None
                
                # Wait a bit before checking again
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error getting current sketch: {e}")
                return None
    
    def _sketch_already_saved(self, sketch_value):
        """Check if sketch is already in sketches_sparse.txt"""
        if not os.path.exists(self.sketches_file):
            return False
        
        try:
            with open(self.sketches_file, 'r', encoding='utf-8') as f:
                saved_sketches = [line.strip() for line in f.readlines() if line.strip()]
            return sketch_value in saved_sketches
        except Exception as e:
            print(f"Error checking saved sketches: {e}")
            return False
    
    def _save_sketch(self, sketch_value):
        """Save sketch parameter to sketches_sparse.txt"""
        try:
            # Append to sketches_sparse.txt (one per line)
            with open(self.sketches_file, 'a', encoding='utf-8') as f:
                f.write(f"{sketch_value}\n")
            
            print(f"✓ Sketch saved to {os.path.abspath(self.sketches_file)}")
            return True
        except Exception as e:
            print(f"Error saving sketch: {e}")
            return False
    
    def _analyze_frame_darkness(self, frame):
        """Analyze a frame and return percentage of dark pixels
        
        Args:
            frame: PIL Image
            
        Returns:
            float: Percentage of dark pixels (0-100)
        """
        if frame is None:
            return 100.0  # Consider None frames as 100% dark
        
        try:
            # Convert to numpy array
            img_array = np.array(frame)
            
            # Handle different image modes
            if len(img_array.shape) == 3:
                # RGB/RGBA - calculate brightness (luminance)
                if img_array.shape[2] == 4:
                    # RGBA - ignore alpha channel
                    rgb = img_array[:, :, :3]
                else:
                    rgb = img_array
                
                # Calculate luminance: 0.299*R + 0.587*G + 0.114*B
                brightness = (0.299 * rgb[:, :, 0] + 
                              0.587 * rgb[:, :, 1] + 
                              0.114 * rgb[:, :, 2])
            else:
                # Grayscale
                brightness = img_array
            
            # Count dark pixels (below threshold)
            total_pixels = brightness.size
            dark_pixels = np.sum(brightness < self.dark_pixel_brightness)
            
            dark_percentage = (dark_pixels / total_pixels) * 100.0
            return dark_percentage
            
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return 100.0  # Consider errors as dark
    
    def _analyze_sketch(self):
        """Analyze frames over time and return average dark pixel percentage"""
        print(f"[ANALYSIS] Starting frame analysis for {self.analysis_duration}s...")
        
        dark_percentages = []
        num_samples = int(self.analysis_duration / self.analysis_interval)
        start_time = time.time()
        
        for i in range(num_samples):
            # Get current frame
            try:
                frame = self.get_frame()
                dark_pct = self._analyze_frame_darkness(frame)
                dark_percentages.append(dark_pct)
                
                if (i + 1) % 10 == 0:  # Log every 10 samples
                    print(f"[ANALYSIS] Sample {i+1}/{num_samples}: {dark_pct:.1f}% dark pixels")
            except Exception as e:
                print(f"[ANALYSIS] Error getting frame: {e}")
                dark_percentages.append(100.0)  # Consider errors as dark
            
            # Wait for next sample
            elapsed = time.time() - start_time
            sleep_time = self.analysis_interval - (elapsed % self.analysis_interval)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        avg_dark_pct = np.mean(dark_percentages) if dark_percentages else 100.0
        print(f"[ANALYSIS] Analysis complete: average {avg_dark_pct:.1f}% dark pixels")
        
        return avg_dark_pct
    
    def _refresh_and_analyze_loop(self):
        """Main loop for refreshing page and automatically analyzing/saving sketches"""
        refresh_interval = 10.0  # Refresh every 10 seconds
        
        while self.refresh_active:
            try:
                print("\n" + "="*60)
                print("[REFRESH] Loading new sketch...")
                
                # Refresh the page
                if not self._refresh_hydra():
                    print("[ERROR] Failed to refresh Hydra, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Wait for page to load and JavaScript to append sketch_id to URL
                print("[REFRESH] Waiting for sketch_id to appear in URL...")
                sketch_value = self._get_current_sketch(wait_for_sketch=True, max_wait=10.0)
                
                if not sketch_value:
                    print("[WARNING] No sketch_id found in URL after waiting")
                    time.sleep(refresh_interval)
                    continue
                
                print(f"[SKETCH] Found sketch_id: {sketch_value[:50]}...")
                
                # Check if already saved
                if self._sketch_already_saved(sketch_value):
                    print(f"[SKIP] Sketch already saved, skipping analysis")
                    time.sleep(refresh_interval)
                    continue
                
                # Wait a moment for frames to start being captured
                print("[ANALYSIS] Waiting for frames to be captured...")
                time.sleep(1.0)
                
                # Analyze frames for darkness
                avg_dark_pct = self._analyze_sketch()
                
                # Decide whether to save
                if avg_dark_pct > self.dark_threshold:
                    print(f"[SAVE] Sketch is bright enough ({avg_dark_pct:.1f}% < {self.dark_threshold}% dark), saving...")
                    if self._save_sketch(sketch_value):
                        print(f"[SUCCESS] Sketch saved successfully!")
                    else:
                        print(f"[ERROR] Failed to save sketch")
                else:
                    print(f"[SKIP] Sketch too dark ({avg_dark_pct:.1f}% >= {self.dark_threshold}% dark), skipping")
                
                # Wait before next refresh
                print(f"[WAIT] Waiting {refresh_interval}s before next refresh...")
                time.sleep(refresh_interval)
                
            except Exception as e:
                print(f"[ERROR] Error in refresh loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    def update(self):
        """Update and return current frame"""
        # Get frame from parent class
        frame = super().update()
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up Hydra Mask Mode...")
        
        # Stop refresh loop
        self.refresh_active = False
        
        # Wait for threads
        if hasattr(self, 'refresh_thread'):
            self.refresh_thread.join(timeout=2.0)
        
        # Clean up parent
        super().cleanup()
        
        print("Hydra Mask Mode cleaned up")

