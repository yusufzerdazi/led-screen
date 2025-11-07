"""
Hydra Mask Generation Mode.

Refreshes Hydra page periodically and allows saving sketch parameters.
When user types Y, saves the sketch parameter from the URL to sketches.txt.
"""

from .website_mode import WebsiteMode
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
        self.waiting_for_approval = False
        self.input_thread = None  # Thread for user input
        self.approval_lock = threading.Lock()
        self.user_response = None
        self.refresh_active = True
        
        # Sketch saving - save to sketches.txt in same folder as this file
        script_dir = os.path.dirname(os.path.abspath(__file__))  # client/modes/
        self.sketches_file = os.path.join(script_dir, "sketches.txt")
        
        print(f"Sketches will be saved to: {os.path.abspath(self.sketches_file)}")
        
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
        
        # Start refresh and input monitoring thread
        self.refresh_active = True
        self.refresh_thread = threading.Thread(target=self._refresh_and_approve_loop, daemon=True)
        self.refresh_thread.start()
        
        print("Hydra Mask Mode initialized. Refreshing page periodically.")
        print("Type 'Y' to save the current sketch parameter to sketches.txt")
    
    def _refresh_hydra(self):
        """Refresh the Hydra page"""
        if not self.driver:
            return False
        
        try:
            # Just refresh the current page
            self.driver.refresh()
            time.sleep(2)  # Wait for Hydra to reload
            
            # Hide UI elements
            self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
            self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
            self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
            
            return True
        except Exception as e:
            print(f"Error refreshing Hydra: {e}")
            return False
    
    def _get_current_sketch(self):
        """Get the sketch parameter from the current URL"""
        if not self.driver:
            return None
        
        try:
            current_url = self.driver.current_url
            parsed_url = urllib.parse.urlparse(current_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Get sketch parameter
            if 'sketch' in query_params:
                sketch_value = query_params['sketch'][0]
                return sketch_value
            return None
        except Exception as e:
            print(f"Error getting current sketch: {e}")
            return None
    
    def _save_sketch(self, sketch_value):
        """Save sketch parameter to sketches.txt"""
        try:
            # Append to sketches.txt (one per line)
            with open(self.sketches_file, 'a', encoding='utf-8') as f:
                f.write(f"{sketch_value}\n")
            
            print(f"\n✓ Sketch saved to {os.path.abspath(self.sketches_file)}")
            return True
        except Exception as e:
            print(f"Error saving sketch: {e}")
            return False
    
    def _prompt_user_approval(self):
        """Prompt user for approval in console"""
        sketch_value = self._get_current_sketch()
        print("\n" + "-"*60)
        if sketch_value:
            print(f"Current sketch parameter: {sketch_value[:50]}...")
        else:
            print("No sketch parameter found in current URL")
        print("-"*60)
        print("\nType 'Y' to save the current sketch parameter to sketches.txt")
        print("Type anything else to continue (page will refresh)")
        print("Waiting for your response...")
        print("> ", end='', flush=True)  # Print prompt and flush
        
        self.waiting_for_approval = True
        self.user_response = None
        
        # Check if stdin is available and is a TTY
        import sys
        if sys.stdin.isatty():
            # stdin is a terminal - use background thread
            input_thread = threading.Thread(target=self._get_user_input, daemon=False)
            input_thread.start()
            self.input_thread = input_thread
        else:
            # stdin is not a terminal (redirected or background process)
            # Try to read from stdin anyway, but warn user
            print("\nWARNING: stdin doesn't appear to be a terminal.")
            print("If running via script, ensure stdin is connected.")
            print("Attempting to read input anyway...")
            input_thread = threading.Thread(target=self._get_user_input, daemon=False)
            input_thread.start()
            self.input_thread = input_thread
        
        # Wait for response (with timeout check in main loop)
        return True
    
    def _get_user_input(self):
        """Get user input in a separate thread"""
        import sys
        
        try:
            # Flush stdout first to ensure prompt is visible
            sys.stdout.flush()
            sys.stderr.flush()  # Also flush stderr
            
            # Debug: Check stdin status
            if not sys.stdin.isatty():
                print("\n[DEBUG] stdin is not a TTY - may have issues reading input")
                print("[DEBUG] If running via script, try running without '&' or ensure stdin is connected")
            
            # Try to read from stdin
            # Use readline() which blocks properly
            try:
                # Make sure stdin is not closed
                if sys.stdin.closed:
                    print("\nERROR: stdin is closed. Cannot read input.")
                    with self.approval_lock:
                        self.user_response = None
                        self.waiting_for_approval = False
                    return
                
                # Read a line (this will block until input is available)
                line = sys.stdin.readline()
                
                if not line:
                    # EOF
                    print("\nEOF reached on stdin.")
                    with self.approval_lock:
                        self.user_response = None
                        self.waiting_for_approval = False
                    return
                
                response = line.strip().upper()
                
            except (EOFError, KeyboardInterrupt, OSError) as e:
                # Handle Ctrl+C, EOF, or I/O errors
                print(f"\nInput error: {e}.")
                with self.approval_lock:
                    self.user_response = None
                    self.waiting_for_approval = False
                return
            
            # Accept any response, but only Y triggers save
            with self.approval_lock:
                self.user_response = response
                self.waiting_for_approval = False
                print(f"\nReceived: {response}")
                    
        except Exception as e:
            print(f"\nError getting user input: {e}")
            import traceback
            traceback.print_exc()
            with self.approval_lock:
                self.user_response = None
                self.waiting_for_approval = False
    
    def _check_user_response(self):
        """Check if user has responded"""
        with self.approval_lock:
            if self.user_response is not None:
                response = self.user_response
                self.user_response = None
                return response
            return None
    
    def _refresh_and_approve_loop(self):
        """Main loop for refreshing page and handling user input"""
        refresh_interval = 10.0  # Refresh every 10 seconds
        
        while self.refresh_active:
            try:
                # Refresh the page
                if not self._refresh_hydra():
                    print("Failed to refresh Hydra, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Wait a moment for visual to render
                time.sleep(2)
                
                # Prompt user for approval
                self._prompt_user_approval()
                
                # Wait for user response (check periodically)
                timeout = refresh_interval  # Wait until next refresh
                start_wait = time.time()
                
                while self.waiting_for_approval and (time.time() - start_wait < timeout):
                    response = self._check_user_response()
                    
                    if response == 'Y':
                        # User wants to save sketch
                        sketch_value = self._get_current_sketch()
                        if sketch_value:
                            if self._save_sketch(sketch_value):
                                print("Sketch saved successfully!")
                            else:
                                print("Error saving sketch")
                        else:
                            print("No sketch parameter found in current URL")
                        break
                    elif response:
                        # User typed something else - just continue
                        break
                    
                    time.sleep(0.5)
                
                # Wait before next refresh
                time.sleep(refresh_interval)
                
            except Exception as e:
                print(f"Error in refresh loop: {e}")
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

