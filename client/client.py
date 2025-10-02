from ast import arg
import json
from io import BytesIO
import base64
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import argparse
import requests
import socket
import io
import urllib.parse
import random

# from pyppeteer import launchimport socket

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

import numpy as np
from threading import Thread
import threading
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from text_scroller import TextScroller

from ai_helper import AiHelper

# Configure ChromeOptions
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")  # No sandbox for Pi
#chrome_options.add_argument("--ignore-gpu-blacklist")
#chrome_options.add_argument("--enable-webgl")
#chrome_options.add_argument("--use-fake-device-for-media-stream");
#chrome_options.add_argument("--use-fake-ui-for-media-stream")

chrome_options.add_experimental_option("prefs", { \
    "profile.default_content_setting_values.media_stream_mic": 1, 
    "profile.default_content_setting_values.media_stream_camera": 1
})

# Initialize the WebDriver instance
service = Service('/usr/bin/chromedriver')  # Path to Chromium's driver

import ws2812
import simulation
import mqtt

lock = threading.RLock()

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

class Client:
    def __init__(self, leds, server=False):
        self.width = 40
        self.height = 30
        self.server = server
        self.mqtt = mqtt.Mqtt(self.on_message)
        self.leds = leds
        
        # Set strip delay for synchronization (adjust as needed)
        self.leds.set_strip_delay(0.001)  # 1ms delay between strips
        
        self.text_scroller = TextScroller(self.width, self.height)
        self.display_mode = None
                
        # Add monitoring variables
        self.last_frame = None
        self.static_frame_count = 0
        self.max_static_frames = 50  # About 5 seconds at 0.05s refresh rate
        self.monitoring_active = True
        
        # Initialize AI helper
        self.ai_helper = AiHelper()
        
        # Start monitoring thread
        self.monitor_thread = Thread(target=self.monitor_display)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Music event visualization settings
        self.visualization_intensity = 0.5  # Control overall brightness
        self.sparse_mode = True  # Enable sparse patterns for music events
        
        # Music event mode - no text scrolling needed
        self.music_mode = True
        
        # Add camera snapshot
        self.camera_snapshot = None
        self.camera_snapshot_time = 0
        self.camera_snapshot_interval = 1  # Update snapshot every second

    def monitor_display(self):
        """Thread function to monitor display for static frames"""
        while self.monitoring_active:
            if self.display_mode == 'website' and self.last_frame is not None:
                try:
                    # Check if frame is static
                    if self.check_static_frame(self.last_frame):
                        self.static_frame_count += 1
                        if self.static_frame_count >= self.max_static_frames:
                            self.handle_visualization_failure()
                    else:
                        self.static_frame_count = 0
                except Exception as e:
                    print(f"Error in monitor thread: {e}")
            time.sleep(0.1)  # Check every 0.1s
        
    def init(self):
        self.mqtt.connect()
        self.leds.init()
        
        # Set default mode to music visualization
        self.display_mode = 'frequency'  # Start with frequency-based music visualization

    def load_camera(self):
        self.camera = Picamera2()
        preview_config = self.camera.create_preview_configuration(main={"size": (120, 80)}, lores={"size": (120, 80)}, display="lores")
        self.camera.configure(preview_config)
        self.camera.start()  # Start camera when loaded

    def start_camera_stream(self):
        self.camera = Picamera2()
        video_config = self.camera.create_preview_configuration(main={"size": (120, 80)}, lores={"size": (120, 80)}, display="lores")
        self.camera.configure(video_config)
        encoder = H264Encoder(1000000)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", 10001))
            sock.listen()

            self.camera.encoders = encoder

            print("1")
            conn, addr = sock.accept()
            print("2")
            stream = conn.makefile("wb")
            print("3")
            encoder.output = FileOutput(stream)
            print("4")
            self.camera.start_encoder(encoder)
            print("starting stream")
            self.camera.start()
            print("streaming")
    
    def load_website(self, url = None):
        if url != None:
            self.url = url
            # Only create new browser instance for initial load
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_window_size(240, 160)
            self.driver.get(self.url)
            
            # Inject jQuery
            jquery_js = """
                if (typeof jQuery === 'undefined') {
                    var script = document.createElement('script');
                    script.src = 'https://code.jquery.com/jquery-3.6.0.min.js';
                    document.head.appendChild(script);
                }
            """
            self.driver.execute_script(jquery_js)
            
            # Wait for jQuery to load
            self.driver.execute_script("""
                return new Promise((resolve) => {
                    function checkJQuery() {
                        if (window.jQuery) {
                            resolve();
                        } else {
                            setTimeout(checkJQuery, 100);
                        }
                    }
                    checkJQuery();
                });
            """)
            
            # Hide UI elements
            self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
            self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")

    def update_camera_snapshot(self):
        """Update the base64 encoded camera snapshot"""
        try:
            # Only update if enough time has passed
            current_time = time.time()
            if current_time - self.camera_snapshot_time >= self.camera_snapshot_interval:
                # Create a temporary buffer for the image
                buffer = BytesIO()
                
                # Capture directly to buffer with small size
                self.camera.capture_file(buffer, format='png')
                buffer.seek(0)
                
                # Convert to base64
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Store with data URL prefix for PNG
                self.camera_snapshot = "data:image/png;base64, " + base64_image
                self.camera_snapshot_time = current_time
                
        except Exception as e:
            print(f"Error updating camera snapshot: {e}")

    def update_hydra_code(self, code=None):
        """Update the Hydra editor with new code"""
        try:
            if code is None:
                self.url = "http://localhost:5173"
                return
                
            # Update camera snapshot
            self.update_camera_snapshot()
            
            # Replace camera token with actual snapshot if present
            if self.camera_snapshot and "CAMERA_FEED_TOKEN" in code:
                code = code.replace("CAMERA_FEED_TOKEN", self.camera_snapshot)
            
            print(code)

            # Use jQuery to update editor and run code
            # js_code = f"""
            #     // Click editor
            #     $('.CodeMirror').click();
                
            #     // Select all text and delete
            #     for (var i = 0; i < $('.CodeMirror').length; i++) {{
            #         $('.CodeMirror')[i].CodeMirror.setValue('');
            #     }}
                
            #     // Set new code
            #     $('.CodeMirror')[0].CodeMirror.setValue(`{code}`);
                
            #     // Click run button
            #     $('#run-icon').click();
            # """
            # self.driver.execute_script(js_code)

            new_code = base64.b64encode(code.encode('utf-8'))
            self.url = "http://localhost:5173?code=" + urllib.parse.quote_plus(new_code)
            
            # Load new URL
            self.driver.get(self.url)
            
            # Hide UI elements again after reload
            self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
            self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")

        except Exception as e:
            print(f"Error updating code: {e}")

    def on_message(self, client, userdata, msg):
        decoded = json.loads(msg.payload.decode())
        if decoded['type'] == "frequency":
            self.display_mode = 'frequency'
            self.frequency_display(decoded)
        if decoded['type'] == "image":
            self.rgb_display(decoded)
        if decoded['type'] == "rgb":
            self.image_display(decoded)
        if decoded['type'] == "blackout":
            self.leds.blackout()
        if decoded['type'] == "hydra":
            print(decoded)
            try:
                content = json.loads(decoded['content'])            
                if 'code' in content:
                    self.update_hydra_code(content['code'])
                
                # Update last visualization time
                self.last_visualization_time = time.time()
            except Exception as e:
                print(f"Error processing hydra message: {e}")

    def update_display(self):
        """Main display update method"""
        if self.display_mode == 'website':
            self.website_display()
        elif self.display_mode == 'camera':
            self.camera_display()
        elif self.display_mode == 'dashboard':
            self.dashboard_display()
        elif self.display_mode == 'frequency':
            # Frequency display is handled by mqtt messages
            pass
        elif self.display_mode == 'music':
            # Music visualization mode - focus on sparse patterns
            self.music_visualization_display()

    def frequency_display(self, msg):
        """Enhanced frequency display for music events - sparse patterns"""
        # Clear screen first
        self.leds.blackout()
        
        # Get frequency data
        frequencies = msg["frequencies"]
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Create sparse pulsing center pattern
        max_freq = max(frequencies) if frequencies else 0
        intensity = min(max_freq * self.visualization_intensity, 1.0)
        
        # Pulsing center effect
        radius = int(intensity * 8)  # Max radius of 8 pixels
        for x in range(max(0, center_x - radius), min(self.width, center_x + radius + 1)):
            for y in range(max(0, center_y - radius), min(self.height, center_y + radius + 1)):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if distance <= radius:
                    # Fade intensity based on distance from center
                    fade = 1.0 - (distance / radius) if radius > 0 else 1.0
                    brightness = int(intensity * fade * 100)
                    
                    # Color based on frequency bands
                    if len(frequencies) > 0:
                        freq_index = min(int((x / self.width) * len(frequencies)), len(frequencies) - 1)
                        freq_value = frequencies[freq_index]
                        
                        # Color mapping: bass=red, mid=green, treble=blue
                        if freq_index < len(frequencies) // 3:  # Bass
                            r, g, b = brightness, 0, 0
                        elif freq_index < 2 * len(frequencies) // 3:  # Mid
                            r, g, b = 0, brightness, 0
                        else:  # Treble
                            r, g, b = 0, 0, brightness
                    else:
                        r, g, b = brightness, brightness // 2, brightness // 3
                    
                    self.leds.set_pixel_color(x, y, r, g, b)

    def music_visualization_display(self):
        """Display method for music visualization mode"""
        # This would be called when in music mode
        # The actual visualization is handled by the Hydra code
        pass
    
    def image_display(self, msg):
        for pix in msg["pixels"]:
            self.leds.set_pixel_color(pix[0][0], pix[0][1], pix[1][0], pix[1][1], pix[1][2])

    def rgb_display(self, msg):
        im = Image.open(BytesIO(base64.b64decode(msg['image'])))
        self.pil_display(im)

    def bytes_display(self, img):
        im = ImageEnhance.Contrast(Image.open(BytesIO(base64.b64decode(img)))).enhance(150)
        self.pil_display(im)

    def pil_display(self, pil):
        im = pil.resize((self.width, self.height), Image.LANCZOS)
        for i in range(self.width):
            for j in range(self.height):
                pix = im.getpixel((i, self.height - j - 1))
                self.leds.set_pixel_color(i, j, pix[0], pix[1], pix[2])
    
    def camera_display(self):
        image = self.camera.capture_file("cam.png")
        self.pil_display(Image.open("cam.png"))
        #self.image_display({"pixels": [((x, y), (image[x][y][0], image[x][y][1], image[x][y][2])) for x in range(len(image)) for y in range(len(image[0]))]})

    def dashboard_display(self):
        sensor_response = requests.get("http://192.168.0.46/api/45F3isezBAfXK82b401E9MfiyFgAMCIs7nIGtoUV/sensors/12").json()

        im = Image.new("RGB", (60,40))
        draw = ImageDraw.Draw(im)
        fn = ImageFont.truetype('')
        draw.text((0,0), "Hello", font=fn)
        del draw

        self.pil_display(im)

      
    def website_display(self):
        try:
            image = self.driver.get_screenshot_as_base64()
            frame = Image.open(BytesIO(base64.b64decode(image)))
            self.last_frame = frame
            self.bytes_display(image)
        except Exception as e:
            print(f"Error in website display: {e}")
            self.handle_visualization_failure()

    def check_static_frame(self, frame):
        """Check if frame is static (black or single color) or has error logs"""
        try:
            # Then check frame
            small_frame = frame.resize((4, 3), Image.LANCZOS)
            pixels = list(small_frame.getdata())
            
            # Check if all pixels are the same or black
            first_pixel = pixels[0]
            is_black = first_pixel == (0, 0, 0)
            all_same = all(pixel == first_pixel for pixel in pixels)
            
            return is_black and all_same
        except Exception as e:
            print(f"Error checking frame: {e}")
            return False

    def handle_visualization_failure(self):
        """Handle failed visualization by generating new music visualization"""
        print("Visualization failed, generating new music visualization")
        
        # Reset counters
        self.static_frame_count = 0
        self.last_frame = None
        
        # Generate new music visualization
        new_visualization = self.generate_music_visualization()
        if new_visualization:
            try:
                content = json.loads(new_visualization)
                if 'code' in content:
                    self.update_hydra_code(content['code'])
            except Exception as e:
                print(f"Error processing new visualization: {e}")
                # Use fallback visualization
                fallback = self.ai_helper.generate_failure_visualization()
                if fallback and 'code' in fallback:
                    self.update_hydra_code(fallback['code'])
        else:
            # Use fallback visualization
            fallback = self.ai_helper.generate_failure_visualization()
            if fallback and 'code' in fallback:
                self.update_hydra_code(fallback['code'])

    def generate_music_visualization(self):
        """Generate a new music visualization when needed"""
        try:
            # Generate visualization based on current audio or random music prompt
            music_prompts = [
                "Create a sparse pulsing center pattern that reacts to bass frequencies",
                "Generate a wave pattern that responds to mid-range frequencies with low brightness",
                "Make a particle effect that pulses with the beat in the center of the screen",
                "Create a geometric pattern that breathes with the music",
                "Generate a color gradient that shifts with frequency changes"
            ]
            
            prompt = random.choice(music_prompts)
            response = self.ai_helper.generate_visualization(prompt)
            if response:
                return response
        except Exception as e:
            print(f"Error generating music visualization: {e}")
        return None

    def cleanup(self):
        """Stop monitoring thread and cleanup"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)

def start(args, client):
    while True:
        client.update_display()
        client.leds.show()

if __name__ == '__main__':
    client = None
    try:
        parser = argparse.ArgumentParser(description='LED Screen Client')
        parser.add_argument('--website', metavar='N', type=str, nargs='+',
                            help='Website to display')
        parser.add_argument('--mode', metavar='N', type=str, nargs='+',
                            help='Use camera mode')
        parser.add_argument('--simulate', type=bool, action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--server', type=bool, action=argparse.BooleanOptionalAction, default=False)
    
        args = parser.parse_args()
        
        server = args.server
        simulate = args.simulate

        leds = simulation.Leds(40, 30) if simulate else ws2812.Leds(40, 30)

        client = Client(leds, server)

        if(args.mode and "camera" in args.mode):
            client.load_camera()
            client.camera.start()
            client.display_mode = 'camera'
        elif(args.mode and "website" in args.mode):
            client.load_camera()
            client.camera.start()
            if(args.website):
                client.load_website(args.website[0])
                client.display_mode = 'website'
        elif(args.mode and "dashboard" in args.mode):
            client.display_mode = 'dashboard'

        client.init()

        thread = Thread(target=start, args=(args, client))
        thread.start()
        thread.join()

    except KeyboardInterrupt:
        print("Exiting LED client")
    finally:
        if client:
            if(args.mode == "camera"):
                client.camera.stop()
            client.cleanup()  # Stop monitoring thread
            client.leds.blackout()
            client.leds.show()