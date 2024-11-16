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
        self.text_scroller = TextScroller(self.width, self.height)
        self.display_mode = None
        
        # Add cooldown tracking
        self.last_visualization_time = 0
        self.cooldown_period = 120  # 2 minutes in seconds
        
        # Add monitoring variables
        self.last_frame = None
        self.static_frame_count = 0
        self.max_static_frames = 100  # About 5 seconds at 0.05s refresh rate
        self.failure_quips = [
            "That's never happened before :(",
            "Oops, my bad!",
            "Let me try again...",
            "Well this is awkward",
            "Time for Plan B!"
        ]

    def can_show_visualization(self):
        """Check if enough time has passed since last visualization"""
        current_time = time.time()
        time_since_last = current_time - self.last_visualization_time
        return time_since_last >= self.cooldown_period
    
    def init(self):
        self.mqtt.connect()
        self.leds.init()

    def load_camera(self):
        self.camera = Picamera2()
        preview_config = self.camera.create_preview_configuration(main={"size": (120, 80)}, lores={"size": (120, 80)}, display="lores")
        self.camera.configure(preview_config)

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
            
            # Hide UI elements
            self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
            self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")

    def update_hydra_code(self, code):
        """Update the Hydra editor with new code by reloading with code parameter"""
        try:
            # Encode code as base64 and construct URL
            print(code)
            new_code = base64.b64encode(code.encode('utf-8'))
            self.url = "http://localhost:5173?code=" + urllib.parse.quote_plus(new_code.decode('utf-8'))
            
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
                if 'display' in content and content['display']:
                    # Check cooldown before showing new visualization
                    if not self.can_show_visualization():
                        print(f"Cooldown active. Please wait {self.cooldown_period - (time.time() - self.last_visualization_time):.0f} seconds")
                        return
                    
                    if 'code' in content:
                        self.update_hydra_code(content['code'])

                    if 'quip' in content:
                        self.text_scroller.start_scroll(content['quip'])
                        self.display_mode = 'scroll'
                    
                    # Update last visualization time
                    self.last_visualization_time = time.time()
            except Exception as e:
                print(f"Error processing hydra message: {e}")

    def update_display(self):
        """Main display update method"""
        if self.display_mode == 'scroll':
            if self.text_scroller.is_scrolling:
                frame = self.text_scroller.get_frame()
                if frame:
                    self.pil_display(frame)
            else:
                self.display_mode = 'website'
        elif self.display_mode == 'website':
            self.website_display()
        elif self.display_mode == 'camera':
            self.camera_display()
        elif self.display_mode == 'dashboard':
            self.dashboard_display()
        elif self.display_mode == 'frequency':
            # Frequency display is handled by mqtt messages
            pass

    def frequency_display(self, msg):
        x = 0
        for f in msg["frequencies"]:
            for y in range(30):
                if int(f * 4) > y:
                    self.leds.set_pixel_color(x, y, 100, 255 - 6 * x, 255 - 8 * y)
                else:  
                    self.leds.set_pixel_color(x, y, 0, 0, 0)
            x += 1
    
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
            
            # Check if frame is static (black or single color)
            is_static = self.check_static_frame(frame)
            
            if is_static:
                self.static_frame_count += 1
                if self.static_frame_count >= self.max_static_frames:
                    self.handle_visualization_failure()
                    return
            else:
                self.static_frame_count = 0
                
            self.last_frame = frame
            self.bytes_display(image)
            
        except Exception as e:
            print(f"Error in website display: {e}")
            self.handle_visualization_failure()

    def check_static_frame(self, frame):
        """Check if frame is static (black or single color)"""
        try:
            # Resize for faster processing
            small_frame = frame.resize((4, 3), Image.LANCZOS)
            pixels = list(small_frame.getdata())
            
            # Check if all pixels are the same or black
            first_pixel = pixels[0]
            is_black = first_pixel == (0, 0, 0)
            all_same = all(pixel == first_pixel for pixel in pixels)
            
            return is_black or all_same
        except Exception as e:
            print(f"Error checking frame: {e}")
            return False

    def handle_visualization_failure(self):
        """Handle failed visualization by showing quip and requesting new one"""
        print("Visualization failed, requesting new one")
        
        # Reset counters
        self.static_frame_count = 0
        self.last_frame = None
        
        # Show failure quip
        import random
        failure_quip = random.choice(self.failure_quips)
        self.text_scroller.start_scroll(failure_quip)
        self.display_mode = 'scroll'
        
        # Request new visualization with retry prompt
        retry_message = {
            "type": "hydra",
            "content": json.dumps({
                "code": "",
                "description": "Previous visualization failed",
                "quip": failure_quip,
                "display": True,
                "retry": True
            })
        }
        self.mqtt.client.publish("mqtt/led-screen", json.dumps(retry_message))

def start(args, client):
    while True:
        client.update_display()
        client.leds.show()
        time.sleep(0.05)  # Small delay to control refresh rate

if __name__ == '__main__':
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
            if(args.website):
                client.load_website(args.website[0])
                client.display_mode = 'website'
        elif(args.mode and "dashboard" in args.mode):
            client.display_mode = 'dashboard'

        client.init()
        #client.leds.start()

        thread = Thread(target=start, args=(args, client))
        thread.start()
        

    except KeyboardInterrupt:
        print("Exiting LED client")
        if(args.mode == "camera"):
            client.camera.stop()
        client.leds.blackout()
        client.leds.show()