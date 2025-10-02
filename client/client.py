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

# Camera imports removed for music visualizer

import numpy as np
from threading import Thread
import threading
import time

# Selenium imports removed for music visualizer

from text_scroller import TextScroller

from ai_helper import AiHelper

# Chrome/Selenium configuration removed for music visualizer

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
        
        # Music visualizer variables
        self.last_visualization_time = time.time()
        self.visualization_interval = 15  # Generate new visualization every 15 seconds
        self.current_visualization = None
        
        # Start visualization generation thread
        self.visualization_thread = Thread(target=self.generate_visualizations)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        
        # Music visualizer state
        self.audio_levels = [0] * 8  # 8 frequency bands
        self.last_audio_update = time.time()

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
        
        # Show startup greeting
        greeting = self.ai_helper.generate_greeting()
        self.queue_text(greeting)

    def generate_visualizations(self):
        """Thread to generate new visualizations periodically"""
        while self.monitoring_active:
            current_time = time.time()
            if current_time - self.last_visualization_time >= self.visualization_interval:
                # Generate new visualization
                prompts = [
                    "Create a sparse, center-focused music visualizer using osc() and src() functions. Use a.fft[0] for bass and a.fft[4] for treble. Make it pulse from the center with dark edges.",
                    "Generate a music visualizer with radial patterns using osc() and modulate(). Use warm colors for bass (a.fft[0]) and cool colors for treble (a.fft[4]). Keep it sparse and center-focused.",
                    "Create a sparse music visualizer using osc() with different frequencies. Use a.fft[0] for center pulsing and a.fft[4] for outer effects. Avoid using background() function.",
                    "Design a music visualizer with center-focused patterns using osc() and src(). Use a.fft[0] for intensity and a.fft[4] for color variation. Make it pulse from center.",
                    "Generate a music visualizer with concentric circles using osc() and modulate(). Use a.fft[0] for bass pulsing and a.fft[4] for treble effects. Keep edges dark."
                ]
                import random
                prompt = random.choice(prompts)
                print(f"Generating new visualization: {prompt}")
                response = self.ai_helper.generate_visualization(prompt)
                if response:
                    try:
                        content = json.loads(response)
                        if 'code' in content:
                            self.current_visualization = content['code']
                            self.update_hydra_code(content['code'])
                            self.last_visualization_time = current_time
                            print(f"New visualization generated: {content.get('description', 'Music visualizer')}")
                    except Exception as e:
                        print(f"Error processing visualization: {e}")
            time.sleep(5)  # Check every 5 seconds
    
    def load_website(self, url = None):
        if url != None:
            self.url = url
            # Set up webdriver for Hydra
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.service import Service
                from selenium.webdriver.chrome.options import Options
                
                # Configure ChromeOptions
                chrome_options = Options()
                chrome_options.add_argument("--headless")  # Run in headless mode
                chrome_options.add_argument("--no-sandbox")  # No sandbox for Pi
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=240,160")
                
                # Initialize the WebDriver instance
                service = Service('/usr/bin/chromedriver')  # Path to Chromium's driver
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                self.driver.set_window_size(240, 160)
                self.driver.get(self.url)
                
                print(f"Loaded Hydra visualizer at: {url}")
            except Exception as e:
                print(f"Error setting up webdriver: {e}")
                self.driver = None

    def update_audio_levels(self, frequencies):
        """Update audio frequency levels for visualization"""
        self.audio_levels = frequencies[:8]  # Use first 8 frequency bands
        self.last_audio_update = time.time()

    def update_hydra_code(self, code=None):
        """Update the Hydra editor with new code"""
        try:
            if code is None:
                self.url = "http://localhost:5173"
                return
                
            print(f"Updating Hydra code: {code}")

            # Encode code for URL
            new_code = base64.b64encode(code.encode('utf-8'))
            self.url = "http://localhost:5173?code=" + urllib.parse.quote_plus(new_code)
            
            print(f"New URL: {self.url}")

        except Exception as e:
            print(f"Error updating code: {e}")

    def on_message(self, client, userdata, msg):
        decoded = json.loads(msg.payload.decode())
        if decoded['type'] == "frequency":
            self.display_mode = 'frequency'
            self.frequency_display(decoded)
            # Update audio levels for visualization
            self.update_audio_levels(decoded.get('frequencies', []))
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
                    self.current_visualization = content['code']
                
                # Update last visualization time
                self.last_visualization_time = time.time()
            except Exception as e:
                print(f"Error processing hydra message: {e}")

    def update_display(self):
        """Main display update method"""
        if self.display_mode == 'frequency':
            # Frequency display is handled by mqtt messages
            pass
        elif self.display_mode == 'website':
            self.website_display()
        elif self.display_mode == 'dashboard':
            self.dashboard_display()

    def frequency_display(self, msg):
        """Create sparse, center-focused music visualization"""
        # Clear all pixels first
        for x in range(self.width):
            for y in range(self.height):
                self.leds.set_pixel_color(x, y, 0, 0, 0)
        
        # Get frequency data
        frequencies = msg.get("frequencies", [])
        if not frequencies:
            return
            
        # Calculate center position
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Use bass frequencies (first 4) for center pulsing
        bass_level = sum(frequencies[:4]) / 4 if len(frequencies) >= 4 else 0
        bass_intensity = min(int(bass_level * 255), 255)
        
        # Use treble frequencies (last 4) for outer ring
        treble_level = sum(frequencies[-4:]) / 4 if len(frequencies) >= 4 else 0
        treble_intensity = min(int(treble_level * 255), 255)
        
        # Create pulsing center based on bass
        if bass_intensity > 10:  # Only light up if there's significant bass
            # Center pulsing circle
            radius = int(bass_intensity / 50) + 1  # Scale radius based on bass
            for x in range(max(0, center_x - radius), min(self.width, center_x + radius + 1)):
                for y in range(max(0, center_y - radius), min(self.height, center_y + radius + 1)):
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if distance <= radius:
                        # Create pulsing effect with bass
                        intensity = int(bass_intensity * (1 - distance / radius))
                        # Use warm colors for bass
                        r = intensity
                        g = int(intensity * 0.3)
                        b = int(intensity * 0.1)
                        self.leds.set_pixel_color(x, y, r, g, b)
        
        # Add outer ring based on treble
        if treble_intensity > 10:  # Only light up if there's significant treble
            outer_radius = int(treble_intensity / 30) + 3
            for x in range(max(0, center_x - outer_radius), min(self.width, center_x + outer_radius + 1)):
                for y in range(max(0, center_y - outer_radius), min(self.height, center_y + outer_radius + 1)):
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if 2 <= distance <= outer_radius:  # Ring, not filled
                        # Use cool colors for treble
                        intensity = int(treble_intensity * 0.5)
                        r = int(intensity * 0.2)
                        g = int(intensity * 0.8)
                        b = intensity
                        self.leds.set_pixel_color(x, y, r, g, b)
    
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
    
    def music_visualizer_display(self):
        """Display music-reactive visualization"""
        # This will be handled by the Hydra visualizer
        # The actual display is managed by the website_display method
        pass

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
            # Generate initial visualization if we don't have one
            if not hasattr(self, 'current_visualization') or not self.current_visualization:
                print("Generating initial music visualization...")
                prompt = "Create a sparse, center-focused music visualizer that reacts to audio input. Focus on pulsing patterns in the center with dark edges."
                response = self.ai_helper.generate_visualization(prompt)
                if response:
                    try:
                        content = json.loads(response)
                        if 'code' in content:
                            self.current_visualization = content['code']
                            self.update_hydra_code(content['code'])
                            print(f"Generated visualization: {content.get('description', 'Music visualizer')}")
                    except Exception as e:
                        print(f"Error processing initial visualization: {e}")
            
            # Try to get screenshot from Hydra if available
            if hasattr(self, 'driver') and self.driver:
                try:
                    # Wait a moment for Hydra to render
                    time.sleep(0.1)
                    image = self.driver.get_screenshot_as_base64()
                    frame = Image.open(BytesIO(base64.b64decode(image)))
                    self.last_frame = frame
                    self.bytes_display(image)
                    print("Displaying Hydra visualization")
                    return
                except Exception as e:
                    print(f"Error getting Hydra screenshot: {e}")
                    # If Hydra fails, continue to fallback
            
            # Fallback: Create a more interesting pattern
            center_x = self.width // 2
            center_y = self.height // 2
            
            # Clear all pixels
            for x in range(self.width):
                for y in range(self.height):
                    self.leds.set_pixel_color(x, y, 0, 0, 0)
            
            # Create multiple patterns for music visualization
            import math
            current_time = time.time()
            
            # Bass pattern - pulsing center
            bass_intensity = int(128 + 127 * (0.5 + 0.5 * math.sin(current_time * 2)))
            bass_radius = 2 + int(3 * (0.5 + 0.5 * math.sin(current_time * 1.5)))
            
            for x in range(max(0, center_x - bass_radius), min(self.width, center_x + bass_radius + 1)):
                for y in range(max(0, center_y - bass_radius), min(self.height, center_y + bass_radius + 1)):
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if distance <= bass_radius:
                        intensity = int(bass_intensity * (1 - distance / bass_radius))
                        r = intensity
                        g = int(intensity * 0.3)
                        b = int(intensity * 0.1)
                        self.leds.set_pixel_color(x, y, r, g, b)
            
            # Treble pattern - outer ring
            treble_intensity = int(100 + 100 * (0.5 + 0.5 * math.sin(current_time * 3)))
            treble_radius = 6 + int(2 * (0.5 + 0.5 * math.sin(current_time * 2.5)))
            
            for x in range(max(0, center_x - treble_radius), min(self.width, center_x + treble_radius + 1)):
                for y in range(max(0, center_y - treble_radius), min(self.height, center_y + treble_radius + 1)):
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if 4 <= distance <= treble_radius:  # Ring, not filled
                        intensity = int(treble_intensity * 0.7)
                        r = int(intensity * 0.2)
                        g = int(intensity * 0.8)
                        b = intensity
                        self.leds.set_pixel_color(x, y, r, g, b)
            
            # Add some sparkles for high frequencies
            if int(current_time * 10) % 3 == 0:  # Occasional sparkles
                for _ in range(3):
                    sparkle_x = center_x + int((random.random() - 0.5) * 8)
                    sparkle_y = center_y + int((random.random() - 0.5) * 8)
                    if 0 <= sparkle_x < self.width and 0 <= sparkle_y < self.height:
                        self.leds.set_pixel_color(sparkle_x, sparkle_y, 255, 255, 255)
            
        except Exception as e:
            print(f"Error in website display: {e}")

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
        """Handle failed visualization by requesting new one"""
        print("Visualization failed, requesting new one")
        
        # Generate new visualization
        prompt = "Create a sparse, center-focused music visualizer that reacts to audio input. Focus on pulsing patterns in the center with dark edges."
        response = self.ai_helper.generate_visualization(prompt)
        if response:
            try:
                content = json.loads(response)
                if 'code' in content:
                    self.current_visualization = content['code']
                    self.update_hydra_code(content['code'])
            except Exception as e:
                print(f"Error processing new visualization: {e}")

    def check_interjections(self):
        """Thread removed for music visualizer"""
        pass

    def cleanup(self):
        """Stop monitoring thread and cleanup"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        if hasattr(self, 'visualization_thread'):
            self.visualization_thread.join(timeout=1.0)

    def queue_text(self, text):
        """Text functionality removed for music visualizer"""
        pass

    def process_text_queue(self):
        """Text functionality removed for music visualizer"""
        pass

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
                            help='Use music visualizer mode')
        parser.add_argument('--simulate', type=bool, action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--server', type=bool, action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--test', type=bool, action=argparse.BooleanOptionalAction, default=False)
    
        args = parser.parse_args()
        
        server = args.server
        simulate = args.simulate
        test_mode = args.test

        leds = simulation.Leds(40, 30) if simulate else ws2812.Leds(40, 30, 0.1)

        client = Client(leds, server)

        if(args.mode and "music" in args.mode):
            # Set up music visualizer in website mode
            client.display_mode = 'website'
            client.load_website("http://localhost:5173")
        elif(args.mode and "website" in args.mode):
            if(args.website):
                client.load_website(args.website[0])
                client.display_mode = 'website'
        elif(args.mode and "dashboard" in args.mode):
            client.display_mode = 'dashboard'
        else:
            # Default to music visualizer in website mode
            client.display_mode = 'website'
            client.load_website("http://localhost:5173")

        client.init()

        thread = Thread(target=start, args=(args, client))
        thread.start()
        thread.join()

    except KeyboardInterrupt:
        print("Exiting LED client")
    finally:
        if client:
            client.cleanup()  # Stop monitoring thread
            client.leds.blackout()
            client.leds.show()
