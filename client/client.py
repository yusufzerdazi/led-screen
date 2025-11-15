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

# Hardware-specific audio imports (only needed on real hardware, not in simulation)
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not available - audio capture will be disabled in simulation mode")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("librosa not available - advanced audio analysis features will be limited")

import wave
import struct
import numpy as np

# Selenium imports removed for music visualizer

from ai_helper import AiHelper

# Chrome/Selenium configuration removed for music visualizer

# Hardware-specific LED imports (ws2812 requires spidev which is Pi-specific)
try:
    import ws2812
    WS2812_AVAILABLE = True
except ImportError as e:
    WS2812_AVAILABLE = False
    print(f"ws2812 hardware module not available: {e}")
    print("Real LED hardware will not work, but simulation mode is available")

import simulation
import mqtt

# Speech-to-text import (optional)
try:
    from speech_to_text import SpeechToText
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("Speech-to-text not available - install SpeechRecognition and openai-whisper")

DEFAULT_TARGET_FPS = 24.0

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
        
        # Initialize MQTT only if server mode is enabled
        if self.server:
            self.mqtt = mqtt.Mqtt(self.on_message)
            print("MQTT enabled - will listen for remote messages")
        else:
            self.mqtt = None
            print("MQTT disabled - running in standalone mode")
        
        self.leds = leds
        self.target_fps = DEFAULT_TARGET_FPS
        self.frame_interval = 1.0 / self.target_fps
        
        # Current display mode
        self.current_mode = None
        
        # Set strip delay for synchronization (adjust as needed)
        self.leds.set_strip_delay(0.001)  # 1ms delay between strips
        
        self.display_mode = None
                
        # Add monitoring variables
        self.last_frame = None
        self.static_frame_count = 0
        self.max_static_frames = 50  # About 5 seconds at 0.05s refresh rate
        self.monitoring_active = True
        
        # AI helper removed - using stock visuals instead
        
        # Generic display state (used by legacy code and fallbacks)
        self.display_mode = None
        self.last_frame = None
        
        # Start monitoring thread (generic, used by all modes)
        self.monitoring_active = True
        self.monitor_thread = Thread(target=self.monitor_display)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Speech-to-text (optional, can be enabled per mode)
        self.speech_to_text = None
        self.stt_enabled = False
        
        # Check if simulation mode
        self.is_simulation = isinstance(leds, simulation.Leds)
        if self.is_simulation:
            print("Running in simulation mode")





    def monitor_display(self):
        """Thread function to monitor display for static frames"""
        while self.monitoring_active:
            # Monitoring disabled - no fallback needed
            time.sleep(0.1)  # Check every 0.1s
        
    def init(self):
        # Connect to MQTT if server mode is enabled
        if self.mqtt:
            self.mqtt.connect()
        
        self.leds.init()
        
        # Audio capture handled by current mode if needed
        
        # Speech-to-text can be enabled by modes if needed
        # No AI greeting - using stock visuals

    
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
                chrome_options.add_argument("--headless=new")  # Headless with better WebGL support
                chrome_options.add_argument("--no-sandbox")  # No sandbox for Pi
                chrome_options.add_argument("--use-gl=egl")  # Enable EGL for WebGL
                chrome_options.add_argument("--enable-webgl")
                chrome_options.add_argument("--ignore-gpu-blocklist")
                chrome_options.add_argument("--window-size=240,160")
                
                # Fallback if webdriver-manager is not installed
                import shutil
                chromedriver_path = shutil.which('chromedriver')
                if not chromedriver_path:
                    for path in ['/usr/bin/chromedriver', '/usr/local/bin/chromedriver']:
                        import os
                        if os.path.exists(path):
                            chromedriver_path = path
                            break
                
                if chromedriver_path:
                    service = Service(chromedriver_path)
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                else:
                    # Try without explicit service path (uses PATH)
                    self.driver = webdriver.Chrome(options=chrome_options)
                self.driver.set_window_size(240, 160)
                self.driver.get(self.url)
                
                # Hide UI elements to get clean visualization
                try:
                    self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                    self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                    # Auto-run any existing code
                    try:
                        self.driver.execute_script("""
                            // Auto-run the current code if hydraSynth is available
                            if (typeof hydraSynth !== 'undefined' && hydraSynth) {
                                try {
                                    eval(hydraSynth.getCode());
                                } catch(e) {
                                    console.log('Auto-run failed:', e);
                                }
                            }
                        """)
                        print("Auto-executed initial Hydra code")
                    except Exception as e:
                        print(f"Could not auto-run initial code: {e}")
                    print("Hidden Hydra UI overlays")
                except Exception as e:
                    print(f"Could not hide overlays: {e}")
                
                print(f"Loaded Hydra visualizer at: {url}")
            except Exception as e:
                print(f"Error setting up webdriver: {e}")
                self.driver = None

    
    
    
    def update_audio_levels(self, frequencies):
        """Update audio frequency levels for visualization (MQTT fallback)"""
        # Delegate to current mode if it has this method
        if self.current_mode and hasattr(self.current_mode, 'update_audio_levels'):
            return self.current_mode.update_audio_levels(frequencies)

    def update_hydra_code(self, code=None):
        """Update the Hydra editor with new code"""
        # Delegate to current mode if it has this method
        if self.current_mode and hasattr(self.current_mode, 'update_hydra_code'):
            return self.current_mode.update_hydra_code(code)

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
            except Exception as e:
                print(f"Error processing hydra message: {e}")

    def set_mode(self, mode):
        """
        Set the current display mode.
        
        Args:
            mode: A BaseMode instance
        """
        # Clean up old mode if exists
        if self.current_mode:
            self.current_mode.cleanup()
        
        self.current_mode = mode
        print(f"Mode set to: {mode.__class__.__name__}")
        
        # Initialize the new mode
        if self.current_mode:
            self.current_mode.init()
    
    def update_display(self):
        """Main display update method"""
        if self.current_mode:
            # Get frame from mode and display it
            frame = self.current_mode.update()
            if frame:
                self.pil_display(frame)
        elif self.display_mode == 'frequency':
            # Fallback for MQTT frequency mode
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
        if pil is None:
            print("Error: PIL image is None")
            return
            
        im = pil.resize((self.width, self.height), Image.LANCZOS)
        
        # Simple display - just show the image as-is
        for i in range(self.width):
            for j in range(self.height):
                pix = im.getpixel((i, self.height - j - 1))
                self.leds.set_pixel_color(i, j, pix[0], pix[1], pix[2])

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
            # No initial loading needed - just use whatever is on the page
            
            # Try to get screenshot from Hydra if available
            if hasattr(self, 'driver') and self.driver:
                try:
                    # Non-blocking FPS cap for screenshots
                    now = time.time()
                    # FPS limiting handled by current mode if needed
                    image = self.driver.get_screenshot_as_base64()
                    frame = Image.open(BytesIO(base64.b64decode(image)))
                    
                    # Check if frame is valid
                    if frame is None:
                        print("Error: Screenshot frame is None")
                        return
                    
                    # Visual processing handled by current mode if needed
                    
                    # Display the frame directly
                    self.last_frame = frame
                    # Render the already-decoded PIL frame directly
                    self.pil_display(frame)
                    # Display mode logging handled by current mode
                    return
                except Exception as e:
                    print(f"Error getting Hydra screenshot: {e}")
                    return
            else:
                print("No driver available for screenshots")
                return
            
        except Exception as e:
            print(f"Error in website display: {e}")


    def check_interjections(self):
        """Thread removed for music visualizer"""
        pass

    def enable_speech_to_text(self, model_size="base", on_text_callback=None):
        """Enable speech-to-text listening
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            on_text_callback: Optional callback function(text) called when text is transcribed
        """
        if not STT_AVAILABLE:
            print("Speech-to-text not available")
            return False
        
        if self.speech_to_text:
            print("Speech-to-text already enabled")
            return True
        
        try:
            self.speech_to_text = SpeechToText(model_size=model_size, use_whisper=True)
            
            # Default callback: just log the text
            def default_callback(text):
                print(f"[STT] Transcribed speech: {text}")
            
            callback = on_text_callback if on_text_callback else default_callback
            self.speech_to_text.start_listening(on_text_callback=callback)
            self.stt_enabled = True
            print("Speech-to-text enabled and listening")
            return True
        except Exception as e:
            print(f"Error enabling speech-to-text: {e}")
            return False
    
    def disable_speech_to_text(self):
        """Disable speech-to-text listening"""
        if self.speech_to_text:
            self.speech_to_text.stop_listening()
            self.speech_to_text = None
            self.stt_enabled = False
            print("Speech-to-text disabled")

    def cleanup(self):
        """Stop monitoring thread and cleanup"""
        self.monitoring_active = False
        
        # Clean up current mode
        if self.current_mode:
            self.current_mode.cleanup()
        
        # Stop speech-to-text
        self.disable_speech_to_text()
        
        # Stop audio capture (legacy cleanup for non-refactored code)
        # Audio cleanup handled by current mode if needed
        if hasattr(self, 'audio_stream') and self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
            print("Audio capture stopped")
        
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

def start(args, client, console_ui=None):
    """Main update loop"""
    import time
    
    def update():
        frame_start = time.time()
        client.update_display()
        client.leds.show()
        frame_time = time.time() - frame_start
        
        # Track performance in console UI
        if console_ui:
            console_ui.track_performance('LED Display', frame_time)
    
    # Check if we're using simulation mode
    if hasattr(client.leds, 'start_event_loop'):
        # Simulation mode - use Qt event loop
        client.leds.start_event_loop(update, client.frame_interval)
    else:
        # Real hardware mode - run as fast as possible (no FPS limiting)
        # LED timing is critical and adding delays causes glitches
        while True:
            update()

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
        parser.add_argument('--console', type=bool, action=argparse.BooleanOptionalAction, default=False,
                            help='Enable Kaleidoscape console UI for monitoring and debugging')
        parser.add_argument('--stt', dest='enable_stt', type=bool, action=argparse.BooleanOptionalAction,
                            default=True, help='Enable on-device speech-to-text (Whisper). Use --no-stt to skip.')
        args = parser.parse_args()
        
        server = args.server
        simulate = args.simulate
        test_mode = args.test

        # Initialize LED hardware or simulation
        if simulate:
            leds = simulation.Leds(40, 30)
        else:
            if WS2812_AVAILABLE:
                leds = ws2812.Leds(40, 30, 1)  # 10% brightness (matching main branch)
            else:
                print("WARNING: ws2812 hardware not available, falling back to simulation mode")
                print("To use real hardware, ensure you're on a Raspberry Pi with required dependencies")
                simulate = True  # Update flag to reflect actual mode
                leds = simulation.Leds(40, 30)

        client = Client(leds, server)

        # Determine which mode to use
        from modes import get_mode, list_modes
        
        mode_name = None
        mode_config = {}
        
        if args.mode:
            # Get mode name from arguments
            mode_name = args.mode[0] if isinstance(args.mode, list) else args.mode
            
            # Handle website mode with URL
            if mode_name == 'website' and args.website:
                mode_config['url'] = args.website[0]
        else:
            # Default mode
            mode_name = 'tush'
        
        # Create mode instance
        mode = get_mode(mode_name, client.width, client.height)
        if not mode:
            print(f"Unknown mode: {mode_name}")
            print(f"Available modes: {', '.join(list_modes())}")
            exit(1)
        
        # Allow CLI to override mode-specific features (e.g., speech-to-text)
        if mode_name == 'decompression':
            mode_config['enable_stt'] = args.enable_stt
        
        # Set up and initialize mode
        mode.setup(**mode_config)

        # Ensure runtime attribute override for modes that inspect enable_stt during init
        if hasattr(mode, 'enable_stt') and 'enable_stt' in mode_config:
            mode.enable_stt = mode_config['enable_stt']
        client.set_mode(mode)

        client.init()
        
        # Ensure decompression mode starts in its configured base status
        if mode_name == 'decompression' and hasattr(mode, 'set_status'):
            mode.set_status('people_kaleidoscope', is_base_status=True)

        # Start console UI if requested (after initialization)
        console_ui = None
        if args.console:
            try:
                from console_ui import KaleidoscapeUI
                console_ui = KaleidoscapeUI(client, mode)
                
                # Start display loop in background thread
                display_thread = threading.Thread(target=lambda: start(args, client, console_ui), daemon=True)
                display_thread.start()
                
                # Store console UI reference in mode for service control
                # This allows the mode to check service enabled/disabled status
                mode._console_ui_ref = console_ui
                
                # Run console UI in main thread (blocks here) - no prints before this
                console_ui.run()
            except ImportError as e:
                print(f"Warning: Could not start console UI: {e}")
                print("Install rich library: pip install rich")
                # Fall back to normal mode
                start(args, client, None)
            except Exception as e:
                print(f"Warning: Console UI error: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to normal mode
                start(args, client, None)
        else:
            # Normal mode - run display loop in main thread
            start(args, client, console_ui)

    except KeyboardInterrupt:
        print("Exiting LED client")
    finally:
        if client:
            client.cleanup()  # Stop monitoring thread
            client.leds.blackout()
            client.leds.show()
