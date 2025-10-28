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
import pyaudio
import wave
import struct
import librosa
import numpy as np

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
        
        # AI helper removed - using stock visuals instead
        
        # Start monitoring thread
        self.monitor_thread = Thread(target=self.monitor_display)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Music visualizer variables
        self.last_visualization_time = time.time()
        self.visualization_interval = 300  # Refresh page every 30 seconds
        self.current_visualization = None
        
        # Fade to black before refresh
        self.fade_start_time = None
        self.fade_duration = 2.0  # 2 seconds fade
        self.is_fading = False
        self.fade_in_start_time = None
        self.is_fading_in = False
        
        # Sparse visual detection
        self.sparse_threshold = 0.4  # If less than 40% of pixels are lit up, consider sparse
        self.is_sparse_visual = False
        self.sparse_fade_start_time = None
        self.last_visual_load_time = None
        self.sparse_detection_delay = 5.0  # Wait 5 seconds after load before detecting
        self.mode_transition_start = None
        self.mode_transition_duration = 2.0  # 2 second transition between modes
        
        # Post-processing variables for shapes
        self.shape_center_x = self.width // 2
        self.shape_center_y = self.height // 2
        self.shape_size = 6   # Much smaller - about 1/4 screen area
        self.shape_size_target = 8
        self.shape_size_speed = 0.05  # Slower size changes
        self.shape_move_speed = 0.08   # Faster movement
        self.shape_move_target_x = self.width // 2
        self.shape_move_target_y = self.height // 2
        
        # Home position and BPM matching
        self.home_x = self.width // 2
        self.home_y = self.height // 2
        self.return_to_home_time = time.time()
        self.return_to_home_interval = 8  # Return home every 8 seconds
        self.at_home = True
        self.home_duration = 2  # Stay at home for 2 seconds
        
        # BPM detection using librosa (professional audio analysis)
        self.bpm = 135  # Default BPM (middle of range)
        self.bpm_history = []  # Store recent BPM readings
        self.bpm_window_size = 5  # Number of readings to average
        self.last_bpm_update = time.time()
        self.pulse_phase = 0.0
        self.pulse_speed = 0.1
        self.min_bpm = 110  # Minimum valid BPM
        self.max_bpm = 160  # Maximum valid BPM
        
        # BPM indicator pixel
        self.bpm_indicator_on = False
        self.last_beat_time = time.time()
        
        # Audio buffer for librosa analysis
        self.audio_buffer = []
        self.buffer_size = 4096  # Buffer size for analysis
        self.analysis_interval = 0.5  # Analyze every 0.5 seconds
        
        # Multiple shapes and paths
        self.num_shapes = 3  # Number of simultaneous shapes
        self.shape_positions = []
        self.shape_sizes = []
        self.shape_paths = []  # 0=random, 1=spiral, 2=figure8, 3=orbit
        self.path_phase = 0.0
        
        # Initialize multiple shapes
        for i in range(self.num_shapes):
            self.shape_positions.append([self.width // 2, self.height // 2])
            self.shape_sizes.append(6)
            self.shape_paths.append(0)  # Start with random movement
        
        # Audio capture setup
        self.audio_enabled = True
        self.audio_thread = None
        self.audio_stream = None
        self.audio_data = []
        self.audio_lock = threading.Lock()
        
        # Initialize PyAudio
        try:
            self.p = pyaudio.PyAudio()
            self.audio_enabled = True
            print("PyAudio initialized successfully")
        except Exception as e:
            print(f"PyAudio initialization failed: {e}")
            self.audio_enabled = False
        
        # Shape switching with smooth transitions
        self.current_shape = 0  # 0=circle, 1=square, 2=triangle, 3=rectangle, 4=stars, 5=heart, 6=spiral, 7=wave, 8=overlay
        self.target_shape = 0
        self.shape_switch_time = time.time()
        self.shape_switch_interval = 60  # Switch shapes every 60 seconds
        self.shape_transition_progress = 0.0  # 0.0 = current shape, 1.0 = target shape
        self.shape_transition_speed = 0.02  # How fast to transition between shapes
        
        # BPM sync variety - different effects for different shapes
        self.bpm_effect_type = 0  # 0=size, 1=opacity, 2=movement, 3=rotation
        self.bpm_effect_switch_time = time.time()
        self.bpm_effect_switch_interval = 45  # Switch BPM effects every 45 seconds
        
        # Tempo responsiveness modes
        self.tempo_mode = 0  # 0=normal, 1=double speed
        self.tempo_switch_time = time.time()
        self.tempo_switch_interval = 600  # Switch tempo modes every 10 minutes
        self.tempo_start_time = None
        self.tempo_max_duration = 300  # Fast tempo lasts max 5 minutes
        
        # Strobe mode
        self.strobe_mode = False
        self.strobe_switch_time = time.time()
        self.strobe_switch_interval = 900  # Switch strobe mode every 15 minutes
        self.strobe_start_time = None
        self.strobe_max_duration = 20  # Strobe lasts max 20 seconds
        self.strobe_phase = 0.0
        
        # Debug flags for testing special effects (set to True to enable)
        self.debug_strobe = False
        self.debug_fast_tempo = False
        self.debug_overlay = False
        
        # Custom overlay image
        self.overlay_image = None
        self.overlay_switch_time = time.time()
        self.overlay_switch_interval = 900  # Switch to overlay every 15 minutes
        self.overlay_start_time = None
        self.overlay_max_duration = 30  # Overlay lasts max 30 seconds
        self.use_overlay = False
        
        # Speed changes
        self.speed_change_time = time.time()
        self.speed_change_interval = 300  # Change speed every 5 minutes
        self.speed_multiplier = 1.0
        
        # Rectangle scrolling
        self.rect_scroll_x = 0
        self.rect_scroll_speed = 1.0  # Faster scrolling
        self.rect_scroll_direction = 1  # 1 = right, -1 = left
        
        # Start visualization generation thread
        self.visualization_thread = Thread(target=self.generate_visualizations)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        
        # Music visualizer state
        self.audio_levels = [0] * 8  # 8 frequency bands
        self.last_audio_update = time.time()
        
        # Runtime config
        self.verbose = True
        self.screenshot_interval = 0.04  # ~25 FPS max capture
        self.last_screenshot_time = 0
        self._hydra_display_announced = False
        self.black_regen_cooldown = 10
        self.last_black_regen = 0
        self.hydra_docs_cache = None
        self.hydra_docs_path = "/home/yusuf/Code/hydra"
        self.last_screenshot_analysis = None

    def get_hydra_docs(self):
        """Extract comprehensive Hydra documentation from local repo"""
        if self.hydra_docs_cache:
            return self.hydra_docs_cache
        
        try:
            # Read README.md for basic functions and examples
            readme_path = f"{self.hydra_docs_path}/README.md"
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            
            # Extract comprehensive documentation
            docs = []
            lines = readme_content.split('\n')
            in_code_block = False
            current_section = ""
            current_heading = ""
            
            for line in lines:
                # Track section headings
                if line.startswith('##') or line.startswith('####'):
                    current_heading = line
                
                # Capture code blocks
                if line.startswith('```javascript'):
                    in_code_block = True
                    current_section = current_heading + '\n' + line + '\n'
                elif line.startswith('```') and in_code_block:
                    in_code_block = False
                    current_section += line + '\n'
                    docs.append(current_section.strip())
                    current_section = ""
                elif in_code_block:
                    current_section += line + '\n'
                # Capture important sections
                elif any(keyword in line.lower() for keyword in [
                    'audio', 'fft', 'osc', 'src', 'out', 'render', 'modulate', 'diff', 'blend', 
                    'add', 'mult', 'rotate', 'scale', 'pixelate', 'kaleid', 'repeat', 'color',
                    'brightness', 'contrast', 'hue', 'saturate', 'posterize', 'invert'
                ]):
                    if line.strip() and not line.startswith('#'):
                        docs.append(line)
            
            # Add comprehensive function reference with REAL audio functions from Hydra repo
            function_reference = """
## Hydra Function Reference (from official repo):

### Audio Input (a object) - ALWAYS USE THESE:
- a.fft[0] - bass frequencies (lowest frequency bin)
- a.fft[1] - low-mid frequencies  
- a.fft[2] - mid frequencies
- a.fft[3] - high-mid frequencies
- a.fft[4] - treble frequencies
- a.fft[5] - high frequencies
- a.setBins(6) - set number of frequency bins
- a.setCutoff(4) - set minimum detection level
- a.setScale(2) - set detection range
- a.setSmooth(0.8) - set smoothing (0-1)
- a.show() - show fft bins
- a.hide() - hide audio waveform

### Basic Generators:
- osc(frequency, sync, offset) - oscillator
- noise(scale, offset) - noise texture
- shape(radius, sides, smoothing) - geometric shapes
- voronoi(scale, speed, blending) - voronoi patterns
- gradient() - color gradients

### Sources:
- src(s0) - source buffer
- s0.initVideo(url) - video input
- s0.initImage(url) - image input
- s0.initScreen() - screen capture
- NOTE: Do NOT use s0.initCam() - camera input is disabled

### Transform Functions:
- .rotate(angle) - rotation
- .scale(x, y) - scaling
- .scroll(x, y) - scrolling
- .pixelate(x, y) - pixelation
- .repeat(x, y) - repetition
- .kaleid(sides) - kaleidoscope effect

### Color Functions:
- .color(r, g, b) - color tinting
- .colorama(amount) - color cycling
- .hue(shift) - hue shifting
- .saturate(amount) - saturation
- .contrast(amount) - contrast
- .brightness(amount) - brightness
- .posterize(bins) - posterization
- .invert() - color inversion

### Blend Functions:
- .blend(texture) - normal blend
- .diff(texture) - difference blend
- .add(texture) - additive blend
- .mult(texture) - multiply blend
- .layer(texture) - layer blend
- .mask(texture) - masking

### Modulation:
- .modulate(texture, amount) - coordinate modulation
- .modulateRotate(texture, amount) - rotation modulation
- .modulateScale(texture, amount) - scale modulation
- .modulateScroll(texture, amount) - scroll modulation

### Output:
- .out(o0) - output to buffer o0
- .out(o1) - output to buffer o1
- .out(o2) - output to buffer o2
- .out(o3) - output to buffer o3
- render(o0) - render buffer to screen
- render() - render all buffers

### Advanced Functions:
- .thresh(amount) - threshold
- .blur(amount) - blur
- .sharpen(amount) - sharpening
- .sobel() - edge detection
- .luma() - luminance
- .feedback(amount) - feedback loop
- .time() - time variable
- .mouse.x, .mouse.y - mouse coordinates

### Audio Examples from Hydra docs:
- osc(10, 0, () => (a.fft[0]*4)).out() - bass-reactive oscillator
- osc(() => (a.fft[0]*4), 0, 0.8).out() - frequency controlled by bass
- Use a.fft[0] for bass, a.fft[4] for treble, a.fft[2] for mid frequencies
"""
            
            # Combine documentation
            all_docs = docs + [function_reference]
            self.hydra_docs_cache = "\n".join(all_docs[:15])  # Limit to prevent token overflow
            return self.hydra_docs_cache
        except Exception as e:
            print(f"Error loading Hydra docs: {e}")
            return "Basic Hydra functions: osc(), src(), out(), render(). Use .out(o0) and render(o0) for output."

    def analyze_screenshot(self, frame):
        """Analyze screenshot to provide feedback for next generation"""
        try:
            # Resize to small size for analysis
            small_frame = frame.resize((8, 6), Image.LANCZOS)
            pixels = list(small_frame.getdata())
            
            # Calculate brightness and coverage
            total_brightness = 0
            bright_pixels = 0
            center_pixels = 0
            edge_pixels = 0
            
            width, height = small_frame.size
            center_x, center_y = width // 2, height // 2
            
            for i, pixel in enumerate(pixels):
                x, y = i % width, i // width
                brightness = sum(pixel) / 3  # Average RGB
                total_brightness += brightness
                
                if brightness > 100:  # Bright pixel threshold
                    bright_pixels += 1
                
                # Check if pixel is in center or edge
                distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if distance_from_center <= 2:  # Center area
                    center_pixels += 1
                else:  # Edge area
                    edge_pixels += 1
            
            avg_brightness = total_brightness / len(pixels)
            coverage = bright_pixels / len(pixels)
            center_ratio = center_pixels / (center_pixels + edge_pixels) if (center_pixels + edge_pixels) > 0 else 0
            
            # Generate feedback
            feedback = []
            if avg_brightness > 150:
                feedback.append("The previous visual was too bright (average brightness: {:.1f})".format(avg_brightness))
            if coverage > 0.7:
                feedback.append("The previous visual covered too much of the screen ({:.1%} coverage)".format(coverage))
            if center_ratio < 0.3:
                feedback.append("The previous visual was too edge-focused, need more center activity")
            
            if feedback:
                return "Previous visual feedback: " + "; ".join(feedback) + ". Make the next visual darker, more center-focused, and less full-screen."
            else:
                return "Previous visual was good - maintain similar brightness and center-focus."
                
        except Exception as e:
            print(f"Error analyzing screenshot: {e}")
            return ""

    def capture_current_screenshot(self):
        """Capture current screenshot for analysis"""
        try:
            if hasattr(self, 'driver') and self.driver:
                image = self.driver.get_screenshot_as_base64()
                frame = Image.open(BytesIO(base64.b64decode(image)))
                self.last_screenshot_analysis = self.analyze_screenshot(frame)
                return self.last_screenshot_analysis
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
        return ""

    def regenerate_visualization(self, reason=""):
        try:
            prompts = [
                "Create a sparse, center-focused visual with pulsing patterns and dark edges. MUST use audio input (a.fft[0] for bass, a.fft[4] for treble).",
                "Generate a visual with radial patterns and subtle color modulation; keep it sparse and center-focused. MUST use audio input (a.fft[0] for bass, a.fft[4] for treble).",
                "Create a minimal visual using oscillating shapes with a pulsing center and understated outer effects. MUST use audio input (a.fft[0] for bass, a.fft[4] for treble).",
                "Design a center-focused visual with layered patterns and gentle intensity changes that pulse outward. MUST use audio input (a.fft[0] for bass, a.fft[4] for treble).",
                "Generate concentric, evolving patterns with tasteful color shifts and dark edges, emphasizing center activity. MUST use audio input (a.fft[0] for bass, a.fft[4] for treble)."
            ]
            import random
            
            # Capture current screenshot for feedback
            screenshot_feedback = self.capture_current_screenshot()
            
            prompt = " Follow the Hydra documentation. Use only documented Hydra APIs. Include at least one .out(o0) and render(o0)."
            if screenshot_feedback:
                prompt += "\n\n" + screenshot_feedback
            prompt += "\n\nHydra Documentation:\n" + self.get_hydra_docs()
            if self.verbose:
                print(f"Regenerating visualization ({reason})")
            response = self.ai_helper.generate_visualization(prompt)
            if response:
                content = json.loads(response)
                if 'code' in content:
                    self.current_visualization = content['code']
                    self.update_hydra_code(content['code'])
                    self.last_visualization_time = time.time()
                    return True
        except Exception as e:
            if self.verbose:
                print(f"Error regenerating visualization: {e}")
        return False

    def monitor_display(self):
        """Thread function to monitor display for static frames"""
        while self.monitoring_active:
            # Monitoring disabled - no fallback needed
            time.sleep(0.1)  # Check every 0.1s
        
    def init(self):
        self.mqtt.connect()
        self.leds.init()
        
        # Start audio capture
        if self.audio_enabled:
            self.start_audio_capture()
        else:
            print("Audio capture disabled - using MQTT audio only")
        
        # No AI greeting - using stock visuals

    def generate_visualizations(self):
        """Thread to refresh page every 30 seconds with fade to black"""
        while self.monitoring_active:
            current_time = time.time()
            if current_time - self.last_visualization_time >= self.visualization_interval:
                # Start fade to black
                if not self.is_fading:
                    self.is_fading = True
                    self.fade_start_time = current_time
                    print("Starting fade to black before refresh...")
                
                # Check if fade is complete
                if self.is_fading and current_time - self.fade_start_time >= self.fade_duration:
                    # Fade complete, now navigate to original URL for new visual
                    if hasattr(self, 'driver') and self.driver:
                        try:
                            print("Navigating to original URL for new visual...")
                            self.driver.get("http://localhost:5173")  # Navigate to original URL
                            time.sleep(2)
                            # Hide UI elements
                            self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                            self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                            self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                            print("New visual loaded")
                            
                            # Start fade in
                            self.is_fading_in = True
                            self.fade_in_start_time = current_time
                            self.last_visual_load_time = current_time  # Track when visual loaded
                            print("Starting fade in...")
                        except Exception as e:
                            print(f"Error loading new visual: {e}")
                    
                    # Reset fade out state
                    self.is_fading = False
                    self.fade_start_time = None
                    self.last_visualization_time = current_time
            
            # Check if fade in is complete
            if self.is_fading_in and self.fade_in_start_time and current_time - self.fade_in_start_time >= self.fade_duration:
                self.is_fading_in = False
                self.fade_in_start_time = None
                print("Fade in complete")
            
            time.sleep(0.1)  # Check more frequently for smooth fade
    
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
                
                # Initialize the WebDriver instance
                service = Service('/usr/bin/chromedriver')  # Path to Chromium's driver
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
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

    def start_audio_capture(self):
        """Start audio capture thread"""
        if not self.audio_enabled:
            print("Audio capture disabled")
            return
            
        try:
            # Audio stream configuration (optimized for 120-150 BPM)
            self.CHUNK = 2048  # Larger chunk for better frequency resolution
            self.FORMAT = pyaudio.paInt16
            self.CHANNELS = 1
            self.RATE = 44100
            
            self.audio_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            # Start audio processing thread
            self.audio_thread = Thread(target=self.audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("Audio capture started successfully")
            
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            self.audio_enabled = False
    
    def audio_capture_loop(self):
        """Main audio capture and processing loop"""
        print("Audio capture loop started")
        
        while self.audio_enabled and self.audio_stream:
            try:
                # Read audio data
                data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Process audio for frequency analysis
                self.process_audio_data(audio_array)
                
            except Exception as e:
                print(f"Error in audio capture loop: {e}")
                break
    
    def process_audio_data(self, audio_data):
        """Process audio data using librosa for professional BPM detection"""
        try:
            current_time = time.time()
            
            # Add to audio buffer
            self.audio_buffer.extend(audio_data)
            
            # Keep buffer size manageable
            if len(self.audio_buffer) > self.buffer_size * 2:
                self.audio_buffer = self.audio_buffer[-self.buffer_size:]
            
            # Analyze every analysis_interval seconds
            if current_time - self.last_bpm_update >= self.analysis_interval:
                if len(self.audio_buffer) >= self.buffer_size:
                    # Convert to float32 for librosa
                    audio_float = np.array(self.audio_buffer[-self.buffer_size:], dtype=np.float32)
                    audio_float = audio_float / 32768.0  # Normalize to [-1, 1]
                    
                    # Use librosa for professional BPM detection
                    try:
                        # Get tempo and beats using librosa
                        tempo, beats = librosa.beat.beat_track(
                            y=audio_float, 
                            sr=self.RATE,
                            units='time',
                            hop_length=512,
                            start_bpm=135  # Start with expected BPM
                        )
                        
                        # Extract scalar value from tempo (it might be a numpy array)
                        if hasattr(tempo, '__len__') and len(tempo) > 0:
                            tempo_value = float(tempo[0])  # Extract first element from array
                        else:
                            tempo_value = float(tempo)  # Already a scalar
                        
                        # Validate BPM is in expected range
                        if self.min_bpm <= tempo_value <= self.max_bpm:
                            # Add to BPM history for smoothing
                            self.bpm_history.append(tempo_value)
                            if len(self.bpm_history) > self.bpm_window_size:
                                self.bpm_history.pop(0)
                            
                            # Calculate smoothed BPM
                            smoothed_bpm = np.mean(self.bpm_history)
                            old_bpm = self.bpm
                            self.bpm = smoothed_bpm
                            self.pulse_speed = self.bpm / 60.0
                            
                            # BPM logging removed
                            pass
                        else:
                            pass  # BPM outside range - no logging
                            
                    except Exception as e:
                        print(f"Librosa analysis error: {e}")
                
                # Get frequency bands using librosa
                try:
                    # Convert buffer to librosa format
                    audio_float = np.array(self.audio_buffer[-self.buffer_size:], dtype=np.float32)
                    audio_float = audio_float / 32768.0
                    
                    # Get spectral features
                    spectral_centroids = librosa.feature.spectral_centroid(y=audio_float, sr=self.RATE)[0]
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_float, sr=self.RATE)[0]
                    mfccs = librosa.feature.mfcc(y=audio_float, sr=self.RATE, n_mfcc=8)
                    
                    # Create 8 frequency bands from spectral features
                    bands = []
                    for i in range(8):
                        if i < len(mfccs):
                            bands.append(np.mean(mfccs[i]))
                        else:
                            bands.append(0.0)
                    
                    # Normalize bands
                    max_band = max(bands) if max(bands) > 0 else 1
                    normalized_bands = [max(0, b / max_band) for b in bands]
                    
                    # Update audio levels
                    with self.audio_lock:
                        self.audio_levels = normalized_bands
                        self.last_audio_update = time.time()
                    
                    # Reduced audio logging - only log when bass is very high
                    bass_level = sum(normalized_bands[:2]) / 2
                    if bass_level > 0.5:  # Only log very high bass
                        print(f"High bass: {bass_level:.2f}")
                        
                except Exception as e:
                    print(f"Librosa spectral analysis error: {e}")
                
                self.last_bpm_update = current_time
            
        except Exception as e:
            print(f"Error processing audio data: {e}")
    
    def update_audio_levels(self, frequencies):
        """Update audio frequency levels for visualization (MQTT fallback)"""
        self.audio_levels = frequencies[:8]  # Use first 8 frequency bands
        self.last_audio_update = time.time()
        
        # MQTT fallback - use simple frequency analysis
        bass_level = sum(frequencies[:2]) / 2 if len(frequencies) >= 2 else 0
        
        # Only log when there's significant audio
        if bass_level > 0.1:
            print(f"MQTT audio levels: {frequencies[:8]}")
            print(f"MQTT bass level: {bass_level:.2f}")
            # MQTT audio note removed

    def update_hydra_code(self, code=None):
        """Update the Hydra editor with new code"""
        try:
            if code is None:
                self.url = "http://localhost:5173"
                return
                
            print(f"Updating Hydra code: {code}")

            # Encode code for URL (ensure string, not bytes)
            new_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
            self.url = "http://localhost:5173?code=" + urllib.parse.quote_plus(new_code)
            
            # Reload the page with new code and hide overlays
            if hasattr(self, 'driver') and self.driver:
                try:
                    self.driver.get(self.url)
                    # Wait a moment for the page to load
                    time.sleep(2)
                    # Hide UI elements
                    self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                    self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                    # Auto-run the code if it's not already running
                    try:
                        self.driver.execute_script("""
                            // Check if code is already running, if not, run it
                            if (typeof hydraSynth !== 'undefined' && hydraSynth) {
                                // Try to run the current code
                                try {
                                    eval(hydraSynth.getCode());
                                } catch(e) {
                                    console.log('Auto-run failed:', e);
                                }
                            }
                        """)
                        print("Auto-executed Hydra code")
                    except Exception as e:
                        print(f"Could not auto-run code: {e}")
                    print("Updated Hydra with new code and hidden overlays")
                except Exception as e:
                    print(f"Error updating Hydra page: {e}")
            
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
        if pil is None:
            print("Error: PIL image is None")
            return
            
        im = pil.resize((self.width, self.height), Image.LANCZOS)
        
        # Handle mode transitions and sparse visuals
        current_time = time.time()
        transition_progress = 0.0
        
        # Check if we're in a mode transition
        if self.mode_transition_start:
            transition_progress = (current_time - self.mode_transition_start) / self.mode_transition_duration
            if transition_progress >= 1.0:
                self.mode_transition_start = None
                transition_progress = 1.0
        
        # Check if this is a sparse visual - if so, show full image with fade and opacity pulsing
        if self.is_sparse_visual:
            # Apply fade in effect for sparse visuals
            fade_multiplier = 1.0
            if self.sparse_fade_start_time:
                fade_progress = (current_time - self.sparse_fade_start_time) / 3.0  # 3 second fade
                fade_multiplier = min(1.0, fade_progress)
            
            # Apply BPM-synced opacity pulsing for sparse visuals (reduced intensity but still peaks at 100%)
            pulse_intensity = (math.sin(self.pulse_phase) + 1) / 2  # 0 to 1
            opacity_multiplier = 0.7 + pulse_intensity * 0.3  # Pulse between 70% and 100% opacity
            
            # Apply strobe effect if enabled
            strobe_multiplier = 1.0
            if self.strobe_mode:
                strobe_intensity = (math.sin(self.strobe_phase) + 1) / 2  # 0 to 1
                strobe_multiplier = 0.1 + strobe_intensity * 0.9  # Strobe between 10% and 100%
            
            # Combine fade, opacity, and strobe effects
            final_multiplier = fade_multiplier * opacity_multiplier * strobe_multiplier
            
            # Display the full image with opacity pulsing
            for i in range(self.width):
                for j in range(self.height):
                    pix = im.getpixel((i, self.height - j - 1))
                    r = int(pix[0] * final_multiplier)
                    g = int(pix[1] * final_multiplier)
                    b = int(pix[2] * final_multiplier)
                    self.leds.set_pixel_color(i, j, r, g, b)
            return
        
        # Apply shape post-processing for non-sparse visuals
        processed_im = self.apply_shape_mask(im)
        
        # Check if shape mask processing failed
        if processed_im is None:
            print("Error: Shape mask processing failed, using original image")
            processed_im = im
        
        # If transitioning from sparse to dense, blend between full image and processed
        if self.mode_transition_start and not self.is_sparse_visual:
            # Blend between original image and processed image during transition
            blend_factor = transition_progress  # 0 = full image, 1 = processed image
            
            for i in range(self.width):
                for j in range(self.height):
                    # Get original pixel
                    orig_pix = im.getpixel((i, self.height - j - 1))
                    # Get processed pixel
                    proc_pix = processed_im.getpixel((i, self.height - j - 1))
                    
                    # Blend between them
                    r = int(orig_pix[0] * (1 - blend_factor) + proc_pix[0] * blend_factor)
                    g = int(orig_pix[1] * (1 - blend_factor) + proc_pix[1] * blend_factor)
                    b = int(orig_pix[2] * (1 - blend_factor) + proc_pix[2] * blend_factor)
                    
                    self.leds.set_pixel_color(i, j, r, g, b)
        else:
            # Normal processing for dense visuals
            for i in range(self.width):
                for j in range(self.height):
                    pix = processed_im.getpixel((i, self.height - j - 1))
                    self.leds.set_pixel_color(i, j, pix[0], pix[1], pix[2])
    
    def apply_shape_mask(self, im):
        """Apply different shape masks that switch and animate"""
        import math
        import random
        
        # Check if image is valid
        if im is None:
            print("Error: Image is None in apply_shape_mask")
            return None
            
        # Update shape animation and switching
        self.update_shape_animation()
        
        # Create a copy to work with
        result = im.copy()
        pixels = result.load()
        
        # Debug: Print shape info occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        # Reduced debug logging - only log every 500 frames
        if self._debug_counter % 500 == 0:  # Every 500 frames
            print(f"Shape: {self.current_shape}, Size: {self.shape_size:.1f}")
        
        # Add BPM indicator pixel in top left
        self.update_bpm_indicator()
        
        # Apply fade effects
        fade_multiplier = 1.0
        current_time = time.time()
        
        if self.is_fading and self.fade_start_time:
            # Fade out to black
            fade_progress = (current_time - self.fade_start_time) / self.fade_duration
            fade_multiplier = max(0.0, 1.0 - fade_progress)  # Fade from 1.0 to 0.0
        elif self.is_fading_in and self.fade_in_start_time:
            # Fade in from black
            fade_progress = (current_time - self.fade_in_start_time) / self.fade_duration
            fade_multiplier = min(1.0, fade_progress)  # Fade from 0.0 to 1.0
        
        # Apply multiple shape masks with BPM-synced effects
        for x in range(self.width):
            for y in range(self.height):
                max_strength = 0.0
                combined_brightness = 1.0
                
                # BPM indicator pixel disabled
                # if x == 0 and y == 0 and self.bpm_indicator_on:
                #     pixels[x, y] = (255, 255, 0)  # Bright yellow
                #     continue
                
                # Check each shape
                for i in range(self.num_shapes):
                    pos = self.shape_positions[i]
                    size = self.shape_sizes[i]
                    
                    # Apply BPM movement offset if in movement mode
                    if self.bpm_effect_type == 2 and hasattr(self, 'shape_movement_offset'):
                        pos = [pos[0] + self.shape_movement_offset, pos[1] + self.shape_movement_offset]
                    
                    # Get shape strength with smooth transition between shapes
                    if self.current_shape != self.target_shape:
                        # Blend between current and target shapes
                        current_strength = self.get_shape_strength(x, y, self.current_shape, pos, size)
                        target_strength = self.get_shape_strength(x, y, self.target_shape, pos, size)
                        shape_strength = current_strength * (1 - self.shape_transition_progress) + target_strength * self.shape_transition_progress
                    else:
                        # No transition, use current shape
                        shape_strength = self.get_shape_strength(x, y, self.current_shape, pos, size)
                    
                    if shape_strength > 0:
                        # BPM-synced brightness pulsing
                        pulse_intensity = (math.sin(self.pulse_phase + i * 0.5) + 1) / 2
                        
                        # Apply BPM effect type (reduced intensity but still peaks at 100%)
                        if self.bpm_effect_type == 1:  # Opacity effect
                            brightness_mult = 0.7 + pulse_intensity * 0.3  # Pulse between 70% and 100%
                        else:  # Other effects use normal brightness
                            brightness_mult = 0.8 + pulse_intensity * 0.2  # Pulse between 80% and 100%
                        
                        # Apply strobe effect if enabled
                        if self.strobe_mode:
                            strobe_intensity = (math.sin(self.strobe_phase + i * 0.3) + 1) / 2
                            strobe_mult = 0.1 + strobe_intensity * 0.9  # Strobe between 10% and 100%
                            brightness_mult *= strobe_mult
                        
                        max_strength = max(max_strength, shape_strength)
                        combined_brightness *= brightness_mult
                
                # Apply combined effect
                original_pixel = pixels[x, y]
                
                if max_strength > 0:
                    # Inside at least one shape - apply BPM-synced effect (brightness only, no color change)
                    final_strength = max_strength * combined_brightness * fade_multiplier
                    # Keep original colors, only adjust brightness
                    r = int(original_pixel[0] * final_strength)
                    g = int(original_pixel[1] * final_strength)
                    b = int(original_pixel[2] * final_strength)
                    pixels[x, y] = (r, g, b)
                else:
                    # Outside all shapes - completely black
                    pixels[x, y] = (0, 0, 0)
        
        return result
    
    def load_overlay_image(self):
        """Load the custom overlay image"""
        try:
            if self.overlay_image is None:
                self.overlay_image = Image.open("tush.png")
                print(f"Loaded overlay image: tush.png (size: {self.overlay_image.size})")
            else:
                print("Overlay image already loaded")
        except Exception as e:
            print(f"Could not load overlay image: {e}")
            self.overlay_image = None
    
    def detect_sparse_visual(self, im):
        """Detect if the visual is sparse (less than 40% lit up)"""
        if im is None:
            return False
            
        # Resize to small size for analysis
        small_im = im.resize((8, 6), Image.LANCZOS)
        pixels = list(small_im.getdata())
        
        # Count lit pixels
        lit_pixels = 0
        for pixel in pixels:
            brightness = sum(pixel) / 3  # Average RGB
            if brightness > 30:  # Lit pixel threshold
                lit_pixels += 1
        
        # Calculate lit coverage
        lit_coverage = lit_pixels / len(pixels)
        is_sparse = lit_coverage < self.sparse_threshold
        
        if is_sparse:
            print(f"Sparse visual detected (lit coverage: {lit_coverage:.2f}) - showing full visual")
        
        return is_sparse
    
    def update_bpm_indicator(self):
        """Update BPM indicator pixel based on beat detection"""
        current_time = time.time()
        beat_interval = 60.0 / self.bpm if self.bpm > 0 else 0.5  # Time between beats
        
        # Check if it's time for a new beat
        if current_time - self.last_beat_time >= beat_interval:
            self.bpm_indicator_on = True
            self.last_beat_time = current_time
        else:
            # Turn off indicator after a short duration
            if current_time - self.last_beat_time >= 0.1:  # 100ms flash
                self.bpm_indicator_on = False
    
    def get_shape_strength(self, x, y, shape_type, pos, size):
        """Get the strength (0.0 to 1.0) for a pixel in a given shape with edge gradient"""
        import math
        
        if shape_type == 0:  # Circle
            center_x = int(pos[0])
            center_y = int(pos[1])
            radius = int(size)
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            if distance <= radius:
                # Inside circle - gradient from center to edge
                normalized_dist = distance / radius if radius > 0 else 0
                return 1.0 - (normalized_dist * 0.7)  # 0.3 to 1.0
            elif distance <= radius + 3:  # Edge gradient
                # Edge falloff
                edge_dist = distance - radius
                edge_strength = 1.0 - (edge_dist / 3.0)
                return max(0, edge_strength * 0.3)  # Fade to 0 over 3 pixels
            else:
                return 0.0
                
        elif shape_type == 1:  # Square
            center_x = int(pos[0])
            center_y = int(pos[1])
            half_size = int(size)
            
            # Distance from square edge
            dist_x = abs(x - center_x)
            dist_y = abs(y - center_y)
            edge_dist = max(dist_x, dist_y)
            
            if edge_dist <= half_size:
                # Inside square - gradient from center
                normalized_dist = edge_dist / half_size if half_size > 0 else 0
                return 1.0 - (normalized_dist * 0.7)
            elif edge_dist <= half_size + 3:  # Edge gradient
                edge_falloff = edge_dist - half_size
                edge_strength = 1.0 - (edge_falloff / 3.0)
                return max(0, edge_strength * 0.3)
            else:
                return 0.0
                
        elif shape_type == 2:  # Triangle
            center_x = int(pos[0])
            center_y = int(pos[1])
            triangle_size = int(size)
            
            dx = x - center_x
            dy = y - center_y
            
            # Triangle distance calculation
            if abs(dx) <= triangle_size and dy >= -triangle_size and dy <= triangle_size:
                triangle_dist = abs(dx) - (triangle_size - abs(dy))
                if triangle_dist <= 0:  # Inside triangle
                    # Distance from center
                    distance = math.sqrt(dx*dx + dy*dy)
                    normalized_dist = distance / triangle_size if triangle_size > 0 else 0
                    return 1.0 - (normalized_dist * 0.7)
                elif triangle_dist <= 3:  # Edge gradient
                    edge_strength = 1.0 - (triangle_dist / 3.0)
                    return max(0, edge_strength * 0.3)
            
            return 0.0
            
        elif shape_type == 3:  # Rectangle (scrolling)
            rect_width = 3  # Much thinner rectangle
            rect_x = int(self.rect_scroll_x)
            
            if x >= rect_x and x < rect_x + rect_width and y >= 0 and y < self.height:
                # Inside rectangle - gradient from left to right
                local_x = x - rect_x
                return 1.0 - (local_x / rect_width * 0.5)
            elif x >= rect_x - 3 and x < rect_x + rect_width + 3 and y >= 0 and y < self.height:
                # Edge gradient for rectangle
                if x < rect_x:  # Left edge
                    edge_dist = rect_x - x
                    edge_strength = 1.0 - (edge_dist / 3.0)
                else:  # Right edge
                    edge_dist = x - (rect_x + rect_width)
                    edge_strength = 1.0 - (edge_dist / 3.0)
                return max(0, edge_strength * 0.3)
            else:
                return 0.0
                
        elif shape_type == 4:  # Stars - multiple small circles
            # Create multiple small circles scattered around
            star_positions = [
                [pos[0] - size, pos[1] - size],
                [pos[0] + size, pos[1] - size], 
                [pos[0] - size, pos[1] + size],
                [pos[0] + size, pos[1] + size],
                [pos[0], pos[1] - size * 1.5],
                [pos[0], pos[1] + size * 1.5],
                [pos[0] - size * 1.5, pos[1]],
                [pos[0] + size * 1.5, pos[1]]
            ]
            
            max_star_strength = 0.0
            for star_pos in star_positions:
                star_dx = x - star_pos[0]
                star_dy = y - star_pos[1]
                star_distance = math.sqrt(star_dx*star_dx + star_dy*star_dy)
                star_radius = size * 0.3  # Small stars
                
                if star_distance <= star_radius:
                    star_strength = 1.0 - (star_distance / star_radius * 0.5)
                    max_star_strength = max(max_star_strength, star_strength)
            
            return max_star_strength
            
        elif shape_type == 5:  # Heart shape
            center_x = int(pos[0])
            center_y = int(pos[1])
            heart_size = int(size)
            
            # Heart equation: (x^2 + y^2 - 1)^3 - x^2*y^3 = 0
            dx = (x - center_x) / heart_size
            dy = (y - center_y) / heart_size
            
            # Heart shape approximation
            heart_eq = (dx*dx + dy*dy - 1)**3 - dx*dx * dy*dy*dy
            if heart_eq <= 0.1:  # Inside heart
                distance = math.sqrt(dx*dx + dy*dy)
                return 1.0 - (distance * 0.3)
            elif heart_eq <= 0.3:  # Edge gradient
                edge_strength = 1.0 - ((heart_eq - 0.1) / 0.2)
                return max(0, edge_strength * 0.4)
            else:
                return 0.0
                
        elif shape_type == 6:  # Spiral
            center_x = int(pos[0])
            center_y = int(pos[1])
            spiral_size = int(size)
            
            dx = x - center_x
            dy = y - center_y
            distance = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)
            
            # Spiral equation: r = a * θ
            spiral_radius = abs(angle) * spiral_size * 0.3
            spiral_width = spiral_size * 0.2
            
            if abs(distance - spiral_radius) <= spiral_width:
                spiral_strength = 1.0 - (abs(distance - spiral_radius) / spiral_width * 0.5)
                return max(0, spiral_strength)
            else:
                return 0.0
                
        elif shape_type == 7:  # Wave
            center_x = int(pos[0])
            center_y = int(pos[1])
            wave_size = int(size)
            
            # Wave pattern: y = sin(x) with some amplitude
            wave_x = (x - center_x) / wave_size * 4  # Scale for wave frequency
            wave_y = math.sin(wave_x) * wave_size * 0.3
            wave_center_y = center_y + wave_y
            
            # Check if pixel is near the wave
            wave_distance = abs(y - wave_center_y)
            wave_width = wave_size * 0.15
            
            if wave_distance <= wave_width:
                wave_strength = 1.0 - (wave_distance / wave_width * 0.5)
                return max(0, wave_strength)
            else:
                return 0.0
                
        elif shape_type == 8:  # Overlay image - static and large
            if self.overlay_image is None:
                return 0.0
                
            # Static position - center of screen
            center_x = 20  # Center of 40-pixel wide screen
            center_y = 15  # Center of 30-pixel tall screen
            
            # Calculate position relative to center
            rel_x = x - center_x
            rel_y = y - center_y
            
            # Scale to overlay image coordinates - maintain aspect ratio
            img_width, img_height = self.overlay_image.size
            # Scale to 60% of screen size while maintaining aspect ratio
            screen_width = 40
            screen_height = 30
            max_width = screen_width * 0.6  # 60% of screen width
            max_height = screen_height * 0.6  # 60% of screen height
            
            # Calculate scale factors for both dimensions
            scale_x = img_width / max_width
            scale_y = img_height / max_height
            
            # Use the smaller scale to fit within bounds while maintaining aspect ratio
            scale = min(scale_x, scale_y)
            
            # Calculate final dimensions
            final_width = img_width / scale
            final_height = img_height / scale
            
            # Convert to image coordinates using the aspect-ratio-preserving scale
            img_x = int(rel_x * scale + img_width // 2)
            img_y = int(rel_y * scale + img_height // 2)
            
            # Check if within image bounds
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                # Get pixel from overlay image
                pixel = self.overlay_image.getpixel((img_x, img_y))
                if len(pixel) >= 3:  # RGB or RGBA
                    # Use alpha channel if available, otherwise use brightness
                    if len(pixel) == 4:  # RGBA
                        alpha = pixel[3] / 255.0
                    else:  # RGB
                        brightness = sum(pixel[:3]) / (3 * 255.0)
                        alpha = brightness
                    
                    # Return alpha as strength (0.0 to 1.0)
                    return alpha
                else:
                    return 0.0
            else:
                return 0.0
        
        return 0.0
    
    def update_shape_animation(self):
        """Update multiple shapes with BPM matching, home position, and various paths"""
        import math
        import random
        
        current_time = time.time()
        
        # Update shape switching
        if current_time - self.shape_switch_time >= self.shape_switch_interval:
            # Check if we should use overlay
            if self.use_overlay:
                self.target_shape = 8  # Overlay shape
                self.load_overlay_image()  # Load the image
            else:
                self.target_shape = (self.current_shape + 1) % 8  # Cycle through 8 shapes
            self.shape_switch_time = current_time
            shape_names = ["circle", "square", "triangle", "rectangle", "stars", "heart", "spiral", "wave", "overlay"]
            print(f"Switching to shape: {shape_names[self.target_shape]}")
        
        # Force overlay shape if overlay mode is enabled but not currently using overlay
        if self.use_overlay and self.current_shape != 8:
            self.target_shape = 8  # Overlay shape
            self.load_overlay_image()  # Load the image
            print("Forcing overlay shape")
        
        # Update shape transition with smooth blending
        if self.current_shape != self.target_shape:
            self.shape_transition_progress += self.shape_transition_speed
            if self.shape_transition_progress >= 1.0:
                self.current_shape = self.target_shape
                self.shape_transition_progress = 0.0
        
        # Update BPM effect variety
        if current_time - self.bpm_effect_switch_time >= self.bpm_effect_switch_interval:
            self.bpm_effect_type = (self.bpm_effect_type + 1) % 4  # Cycle through effects
            self.bpm_effect_switch_time = current_time
            effect_names = ["size", "opacity", "movement", "rotation"]
            print(f"BPM effect: {effect_names[self.bpm_effect_type]}")
        
        # Debug fast tempo mode override
        if self.debug_fast_tempo:
            self.tempo_mode = 1  # Force double speed
            if not self.tempo_start_time:
                self.tempo_start_time = current_time
                print("DEBUG: Fast tempo mode FORCED ON")
        
        # Update tempo responsiveness with duration limits (only if not in debug mode)
        if not self.debug_fast_tempo and current_time - self.tempo_switch_time >= self.tempo_switch_interval:
            self.tempo_mode = (self.tempo_mode + 1) % 2  # Toggle between normal and double speed
            self.tempo_switch_time = current_time
            self.tempo_start_time = current_time if self.tempo_mode == 1 else None
            tempo_names = ["normal", "double speed"]
            print(f"Tempo mode: {tempo_names[self.tempo_mode]}")
        
        # Check if fast tempo has exceeded max duration (only if not in debug mode)
        if (not self.debug_fast_tempo and self.tempo_mode == 1 and self.tempo_start_time and 
            current_time - self.tempo_start_time >= self.tempo_max_duration):
            self.tempo_mode = 0  # Switch back to normal
            self.tempo_start_time = None
            print("Fast tempo duration exceeded - switching to normal")
        
        # Debug strobe mode override
        if self.debug_strobe:
            self.strobe_mode = True
            if not self.strobe_start_time:
                self.strobe_start_time = current_time
                print("DEBUG: Strobe mode FORCED ON")
        
        # Update strobe mode with duration limits (only if not in debug mode)
        if not self.debug_strobe and current_time - self.strobe_switch_time >= self.strobe_switch_interval:
            self.strobe_mode = not self.strobe_mode
            self.strobe_switch_time = current_time
            self.strobe_start_time = current_time if self.strobe_mode else None
            print(f"Strobe mode: {'ON' if self.strobe_mode else 'OFF'}")
        
        # Check if strobe has exceeded max duration (only if not in debug mode)
        if (not self.debug_strobe and self.strobe_mode and self.strobe_start_time and 
            current_time - self.strobe_start_time >= self.strobe_max_duration):
            self.strobe_mode = False
            self.strobe_start_time = None
            print("Strobe duration exceeded - switching OFF")
        
        # Debug overlay mode (force overlay if debug flag is set)
        if self.debug_overlay:
            if not self.use_overlay:
                self.use_overlay = True
                self.load_overlay_image()
                print("DEBUG: Overlay mode FORCED ON")
        
        # Update overlay mode with duration limits
        if not self.debug_overlay and current_time - self.overlay_switch_time >= self.overlay_switch_interval:
            self.use_overlay = not self.use_overlay
            self.overlay_switch_time = current_time
            self.overlay_start_time = current_time if self.use_overlay else None
            print(f"Overlay mode: {'ON' if self.use_overlay else 'OFF'}")
        
        # Check if overlay has exceeded max duration
        if (not self.debug_overlay and self.use_overlay and self.overlay_start_time and 
            current_time - self.overlay_start_time >= self.overlay_max_duration):
            self.use_overlay = False
            self.overlay_start_time = None
            print("Overlay duration exceeded - switching OFF")
        
        # Update BPM pulse phase - properly synced to BPM with tempo modes
        # Calculate phase increment based on BPM (beats per minute to radians per frame)
        tempo_multiplier = 2.0 if self.tempo_mode == 1 else 1.0  # Double speed for tempo mode 1
        bpm_radians_per_second = (self.bpm / 60.0) * 2 * math.pi * tempo_multiplier
        frame_rate = 20  # Approximate frame rate
        phase_increment = bpm_radians_per_second / frame_rate
        self.pulse_phase += phase_increment
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase -= 2 * math.pi
        
        # Update strobe phase
        if self.strobe_mode:
            strobe_frequency = self.bpm / 60.0 * 4  # 4x BPM for strobe effect
            self.strobe_phase += strobe_frequency / frame_rate
            if self.strobe_phase > 2 * math.pi:
                self.strobe_phase -= 2 * math.pi
        
        # Update rectangle scrolling - make it BPM-dependent
        bpm_speed_multiplier = self.bpm / 120.0  # Faster BPM = faster scrolling
        self.rect_scroll_x += self.rect_scroll_speed * bpm_speed_multiplier * self.rect_scroll_direction
        if self.rect_scroll_x >= self.width:
            self.rect_scroll_x = -3  # Start from left edge
        elif self.rect_scroll_x < -3:
            self.rect_scroll_x = self.width  # Start from right edge
        
        # Home position logic - make it BPM-dependent (faster BPM = more frequent returns)
        bpm_home_multiplier = max(0.5, min(2.0, self.bpm / 120.0))  # 0.5x to 2x speed
        dynamic_home_interval = self.return_to_home_interval / bpm_home_multiplier
        
        if current_time - self.return_to_home_time >= dynamic_home_interval:
            if not self.at_home:
                # Return to home
                self.at_home = True
                self.return_to_home_time = current_time
                print("Returning to home position")
            elif current_time - self.return_to_home_time >= self.home_duration:
                # Leave home
                self.at_home = False
                self.return_to_home_time = current_time
                # Change paths for all shapes
                for i in range(self.num_shapes):
                    self.shape_paths[i] = random.randint(0, 3)
                print("Leaving home - changing paths")
        
        # Update each shape
        for i in range(self.num_shapes):
            # BPM-synced pulsing with reduced sensitivity
            bass_level = sum(self.audio_levels[:2]) / 2 if len(self.audio_levels) >= 2 else 0
            treble_level = sum(self.audio_levels[-2:]) / 2 if len(self.audio_levels) >= 2 else 0
            
            # Apply minimum threshold for audio reactivity
            if bass_level < 0.1:
                bass_level = 0
            if treble_level < 0.1:
                treble_level = 0
            
            # Apply BPM effects based on current effect type
            pulse_intensity = (math.sin(self.pulse_phase) + 1) / 2  # 0 to 1
            bpm_multiplier = self.bpm / 120.0  # Higher BPM = stronger effects
            
            if self.bpm_effect_type == 0:  # Size pulsing
                base_size = 3 + bass_level * 4
                pulse_amplitude = 3 * bpm_multiplier
                self.shape_sizes[i] = base_size + pulse_intensity * pulse_amplitude
            elif self.bpm_effect_type == 1:  # Opacity pulsing (affects brightness)
                base_size = 4 + bass_level * 2
                self.shape_sizes[i] = base_size
                # Store opacity for later use in rendering
                self.shape_opacity = 0.3 + pulse_intensity * 0.7 * bpm_multiplier
            elif self.bpm_effect_type == 2:  # Movement pulsing
                base_size = 4 + bass_level * 2
                self.shape_sizes[i] = base_size
                # Add movement offset based on BPM
                movement_amplitude = 2 * bpm_multiplier
                self.shape_movement_offset = pulse_intensity * movement_amplitude
            else:  # Rotation pulsing
                base_size = 4 + bass_level * 2
                self.shape_sizes[i] = base_size
                # Add rotation based on BPM
                rotation_amplitude = math.pi * bpm_multiplier
                self.shape_rotation = pulse_intensity * rotation_amplitude
            
            # Log shape info every 50 frames
            if hasattr(self, '_shape_debug_counter'):
                self._shape_debug_counter += 1
            else:
                self._shape_debug_counter = 0
                
            # Shape debug logging removed
            
            if self.at_home:
                # Move towards home
                self.shape_positions[i][0] += (self.home_x - self.shape_positions[i][0]) * 0.1
                self.shape_positions[i][1] += (self.home_y - self.shape_positions[i][1]) * 0.1
            else:
                # Follow assigned path
                self.follow_path(i, current_time, treble_level)
    
    def follow_path(self, shape_index, current_time, treble_level):
        """Make a shape follow a specific path"""
        import math
        import random
        
        pos = self.shape_positions[shape_index]
        path_type = self.shape_paths[shape_index]
        # Make speed BPM-dependent - faster BPM = faster movement
        bpm_speed_multiplier = self.bpm / 120.0
        speed = (0.05 + treble_level * 0.1) * bpm_speed_multiplier
        
        if path_type == 0:  # Random movement
            if not hasattr(self, f'_target_{shape_index}'):
                setattr(self, f'_target_{shape_index}', [random.uniform(-10, self.width + 10), 
                                                       random.uniform(-10, self.height + 10)])
            
            target = getattr(self, f'_target_{shape_index}')
            pos[0] += (target[0] - pos[0]) * speed
            pos[1] += (target[1] - pos[1]) * speed
            
            # Change target when close
            if abs(pos[0] - target[0]) < 2 and abs(pos[1] - target[1]) < 2:
                setattr(self, f'_target_{shape_index}', [random.uniform(-10, self.width + 10), 
                                                       random.uniform(-10, self.height + 10)])
        
        elif path_type == 1:  # Spiral
            if not hasattr(self, f'_spiral_phase_{shape_index}'):
                setattr(self, f'_spiral_phase_{shape_index}', 0.0)
            
            phase = getattr(self, f'_spiral_phase_{shape_index}')
            phase += speed * 0.5 * bpm_speed_multiplier  # BPM affects spiral speed
            setattr(self, f'_spiral_phase_{shape_index}', phase)
            
            radius = 5 + phase * 0.1
            pos[0] = self.home_x + radius * math.cos(phase)
            pos[1] = self.home_y + radius * math.sin(phase)
        
        elif path_type == 2:  # Figure 8
            if not hasattr(self, f'_figure8_phase_{shape_index}'):
                setattr(self, f'_figure8_phase_{shape_index}', 0.0)
            
            phase = getattr(self, f'_figure8_phase_{shape_index}')
            phase += speed * 0.3 * bpm_speed_multiplier  # BPM affects figure-8 speed
            setattr(self, f'_figure8_phase_{shape_index}', phase)
            
            pos[0] = self.home_x + 8 * math.sin(phase)
            pos[1] = self.home_y + 4 * math.sin(2 * phase)
        
        elif path_type == 3:  # Orbit
            if not hasattr(self, f'_orbit_phase_{shape_index}'):
                setattr(self, f'_orbit_phase_{shape_index}', 0.0)
            
            phase = getattr(self, f'_orbit_phase_{shape_index}')
            phase += speed * 0.4 * bpm_speed_multiplier  # BPM affects orbit speed
            setattr(self, f'_orbit_phase_{shape_index}', phase)
            
            radius = 6 + shape_index * 2  # Different orbits for different shapes
            pos[0] = self.home_x + radius * math.cos(phase)
            pos[1] = self.home_y + radius * math.sin(phase)
    
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
            # No initial loading needed - just use whatever is on the page
            
            # Try to get screenshot from Hydra if available
            if hasattr(self, 'driver') and self.driver:
                try:
                    # Non-blocking FPS cap for screenshots
                    now = time.time()
                    if now - self.last_screenshot_time < self.screenshot_interval:
                        return
                    self.last_screenshot_time = now
                    image = self.driver.get_screenshot_as_base64()
                    frame = Image.open(BytesIO(base64.b64decode(image)))
                    
                    # Check if frame is valid
                    if frame is None:
                        print("Error: Screenshot frame is None")
                        return
                    
                    # Check if this is a sparse visual (lots of black space) - but only after delay
                    current_time = time.time()
                    if (self.last_visual_load_time and 
                        current_time - self.last_visual_load_time >= self.sparse_detection_delay):
                        
                        is_sparse = self.detect_sparse_visual(frame)
                        
                        # Check if mode needs to change
                        if is_sparse != self.is_sparse_visual:
                            print(f"Mode change: {'sparse' if is_sparse else 'dense'} visual detected")
                            self.mode_transition_start = current_time
                            self.is_sparse_visual = is_sparse
                            if is_sparse:
                                self.sparse_fade_start_time = current_time
                    
                    # Display the frame directly
                    self.last_frame = frame
                    # Render the already-decoded PIL frame directly
                    self.pil_display(frame)
                    if self.verbose and not self._hydra_display_announced:
                        print("Displaying Hydra visualization")
                        self._hydra_display_announced = True
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"Error getting Hydra screenshot: {e}")
                    return
            else:
                if self.verbose:
                    print("No driver available for screenshots")
                return
            
        except Exception as e:
            print(f"Error in website display: {e}")


    def check_interjections(self):
        """Thread removed for music visualizer"""
        pass

    def cleanup(self):
        """Stop monitoring thread and cleanup"""
        self.monitoring_active = False
        
        # Stop audio capture
        if self.audio_enabled:
            self.audio_enabled = False
            if self.audio_stream:
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

        leds = simulation.Leds(40, 30) if simulate else ws2812.Leds(40, 30, 0.65)

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
