from ast import arg
import json
from io import BytesIO
import base64
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import argparse
import requests
import socket
import io
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
        self.queued_url = None

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
        #threading.Thread(target=self.start_camera_stream).start()

        
        #data = io.BytesIO()
        #self.camera.capture_file(data, format='png')
        #image = base64.b64encode(data.getvalue()).decode()
        
        #print(image)
        
        #url_parts = url.split("=")
        #code = url_parts[1] + '=='
        #print(code)
        #code_decoded = base64.b64decode(code)
        #code.replace("{{image}}", "data:image/png;base64," + image)
        #new_code = base64.b64encode(code.encode('utf-8'))
        #url = url_parts[0] + "=" + str(new_code)
    
        if url != None:
            self.url = url

        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.set_window_size(240, 160)
        self.driver.get(self.url)

        self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
        self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
        
        # self.browser = await launch(headless=True, executablePath='/usr/bin/chromium', options={'args': ['--no-sandbox','--use-fake-ui-for-media-stream', '--allow-file-access-from-files']})
        # self.page = await self.browser.newPage()
        
        # await self.page.setViewport({"width":1200, "height":800})
        # await self.page.goto(url)

        # await self.page.addStyleTag({'content': '#modal{opacity: 0} #editor-container{opacity: 0}'})
    
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
            content = json.loads(decoded['content'])
            if content['display']:
                # Show quip first
                if 'quip' in content:
                    self.text_scroller.start_scroll(content['quip'])
                    self.display_mode = 'scroll'
                
                # Queue the URL for after scrolling
                new_code = base64.b64encode(content['code'].encode('utf-8'))
                self.queued_url = "https://hydra.ojack.xyz?code=" + new_code.decode('utf-8')
                print(self.queued_url)
                self.load_website()

    def update_display(self):
        """Main display update method"""
        if self.display_mode == 'scroll' and self.text_scroller.is_scrolling:
            frame = self.text_scroller.get_frame()
            if frame:
                self.pil_display(frame)
            else:
                # Scrolling finished, switch to queued URL if exists
                if self.queued_url:
                    self.load_website(self.queued_url)
                    self.queued_url = None
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
        image = self.driver.get_screenshot_as_base64()
        self.bytes_display(image)
        #pass
        # with lock:
        #     await self.page.screenshot(path='web.png')
        #     time.sleep(1)
        #     self.pil_display(Image.open("web.png"))

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