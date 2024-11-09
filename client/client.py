from ast import arg
import json
from io import BytesIO
import base64
from PIL import Image, ImageDraw, ImageFont
import argparse
import requests
# from pyppeteer import launch
from picamera2 import Picamera2
import numpy as np
from threading import Thread
import threading
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Configure ChromeOptions
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")  # No sandbox for Pi
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

# Initialize the WebDriver instance
service = Service('/usr/bin/chromedriver')  # Path to Chromium's driver

import ws2812
import simulation
import mqtt

lock = threading.RLock()

class Client:
    def __init__(self, leds, server=False):
        self.width = 40
        self.height = 30
        self.server = server
        if self.server:
            self.mqtt = mqtt.Mqtt(self.on_message)
        self.leds = leds

    def init(self):
        if self.server:
            self.mqtt.connect()
        self.leds.init()

    def load_camera(self):
       self.camera = Picamera2()
       preview_config = self.camera.create_preview_configuration(main={"size": (120, 80)}, lores={"size": (120, 80)}, display="lores")
       self.camera.configure(preview_config)
    
    def load_website(self, url):
        
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.set_window_size(240, 160)
        self.driver.get(url)

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
            self.frequency_display(decoded)
        if decoded['type'] == "image":
            self.rgb_display(decoded)
        if decoded['type'] == "rgb":
            self.image_display(decoded)
        if decoded['type'] == "blackout":
            self.leds.blackout()

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
        im = Image.open(BytesIO(base64.b64decode(img)))
        self.pil_display(im)

    def pil_display(self, pil):
        im = pil.resize((self.width, self.height), Image.NEAREST)
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

        # with lock:
        #     await self.page.screenshot(path='web.png')
        #     time.sleep(1)
        #     self.pil_display(Image.open("web.png"))

def start(args, client):
    while True:
        
        if(args.mode and "camera" in args.mode):
            client.camera_display()
        elif(args.mode and "website" in args.mode):
            client.website_display()
        elif(args.mode and "dashboard" in args.mode):
            client.dashboard_display()


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

        if(args.mode and "website" in args.mode):
            if(args.website):
                client.load_website(args.website[0])
        elif(args.mode and "camera" in args.mode):
            client.load_camera()
            client.camera.start()


        client.leds.start()
        
        thread = Thread(target=start, args=(args, client))
        thread.start()

        client.init()
        

    except KeyboardInterrupt:
        print("Exiting LED client")
        if(args.mode == "camera"):
            client.camera.stop()
        client.leds.blackout()
        client.leds.show()