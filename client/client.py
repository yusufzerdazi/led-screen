from ast import arg
import json
from io import BytesIO
import base64
from PIL import Image
import argparse
import asyncio
from pyppeteer import launch
from picamera2 import Picamera2
import numpy as np

import ws2812
import mqtt

class Client:
    def __init__(self):
        self.width = 40
        self.height = 30
        self.mqtt = mqtt.Mqtt(self.on_message)
        self.leds = ws2812.Leds(40, 30)

    def init(self):
        self.mqtt.connect()
        self.leds.init()

    def load_camera(self):
        self.camera = Picamera2()
        preview_config = self.camera.create_preview_configuration(main={"size": (120, 80)}, lores={"size": (120, 80)}, display="lores")
        self.camera.configure(preview_config)
    
    async def load_website(self, url):
        self.browser = await launch(headless=True, executablePath='/usr/bin/chromium', options={'args': ['--no-sandbox']})
        self.page = await self.browser.newPage()
        await self.page.setViewport({"width":240, "height":160})
        await self.page.goto(url)
        await self.page.waitForSelector('.ytp-large-play-button')
        await self.page.click('.ytp-large-play-button');
    
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

    def pil_display(self, pil):
        im = pil.resize((self.width, self.height), Image.ANTIALIAS)
        for i in range(self.width):
            for j in range(self.height):
                pix = im.getpixel((i, self.height - j - 1))
                self.leds.set_pixel_color(i, j, pix[0], pix[1], pix[2])
    
    def camera_display(self):
        image = self.camera.capture_file("cam.png")
        self.pil_display(Image.open("cam.png"))
        #self.image_display({"pixels": [((x, y), (image[x][y][0], image[x][y][1], image[x][y][2])) for x in range(len(image)) for y in range(len(image[0]))]})
      
    async def website_display(self):
        await self.page.screenshot({'path': 'web.png'})
        self.pil_display(Image.open("web.png"))

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='LED Screen Client')
        parser.add_argument('--website', metavar='N', type=str, nargs='+',
                            help='Website to display')
        parser.add_argument('--mode', metavar='N', type=str, nargs='+',
                            help='Use camera mode')
    
        args = parser.parse_args()
        
        client = Client()
        client.init()
        
        if(args.mode == "website"):
            if(args.website):
                asyncio.get_event_loop().run_until_complete(asyncio.gather(client.load_website(args.website[0])))
        if(args.mode == "camera"):
            client.load_camera()
            client.camera.start()
            
        while True:
            client.leds.show()
            if(args.mode == "website"):
                asyncio.get_event_loop().run_until_complete(asyncio.gather(client.website_display()))
            if(args.mode == "camera"):
                client.camera_display()
    except KeyboardInterrupt:
        print("Exiting LED client")
        if(args.mode == "camera"):
            client.camera.stop()
        client.leds.blackout()
        client.leds.show()