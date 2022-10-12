from ast import arg
import json
from io import BytesIO
import base64
from PIL import Image
import argparse
import asyncio
from pyppeteer import launch

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
    
    async def load_website(self, url):
        self.browser = await launch()
        self.page = await self.browser.newPage()
        await self.page.goto(url)
    
    def on_message(self, client, userdata, msg):
        decoded = json.loads(msg.payload.decode())
        if decoded['type'] == "frequency":
            self.frequency_display(decoded)
        if decoded['type'] == "image":
            self.rgb_display(decoded)
        if decoded['type'] == "rgb":
            self.image_display(decoded)

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
        im = im.resize((self.width, self.height), Image.ANTIALIAS)
        for i in range(self.width):
            for j in range(self.height):
                pix = im.getpixel((i, self.height - j - 1))
                self.leds.set_pixel_color(i, j, pix[0], pix[1], pix[2])

    async def website_display(self):
        await self.page.screenshot({'path': 'web.png'})
        self.pil_display(Image.open("web.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LED Screen Client')
    parser.add_argument('--website', metavar='N', type=str, nargs='+',
                        help='Website to display')

    args = parser.parse_args()
    print(args.website)

    client = Client()
    client.init()

    if(args.website):
        asyncio.get_event_loop().run_until_complete(client.load_website(args.website[0]))

    while True:
        client.leds.show()
        if(args.website):
            asyncio.get_event_loop().run_until_complete(client.website_display)