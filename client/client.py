import json
from io import BytesIO
import base64
from PIL import Image

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
    
    def on_message(self, client, userdata, msg):
        decoded = json.loads(msg.payload.decode())
        if decoded['type'] == "frequency":
            self.frequency_display(decoded)
        if decoded['type'] == "image":
            self.image_display(decoded)
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
        im = im.resize((self.width, self.height), Image.ANTIALIAS)
        for i in range(self.width):
            for j in range(self.height):
                pix = im.getpixel((i, self.height - j - 1))
                self.leds.set_pixel_color(i, j, pix[0], pix[1], pix[2])

if __name__ == '__main__':
    client = Client()
    client.init()

    while True:
        client.leds.show()