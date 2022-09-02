# NeoPixel library strandtest example
# Author: Tony DiCola (tony@tonydicola.com)
#
# Direct port of the Arduino NeoPixel library strandtest example.  Showcases
# various animations on a strip of NeoPixels.
import time
import random
import argparse
import json
import paho.mqtt.client as mqttClient

from rpi_ws281x import ws, Color, Adafruit_NeoPixel

class PiClient:
    def __init__(self, on_message):
        self.broker_address= "raspberrypi.local"
        self.topic = "mqtt/led-screen"
        self.port = 1883
        self.connected = False
        self.on_message = on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to broker")
            self.connected = True
            self.client.subscribe(self.topic)
            self.client.on_message = self.on_message
        else:
            print("Connection failed")

    def connect(self):
        self.client = mqttClient.Client("client")
        self.client.on_connect=self.on_connect
        self.client.connect(self.broker_address, port=self.port)
        self.client.loop_start()


# LED strip configuration:
LED_1_COUNT = 600        # Number of LED pixels.
LED_1_PIN = 18          # GPIO pin connected to the pixels (must support PWM! GPIO 13 and 18 on RPi 3).
LED_1_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_1_DMA = 10          # DMA channel to use for generating signal (Between 1 and 14)
LED_1_BRIGHTNESS = 20  # Set to 0 for darkest and 255 for brightest
LED_1_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_1_CHANNEL = 0       # 0 or 1
LED_1_STRIP = ws.WS2812_STRIP

LED_2_COUNT = 600       # Number of LED pixels.
LED_2_PIN = 13          # GPIO pin connected to the pixels (must support PWM! GPIO 13 or 18 on RPi 3).
LED_2_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_2_DMA = 9          # DMA channel to use for generating signal (Between 1 and 14)
LED_2_BRIGHTNESS = 20  # Set to 0 for darkest and 255 for brightest
LED_2_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_2_CHANNEL = 1       # 0 or 1
LED_2_STRIP = ws.WS2812_STRIP

WIDTH = 40
HEIGHT = 30
a = 0.0
b = 0.0
color = 0

def get_strip(x, y):
    return (x + y * WIDTH > LED_1_COUNT) * 1


def get_pixel_index(x, y):
    reversed = (y % 2 == 0)
    if reversed:
        index = x + y * WIDTH
    else:
        index = (WIDTH - x - 1) + y * WIDTH
    return index % 600

def blackout(strip):
    for i in range(max(strip.numPixels(), strip.numPixels())):
        strip.setPixelColor(i, Color(0, 0, 0))

def frequency_display(client, userdata, msg):
    global blackout
    frequencies = json.loads(msg.payload.decode())
    x = 0
    for f in frequencies["frequencies"]:
        for y in range(30):
            if int(f * 4) > y:
                strips[get_strip(x, y)].setPixelColor(get_pixel_index(x, y), Color(100, 255 - 6 * x, 255 - 8 * y))#Color(100, y * 5, x * 4))
            else:  
                strips[get_strip(x, y)].setPixelColor(get_pixel_index(x, y), Color(0, 0, 0))
        x += 1

# Main program logic follows:
if __name__ == '__main__':
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clear', action='store_true', help='clear the display on exit')
    args = parser.parse_args()

    # Create NeoPixel object with appropriate configuration.
    strips = [
        Adafruit_NeoPixel(LED_2_COUNT, LED_2_PIN, LED_2_FREQ_HZ,
                               LED_2_DMA, LED_2_INVERT, LED_2_BRIGHTNESS,
                               LED_2_CHANNEL, LED_2_STRIP),
        Adafruit_NeoPixel(LED_1_COUNT, LED_1_PIN, LED_1_FREQ_HZ,
                               LED_1_DMA, LED_1_INVERT, LED_1_BRIGHTNESS,
                               LED_1_CHANNEL, LED_1_STRIP)
    ]
    
    # Intialize the library (must be called once before other functions).
    for strip in strips:
        strip.begin()
        blackout(strip)
        strip.show()
    

    client = PiClient(frequency_display)
    client.connect()

    while True:
        for strip in strips:
            strip.show()
            time.sleep(20 / 1000)