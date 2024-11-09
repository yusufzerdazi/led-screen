import time
import board
import busio
import neopixel_spi
import threading
import concurrent.futures
from adafruit_bus_device.spi_device import SPIDevice

print(dir(board))

#from rpi_ws281x import ws, Color, Adafruit_NeoPixel
from pi5neo import Pi5Neo

# LED strip configuration:
LED_1_COUNT = 600        # Number of LED pixels.

LED_2_COUNT = 600       # Number of LED pixels.

lock = threading.RLock()

class Leds:
    def __init__(self, width, height, brightness = 1):
        self.width = width
        self.height = height
        self.brightness = brightness
        
        self.strips = [
            neopixel_spi.NeoPixel_SPI(busio.SPI(board.SCK), 600, auto_write=False),
            neopixel_spi.NeoPixel_SPI(busio.SPI(15), 600, auto_write=False)
        ]

    def get_strip(self, x, y):
        return self.strips[(x + y * self.width >= LED_1_COUNT) * 1]

    def get_pixel_index(self, x, y):
        reversed = (y % 2 == 0)
        if reversed:
            index = x + y * self.width
        else:
            index = (self.width - x - 1) + y * self.width
        return index % LED_1_COUNT

    def set_pixel_color(self, x, y, r, g, b):
        strip = self.get_strip(x, y)
        index = self.get_pixel_index(x, y)
        with lock:
            strip[index] = (min(int(r * self.brightness), 255), min(int(g * self.brightness), 255), min(int(b * self.brightness), 255))
        

    def blackout(self):
        for strip in self.strips:
            strip.deinit()

    def loop(self):
        while True:
            self.show()
            time.sleep(16.0 / 1000)

    def start(self):
        thread = threading.Thread(target=self.loop)
        thread.start()

    def show(self):
        with lock:
            for strip in self.strips:
                strip.show()
            # t1 = threading.Thread(target=self.strips[0].show)
            # t2 = threading.Thread(target=self.strips[1].show)

            # t1.start()
            # t2.start()

            # t1.join()
            # t2.join()


    def init(self):
        self.blackout()
        self.show()