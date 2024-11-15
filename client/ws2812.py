import time
import board
import busio
import neopixel
import threading
import concurrent.futures
from pi5neo import Pi5Neo
#from adafruit_bus_device.spi_device import SPIDevice

# LED strip configuration:
LED_1_COUNT = 600        # Number of LED pixels.

LED_2_COUNT = 600       # Number of LED pixels.

lock = threading.RLock()
print(dir(neopixel))
class Leds:
    def __init__(self, width, height, brightness = 0.1):
        self.width = width
        self.height = height
        self.brightness = brightness
        
        self.strips = [
            Pi5Neo('/dev/spidev0.0', 600, 800),
            Pi5Neo('/dev/spidev5.0', 600, 800)
            #neopixel.NeoPixel_SPI(busio.SPI(board.SCK), 600, auto_write=False, frequency=5120000, reset_time=0.005),
            #neopixel.NeoPixel_SPI(busio.SPI(15), 600, auto_write=False, frequency=5120000, reset_time=0.005)
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
        strip.set_led_color(index, min(int(r * self.brightness), 255), min(int(g * self.brightness), 255), min(int(b * self.brightness), 255))
        #strip[index] = (min(int(r * self.brightness), 255), min(int(g * self.brightness), 255), min(int(b * self.brightness), 255))
        

    def blackout(self):
        for strip in self.strips:
            strip.fill_strip(0, 0, 0)

    def loop(self):
        while True:
            self.show()

    def fill(self, colour):
        for strip in self.strips:
            for i in range(strip.n):
                strip[i] = colour

    def start(self):
        thread = threading.Thread(target=self.loop)
        thread.start()

    def show(self):
        with lock:
           t1 = threading.Thread(target=self.strips[0].update_strip())
           t2 = threading.Thread(target=self.strips[1].update_strip())

           t1.start()
           t2.start()

           t1.join()
           t2.join()


    def init(self):
        self.blackout()
        self.show()