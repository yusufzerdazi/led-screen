import time
import threading
import concurrent.futures
from pi5neo import Pi5Neo

lock = threading.RLock()

class Leds:
    def __init__(self, width, height, brightness = 0.1):
        self.width = width
        self.height = height
        self.brightness = brightness
        
        self.strips = [
            Pi5Neo('/dev/spidev0.0', 600, 800),
            Pi5Neo('/dev/spidev5.0', 600, 800)
        ]

    def get_strip(self, x, y):
        return self.strips[(x + y * self.width >= 600) * 1]

    def get_pixel_index(self, x, y):
        reversed = (y % 2 == 0)
        if reversed:
            index = x + y * self.width
        else:
            index = (self.width - x - 1) + y * self.width
        return index % 600

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