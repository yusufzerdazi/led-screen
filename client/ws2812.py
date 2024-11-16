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

        self.strips[0].strip_delay = 0.0
        self.strips[1].strip_delay = 0.5
        
        # Add delay between strip updates (in seconds)
        self.strip_delay = 0.0
        
        # Create thread pool for parallel updates
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

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

    def update_strip(self, strip):
        """Update a single strip"""
        with lock:
            if strip.strip_delay > 0:
                time.sleep(strip.strip_delay)
            strip.update_strip()

    def show(self):
        """Update all strips in parallel"""
        # Submit both strip updates to thread pool
        futures = [
            self.pool.submit(self.update_strip, strip)
            for strip in self.strips
        ]
        # Wait for both updates to complete
        concurrent.futures.wait(futures)

    def init(self):
        self.blackout()
        self.show()
        
    def set_strip_delay(self, delay):
        """Set the delay between strip updates in seconds"""
        self.strip_delay = delay