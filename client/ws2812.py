import time
from rpi_ws281x import ws, Color, Adafruit_NeoPixel

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

class Leds:
    def __init__(self, width, height, brightness = 1):
        self.width = width
        self.height = height
        self.brightness = brightness
        self.strips = [
            Adafruit_NeoPixel(LED_1_COUNT, LED_1_PIN, LED_1_FREQ_HZ,
                                LED_1_DMA, LED_1_INVERT, LED_1_BRIGHTNESS,
                                LED_1_CHANNEL, LED_1_STRIP),
            Adafruit_NeoPixel(LED_2_COUNT, LED_2_PIN, LED_2_FREQ_HZ,
                                LED_2_DMA, LED_2_INVERT, LED_2_BRIGHTNESS,
                                LED_2_CHANNEL, LED_2_STRIP)
        ]

    def get_strip(self, x, y):
        return self.strips[(x + y * self.width > LED_1_COUNT) * 1]

    def get_pixel_index(self, x, y):
        reversed = (y % 2 == 0)
        if reversed:
            index = x + y * self.width
        else:
            index = (self.width - x - 1) + y * self.width
        return index % LED_1_COUNT
    
    def set_pixel_color(self, x, y, r, g, b):
        strip = self.get_strip(x, y)
        strip.setPixelColor(self.get_pixel_index(x, y), Color(min(int(r * self.brightness), 255), min(int(g * self.brightness), 255), min(int(b * self.brightness), 255)))
    
    def blackout(self):
        for strip in self.strips:
            for i in range(max(strip.numPixels(), strip.numPixels())):
                strip.setPixelColor(i, Color(0, 0, 0))
    
    def show(self):
        for strip in self.strips:
            strip.show()
            time.sleep(16.0 / 1000.0)
            
    def init(self):
        for strip in self.strips:
            strip.begin()
        self.blackout()
        self.show()