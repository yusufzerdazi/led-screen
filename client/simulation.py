import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer

# LED strip configuration:
LED_1_COUNT = 600  # Number of LED pixels.
LED_2_COUNT = 600  # Number of LED pixels.


class ImageWidget(QWidget):
    def __init__(self, width, height):
        super().__init__()

        # Create a QImage and QLabel
        self.image = QImage(width, height, QImage.Format_RGB32)
        self.pixmap_label = QLabel(self)

        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.pixmap_label)
        self.setLayout(layout)

    def set_pixel_color(self, x, y, r, g, b):
        """Set the color of the pixel at (x, y) to the specified RGB values."""
        # Ensure the coordinates are within the bounds of the image
        if 0 <= x < self.image.width() and 0 <= y < self.image.height():
            color = QColor(r, g, b)
            for i in range(x * 10, x * 10 + 10):
                for j in range(y * 10, y * 10 + 10):
                    self.image.setPixel(i, j, color.rgb())

    def update_pixmap(self):
        """Update the pixmap with the current image state."""
        pixmap = QPixmap.fromImage(self.image)
        self.pixmap_label.setPixmap(pixmap)


class Leds:
    def __init__(self, width, height, brightness=1):
        self.width = width
        self.height = height
        self.brightness = brightness

        self.app = QApplication(sys.argv)
        self.window = ImageWidget(10 * self.width, 10 * self.height)
        self.window.setFixedWidth(10 * self.width)
        self.window.setFixedHeight(10 * self.height)
        self.window.show()
        
        # Store update callback for timer
        self.update_callback = None
        self.timer = None

    def get_pixel_index(self, x, y):
        reversed = (y % 2 == 0)
        if reversed:
            index = x + y * self.width
        else:
            index = (self.width - x - 1) + y * self.width
        return index % LED_1_COUNT

    def set_pixel_color(self, x, y, r, g, b):
        self.window.set_pixel_color(x, y, r, g, b)

    def blackout(self):
        pass

    def show(self):
        self.window.update_pixmap()
        # Process Qt events to keep window responsive
        self.app.processEvents()

    def init(self):
        # Don't block here - let the main loop call show() which processes events
        pass
    
    def start_event_loop(self, update_callback):
        """Start Qt event loop with periodic update callback"""
        self.update_callback = update_callback
        
        # Create timer to call update function periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self._timer_update)
        self.timer.start(50)  # Update every 50ms (20 FPS)
        
        # Start Qt event loop (blocks until window closes)
        self.app.exec()
    
    def _timer_update(self):
        """Called by Qt timer to update display"""
        if self.update_callback:
            try:
                self.update_callback()
            except Exception as e:
                print(f"Error in update callback: {e}")
        
    def set_strip_delay(self, delay):
        pass
