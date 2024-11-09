import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QColor

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

    def init(self):
        self.app.exec()
