import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer, Qt

# LED strip configuration:
LED_1_COUNT = 600  # Number of LED pixels.
LED_2_COUNT = 600  # Number of LED pixels.


class ImageWidget(QWidget):
    def __init__(self, width, height):
        super().__init__()

        # Store original dimensions
        self.original_width = width
        self.original_height = height
        self.aspect_ratio = width / height
        
        # Create a QImage and QLabel
        self.image = QImage(width, height, QImage.Format_RGB32)
        self.pixmap_label = QLabel(self)
        self.pixmap_label.setAlignment(Qt.AlignCenter)
        self.pixmap_label.setScaledContents(False)

        # Set layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pixmap_label)
        self.setLayout(layout)
        
        # Enable window resize
        self.setWindowTitle("LED Screen Simulation")
        
    def resizeEvent(self, event):
        """Maintain aspect ratio when window is resized"""
        super().resizeEvent(event)
        self.update_pixmap()

    def set_pixel_color(self, x, y, r, g, b):
        """Set the color of the pixel at (x, y) to the specified RGB values."""
        # Ensure the coordinates are within the bounds of the image
        if 0 <= x < self.image.width() and 0 <= y < self.image.height():
            color = QColor(r, g, b)
            for i in range(x * 10, x * 10 + 10):
                for j in range(y * 10, y * 10 + 10):
                    self.image.setPixel(i, j, color.rgb())

    def update_pixmap(self):
        """Update the pixmap with the current image state, scaled to fit window while maintaining aspect ratio."""
        pixmap = QPixmap.fromImage(self.image)
        
        # Get available size
        available_size = self.size()
        
        # Calculate size that maintains aspect ratio
        scaled_pixmap = pixmap.scaled(
            available_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation  # Use FastTransformation for better performance
        )
        
        self.pixmap_label.setPixmap(scaled_pixmap)


class Leds:
    def __init__(self, width, height, brightness=1):
        self.width = width
        self.height = height
        self.brightness = brightness

        self.app = QApplication(sys.argv)
        self.window = ImageWidget(10 * self.width, 10 * self.height)
        
        # Set initial size but allow resizing
        self.window.resize(10 * self.width, 10 * self.height)
        
        # Add keyboard shortcut for fullscreen (F11 or F)
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        self.fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self.window)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        
        self.fullscreen_shortcut2 = QShortcut(QKeySequence(Qt.Key_F), self.window)
        self.fullscreen_shortcut2.activated.connect(self.toggle_fullscreen)
        
        # ESC key to exit fullscreen
        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self.window)
        self.escape_shortcut.activated.connect(self.exit_fullscreen)
        
        self.window.show()
        
        print("LED Screen Simulation:")
        print("  - Press F11 or F to toggle fullscreen")
        print("  - Window can be resized (aspect ratio maintained)")
        print("  - Press ESC to exit fullscreen")
        
        # Store update callback for timer
        self.update_callback = None
        self.timer = None
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.window.isFullScreen():
            self.window.showNormal()
        else:
            self.window.showFullScreen()
    
    def exit_fullscreen(self):
        """Exit fullscreen mode"""
        if self.window.isFullScreen():
            self.window.showNormal()

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
