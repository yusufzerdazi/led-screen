"""
Base class for display modes.

All display modes should inherit from this class and implement
the required methods.
"""

class BaseMode:
    """Base class for all display modes"""
    
    def __init__(self, width=256, height=144):
        """
        Initialize the mode.
        
        Args:
            width: Display width in pixels
            height: Display height in pixels
        """
        self.width = width
        self.height = height
        self.frame_interval = None
        self.last_frame_time = 0.0
    
    def setup(self, **kwargs):
        """
        Set up the mode with any additional parameters.
        
        This is called after __init__ and can be used to configure
        the mode with command-line arguments or other settings.
        
        Args:
            **kwargs: Mode-specific configuration parameters
        """
        pass
    
    def set_frame_interval(self, interval):
        """Set desired frame interval for the mode"""
        self.frame_interval = interval
    
    def init(self):
        """
        Initialize the mode.
        
        This is called once when the mode is activated.
        Override this to perform any setup needed for the mode.
        """
        pass
    
    def update(self):
        """
        Generate and return the next frame.
        
        This is called every frame to get the image to display.
        Override this to implement the mode's display logic.
        
        Returns:
            PIL.Image or None: The frame to display, or None if no update
        """
        raise NotImplementedError("Mode must implement update()")
    
    def cleanup(self):
        """
        Clean up the mode.
        
        This is called when the mode is being deactivated or
        the program is shutting down.
        Override this to perform any cleanup needed.
        """
        pass
    
    def on_message(self, message):
        """
        Handle MQTT messages.
        
        This is called when an MQTT message is received (if server mode is enabled).
        Override this to handle mode-specific messages.
        
        Args:
            message: Decoded JSON message
        """
        pass

