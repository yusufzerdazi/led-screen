from PIL import Image, ImageDraw, ImageFont
import time

class TextScroller:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
        self.scroll_position = height
        self.current_text = None
        self.is_scrolling = False
        
    def start_scroll(self, text):
        self.current_text = text
        self.scroll_position = self.height
        self.is_scrolling = True
        
    def get_frame(self):
        if not self.is_scrolling:
            return None
            
        # Create image with black background
        image = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw text at current scroll position
        draw.text((2, self.scroll_position), self.current_text, font=self.font, fill=(255, 255, 255))
        
        # Update scroll position
        self.scroll_position -= 1
        
        # Check if scrolling is complete
        text_bbox = draw.textbbox((0, 0), self.current_text, font=self.font)
        text_height = text_bbox[3] - text_bbox[1]
        if self.scroll_position < -text_height:
            self.is_scrolling = False
            
        return image 