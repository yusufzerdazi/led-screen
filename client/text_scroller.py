from PIL import Image, ImageDraw, ImageFont
import textwrap

class TextScroller:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = ImageFont.truetype('slkscr.ttf', 16)
        self.scroll_position = width
        self.current_text = None
        self.is_scrolling = False
        self.scroll_speed = 2
        
        self.y_position = 4
        
    def start_scroll(self, text):
        self.current_text = text.upper()
        self.scroll_position = self.width
        self.is_scrolling = True
        
    def get_frame(self):
        if not self.is_scrolling:
            return None
            
        image = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        draw.text((self.scroll_position, self.y_position), self.current_text, font=self.font, fill=(255, 255, 255))
        
        self.scroll_position -= self.scroll_speed
        
        text_bbox = draw.textbbox((0, 0), self.current_text, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        if self.scroll_position < -text_width:
            self.is_scrolling = False
            
        return image 