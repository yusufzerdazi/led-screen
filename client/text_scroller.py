from PIL import Image, ImageDraw, ImageFont
import textwrap

class TextScroller:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 8)
        self.scroll_position = height
        self.current_text = None
        self.is_scrolling = False
        self.wrapped_text = None
        self.scroll_speed = 2
        
    def start_scroll(self, text):
        self.current_text = text
        self.wrapped_text = textwrap.fill(text, width=12)
        self.scroll_position = self.height
        self.is_scrolling = True
        
    def get_frame(self):
        if not self.is_scrolling:
            return None
            
        image = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        draw.text((2, self.scroll_position), self.wrapped_text, font=self.font, fill=(255, 255, 255))
        
        self.scroll_position -= self.scroll_speed
        
        text_bbox = draw.textbbox((0, 0), self.wrapped_text, font=self.font)
        text_height = text_bbox[3] - text_bbox[1]
        if self.scroll_position < -text_height:
            self.is_scrolling = False
            
        return image 