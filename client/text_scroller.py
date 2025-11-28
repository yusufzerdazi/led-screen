"""
Text scrolling utilities for LED display.

Provides scrolling text with wobbly effects, proper centering, and full exit animation.
"""

import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class TextScroller:
    """Handles scrolling text rendering with effects"""
    
    def __init__(self, width, height, font_path, font_size):
        """Initialize text scroller
        
        Args:
            width: Display width in pixels
            height: Display height in pixels
            font_path: Path to TTF font file
            font_size: Base font size
        """
        self.width = width
        self.height = height
        self.font_path = font_path
        self.font_size = font_size
        self._font_cache = None
    
    def _get_font(self):
        """Get or load the TTF font"""
        if self._font_cache is None and self.font_path:
            try:
                self._font_cache = ImageFont.truetype(self.font_path, self.font_size)
            except Exception as e:
                print(f"Warning: Could not load font {self.font_path}: {e}")
                self._font_cache = None
        return self._font_cache
    
    def _text_to_pixel_map(self, text, font_size_scale=1.0):
        """Render text using TTF font and convert to pixel map
        
        Returns a list of (x, y) tuples representing pixel positions relative to text origin
        """
        font = self._get_font()
        if font is None:
            return []
        
        # Render text at high resolution for better quality
        scale_factor = 4  # Render at 4x resolution for smoother edges
        render_size = int(self.font_size * font_size_scale * scale_factor)
        
        try:
            # Create temporary font at scaled size
            temp_font = ImageFont.truetype(self.font_path, render_size) if self.font_path else None
            if temp_font is None:
                return []
            
            # Get text bounding box to determine canvas size
            temp_img = Image.new('RGB', (render_size * len(text) * 2, render_size * 2), (0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=temp_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Create image for rendering
            img_width = int(text_width + render_size)
            img_height = int(text_height + render_size)
            img = Image.new('RGB', (img_width, img_height), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw text in white
            draw.text((render_size // 2, render_size // 2), text, font=temp_font, fill=(255, 255, 255))
            
            # Convert to numpy array and extract pixel positions
            pixels = np.array(img)
            # Get all white pixels (where text is)
            text_pixels = np.where((pixels[:, :, 0] > 128) & 
                                   (pixels[:, :, 1] > 128) & 
                                   (pixels[:, :, 2] > 128))
            
            # Convert to list of (x, y) coordinates relative to text origin
            pixel_map = []
            for y, x in zip(text_pixels[0], text_pixels[1]):
                # Scale down to final resolution and adjust for origin
                final_x = (x - render_size // 2) / scale_factor
                final_y = (y - render_size // 2) / scale_factor
                pixel_map.append((final_x, final_y))
            
            return pixel_map
            
        except Exception as e:
            print(f"Error rendering text to pixel map: {e}")
            return []
    
    def _apply_wobble_effect(self, px, py, elapsed, text_center_x, text_center_y, wobble_amount=1.0):
        """Apply wobbly effect to pixel positions
        
        Args:
            px, py: Pixel position relative to text origin
            elapsed: Elapsed time for animation
            text_center_x, text_center_y: Center of the text for reference
            wobble_amount: Amount of wobbling (rotation/translation)
        
        Returns (offset_x, offset_y) for the pixel position
        """
        # Base wave effect - per-pixel wave based on position
        wave_freq_x = 2.5
        wave_freq_y = 3.0
        wave_amp = 0.8
        
        # Per-pixel wave based on position
        wave_x = math.sin(elapsed * wave_freq_x + px * 0.3) * wave_amp
        wave_y = math.sin(elapsed * wave_freq_y + py * 0.2) * wave_amp
        
        # Dynamic wobbling - overall rotation only (no translation to avoid affecting scroll position)
        wobble_rot = math.sin(elapsed * 1.2) * wobble_amount * 0.1  # Rotation in radians
        
        # Apply wobbling (rotation around text center)
        wobble_cos = math.cos(wobble_rot)
        wobble_sin = math.sin(wobble_rot)
        
        # Rotate around text center
        # First, translate to center-relative coordinates
        px_rel = px - text_center_x
        py_rel = py - text_center_y
        
        # Rotate around center
        px_rotated = px_rel * wobble_cos - py_rel * wobble_sin
        py_rotated = px_rel * wobble_sin + py_rel * wobble_cos
        
        # Translate back to origin-relative coordinates
        px_rotated_abs = px_rotated + text_center_x
        py_rotated_abs = py_rotated + text_center_y
        
        # Calculate offset (difference between rotated and original)
        # Add wave offset for visual effect
        offset_x = (px_rotated_abs - px) + wave_x
        offset_y = (py_rotated_abs - py) + wave_y
        
        return (offset_x, offset_y)
    
    def render_scrolling_text(self, text, elapsed, scroll_speed=20.0, 
                             wobble_amount=1.0, font_size_scale=1.0, 
                             base_color=(255, 255, 255), color_source=None):
        """Render scrolling text that moves across the screen with wobbly effects
        
        Args:
            text: Text to display
            elapsed: Elapsed time since text started (seconds)
            scroll_speed: Pixels per second
            wobble_amount: Amount of wobbling effect (0 = no wobble)
            font_size_scale: Scale factor for font size
            base_color: RGB color tuple for text (default: white) - used if color_source is None
            color_source: Optional PIL Image to sample colors from (text pixels will use colors from this image)
        
        Returns:
            PIL Image with scrolling text
        """
        # Create black image
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        
        # Get text pixel map
        pixel_map = self._text_to_pixel_map(text, font_size_scale)
        if not pixel_map:
            return img
        
        # Prepare color source if provided
        color_source_pixels = None
        if color_source:
            # Resize color source to match display dimensions if needed
            if color_source.size != (self.width, self.height):
                color_source = color_source.resize((self.width, self.height), Image.LANCZOS)
            color_source_pixels = color_source.load()
        
        # Calculate text bounds
        min_x = min(px for px, py in pixel_map)
        max_x = max(px for px, py in pixel_map)
        min_y = min(py for px, py in pixel_map)
        max_y = max(py for px, py in pixel_map)
        text_width = max_x - min_x
        text_height = max_y - min_y
        text_center_x_rel = (min_x + max_x) / 2
        text_center_y_rel = (min_y + max_y) / 2
        
        # Calculate scroll position (same for wobble and non-wobble)
        # px values are relative to text origin (at render_size//2, which becomes 0 after subtraction)
        # min_x is leftmost pixel (typically 0 or negative), max_x is rightmost pixel (positive)
        # current_x is the x-position of the origin (leftmost pixel if min_x=0)
        # Use non-wobble bounds as source of truth for positioning
        
        # At elapsed=0, we want the leftmost pixel to start just off-screen to the right
        # Leftmost pixel should be at: current_x + min_x = self.width + 1
        # So: start_x + min_x = self.width + 1
        # Therefore: start_x = self.width + 1 - min_x
        start_x = self.width + 1 - min_x
        
        # At end, we want the leftmost pixel to be off-screen left
        # Leftmost pixel should be off-screen: current_x + min_x <= -padding
        # So: end_x <= -min_x - padding
        # Use larger padding to ensure clean exit, especially for short text
        # For short text like "hai", we need more padding to ensure it fully disappears
        exit_padding = max(text_width * 0.8, 15)  # Padding for clean exit (at least 15px, 80% of width)
        end_x = -min_x - exit_padding  # End position (leftmost pixel off-screen left)
        scroll_distance = start_x - end_x  # Total distance to scroll
        
        # Calculate scroll time - ensure it's never zero or negative
        if scroll_distance <= 0:
            scroll_time = 1.0  # Fallback to prevent division by zero
        else:
            scroll_time = scroll_distance / scroll_speed  # Time to complete scroll
        
        # Calculate current x position
        # Don't restrict progress - let it continue smoothly
        scroll_progress = elapsed / scroll_time if scroll_time > 0 else 0.0
        current_x = start_x - (scroll_progress * scroll_distance)
        
        # Center vertically - move up by 14 pixels (1 pixel lower than before)
        # Use consistent vertical centering regardless of wobble
        base_center_y = self.height // 2 - 14
        
        # Draw pixels - wobble is applied as pure visual transformation
        # Math breakdown:
        # - px, py are relative to text origin (center of rendered text)
        # - current_x is the x-position of the text origin (center)
        # - For non-wobble: screen_x = current_x + px, screen_y = base_center_y + py
        # - For wobble: apply offset on top of non-wobble position
        
        pixels = img.load()
        for px, py in pixel_map:
            # Non-wobble screen position (source of truth)
            # Horizontal: text origin at current_x, pixel at px relative to origin
            screen_x_non_wobble = current_x + px
            # Vertical: center at base_center_y, pixel at py relative to center (py is already centered)
            screen_y_non_wobble = base_center_y + py
            
            # Apply wobble as visual offset only
            if wobble_amount > 0:
                wobble_offset_x, wobble_offset_y = self._apply_wobble_effect(
                    px, py, elapsed, text_center_x_rel, text_center_y_rel, wobble_amount
                )
                screen_x = int(screen_x_non_wobble + wobble_offset_x)
                screen_y = int(screen_y_non_wobble + wobble_offset_y)
            else:
                screen_x = int(screen_x_non_wobble)
                screen_y = int(screen_y_non_wobble)
            
            # Only draw if within screen bounds
            if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                # Use color from source if available, otherwise use base_color
                if color_source_pixels:
                    # Sample color from color_source at the screen position
                    # Clamp coordinates to valid range
                    src_x = max(0, min(self.width - 1, screen_x))
                    src_y = max(0, min(self.height - 1, screen_y))
                    pixels[screen_x, screen_y] = color_source_pixels[src_x, src_y]
                else:
                    pixels[screen_x, screen_y] = base_color
        
        return img
