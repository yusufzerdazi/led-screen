#!/usr/bin/env python3
"""
Test script to validate text scrolling math and behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_scroller import TextScroller
from PIL import Image
import math

def test_text_bounds(text, font_path, font_size, width=40, height=30):
    """Test text bounds calculation"""
    scroller = TextScroller(width, height, font_path, font_size)
    pixel_map = scroller._text_to_pixel_map(text)
    
    if not pixel_map:
        print(f"  ERROR: No pixel map generated for '{text}'")
        return None
    
    min_x = min(px for px, py in pixel_map)
    max_x = max(px for px, py in pixel_map)
    min_y = min(py for px, py in pixel_map)
    max_y = max(py for px, py in pixel_map)
    
    text_width = max_x - min_x
    text_height = max_y - min_y
    text_center_x_rel = (min_x + max_x) / 2
    text_center_y_rel = (min_y + max_y) / 2
    
    print(f"  Text: '{text}'")
    print(f"    Bounds: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
    print(f"    Width: {text_width:.2f}, Height: {text_height:.2f}")
    print(f"    Center: ({text_center_x_rel:.2f}, {text_center_y_rel:.2f})")
    
    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'text_width': text_width,
        'text_height': text_height,
        'text_center_x_rel': text_center_x_rel,
        'text_center_y_rel': text_center_y_rel,
        'pixel_map': pixel_map
    }

def test_scroll_positions(text_info, width=40, height=30, scroll_speed=20.0):
    """Test scroll position calculations"""
    min_x = text_info['min_x']
    max_x = text_info['max_x']
    text_width = text_info['text_width']
    text_center_x_rel = text_info['text_center_x_rel']
    
    # Calculate start/end positions (matching the actual code)
    # Leftmost pixel should start at x=41 (just off-screen)
    start_x = width + 1 - min_x
    exit_padding = max(text_width * 0.8, 15)
    end_x = -min_x - exit_padding
    scroll_distance = start_x - end_x
    scroll_time = scroll_distance / scroll_speed
    
    print(f"  Scroll calculations:")
    print(f"    start_x: {start_x:.2f}")
    print(f"    end_x: {end_x:.2f}")
    print(f"    scroll_distance: {scroll_distance:.2f}")
    print(f"    scroll_time: {scroll_time:.2f}s")
    
    # Test at key points
    test_points = [
        (0.0, "Start (elapsed=0)"),
        (scroll_time * 0.5, "Middle (elapsed=50%)"),
        (scroll_time, "End (elapsed=100%)"),
        (scroll_time * 1.1, "Past end (elapsed=110%)"),
    ]
    
    print(f"  Position at key points:")
    for elapsed, label in test_points:
        scroll_progress = elapsed / scroll_time if scroll_time > 0 else 0.0
        current_x = start_x - (scroll_progress * scroll_distance)
        
        # Calculate where leftmost and rightmost pixels would be
        leftmost_x = current_x + min_x
        rightmost_x = current_x + max_x
        
        print(f"    {label}:")
        print(f"      current_x: {current_x:.2f}")
        print(f"      leftmost pixel: {leftmost_x:.2f} (screen: 0-{width})")
        print(f"      rightmost pixel: {rightmost_x:.2f} (screen: 0-{width})")
        
        # Check if fully off-screen
        if elapsed == 0.0:
            # At start, rightmost pixel should be >= width (at or past right edge)
            # This ensures immediate appearance without delay
            if rightmost_x >= width:
                print(f"      ✓ Starts at right edge (rightmost={rightmost_x:.2f} >= {width}) - will appear immediately")
            else:
                print(f"      ✗ ERROR: Starts too far left! rightmost={rightmost_x:.2f} < {width}")
        
        if elapsed >= scroll_time:
            # At end, leftmost pixel should be <= -padding (off-screen left)
            if leftmost_x <= -5:
                print(f"      ✓ Fully off-screen left (leftmost={leftmost_x:.2f} <= -5)")
            else:
                print(f"      ✗ WARNING: Not fully off-screen! leftmost={leftmost_x:.2f} > -5")
    
    return {
        'start_x': start_x,
        'end_x': end_x,
        'scroll_distance': scroll_distance,
        'scroll_time': scroll_time
    }

def test_wobble_effect(text_info, elapsed=0.0, wobble_amount=1.0):
    """Test wobble effect on pixel positions"""
    pixel_map = text_info['pixel_map']
    text_center_x_rel = text_info['text_center_x_rel']
    text_center_y_rel = text_info['text_center_y_rel']
    
    # Import the wobble function
    from text_scroller import TextScroller
    scroller = TextScroller(40, 30, None, 20)
    
    print(f"  Wobble effect (elapsed={elapsed:.2f}, amount={wobble_amount}):")
    
    # Test a few sample pixels
    test_pixels = [
        (text_info['min_x'], 0, "Leftmost pixel"),
        (text_info['max_x'], 0, "Rightmost pixel"),
        (text_center_x_rel, text_center_y_rel, "Center pixel"),
    ]
    
    max_offset_x = 0
    max_offset_y = 0
    
    for px, py, label in test_pixels:
        offset_x, offset_y = scroller._apply_wobble_effect(
            px, py, elapsed, text_center_x_rel, text_center_y_rel, wobble_amount
        )
        max_offset_x = max(max_offset_x, abs(offset_x))
        max_offset_y = max(max_offset_y, abs(offset_y))
        print(f"    {label} ({px:.2f}, {py:.2f}): offset=({offset_x:.2f}, {offset_y:.2f})")
    
    print(f"    Max offsets: x={max_offset_x:.2f}, y={max_offset_y:.2f}")
    return max_offset_x, max_offset_y

def main():
    print("Text Scroller Math Validation")
    print("=" * 60)
    
    # Find font
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "KiwiSoda.ttf")
    if not os.path.exists(font_path):
        font_path = os.path.join(script_dir, "slkscr.ttf")
    if not os.path.exists(font_path):
        print("ERROR: No font file found!")
        return
    
    font_size = 20
    width = 40
    height = 30
    scroll_speed = 20.0
    
    test_texts = ["hai", "i see you smiling"]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Testing: '{text}'")
        print(f"{'='*60}")
        
        # Test 1: Text bounds
        print("\n1. Text Bounds:")
        text_info = test_text_bounds(text, font_path, font_size, width, height)
        if not text_info:
            continue
        
        # Test 2: Scroll positions
        print("\n2. Scroll Position Calculations:")
        scroll_info = test_scroll_positions(text_info, width, height, scroll_speed)
        
        # Test 3: Wobble effect
        print("\n3. Wobble Effect:")
        max_offset_x, max_offset_y = test_wobble_effect(text_info, elapsed=0.0, wobble_amount=1.0)
        
        # Test 4: Check if padding accounts for wobble
        print("\n4. Padding Analysis:")
        exit_padding = max(text_info['text_width'] * 0.3, 8)
        print(f"    exit_padding: {exit_padding:.2f} (30% of width or min 8px)")
        print(f"    max_wobble_offset_x: {max_offset_x:.2f}")
        if exit_padding < max_offset_x:
            print(f"    ⚠ WARNING: exit_padding ({exit_padding:.2f}) < max_wobble_offset ({max_offset_x:.2f})")
            print(f"      Text might not fully exit with wobble enabled!")
        else:
            print(f"    ✓ Padding sufficient for wobble")

if __name__ == '__main__':
    main()

