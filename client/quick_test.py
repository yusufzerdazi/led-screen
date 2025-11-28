#!/usr/bin/env python3
"""
Quick test script to verify the music visualizer is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client import Client
import simulation
import time
import json

def test_visualizer():
    print("Testing music visualizer...")
    
    # Use simulation mode for testing
    leds = simulation.Leds(40, 30)
    client = Client(leds, server=False)
    
    # Initialize
    client.init()
    client.display_mode = 'frequency'
    
    print("Sending test frequency data...")
    
    # Send test frequency data
    test_frequencies = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    msg = {
        'type': 'frequency',
        'frequencies': test_frequencies
    }
    
    client.frequency_display(msg)
    client.leds.show()
    
    print("Test complete! Check the simulation window for LED patterns.")
    print("You should see a pulsing center circle with warm colors (red/orange)")
    print("and an outer ring with cool colors (blue/cyan).")
    
    # Keep the simulation window open
    print("Press Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    test_visualizer()
