#!/usr/bin/env python3
"""
Test script for the music visualizer
This script simulates audio frequency data to test the visualizer
"""

import json
import time
import random
import paho.mqtt.client as mqtt
import threading

class MusicVisualizerTester:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")
        
    def on_disconnect(self, client, userdata, rc):
        print("Disconnected from MQTT broker")
        
    def connect(self):
        try:
            self.client.connect("localhost", 1883, 60)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            return False
            
    def send_frequency_data(self, frequencies):
        """Send frequency data to the visualizer"""
        message = {
            "type": "frequency",
            "frequencies": frequencies
        }
        self.client.publish("led_screen", json.dumps(message))
        print(f"Sent frequency data: {frequencies[:4]}... (bass) {frequencies[-4:]}... (treble)")
        
    def simulate_music(self):
        """Simulate music with varying bass and treble"""
        print("Starting music simulation...")
        
        # Simulate different music patterns
        patterns = [
            # Heavy bass pattern
            lambda t: [0.8, 0.7, 0.6, 0.5, 0.2, 0.1, 0.1, 0.1] if int(t) % 4 < 2 else [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            # High treble pattern  
            lambda t: [0.1, 0.1, 0.1, 0.1, 0.6, 0.7, 0.8, 0.9] if int(t) % 3 == 0 else [0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5],
            # Balanced pattern
            lambda t: [0.4, 0.3, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6] + [random.random() * 0.2 for _ in range(8)],
            # Random pattern
            lambda t: [random.random() for _ in range(8)]
        ]
        
        pattern_index = 0
        start_time = time.time()
        
        while True:
            current_time = time.time() - start_time
            
            # Switch patterns every 10 seconds
            if int(current_time) % 10 == 0 and int(current_time) > 0:
                pattern_index = (pattern_index + 1) % len(patterns)
                print(f"Switching to pattern {pattern_index + 1}")
            
            # Generate frequencies based on current pattern
            frequencies = patterns[pattern_index](current_time)
            
            # Add some randomness
            frequencies = [max(0, min(1, f + random.random() * 0.1 - 0.05)) for f in frequencies]
            
            # Send frequency data
            self.send_frequency_data(frequencies)
            
            time.sleep(0.1)  # 10 FPS
            
    def run(self):
        if not self.connect():
            return
            
        print("Music visualizer tester started!")
        print("This will simulate different music patterns:")
        print("1. Heavy bass (first 4 seconds)")
        print("2. High treble (next 4 seconds)")  
        print("3. Balanced (next 4 seconds)")
        print("4. Random (next 4 seconds)")
        print("Patterns will cycle every 10 seconds")
        print("Press Ctrl+C to stop")
        
        try:
            self.simulate_music()
        except KeyboardInterrupt:
            print("\nStopping music simulation...")
        finally:
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    tester = MusicVisualizerTester()
    tester.run()
