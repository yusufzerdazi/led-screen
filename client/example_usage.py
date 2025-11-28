#!/usr/bin/env python3
"""
Example usage of the music visualizer
This shows how to send frequency data to the visualizer
"""

import json
import time
import paho.mqtt.client as mqtt
import math

class MusicVisualizerExample:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")
        
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
        
    def simulate_song(self):
        """Simulate a song with varying bass and treble"""
        print("Simulating a song...")
        
        # Simulate a 30-second song
        for t in range(300):  # 30 seconds at 10 FPS
            time_in_song = t / 10.0  # Convert to seconds
            
            # Create a song-like pattern
            # Bass drops at 10s, 20s
            bass_intensity = 0.3
            if 8 <= time_in_song <= 12:  # Bass drop 1
                bass_intensity = 0.9
            elif 18 <= time_in_song <= 22:  # Bass drop 2
                bass_intensity = 0.8
            elif 25 <= time_in_song <= 30:  # Final drop
                bass_intensity = 1.0
                
            # Treble varies throughout
            treble_intensity = 0.2 + 0.3 * math.sin(time_in_song * 0.5)
            
            # Create frequency array (8 bands)
            frequencies = [
                bass_intensity * 0.8,  # Sub bass
                bass_intensity * 0.9,  # Bass
                bass_intensity * 0.7,  # Low mid
                bass_intensity * 0.5,  # Mid
                treble_intensity * 0.6,  # High mid
                treble_intensity * 0.8,  # Treble
                treble_intensity * 0.9,  # High treble
                treble_intensity * 0.7   # Air
            ]
            
            # Add some randomness for realism
            frequencies = [max(0, min(1, f + (time.time() % 1 - 0.5) * 0.1)) for f in frequencies]
            
            self.send_frequency_data(frequencies)
            print(f"Time: {time_in_song:.1f}s - Bass: {bass_intensity:.2f}, Treble: {treble_intensity:.2f}")
            
            time.sleep(0.1)  # 10 FPS
            
    def run(self):
        if not self.connect():
            return
            
        print("Music visualizer example started!")
        print("This will simulate a 30-second song with bass drops")
        print("Press Ctrl+C to stop")
        
        try:
            self.simulate_song()
        except KeyboardInterrupt:
            print("\nStopping simulation...")
        finally:
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    example = MusicVisualizerExample()
    example.run()
