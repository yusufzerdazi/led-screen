#!/usr/bin/env python3
"""
Populate default gesture responses for new Burning Man/decompression gestures
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client.response_storage import ResponseStorage

# Default responses for new gestures
GESTURE_RESPONSES = {
    'peace': [
        {'action': {'type': 'status', 'status': 'peace'}, 'tts_response': 'peace and love', 'text_scroller': {'text': 'peace', 'wobble_amount': 0.0, 'scroll_time': 4.0}},
        {'action': {'type': 'status', 'status': 'peace'}, 'tts_response': 'peace out', 'text_scroller': {'text': 'peace out', 'wobble_amount': 0.0, 'scroll_time': 5.5}},
        {'action': {'type': 'status', 'status': 'peace'}, 'tts_response': 'spreading peace', 'text_scroller': {'text': 'peace', 'wobble_amount': 0.0, 'scroll_time': 4.0}},
    ],
    'heart': [
        {'action': {'type': 'status', 'status': 'heart'}, 'tts_response': 'sending love', 'text_scroller': {'text': '<3', 'wobble_amount': 0.0, 'scroll_time': 2.5}},
        {'action': {'type': 'status', 'status': 'heart'}, 'tts_response': 'love you too', 'text_scroller': {'text': 'love', 'wobble_amount': 0.0, 'scroll_time': 3.5}},
        {'action': {'type': 'status', 'status': 'heart'}, 'tts_response': 'radiating love', 'text_scroller': {'text': '<3', 'wobble_amount': 0.0, 'scroll_time': 2.5}},
    ],
    'rock_on': [
        {'action': {'type': 'status', 'status': 'rock_on'}, 'tts_response': 'rock on', 'text_scroller': {'text': 'rock on', 'wobble_amount': 0.0, 'scroll_time': 5.0}},
        {'action': {'type': 'status', 'status': 'rock_on'}, 'tts_response': 'hell yeah', 'text_scroller': {'text': 'hell yeah', 'wobble_amount': 0.0, 'scroll_time': 5.5}},
        {'action': {'type': 'status', 'status': 'rock_on'}, 'tts_response': 'metal', 'text_scroller': {'text': 'metal', 'wobble_amount': 0.0, 'scroll_time': 4.0}},
    ],
    'point': [
        {'action': {'type': 'status', 'status': 'point'}, 'tts_response': 'pointing at you', 'text_scroller': {'text': 'you', 'wobble_amount': 0.0, 'scroll_time': 2.5}},
        {'action': {'type': 'status', 'status': 'point'}, 'tts_response': 'i see you', 'text_scroller': {'text': 'i see you', 'wobble_amount': 0.0, 'scroll_time': 5.5}},
    ],
    'clap': [
        {'action': {'type': 'status', 'status': 'clap'}, 'tts_response': 'clapping with you', 'text_scroller': {'text': 'clap clap', 'wobble_amount': 0.0, 'scroll_time': 6.0}},
        {'action': {'type': 'status', 'status': 'clap'}, 'tts_response': 'applause', 'text_scroller': {'text': 'applause', 'wobble_amount': 0.0, 'scroll_time': 5.5}},
    ],
    'fist_pump': [
        {'action': {'type': 'status', 'status': 'fist_pump'}, 'tts_response': 'yes', 'text_scroller': {'text': 'yes', 'wobble_amount': 0.0, 'scroll_time': 3.0}},
        {'action': {'type': 'status', 'status': 'fist_pump'}, 'tts_response': 'pump it up', 'text_scroller': {'text': 'pump', 'wobble_amount': 0.0, 'scroll_time': 3.5}},
    ],
    'surrender': [
        {'action': {'type': 'status', 'status': 'surrender'}, 'tts_response': 'hands up', 'text_scroller': {'text': 'hands up', 'wobble_amount': 0.0, 'scroll_time': 5.5}},
        {'action': {'type': 'status', 'status': 'surrender'}, 'tts_response': 'i surrender', 'text_scroller': {'text': 'surrender', 'wobble_amount': 0.0, 'scroll_time': 6.0}},
    ],
    'prayer': [
        {'action': {'type': 'status', 'status': 'prayer'}, 'tts_response': 'namaste', 'text_scroller': {'text': 'namaste', 'wobble_amount': 0.0, 'scroll_time': 5.0}},
        {'action': {'type': 'status', 'status': 'prayer'}, 'tts_response': 'blessings', 'text_scroller': {'text': 'blessings', 'wobble_amount': 0.0, 'scroll_time': 6.0}},
    ],
    'air_guitar': [
        {'action': {'type': 'status', 'status': 'air_guitar'}, 'tts_response': 'rock and roll', 'text_scroller': {'text': 'rock', 'wobble_amount': 0.0, 'scroll_time': 3.5}},
        {'action': {'type': 'status', 'status': 'air_guitar'}, 'tts_response': 'air guitar', 'text_scroller': {'text': 'guitar', 'wobble_amount': 0.0, 'scroll_time': 4.5}},
    ],
}

def main():
    storage = ResponseStorage()
    print("Populating new gesture responses...")
    total = 0
    
    for gesture_name, responses in GESTURE_RESPONSES.items():
        print(f"\nAdding {len(responses)} responses for '{gesture_name}':")
        for response in responses:
            storage.save_response('gesture', gesture_name, response)
            print(f"  - {response.get('tts_response', 'no TTS')}")
            total += 1
    
    print(f"\n✓ Added {total} gesture responses")
    print("\nVideos needed (add to client/videos/):")
    for gesture_name in GESTURE_RESPONSES.keys():
        print(f"  - {gesture_name}.mp4")

if __name__ == '__main__':
    main()



