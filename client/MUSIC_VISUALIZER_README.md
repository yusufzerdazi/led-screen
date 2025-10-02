# Music Event Visualizer

This is an updated version of the LED screen client, specifically designed for music events with sound-reactive, sparse lighting patterns.

## Features

- **Sound Reactive**: Responds to audio frequency data with bass and treble visualization
- **Sparse Lighting**: Center-focused patterns with dark edges to avoid blinding the crowd
- **AI-Generated Visuals**: Uses AI to generate Hydra visualizations optimized for music
- **No Camera/Listener**: Removed camera and speech recognition functionality
- **Music-Focused**: Designed specifically for music events and concerts

## Key Changes from Original

1. **Removed Components**:
   - Camera functionality (Picamera2)
   - Speech recognition (listener.py)
   - Selenium webdriver for complex interactions
   - Text scrolling and quips

2. **Updated AI Model**:
   - Changed from GPT-4 to GPT-4o-mini for efficiency
   - Updated prompts to focus on music visualization
   - Removed "quip" functionality
   - Focus on sparse, center-focused patterns

3. **Enhanced Frequency Display**:
   - Bass frequencies create pulsing center circles (warm colors)
   - Treble frequencies create outer rings (cool colors)
   - Sparse lighting to avoid overwhelming the audience

## Usage

### Basic Setup
```bash
# Start the music visualizer
./run.sh
```

### Manual Start
```bash
# Start Hydra server
cd /path/to/hydra && npm run dev &

# Start the visualizer
python3 client.py --mode music
```

### Testing
```bash
# Test with simulated audio data
python3 test_music_visualizer.py
```

## Configuration

The visualizer expects MQTT messages with frequency data:
```json
{
  "type": "frequency",
  "frequencies": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}
```

- First 4 values: Bass frequencies (center pulsing)
- Last 4 values: Treble frequencies (outer rings)

## Visual Patterns

- **Bass Response**: Warm colors (red/orange) in center circles
- **Treble Response**: Cool colors (blue/cyan) in outer rings
- **Sparse Design**: Only center area lights up, edges stay dark
- **Dynamic Scaling**: Pattern size scales with audio intensity

## Requirements

- Python 3.8+
- MQTT broker running on localhost:1883
- Hydra server running on localhost:5173
- LED strip hardware (or simulation mode)

## Dependencies

- pi5neo (LED control)
- pillow (image processing)
- paho-mqtt (MQTT communication)
- openai (AI visualization generation)
- numpy (audio processing)
- python-dotenv (environment variables)

## Environment Variables

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Hardware Setup

The visualizer is designed for a 40x30 LED matrix with two strips:
- Strip 1: 600 LEDs (spidev1.0)
- Strip 2: 600 LEDs (spidev5.0)

For testing without hardware, use simulation mode:
```bash
python3 client.py --mode music --simulate
```
