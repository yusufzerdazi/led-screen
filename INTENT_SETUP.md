# Intent-Based Command System Setup Guide

## Overview

This system replaces the live LLM execution with a much faster intent-based classification system. Instead of generating responses on-the-fly, it:

1. **Classifies user input** into predefined intents using a lightweight ML model
2. **Looks up responses** from a CSV file based on the intent
3. **Executes actions** immediately (no waiting for LLM generation)

## Why Intent Models Are Faster

- **Smaller models**: ~500MB vs several GB for full LLMs
- **Classification only**: No text generation, just category selection
- **Faster inference**: Milliseconds vs seconds
- **CPU-friendly**: Runs efficiently on CPU without GPU
- **Lower latency**: Instant responses vs 1-5 second delays

## Installation Steps

### 1. Install Required Dependencies

```bash
pip install transformers torch
```

**Note**: On Raspberry Pi or systems with limited RAM, you can use a smaller model by modifying the code to use `typeform/distilbert-base-uncased-finetuned-squad2` instead of `facebook/bart-large-mnli`.

### 2. Verify Installation

```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier("hello there", ["greeting", "eye", "people"])
print(result)
```

### 3. CSV Structure

The `intent_mappings.csv` file has the following columns:

- **intent**: The intent name (e.g., "greeting", "eye", "people", "gesture_wave")
- **example_phrases**: Pipe-separated example phrases that match this intent (used for keyword fallback)
- **response_json**: JSON string with the action, TTS response, and text scroller config
- **audio_file**: Optional path to audio file (for future audio playback support)
- **usage_count**: Counter for how often this response is used

### 4. CSV Format Example

```csv
intent,example_phrases,response_json,audio_file,usage_count
greeting,hello|hi|hey,"{""action"": {""type"": ""status"", ""status"": ""wave""}, ""tts_response"": ""hello beautiful human"", ""text_scroller"": null}",,0
eye,eye|look|see,"{""action"": {""type"": ""status"", ""status"": ""eye""}, ""tts_response"": ""opening my eye for you"", ""text_scroller"": null}",,0
```

### 5. Response JSON Structure

Each response JSON follows this format:

```json
{
  "action": {
    "type": "status" | "tts_only",
    "status": "eye" | "people" | "wave" | "smile" | "thumbs_up"
  },
  "tts_response": "Text to speak",
  "text_scroller": {
    "text": "Text to display",
    "wobble_amount": 0.05,
    "scroll_time": 8.0
  } | null
}
```

For gestures:
```json
{
  "action": {
    "type": "status",
    "status": "wave",
    "substate": "hand_waving"
  },
  "tts_response": "hello beautiful",
  "text_scroller": {
    "text": "hai",
    "wobble_amount": 0.0,
    "scroll_time": 5.0
  }
}
```

For periodic TTS triggers:
```json
{
  "type": "random",
  "interval_min": 30.0,
  "interval_max": 90.0,
  "interval": 60.0,
  "response": "welcome to decompression"
}
```

## Usage

### Switching from LLM to Intent Handler

In `decompression_mode.py`, replace:

```python
from llm_command_handler import LLMCommandHandler
self.llm_command_handler = LLMCommandHandler(...)
```

With:

```python
from intent_command_handler import IntentCommandHandler
self.intent_command_handler = IntentCommandHandler(
    intent_csv_path="client/intent_mappings.csv",
    use_intent_model=True  # Set to False to use keyword matching only
)
```

### Adding New Intents

1. **Add intent to CSV**: Add a new row with your intent name and example phrases
2. **Add multiple responses**: Add multiple rows with the same intent but different responses for variety
3. **Reload**: The system will automatically pick up new intents on next command, or call `reload_config()`

### Example: Adding a "dance" Intent

```csv
dance,dance|dancing|move|groove,"{""action"": {""type"": ""status"", ""status"": ""wave""}, ""tts_response"": ""let's dance"", ""text_scroller"": {""text"": ""dance"", ""wobble_amount"": 0.1, ""scroll_time"": 6.0}}",,0
```

## Performance Comparison

| Method | Latency | Model Size | CPU Usage | GPU Required |
|--------|---------|------------|-----------|--------------|
| Full LLM | 1-5 seconds | 2-7 GB | High | Recommended |
| Intent Model | 50-200ms | ~500 MB | Low | No |
| Keyword Matching | <1ms | 0 MB | Minimal | No |

## Fallback Behavior

The system uses a two-tier approach:

1. **Primary**: ML-based intent classification (if model available)
2. **Fallback**: Keyword matching (always available)

If the ML model fails or isn't available, it automatically falls back to keyword matching using the `example_phrases` column in the CSV.

## Troubleshooting

### Model Download Issues

If the model fails to download, it will automatically fall back to keyword matching. Check your internet connection and disk space.

### Low Confidence Scores

If the classifier has low confidence (< 0.3), it falls back to keyword matching. You can adjust the threshold in `intent_command_handler.py`:

```python
if confidence > 0.3:  # Adjust this threshold
```

### Adding Audio Files

To use audio files instead of TTS:

1. Place audio files in a directory (e.g., `client/audio/`)
2. Set the `audio_file` column in CSV to the relative path
3. Modify `_execute_response()` to check for audio files and play them

## Benefits

✅ **10-100x faster** than LLM generation  
✅ **Lower resource usage** - runs on CPU efficiently  
✅ **Predictable responses** - CSV-based, easy to edit  
✅ **No API dependencies** - works offline  
✅ **Easy to extend** - just add CSV rows  
✅ **Fallback support** - keyword matching if model unavailable  


