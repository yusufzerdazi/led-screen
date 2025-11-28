# Intent-Based System FAQ

## Is it true that intent AI models run much faster than full LLMs?

**Yes, absolutely!** Here's why:

### Performance Comparison

| Metric | Full LLM | Intent Model | Improvement |
|--------|----------|-------------|-------------|
| **Latency** | 1-5 seconds | 50-200ms | **10-100x faster** |
| **Model Size** | 2-7 GB | ~500 MB | **4-14x smaller** |
| **CPU Usage** | High | Low | **Much lower** |
| **GPU Required** | Recommended | No | **CPU-friendly** |
| **Memory** | 4-8 GB RAM | 1-2 GB RAM | **2-4x less** |

### Why Intent Models Are Faster

1. **Classification vs Generation**: Intent models only classify text into categories (like "greeting", "eye", "people"), while LLMs generate entire responses word-by-word.

2. **Smaller Models**: Intent classification models are much smaller because they don't need to understand language generation - just categorization.

3. **Single Forward Pass**: Intent models do a single forward pass through the network, while LLMs generate tokens sequentially (each token requires a forward pass).

4. **Optimized for Speed**: Intent models are optimized for low-latency inference, while LLMs prioritize quality over speed.

## Steps to Use Intent Model

### 1. Install Dependencies

```bash
pip install transformers torch
```

**Note**: On Raspberry Pi or low-memory systems, you can use a smaller model. See `INTENT_SETUP.md` for alternatives.

### 2. Use the Intent Handler

```python
from intent_command_handler import IntentCommandHandler

# Initialize handler
handler = IntentCommandHandler(
    intent_csv_path="client/intent_mappings.csv",
    use_intent_model=True  # Set False for keyword-only mode
)

# Set callbacks
handler.set_status_change_callback(your_status_callback)
handler.set_tts_callback(your_tts_callback)

# Process commands (instant!)
success, is_unknown = handler.process_voice_command("hello there")
```

### 3. Model Loading

The model loads automatically on first use (lazy loading). First classification may take 10-30 seconds to download/load the model, then it's cached in memory.

## CSV Structure for Intent Mappings

The CSV file (`intent_mappings.csv`) has these columns:

### Columns

- **intent**: Intent name (e.g., "greeting", "eye", "people", "gesture_wave")
- **example_phrases**: Pipe-separated phrases (e.g., "hello\|hi\|hey") - used for keyword fallback
- **response_json**: JSON string with action, TTS, and text scroller config
- **audio_file**: Optional path to audio file (for future use)
- **usage_count**: Counter for response selection (auto-incremented)

### Example CSV Row

```csv
greeting,hello|hi|hey,"{""action"": {""type"": ""status"", ""status"": ""wave""}, ""tts_response"": ""hello beautiful human"", ""text_scroller"": null}",,0
```

### Response JSON Format

**Voice Command:**
```json
{
  "action": {
    "type": "status",
    "status": "wave"
  },
  "tts_response": "hello beautiful human",
  "text_scroller": null
}
```

**Gesture:**
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

**Periodic TTS:**
```json
{
  "type": "random",
  "interval_min": 30.0,
  "interval_max": 90.0,
  "interval": 60.0,
  "response": "welcome to decompression"
}
```

## Adding New Intents

### Step 1: Add Intent to CSV

Add a new row with your intent:

```csv
dance,dance|dancing|move|groove,"{""action"": {""type"": ""status"", ""status"": ""wave""}, ""tts_response"": ""let's dance"", ""text_scroller"": {""text"": ""dance"", ""wobble_amount"": 0.1, ""scroll_time"": 6.0}}",,0
```

### Step 2: Add Multiple Responses (Optional)

Add more rows with the same intent but different responses for variety:

```csv
dance,dance|dancing|move|groove,"{""action"": {""type"": ""status"", ""status"": ""wave""}, ""tts_response"": ""let's dance"", ""text_scroller"": {""text"": ""dance"", ""wobble_amount"": 0.1, ""scroll_time"": 6.0}}",,0
dance,dance|dancing|move|groove,"{""action"": {""type"": ""status"", ""status"": ""wave""}, ""tts_response"": ""groove with me"", ""text_scroller"": {""text"": ""groove"", ""wobble_amount"": 0.1, ""scroll_time"": 6.0}}",,0
```

### Step 3: Reload (Automatic)

The system automatically picks up new intents. Or call `reload_config()` manually.

## Audio Files

To use audio files instead of TTS:

1. Place audio files in a directory (e.g., `client/audio/greeting_1.wav`)
2. Set the `audio_file` column in CSV: `client/audio/greeting_1.wav`
3. Modify `_execute_response()` to check for audio files and play them

**Note**: Audio file support is not yet implemented in the base handler - you'll need to add it.

## Fallback Behavior

The system uses a two-tier approach:

1. **Primary**: ML-based intent classification (if model available)
2. **Fallback**: Keyword matching (always available)

If the ML model fails or isn't available, it automatically falls back to keyword matching using the `example_phrases` column.

## Troubleshooting

### Model Won't Load

- Check internet connection (first-time download)
- Check disk space (~500 MB needed)
- Check memory (1-2 GB RAM needed)
- Falls back to keyword matching automatically

### Low Confidence Scores

- Adjust threshold in `intent_command_handler.py`:
  ```python
  if confidence > 0.3:  # Lower = more permissive
  ```
- Add more example phrases to CSV
- System falls back to keyword matching if confidence too low

### Slow First Classification

- First classification downloads/loads model (10-30 seconds)
- Subsequent classifications are fast (<200ms)
- Model stays in memory until process exits

## Performance Tips

1. **Use keyword-only mode** on very low-resource systems:
   ```python
   IntentCommandHandler(use_intent_model=False)
   ```

2. **Preload model** by calling classifier once at startup:
   ```python
   handler._get_classifier()  # Preloads model
   ```

3. **Use smaller model** for Raspberry Pi:
   - Modify code to use `typeform/distilbert-base-uncased-finetuned-squad2`
   - Or use keyword-only mode

## Next Steps

1. Read `INTENT_SETUP.md` for detailed setup instructions
2. Read `MIGRATION_TO_INTENT.md` to migrate from LLM handler
3. Edit `intent_mappings.csv` to customize responses
4. Test with various voice commands


