# Migration Guide: LLM to Intent-Based System

## Quick Migration Steps

### 1. Install Dependencies

```bash
pip install transformers torch
```

### 2. Update decompression_mode.py

Replace the LLM handler import and initialization:

**Before:**
```python
from llm_command_handler import LLMCommandHandler
...
self.llm_command_handler = LLMCommandHandler(
    state_manager=self.state_manager,
    llm_api_url="http://localhost:11434",
    model_name="gemma2:2b",
    use_llm=True
)
self.llm_command_handler.set_status_change_callback(self._handle_voice_status_change)
self.llm_command_handler.set_tts_callback(self._handle_voice_tts)
```

**After:**
```python
from intent_command_handler import IntentCommandHandler
...
self.intent_command_handler = IntentCommandHandler(
    intent_csv_path="client/intent_mappings.csv",
    use_intent_model=True  # Set to False to use keyword matching only
)
self.intent_command_handler.set_status_change_callback(self._handle_voice_status_change)
self.intent_command_handler.set_tts_callback(self._handle_voice_tts)
```

### 3. Update All References

Replace all `self.llm_command_handler` with `self.intent_command_handler`:

```bash
# In decompression_mode.py, replace:
sed -i 's/self\.llm_command_handler/self.intent_command_handler/g' client/modes/decompression_mode.py
```

Or manually update:
- Line 22: Import statement
- Line 330-337: Initialization
- Line 341: ConfigManager (may need adjustment)
- Line 738: TTS output device
- Line 797, 831, 853: Gesture triggers
- Line 994: Gesture triggers list
- Line 1003: TTS triggers
- Line 1107: Reload config
- Line 1124: Process voice command
- Line 1184: Pending command callback (not needed for intent handler)
- Line 1779: Execute response

### 4. Remove Think Status Logic (Optional)

Since intent classification is instant, you can remove the "think" status delay logic:

**Remove or simplify:**
- Lines 1129-1194: Unknown command handling with think status
- Lines 1758-1787: Think status completion logic

**Simplified version:**
```python
def _handle_voice_command(self, text):
    """Handle voice commands using intent classification"""
    if not text:
        return
    
    self.logger.info(f"Voice command received: {text}")
    
    # Process command through intent handler (instant, no async needed)
    success, is_unknown = self.intent_command_handler.process_voice_command(text)
    
    if not success:
        self.logger.warning(f"Command not recognized: {text}")
```

### 5. Update ConfigManager (if used)

If `ConfigManager` is used, you may need to update it to work with the intent CSV format, or bypass it entirely since the intent handler manages its own CSV.

### 6. Update Console UI (if needed)

In `console_ui.py`, update references:

```python
# Replace:
if hasattr(self.mode, 'llm_command_handler') and self.mode.llm_command_handler:
    llm_handler = self.mode.llm_command_handler
    ...

# With:
if hasattr(self.mode, 'intent_command_handler') and self.mode.intent_command_handler:
    intent_handler = self.mode.intent_command_handler
    ...
```

## Key Differences

### LLM Handler
- ✅ Generates responses dynamically
- ❌ Slow (1-5 seconds)
- ❌ Requires Ollama/LLM server
- ❌ High resource usage
- ❌ Async callbacks needed

### Intent Handler
- ✅ Instant classification (<200ms)
- ✅ No external server needed
- ✅ Low resource usage
- ✅ Synchronous (no callbacks)
- ❌ Requires CSV setup
- ❌ Fixed set of intents

## Benefits After Migration

1. **10-100x faster** response times
2. **No Ollama dependency** - works offline
3. **Lower CPU/memory** usage
4. **Predictable responses** - easy to test and debug
5. **Easy to extend** - just add CSV rows

## Testing

After migration, test with:

```python
# Test voice commands
handler = IntentCommandHandler()
handler.set_status_change_callback(lambda s: print(f"Status: {s}"))
handler.set_tts_callback(lambda t: print(f"TTS: {t}"))

handler.process_voice_command("hello")
handler.process_voice_command("show me the eye")
handler.process_voice_command("show people")
```

## Rollback

If you need to rollback, simply revert the changes and restart with the LLM handler. The CSV files are independent and won't interfere.


