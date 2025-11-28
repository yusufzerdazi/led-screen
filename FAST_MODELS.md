# Faster LLM Models for Raspberry Pi 5

TinyLlama is taking ~200 seconds which is too slow. Here are faster alternatives:

## Recommended Models (Fastest First)

### 1. **phi3:mini** (3.8B parameters, quantized)
- **Speed**: ~5-15 seconds on Pi 5
- **Quality**: Excellent for structured JSON output
- **Install**: `ollama pull phi3:mini` ⚠️ Note: Use colon `:`, not hyphen `-`
- **Best for**: Structured responses, JSON generation

### 2. **qwen2.5-coder:0.5b** (0.5B parameters)
- **Speed**: ~2-5 seconds on Pi 5  
- **Quality**: Good for simple structured outputs
- **Install**: `ollama pull qwen2.5-coder:0.5b`
- **Best for**: Fastest option, simple commands

### 3. **gemma2:2b** (2B parameters)
- **Speed**: ~5-10 seconds on Pi 5
- **Quality**: Good balance
- **Install**: `ollama pull gemma2:2b`
- **Best for**: General purpose, balanced

### 4. **llama3.2:1b** (1B parameters) - Already Installed!
- **Speed**: ~10-20 seconds on Pi 5
- **Quality**: Better than TinyLlama
- **Install**: Already installed
- **Best for**: Try this first - you already have it!

### 5. **phi-2** (2.7B parameters)
- **Speed**: ~8-15 seconds on Pi 5
- **Quality**: Very good for instruction following
- **Install**: `ollama pull phi-2`
- **Best for**: Instruction following, JSON

## Quick Test

Test a model's speed:
```bash
time ollama run phi3:mini "Generate JSON: {\"test\": \"value\"}"
```

## Configuration

Update `client/modes/decompression_mode.py`:
```python
model_name="phi3:mini"  # or qwen2.5-coder:0.5b, gemma2:2b, etc.
```

## Optimizations Already Applied

1. **Response length limit**: Limited to 256 tokens (enough for JSON)
2. **Shorter prompts**: Reduced system prompt size
3. **Temperature**: Lowered for faster, more deterministic responses
4. **Timeout**: 60 second timeout (should be plenty)

## Expected Performance

- **qwen2.5-coder:0.5b**: 2-5 seconds ⚡ Fastest (already installed!)
- **phi3:mini**: 5-15 seconds ⚡⚡ Very fast (use colon `:` not hyphen `-`)
- **gemma2:2b**: 5-10 seconds ⚡⚡ Fast
- **llama3.2:1b**: 10-20 seconds ⚡ Good (already installed!)
- **tinyllama**: 200 seconds ❌ Too slow

Try `qwen2.5-coder:0.5b` (already installed) or `phi3:mini` for best speed!

**Note**: Ollama model names use colons (`:`) for tags, not hyphens (`-`). Common mistake!

