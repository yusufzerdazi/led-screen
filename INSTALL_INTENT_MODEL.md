# Installing the Intent Classification Model

The intent classification system uses the `facebook/bart-large-mnli` model from Hugging Face. **The model must be pre-installed** before running the code - it will NOT auto-download during runtime.

## Required: Pre-install the Model

**You must install the model before running the application:**

```bash
python3 -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"
```

This will download the model (~1.6GB) and cache it in `~/.cache/huggingface/`. The model will then be available instantly on subsequent runs.

**Important:** The model is preloaded at startup to avoid any dynamic installation delays. If the model is not found, the application will log a warning but continue (intent classification will fail until the model is installed).

### Option 2: Preload in your code

You can also preload the model in your code before first use:

```python
from intent_command_handler import IntentCommandHandler

handler = IntentCommandHandler(intent_csv_path="client/intent_mappings.csv")
handler.preload_model()  # Download/load model now
```

### Option 3: Let it auto-download

If you don't pre-install, the model will automatically download on first use. This may cause a delay (1-2 minutes) on the first run.

## Requirements

Make sure you have the required packages installed:

```bash
pip install transformers torch
```

## Model Location

After installation, the model is cached at:
- Linux/Mac: `~/.cache/huggingface/hub/models--facebook--bart-large-mnli/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\models--facebook--bart-large-mnli/`

## Verification

To verify the model is installed and working:

```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier("hello there", candidate_labels=["greeting", "goodbye", "ask_name"])
print(result)  # Should show classification results
```

