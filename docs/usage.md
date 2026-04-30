# Usage Guide

## Python API

```python
from models.blip_arabic import ArabicSignRecognizer
from PIL import Image

# Load model
recognizer = ArabicSignRecognizer(
    model_path="models/arsl-blip",
    confidence_threshold=0.85,   # Reject predictions below this
    device="cuda",               # or "cpu"
)

# Single image
image = Image.open("sign.jpg")
result = recognizer.predict(image)

print(result["letter"])       # e.g. "nun"
print(result["arabic_char"]) # e.g. "ن"
print(result["confidence"])  # e.g. 0.934
print(result["accepted"])    # True if confidence >= threshold

# With attention heatmap
result = recognizer.predict(image, return_attention=True)
attention = result["attention"]   # numpy array
```

## Command Line

```bash
# Single image
python inference/predict.py --image sign.jpg

# Batch folder
python inference/predict.py --folder ./images/ --output results.json

# Interactive mode
python inference/predict.py --interactive

# Custom threshold
python inference/predict.py --image sign.jpg --threshold 0.90
```

## Web Demo

```bash
python demo/app.py
# Open http://localhost:5000
```

## Real-Time Webcam

```bash
python inference/real_time.py
# Press Q to quit
```

## Result Dictionary

| Field | Type | Description |
|---|---|---|
| `letter` | str | Predicted letter name (English) |
| `arabic_char` | str | Corresponding Arabic character |
| `confidence` | float | Confidence score 0–1 |
| `full_text` | str | Full generated text |
| `accepted` | bool | Whether confidence >= threshold |
| `attention` | ndarray | Cross-attention heatmap (if requested) |
