# API Reference

## ArabicSignRecognizer

```python
class ArabicSignRecognizer:
    def __init__(
        self,
        model_path: str = "models/arsl-blip",
        confidence_threshold: float = 0.85,
        device: str = None
    )
```

### Methods

#### `predict(image, return_attention=False) -> dict`
Predict the Arabic letter in an image.

**Args:**
- `image` (PIL.Image): Input image (any mode, will be converted to RGB 224x224)
- `return_attention` (bool): If True, include cross-attention heatmap

**Returns:** dict with keys `letter`, `arabic_char`, `confidence`, `full_text`, `accepted`, optionally `attention`

#### `predict_batch(images) -> list[dict]`
Predict letters for a list of images.

---

## REST API (Web Demo)

### POST /predict

Predict from an uploaded image.

**Request:** `multipart/form-data`
- `image` (file): Image file
- `attention` (str): "true" to include heatmap

**Response:**
```json
{
  "letter": "nun",
  "arabic_char": "ن",
  "confidence": 0.934,
  "accepted": true,
  "full_text": "arabic sign language letter nun",
  "attention_image": "<base64 PNG>"
}
```

### GET /health

Returns `{"status": "ok", "model": "..."}`.

---

## Attention Visualization

```python
from models.attention_viz import attention_to_heatmap, overlay_heatmap, save_attention_figure

# Convert attention weights to heatmap
heatmap = attention_to_heatmap(result["attention"])

# Overlay on image
overlay = overlay_heatmap(image, heatmap, alpha=0.5, colormap="jet")

# Save side-by-side figure
save_attention_figure(image, result["attention"], "attention.png", title="Predicted: Nun")
```

---

## Confidence Calibration

```python
from inference.confidence import ConfidenceCalibrator, apply_reject_option

cal = ConfidenceCalibrator()
cal.fit(val_confidences, val_correct_flags)

# Calibrate a score
calibrated = cal.calibrate(0.78)

# Apply reject option to a list of results
results = apply_reject_option(results, threshold=0.85)
```
