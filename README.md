# ArSL-BLIP: Arabic Sign Language Recognition with BLIP

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Springer-orange.svg)](https://link.springer.com/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-yellow.svg)]()

A generative vision-language framework for Arabic Sign Language (ArSL) recognition using fine-tuned BLIP (Bootstrapping Language-Image Pre-training). This model achieves **92.19% accuracy** on 32 Arabic letters with interpretable cross-attention visualization and confidence calibration.

> **Note**: This paper is currently under review at *Multimedia Tools and Applications* (Springer), 2026.

---

## 📋 Overview

This repository contains the code, models, and documentation for our paper:

**"Controllable and Interpretable Arabic Sign Language Recognition: A Generative Vision-Language Framework for Assistive Technology"**  
*Under review at Multimedia Tools and Applications (Springer), 2026*

**Authors**: Najiba Tagougui\*, Ansar Hani, Monji Kherallah  
(\*Corresponding Author — University of Sfax, Tunisia)

### Key Features

- ✅ **Generative Recognition**: Outputs natural language descriptions instead of class labels
- ✅ **92.19% Accuracy** on 32 Arabic letters (ArASL dataset)
- ✅ **Real-time Performance**: 47 ms per frame (21 FPS)
- ✅ **Interpretable**: Cross-attention visualization shows what the model focuses on
- ✅ **Confidence Calibration**: Reject option for uncertain predictions (94.1% effective accuracy)
- ✅ **Web Demo**: Interactive web interface for testing

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/NajibaTagougui/ArSL-BLIP.git
cd ArSL-BLIP

# Create conda environment (recommended)
conda create -n arsl python=3.10
conda activate arsl

# Install dependencies
pip install -r requirements.txt
```

### Download Model

```bash
# Download fine-tuned model weights
python scripts/download_model.py

# Or use Hugging Face (coming soon)
# from transformers import BlipProcessor, BlipForConditionalGeneration
# processor = BlipProcessor.from_pretrained("NajibaTagougui/arsl-blip")
# model = BlipForConditionalGeneration.from_pretrained("NajibaTagougui/arsl-blip")
```

### Run Inference

```bash
# Single image inference
python inference/predict.py --image path/to/image.jpg

# Real-time webcam
python inference/real_time.py

# Web demo
python demo/app.py
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Overall Accuracy | 92.19% |
| Effective Accuracy (τ=0.85) | 94.1% |
| Inference Time | 47 ms/image |
| Throughput | 21 FPS |
| Model Size | 946 MB |
| GPU Memory | 7.8 GB |

### Per-Letter Performance (Selected Letters)

| Letter | Arabic | Confidence |
|---|---|---|
| Nun | ن | 89.43% |
| Waw | و | 94.60% |
| La | لا | 98.72% |

---

## 🏗️ Architecture

The model uses BLIP (ViT-B/16 vision encoder + BERT-base text encoder) with three novel contributions:

1. **Template Constraint**: Prevents mode collapse (22.1% → 92.19%)
2. **Cross-Attention Visualization**: Enables interpretable predictions
3. **Confidence Calibration**: Reject option for uncertain predictions

```
Input Image → Vision Encoder (ViT-B/16) → Cross-Attention → Text Decoder → "arabic sign language letter X"
                                                                              ↓
                                                           Attention Maps + Confidence Score
```

---

## 🎯 Usage Examples

### Python API

```python
from models.blip_arabic import ArabicSignRecognizer

# Initialize recognizer
recognizer = ArabicSignRecognizer(model_path="models/arsl-blip")

# Recognize single image
result = recognizer.predict("images/sign.jpg")
print(f"Predicted: {result['letter']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Status: {'Accepted' if result['accepted'] else 'Rejected'}")

# Get attention visualization
result = recognizer.predict("images/sign.jpg", return_attention=True)
attention_map = result['attention']  # Heatmap of attended regions
```

### Command Line

```bash
# Single image
python inference/predict.py --image sign.jpg

# Batch processing
python inference/predict.py --folder ./test_images/ --output results.json

# Interactive mode
python inference/predict.py --interactive
```

---

## 🖥️ Web Demo

```bash
# Launch web interface
python demo/app.py
# Open browser to http://localhost:5000
```

Features:
- Upload images via drag-and-drop
- Real-time prediction with confidence bar
- Attention visualization overlay
- Batch processing for folders

---

## 📁 Dataset

The model is trained on the **ArASL dataset**:

- 32 Arabic letters
- 54,049 original images (108,098 augmented)
- 200 images per class (sampled for training)
- 85% / 15% train/validation split

```bash
# Download from Mendeley Data
python scripts/prepare_dataset.py --source /path/to/ArASL
```

---

## 🧪 Training

```bash
# Prepare configuration
cp configs/train_config.yaml my_config.yaml

# Start training
python training/train.py --config my_config.yaml

# Resume training
python training/train.py --resume checkpoints/latest.pt
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 5e-5 |
| Batch size | 16 (effective) |
| Epochs | 3 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |

---

## 📈 Evaluation

```bash
# Evaluate on test set
python evaluation/evaluate.py --checkpoint models/arsl-blip

# Generate confusion matrix
python evaluation/confusion_analysis.py

# Run ablation studies
python scripts/run_experiments.py --ablation
```

---

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.5.1+
- CUDA 12.1 (recommended)
- 8+ GB GPU memory
- 32+ GB RAM

See `requirements.txt` for the complete list.

---

## 📖 Citation

If you use this code or model in your research, please cite:

```bibtex
@article{tagougui2026arslblip,
  title={Controllable and Interpretable Arabic Sign Language Recognition: A Generative Vision-Language Framework for Assistive Technology},
  author={Tagougui, Najiba and Hani, Ansar and Kherallah, Monji},
  journal={Multimedia Tools and Applications},
  year={2026},
  note={Under review},
  publisher={Springer}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Corresponding Author**: Najiba Tagougui — tag.najiba@gmail.com  
**GitHub Issues**: [Open an issue](https://github.com/NajibaTagougui/ArSL-BLIP/issues)

## 🙏 Acknowledgments

- The ArASL dataset creators
- Salesforce Research for the BLIP model
- *Multimedia Tools and Applications* (Springer)

---

⭐ If you find this work useful, please consider giving us a star!
