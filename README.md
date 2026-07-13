# ArSL-BLIP: Arabic Sign Language Recognition with BLIP




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



### Run Inference

```bash
# Single image inference
python inference/predict.py --image path/to/image.jpg

# Real-time webcam
python inference/real_time.py

# Web demo
python demo/app.py
```

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

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.5.1+
- CUDA 12.1 (recommended)
- 8+ GB GPU memory
- 32+ GB RAM

See `requirements.txt` for the complete list.

