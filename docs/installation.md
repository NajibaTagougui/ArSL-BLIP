# Installation

## Requirements

- Python 3.10+
- PyTorch 2.5.1+ (with CUDA 12.1 recommended)
- 8+ GB GPU memory for inference, 16+ GB for training
- 32+ GB RAM

## Steps

### 1. Clone the repository

```bash
git clone https://github.com/NajibaTagougui/ArSL-BLIP.git
cd ArSL-BLIP
```

### 2. Create a virtual environment (recommended)

Using conda:
```bash
conda create -n arsl python=3.10
conda activate arsl
```

Or using venv:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3. Install PyTorch

Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the model

```bash
python scripts/download_model.py
```

> **Note**: The Hugging Face model will be made publicly available upon paper acceptance.

### 6. Verify installation

```bash
python -c "from models.blip_arabic import ArabicSignRecognizer; print('OK')"
```

## CPU-only Installation

If you don't have a GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Inference will be slower (~500 ms/image vs 47 ms on GPU).
