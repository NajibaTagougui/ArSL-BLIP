"""
Flask Web Demo for ArSL-BLIP

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Usage:
    python demo/app.py
    Open http://localhost:5000
"""

import base64
import io
import os
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

from models.blip_arabic import ArabicSignRecognizer
from models.attention_viz import attention_to_heatmap, overlay_heatmap

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

MODEL_PATH = os.environ.get("ARSL_MODEL_PATH", "models/arsl-blip")
THRESHOLD = float(os.environ.get("ARSL_THRESHOLD", "0.85"))

recognizer: ArabicSignRecognizer = None


def get_recognizer() -> ArabicSignRecognizer:
    global recognizer
    if recognizer is None:
        recognizer = ArabicSignRecognizer(model_path=MODEL_PATH, confidence_threshold=THRESHOLD)
    return recognizer


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    return_attention = request.form.get("attention", "false").lower() == "true"

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    rec = get_recognizer()
    result = rec.predict(image, return_attention=return_attention)

    response = {
        "letter": result["letter"],
        "arabic_char": result["arabic_char"],
        "confidence": round(result["confidence"], 4),
        "accepted": result["accepted"],
        "full_text": result["full_text"],
    }

    # Attention overlay
    if return_attention and result.get("attention") is not None:
        heatmap = attention_to_heatmap(result["attention"])
        overlay = overlay_heatmap(image, heatmap)
        response["attention_image"] = pil_to_base64(overlay)

    return jsonify(response)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


def main():
    host = os.environ.get("ARSL_HOST", "0.0.0.0")
    port = int(os.environ.get("ARSL_PORT", "5000"))
    debug = os.environ.get("ARSL_DEBUG", "false").lower() == "true"

    print(f"Starting ArSL-BLIP demo at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
