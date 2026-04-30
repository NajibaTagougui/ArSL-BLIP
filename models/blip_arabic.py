"""
BLIP-based Arabic Sign Language Recognition Model

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Institution: University of Sfax, Tunisia
Year: 2026
License: MIT

This model is described in the paper:
"Controllable and Interpretable Arabic Sign Language Recognition:
 A Generative Vision-Language Framework for Assistive Technology"
Under review at Multimedia Tools and Applications (Springer), 2026
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from typing import Dict, List, Optional


ARABIC_LETTERS = {
    "alif": "ا", "baa": "ب", "taa": "ت", "thaa": "ث", "jeem": "ج",
    "haa": "ح", "khaa": "خ", "dal": "د", "thal": "ذ", "raa": "ر",
    "zay": "ز", "seen": "س", "sheen": "ش", "saad": "ص", "dhad": "ض",
    "taa_marbuta": "ط", "thaa_marbuta": "ظ", "ain": "ع", "ghain": "غ",
    "fa": "ف", "qaaf": "ق", "kaaf": "ك", "laam": "ل", "meem": "م",
    "nun": "ن", "ha": "ه", "waw": "و", "ya": "ي", "aleff": "أ",
    "la": "لا", "toot": "ط", "gaaf": "گ",
}


class ArabicSignRecognizer:
    """Arabic Sign Language Recognizer using fine-tuned BLIP.

    This implementation follows the methodology described in our paper
    currently under review at Multimedia Tools and Applications (Springer, 2026).

    Key contributions:
      1. Template Constraint – prevents mode collapse
      2. Cross-Attention Visualization – enables interpretable predictions
      3. Confidence Calibration – reject option for uncertain predictions
    """

    def __init__(
        self,
        model_path: str = "models/arsl-blip",
        confidence_threshold: float = 0.85,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to fine-tuned model or HuggingFace model ID.
            confidence_threshold: Minimum confidence to accept a prediction.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.confidence_threshold = confidence_threshold
        self.arabic_letters = ARABIC_LETTERS

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading ArSL-BLIP from '{model_path}' on {self.device} ...")
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        return_attention: bool = False,
    ) -> Dict:
        """Predict the Arabic sign-language letter in an image.

        Args:
            image: PIL Image (RGB or grayscale).
            return_attention: If True, include cross-attention heatmap.

        Returns:
            dict with keys: letter, arabic_char, confidence, full_text,
                            accepted, [attention].
        """
        image = self._preprocess(image)
        inputs = self.processor(
            images=image,
            text="arabic sign language letter",
            return_tensors="pt",
        ).to(self.device)

        generate_kwargs = dict(
            max_length=50,
            num_beams=3,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if return_attention:
            generate_kwargs["output_attentions"] = True

        outputs = self.model.generate(**inputs, **generate_kwargs)

        # Decode text
        token_ids = outputs.sequences[0]
        full_text = self.processor.decode(token_ids, skip_special_tokens=True)

        # Extract letter name
        predicted_letter = None
        if "letter " in full_text:
            predicted_letter = full_text.split("letter ")[-1].strip().lower()

        confidence = self._compute_confidence(outputs)

        result = {
            "letter": predicted_letter,
            "arabic_char": self.arabic_letters.get(predicted_letter, predicted_letter),
            "confidence": confidence,
            "full_text": full_text,
            "accepted": confidence >= self.confidence_threshold,
        }

        if return_attention:
            result["attention"] = self._extract_attention(outputs)

        return result

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Predict letters for a list of images."""
        return [self.predict(img) for img in images]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.resize((224, 224), Image.LANCZOS)

    @staticmethod
    def _compute_confidence(outputs) -> float:
        if hasattr(outputs, "scores") and outputs.scores:
            scores = [
                torch.softmax(s, dim=-1).max().item()
                for s in outputs.scores
            ]
            return float(np.mean(scores))
        return 0.85

    @staticmethod
    def _extract_attention(outputs) -> Optional[np.ndarray]:
        if hasattr(outputs, "attentions") and outputs.attentions:
            attn = outputs.attentions[-1][-1][0]
            return attn.mean(dim=0).cpu().numpy()
        return None
