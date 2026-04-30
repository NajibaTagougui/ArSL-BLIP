"""
Cross-Attention Visualization

Generates heatmaps showing which image regions the model focuses on
when predicting each Arabic letter.

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Optional


def attention_to_heatmap(
    attention: np.ndarray,
    image_size: tuple = (224, 224),
) -> np.ndarray:
    """Convert raw attention weights to a resized heatmap (H×W, float 0-1)."""
    # attention shape: (num_heads, seq_len) or (seq_len,)
    if attention.ndim == 2:
        attn = attention.mean(axis=0)
    else:
        attn = attention

    # Reshape to square grid
    n = int(np.sqrt(attn.shape[-1]))
    if n * n != attn.shape[-1]:
        # Trim to nearest square
        attn = attn[: n * n]
    heatmap = attn.reshape(n, n)

    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Resize to image_size using PIL
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        image_size, Image.BILINEAR
    )
    return np.array(heatmap_img) / 255.0


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a heatmap on an image.

    Args:
        image: Original PIL Image.
        heatmap: 2D float array normalized to [0, 1].
        alpha: Blend factor (0 = image only, 1 = heatmap only).
        colormap: Matplotlib colormap name.

    Returns:
        Blended PIL Image.
    """
    # Convert image to RGB numpy
    img_arr = np.array(image.convert("RGB").resize((224, 224))) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap)[..., :3]  # Drop alpha channel

    # Blend
    blended = (1 - alpha) * img_arr + alpha * colored
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def save_attention_figure(
    image: Image.Image,
    attention: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Save a side-by-side figure: original image | attention overlay."""
    heatmap = attention_to_heatmap(attention, image_size=(224, 224))
    overlay = overlay_heatmap(image, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.resize((224, 224)))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Cross-Attention Heatmap")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention figure → {output_path}")
