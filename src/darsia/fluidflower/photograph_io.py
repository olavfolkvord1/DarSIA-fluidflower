"""Photograph decoding for retained annotation and display surfaces."""
from pathlib import Path
import numpy as np


def load_reduced(path: Path, factor: int) -> np.ndarray:
    """Read a photograph at 1/factor resolution as float32 RGB in [0, 1].

    JPEG files are decoded at reduced size directly (DCT scaling).
    """

    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        if factor > 1:
            image.draft("RGB", (width // factor, height // factor))
        image = image.convert("RGB")
        target = (max(1, width // factor), max(1, height // factor))
        if image.size != target:
            image = image.resize(target, Image.BOX)
        return np.asarray(image, dtype=np.float32) / 255.0
