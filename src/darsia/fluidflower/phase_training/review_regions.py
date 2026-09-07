"""Native-resolution review-region loading and deterministic pixel sampling."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def load_review_rgb(path: Path, content_box: Sequence[int]) -> np.ndarray:
    """Load the registered review content as float32 RGB without resampling."""

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    x, y, width, height = (int(value) for value in content_box)
    cropped = rgb[y : y + height, x : x + width]
    if cropped.shape[:2] != (height, width):
        raise ValueError(
            f"Content crop {content_box} falls outside {path}: {rgb.shape}"
        )
    return np.asarray(cropped, dtype=np.float32)


def annotation_mask(
    annotation: Mapping[str, Any],
    *,
    canvas_wh: Sequence[int],
    content_box: Sequence[int],
) -> np.ndarray:
    """Rasterize a normalized review box or polygon in content coordinates."""

    canvas_width, canvas_height = (float(value) for value in canvas_wh)
    content_x, content_y, width, height = (int(value) for value in content_box)
    points = annotation.get("points") or []
    if not points:
        x = float(annotation["x"])
        y = float(annotation["y"])
        box_width = float(annotation["width"])
        box_height = float(annotation["height"])
        points = (
            (x, y),
            (x + box_width, y),
            (x + box_width, y + box_height),
            (x, y + box_height),
        )
    polygon = np.asarray(
        [
            (
                float(point[0]) * canvas_width - content_x,
                float(point[1]) * canvas_height - content_y,
            )
            for point in points
        ],
        dtype=np.float32,
    )
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    integer_polygon = np.rint(polygon).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [integer_polygon], color=1)
    return mask.astype(bool)


def sample_mask(
    mask: np.ndarray,
    *,
    limit: int,
    annotation_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample pixels reproducibly so repeated training uses identical evidence."""

    rows, columns = np.nonzero(mask)
    if not rows.size:
        raise ValueError(f"Annotation {annotation_id} has no in-frame pixels")
    if rows.size > limit:
        seed = int.from_bytes(
            hashlib.sha256(annotation_id.encode("utf-8")).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(rows.size, size=limit, replace=False))
        rows = rows[selected]
        columns = columns[selected]
    return rows, columns


# Private aliases retain compatibility with the original campaign trainer.
_annotation_mask = annotation_mask
_load_rgb = load_review_rgb
_sample_mask = sample_mask


__all__ = ["annotation_mask", "load_review_rgb", "sample_mask"]
