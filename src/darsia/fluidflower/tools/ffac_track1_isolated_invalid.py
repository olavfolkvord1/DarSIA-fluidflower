"""Opt-in local repair of isolated invalid Track1 scientific pixels on CPU."""
from __future__ import annotations

from hashlib import sha256

import cv2
import numpy as np

POLICY = "fill_isolated_invalid_v1"
CONTRACT = "transported_safe_B_sand_core_enclosed_filled_v1_isolated_invalid_filled_v1"
MAX_AREA_PX = 50


def validate_policy(policy: str) -> str:
    if policy not in ("keep", POLICY):
        raise ValueError(f"Unknown isolated-invalid policy: {policy!r}")
    return policy


def policy_contract() -> dict:
    return dict(contract=CONTRACT, policy=POLICY, maximum_component_area_px=MAX_AREA_PX,
                connectivity=8, neighbourhood_size_px=5,
                neighbourhood="all-channel finite pixels inside photo-sand support",
                interpolation="iterative per-channel median in log space",
                outside_photo_sand_filled=False, fully_enclosed_by_valid_pixels=True,
                hard_floor_applied=False, original_finite_channels_preserved=True)


def fill_isolated_invalid(log_rgb: np.ndarray, safe_filled: np.ndarray):
    """Return a copy with enclosed <=50-pixel NaN components locally filled.

    Neighbours are complete finite RGB pixels within the fixed sand domain.
    Each sweep is simultaneous; repaired pixels can support the next sweep.
    Finite channels and every non-filled pixel retain their original bits.
    """
    signal = np.asarray(log_rgb)
    safe = np.asarray(safe_filled)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError("Isolated-invalid repair needs CHW float32 log RGB")
    if safe.dtype != np.dtype(bool) or safe.shape != signal.shape[1:]:
        raise TypeError("Isolated-invalid repair needs a native boolean photo-sand mask")
    if np.isinf(signal).any():
        raise ValueError("Infinite scientific values must not be repaired")
    result = signal.copy()
    valid = np.isfinite(signal).all(axis=0) & safe
    invalid = ~np.isfinite(signal).all(axis=0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        invalid.astype(np.uint8), connectivity=8)
    filled = np.zeros(safe.shape, bool)
    filled_components = largest = maximum_sweeps = 0
    height, width = safe.shape
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area > MAX_AREA_PX or x == 0 or y == 0 or x+w == width or y+h == height:
            continue
        component = labels[y:y+h, x:x+w] == label
        if not safe[y:y+h, x:x+w][component].all():
            continue
        patch = labels[y-1:y+h+1, x-1:x+w+1] == label
        ring = cv2.dilate(patch.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~patch
        if not valid[y-1:y+h+1, x-1:x+w+1][ring].all():
            continue
        rows, cols = np.nonzero(component)
        pending = list(zip(rows+y, cols+x))
        sweeps = 0
        while pending:
            updates = []
            for row, col in pending:
                y0, y1, x0, x1 = max(0,row-2), min(height,row+3), max(0,col-2), min(width,col+3)
                donors = valid[y0:y1, x0:x1]
                if donors.any():
                    median = np.median(result[:, y0:y1, x0:x1][:, donors], axis=1)
                    updates.append((row, col, median))
            if not updates:
                raise ValueError("Enclosed isolated-invalid component has no finite local donor")
            updated = set()
            for row, col, median in updates:
                absent = ~np.isfinite(result[:, row, col])
                result[absent, row, col] = median[absent]
                valid[row, col] = True
                filled[row, col] = True
                updated.add((row, col))
            pending = [pixel for pixel in pending if pixel not in updated]
            sweeps += 1
        filled_components += 1
        largest = max(largest, area)
        maximum_sweeps = max(maximum_sweeps, sweeps)
    metadata = dict(policy_contract(), filled_pixel_count=int(filled.sum()),
                    filled_component_count=filled_components,
                    largest_filled_component_px=largest, maximum_fill_sweeps=maximum_sweeps,
                    filled_mask_packbits_sha256=sha256(np.packbits(filled, bitorder="little").tobytes()).hexdigest())
    return result, filled, metadata
