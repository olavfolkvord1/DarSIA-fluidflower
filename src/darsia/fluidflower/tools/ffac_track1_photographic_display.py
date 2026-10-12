"""Fixed photographic presentation of corrected Track1 camera-linear RAW.

This transfer is display-only. The scientific log-camera RGB array and its
source validity are never changed by rendering a photograph.
"""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np


CONTRACT = "ffac-track1-fixed-photographic-display-v1"
DISPLAY_MODES = ("raw_log", "photographic_v1")
MATRIX = np.array([
    [0.00029792297036576425, -5.479831037268461e-05, 4.557770855244091e-06],
    [-4.7784972644298926e-05, 0.0001882330901444948, -5.8189790197807505e-05],
    [-9.052150215971802e-05, -0.00015616390509033662, 0.00045598006872025245],
], dtype=np.float64)
MATRIX.setflags(write=False)
_MATRIX_SHA256 = sha256(json.dumps(MATRIX.tolist(), separators=(",", ":")).encode()).hexdigest()


def validate_display_mode(mode: str) -> str:
    if mode not in DISPLAY_MODES:
        raise ValueError(f"Unknown Track1 display mode: {mode!r}")
    return mode


def display_contract() -> dict:
    """Serializable fixed transfer, independent of run, frame and sand mask."""
    return dict(
        contract=CONTRACT,
        matrix_rows_camera_rgb_cols_linear_srgb=MATRIX.tolist(),
        matrix_definition_sha256=_MATRIX_SHA256,
        matrix_source_json_path=(r"C:\Users\olav_\Documents\Claude\Projects\FF Albus"
                                r"\ffac_track1_ac28_display_check_20260930"
                                r"\results\display_matrix_v1.json"),
        matrix_source_json_sha256="354ecdecba97bb42375acba7fd220b244dd4f5137981ea99eea90d45896b8f21",
        same_for_all_runs_and_frames=True,
        definition="round(255*sRGB_OETF(clip(exp(L).T @ M,0,1)))",
        sensor_invalid_rgb8=[128, 128, 128],
        display_only_never_features=True,
        raw_detail_display="existing-fixed-log-camera-channel-view",
        fit_source="Claude display_matrix_v1.json; AC28 frames 0 and 14",
    )


def photographic_display(log_camera_rgb: np.ndarray, *, tile_rows: int = 128) -> np.ndarray:
    """Return full-frame HWC float32 sRGB quantized to the fixed 8-bit view.

    CHW log-camera RGB remains float32 and untouched. Sensor-invalid pixels
    become middle grey; a model-domain mask must never be passed here.
    Tiling bounds transient float64 RGB allocations on CPU-only machines.
    """
    signal = np.asarray(log_camera_rgb)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError("Photographic display requires CHW float32 log-camera RGB")
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    _, height, width = signal.shape
    output = np.empty((height, width, 3), dtype=np.float32)
    for start in range(0, height, tile_rows):
        end = min(start + tile_rows, height)
        tile = np.moveaxis(signal[:, start:end], 0, -1)
        valid = np.isfinite(tile).all(axis=-1)
        # Replace invalid channels only in the transient rendering array.
        linear = np.exp(np.where(valid[..., None], tile, 0).astype(np.float64))
        srgb_linear = np.clip(linear @ MATRIX, 0, 1)
        encoded = np.where(srgb_linear <= 0.0031308,
                           12.92 * srgb_linear,
                           1.055 * np.power(srgb_linear, 1 / 2.4) - 0.055)
        encoded = np.rint(255 * encoded) / 255
        encoded[~valid] = 128 / 255
        output[start:end] = encoded.astype(np.float32)
    return output


__all__ = ["CONTRACT", "MATRIX", "display_contract", "photographic_display",
           "validate_display_mode"]
