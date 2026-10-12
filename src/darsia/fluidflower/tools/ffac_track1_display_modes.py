"""Fixed photographic modes for corrected Track1 camera-linear RAW.

This is presentation only: neither the scientific log-camera input nor its
validity is modified. All runs and frames share the exact same colour matrix,
gain and tone curve. The separate scientific isolated-invalid fill is applied
by the source processor, never by this renderer.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import warnings

import cv2
import numpy as np


CONTRACT = "ffac-track1-display-modes-v1"
DISPLAY_MODE = "natural_modes_v1"
DISPLAY_MODES = ("natural", "lift1", "lift2")
CLAUDE_CONTRACT_SHA256 = "69e72fe789a5c455d343aaa3db17e92d78b262c0c3f6566a5e7729ace7ab01df"
CLAUDE_CONTRACT_PATH = (
    r"C:\Users\olav_\Documents\Claude\Projects\FF Albus"
    r"\ffac_track1_rerun_check_20260930\results\display_modes_v1_contract.json"
)
M1 = np.array([
    [0.00029792297036576425, -5.479831037268461e-05, 4.557770855244091e-06],
    [-4.7784972644298926e-05, 0.0001882330901444948, -5.8189790197807505e-05],
    [-9.052150215971802e-05, -0.00015616390509033662, 0.00045598006872025245],
], dtype=np.float64)
A = np.array([
    [0.7482367985193791, -0.046057072251059435, -0.0033417078351086617],
    [-0.016847507193594402, 0.7967817054734453, -0.05549901039949279],
    [0.08512591962169573, 0.19570119026229627, 1.17028321514431],
], dtype=np.float64)
# Keep the exact exported values, rather than the six-digit matrix in the note.
M_NATURAL = np.array([
    [0.00022422812891554153, -5.6491789785883e-05, 7.379563203284224e-06],
    [-4.387919269217012e-05, 0.0001407937073266365, -7.83856015712351e-05],
    [-2.6284823779787712e-05, -3.1023545081590044e-05, 0.0005425952594693075],
], dtype=np.float64)
Y_REFERENCE = 0.014062383957633609
GAINS = {"natural": 1.0, "lift1": 4.266701875070739, "lift2": 7.111169791784565}
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
for _constant in (M1, A, M_NATURAL, LUMINANCE_WEIGHTS):
    _constant.setflags(write=False)
_MATRIX_SHA256 = sha256(json.dumps(M_NATURAL.tolist(), separators=(",", ":")).encode()).hexdigest()


def display_contract() -> dict:
    """Return the immutable transfer and cleanup policy for source binding."""
    return {
        "contract": CONTRACT,
        "display_mode": DISPLAY_MODE,
        "default_mode": "natural",
        "modes": {
            name: {"gain": gain, "tone": "clip" if name == "natural" else "extended_reinhard_luminance",
                   "white": None if name == "natural" else 0.95 * gain}
            for name, gain in GAINS.items()
        },
        "matrix_rows_camera_rgb_cols_linear_srgb": M_NATURAL.tolist(),
        "matrix_definition_sha256": _MATRIX_SHA256,
        "matrix_definition": "M_natural=M1@A; linear_srgb=exp(log_camera_rgb).T@M_natural",
        "M1": M1.tolist(),
        "A": A.tolist(),
        "matrix_source_json_path": CLAUDE_CONTRACT_PATH,
        "matrix_source_json_sha256": CLAUDE_CONTRACT_SHA256,
        "renderer_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "sand_reference_luminance": Y_REFERENCE,
        "same_for_all_runs_and_frames": True,
        "adaptive_gain": False,
        "invalid_components": {
            "connectivity": 8, "max_fill_area_px": 64,
            "fill": "iterated_valid_5x5_channel_median_linear_srgb",
            "unfilled_rgb8": [40, 40, 40],
        },
        "highlights": {
            "scope": "outside_photo_sand_only",
            "preferred": "native_raw_any_channel_saturation_mask_to_max_linear_srgb",
            "fallback": "w=clip((min(linear_R,linear_B)-0.55)/0.25,0,1); blend_to_max_linear_srgb",
            "max_channel_threshold_forbidden": True,
        },
        "chroma": {"colour_space": "OpenCV_RGB8_YCrCb", "median_size_px": 5,
                   "luminance_filter": "none", "roundtrip_luminance_quantisation": "at_most_1_rgb8_unit"},
        "display_only_never_features": True,
        "raw_detail_display": "unchanged-existing-fixed-log-camera-channel-view",
    }


display_modes_contract = display_contract


def _fill_small_invalid(linear: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fill only small display holes; retain all valid pixels byte-for-byte.

    Sparse 5x5 neighbourhoods avoid three whole-frame inpainting passes. Each
    iteration uses the previous iteration's valid pixels, so filling is not
    dependent on pixel traversal order. Larger invalid regions are never used
    as interpolation neighbours.
    """
    invalid = ~valid
    invalid_count = int(np.count_nonzero(invalid))
    if not invalid_count:
        return invalid, {"invalid_input_pixels": 0, "filled_invalid_pixels": 0,
                         "largest_filled_component_px": 0, "unfilled_invalid_pixels": 0,
                         "fill_iterations": 0}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(invalid, dtype=np.uint8), connectivity=8)
    small_ids = np.flatnonzero((stats[:, cv2.CC_STAT_AREA] <= 64) & (np.arange(count) != 0))
    lookup = np.zeros(count, dtype=bool)
    lookup[small_ids] = True
    rows, cols = np.nonzero(lookup[labels])
    del lookup
    filled_mask = np.zeros(valid.shape, dtype=bool)
    height, width = valid.shape
    iterations = 0
    offsets = np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)], dtype=np.int64)
    # A connected component contains at most 64 pixels, so this also bounds the
    # longest possible propagation from a valid neighbourhood into that hole.
    while rows.size and iterations < 64:
        nr = rows[:, None] + offsets[None, :, 0]
        nc = cols[:, None] + offsets[None, :, 1]
        inside = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
        nr = np.clip(nr, 0, height - 1)
        nc = np.clip(nc, 0, width - 1)
        neighbours = linear[nr, nc].copy()
        neighbours[~(inside & valid[nr, nc])] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # no valid neighbour yet
            medians = np.nanmedian(neighbours, axis=1)
        ready = np.isfinite(medians).all(axis=-1)
        if not np.any(ready):
            break
        linear[rows[ready], cols[ready]] = medians[ready]
        valid[rows[ready], cols[ready]] = True
        filled_mask[rows[ready], cols[ready]] = True
        rows, cols = rows[~ready], cols[~ready]
        iterations += 1
    filled_ids = np.unique(labels[filled_mask])
    largest = int(stats[filled_ids, cv2.CC_STAT_AREA].max()) if filled_ids.size else 0
    return ~valid, {"invalid_input_pixels": invalid_count,
                    "filled_invalid_pixels": int(np.count_nonzero(filled_mask)),
                    "largest_filled_component_px": largest,
                    "unfilled_invalid_pixels": int(np.count_nonzero(~valid)),
                    "fill_iterations": iterations}


def _neutralise_highlights(linear: np.ndarray, photo_sand: np.ndarray,
                          valid: np.ndarray, raw_saturated_mask: np.ndarray | None) -> dict:
    """Neutralise white clipping outside sand without whitening yellow/orange."""
    outside = (~photo_sand) & valid
    if raw_saturated_mask is not None:
        selected = outside & raw_saturated_mask
        linear[selected] = np.max(linear[selected], axis=-1, keepdims=True)
        return {"policy": "native_raw_any_channel_saturation_mask",
                "pixels_affected": int(np.count_nonzero(selected)), "raw_saturation_mask_available": True}
    weight = np.clip((np.minimum(linear[..., 0], linear[..., 2]) - 0.55) / 0.25, 0, 1)
    selected = outside & (weight > 0)
    if np.any(selected):
        rgb = linear[selected]
        w = weight[selected, None]
        linear[selected] = rgb * (1 - w) + np.max(rgb, axis=-1, keepdims=True) * w
    return {"policy": "allowed_min_linear_RB_fallback",
            "pixels_affected": int(np.count_nonzero(selected)), "raw_saturation_mask_available": False,
            "limitation": "Frozen RAW renderer exposes saturation counts, not a registered per-pixel saturation mask"}


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.power(linear, 1 / 2.4) - 0.055)


def _median_chroma_rgb8(rgb8: np.ndarray) -> np.ndarray:
    """Median only the two 8-bit chroma planes; keep Y completely unfiltered."""
    ycrcb = cv2.cvtColor(np.ascontiguousarray(rgb8), cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 1] = cv2.medianBlur(ycrcb[..., 1], 5)
    ycrcb[..., 2] = cv2.medianBlur(ycrcb[..., 2], 5)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def render_display_modes(log_camera_rgb: np.ndarray, photo_sand: np.ndarray, *,
                         raw_saturated_mask: np.ndarray | None = None,
                         tile_rows: int = 128) -> tuple[dict[str, np.ndarray], dict]:
    """Return three full-frame HWC float32 photographs quantized to 8-bit.

    Input is full-frame CHW float32 log-camera RGB, not the sand-masked sidecar.
    ``photo_sand`` is used only to protect sand from highlight neutralisation.
    CPU tiling bounds the float64 transformation temporaries. The return value
    never enters scientific features or phase classification.
    """
    signal = np.asarray(log_camera_rgb)
    sand = np.asarray(photo_sand)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError("Display modes require CHW float32 log-camera RGB, never uint8")
    height, width = signal.shape[1:]
    if height < 1 or width < 1:
        raise ValueError("Display modes require a non-empty image")
    if sand.dtype != np.bool_ or sand.shape != (height, width):
        raise TypeError("Photo-sand mask must be native-shape boolean")
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    if raw_saturated_mask is not None:
        raw_saturated_mask = np.asarray(raw_saturated_mask)
        if raw_saturated_mask.dtype != np.bool_ or raw_saturated_mask.shape != sand.shape:
            raise TypeError("RAW saturation mask must be native-shape boolean")

    linear = np.empty((height, width, 3), dtype=np.float32)
    valid = np.empty((height, width), dtype=bool)
    input_nonfinite = 0
    transform_nonfinite = 0
    for start in range(0, height, tile_rows):
        end = min(start + tile_rows, height)
        tile = np.moveaxis(signal[:, start:end], 0, -1)
        tile_valid = np.isfinite(tile).all(axis=-1)
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = np.exp(np.where(tile_valid[..., None], tile, 0).astype(np.float64)) @ M_NATURAL
            transformed = transformed.astype(np.float32)
        transform_valid = np.isfinite(transformed).all(axis=-1)
        input_nonfinite += int(np.count_nonzero(~tile_valid))
        transform_nonfinite += int(np.count_nonzero(tile_valid & ~transform_valid))
        tile_valid &= transform_valid
        transformed[~tile_valid] = np.nan
        # Negative calibrated components have no photographic intensity. This
        # display clamp leaves the scientific camera/log array unchanged.
        linear[start:end] = np.maximum(transformed, 0)
        valid[start:end] = tile_valid
    unfilled, fill_metadata = _fill_small_invalid(linear, valid)
    highlight_metadata = _neutralise_highlights(linear, sand, valid, raw_saturated_mask)
    modes = {}
    for name, gain in GAINS.items():
        rgb8 = np.empty((height, width, 3), dtype=np.uint8)
        for start in range(0, height, tile_rows):
            end = min(start + tile_rows, height)
            rgb = np.where(valid[start:end, :, None], linear[start:end], 0).astype(np.float64)
            if name != "natural":
                luminance = rgb @ LUMINANCE_WEIGHTS
                gained = gain * luminance
                toned = gained * (1 + gained / (0.95 * gain) ** 2) / (1 + gained)
                ratio = np.divide(toned, luminance, out=np.zeros_like(toned), where=luminance > 0)
                rgb *= ratio[..., None]
                maximum = np.max(rgb, axis=-1, keepdims=True)
                rgb /= np.maximum(maximum, 1)
            rgb8[start:end] = np.rint(255 * _srgb_encode(rgb)).astype(np.uint8)
        rgb8[unfilled] = 40
        rgb8 = _median_chroma_rgb8(rgb8)
        # Chroma filtering must not tint the deliberately neutral invalid area.
        rgb8[unfilled] = 40
        modes[name] = rgb8.astype(np.float32) / np.float32(255)
    metadata = {
        "contract": display_contract(),
        "display_only": True,
        "input_shape_chw": list(signal.shape),
        "input_dtype": str(signal.dtype),
        "input_nonfinite_pixels": input_nonfinite,
        "transform_nonfinite_pixels": transform_nonfinite,
        "invalid_cleanup": fill_metadata,
        "highlight_cleanup": highlight_metadata,
        "mode_keys": list(DISPLAY_MODES),
        "output_dtype": "float32",
        "output_representation": "HWC_sRGB8_quantized_div255",
    }
    return modes, metadata


__all__ = ["CONTRACT", "DISPLAY_MODE", "DISPLAY_MODES", "CLAUDE_CONTRACT_SHA256",
           "M1", "A", "M_NATURAL", "Y_REFERENCE", "GAINS", "display_contract",
           "display_modes_contract", "render_display_modes"]
