"""Scientific CPU flat-field features for the FF_AC gas-within-CO2 head.

Only source-bound CHW float32 log-camera sidecars are accepted.  The colour
matrix, frame-specific baseline warp and linear 5x5 smoothing follow the
Track1 change view.  Scientific invalid pixels are never filled here.
Current frames are converted in haloed row tiles, so sparse training samples
and native inference do not require a full current RGB or feature tensor.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from .tools import ffac_track1_display_modes as display
from .tools.ffac_v5_temporal_alignment import _h


FEATURE_VERSION = "baseline-relative-bright-grain-v5+flatfield-v1"
FLATFIELD_ONLY_FEATURE_VERSION = "flatfield-gas-v1"
FLATFIELD_FEATURE_NAMES = ("ff_log_r", "ff_log_g", "ff_log_b", "ff_log_y")
BOX_SIZE = 5
DENOMINATOR_FLOOR = np.float32(1e-6)
CHANGE_DISPLAY_CONTRACT = "ffac-track1-display-change-flatfield-v1"
_BOX_KERNEL = np.ones((BOX_SIZE, BOX_SIZE), np.uint8)
_LUMINANCE = display.LUMINANCE_WEIGHTS.astype(np.float32)
_LUMINANCE.setflags(write=False)


@dataclass(frozen=True)
class FlatfieldReference:
    """Unsmoothed calibrated RGB on the pre-injection source grid."""

    linear_rgb: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class RegisteredFlatfieldReference:
    """Baseline linear RGB and luminance warped before box smoothing."""

    rgb_box5: np.ndarray
    luminance_box5: np.ndarray
    valid: np.ndarray
    box_valid: np.ndarray
    alignment: dict


def flatfield_contract() -> dict:
    """Return the versioned feature definition and exact source-code SHA."""
    return {
        "schema": 1,
        "generic_version": FEATURE_VERSION,
        "flatfield_only_version": FLATFIELD_ONLY_FEATURE_VERSION,
        "feature_names": list(FLATFIELD_FEATURE_NAMES),
        "input": "source-bound-CHW-float32-log-camera-rgb",
        "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "display_contract": display.CONTRACT,
        "change_display_contract": CHANGE_DISPLAY_CONTRACT,
        "matrix_rows_camera_rgb_cols_linear_srgb": display.M_NATURAL.tolist(),
        "matrix_definition_sha256": display.display_contract()[
            "matrix_definition_sha256"
        ],
        "matrix_code_sha256": sha256(Path(display.__file__).read_bytes()).hexdigest(),
        "alignment_code_sha256": sha256(
            Path(__file__)
            .with_name("tools")
            .joinpath("ffac_v5_temporal_alignment.py")
            .read_bytes()
        ).hexdigest(),
        "linear_rgb": "max(exp(log_camera_rgb).T @ M_natural, 0)",
        "reference": "native_references/<run>/pre_injection.log_linear_rgb.npy",
        "registration": "H_v5_B_to_frame_native; INTER_LINEAR; zero border; all-neighbour support",
        "operator_order": "calibrate-linear-colour; warp-baseline; box5-both; log-ratio",
        "box_size_px": BOX_SIZE,
        "box_border": "BORDER_REFLECT_101",
        "luminance_weights": display.LUMINANCE_WEIGHTS.tolist(),
        "denominator_floor": float(DENOMINATOR_FLOOR),
        "numerator_floor": None,
        "invalid_policy": "scientific-mask-and-finite-source; no repair; require complete box support",
        "nonpositive_numerator_policy": "invalid; NaN features; never a training or prediction pixel",
        "device": "cpu",
        "output_dtype": "float32",
        "gas_head_only": True,
    }


def _require_signal(value: np.ndarray, name: str) -> np.ndarray:
    signal = np.asarray(value)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError(f"{name} requires CHW float32 log-camera RGB")
    if min(signal.shape[1:]) < 1:
        raise ValueError(f"{name} requires a nonempty image")
    return signal


def _require_mask(value: np.ndarray | None, shape: tuple[int, int], name: str):
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise TypeError(f"{name} must be source HW bool with the exact image shape")
    return mask


def _require_tile_rows(tile_rows: int) -> int:
    if not isinstance(tile_rows, int) or isinstance(tile_rows, bool) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    return tile_rows


def _linear_tile(signal: np.ndarray, valid_mask: np.ndarray | None):
    """Match the display's calibrated float32 linear values without repair."""
    tile = np.moveaxis(signal, 0, -1)
    valid = np.isfinite(tile).all(axis=-1)
    if valid_mask is not None:
        valid &= valid_mask
    with np.errstate(over="ignore", invalid="ignore"):
        linear = (
            np.exp(np.where(valid[..., None], tile, 0).astype(np.float64))
            @ display.M_NATURAL
        )
        linear = linear.astype(np.float32)
    valid &= np.isfinite(linear).all(axis=-1)
    # Zero is only a transport value outside support.  The support mask
    # prevents invalid source values from entering a usable smoothing box.
    linear[~valid] = 0
    np.maximum(linear, 0, out=linear)
    return linear, valid


def _box_valid(valid: np.ndarray) -> np.ndarray:
    return cv2.erode(
        valid.astype(np.uint8), _BOX_KERNEL, borderType=cv2.BORDER_REFLECT_101
    ).astype(bool)


def prepare_flatfield_reference(
    reference_log_camera_rgb: np.ndarray,
    *,
    source_valid_mask: np.ndarray | None = None,
    tile_rows: int = 128,
) -> FlatfieldReference:
    """Calibrate the exact run baseline, retaining its scientific support."""
    signal = _require_signal(reference_log_camera_rgb, "Flatfield reference")
    tile_rows = _require_tile_rows(tile_rows)
    height, width = signal.shape[1:]
    source_valid = _require_mask(
        source_valid_mask, (height, width), "Reference validity"
    )
    linear = np.empty((height, width, 3), np.float32)
    valid = np.empty((height, width), bool)
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        linear[start:stop], valid[start:stop] = _linear_tile(
            signal[:, start:stop],
            None if source_valid is None else source_valid[start:stop],
        )
    linear.setflags(write=False)
    valid.setflags(write=False)
    return FlatfieldReference(linear, valid)


def _shape_hw(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("Flatfield target shape must contain height and width")
    result = tuple(map(int, value))
    if tuple(value) != result or min(result) < 1:
        raise ValueError("Flatfield target shape must be positive integers")
    return result


def _origin_xy(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("Flatfield source and target origins must contain x and y")
    result = tuple(map(int, value))
    if tuple(value) != result:
        raise ValueError("Flatfield origins must be integer native coordinates")
    return result


def register_flatfield_reference(
    prepared: FlatfieldReference,
    h_baseline_to_current: np.ndarray,
    *,
    target_shape_hw: tuple[int, int],
    source_origin_xy: tuple[int, int] = (0, 0),
    target_origin_xy: tuple[int, int] = (0, 0),
) -> RegisteredFlatfieldReference:
    """Transport calibrated baseline colour with the change view's exact H."""
    if not isinstance(prepared, FlatfieldReference):
        raise TypeError("Expected an unregistered FlatfieldReference")
    source, source_valid = prepared.linear_rgb, prepared.valid
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[-1] != 3
        or source_valid.dtype != np.bool_
        or source_valid.shape != source.shape[:2]
        or not np.isfinite(source).all()
        or np.any(source < 0)
    ):
        raise TypeError("Unregistered flatfield reference is malformed")
    height, width = _shape_hw(target_shape_hw)
    origin, target_origin = _origin_xy(source_origin_xy), _origin_xy(target_origin_xy)
    transform = _h(h_baseline_to_current)
    source_to_native = np.array(
        [[1.0, 0.0, origin[0]], [0.0, 1.0, origin[1]], [0.0, 0.0, 1.0]]
    )
    native_to_target = np.array(
        [[1.0, 0.0, -target_origin[0]], [0.0, 1.0, -target_origin[1]], [0.0, 0.0, 1.0]]
    )
    source_to_current = _h(native_to_target @ transform @ source_to_native)
    footprint = cv2.warpPerspective(
        source_valid.astype(np.float32),
        source_to_current,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    valid = footprint >= np.float32(1 - 1e-6)
    del footprint

    # Warp one channel at a time to bound peak native-frame memory.  The
    # independently warped Y matches the display's float32 luminance path.
    rgb_box5 = np.empty((height, width, 3), np.float32)
    luminance_box5 = None
    for channel in range(4):
        field = source[..., channel] if channel < 3 else source @ _LUMINANCE
        aligned = cv2.warpPerspective(
            field,
            source_to_current,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        aligned[~valid] = 0
        smoothed = cv2.blur(
            aligned, (BOX_SIZE, BOX_SIZE), borderType=cv2.BORDER_REFLECT_101
        )
        if channel < 3:
            rgb_box5[..., channel] = smoothed
        else:
            luminance_box5 = smoothed
        del field, aligned, smoothed
    box_valid = _box_valid(valid)
    for array in (rgb_box5, luminance_box5, valid, box_valid):
        array.setflags(write=False)
    return RegisteredFlatfieldReference(
        rgb_box5,
        luminance_box5,
        valid,
        box_valid,
        {
            "policy": "warp-calibrated-linear-rgb-and-luminance-then-box5",
            "source_origin_xy": list(origin),
            "target_origin_xy": list(target_origin),
            "target_shape_hw": [height, width],
            "H_v5_B_to_frame_native": transform.tolist(),
        },
    )


def _require_registered(
    reference: RegisteredFlatfieldReference, shape: tuple[int, int]
):
    if not isinstance(reference, RegisteredFlatfieldReference):
        raise TypeError("reference must be a per-frame RegisteredFlatfieldReference")
    if (
        reference.rgb_box5.shape != (*shape, 3)
        or reference.luminance_box5.shape != shape
        or reference.valid.shape != shape
        or reference.box_valid.shape != shape
    ):
        raise ValueError(
            "Registered pre-injection reference does not match current geometry"
        )
    if (
        reference.rgb_box5.dtype != np.float32
        or reference.luminance_box5.dtype != np.float32
        or reference.valid.dtype != np.bool_
        or reference.box_valid.dtype != np.bool_
    ):
        raise TypeError("Registered pre-injection reference is malformed")


def _feature_tile(signal, reference, current_valid, start, stop):
    halo_start, halo_stop = max(0, start - 2), min(signal.shape[1], stop + 2)
    offset, length = start - halo_start, stop - start
    linear, valid = _linear_tile(
        signal[:, halo_start:halo_stop],
        None if current_valid is None else current_valid[halo_start:halo_stop],
    )
    blurred = cv2.blur(linear, (BOX_SIZE, BOX_SIZE), borderType=cv2.BORDER_REFLECT_101)[
        offset : offset + length
    ]
    usable = (
        _box_valid(valid)[offset : offset + length] & reference.box_valid[start:stop]
    )
    numerator_y = blurred @ _LUMINANCE
    usable &= (blurred > 0).all(axis=-1) & (numerator_y > 0)
    usable &= np.isfinite(blurred).all(axis=-1) & np.isfinite(numerator_y)
    features = np.full((*blurred.shape[:2], 4), np.nan, np.float32)
    # Difference of logs computes the exact log ratio while avoiding overflow
    # when a large finite numerator is divided by the fixed small floor.
    for channel in range(4):
        numerator = blurred[..., channel] if channel < 3 else numerator_y
        denominator = (
            reference.rgb_box5[start:stop, :, channel]
            if channel < 3
            else reference.luminance_box5[start:stop]
        )
        logged = np.full(numerator.shape, np.nan, np.float32)
        np.log(numerator, out=logged, where=usable)
        logged -= np.log(np.maximum(denominator, DENOMINATOR_FLOOR))
        features[..., channel] = logged
    usable &= np.isfinite(features).all(axis=-1)
    features[~usable] = np.nan
    return features, usable


def iter_flatfield_feature_tiles(
    current_log_camera_rgb: np.ndarray,
    reference: RegisteredFlatfieldReference,
    *,
    current_valid_mask: np.ndarray | None = None,
    tile_rows: int = 128,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(start, stop, Htile W 4 features, Htile W valid)`` on CPU."""
    signal = _require_signal(current_log_camera_rgb, "Flatfield current frame")
    tile_rows = _require_tile_rows(tile_rows)
    shape = signal.shape[1:]
    current_valid = _require_mask(current_valid_mask, shape, "Current validity")
    _require_registered(reference, shape)
    for start in range(0, shape[0], tile_rows):
        stop = min(start + tile_rows, shape[0])
        features, valid = _feature_tile(signal, reference, current_valid, start, stop)
        yield start, stop, features, valid


def extract_flatfield_features(
    current_log_camera_rgb: np.ndarray,
    reference: RegisteredFlatfieldReference,
    *,
    current_valid_mask: np.ndarray | None = None,
    selection: tuple[np.ndarray, np.ndarray] | None = None,
    tile_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return features and support on the native grid or at selected pixels.

    ``selection=(rows, columns)`` returns N by 4 features and N validity flags
    in request order, preserving duplicates.  Only tiles touched by selected
    rows are evaluated.  Without selection the result is H by W by 4 plus H
    by W validity.  Every unusable row contains NaN in all four features.
    """
    signal = _require_signal(current_log_camera_rgb, "Flatfield current frame")
    tile_rows = _require_tile_rows(tile_rows)
    height, width = signal.shape[1:]
    current_valid = _require_mask(
        current_valid_mask, (height, width), "Current validity"
    )
    _require_registered(reference, (height, width))
    if selection is None:
        features = np.empty((height, width, 4), np.float32)
        valid = np.empty((height, width), bool)
        for start, stop, tile_features, tile_valid in iter_flatfield_feature_tiles(
            signal, reference, current_valid_mask=current_valid, tile_rows=tile_rows
        ):
            features[start:stop], valid[start:stop] = tile_features, tile_valid
        return features, valid
    if len(selection) != 2:
        raise ValueError("selection must contain rows and columns")
    rows, columns = (np.asarray(index) for index in selection)
    if (
        rows.ndim != 1
        or columns.shape != rows.shape
        or rows.dtype.kind not in "iu"
        or columns.dtype.kind not in "iu"
    ):
        raise TypeError(
            "selection must be equally sized one-dimensional integer rows and columns"
        )
    if (
        np.any(rows < 0)
        or np.any(rows >= height)
        or np.any(columns < 0)
        or np.any(columns >= width)
    ):
        raise IndexError("Flatfield selection is outside the native image")
    features = np.empty((rows.size, 4), np.float32)
    valid = np.empty(rows.size, bool)
    tile_ids = rows // tile_rows
    for tile_id in np.unique(tile_ids):
        start = int(tile_id) * tile_rows
        stop = min(start + tile_rows, height)
        selected = np.flatnonzero(tile_ids == tile_id)
        tile_features, tile_valid = _feature_tile(
            signal, reference, current_valid, start, stop
        )
        local_rows, local_columns = rows[selected] - start, columns[selected]
        features[selected] = tile_features[local_rows, local_columns]
        valid[selected] = tile_valid[local_rows, local_columns]
    return features, valid


def variant_feature_names(
    variant: str, current_feature_names: Sequence[str]
) -> tuple[str, ...]:
    """Choose gas features without changing the independent CO2 contract."""
    if variant in {"G0prime", "G0'", "G0′", "G1"}:
        return tuple(current_feature_names)
    if variant == "G2":
        return (*current_feature_names, *FLATFIELD_FEATURE_NAMES)
    if variant == "G3":
        return FLATFIELD_FEATURE_NAMES
    raise ValueError(f"Unknown gas feature variant {variant!r}")


def select_variant_features(
    variant: str,
    current_features: np.ndarray,
    flatfield_features: np.ndarray | None = None,
):
    """Select a variant's last feature axis on sparse or native matrices."""
    current = np.asarray(current_features)
    if current.dtype != np.float32 or current.ndim < 2:
        raise TypeError("Current gas features must be float32 with a last feature axis")
    if variant in {"G0prime", "G0'", "G0′", "G1"}:
        return current
    if variant not in {"G2", "G3"}:
        raise ValueError(f"Unknown gas feature variant {variant!r}")
    flatfield = np.asarray(flatfield_features)
    if flatfield.dtype != np.float32 or flatfield.shape != (*current.shape[:-1], 4):
        raise TypeError(
            "Flatfield gas features must be same-grid float32 with four columns"
        )
    return (
        flatfield if variant == "G3" else np.concatenate((current, flatfield), axis=-1)
    )


__all__ = [
    "FEATURE_VERSION",
    "FLATFIELD_ONLY_FEATURE_VERSION",
    "FLATFIELD_FEATURE_NAMES",
    "BOX_SIZE",
    "DENOMINATOR_FLOOR",
    "FlatfieldReference",
    "RegisteredFlatfieldReference",
    "flatfield_contract",
    "prepare_flatfield_reference",
    "register_flatfield_reference",
    "iter_flatfield_feature_tiles",
    "extract_flatfield_features",
    "variant_feature_names",
    "select_variant_features",
]
