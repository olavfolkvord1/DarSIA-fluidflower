"""CPU-only, display-only FF_AC change-from-pre-injection photograph.

The fixed transform follows ``ffac-track1-display-change-flatfield-v1``.  It
uses the same camera-linear colour calibration and lift1 transfer as the three
Track1 photographic modes.  It never changes scientific log-camera arrays,
phase maps, sidecars, or labels.  Calibrated baseline luminance is prepared
once per run, but is *registered and then smoothed for each frame*.  Smoothing
before the per-frame warp would not match the scientific feature geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import ffac_track1_display_modes as display


CONTRACT = "ffac-track1-display-change-flatfield-v1"
BOX_SIZE = 5
MIN_REFERENCE_LUMINANCE = 1e-6
_SRGB_EOTF_LUT = np.arange(256, dtype=np.float32) / np.float32(255)
_SRGB_EOTF_LUT = np.where(
    _SRGB_EOTF_LUT <= np.float32(.04045),
    _SRGB_EOTF_LUT / np.float32(12.92),
    np.power((_SRGB_EOTF_LUT + np.float32(.055)) / np.float32(1.055),
             np.float32(2.4))).astype(np.float32)
_SRGB_EOTF_LUT.setflags(write=False)


@dataclass(frozen=True)
class FlatfieldReference:
    """Unsmoothed calibrated luminance of an exact pre-injection reference."""

    luminance: np.ndarray
    valid: np.ndarray
    invalid_cleanup: dict


@dataclass(frozen=True)
class RegisteredFlatfieldReference:
    """Reference luminance transported to one frame, then box-smoothed."""

    luminance_box5: np.ndarray
    valid: np.ndarray
    invalid_cleanup: dict
    alignment: dict


def _require_log_signal(value: np.ndarray, *, name: str) -> np.ndarray:
    signal = np.asarray(value)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError(f"{name} requires CHW float32 log-camera RGB, never uint8")
    if min(signal.shape[1:]) < 1:
        raise ValueError(f"{name} requires a non-empty image")
    return signal


def _calibrated_linear(signal: np.ndarray, *, tile_rows: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Match display-modes v1 through its isolated-invalid cleanup stage.

    Highlight neutralisation is deliberately omitted here: it acts only
    outside photo sand, where the existing lift1 image is shown unchanged.
    """
    height, width = signal.shape[1:]
    linear = np.empty((height, width, 3), np.float32)
    valid = np.empty((height, width), bool)
    nonfinite_input = 0
    nonfinite_transform = 0
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        tile = np.moveaxis(signal[:, start:stop], 0, -1)
        tile_valid = np.isfinite(tile).all(axis=-1)
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.exp(np.where(tile_valid[..., None], tile, 0).astype(np.float64)) @ display.M_NATURAL
            converted = converted.astype(np.float32)
        converted_valid = np.isfinite(converted).all(axis=-1)
        nonfinite_input += int(np.count_nonzero(~tile_valid))
        nonfinite_transform += int(np.count_nonzero(tile_valid & ~converted_valid))
        tile_valid &= converted_valid
        converted[~tile_valid] = np.nan
        linear[start:stop] = np.maximum(converted, 0)
        valid[start:stop] = tile_valid
    unfilled, fill = display._fill_small_invalid(linear, valid)
    # A large invalid region remains unavailable.  Keep its NaNs out of the
    # 5x5 box filter so they cannot contaminate otherwise valid neighbours.
    linear[unfilled] = 0
    return linear, ~unfilled, {
        "input_nonfinite_pixels": nonfinite_input,
        "transform_nonfinite_pixels": nonfinite_transform,
        "invalid_cleanup": fill,
    }


def _box5_fully_valid(valid: np.ndarray) -> np.ndarray:
    """Require all pixels touched by the fixed 5x5 box to have source support."""
    return cv2.erode(
        valid.astype(np.uint8), np.ones((BOX_SIZE, BOX_SIZE), np.uint8),
        borderType=cv2.BORDER_REFLECT_101).astype(bool)


def prepare_flatfield_reference(reference_log_camera_rgb: np.ndarray, *,
                                source_valid_mask: np.ndarray | None = None,
                                tile_rows: int = 128) -> FlatfieldReference:
    """Prepare the run's pre-injection reference before per-frame registration.

    The caller must supply the same source-bound pre-injection reference used
    by feature support.  It remains on its source grid here; never blur before
    transporting it with each frame's source-bound homography.
    """
    reference = _require_log_signal(reference_log_camera_rgb, name="Flatfield reference")
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    linear, valid, metadata = _calibrated_linear(reference, tile_rows=tile_rows)
    if source_valid_mask is not None:
        source_valid = np.asarray(source_valid_mask)
        if source_valid.dtype != np.bool_ or source_valid.shape != valid.shape:
            raise TypeError("Flatfield source valid mask must be source HW bool")
        # Preserve the deliberate display-v2 repair of tiny, fully enclosed
        # nonfinite pixels even when the original scientific mask marks them
        # invalid.  Do not extend into any other source-invalid pixel.
        original_finite = np.isfinite(reference).all(axis=0)
        repaired_isolated = valid & ~original_finite
        valid &= source_valid | repaired_isolated
    luminance = linear @ display.LUMINANCE_WEIGHTS.astype(np.float32)
    luminance[~valid] = 0
    luminance.setflags(write=False)
    valid.setflags(write=False)
    return FlatfieldReference(luminance, valid, metadata)


def register_flatfield_reference(
    prepared: FlatfieldReference, h_baseline_to_current: np.ndarray, *,
    target_shape_hw: tuple[int, int], source_origin_xy: tuple[int, int] = (0, 0),
    target_origin_xy: tuple[int, int] = (0, 0),
) -> RegisteredFlatfieldReference:
    """Warp baseline luminance to the exact current frame, then smooth it.

    The homography and all-four-neighbour validity policy match
    ``ffac_v5_temporal_alignment.warp_reference_to_current``.  Warping the
    linear luminance is equivalent to taking luminance after bilinear RGB
    warping, without the extra memory needed for a 3-channel baseline warp.
    """
    if not isinstance(prepared, FlatfieldReference):
        raise TypeError("Expected an unregistered FlatfieldReference")
    source = np.asarray(prepared.luminance)
    source_valid = np.asarray(prepared.valid)
    if (source.dtype != np.float32 or source.ndim != 2 or
            source_valid.dtype != np.bool_ or source_valid.shape != source.shape or
            not np.isfinite(source).all()):
        raise TypeError("Unregistered flatfield reference is malformed")
    if len(target_shape_hw) != 2 or min(target_shape_hw) < 1:
        raise ValueError("Flatfield reference needs a positive target shape")
    height, width = map(int, target_shape_hw)
    if tuple(target_shape_hw) != (height, width):
        raise ValueError("Flatfield target shape must be integer")
    if len(source_origin_xy) != 2 or len(target_origin_xy) != 2:
        raise ValueError("Flatfield source and target origins must contain x and y")
    origin = tuple(map(int, source_origin_xy))
    target_origin = tuple(map(int, target_origin_xy))
    if tuple(source_origin_xy) != origin or tuple(target_origin_xy) != target_origin:
        raise ValueError("Flatfield source and target origins must be integer")
    from .ffac_v5_temporal_alignment import _h

    transform = _h(h_baseline_to_current)
    source_to_native = np.array([[1., 0., origin[0]], [0., 1., origin[1]],
                                 [0., 0., 1.]], np.float64)
    native_to_target = np.array([[1., 0., -target_origin[0]],
                                 [0., 1., -target_origin[1]],
                                 [0., 0., 1.]], np.float64)
    source_to_current = _h(native_to_target @ transform @ source_to_native)
    aligned = cv2.warpPerspective(
        source, source_to_current, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    footprint = cv2.warpPerspective(
        source_valid.astype(np.float32), source_to_current, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0)
    valid = footprint >= np.float32(1 - 1e-6)
    aligned[~valid] = 0
    smoothed = cv2.blur(aligned, (BOX_SIZE, BOX_SIZE))
    smoothed.setflags(write=False)
    valid.setflags(write=False)
    return RegisteredFlatfieldReference(
        smoothed, valid, prepared.invalid_cleanup,
        {"policy": "warp-calibrated-linear-luminance-then-box5",
         "source_origin_xy": list(origin),
         "target_origin_xy": list(target_origin),
         "target_shape_hw": [height, width],
         "valid_pixels": int(np.count_nonzero(valid)),
         "H_baseline_to_current": transform.tolist()},
    )


def _decode_lift1(value: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != (*shape_hw, 3):
        raise ValueError("Existing lift1 image must have the native analysis shape")
    if image.dtype == np.uint8:
        return image
    if image.dtype != np.float32 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise TypeError("Existing lift1 image must be uint8 RGB or finite float32 RGB in [0, 1]")
    return np.rint(image * 255).astype(np.uint8)


def smoothed_sand_luminance_cv(rgb8: np.ndarray, photo_sand: np.ndarray) -> float:
    """Coefficient of variation for the frame-0 acceptance check.

    Evaluate the actual display image (after the fixed tone/8-bit encoding),
    decoding sRGB before a 5x5 luminance box filter.  This is diagnostic only;
    it never influences any display transform or scientific result.
    """
    rgb = np.asarray(rgb8)
    sand = np.asarray(photo_sand)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise TypeError("Flatfield CV requires HWC uint8 RGB")
    if sand.dtype != np.bool_ or sand.shape != rgb.shape[:2] or not sand.any():
        raise ValueError("Flatfield CV requires a nonempty native photo-sand mask")
    # A 256-entry EOTF lookup avoids several full-resolution HWC temporaries.
    luminance = _SRGB_EOTF_LUT[rgb[..., 0]]
    luminance *= np.float32(display.LUMINANCE_WEIGHTS[0])
    for channel in (1, 2):
        luminance += (_SRGB_EOTF_LUT[rgb[..., channel]] *
                      np.float32(display.LUMINANCE_WEIGHTS[channel]))
    smoothed = cv2.blur(luminance, (BOX_SIZE, BOX_SIZE))
    values = smoothed[sand].astype(np.float64)
    mean = float(np.mean(values))
    if mean <= 0 or not np.isfinite(mean):
        raise ValueError("Flatfield sand luminance has no positive finite mean")
    return float(np.std(values) / mean)


def render_change_flatfield(current_log_camera_rgb: np.ndarray,
                            reference: RegisteredFlatfieldReference,
                            photo_sand: np.ndarray, *,
                            lift1_rgb: np.ndarray | None = None,
                            tile_rows: int = 128) -> tuple[np.ndarray, dict]:
    """Render the fixed change view as HWC uint8 RGB.

    ``lift1_rgb`` should be the existing analysis-grid lift1 display; its
    pixels are copied exactly outside photo sand.  If omitted, the exact v1
    renderer supplies it.  The input arrays are never modified.
    """
    signal = _require_log_signal(current_log_camera_rgb, name="Flatfield current frame")
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    height, width = signal.shape[1:]
    sand = np.asarray(photo_sand)
    if sand.dtype != np.bool_ or sand.shape != (height, width):
        raise TypeError("Photo-sand mask must be native-shape boolean")
    if not isinstance(reference, RegisteredFlatfieldReference):
        raise TypeError("reference must be a per-frame RegisteredFlatfieldReference")
    prepared = reference
    if prepared.luminance_box5.shape != (height, width) or prepared.valid.shape != (height, width):
        raise ValueError("Pre-injection reference does not match current native geometry")
    if (prepared.luminance_box5.dtype != np.float32 or prepared.valid.dtype != np.bool_
            or not np.isfinite(prepared.luminance_box5).all()):
        raise TypeError("Prepared pre-injection reference is malformed")

    if lift1_rgb is None:
        modes, _ = display.render_display_modes(signal, sand, tile_rows=tile_rows)
        lift1 = modes["lift1"]
    else:
        lift1 = _decode_lift1(lift1_rgb, (height, width))
    output = np.array(lift1, dtype=np.uint8, copy=True)
    linear, valid, input_meta = _calibrated_linear(signal, tile_rows=tile_rows)
    # A centre pixel with a large invalid region one or two pixels away has
    # a zero-contaminated box result.  Leave it on the already valid lift1
    # photograph instead of presenting a false colour change.  Reflect at
    # image edges, matching cv2.blur's border semantics.
    usable = sand & _box5_fully_valid(valid) & _box5_fully_valid(prepared.valid)
    gain = display.GAINS["lift1"]
    white = 0.95 * gain
    # Haloed tiles reproduce a full-frame OpenCV box filter while keeping the
    # extra current-frame memory bounded independently of image height.
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        halo_start, halo_stop = max(0, start - 2), min(height, stop + 2)
        # BORDER_DEFAULT is the same reflected boundary used by cv2.blur in
        # the read-only prototype.  At interior tile boundaries the halo
        # supplies the real neighbouring rows.
        blurred = cv2.blur(linear[halo_start:halo_stop], (BOX_SIZE, BOX_SIZE))[
            start - halo_start:stop - halo_start]
        divisor = np.maximum(prepared.luminance_box5[start:stop],
                             np.float32(MIN_REFERENCE_LUMINANCE))
        flatfield = blurred * (np.float32(display.Y_REFERENCE) / divisor)[..., None]
        luminance = flatfield @ display.LUMINANCE_WEIGHTS.astype(np.float32)
        gained = gain * luminance.astype(np.float64)
        toned = gained * (1 + gained / white ** 2) / (1 + gained)
        ratio = np.divide(toned, luminance, out=np.zeros_like(toned), where=luminance > 0)
        rgb = flatfield.astype(np.float64) * ratio[..., None]
        maximum = np.max(rgb, axis=-1, keepdims=True)
        rgb /= np.maximum(maximum, 1)
        rgb8 = np.rint(255 * display._srgb_encode(rgb)).astype(np.uint8)
        # Unlike the three photographic source modes, the flatfield view
        # intentionally uses the box filter on linear colour itself.  Its
        # fixed transform contains no additional chroma filter.
        selected = usable[start:stop]
        output[start:stop][selected] = rgb8[selected]
    return output, {
        "contract": CONTRACT,
        "display_only": True,
        "source_mode": display.CONTRACT,
        "reference_luminance": display.Y_REFERENCE,
        "box_size_px": BOX_SIZE,
        "lift_gain": gain,
        "lift_white": white,
        "sand_pixels": int(np.count_nonzero(sand)),
        "flatfield_pixels": int(np.count_nonzero(usable)),
        "sand_pixels_without_valid_pair": int(np.count_nonzero(sand & ~usable)),
        "current_cleanup": input_meta,
        "reference_cleanup": prepared.invalid_cleanup,
        "reference_alignment": prepared.alignment,
        "outside_photo_sand": "existing_lift1_unchanged",
    }


__all__ = ["CONTRACT", "BOX_SIZE", "FlatfieldReference",
           "RegisteredFlatfieldReference", "prepare_flatfield_reference",
           "register_flatfield_reference", "render_change_flatfield",
           "smoothed_sand_luminance_cv"]
