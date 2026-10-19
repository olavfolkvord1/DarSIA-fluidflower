"""ARW positioning on the existing calibrated U/native grids.

Scientific samples remain black-subtracted camera DN.  The registration views
come from the same RAW acquisition and the existing shape-correction objects;
no companion photograph is an input.  Legacy JPEG-positioned renderers are not
changed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from . import registration
from .io import sha256_file


CONTRACT = "darsia-raw-positioning-existing-grid-v1"


@dataclass(frozen=True)
class Sensor:
    """Verified, unbalanced integer RGB on the visible sensor grid."""

    rgb: np.ndarray
    source: Mapping[str, Any]
    metadata: Mapping[str, Any]
    distortion: tuple[int, ...]
    ca_red: tuple[int, ...]
    ca_blue: tuple[int, ...]
    profile_id: str


def _version(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def read_sensor(path: str | Path, *, expected_sha256: str,
                parameters: Mapping, profile_id: str) -> Sensor:
    """Apply the established integer EA sensor arithmetic, with RAW-only IO."""
    from darsia.fluidflower.image_formats import image_family
    from .raw_camera import read_camera_identity

    path = Path(path)
    if image_family(path) != "raw":
        raise ValueError("RAW positioning requires a camera RAW acquisition")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("An exact lowercase RAW SHA-256 is required")
    before = _version(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("RAW acquisition SHA-256 differs from its binding")
    common = parameters["common"]
    policy = common["sensor"]
    profile = common["profiles"][profile_id]
    registration.check_demosaic_order(parameters=parameters)
    distortion, red, blue = registration.read_lens_tables(
        path, parameters=common["lens"])
    if distortion != tuple(profile["distortion"]):
        raise ValueError("RAW lens table differs from the calibrated profile")
    identity = read_camera_identity(path)
    try:
        import rawpy
    except ImportError as error:
        raise ImportError("RAW positioning requires rawpy (CPU LibRaw)") from error
    with rawpy.imread(str(path)) as raw:
        original = raw.raw_image_visible.copy()
        if original.dtype != np.uint16 or original.ndim != 2:
            raise ValueError("RAW visible mosaic must be two-dimensional uint16")
        if original.shape != (policy["raw_wh"][1], policy["raw_wh"][0]):
            raise ValueError("RAW visible shape differs from its calibration")
        if (not np.array_equal(raw.raw_pattern, policy["cfa_pattern"])
                or raw.color_desc != policy["color_description"].encode("ascii")):
            raise ValueError("RAW CFA/channel description differs from calibration")
        black = tuple(int(v) for v in raw.black_level_per_channel)
        if black != tuple(policy["black_levels"]):
            raise ValueError("RAW black levels differ from calibration")
        level = int(policy["black_level"])
        white = int(raw.white_level)
        if not level < white <= policy["white_maximum"]:
            raise ValueError("RAW white level differs from the sensor contract")
        if white - level >= 2**14:
            raise ValueError("RAW values exceed the verified uint16 demosaic headroom")
        crop = (int(raw.sizes.crop_left_margin), int(raw.sizes.crop_top_margin),
                int(raw.sizes.crop_width), int(raw.sizes.crop_height))
        if crop[2:] != tuple(policy["camera_wh"]):
            raise ValueError("ARW crop does not define the calibrated U extent")
        if crop[:2] != tuple(common["lens"]["raw_offset_xy"]):
            raise ValueError("ARW crop origin differs from the lens calibration")
        metadata = dict(make=identity.make, model=identity.model,
                        orientation=identity.orientation,
                        raw_visible_wh=list(policy["raw_wh"]),
                        crop_xywh=list(crop), black_levels=list(black),
                        black_level=level, white_level=white,
                        sensor_saturated_count=int(np.count_nonzero(original >= white)),
                        camera_whitebalance=[float(x) for x in raw.camera_whitebalance],
                        rgb_xyz_matrix=np.asarray(raw.rgb_xyz_matrix, np.float64).tolist())
    # Same arithmetic and operation order as registered_edge_aware_sensor_linear.
    subtracted = np.maximum(original.astype(np.int32) - level, 0).astype(np.uint16)
    del original
    rgb = cv2.demosaicing(
        subtracted, getattr(cv2, parameters["edge_aware"]["demosaic_code"]))
    if _version(path) != before or sha256_file(path) != expected_sha256:
        raise ValueError("RAW acquisition changed while it was decoded")
    rgb.setflags(write=False)
    return Sensor(rgb, dict(path=str(path), sha256=expected_sha256), metadata,
                  distortion, red, blue, profile_id)


def sample_sensor(sensor: Sensor, homography: np.ndarray, *,
                  shape_hw: Sequence[int], parameters: Mapping,
                  channels: Sequence[int] = (0, 1, 2)) -> np.ndarray:
    """Sample sensor DN once, using the original per-channel lens/warp math."""
    common = parameters["common"]
    height, width = (int(v) for v in shape_hw)
    h = np.asarray(homography, np.float64)
    if (height <= 0 or width <= 0 or h.shape != (3, 3)
            or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12):
        raise ValueError("A finite invertible native-to-U homography is required")
    selected = tuple(channels)
    if not selected or len(set(selected)) != len(selected) or any(
            c not in (0, 1, 2) for c in selected):
        raise ValueError("Sensor channels must be distinct RGB indices")
    rows = int(common["sensor"]["tile_rows"])
    if rows <= 0:
        raise ValueError("A positive configured tile-row count is required")
    coefficients = common["profiles"][sensor.profile_id]["coefficients"]
    output = np.empty((len(selected), height, width), np.float32)
    for index, channel in enumerate(selected):
        plane = sensor.rgb[..., channel].astype(np.float32)
        ca = (sensor.ca_red, None, sensor.ca_blue)[channel]
        for y0 in range(0, height, rows):
            y1 = min(y0 + rows, height)
            yy, xx = np.mgrid[y0:y1, 0:width].astype(np.float32)
            rx, ry = registration.raw_coordinates(
                xx, yy, h, coefficients, sensor.distortion, ca,
                parameters=common["lens"])
            output[index, y0:y1] = cv2.remap(
                plane, rx.astype(np.float32), ry.astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan)
    return output


def camera_u(sensor: Sensor, *, parameters: Mapping) -> np.ndarray:
    """Render the calibrated camera-equivalent U grid without a JPEG."""
    width, height = parameters["common"]["sensor"]["camera_wh"]
    return sample_sensor(sensor, np.eye(3), shape_hw=(height, width),
                         parameters=parameters)


def load_shape_chain(assets: Sequence[Mapping]) -> list:
    """Load only the existing SHA-bound geometric correction objects."""
    import darsia
    from darsia.presets.workflows.rig import Rig

    if not assets:
        raise ValueError("The existing native shape chain is required")
    chain = []
    rig = Rig()
    for asset in assets:
        path = Path(asset["path"])
        if path.suffix.lower() != ".npz" or sha256_file(path) != asset["sha256"]:
            raise ValueError("Native shape correction changed: " + str(path))
        correction = darsia.read_correction(path)
        if not rig._is_shape_correction(correction):
            raise ValueError("Native review permits geometric corrections only")
        chain.append(correction)
    return chain


def native_review_rgb(camera_rgb: np.ndarray, chain: Sequence, *,
                      white_range_dn: int, shape_hw: Sequence[int],
                      image_metadata: Mapping) -> np.ndarray:
    """Use the saved per-frame shape/motion chain, without fitting a new rig."""
    import darsia

    camera_rgb = np.asarray(camera_rgb)
    if (camera_rgb.dtype != np.float32 or camera_rgb.ndim != 3
            or camera_rgb.shape[0] != 3 or not np.isfinite(camera_rgb).all()
            or white_range_dn <= 0 or not chain):
        raise ValueError("A finite camera-U RGB view and existing shape chain are required")
    scale = np.float32(white_range_dn)
    image = darsia.OpticalImage(
        np.moveaxis(camera_rgb, 0, -1) / scale,
        transformations=list(chain), color_space="RGB", **dict(image_metadata))
    review = np.moveaxis(np.asarray(image.img, np.float32), -1, 0) * scale
    if review.shape != (3, *shape_hw) or not np.isfinite(review).all():
        raise ValueError("RAW-derived review does not share the existing native grid")
    return review


def native_review_green(camera_rgb: np.ndarray, chain: Sequence, **kwargs) -> np.ndarray:
    """Green positioning view of :func:`native_review_rgb`."""
    return native_review_rgb(camera_rgb, chain, **kwargs)[1]


def render_native(sensor: Sensor, *, shape_assets: Sequence[Mapping],
                  shape_hw: Sequence[int], image_metadata: Mapping,
                  parameters: Mapping) -> tuple[np.ndarray, dict]:
    """Position from RAW-derived views, then sample scientific DN directly."""
    from darsia._opencv_runtime import scientific_thread, completed_frame

    before = scientific_thread("raw_positioning.render_native:before")
    view = camera_u(sensor, parameters=parameters)
    review = native_review_green(
        view, load_shape_chain(shape_assets),
        white_range_dn=sensor.metadata["white_level"] - sensor.metadata["black_level"],
        shape_hw=shape_hw, image_metadata=image_metadata)
    h, quality = registration.review_to_camera(
        review, view[1], parameters=parameters["common"])
    scientific = sample_sensor(sensor, h, shape_hw=shape_hw, parameters=parameters)
    certificate = dict(
        contract=CONTRACT, source_kind="raw", raw_acquisition=dict(sensor.source),
        camera_u_wh=list(parameters["common"]["sensor"]["camera_wh"]),
        native_shape_hw=list(shape_hw), native_to_camera_u_h=h.tolist(),
        positioning_inputs="RAW-derived camera U and saved-shape-chain native review",
        shape_corrections=[dict(x) for x in shape_assets],
        registration=quality, sensor_metadata=dict(sensor.metadata),
        profile_id=sensor.profile_id,
        white_balance_applied=False, gamma_applied=False,
        scientific_units="black-subtracted camera DN",
        scientific_interpolation="one direct per-channel sensor-to-native bilinear float32 warp",
        new_rig_fit=False, new_photometric_fit=False, training=False,
        opencv_ipp=completed_frame(before, "raw_positioning.render_native:after"))
    return scientific, certificate
