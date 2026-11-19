"""Each raw photograph registered onto a setup's reference and sampled once from its
sensor.

A setup's reference is one colour channel of its baseline photograph on the grid
the photographs are mapped onto: the baseline's camera view (its sensor values
through the camera's lens model on the camera grid), or that view cropped to the
four rig points by the official curvature correction's crop
(:class:`darsia.CurvatureCorrection`) in the given precision, which places the rig
in metres, as our production's setup cropped it: in single precision its transform
is solved from the rig points and its positions are mapped in single precision, in
double precision it is OpenCV's transform and warp
(:func:`darsia.warp_perspective`). The setup's homography registers the reference
to the baseline's camera view, from the rig points for a crop
(:func:`darsia.rig_points_homography`) and the identity otherwise.

Each photograph's camera view, through its own lens model (the tables recorded in
its raw file, or a configured model; :func:`darsia.lens_coordinates`), is
registered to the reference from the setup's homography
(:func:`darsia.projective_registration`). Its sensor values are then sampled once
onto the reference's grid through that homography and its lens model
(:func:`darsia.registration_mapping`); a value that draws on an invalid photosite is
NaN. The registration compares the views of all values, without the decode's
invalid values, and the sampling divides by the photograph's white range after the
interpolation, as our production did. The baseline itself is sampled through the
setup's homography.

All arithmetic runs in the given floating precision (single precision at least).

"""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

import darsia
from darsia.image.raw import raw_stage


@dataclass(frozen=True)
class FrameRegistration:
    """A setup's reference, to which each raw photograph is registered."""

    reference: np.ndarray
    """One colour channel (rows, cols) of the setup's reference, in units of the
    white range."""
    homography: np.ndarray
    """The setup's homography from the reference's grid to the baseline's camera
    view, from which each photograph's registration starts."""
    settings: "darsia.ProjectiveRegistrationSettings"
    """Settings of the registration on the reference's grid."""
    lens: tuple
    """The configured lens models, in order of preference
    (:func:`darsia.lens_coordinates`)."""
    camera_shape: tuple[int, int]
    """Rows and columns of the camera grid, on which the views are compared."""
    channel: int
    """The colour channel of the reference and the views."""
    demosaic: str
    """Interpolation of the colour filter array (:func:`darsia.read_raw_sensor`)."""
    dimensions: tuple[float, float]
    """Physical height and width of the reference's grid."""
    origin: tuple[float, float]
    """Cartesian coordinates of the top-left pixel of the reference's grid."""


def camera_view(
    values: np.ndarray, lens: Callable, camera_shape: tuple[int, int], channels
) -> np.ndarray:
    """Sensor values on the camera grid through a lens model, with the identity
    homography, in the values' floating precision.

    Parameters
    ----------
    values : array
        Sensor values (rows, cols, channels) of the given colour channels.
    lens : callable
        The lens model's coordinate function (:func:`darsia.lens_coordinates`).
    camera_shape : tuple of int
        Rows and columns of the camera grid.
    channels : tuple of int
        The colour channels of ``values`` (0 red, 1 green, 2 blue).

    Returns
    -------
    array
        The view (rows, cols, channels) on the camera grid, NaN where the sampling
        leaves the sensor.

    """
    dtype = np.result_type(values, np.float32)
    mapping = darsia.registration_mapping(
        np.eye(3, dtype=dtype),
        camera_shape,
        values.shape[:2],
        [float(n) for n in camera_shape],
        coordinates=partial(lens, channels=tuple(channels)),
    )
    return mapping.correct_array(values)


def reference_grid(
    camera_shape: tuple[int, int],
    corners: Optional[Sequence] = None,
    size: Optional[tuple[float, float]] = None,
) -> darsia.Image:
    """An empty image on a setup's reference grid: the official crop of the camera
    grid to the rig points, or the camera grid itself without them.

    Parameters
    ----------
    camera_shape : tuple of int
        Rows and columns of the camera grid.
    corners : sequence, optional
        The rig points (row, column) in the camera grid, in the order top left,
        bottom left, bottom right, top right.
    size : tuple of float, optional
        Physical width and height of the crop.

    Returns
    -------
    darsia.Image
        Zeros on the grid, with its physical dimensions and origin.

    """
    zeros = np.zeros(tuple(camera_shape), np.uint8)
    if corners is None:
        return darsia.Image(zeros, dimensions=[float(n) for n in camera_shape])
    width, height = size
    cropped = darsia.extract_quadrilateral_ROI(
        zeros, pts_src=darsia.make_voxel(corners), width=width, height=height
    )
    return darsia.Image(cropped, dimensions=[height, width], origin=[0.0, height])


def setup_frame_registration(
    path: Path,
    settings: Mapping[str, "darsia.ProjectiveRegistrationSettings"],
    lens: Sequence,
    camera_shape: tuple[int, int],
    channel: int,
    demosaic: str,
    dtype,
    corners: Optional[Sequence] = None,
    size: Optional[tuple[float, float]] = None,
) -> tuple[FrameRegistration, np.ndarray, "darsia.ProjectiveRegistration"]:
    """A setup's registration from its baseline raw file, and the baseline on the
    reference's grid.

    The baseline's camera view (all values, divided by the white range) is the
    reference, or the official curvature correction's crop of it to the rig points
    in the given precision (:func:`reference_grid`, :func:`darsia.warp_perspective`).
    With rig points, the reference is registered to the baseline's view from them;
    the baseline is then sampled once from its sensor through that homography.

    Parameters
    ----------
    path : Path
        The baseline's raw file.
    settings : Mapping[str, ProjectiveRegistrationSettings]
        The registration's settings for each grid size, by ``"<rows>x<cols>"``.
    lens : sequence
        The configured lens models, in order of preference.
    camera_shape : tuple of int
        Rows and columns of the camera grid.
    channel : int
        The colour channel that is registered, e.g. 1 for green.
    demosaic : str
        Interpolation of the colour filter array.
    dtype : numpy dtype
        Floating precision of the calculation.
    corners : sequence, optional
        The rig points (row, column) in the camera grid; without them the
        reference is the camera view.
    size : tuple of float, optional
        Physical width and height of the crop to the rig points.

    Returns
    -------
    FrameRegistration
        The setup's reference and homography.
    array
        The baseline (rows, cols, 3) on the reference's grid, divided by its white
        range, NaN where invalid.
    ProjectiveRegistration
        The setup's registration and its quality; the identity without rig points.

    Raises
    ------
    ValueError
        If no settings are configured for the reference's grid size.

    """
    dtype = np.dtype(dtype)
    sensor = darsia.read_raw_sensor(path, demosaic)
    coordinates = darsia.lens_coordinates(path, lens)
    white = dtype.type(sensor.white_range)
    view = camera_view(
        sensor.values[..., channel : channel + 1].astype(dtype),
        coordinates,
        camera_shape,
        (channel,),
    )[..., 0]
    grid = reference_grid(camera_shape, corners, size)
    rows, cols = grid.shape[:2]
    if f"{rows}x{cols}" not in settings:
        raise ValueError(f"No registration settings for a {rows} x {cols} grid.")
    chosen = settings[f"{rows}x{cols}"]
    reference = view / white
    found = darsia.ProjectiveRegistration(np.eye(3, dtype=dtype), 0, 1.0, {})
    if corners is not None:
        names = ("top_left", "bottom_left", "bottom_right", "top_right")
        crop = dict(
            zip(names, (list(c) for c in corners)), width=size[0], height=size[1]
        )
        crop.update(in_meters=True, precision=dtype)
        stages = {"init": None, "crop": crop, "bulge": None, "stretch": None}
        reference = darsia.CurvatureCorrection(config=stages).correct_array(reference)
        initial = darsia.rig_points_homography(np.asarray(corners), (rows, cols), dtype)
        found = darsia.projective_registration(reference * white, view, initial, chosen)
    registration = FrameRegistration(
        reference=reference,
        homography=found.homography,
        settings=chosen,
        lens=tuple(lens),
        camera_shape=tuple(camera_shape),
        channel=channel,
        demosaic=demosaic,
        dimensions=tuple(grid.dimensions),
        origin=tuple(float(v) for v in grid.origin),
    )
    return (
        registration,
        _sample(sensor, coordinates, found.homography, registration, dtype),
        found,
    )


def register_raw_frame(
    path: Path, registration: FrameRegistration, dtype
) -> tuple[np.ndarray, "darsia.ProjectiveRegistration"]:
    """Register a raw photograph to a setup's reference and sample it once onto the
    reference's grid.

    Parameters
    ----------
    path : Path
        The photograph's raw file.
    registration : FrameRegistration
        The setup's reference.
    dtype : numpy dtype
        Floating precision of the calculation.

    Returns
    -------
    array
        The photograph (rows, cols, 3) on the reference's grid, divided by its white
        range, NaN where it draws on an invalid photosite or leaves the sensor.
    ProjectiveRegistration
        Its homography from the reference's grid to its camera view, and the
        registration's quality.

    """
    dtype = np.dtype(dtype)
    sensor = darsia.read_raw_sensor(path, registration.demosaic)
    with raw_stage("registration_and_sampling"):
        coordinates = darsia.lens_coordinates(path, registration.lens)
        white = dtype.type(sensor.white_range)
        channel = registration.channel
        view = camera_view(
            sensor.values[..., channel : channel + 1].astype(dtype),
            coordinates,
            registration.camera_shape,
            (channel,),
        )[..., 0]
        found = darsia.projective_registration(
            np.asarray(registration.reference, dtype) * white,
            view,
            registration.homography,
            registration.settings,
        )
        return _sample(sensor, coordinates, found.homography, registration, dtype), found


FRAME_REGISTRATION = "registration.json"
"""The file of a photograph's stored registration."""


def write_frame_registration(
    folder: Path, found: "darsia.ProjectiveRegistration"
) -> None:
    """Store a photograph's registration in ``folder`` (:data:`FRAME_REGISTRATION`):
    its 3 x 3 homography, from which its mapping is made again, and its quality."""
    record = {
        "homography": np.asarray(found.homography).tolist(),
        "dtype": np.asarray(found.homography).dtype.name,
        "matches": int(found.matches),
        "inlier_fraction": float(found.inlier_fraction),
        "heldout": {k: np.asarray(v).tolist() for k, v in found.heldout.items()},
    }
    (Path(folder) / FRAME_REGISTRATION).write_text(json.dumps(record), "utf-8")


def read_frame_registration(folder: Path) -> "darsia.ProjectiveRegistration":
    """A photograph's stored registration (:func:`write_frame_registration`)."""
    record = json.loads((Path(folder) / FRAME_REGISTRATION).read_text("utf-8"))
    dtype = np.dtype(record["dtype"])
    return darsia.ProjectiveRegistration(
        np.asarray(record["homography"], dtype),
        record["matches"],
        record["inlier_fraction"],
        {k: np.asarray(v, dtype) for k, v in record["heldout"].items()},
    )


def _sample(sensor, coordinates, homography, registration, dtype) -> np.ndarray:
    """Sensor values sampled once onto the reference's grid through a homography
    and a lens model, then divided by the white range; NaN where invalid."""
    values = sensor.values.astype(dtype)
    values[sensor.invalid] = np.nan
    mapping = darsia.registration_mapping(
        homography,
        registration.reference.shape,
        values.shape[:2],
        list(registration.dimensions),
        list(registration.origin),
        coordinates=coordinates,
    )
    sampled = mapping.correct_array(values)
    sampled /= dtype.type(sensor.white_range)
    return sampled
