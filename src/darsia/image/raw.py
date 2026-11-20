"""Decoding of camera raw files into linear images in the camera's own colours.

A camera raw file holds one value per photosite of the sensor's colour filter array.
The decode reads the visible photosites with LibRaw, subtracts the black level,
demosaics the 16-bit values with OpenCV's edge-aware Bayer interpolation, and divides
by the white range (white level minus black level) in the floating type the caller
chooses, so that the white level becomes 1. Nothing else is applied: no white balance,
no colour matrix, no gamma, no crop and no rotation. The image is linear in the
camera's red, green and blue channels, on the grid of the visible sensor in the
sensor's orientation. A geometric and lens correction, e.g.,
:class:`darsia.ImageMappingCorrection`, refers to this grid.

Values that draw on a clipped photosite, at or above the white level, are invalid
and set to NaN in their channel. The outermost photosites count as invalid too,
since the edge-aware demosaic does not interpolate there but copies the neighbouring
rows and columns.

LibRaw is used through the optional package ``rawpy`` (``pip install darsia[raw]``).

"""

from pathlib import Path

import cv2
import numpy as np

import darsia

try:
    import rawpy
except ImportError:  # Optional dependency, installed with darsia[raw].
    rawpy = None

RAW_SUFFIXES = (
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".dcr",
    ".dng",
    ".erf",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rw2",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
)
"""File suffixes (lower case) of camera raw formats, read by :func:`imread_from_raw`."""


def imread_from_raw(
    path: Path | list[Path],
    demosaic: str,
    dtype: np.dtype | str,
    transformations: list | None = None,
    **kwargs,
) -> darsia.OpticalImage:
    """Read a camera raw file, or a time series of them, as a linear optical image.

    Parameters
    ----------
    path : Path or list of Path
        Path to the raw file, or paths to the raw files of a time series.
    demosaic : str
        Interpolation of the colour filter array; ``"edge_aware"`` is OpenCV's
        edge-aware Bayer demosaic.
    dtype : numpy dtype or str
        Floating type of the image, e.g., ``"float32"``.
    transformations : list of callable, optional
        Corrections applied to the decoded image.
    **kwargs : dict
        Keyword arguments of :class:`darsia.OpticalImage`, e.g., ``date``, a list
        of dates for a time series. Dates are not read from the raw files.

    Returns
    -------
    OpticalImage
        Black-subtracted camera RGB divided by the white range, NaN where invalid.
        For a list of paths, a space-time image with the files along the time axis,
        as for other optical images.

    Raises
    ------
    ImportError
        If ``rawpy`` is not installed.
    ValueError
        If the demosaic is unknown, the colour filter array is not an RGB Bayer
        array, the colours have different black levels, or the white level does not
        exceed the black level.

    """
    if rawpy is None:
        raise ImportError("Reading camera raw files requires rawpy (darsia[raw]).")
    if demosaic != "edge_aware":
        raise ValueError(f"Unknown demosaic {demosaic!r}; 'edge_aware' is supported.")
    if isinstance(path, list):
        image = np.stack([_decode(p, dtype) for p in path], axis=2)
        kwargs["series"] = True
    else:
        image = _decode(path, dtype)
    return darsia.OpticalImage(
        image, transformations=transformations, color_space="RGB", **kwargs
    )


def raw_display_matrix(path: Path) -> np.ndarray:
    """Read an as-shot display matrix without demosaicing or postprocessing.

    A row of black-subtracted, white-range-normalized camera RGB multiplied by
    this matrix gives linear sRGB. Exposure is the camera's recorded sensor
    scale, with white-balance gains relative to green. No channel-wise or
    image-dependent brightness normalization is applied. A comparison must use
    the same matrix for both its before and after samples.
    """
    if rawpy is None:
        raise ImportError("Reading camera colour metadata requires rawpy.")
    with rawpy.RawPy() as raw:
        raw.open_file(str(path))
        return _raw_display_matrix(raw)


def _raw_display_matrix(raw) -> np.ndarray:
    """Camera metadata in the decoder's RGB order, using LibRaw's colour model.

    LibRaw's cam_xyz (rawpy rgb_xyz_matrix) maps XYZ to camera responses.
    Compose with the standard sRGB-to-XYZ primaries, normalize each camera
    response to a neutral white and invert, as in LibRaw cam_xyz_coeff. Apply
    the recorded as-shot gains before this colour conversion. This does not
    replace the scientific decoder or the fitted light/colour correction.
    """
    colours = raw.color_desc.decode().rstrip("\x00")
    if any(colour not in colours for colour in "RGB"):
        raise ValueError("Camera display requires RGB colour metadata.")
    channels = [colours.index(colour) for colour in "RGB"]
    gains = np.asarray(raw.camera_whitebalance, dtype=float)[channels]
    xyz_to_camera = np.asarray(raw.rgb_xyz_matrix, dtype=float)[channels, :]
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("The RAW file has no valid as-shot white balance.")
    # Embedded camera profiles (notably DNG) already supply camera-to-sRGB.
    # LibRaw prefers cmatrix for as-shot display; cam_xyz may be all zero there.
    embedded = np.asarray(getattr(raw, "color_matrix", np.zeros((3, 4))), dtype=float)
    if embedded.shape == (3, 4) and np.any(embedded):
        embedded = embedded[:, channels]
        if not np.isfinite(embedded).all() or np.linalg.matrix_rank(embedded) != 3:
            raise ValueError("The embedded camera display matrix is not invertible.")
        return np.diag(gains / gains[1]) @ embedded.T
    if xyz_to_camera.shape != (3, 3) or not np.isfinite(xyz_to_camera).all():
        raise ValueError("The RAW file has no valid camera colour matrix.")
    xyz_from_srgb = np.array(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ]
    )
    camera_from_srgb = xyz_to_camera @ xyz_from_srgb
    neutral = camera_from_srgb.sum(axis=1)
    if np.any(neutral <= 0) or np.linalg.matrix_rank(camera_from_srgb) != 3:
        raise ValueError("The RAW camera colour matrix is not invertible.")
    camera_from_srgb /= neutral[:, None]
    return np.diag(gains / gains[1]) @ np.linalg.inv(camera_from_srgb).T


def _decode(path: Path, dtype: np.dtype | str) -> np.ndarray:
    """Decode one raw file with the edge-aware demosaic.

    Parameters
    ----------
    path : Path
        Path to the raw file.
    dtype : numpy dtype or str
        Floating type of the image.

    Returns
    -------
    array
        Black-subtracted camera RGB divided by the white range, NaN where invalid,
        shape (rows, cols, 3).

    """
    with rawpy.imread(str(path)) as raw:
        if np.shape(raw.raw_pattern) != (2, 2):
            raise ValueError(f"{path} has no Bayer colour filter array.")
        mosaic = raw.raw_image_visible.copy()
        # RGB index of each photosite of the 2 x 2 cell at the visible origin.
        rgb_index = np.array(["RGB".find(c) for c in raw.color_desc.decode()])
        phase = rgb_index[raw.raw_colors_visible[:2, :2]]
        black_levels = set(raw.black_level_per_channel)
        white = int(raw.white_level)

    # A Bayer cell has two green photosites on one diagonal, red and blue on the other.
    if (
        sorted(phase.ravel()) != [0, 1, 1, 2]
        or (phase == phase[::-1]).any()
        or (phase == phase[:, ::-1]).any()
    ):
        raise ValueError(f"{path} has no RGB Bayer colour filter array.")
    if len(black_levels) != 1:
        raise ValueError(f"{path} has different black levels per colour.")
    black = black_levels.pop()
    if white <= black:
        raise ValueError(f"{path} has no white level above its black level.")

    # OpenCV names a Bayer layout by the second row's second and third photosites.
    code = getattr(cv2, f"COLOR_Bayer{'RGB'[phase[1, 1]]}{'RGB'[phase[1, 0]]}2RGB_EA")
    subtracted = mosaic - np.minimum(mosaic, black)
    image = cv2.demosaicing(subtracted, code).astype(dtype)
    image /= image.dtype.type(white - black)
    image[_invalid(subtracted, mosaic >= white, phase)] = np.nan
    return image


def _invalid(mosaic: np.ndarray, clipped: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Values of the edge-aware demosaic that draw on an invalid photosite.

    This follows OpenCV's edge-aware demosaic of 16-bit mosaics. Each photosite keeps
    its own colour. At a green photosite, red and blue are the means of the two
    horizontal or the two vertical neighbours of that colour. At a red or blue
    photosite, the other of the two is the mean of the four diagonal neighbours, and
    green is the mean of the vertical pair if the horizontal difference is larger,
    else of the horizontal pair. Clipped and outermost photosites are invalid, and so
    is a green value whose direction compares an outermost photosite. The outermost
    rows and columns copy their neighbours, as in OpenCV.

    Parameters
    ----------
    mosaic : array
        Black-subtracted photosite values, shape (rows, cols).
    clipped : array
        Photosites at or above the white level, same shape.
    phase : array
        RGB index of each photosite of the 2 x 2 Bayer cell.

    Returns
    -------
    array
        True where a value is invalid, shape (rows, cols, 3).

    """
    bad = clipped.copy()
    bad[[0, -1], :] = True
    bad[:, [0, -1]] = True
    centre = bad[1:-1, 1:-1]
    horizontal = bad[1:-1, :-2] | bad[1:-1, 2:]
    vertical = bad[:-2, 1:-1] | bad[2:, 1:-1]
    diagonal = bad[:-2, :-2] | bad[:-2, 2:] | bad[2:, :-2] | bad[2:, 2:]

    # Absolute differences in unsigned arithmetic, without overflow.
    left, right = mosaic[1:-1, :-2], mosaic[1:-1, 2:]
    up, down = mosaic[:-2, 1:-1], mosaic[2:, 1:-1]
    larger_across = np.maximum(left, right) - np.minimum(left, right) > np.maximum(
        up, down
    ) - np.minimum(up, down)
    green = np.where(larger_across, vertical, horizontal)
    green[[0, -1], :] = True
    green[:, [0, -1]] = True

    rows, cols = np.ogrid[1 : mosaic.shape[0] - 1, 1 : mosaic.shape[1] - 1]
    own = phase[rows % 2, cols % 2]
    beside = phase[rows % 2, (cols + 1) % 2]
    invalid = np.empty((*mosaic.shape, 3), dtype=bool)
    for channel in range(3):
        paired = np.where(beside == channel, horizontal, vertical)
        other = green if channel == 1 else diagonal
        invalid[1:-1, 1:-1, channel] = np.where(
            own == channel, centre, np.where(own == 1, paired, other)
        )
    invalid[1:-1, 0] = invalid[1:-1, 1]
    invalid[1:-1, -1] = invalid[1:-1, -2]
    invalid[0] = invalid[1]
    invalid[-1] = invalid[-2]
    return invalid
