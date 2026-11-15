"""
Module containing auxiliary methods to extract ROIs from darsia Images.
"""

from typing import Literal, Optional, Union
from warnings import warn

import cv2
import numpy as np

import darsia


def _to_reverse_matrix_indexing(pixel: np.ndarray) -> np.ndarray:
    """Auxiliary routine: Convert pixel coordinates to reverse matrix indexing format.

    Parameters
    ----------
    pixel : np.ndarray
        Pixel coordinates.

    Returns
    -------
        Pixel converted to 'reverse matrix' indexing.
    """
    return np.fliplr(np.atleast_2d(pixel))


InterpolationOption = Literal["inter_nearest", "inter_linear", "inter_area"]
"""Interpolation options considered in below warp."""


def quadrilateral_transform(
    source: np.ndarray, target: np.ndarray, dtype: np.dtype | type
) -> np.ndarray:
    """The perspective transform mapping four points onto four others, solved from
    its eight projective equations in a given floating precision.

    Parameters
    ----------
    source, target : array
        Four points each, as (column, row) rows.
    dtype : numpy dtype or type
        Floating precision of the equations and their solution.

    Returns
    -------
    array
        3 x 3 homography of (column, row, 1) from the source to the target points.

    """
    dtype = np.dtype(dtype)
    source, target = np.asarray(source, dtype), np.asarray(target, dtype)
    xy1 = np.column_stack((source, np.ones(4, dtype=dtype)))
    system = np.zeros((8, 8), dtype=dtype)
    system[::2, :3], system[1::2, 3:6] = xy1, xy1
    system[::2, 6:] = -source * target[:, :1]
    system[1::2, 6:] = -source * target[:, 1:]
    values = darsia.linalg.solve(system, target.ravel())
    return np.append(values, dtype.type(1)).reshape(3, 3)


def project_xy(
    matrix: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map positions through a homography, elementwise and without homogeneous
    arrays, in the precision of the matrix and the positions.

    Parameters
    ----------
    matrix : array
        3 x 3 homography of (x, y, 1).
    x, y : array
        Positions, broadcast against each other, e.g. a row of columns and a
        column of rows.

    Returns
    -------
    tuple of arrays
        The mapped x and y; not finite where a position maps to infinity.

    """
    den = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / den,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / den,
        )


def warp_perspective(
    image: np.ndarray,
    matrix: np.ndarray,
    size: tuple[int, int],
    precision: np.dtype | type | str | None = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Warp an image by a homography onto a grid of a given size, zero outside.

    In single precision, the inverse of the homography is solved in single
    precision, each pixel centre of the grid is mapped through it in single
    precision (:func:`project_xy`), and the image is sampled there by OpenCV's
    remap. In double precision or without a precision, it is OpenCV's perspective
    warp, the calculation our production made without a chosen precision.

    Parameters
    ----------
    image : array
        Image (rows, cols) or (rows, cols, channels); single precision for the
        remap.
    matrix : array
        3 x 3 homography from the image's pixel centres (column, row, 1) to the
        grid's.
    size : tuple of int
        Width and height of the grid.
    precision : numpy dtype, optional
        Floating precision of the calculation.
    interpolation : int
        OpenCV's interpolation flag, bilinear by default.

    Returns
    -------
    array
        The warped image on the grid.

    """
    width, height = size
    if precision is None or np.dtype(precision) != np.float32:
        return cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    matrix = np.asarray(matrix, np.float32)
    inverse = darsia.linalg.solve(matrix, np.eye(3, dtype=matrix.dtype))
    x = np.arange(width, dtype=matrix.dtype)[None, :]
    y = np.arange(height, dtype=matrix.dtype)[:, None]
    map_x, map_y = project_xy(inverse, x, y)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def extract_quadrilateral_ROI(
    img_src: np.ndarray,
    pts_src: Optional[Union[list, darsia.VoxelArray, np.ndarray]] = None,
    indexing: Literal["matrix", "reverse matrix"] = "reverse matrix",
    interpolation: InterpolationOption = "inter_linear",
    **kwargs,
) -> np.ndarray:
    """Extract a quadrilateral ROI via a perspective transform.

    The corner points of a (by default square) object are assumed known.

    Parameters
    ----------
    img_src : np.ndarray
        Source image.
    pts_src : array, optional
        N points in pixels; several indexing formats are accepted.
    indexing : {"matrix", "reverse matrix"}
        Indexing of the pixels (only relevant when ``pts_src`` is a list or
        ``np.ndarray``).
    interpolation : InterpolationOption
        Interpolation method, adopted from OpenCV.
    **kwargs
        Physical geometry of the target object::

            width (int | float)      width of the physical object
            height (int | float)     height of the physical object
            pts_src (array)          N >= 4 points in (col, row) pixel format
            pts_dst (array)          N >= 4 points in (col, row) pixel format
            precision (dtype)        in single precision, the transform is
                                     solved from its four corners in single
                                     precision (quadrilateral_transform) and
                                     the image warped in it (warp_perspective);
                                     in double precision or by default,
                                     OpenCV's perspective transform and warp
    """

    # FIXME: Implementation hardcoded for 2d.

    # ! ---- Properties of image

    scalar = len(img_src.shape) == 2

    # ! ---- Size of new image

    # Determine current dimensions and use as default size.
    original_shape = img_src.shape[:2]

    if "width" in kwargs and "height" in kwargs:
        # Determine target image dimensions based on physical units.
        target_width: Optional[float] = kwargs.get("width")
        target_height: Optional[float] = kwargs.get("height")

        # Aim at comparably many pixels as in the provided
        # image, modulo the ratio.
        assert target_width is not None  # required by mypy
        assert target_height is not None  # required by mypy
        aspect_ratio = target_width / target_height

        # Try to keep this aspect ratio, but do not use more pixels than before.
        # Convert to number of pixels
        original_height, original_width = original_shape
        width = min(original_width, int(aspect_ratio * float(original_height)))
        height = min(original_height, int(1.0 / aspect_ratio * float(original_width)))

    else:
        # Determine target image dimensions based on voxel numbers.
        target_shape = kwargs.get("shape", original_shape)
        height, width = target_shape

    # ! ---- Mapping

    # Fetch corner points in the provided image
    if pts_src is None:
        pts_src = darsia.make_voxel(
            [
                [0, 0],
                [original_shape[0], 0],
                [original_shape[0], original_shape[1]],
                [0, original_shape[1]],
            ]
        )
        if indexing == "reverse matrix":
            pts_src = np.fliplr(pts_src)
    else:
        if isinstance(pts_src, darsia.VoxelArray):
            # Voxels use by definition matrix indexing
            pts_src = _to_reverse_matrix_indexing(pts_src)
        else:
            warn("Try using darsia.VoxelArray instead of list or np.ndarray.")
            if isinstance(pts_src, list):
                pts_src = np.array(pts_src)
            if indexing == "matrix":
                pts_src = _to_reverse_matrix_indexing(pts_src)

    # Assign corner points as destination points if none are provided.
    if "pts_dst" in kwargs:
        pts_dst = kwargs.get("pts_dst")
        if isinstance(pts_dst, darsia.VoxelArray):
            # Voxels use by definition matrix indexing
            pts_dst = _to_reverse_matrix_indexing(pts_dst)
        else:
            warn("Try using darsia.VoxelArray instead of list or np.ndarray.")
            if isinstance(pts_dst, list):
                pts_dst = np.array(pts_dst)
            if indexing == "matrix":
                pts_dst = _to_reverse_matrix_indexing(pts_dst)
    else:
        # If no destination points are provided, assume implicitly that
        # the target image is a square with the same size as the source image.
        pts_dst = np.array(
            [
                [0, 0],
                [0, height - 1],
                [width - 1, height - 1],
                [width - 1, 0],
            ]
        )

    assert pts_src.shape[0] == pts_dst.shape[0]
    precision = kwargs.get("precision")
    single = precision is not None and np.dtype(precision) == np.float32
    if single:
        P = quadrilateral_transform(pts_src, pts_dst, np.float32)
    else:
        P = cv2.getPerspectiveTransform(
            pts_src.astype(np.float32), pts_dst.astype(np.float32)
        )

    # ! ---- Warp image

    # Take care of data type - cv2 requires np.float32 objects.
    # However, when using input images with integer dtype, it is
    # intended that it remains like this. One may indeed loose
    # some information. However, since data type changes are
    # challenging to keep track of, the intention is that this
    # routine returns arrays of same dtype again.
    dtype = img_src.dtype

    interpolation_flag = None  # type: ignore
    if interpolation == "inter_nearest":
        interpolation_flag = cv2.INTER_NEAREST
    elif interpolation == "inter_linear":
        interpolation_flag = cv2.INTER_LINEAR
    elif interpolation == "inter_area":
        interpolation_flag == cv2.INTER_AREA
    else:
        raise NotImplementedError

    # Warp source image. Warping may convert a 3-tensor to a 2-tensor.
    # Force to use a 3-tensor structure.
    warped = warp_perspective(
        img_src.astype(np.float32),
        P,
        (width, height),
        np.float32 if single else None,
        interpolation_flag,  # type: ignore
    )
    img_dst = np.atleast_3d(warped).astype(dtype)

    # Reduce to scalar image array, if input provided in such format.
    if scalar:
        img_dst = img_dst[:, :, 0]

    return img_dst


def restrict_image_to_landmarks(
    img: np.ndarray,
    landmarks: "darsia.VoxelArray",
    width: float,
    height: float,
    indexing: Literal["matrix", "reverse matrix"] = "matrix",
) -> np.ndarray:
    """Extract the image region spanned by 4 ordered landmark points.

    The landmarks are assumed anti-clockwise oriented, starting at a known
    reference corner (e.g. the brown swatch of a color checker). If they are
    (approximately) axis-aligned, perform a plain crop and rotate so the
    reference corner lands top-left. Otherwise fall back to a full
    perspective extraction via :func:`extract_quadrilateral_ROI`.

    Parameters
    ----------
    img : np.ndarray
        Source image.
    landmarks : darsia.VoxelArray
        4 ordered corner landmark points, matrix-indexed.
    width, height : float
        Physical dimensions of the target object, used by the perspective
        extraction fallback.
    indexing : {"matrix", "reverse matrix"}
        Forwarded to :func:`extract_quadrilateral_ROI`.

    Returns
    -------
    np.ndarray
        Restricted image.
    """
    row_pixels = np.sort(landmarks[:, 0])
    col_pixels = np.sort(landmarks[:, 1])
    row_diff = max(row_pixels[1] - row_pixels[0], row_pixels[3] - row_pixels[2])
    col_diff = max(col_pixels[1] - col_pixels[0], col_pixels[3] - col_pixels[2])
    roi_is_box = row_diff < 0.01 * img.shape[0] and col_diff < 0.01 * img.shape[1]
    atol = max(0.01 * img.shape[0], 0.01 * img.shape[1])

    if roi_is_box:
        # landmarks are more or less a box
        roi_slices = (
            slice(row_pixels[0], row_pixels[3]),
            slice(col_pixels[0], col_pixels[3]),
        )
        box_img = img[roi_slices]
        # need to extract a box with the reference landmark first - assume
        # the first point in landmarks is the reference corner
        if np.allclose([row_pixels[0], col_pixels[0]], [landmarks[0]], atol=atol):
            # reference corner is in the upper left corner
            return box_img
        elif np.allclose([row_pixels[0], col_pixels[3]], [landmarks[0]], atol=atol):
            # reference corner is in the upper right corner - rotate 90 degrees clockwise
            return np.rot90(box_img, 1)
        elif np.allclose([row_pixels[3], col_pixels[3]], [landmarks[0]], atol=atol):
            # reference corner is in the lower right corner - rotate 180 degrees
            return np.rot90(box_img, -2)
        elif np.allclose([row_pixels[3], col_pixels[0]], [landmarks[0]], atol=atol):
            # reference corner is lower left - rotate 90 degrees counterclockwise
            return np.rot90(box_img, -1)
        else:
            raise ValueError("The reference landmark is not in the corner of the ROI.")

    else:
        return extract_quadrilateral_ROI(
            img, pts_src=landmarks, width=width, height=height, indexing=indexing
        )
