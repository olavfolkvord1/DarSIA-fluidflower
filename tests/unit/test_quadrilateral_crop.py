"""Tests of the quadrilateral crop in a chosen precision."""

import cv2
import numpy as np
import scipy.ndimage

import darsia


def test_a_crop_in_a_precision_samples_its_corner_transform_solved_in_it():
    """In single precision, the crop samples the image with OpenCV's remap and a
    zero border at the positions of its corner transform, solved from its eight
    projective equations and inverted in single precision; by default it is
    OpenCV's perspective transform and warp, as before."""
    rng = np.random.default_rng(7)
    image = cv2.GaussianBlur(rng.random((300, 450)), (0, 0), 2.0).astype(np.float32)
    corners = [[13, -12], [287, 6], [281, 447], [9, 431]]  # (row, column)
    crop = dict(pts_src=darsia.make_voxel(corners), width=1.5, height=1.0)

    cropped = darsia.extract_quadrilateral_ROI(image, precision=np.float32, **crop)

    rows, cols = cropped.shape
    h = darsia.rig_points_homography(np.array(corners), (rows, cols), np.float32)
    x, y = np.arange(cols, dtype=np.float32)[None, :], np.arange(rows)[:, None]
    y = y.astype(np.float32)
    den = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    sx = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / den
    sy = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / den
    expected = cv2.remap(
        image, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    np.testing.assert_array_equal(cropped, expected)

    default = darsia.extract_quadrilateral_ROI(image, **crop)
    source = np.fliplr(np.array(corners)).astype(np.float32)
    target = np.float32([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]])
    warp = cv2.getPerspectiveTransform(source, target)
    np.testing.assert_array_equal(
        default, cv2.warpPerspective(image, warp, (cols, rows), flags=cv2.INTER_LINEAR)
    )
    assert not np.array_equal(default, cropped)


def test_the_curvature_crop_is_computed_in_its_configured_precision():
    """A curvature correction's crop with a precision crops its coordinate meshes
    in that precision, and samples the image at them, as without one."""
    rng = np.random.default_rng(8)
    image = cv2.GaussianBlur(rng.random((300, 450)), (0, 0), 2.0).astype(np.float32)
    corners = [[13, -12], [287, 6], [281, 447], [9, 431]]
    names = ("top_left", "bottom_left", "bottom_right", "top_right")
    crop = dict(zip(names, corners), width=1.5, height=1.0, precision="float32")
    stages = {"init": None, "crop": crop, "bulge": None, "stretch": None}

    corrected = darsia.CurvatureCorrection(config=stages).correct_array(image)

    x, y = np.meshgrid(
        np.arange(450, dtype=np.float32), np.arange(300, dtype=np.float32)
    )
    shape = dict(pts_src=darsia.make_voxel(corners), width=1.5, height=1.0)
    cx, cy = (
        darsia.extract_quadrilateral_ROI(mesh, precision=np.float32, **shape)
        for mesh in (x, y)
    )
    expected = scipy.ndimage.map_coordinates(image, [cy.ravel(), cx.ravel()], order=1)
    np.testing.assert_array_equal(corrected, expected.reshape(cx.shape))
    default = {**stages, "crop": {k: v for k, v in crop.items() if k != "precision"}}
    assert not np.array_equal(
        darsia.CurvatureCorrection(config=default).correct_array(image), corrected
    )


def test_a_warp_in_double_precision_is_opencvs_perspective_warp():
    """In double precision, as without a precision, a warp is OpenCV's perspective
    warp with a zero border, and a crop OpenCV's perspective transform and warp,
    the calculation our production made without a chosen precision; in single
    precision the positions are mapped in single precision and sampled by remap."""
    rng = np.random.default_rng(9)
    image = cv2.GaussianBlur(rng.random((300, 450)), (0, 0), 2.0).astype(np.float32)
    angle = 0.01
    matrix = np.array(
        [
            [1.002 * np.cos(angle), -np.sin(angle), 6.3],
            [np.sin(angle), 0.998 * np.cos(angle), -4.1],
            [2e-6, -1e-6, 1.0],
        ]
    )
    expected = cv2.warpPerspective(
        image,
        matrix,
        (450, 300),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    for precision in (None, np.float64, "float64"):
        warped = darsia.warp_perspective(image, matrix, (450, 300), precision)
        np.testing.assert_array_equal(warped, expected)
    single = matrix.astype(np.float32)
    inverse = darsia.linalg.solve(single, np.eye(3, dtype=np.float32))
    x = np.arange(450, dtype=np.float32)[None, :]
    y = np.arange(300, dtype=np.float32)[:, None]
    den = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
    sx = (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]) / den
    sy = (inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]) / den
    remapped = cv2.remap(
        image, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    found = darsia.warp_perspective(image, single, (450, 300), np.float32)
    np.testing.assert_array_equal(found, remapped)
    assert not np.array_equal(found, expected)

    corners = [[13, -12], [287, 6], [281, 447], [9, 431]]
    crop = dict(pts_src=darsia.make_voxel(corners), width=1.5, height=1.0)
    np.testing.assert_array_equal(
        darsia.extract_quadrilateral_ROI(image, precision=np.float64, **crop),
        darsia.extract_quadrilateral_ROI(image, **crop),
    )
    names = ("top_left", "bottom_left", "bottom_right", "top_right")
    table = dict(zip(names, corners), width=1.5, height=1.0)
    stages = {"init": None, "bulge": None, "stretch": None}
    double = {**stages, "crop": {**table, "precision": "float64"}}
    np.testing.assert_array_equal(
        darsia.CurvatureCorrection(config=double).correct_array(image),
        darsia.CurvatureCorrection(config={**stages, "crop": table}).correct_array(
            image
        ),
    )
