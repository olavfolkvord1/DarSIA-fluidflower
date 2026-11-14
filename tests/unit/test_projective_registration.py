"""Unit tests for the projective registration to the run grid."""

import numpy as np

import darsia

H = np.array(
    [
        [0.98755, 0.00516, 57.013],
        [-0.000136, 0.99112, 43.001],
        [2.5e-6, 9.7e-6, 1.0],
    ]
)
"""Homography from the run grid to the camera image, like a setup's."""

RANSAC = {
    "confidence": 0.995,
    "iterations": 2000,
    "refinement_steps": 10,
    "refinement_tolerance": float(np.finfo(np.float32).eps),
}
"""The RANSAC controls our production ran with (OpenCV's)."""


def _project(h: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Homography applied to pixel centres, in double precision."""
    den = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / den, (
        h[1, 0] * x + h[1, 1] * y + h[1, 2]
    ) / den


def test_ransac_recovers_a_known_homography_among_outliers():
    """The inliers are the true matches, and their homography is recovered."""
    rng = np.random.default_rng(1)
    source = rng.uniform([0, 0], [1100, 640], (300, 2))
    target = np.column_stack(_project(H, source[:, 0], source[:, 1]))
    target += 0.05 * rng.standard_normal(target.shape)
    outlier = rng.random(300) < 0.3
    target[outlier] += rng.uniform(20, 60, (outlier.sum(), 2))

    for dtype in (np.float32, np.float64):
        h, inliers = darsia.ransac_homography(
            source.astype(dtype), target.astype(dtype), 1.0, **RANSAC
        )
        assert h.dtype == dtype
        np.testing.assert_array_equal(inliers[:, 0] == 1, ~outlier)
        mapped = np.column_stack(_project(h.astype(float), *source.T))
        exact = np.column_stack(_project(H, *source.T))
        assert np.abs(mapped - exact).max() < 0.05
