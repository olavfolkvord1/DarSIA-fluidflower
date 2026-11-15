"""Unit tests for depth measurements interpolated linearly inside their hull."""

import numpy as np
import pytest

import darsia


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linear_inside_the_hull_and_the_nearest_measurement_outside(dtype):
    """A plane is reproduced inside the hull, the nearest measurement outside."""
    x, y = np.array([[0, 2, 0, 2.2, 1.1, 0.9], [0, 0.25, 1, 1.9, 0.9, -0.1]])
    measurements = (x, y, 1 + x + 2 * y)
    points = np.array(
        [[0.5, 0.5], [1.5, 0.6], [1.0, 1.2], [2.2, 1.9]]
        + [[3.0, 0.2], [1.0, -2.0], [-1.0, 0.5]],
        dtype=dtype,
    )

    values, outside = darsia.interpolate_linear_in_hull(measurements, points)

    assert values.dtype == dtype
    np.testing.assert_array_equal(outside, [False] * 4 + [True] * 3)
    np.testing.assert_allclose(values[:4], [2.5, 3.7, 4.4, 7.0], rtol=1e-6)
    # The last point is as near to (0, 0) as to (0, 1): the first one counts.
    np.testing.assert_array_equal(values[4:], np.array([3.5, 1.7, 1.0], dtype))
