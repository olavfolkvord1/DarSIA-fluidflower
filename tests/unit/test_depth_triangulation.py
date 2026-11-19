"""Unit tests for depth measurements interpolated linearly inside their hull."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.presets.workflows.setup.setup_depth import setup_depth_map


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


@pytest.mark.parametrize("interpolation", [None, "triangulation"])
def test_the_depth_map_follows_the_configured_interpolation(tmp_path, interpolation):
    """The setup interpolates as [depth].interpolation says, by RBF without it."""
    x, y = np.meshgrid([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    table = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "depth": (x * y).ravel()})
    table.to_csv(tmp_path / "depth.csv", index=False)
    choice = "" if interpolation is None else f'interpolation = "{interpolation}"\n'
    (tmp_path / "config.toml").write_text(
        f'[depth]\nmeasurements = "{(tmp_path / "depth.csv").as_posix()}"\n'
        f'depth_map = "{(tmp_path / "depth_map.npz").as_posix()}"\n'
        f"resolution = [6, 8]\n{choice}"
        f'[rig]\ndim = 2\nwidth = 1.0\nheight = 1.0\npath = "{tmp_path.as_posix()}"\n'
    )

    setup_depth_map(tmp_path / "config.toml", key="depth")

    proxy = darsia.Image(
        img=np.zeros((6, 8)), width=1.0, height=1.0, scalar=True, space_dim=2
    )
    maps = {
        method: darsia.interpolate_to_image_from_csv(
            tmp_path / "depth.csv", "depth", proxy, method=method
        ).img
        for method in ("rbf", "triangulation")
    }
    depth_map = darsia.imread(Path(tmp_path / "depth_map.npz"))
    np.testing.assert_array_equal(depth_map.img, maps[interpolation or "rbf"])
    assert not np.allclose(maps["rbf"], maps["triangulation"])
