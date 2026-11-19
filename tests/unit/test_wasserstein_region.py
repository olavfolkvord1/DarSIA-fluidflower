"""A transport region in the Beckmann solver: no flux leaves it."""

import numpy as np
import pytest

import darsia

OPTIONS = {
    "mobility_mode": darsia.MobilityMode.CELL_BASED,
    "l1_mode": darsia.L1Mode.RAVIART_THOMAS,
    "bregman_update": lambda iter: iter in [0, 10, 20],
    "num_iter": 200,
    "tol_distance": 1e-8,
    "tol_increment": 1e-6,
    "linear_solver": "direct",
    "return_info": True,
}
ROWS, COLS = 6, 9


def _image(array):
    """A scalar image of unit cells, top row first."""
    rows, cols = np.shape(array)
    return darsia.ScalarImage(
        np.asarray(array, float),
        dimensions=[float(rows), float(cols)],
        origin=[0.0, float(rows)],
    )


def _solve(source, target, weight, region=None):
    options = OPTIONS if region is None else dict(OPTIONS, region=region)
    return darsia.wasserstein_distance(
        _image(source),
        _image(target),
        weight=_image(weight),
        method="bregman",
        options=options,
    )


def test_a_region_blocks_the_path_a_finite_penalty_allows():
    # Two arms joined only along the bottom; the gap between them is outside.
    region = np.zeros((ROWS, COLS), bool)
    region[:, 0:2] = region[:, 7:9] = region[4:6, :] = True
    source, target = np.zeros((ROWS, COLS)), np.zeros((ROWS, COLS))
    source[0, 0] = target[0, 8] = 1.0
    weight = 1.0 + np.arange(ROWS * COLS).reshape(ROWS, COLS) / 10
    free, _ = _solve(source, target, weight)
    every, _ = _solve(source, target, weight, region=np.ones((ROWS, COLS), bool))
    assert np.float64(every).tobytes() == np.float64(free).tobytes()

    straight, _ = _solve(source, target, np.ones((ROWS, COLS)))
    masked, info = _solve(source, target, np.where(region, 1.0, np.nan), region)
    assert info["converged"]
    assert np.all(info["transport_density"][~region] == 0)
    assert masked > 1.6 * straight
    penalised, _ = _solve(source, target, np.where(region, 1.0, 2.0))
    assert penalised < masked


def test_each_part_of_a_region_moves_its_own_mass():
    region = np.zeros((ROWS, COLS), bool)
    region[:, 0:4] = region[:, 5:9] = True
    source, target = np.zeros((ROWS, COLS)), np.zeros((ROWS, COLS))
    source[0, 0] = source[0, 5] = target[5, 3] = target[5, 8] = 0.5
    together, _ = _solve(source, target, np.ones((ROWS, COLS)), region)
    left, _ = _solve(source[:, 0:4], target[:, 0:4], np.ones((ROWS, 4)))
    right, _ = _solve(source[:, 5:9], target[:, 5:9], np.ones((ROWS, 4)))
    assert together == pytest.approx(left + right, rel=1e-9)

    target[5, 3], target[5, 8] = 0.0, 1.0
    with pytest.raises(ValueError, match="different masses"):
        _solve(source, target, np.ones((ROWS, COLS)), region)
