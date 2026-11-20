import struct
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sps

import darsia
import darsia.measure.beckmann_linalg as linalg
from darsia.measure.beckmann_linalg import _darcy_lu
from darsia.measure.beckmann_problem import MobilityMode, _profile_memory
from darsia.measure.wasserstein import _matching_grid


def _pair():
    first = np.zeros((8, 9), dtype=float)
    first[2:5, 2:5] = 1.0
    second = np.roll(first, 2, axis=1)
    return [
        darsia.ScalarImage(values, dimensions=[0.024, 0.027])
        for values in (first, second)
    ]


def test_reused_grid_darcy_and_pressure_system_preserve_the_bregman_solution():
    _matching_grid.cache_clear()
    _darcy_lu.cache_clear()
    first, second = _pair()
    options = {"return_info": True, "num_iter": 8, "bregman_update": lambda _: False}
    complete, info = darsia.wasserstein_distance(
        first, second, method="bregman", options=options
    )
    reduced, detail = darsia.wasserstein_distance(
        first, second, method="bregman", options={**options, "return_pressure": False}
    )
    assert struct.pack(">d", complete) == struct.pack(">d", reduced)
    assert detail["number_iterations"] == info["number_iterations"]
    assert detail["converged"] == info["converged"]
    assert "pressure" in info and "pressure" not in detail
    np.testing.assert_array_equal(detail["flux"], info["flux"])
    assert _matching_grid.cache_info().hits == _darcy_lu.cache_info().hits == 1
    assert info["grid"] is not detail["grid"]
    assert info["grid"].shape == first.num_voxels


def test_cached_darcy_lu_preserves_the_direct_solvers_csc_state():
    matrix = sps.csc_matrix(
        (
            [0.2, 2.0, 0.1, 0.3, 3.0, 4.0, 0.1],
            [1, 0, 2, 2, 1, 2, 0],
            [0, 3, 5, 7],
        ),
        shape=(3, 3),
    )
    rhs = np.array([7.0, 3.0, 1.0])
    reference, actual = matrix.copy(), matrix.copy()
    plain = darsia.BeckmannDirectSolver({})
    cached = darsia.BeckmannDirectSolver({})
    plain.setup(reference)
    for actual in (actual, matrix.copy()):
        cached.setup(actual, cache_darcy=True)
        assert actual.has_canonical_format == reference.has_canonical_format
        assert actual.has_sorted_indices == reference.has_sorted_indices
        for name in ("data", "indices", "indptr"):
            assert getattr(actual, name).tobytes() == getattr(reference, name).tobytes()
        assert cached(rhs).tobytes() == plain(rhs).tobytes()


@pytest.mark.parametrize("failure", ["factorization", "residual"])
def test_symmetric_pressure_ordering_falls_back_and_reports(failure, monkeypatch):
    matrix = sps.csc_matrix([[4.0, -1.0], [-1.0, 3.0]])
    rhs = np.array([1.0, 5.0])
    direct = darsia.BeckmannDirectSolver({})
    direct.setup(matrix.copy())
    expected = direct(rhs)
    original = linalg._direct_factor

    def factor(system, symmetric):
        if not symmetric:
            return original(system, False)
        if failure == "factorization":
            raise RuntimeError("controlled symmetric failure")
        return SimpleNamespace(solve=lambda values: np.zeros_like(values))

    monkeypatch.setattr(linalg, "_direct_factor", factor)
    solver = darsia.BeckmannDirectSolver({"direct_solver_ordering": "symmetric"})
    solver.setup(matrix.copy(), symmetric_pressure=True)
    assert solver(rhs).tobytes() == expected.tobytes()
    assert solver.symmetric_fallbacks == {failure: 1}
    solver.setup(matrix.copy())
    assert solver(rhs).tobytes() == expected.tobytes()
    assert solver.symmetric_fallbacks == {failure: 1}


def test_symmetric_ordering_option_is_applied_only_to_bregman_pressure():
    first, second = _pair()
    options = {
        "return_solver_info": True,
        "num_iter": 8,
        "direct_solver_ordering": "symmetric",
    }
    _, pressure = darsia.wasserstein_distance(
        first, second, method="bregman", options=options
    )
    _, mixed = darsia.wasserstein_distance(
        first, second, method="bregman", options={**options, "formulation": "full"}
    )
    assert pressure["symmetric_factorizations"] > 0
    assert mixed["symmetric_factorizations"] == 0


@pytest.mark.parametrize(
    "mobility_mode", [MobilityMode.CELL_BASED, MobilityMode.CELL_BASED_ARITHMETIC]
)
def test_bregman_reuses_only_the_density_of_the_same_flux(mobility_mode):
    first, second = _pair()
    options = {
        "return_info": True,
        "num_iter": 8,
        "bregman_update": lambda iteration: iteration in (0, 3, 5),
        "mobility_mode": mobility_mode,
    }
    observed, reused = darsia.wasserstein_distance(
        first, second, method="bregman", options=options
    )
    reference, independent = darsia.wasserstein_distance(
        first,
        second,
        method="bregman",
        options={**options, "callbacks": [lambda _: None]},
    )
    assert struct.pack(">d", observed) == struct.pack(">d", reference)
    assert reused["number_iterations"] == independent["number_iterations"]
    assert reused["converged"] == independent["converged"]
    for key in ("flux", "pressure"):
        assert reused[key].tobytes() == independent[key].tobytes()
    assert (
        np.asarray(reused["convergence_history"]["distance"]).tobytes()
        == np.asarray(independent["convergence_history"]["distance"]).tobytes()
    )


@pytest.mark.parametrize("shape", [(1, 7), (17, 19)])
@pytest.mark.parametrize("weighted", [False, True])
def test_rt0_buffers_preserve_quadrature_bytes(shape, weighted):
    rng = np.random.default_rng(8322)
    grid = darsia.Grid(list(shape), [0.003, 0.003])
    weight = (
        darsia.ScalarImage(
            rng.uniform(0.5, 12, shape).astype(np.float32),
            dimensions=[length * 0.003 for length in shape],
        )
        if weighted
        else None
    )
    solver = darsia.BeckmannBregmanSolver(grid, weight)
    flux = np.ldexp(
        rng.standard_normal(grid.num_faces), rng.integers(-570, 450, grid.num_faces)
    )
    flux[::7] = -0.0
    density = solver.transport_density(flux, flatten=False)
    solver.callbacks = [lambda _: None]
    reference = solver.transport_density(flux, flatten=False)
    assert density.tobytes() == reference.tobytes()


def test_memory_profiling_stops_after_a_solve_and_its_exception():
    pref = SimpleNamespace(options={})

    @_profile_memory
    def inspect(solver, might_fail=False):
        enabled = tracemalloc.is_tracing()
        if might_fail:
            raise RuntimeError("probe")
        return enabled, None, {}

    assert not tracemalloc.is_tracing()
    assert inspect(pref) == (False, None, {"peak_memory_consumption": 0.0})
    pref.options["profile_memory"] = True
    assert inspect(pref)[0]
    assert not tracemalloc.is_tracing()
    with pytest.raises(RuntimeError, match="probe"):
        inspect(pref, might_fail=True)
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        assert inspect(pref)[0]
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()
