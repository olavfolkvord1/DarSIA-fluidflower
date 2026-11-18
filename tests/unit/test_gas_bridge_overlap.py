"""Refuse undefined bridge interference and preserve fixed model-gas meetings."""

from types import SimpleNamespace

import numpy as np
import pytest

from darsia.multiphase import gas_allocation as allocation
from darsia.multiphase.gas_bridge import GasBridge, bridge_scores, hidden_bridges
from darsia.multiphase.gas_parameters import GasAllocationSettings, parameters_at_depth

SETTINGS = parameters_at_depth(GasAllocationSettings(1, 1, 1, 1, 1, 1, 1, 1), 5)


def scene(columns):
    shape = (70, 90)
    co2 = np.zeros(shape, bool)
    co2[10:41, 10:80] = True
    links = {
        index: SimpleNamespace(
            path_coordinates=np.column_stack(
                (np.arange(55, 39, -1), np.full(16, col))
            ).astype(np.int32)
        )
        for index, col in enumerate(columns)
    }
    bridges = hidden_bridges(
        SimpleNamespace(contour=np.ones(shape, bool), links=links),
        co2,
        ~co2,
        np.ones(shape, bool),
        (1, 1),
        SETTINGS,
    )
    base = np.full(shape, 0.2)
    base[:, columns[-1] :] = 0.8
    return co2, bridges, base


def solve(monkeypatch, co2, bridges, base, volume, gas=None):
    gas = np.zeros_like(co2) if gas is None else gas
    hidden = np.logical_or.reduce([bridge.hidden for bridge in bridges])
    saturation = gas.astype(float)
    monkeypatch.setattr(allocation, "_round_score", lambda *args: base.copy())
    allocation._add(
        saturation,
        volume,
        gas,
        (co2 | hidden) & ~gas,
        np.ones(co2.shape),
        np.ones(co2.shape),
        (co2, base),
        co2,
        np.ones(co2.shape),
        (1, 1),
        SETTINGS,
        bridges=bridges,
    )
    return saturation


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_active_hidden_fields_refuse_before_allocation(
    monkeypatch, reverse
):
    co2, bridges, base = scene((44, 46))
    if reverse:
        bridges.reverse()
    with pytest.raises(ValueError, match="incompatible overlapping hidden support"):
        solve(monkeypatch, co2, bridges, base, 100)


def test_distinct_hidden_fields_still_refuse_meeting_collar_interference():
    shape = (1, 5)
    a = np.array([[True, False, False, False, False]])
    b = a[:, ::-1].copy()
    profile = np.ones(shape, np.float32)
    bridges = [
        GasBridge(a, (0, 1), profile, np.array([[0, 1, 0, 0.2, 0]], np.float32), 1),
        GasBridge(b, (0, 3), profile, np.array([[0, 0.2, 0, 1, 0]], np.float32), 1),
    ]
    with pytest.raises(ValueError, match="transition changes another bridge's meeting"):
        bridge_scores(
            np.ones(shape), np.ones(shape, bool), np.zeros(shape, bool), bridges
        )


@pytest.mark.parametrize("other_fixed", [False, True])
def test_fixed_gas_meetings_do_not_receive_visible_collar_blends(other_fixed):
    shape = (1, 5)
    a = np.array([[True, False, False, False, False]])
    b = a[:, ::-1].copy()
    profile = np.ones(shape, np.float32)
    gas = np.array([[False, True, False, other_fixed, False]])
    candidate = a | b
    candidate[0, 3] = not other_fixed
    base = np.zeros(shape)
    base[0, 3] = 1
    bridges = [
        GasBridge(a, (0, 1), profile, np.array([[0, 1, 0, 0.2, 0]], np.float32), 1),
        GasBridge(b, (0, 3), profile, np.array([[0, 0.2, 0, 1, 0]], np.float32), 1),
    ]
    _, fixed, _ = bridge_scores(base, candidate, gas, bridges)
    np.testing.assert_array_equal(fixed, (a | (b & other_fixed)).astype(float))


def test_separated_bridges_preserve_each_meeting_and_exact_total(monkeypatch):
    co2, bridges, base = scene((20, 60))
    result = solve(monkeypatch, co2, bridges, base, 100)
    assert np.isclose(result.sum(), 100, rtol=1e-12)
    for column in (20, 60):
        np.testing.assert_allclose(
            result[42:56, column], result[40, column], rtol=1e-12
        )
    np.testing.assert_array_equal(
        result, solve(monkeypatch, co2, bridges[::-1], base, 100)
    )


def test_identical_fields_are_deduplicated_without_reweighting(monkeypatch):
    co2, bridges, base = scene((45,))
    bridge = bridges[0]
    duplicate = GasBridge(
        bridge.hidden.copy(),
        bridge.meeting,
        bridge.profile.copy(),
        bridge.transition.copy(),
        bridge.length_mm,
    )
    original = solve(monkeypatch, co2, bridges, base, 100)
    np.testing.assert_array_equal(
        original, solve(monkeypatch, co2, [bridge, duplicate], base, 100)
    )


@pytest.mark.parametrize("meeting_score", [0, np.nan])
def test_fixed_gas_meeting_needs_no_invented_positive_score(monkeypatch, meeting_score):
    co2, bridges, base = scene((45,))
    bridge = bridges[0]
    gas = np.zeros_like(co2)
    gas[bridge.meeting] = True
    base[bridge.meeting] = meeting_score
    fixed_volume = float(bridge.profile[bridge.hidden].sum(dtype=np.float64))
    result = solve(monkeypatch, co2, bridges, base, fixed_volume + 10, gas)
    assert np.isclose(result[~gas].sum(), fixed_volume + 10, rtol=1e-12)
    assert result[bridge.meeting] == 1
    np.testing.assert_array_equal(result[bridge.hidden], bridge.profile[bridge.hidden])
    assert np.all(result <= 1) and np.all(np.isfinite(result))
    with pytest.raises(ValueError, match="cannot fill the fixed gas bridge"):
        solve(monkeypatch, co2, bridges, base, fixed_volume / 2, gas)
