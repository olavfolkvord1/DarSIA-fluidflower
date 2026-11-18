"""The hidden bridge shares the visible volume solve and preserves optics."""

from types import SimpleNamespace

import numpy as np
import pytest

from darsia.multiphase import gas_allocation as allocation
from darsia.multiphase.gas_bridge import hidden_bridges
from darsia.multiphase.phase_rules import PhaseLabel, PhaseRule
from tests.unit.test_gas_allocation import SETTINGS, _frame, _run


def scene():
    shape = (70, 90)
    co2 = np.zeros(shape, bool)
    co2[10:41, 32:59] = True
    gas = np.zeros(shape, bool)
    gas[10:20, 40:50] = True
    points = np.column_stack((np.arange(55, 14, -1), np.full(41, 45))).astype(np.int32)
    zone = SimpleNamespace(
        contour=np.ones(shape, bool),
        links={0: SimpleNamespace(path_coordinates=points)},
    )
    bridges = hidden_bridges(zone, co2, ~co2, np.ones(shape, bool), (1, 1), SETTINGS)
    assert len(bridges) == 1 and bridges[0].meeting == (40, 45)
    hidden = bridges[0].hidden
    cap = np.full(shape, 0.8)
    saturation = np.where(gas, cap, 0)
    limit = co2 | hidden
    return co2, gas, bridges, cap, saturation, limit


@pytest.mark.parametrize("volume", [30.0, 180.0])
def test_hidden_fraction_copies_meeting_and_total_volume_includes_bridge(volume):
    co2, gas, bridges, cap, saturation, limit = scene()
    hidden = bridges[0].hidden
    allocation._add(
        saturation,
        volume,
        gas,
        limit & ~gas,
        cap,
        cap,
        (co2, np.where(co2, 0.4, 0)),
        gas,
        allocation._inward_weight(limit, (1, 1), SETTINGS.fade_length_mm),
        (1, 1),
        SETTINGS,
        bridges=bridges,
    )
    assert np.isclose(saturation[~gas].sum(), volume, rtol=1e-7)
    np.testing.assert_allclose(saturation[42:56, 45], saturation[40, 45], rtol=1e-7)
    assert 0 < saturation[50, 46] < saturation[50, 45]
    assert not saturation[~limit].any()
    assert np.all(saturation <= cap)
    assert saturation[hidden].sum() > 0


def test_taper_survives_a_capped_meeting_point():
    co2, gas, bridges, cap, saturation, limit = scene()
    allocation._add(
        saturation,
        1e8,
        gas,
        limit & ~gas,
        cap,
        cap,
        (co2, np.where(co2, 0.4, 0)),
        gas,
        allocation._inward_weight(limit, (1, 1), SETTINGS.fade_length_mm),
        (1, 1),
        SETTINGS,
        bridges=bridges,
    )
    assert saturation[40, 45] == cap[40, 45]
    assert saturation[50, 45] == cap[50, 45]
    np.testing.assert_allclose(
        saturation[bridges[0].hidden],
        cap[bridges[0].hidden] * bridges[0].profile[bridges[0].hidden],
    )


def test_no_link_or_unknown_support_cannot_create_a_hidden_bridge():
    co2, _, bridges, _, _, _ = scene()
    empty = SimpleNamespace(contour=np.ones(co2.shape, bool), links={})
    assert not hidden_bridges(empty, co2, ~co2, np.ones_like(co2), (1, 1), SETTINGS)
    assert not (bridges[0].hidden & co2).any()
    assert not hidden_bridges(
        SimpleNamespace(
            contour=empty.contour,
            links={
                0: SimpleNamespace(
                    path_coordinates=np.array([[55, 45], [40, 45]], np.int32)
                )
            },
        ),
        co2,
        np.zeros_like(co2),
        np.ones_like(co2),
        (1, 1),
        SETTINGS,
    )


def test_bridge_is_optically_water_and_disappears_after_injection(tmp_path):
    frame = _frame([np.s_[10:40, 20:60]], [np.s_[20:30, 30:50]])
    records = []
    maps = _run(
        tmp_path,
        [frame, frame],
        [0, 3],
        [3e-6, 3e-6],
        end_h=2,
        record=lambda index, record: records.append(record),
    )
    labels, rules, saturation, optical_co2, before = maps[0]
    bridge = rules == PhaseRule.GAS_BRIDGE
    assert bridge.any() and (saturation[bridge] > 0).all()
    assert (labels[bridge] == PhaseLabel.WATER).all()
    assert (before[bridge] == PhaseLabel.WATER).all()
    assert not optical_co2[bridge].any()
    assert records[0].bridge_area_mm2 == int(bridge.sum())
    assert records[0].bridge_length_mm > 0 and records[0].bridge_volume_m3 > 0
    assert np.isclose(records[0].placed_m3, records[0].target_m3, rtol=1e-7)
    assert records[0].disconnected_added_area_mm2 == 0
    assert not (maps[1][1] == PhaseRule.GAS_BRIDGE).any()
    assert not maps[1][2][bridge].any()
    assert records[1].bridge_volume_m3 == 0
