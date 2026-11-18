"""Proposed active-meeting fallback for an otherwise unseeded gas component."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from darsia.multiphase import gas_allocation as allocation
from darsia.multiphase.gas_zone import GasLink, GasZone
from darsia.multiphase.phase_rules import PhaseLabel, PhaseRule
from tests.unit.test_gas_limit import RULES, SETTINGS, START, _model, _scene


def _fixture(*, hidden=True, refusal=None):
    image, positions, domain = _scene((125, 175))
    labels = np.full(domain.shape, PhaseLabel.WATER, np.uint8)
    labels[10:30, 10:30] = PhaseLabel.GAS  # Old large disconnected plume.
    labels[50:90, 100:140] = PhaseLabel.DISSOLVED
    for row in (55, 65):
        for col in (105, 115, 125):
            labels[row : row + 2, col : col + 3] = PhaseLabel.GAS
    meeting = (89, 120) if hidden else (80, 120)
    anchor = (110, 120) if hidden else meeting
    old_port = (float(positions.x[35, 20]), float(positions.y[35, 20]))
    new_port = (float(positions.x[anchor]), float(positions.y[anchor]))
    protocol = SimpleNamespace(
        df=pd.DataFrame(
            [
                [
                    1,
                    *old_port,
                    START - pd.Timedelta(hours=3),
                    START - pd.Timedelta(hours=1),
                ],
                [1, *new_port, START, START + pd.Timedelta(hours=2)],
            ],
            columns=["rate_kg_s", "location_x", "location_y", "start", "end"],
        )
    )
    points = np.column_stack(
        (
            np.arange(anchor[0], meeting[0] - 1, -1),
            np.full(anchor[0] - meeting[0] + 1, meeting[1]),
        )
    ).astype(np.int32)
    line = np.zeros(domain.shape, bool)
    line[points[:, 0], points[:, 1]] = True
    contour = np.ones(domain.shape, bool)
    probability = np.full(domain.shape, 0.4, np.float32)
    swi = np.full(domain.shape, 0.2, np.float32)
    if refusal == "unknown":
        labels[meeting] = PhaseLabel.UNKNOWN
    elif refusal == "seal":
        image.img[meeting] = 2
    elif refusal == "cap_zero":
        swi[meeting] = 1
    elif refusal == "zero_score":
        probability[45:115, 95:145] = 0
    elif refusal == "outside_contour":
        contour[meeting] = False
    link = GasLink(line, np.zeros(domain.shape, np.float32), {}, points)
    zone = GasZone(contour, {new_port: link}, {})
    frame = _model(labels, probability)
    pore = np.full(domain.shape, 1e-9)
    cap = 1 - swi
    model_volume = float(
        np.sum(pore[labels == PhaseLabel.GAS] * cap[labels == PhaseLabel.GAS])
    )
    target = model_volume + 1e-7
    date = START + pd.Timedelta(hours=3 if refusal == "inactive" else 1)
    return dict(
        image=image,
        positions=positions,
        domain=domain,
        protocol=protocol,
        frame=frame,
        zone=zone,
        pore=pore,
        swi=swi,
        target=target,
        date=date,
        meeting=meeting,
        model_volume=model_volume,
    )


def _allocate(scene):
    records = []
    result = list(
        allocation.free_gas(
            [None],
            [scene["date"]],
            scene["image"],
            scene["domain"],
            1.0,
            {1: 1.0, 2: 0.01},
            scene["protocol"],
            RULES,
            scene["pore"],
            scene["swi"],
            [scene["target"]],
            SETTINGS,
            gas_threshold=0.66,
            model_frames=[scene["frame"]],
            zones=[scene["zone"]],
            depth_m=np.full(scene["domain"].shape, 0.005),
            positions=scene["positions"],
            record=lambda index, value: records.append(value),
        )
    )[0]
    return result, records[0]


@pytest.mark.parametrize("hidden", [True, False])
def test_active_link_seeds_small_disconnected_plume_and_closes_volume(hidden):
    """Old large gas cannot suppress an approved new plume's meeting fallback."""
    scene = _fixture(hidden=hidden)
    maps, record = _allocate(scene)
    labels, rules, saturation, co2, before = maps
    assert saturation[scene["meeting"]] > 0
    assert saturation[scene["frame"].labels == PhaseLabel.DISSOLVED].sum() > 0
    assert np.isclose(record.placed_m3, scene["target"], rtol=2e-7, atol=1e-15)
    assert record.unplaced_m3 < 1e-13
    assert record.disconnected_added_area_mm2 == 0
    assert np.all(saturation <= 1 - scene["swi"])
    np.testing.assert_array_equal(labels, scene["frame"].labels)
    np.testing.assert_array_equal(before, scene["frame"].before)
    inferred = rules == PhaseRule.GAS_BRIDGE
    if hidden:
        assert inferred.any() and record.bridge_volume_m3 > 0
        assert (labels[inferred] == PhaseLabel.WATER).all()
        assert not co2[inferred].any()
    else:
        assert not inferred.any() and record.bridge_volume_m3 == 0


@pytest.mark.parametrize(
    "refusal",
    [
        "inactive",
        "unknown",
        "seal",
        "cap_zero",
        "zero_score",
        "outside_contour",
        "gravity",
    ],
)
def test_invalid_or_inactive_meeting_grants_no_fallback(refusal, monkeypatch):
    scene = _fixture(refusal=refusal)
    if refusal == "gravity":
        below = np.zeros(scene["domain"].shape, bool)
        below[scene["meeting"]] = True
        monkeypatch.setattr(
            allocation, "_gravity_limit", lambda *args: (below, np.zeros_like(below))
        )
    maps, record = _allocate(scene)
    assert maps[2][scene["meeting"]] == 0
    assert not maps[2][scene["frame"].labels == PhaseLabel.DISSOLVED].any()
    assert record.unplaced_m3 > 0
    assert np.isclose(record.placed_m3, scene["model_volume"], rtol=2e-7)
    assert record.bridge_volume_m3 == 0


def test_existing_seeded_components_do_not_gain_fallback_seeds():
    gas = np.zeros((15, 20), bool)
    gas[2:5, 2:5] = True
    candidate = np.zeros_like(gas)
    candidate[2:8, 2:8] = True
    seeds = gas.copy()
    before = seeds.copy()
    active, effective = allocation._connected_candidates(
        gas, candidate, seeds, [(7, 7)]
    )
    np.testing.assert_array_equal(effective, before)
    np.testing.assert_array_equal(seeds, before)
    np.testing.assert_array_equal(active, candidate)


def test_fallback_does_not_seed_an_unlinked_third_component_or_mutate_old_seeds():
    gas = np.zeros((20, 30), bool)
    gas[2:5, 2:5] = True
    candidate = np.zeros_like(gas)
    candidate[2:7, 2:7] = candidate[10:15, 10:15] = candidate[10:15, 22:27] = True
    seeds = gas.copy()
    before = seeds.copy()
    active, effective = allocation._connected_candidates(
        gas, candidate, seeds, [(12, 12)]
    )
    assert active[12, 12] and not active[12, 24]
    assert effective[12, 12] and effective[gas].all()
    np.testing.assert_array_equal(seeds, before)


def test_existing_seeds_and_no_active_meetings_reproduce_original_addition():
    rng = np.random.default_rng(519)
    gas = rng.random((30, 40)) > 0.8
    candidate = rng.random(gas.shape) > 0.4
    seeds = gas & (rng.random(gas.shape) > 0.5)
    expected = candidate & allocation._touching(gas | candidate, seeds)
    made, effective = allocation._connected_candidates(gas, candidate, seeds, [])
    np.testing.assert_array_equal(made, expected)
    np.testing.assert_array_equal(effective, seeds)
