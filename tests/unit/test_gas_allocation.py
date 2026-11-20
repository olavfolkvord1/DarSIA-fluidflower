"""Unit tests for the gas allocation on synthetic runs."""

import datetime
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.multiphase.gas_parameters import parameters_at_depth
from darsia.multiphase import gas_allocation as allocation_module
from darsia.utils import storage

W, D, G, U = darsia.PhaseLabel
RULE = darsia.PhaseRule

TABLES = tomllib.loads(
    """
    # The values the phase rules and the gas allocation ran with in production.
    [phase_rules]
    co2_threshold = 0.8
    smallest_gas_patch_mm2 = 25.0
    neighbour_window_h = 1.0
    neighbours_per_side = 2
    small_bubble_max_mm2 = 25.0
    small_bubble_min_height_mm = 30.0
    chart_margin_mm = 2.0
    chart_extend_below_mm = 35.0
    free_water_facies = [5]
    min_component_mm2 = 25.0
    max_hole_mm2 = 25.0
    min_gas_component_mm2 = 2.5
    seal_permeability_fraction = 0.05
    port_radius_mm = 15.0
    port_zone_half_width_mm = 38.0
    port_zone_depth_mm = 10.0
    finger_width_mm = 10.0
    pitch_stride = 64

    [gas_allocation]
    seed_mm2 = 100.0
    persistent_seed_mm2 = 25.0
    seed_window_h = 1.01
    fallback_seed_mm2 = 25.0
    lattice_mm = 1.0
    cost_up_per_mm = 0.5
    cost_side_per_mm = 1.0
    cost_down_per_mm = 3.0
    """
)
RULES = darsia.PhaseRuleSettings(**TABLES["phase_rules"])

BASE_SETTINGS = darsia.GasAllocationSettings(**TABLES["gas_allocation"])
SETTINGS = parameters_at_depth(BASE_SETTINGS, 5.0)
GAS_THRESHOLD = 0.66
"""Gas threshold of our production phase model, from its folder."""

START = datetime.datetime(2000, 1, 1)
SHAPE = (60, 80)
"""Rows and columns of a 1 mm grid; row r lies at height 59.5 - r mm."""
PORE = 1e-6 * 0.02 * 0.4
"""Pore volume of a pixel [m3]: 1 mm2, 20 mm deep, porosity 0.4."""
SWI = np.float32(0.2)


def _frame(co2=(), gas=(), level=0.9):
    """Probabilities: low everywhere, CO2 and gas at ``level`` in the listed boxes."""
    probabilities = np.full((2, *SHAPE), 0.1, np.float32)
    for index, boxes in enumerate((co2, gas)):
        for box in boxes:
            probabilities[index][box] = 0.9 if index == 0 else level
    return tuple(probabilities)


def _run(
    tmp_path,
    frames,
    hours,
    volumes,
    *,
    end_h=9.0,
    facies=None,
    positions=None,
    record=None,
    settings=BASE_SETTINGS,
    saturation_before=None,
    start=0,
    periods=None,
):
    """Phase and rule maps and saturations of a run, one port at height 4.5 mm."""
    csv = tmp_path / "protocol.csv"
    pd.DataFrame(
        [
            {
                "id": k + 1,
                "location_x": period[2] if len(period) > 2 else 0.0405,
                "location_y": period[3] if len(period) > 2 else 0.0045,
                "start": START + datetime.timedelta(hours=period[0]),
                "end": START + datetime.timedelta(hours=period[1]),
                "rate_g/s": 0.01,
            }
            for k, period in enumerate(periods or [(-1.0, end_h)])
        ]
    ).to_csv(csv, index=False)
    return list(
        darsia.free_gas(
            frames,
            [START + datetime.timedelta(hours=h) for h in hours],
            darsia.Image(
                np.ones(SHAPE, int) if facies is None else facies,
                dimensions=[0.06, 0.08],
            ),
            np.ones(SHAPE, bool),
            1.0,
            {1: 1.0, 2: 0.01},
            darsia.InjectionProtocol(csv),
            RULES,
            np.full(SHAPE, PORE),
            np.full(SHAPE, SWI),
            volumes,
            settings,
            gas_threshold=GAS_THRESHOLD,
            depth_m=np.full(SHAPE, 0.02),
            positions=positions,
            record=record,
            saturation_before=saturation_before,
            start=start,
        )
    )


def test_the_peel_reads_the_measured_heights():
    """Under a seal, a pixel's depth below its run's top is taken between the
    measured heights of its own column; in a peel's cell, the pixel lowest by its
    measured height goes first."""
    rows, cols = 12, 4
    y = np.broadcast_to(0.012 - np.arange(rows)[:, None] * 0.001, (rows, cols)).copy()
    y[:, 2] = 0.012 - np.arange(rows) * 0.0015  # one column's pixels lie further apart
    model = np.zeros((rows, cols), bool)
    model[4:6, :] = True
    tilted = y + np.arange(cols)[None, :] * -0.0012  # the right lies lower
    room = np.ones((rows, cols))
    order_rows, order_cols, _ = allocation_module._peel_order(
        model, tilted, room, (1.0, 1.0), replace(SETTINGS, lattice_mm=50.0)
    )
    heights = tilted[order_rows, order_cols]
    assert (order_rows[0], order_cols[0]) == np.unravel_index(
        np.where(model, tilted, np.inf).argmin(), (rows, cols)
    )
    assert (np.diff(heights) >= 0).all()


def test_the_allocation_reads_the_measured_positions(tmp_path, monkeypatch):
    """Identical centres give identical maps; calibrated centres locate the port.

    Moving a port inside a wide depth margin need not change the final mass.
    Check its actual physical endpoint instead of requiring such a change.
    """
    from darsia.multiphase import gas_zone

    endpoints = []
    original = gas_zone._injector_link

    def observed_endpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        endpoints.append(tuple(result[2]["port_pixel"]))
        return result

    monkeypatch.setattr(gas_zone, "_injector_link", observed_endpoint)
    frame = _frame([np.s_[10:60, 20:60]], [np.s_[30:40, 30:50]])
    volumes = [3 * 10 * 20 * PORE * (1 - float(SWI))]
    centres = darsia.grid_positions(
        darsia.Image(np.zeros(SHAPE), dimensions=[0.06, 0.08])
    )
    plain = _run(tmp_path, [frame], [0.0], volumes)
    same = _run(tmp_path, [frame], [0.0], volumes, positions=centres)
    for a, b in zip(plain[0], same[0]):
        np.testing.assert_array_equal(a, b)
    sheared = darsia.GridPositions(
        (centres.x + 0.004 * np.linspace(0, 1, SHAPE[0])[:, None]).astype(np.float32),
        centres.y.astype(np.float32),
        centres.pitch_mm,
    )
    _run(tmp_path, [frame], [0.0], volumes, positions=sheared)
    assert endpoints[0] == endpoints[1] != endpoints[2]
    row, col = endpoints[2]
    distance_mm = 1000 * np.hypot(
        sheared.x[row, col] - 0.0405, sheared.y[row, col] - 0.0045
    )
    assert distance_mm <= np.sqrt(0.5)


def test_added_gas_joins_the_model_gas_by_its_score(tmp_path):
    """Model gas stays; a request larger than the footprint's room is capped."""
    facies = np.ones(SHAPE, int)
    facies[40:42] = 2  # a seal across the CO2
    # The fixed smoothed contour joins the bodies. The seed-connected gas cannot
    # cross the seal to isolated CO2 below it.
    bodies = [np.s_[22:34, 34:39], np.s_[22:34, 41:46]]
    probabilities = _frame(co2=[np.s_[20:52, 10:70]], gas=bodies)
    probabilities[1][20:40, 10:70] += np.linspace(0, 0.3, 60, dtype=np.float32)
    records = []
    labels, _, saturation, _, _ = _run(
        tmp_path,
        [probabilities],
        [0.0],
        [1.0],
        facies=facies,
        record=lambda index, value: records.append(value),
    )[0]
    placed = np.sum(PORE * saturation.astype(float))
    assert placed < 1.0
    assert records[0].unplaced_m3 > 0
    assert np.isclose(placed + records[0].unplaced_m3, 1.0)
    for body in bodies:
        assert (saturation[body] == 1 - SWI).all() and (labels[body] == G).all()
    added = (saturation > 0) & (labels == D)
    assert added[25:31, 39:41].any() and not added[40:].any()
    assert not added[:20].any()
    assert records[0].disconnected_added_area_mm2 == 0


def test_removed_gas_peels_the_lower_edges_first(tmp_path):
    """Whole pixels go from the bottom and sides, one partial pixel, freed as dissolved."""
    probabilities = _frame(co2=[np.s_[10:50, 20:60]], gas=[np.s_[20:40, 30:50]])
    model = 400 * PORE * (1 - SWI)
    kept, gone = _run(tmp_path, [probabilities] * 2, [0.0, 3.0], [0.4 * model, 0.0])
    kept, kept_rules, kept_saturation, kept_co2, kept_before = kept
    gone, gone_rules, gone_saturation, gone_co2, gone_before = gone
    assert np.isclose(np.sum(PORE * kept_saturation.astype(float)), 0.4 * model)
    partial = (kept_saturation > 0) & (kept_saturation < 1 - SWI)
    assert partial.sum() <= 1
    assert ((kept == G) == (kept_saturation > 0)).all()
    assert not kept_saturation[33:40].any() and (kept_saturation[20, 35:45] > 0).all()
    assert not (gone == G).any() and not gone_saturation.any()
    # The rule map records the peel, and the model's gas is the gas before it.
    gas = np.zeros(SHAPE, bool)
    gas[20:40, 30:50] = True
    assert ((kept_rules == RULE.PEELED) == (gas & (kept == D))).all()
    assert (darsia.model_gas(kept, kept_rules) == gas).all()
    assert (darsia.model_gas(gone, gone_rules) == gas).all()
    # The CO2 before the rules, which the dilution law reads, stays the model's,
    # and so do the labels before the rules, on which the colour is read.
    co2 = np.zeros(SHAPE, bool)
    co2[10:50, 20:60] = True
    assert (kept_co2 == co2).all() and (gone_co2 == co2).all()
    model = np.where(gas, G, np.where(co2, D, W))
    np.testing.assert_array_equal(kept_before, model)
    np.testing.assert_array_equal(gone_before, model)


@pytest.mark.parametrize("end_h", [1.0, 3.0, 4.0])
def test_fixed_contour_keeps_later_model_gas_after_injection(tmp_path, end_h):
    """All selected post-rule gas contributes, irrespective of injection end."""
    left, right = np.s_[20:40, 10:30], np.s_[20:40, 52:72]
    before = _frame(co2=[left], gas=[left])
    after = _frame(co2=[left, right], gas=[left, right], level=0.99)
    (_, _, saturation, _, _), (labels, rules, _, _, _) = _run(
        tmp_path, [before, after], [0.5, 3.0], [None, None], end_h=end_h
    )
    assert (saturation[left] == 1 - SWI).all()
    assert (labels[left] == G).all() and (labels[right] == G).all()
    assert not (rules == darsia.PhaseRule.GAS_LIMIT).any()


@pytest.mark.parametrize("start", [1, 3])
def test_a_resumed_run_equals_the_whole_run(tmp_path, start):
    """The full prepass reconstructs the fixed contour before resuming.

    Every later map and saturation equals the whole run's, including peeled gas.
    """
    left, right = np.s_[20:40, 10:30], np.s_[20:40, 52:72]
    frames = [
        _frame(co2=[left], gas=[left]),
        _frame(co2=[left], gas=[np.s_[22:38, 12:28]]),
        _frame(co2=[left, right], gas=[left, right], level=0.7),
        _frame(co2=[left, right], gas=[left, np.s_[24:36, 56:68]], level=0.7),
        _frame(co2=[left, right], gas=[left, right], level=0.75),
    ]
    hours, volumes = [0.0, 0.5, 1.5, 2.0, 3.0], [None, 3e-7, 2e-7, None, 4e-7]
    whole = _run(tmp_path, frames, hours, volumes, end_h=1.0)
    csv = tmp_path / "protocol.csv"
    args = (
        frames,
        [START + datetime.timedelta(hours=h) for h in hours],
        darsia.Image(np.ones(SHAPE, int), dimensions=[0.06, 0.08]),
        np.ones(SHAPE, bool),
        1.0,
        {1: 1.0, 2: 0.01},
        darsia.InjectionProtocol(csv),
        RULES,
        np.full(SHAPE, PORE),
        np.full(SHAPE, SWI),
        volumes,
        BASE_SETTINGS,
    )
    resumed = list(
        darsia.free_gas(
            *args,
            gas_threshold=GAS_THRESHOLD,
            start=start,
            depth_m=np.full(SHAPE, 0.02),
        )
    )
    assert len(resumed) == len(frames) - start
    for made, again in zip(whole[start:], resumed):
        for array, other in zip(made, again):
            np.testing.assert_array_equal(array, other)
    assert (whole[3][0][24:36, 56:68] == G).all()


@pytest.mark.parametrize("compress", [False, True])
def test_a_held_out_variant_stores_only_the_maps_that_differ(
    tmp_path, monkeypatch, compress
):
    """A frame's maps without a held-out run's estimate store only those that differ
    from its maps, in dtype or a single bit, and read back as a full write of them,
    plain and compressed; an equal variant stores none, a variant stored whole
    reads as before, and a variant needs the frame's maps stored first."""
    monkeypatch.setattr(storage, "_current", [darsia.StorageSettings(compress)])
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 4, SHAPE).astype(np.uint8)
    maps = dict(
        labels=labels,
        rules=np.zeros(SHAPE, np.uint8),
        saturation=rng.random(SHAPE).astype(np.float32),
        co2=labels > 0,
        before=labels.copy(),
    )
    with pytest.raises(FileNotFoundError):
        darsia.write_phase_maps(tmp_path, *maps.values(), held_out="r2")
    darsia.write_phase_maps(tmp_path, *maps.values())
    variant = dict(maps, rules=maps["rules"].astype(np.int16))  # same values
    variant["saturation"] = maps["saturation"].copy()
    variant["saturation"].view(np.uint32)[3, 4] ^= 1  # one bit
    darsia.write_phase_maps(tmp_path, *variant.values(), held_out="r2")
    darsia.write_phase_maps(tmp_path, *maps.values(), held_out="r3")
    np.savez(tmp_path / "maps.without-r4.npz", **variant)  # stored whole
    for held_out, stored in (("r2", ["rules", "saturation"]), ("r3", [])):
        with np.load(tmp_path / f"maps.without-{held_out}.npz") as own:
            assert own.files == stored
    for held_out, made in (
        (None, maps),
        ("r2", variant),
        ("r3", maps),
        ("r4", variant),
    ):
        found = darsia.read_phase_maps(tmp_path, held_out)
        assert list(found) == list(made)
        for name, values in made.items():
            assert found[name].dtype == values.dtype
            assert found[name].tobytes() == values.tobytes()


def test_no_growth_after_injection_keeps_gone_gas_gone(tmp_path):
    """After every injection ends a frame holds at most the frame before's gas:
    once the gas is gone it stays gone, whatever the overflow or the model say.
    During the injection and without the setting nothing changes."""
    probabilities = _frame(co2=[np.s_[10:50, 20:60]], gas=[np.s_[20:40, 30:50]])
    model = 400 * PORE * (1 - SWI)
    frames, hours = [probabilities] * 4, [0.5, 2.0, 3.0, 4.0]
    volumes = [0.4 * model, 0.0, 0.3 * model, None]
    no_growth = replace(BASE_SETTINGS, growth_only_from_injecting_ports=True)
    found, plain = [], []
    _run(tmp_path, frames, hours, volumes, end_h=1.0, settings=no_growth,
         record=lambda i, r: found.append(r))
    _run(tmp_path, frames, hours, volumes, end_h=1.0,
         record=lambda i, r: plain.append(r))
    assert found[0].placed_m3 == pytest.approx(plain[0].placed_m3)
    assert found[0].growth_limit_m3 == 0 and found[1].placed_m3 == 0
    assert plain[2].placed_m3 > 0 and plain[3].placed_m3 > 0
    assert found[2].placed_m3 == 0 and found[3].placed_m3 == 0
    assert found[2].growth_limit_m3 == 0 and found[3].growth_limit_m3 == 0
    assert found[2].target_m3 == pytest.approx(0.3 * model)
    assert found[3].target_m3 is None and found[2].unplaced_m3 == 0


def test_no_growth_bounds_by_the_frame_before_a_resumed_start(tmp_path):
    """A resumed run takes the bound from the stored frame before its start."""
    probabilities = _frame(co2=[np.s_[10:50, 20:60]], gas=[np.s_[20:40, 30:50]])
    model = 400 * PORE * (1 - SWI)
    no_growth = replace(BASE_SETTINGS, growth_only_from_injecting_ports=True)
    found = []
    before = np.zeros(SHAPE, np.float32)
    before[20:24, 30:50] = 1 - SWI  # 80 of the 400 gas pixels: 0.2 of the model
    _run(tmp_path, [probabilities] * 2, [2.0, 3.0], [None, 0.5 * model],
         end_h=1.0, settings=no_growth, start=1, saturation_before=before,
         record=lambda i, r: found.append(r))
    assert found[-1].placed_m3 == pytest.approx(0.2 * model)
    assert found[-1].growth_limit_m3 == pytest.approx(0.2 * model)
    with pytest.raises(ValueError, match="saturation of the frame before"):
        _run(tmp_path, [probabilities] * 2, [2.0, 3.0], [None, None],
             end_h=1.0, settings=no_growth, start=1)


@pytest.mark.parametrize("requested", [None, 400 * PORE * (1 - SWI)])
def test_the_first_frame_waits_for_an_active_port(tmp_path, requested):
    gas = np.s_[20:40, 30:50]
    frame = _frame(co2=[np.s_[10:50, 20:60]], gas=[gas])
    settings = replace(BASE_SETTINGS, growth_only_from_injecting_ports=True)
    (labels, rules, saturation, _, before), = _run(
        tmp_path, [frame], [0.5], [requested], settings=settings, periods=[(1.0, 2.0)]
    )
    assert np.all(before[gas] == G)
    assert np.all(labels[gas] == D)
    assert np.all(rules[gas] == RULE.PEELED)
    assert not saturation.any()


def test_no_growth_holds_between_injections_but_not_during_them(tmp_path):
    """Between two injections and after the last the gas cannot grow; while a
    port injects it may."""
    probabilities = _frame(co2=[np.s_[10:50, 20:60]], gas=[np.s_[20:40, 30:50]])
    model = 400 * PORE * (1 - SWI)
    no_growth = replace(BASE_SETTINGS, growth_only_from_injecting_ports=True)
    found = []
    _run(tmp_path, [probabilities] * 4, [0.5, 1.5, 2.5, 3.5],
         [0.4 * model, 0.6 * model, 0.6 * model, 0.8 * model],
         settings=no_growth, periods=[(-1.0, 1.0), (2.0, 3.0)],
         record=lambda i, r: found.append(r))
    assert found[1].placed_m3 == pytest.approx(found[0].placed_m3)
    assert found[1].growth_limit_m3 == pytest.approx(found[0].placed_m3)
    assert found[2].growth_limit_m3 == 0
    assert found[2].placed_m3 == pytest.approx(0.6 * model)
    assert found[3].placed_m3 == pytest.approx(found[2].placed_m3)


def test_rising_reach_goes_up_and_sideways_never_down():
    """From a seed the reach climbs and spreads along each row, and a seal row
    with a gap lets it through only at the gap."""
    passable = np.ones((6, 5), bool)
    passable[2, :4] = False  # a seal with a gap at the right
    seeds = np.zeros((6, 5), bool)
    seeds[4, 1] = True
    reach = darsia.rising_reach(passable, seeds, top_first=True)
    assert not reach[5].any()  # below the seed
    assert reach[4].all() and reach[3].all()
    assert not reach[2, :4].any() and reach[2, 4]
    assert reach[1].all() and reach[0].all()


def test_gas_below_a_seal_cannot_grow_while_only_a_port_above_injects(tmp_path):
    """Only the upper port injects at the second frame: the lower box, which its
    gas cannot reach by rising, keeps at most its earlier gas."""
    facies = np.ones(SHAPE, int)
    facies[28:33] = 2  # a seal band at 27.5-31.5 mm
    lower_co2, upper_co2 = np.s_[35:55, 20:60], np.s_[5:24, 20:60]
    first = _frame(co2=[lower_co2], gas=[np.s_[35:40, 30:50]])
    second = _frame(
        co2=[lower_co2, upper_co2], gas=[np.s_[35:45, 30:50], np.s_[5:10, 30:50]]
    )
    model = 100 * PORE * (1 - SWI)
    rules = replace(BASE_SETTINGS, growth_only_from_injecting_ports=True)
    found = []
    maps = _run(
        tmp_path,
        [first, second],
        [0.5, 2.0],
        [model, 6 * model],
        facies=facies,
        settings=rules,
        periods=[(-1.0, 1.0, 0.0405, 0.0045), (1.5, 3.0, 0.0405, 0.0345)],
        record=lambda i, r: found.append(r),
    )
    lower = np.zeros(SHAPE, bool)
    lower[33:] = True
    held = np.sum(PORE * maps[0][2][lower].astype(float))
    later = np.sum(PORE * maps[1][2][lower].astype(float))
    assert held > 0
    assert later <= held + 1e-15
    assert found[1].growth_limit_m3 == pytest.approx(held)


def test_unreachable_model_gas_cannot_move_to_a_disconnected_patch(tmp_path):
    old, new = np.s_[18:28, 10:25], np.s_[18:28, 55:70]
    frames = [_frame(co2=[old], gas=[old]), _frame(co2=[new], gas=[new])]
    facies = np.ones(SHAPE, int)
    facies[:, 39:42] = 2
    settings = replace(BASE_SETTINGS, growth_only_from_injecting_ports=True)
    maps = _run(
        tmp_path,
        frames,
        [0.5, 1.5],
        [None, None],
        facies=facies,
        settings=settings,
        periods=[(-1.0, 1.0, 0.0205, 0.0045)],
    )
    assert np.all(maps[0][2][old] > 0)
    assert not maps[0][2][new].any()
    assert np.all(maps[1][4][new] == G)
    assert np.all(maps[1][1][new] == RULE.PEELED)
    assert not maps[1][2].any()


def test_an_unsupported_gas_pool_can_rise_within_its_held_volume(tmp_path):
    old, higher = np.s_[28:38, 30:50], np.s_[15:28, 30:50]
    frames = [_frame(co2=[old], gas=[old]), _frame(co2=[higher, old], gas=[higher, old])]
    settings = replace(BASE_SETTINGS, growth_only_from_injecting_ports=True)
    found = []
    maps = _run(
        tmp_path,
        frames,
        [0.5, 1.5],
        [None, None],
        settings=settings,
        periods=[(-1.0, 1.0)],
        record=lambda i, value: found.append(value),
    )
    assert maps[0][2][old].any()
    assert not maps[0][2][higher].any()
    assert not maps[1][2][old].any()
    assert maps[1][2][higher].any()
    assert found[1].placed_m3 == pytest.approx(found[0].placed_m3)
    assert found[1].growth_limit_m3 == pytest.approx(found[0].placed_m3)
    assert found[1].growth_peeled_m3 > 0


def test_supported_inferred_gas_can_stay_when_injection_stops(tmp_path):
    co2, gas = np.s_[10:50, 20:60], np.s_[20:40, 30:50]
    frame = _frame(co2=[co2], gas=[gas])
    narrowed = _frame(co2=[gas], gas=[gas])
    model = 400 * PORE * (1 - SWI)
    settings = replace(BASE_SETTINGS, growth_only_from_injecting_ports=True)
    found = []
    maps = _run(
        tmp_path,
        [frame, frame, narrowed],
        [0.5, 1.5, 2.0],
        [1.5 * model, 2 * model, 2 * model],
        settings=settings,
        periods=[(-1.0, 1.0)],
        record=lambda i, value: found.append(value),
    )
    assert found[0].placed_m3 == pytest.approx(1.5 * model)
    assert model < found[1].placed_m3 <= found[0].placed_m3
    assert 0 < found[1].growth_retained_m3 <= 0.5 * model
    assert found[1].placed_m3 == pytest.approx(model + found[1].growth_retained_m3)
    assert found[1].model_m3 == pytest.approx(model)
    assert np.all(maps[1][2][maps[0][0] == D] <= maps[0][2][maps[0][0] == D])
    assert np.all(maps[1][0][maps[0][0] == D] == D)
    assert found[2].placed_m3 == pytest.approx(model)
    assert found[2].growth_retained_m3 == 0


def test_an_unsupported_pool_cannot_spend_another_pools_previous_volume():
    shape = (6, 9)
    passable = np.ones(shape, bool)
    passable[:, 4] = False
    driven = np.zeros(shape, bool)
    previous = np.zeros(shape, np.float32)
    previous[4, 1] = 0.7
    previous[4, 6:9] = 0.7
    model = np.zeros(shape, bool)
    model[3, :4] = True
    cap = model.astype(np.float32)
    pore = np.ones(shape)
    y = np.broadcast_to(np.arange(shape[0], 0, -1)[:, None], shape)
    bound, peeled = allocation_module._hold_model_gas(
        cap, model, pore, passable, driven, previous, y, True
    )
    assert bound == pytest.approx(2.8)
    assert peeled == pytest.approx(3.3)
    assert cap.sum() == pytest.approx(0.7)
