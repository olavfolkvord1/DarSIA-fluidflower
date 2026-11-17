"""Allocated bridge gas stays known without becoming optical or dissolved CO2."""

from dataclasses import fields, replace

import numpy as np
import pytest

import darsia


def _frame(dtype):
    labels = np.array([[0, 0, 0, 1, 2], [1, 255, 0, 0, 0]], dtype=np.uint8)
    shape = labels.shape
    sand = darsia.Image(np.ones(shape, np.uint8), dimensions=[0.01, 0.025])
    domain = np.ones(shape, bool)
    domain[1, 3] = False
    excluded = np.zeros(shape, bool)
    excluded[1, 4] = True
    depth = np.full(shape, 0.01, dtype)
    depth[1, 2] = np.nan
    geometry = darsia.run_geometry(
        sand,
        np.full(shape, 25e-6, dtype),
        depth,
        np.full(shape, 0.4, dtype),
        np.full(shape, 0.2, dtype),
        domain,
        excluded,
    )
    relative = darsia.ScalarImage(
        np.full(shape, 0.5, dtype), dimensions=sand.dimensions
    )
    relative.img[0, :2] = np.nan  # hidden pixels have no dissolved reading
    co2 = darsia.PrecisionCO2MassAnalysis(relative, datum_m=0.01)
    calibration = darsia.MassCalibration(
        gain=1.0,
        zero_signal=0.0,
        saturation_signal=1.0,
        line_share=1.0,
        beta=0.5,
        beta_growth=0.0,
        anchor_h=1.0,
        anchor_pore_volume_m3=1e-4,
    )
    rules = np.zeros(shape, np.uint8)
    rules[0, :2] = darsia.PhaseRule.GAS_BRIDGE
    saturation = np.zeros(shape, dtype)
    saturation[0] = [0.2, 0.4, 0, 0.1, 0.8]
    measured = np.ones(shape, bool)
    measured[0, :2] = measured[1, 0] = False
    return relative, labels, rules, saturation, measured, geometry, co2, calibration


def _mass(frame):
    return darsia.mass_maps(*frame, factor=0.8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bridge_mass_keeps_volume_without_optical_or_aqueous_evidence(
    dtype, tmp_path
):
    frame = _frame(dtype)
    relative, labels, rules, saturation, measured, geometry, co2, calibration = frame
    before, original_rules, original_sg = (
        labels.copy(),
        rules.copy(),
        saturation.copy(),
    )
    maps = _mass(frame)
    bridge = rules == darsia.PhaseRule.GAS_BRIDGE
    ordinary = rules.copy()
    ordinary[bridge] = darsia.PhaseRule.UNCHANGED
    reference = _mass((relative, labels, ordinary, *frame[3:]))

    assert np.all(labels[bridge] == darsia.PhaseLabel.WATER)
    assert not darsia.model_gas(labels, rules)[bridge].any()
    np.testing.assert_array_equal(labels, before)
    np.testing.assert_array_equal(rules, original_rules)
    np.testing.assert_array_equal(saturation, original_sg)
    np.testing.assert_array_equal(maps.saturation_g.img[bridge], saturation[bridge])
    assert maps.saturation_g.img.dtype == dtype
    assert np.all(maps.concentration_aq.img[bridge] == 0)
    assert np.all(maps.mass_aq.img[bridge] == 0)
    np.testing.assert_array_equal(
        maps.mass_g.img[bridge],
        co2.density_gaseous_co2[bridge] * saturation[bridge],
    )
    for name in ("mass", "mass_g", "mass_aq", "concentration_aq", "saturation_g"):
        np.testing.assert_array_equal(
            getattr(maps, name).img[~bridge], getattr(reference, name).img[~bridge]
        )
    pores = geometry.pore_volume_m3
    assert np.sum(pores[bridge] * maps.saturation_g.img[bridge]) == np.sum(
        pores[bridge] * saturation[bridge]
    )
    assert all(
        mask[bridge].all() for mask in darsia.known_mass(maps, labels).values()
    )

    # Existing statistics count allocated saturation everywhere counted, but
    # model-gas water and dilution evidence still use the unchanged optical maps.
    observed = np.isin(before, (darsia.PhaseLabel.DISSOLVED, darsia.PhaseLabel.GAS))
    stats = darsia.gas_statistics(
        observed,
        darsia.model_gas(labels, rules),
        saturation,
        np.ones(labels.shape, int),
        geometry.counted,
        pores,
        co2.density_gaseous_co2,
        co2.solubility_co2,
    )
    assert geometry.mass_g(maps.mass_g.img) == pytest.approx(
        stats.free_gas_g, rel=5e-7
    )
    assert stats.co2_pore_volume_m3 == float(np.sum(pores[observed]))
    without_bridge = saturation.copy()
    without_bridge[bridge] = 0
    old_stats = darsia.gas_statistics(
        observed,
        darsia.model_gas(labels, ordinary),
        without_bridge,
        np.ones(labels.shape, int),
        geometry.counted,
        pores,
        co2.density_gaseous_co2,
        co2.solubility_co2,
    )
    assert stats.gas_water_g == old_stats.gas_water_g
    assert stats.sands == old_stats.sands
    assert stats.free_gas_g > old_stats.free_gas_g
    settings = darsia.UnknownMassSettings(column_reach_mm=20, intervals=32)
    unknown = darsia.unknown_statistics(
        relative.img, labels, rules, measured, geometry, co2, settings
    )
    old_unknown = darsia.unknown_statistics(
        relative.img, labels, ordinary, measured, geometry, co2, settings
    )
    for field in fields(unknown):
        np.testing.assert_equal(
            getattr(unknown, field.name), getattr(old_unknown, field.name)
        )

    stored = tmp_path / "mass"
    stored.mkdir()
    darsia.write_mass_maps(stored, maps)
    read = darsia.read_mass_maps(stored)
    np.testing.assert_array_equal(read.saturation_g.img, maps.saturation_g.img)
    np.testing.assert_array_equal(read.mass_g.img, maps.mass_g.img)
    copy = darsia.MassCopy(geometry, darsia.MassCopySettings(copy_mm=5.0))
    cells = copy(read, labels)
    assert np.sum(cells["mass_free_gas_g"]) == pytest.approx(
        stats.free_gas_g, rel=5e-7
    )
    np.testing.assert_array_equal(
        cells["phase_gas_area_fraction"],
        copy(reference, labels)["phase_gas_area_fraction"],
    )


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "dissolved",
        "gas",
        "no_swi",
        "infinite_swi",
        "nan_sg",
        "infinite_sg",
        "zero",
        "negative",
        "over_cap",
    ],
)
def test_counted_malformed_bridge_is_refused_instead_of_becoming_zero_mass(bad):
    frame = list(_frame(np.float32))
    _, labels, _, saturation, _, geometry, _, _ = frame
    if bad in {"unknown", "dissolved", "gas"}:
        labels[0, 0] = {
            "unknown": darsia.PhaseLabel.UNKNOWN,
            "dissolved": darsia.PhaseLabel.DISSOLVED,
            "gas": darsia.PhaseLabel.GAS,
        }[bad]
    elif bad in {"no_swi", "infinite_swi"}:
        swi = geometry.swi.copy()
        swi[0, 0] = np.nan if bad == "no_swi" else np.inf
        frame[5] = replace(geometry, swi=swi)
    else:
        saturation[0, 0] = {
            "nan_sg": np.nan,
            "infinite_sg": np.inf,
            "zero": 0,
            "negative": -0.1,
            "over_cap": 0.9,
        }[bad]
    with pytest.raises(ValueError, match="Counted GAS_BRIDGE"):
        _mass(frame)


def test_marker_outside_counted_and_unmarked_water_keep_existing_zero_treatment():
    frame = _frame(np.float32)
    relative, labels, rules, saturation, measured, geometry, co2, calibration = frame
    rules[:] = 0
    # No marker means the old WATER behavior, even with an unused nonzero Sg.
    saturation[0, 2] = 0.5
    reference = _mass(frame)
    for point in ((1, 2), (1, 3), (1, 4)):
        assert not geometry.counted[point]
        rules[point] = darsia.PhaseRule.GAS_BRIDGE
        labels[point] = darsia.PhaseLabel.UNKNOWN
        saturation[point] = np.nan
    maps = _mass(frame)
    assert maps.mass.img[0, 2] == maps.saturation_g.img[0, 2] == 0
    for point in ((1, 2), (1, 3), (1, 4)):
        assert maps.mass.img[point] == maps.saturation_g.img[point] == 0
    np.testing.assert_array_equal(maps.mass.img, reference.mass.img)
    # The existing counted UNKNOWN pixel remains unknown to known_mass.
    assert not any(mask[1, 1] for mask in darsia.known_mass(maps, labels).values())
