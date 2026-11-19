"""Physical gas lengths follow measured rig depth, independent of a project."""

from dataclasses import fields
from typing import get_type_hints

import numpy as np
import pytest

from darsia.multiphase.gas_parameters import GasAllocationSettings, gas_parameters
from tests.unit.test_gas_allocation import BASE_SETTINGS


@pytest.mark.parametrize("depth_mm", [2.5, 7.0, 18.0])
def test_every_geometric_length_scales_with_measured_depth(depth_mm):
    depth = np.full((8, 11), depth_mm / 1000)
    depth[0, :4] = [0, -1, np.nan, np.inf]
    resolved = gas_parameters(BASE_SETTINGS, depth)
    expected = {
        "depth_mm": 1,
        "closing_radius_mm": 2,
        "pre_rounding_margin_mm": 0,
        "contour_sigma_mm": 1,
        "fade_length_mm": 1,
        "probability_sigma_mm": 1,
        "link_width_mm": 1,
        "trace_sigma_mm": 1 / 3,
        "path_sigma_mm": 1 / 2,
        "outlier_dilation_mm": 1 / 3,
        "outlier_closing_radius_mm": 1,
        "outlier_sigma_mm": 1 / 3,
    }
    assert resolved.geometry_record() == pytest.approx(
        {key: factor * depth_mm for key, factor in expected.items()}
    )
    for field in fields(GasAllocationSettings):
        assert getattr(resolved, field.name) == getattr(BASE_SETTINGS, field.name)


def test_depth_uses_grid_median_and_cannot_be_a_series_length():
    values = np.array([[1, 2, 3], [8, np.nan, 0]]) / 1000
    assert gas_parameters(BASE_SETTINGS, values).depth_mm == pytest.approx(2.5)
    types = get_type_hints(GasAllocationSettings)
    assert {f.name for f in fields(GasAllocationSettings) if types[f.name] is not bool} == {
        "seed_mm2",
        "persistent_seed_mm2",
        "seed_window_h",
        "fallback_seed_mm2",
        "lattice_mm",
        "cost_up_per_mm",
        "cost_side_per_mm",
        "cost_down_per_mm",
    }


@pytest.mark.parametrize("depth", [None, [[0, -1], [np.nan, np.inf]], [0.01]])
def test_missing_or_unusable_depth_refuses_instead_of_guessing(depth):
    with pytest.raises(ValueError, match="depth"):
        gas_parameters(BASE_SETTINGS, depth)


def test_public_contours_scale_with_the_physical_rig_and_record_actual_depth():
    """Scaling a rig and its depth together preserves its pixel-space contour."""
    from darsia.multiphase.gas_zone import gas_model_zones
    from darsia.multiphase.phase_rules import PhaseLabel
    from tests.unit.test_gas_allocation import RULES, START
    from tests.unit.test_gas_limit import _model, _protocol, _scene

    contours = []
    for scale in (1, 2, 3):
        image, positions, domain = _scene((40, 50), (scale, scale))
        labels = np.full(domain.shape, PhaseLabel.WATER, np.uint8)
        labels[15:22, 18:30] = PhaseLabel.GAS
        zone = next(
            gas_model_zones(
                [_model(labels)],
                [START],
                image,
                domain,
                {1: 1.0},
                _protocol(positions),
                RULES,
                BASE_SETTINGS,
                
                depth_m=np.full(domain.shape, 0.003 * scale),
            )
        )
        assert zone.diagnostics["geometry"]["depth_mm"] == pytest.approx(3 * scale)
        assert zone.diagnostics["depth_margin"]["depth_quantum_mm"] == scale
        contours.append(zone.contour)
    for contour in contours[1:]:
        np.testing.assert_array_equal(contour, contours[0])
