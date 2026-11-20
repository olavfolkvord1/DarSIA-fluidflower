"""Hidden gas is visible in the gas contour without inventing optical CO2."""

from dataclasses import replace

import numpy as np
import pytest

from darsia.multiphase.phase_rules import PhaseLabel, PhaseRule
from darsia.presets.workflows.media.panels import PhotographSettings, outline_masks
from darsia.presets.workflows.media.tiles import (
    Frame,
    Panel,
    SandGeometry,
    TileSettings,
    draw_tile,
)
from tests.unit.test_media import STYLE


def test_only_explicit_known_water_bridge_enters_red_contour():
    labels = np.array(
        [
            [
                PhaseLabel.WATER,
                PhaseLabel.WATER,
                PhaseLabel.UNKNOWN,
                PhaseLabel.WATER,
                PhaseLabel.WATER,
                PhaseLabel.DISSOLVED,
            ]
        ],
        np.uint8,
    )
    rules = np.array(
        [
            [
                PhaseRule.GAS_BRIDGE,
                0,
                PhaseRule.GAS_BRIDGE,
                PhaseRule.GAS_BRIDGE,
                PhaseRule.GAS_BRIDGE,
                0,
            ]
        ],
        np.uint8,
    )
    saturation = np.array([[0.45, 1.0, 1.0, 0.449, np.nan, 0.45]])
    swi = np.full_like(saturation, 0.5)
    settings = PhotographSettings(
        tuple(map(tuple, np.eye(3))), 128, co2_outline=True, gas_contour=True
    )
    original = [a.copy() for a in (labels, rules, saturation, swi)]
    co2, gas = outline_masks(labels, saturation, swi, settings, rules=rules)
    np.testing.assert_array_equal(co2, [[0, 0, 0, 0, 0, 1]])
    np.testing.assert_array_equal(gas, [[1, 0, 0, 0, 0, 1]])
    old_co2, old_gas = outline_masks(labels, saturation, swi, settings)
    np.testing.assert_array_equal(old_co2, co2)
    np.testing.assert_array_equal(old_gas, [[0, 0, 0, 0, 0, 1]])
    for before, after in zip(original, (labels, rules, saturation, swi)):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="grid"):
        outline_masks(labels, saturation, swi, settings, rules=rules[:, :2])


def test_bridge_rule_reaches_the_actual_image_tile():
    shape = (24, 40)
    labels = np.full(shape, PhaseLabel.WATER, np.uint8)
    rules = np.zeros(shape, np.uint8)
    rules[8:16, 10:30] = PhaseRule.GAS_BRIDGE
    saturation = np.zeros(shape, np.float32)
    saturation[8:16, 10:30] = 1.0
    sand = SandGeometry(
        np.ones(shape, np.uint8),
        [1],
        np.ones(shape),
        np.ones(shape),
        np.ones(shape),
        np.zeros(shape),
    )
    frame = Frame(
        "case",
        1.0,
        1.0,
        shape,
        (0, 0, 40, 24),
        image=np.full((*shape, 3), 0.15, np.float32),
        phase=labels,
        saturation=saturation,
        sand=sand,
        rules=rules,
    )
    settings = PhotographSettings(
        tuple(map(tuple, np.eye(3))), 128, co2_outline=True, gas_contour=True
    )
    args = dict(style=STYLE, photograph_settings=settings)
    image, _ = draw_tile(
        Panel("image", "Photo", "none"), frame, TileSettings(400, 40, 10), **args
    )
    without, _ = draw_tile(
        Panel("image", "Photo", "none"),
        replace(frame, rules=None),
        TileSettings(400, 40, 10),
        **args,
    )
    red = (image[:, :, 0] > 180) & (image[:, :, 1] < 100) & (image[:, :, 2] < 100)
    assert red.any()
    assert not (
        (without[:, :, 0] > 180) & (without[:, :, 1] < 100) & (without[:, :, 2] < 100)
    ).any()
