"""Display uses recorded colour metadata without touching scientific samples."""

from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.image.raw import _raw_display_matrix
from darsia.presets.workflows.media.panels import (
    PhotographSettings,
    photograph,
    source_photograph_settings,
)


def test_as_shot_display_preserves_scientific_decode_and_unknowns(tmp_path):
    pytest.importorskip("rawpy")
    from tests.unit.test_raw_decode import BLACK, LAYOUTS, write_dng

    path = tmp_path / "camera.dng"
    mosaic = np.full((32, 48), BLACK + 300, np.uint16)
    write_dng(path, mosaic, LAYOUTS["BGGR"][0])
    scientific = darsia.read_raw_sensor(path, "edge_aware")
    shown = darsia.read_raw_sensor(path, "edge_aware", display=True)
    np.testing.assert_array_equal(shown.values, scientific.values)
    np.testing.assert_array_equal(shown.invalid, scientific.invalid)
    assert shown.white_range == scientific.white_range
    np.testing.assert_allclose(shown.display_matrix, darsia.raw_display_matrix(path))
    values = shown.values.astype(np.float32) / shown.white_range
    values[shown.invalid] = np.nan
    saved = values.copy()
    settings = source_photograph_settings(PhotographSettings(), path)
    result = photograph(values, settings)
    np.testing.assert_array_equal(values, saved)
    assert result[5, 5, 2] > result[5, 5, 1]
    assert np.all(result[0, 0] == np.float32(128 / 255))
    # A darker correction stays darker; no per-side normalization erases it.
    darker = photograph(values * 0.5, settings)
    assert np.all(darker[5, 5] < result[5, 5])


def test_missing_white_balance_is_not_replaced_by_an_invented_one():
    raw = SimpleNamespace(
        color_desc=b"RGBG",
        camera_whitebalance=[0, 0, 0, 0],
        rgb_xyz_matrix=np.eye(4, 3),
    )
    with pytest.raises(ValueError, match="as-shot white balance"):
        _raw_display_matrix(raw)


def test_colour_metadata_order_matches_decoder_rgb():
    gains = [2, 1, 4, 1]
    xyz = np.array([[0.8, -0.2, -0.1], [-0.3, 1.2, 0.2], [-0.05, 0.15, 0.5], [0, 0, 0]])
    ordered = SimpleNamespace(
        color_desc=b"RGBG", camera_whitebalance=gains, rgb_xyz_matrix=xyz
    )
    order = [2, 1, 0, 3]
    reversed_ = SimpleNamespace(
        color_desc=b"BGRG",
        camera_whitebalance=np.array(gains)[order],
        rgb_xyz_matrix=xyz[order],
    )
    np.testing.assert_allclose(
        _raw_display_matrix(ordered), _raw_display_matrix(reversed_)
    )
