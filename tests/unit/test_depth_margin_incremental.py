"""Incremental half-disc unions must reproduce complete-prefix geometry exactly."""

import numpy as np
import pandas as pd
import pytest

from darsia.multiphase import gas_zone
from darsia.multiphase.phase_rules import PhaseLabel
from tests.unit.test_gas_limit import RULES, SETTINGS, START, _model, _protocol, _scene


def _direct(bare, depth, path, pitch):
    """Enumerate the same quantised source discs without any convolution/cache."""
    valid = np.isfinite(depth) & (depth > 0)
    source = bare & path & valid
    quantum = float(min(pitch))
    farthest = np.hypot(*((np.asarray(bare.shape) - 1) * pitch))
    last = int(np.ceil(farthest / quantum))
    values = 1000 * depth[source]
    levels = np.floor(np.minimum(values / quantum, last) + 0.5).astype(np.int64)
    rows, cols = np.indices(bare.shape)
    union = np.zeros_like(bare)
    for (row, col), level in zip(np.argwhere(source), levels):
        radius = level * quantum
        union |= (rows <= row) & (
            ((rows - row) * pitch[0]) ** 2 + ((cols - col) * pitch[1]) ** 2 <= radius**2
        )
    return bare | (union & path & valid)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pitch", [(1.0, 1.0), (2.0, 0.5), (0.3, 1.7)])
@pytest.mark.parametrize("seed", [1, 7, 32])
def test_incremental_variable_depth_matches_full_and_direct(dtype, pitch, seed):
    """Changing depths across space, invalid support and anisotropy preserve bits."""
    rng = np.random.default_rng(seed)
    shape = (31, 47)
    depths = [0, -1, np.nan, np.inf, 0.0002, 0.0015, 0.0032, 0.0071]
    depth = rng.choice(depths, shape).astype(dtype)
    depth[5, 7] = 10  # Geometric grid-diagonal cap, not a physical size change.
    path = rng.random(shape) > 0.15
    path[:, 20:22] = False
    depth.flags.writeable = path.flags.writeable = False
    bare = np.zeros(shape, bool)
    previous = None
    for index in range(12):
        grown = bare | (rng.random(shape) < 0.008)
        if index in (2, 5, 8):
            grown = bare.copy()  # No new source, but diagnostics remain complete.
        grown[5, 7] |= index > 6
        grown.flags.writeable = False
        full, expected_record = gas_zone._depth_margin(grown, depth, path, pitch)
        made, record = gas_zone._depth_margin(
            grown, depth, path, pitch, previous=previous
        )
        np.testing.assert_array_equal(made, full)
        np.testing.assert_array_equal(made, _direct(grown, depth, path, pitch))
        assert record == expected_record
        if previous is not None:
            assert not (previous[1] & ~made).any()
        previous = grown, made
        bare = grown


def test_new_bare_inside_old_margin_is_still_a_new_source():
    """Subtracting prior expansion instead of bare pixels would lose this growth."""
    bare = np.zeros((45, 55), bool)
    bare[25, 25] = True
    depth = np.full(bare.shape, 0.002)
    depth[23, 25] = 0.008
    path = np.ones_like(bare)
    expanded, _ = gas_zone._depth_margin(bare, depth, path, (1, 1))
    assert expanded[23, 25] and not expanded[15, 25]
    grown = bare.copy()
    grown[23, 25] = True
    made, record = gas_zone._depth_margin(
        grown, depth, path, (1, 1), previous=(bare, expanded)
    )
    full, full_record = gas_zone._depth_margin(grown, depth, path, (1, 1))
    assert made[15, 25]
    np.testing.assert_array_equal(made, full)
    assert record == full_record


def test_unchanged_prefix_performs_no_new_dilation_and_never_recurses(monkeypatch):
    bare = np.zeros((40, 60), bool)
    bare[20, 30] = True
    depth = np.full(bare.shape, 0.004)
    path = np.ones_like(bare)
    expanded, record = gas_zone._depth_margin(bare, depth, path, (1, 1))

    def forbidden(*args):
        raise AssertionError("Repeated bare mask must not be dilated again")

    monkeypatch.setattr(gas_zone, "_half_disc_dilation", forbidden)
    for _ in range(6):
        result, repeated = gas_zone._depth_margin(
            bare, depth, path, (1, 1), previous=(bare, expanded)
        )
        np.testing.assert_array_equal(result, expanded)
        assert repeated == record
        assert result[16, 30] and not result[15, 30]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_invalid_depth_and_excluded_sources_do_not_reuse_support(dtype):
    """Fixed invalid inputs retain bare pixels but grant no margin or recursion."""
    bare = np.zeros((11, 13), bool)
    bare[5, 5:10] = True
    depth = np.full(bare.shape, 0.003, dtype)
    depth[5, 5:9] = [0, -1, np.nan, np.inf]
    path = np.ones_like(bare)
    path[5, 9] = False
    full, expected = gas_zone._depth_margin(bare, depth, path, (0.5, 1.5))
    np.testing.assert_array_equal(full, bare)
    assert expected["source_depth_mm"] is None and expected["added_pixels"] == 0
    full.flags.writeable = bare.flags.writeable = False
    repeated, record = gas_zone._depth_margin(
        bare, depth, path, (0.5, 1.5), previous=(bare, full)
    )
    np.testing.assert_array_equal(repeated, bare)
    assert record == expected
    with pytest.raises(ValueError, match="share the gas contour's grid"):
        gas_zone._depth_margin(
            bare, depth[:-1], path, (0.5, 1.5), previous=(bare, full)
        )


@pytest.mark.parametrize("pitch", [(1.0, 1.0), (1.5, 0.75)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complete_zone_sequence_keeps_contours_links_and_diagnostics(
    monkeypatch, pitch, dtype
):
    """Raw seal gas, UNKNOWN, chart and disappearing/late gas stay identical."""
    image, positions, domain = _scene((100, 130), pitch)
    image.img[:, 78:81] = 2
    chart = np.zeros(domain.shape, bool)
    chart[30:36, 10:20] = True
    depth = np.full(domain.shape, 0.003, dtype)
    depth[:45] = 0.0043
    depth[20:25, 55:60] = np.nan
    first = np.full(domain.shape, PhaseLabel.WATER, np.uint8)
    first[28:49, 28:55] = PhaseLabel.GAS
    first[47:82, 36:47] = PhaseLabel.DISSOLVED
    first[35:38, 78:81] = PhaseLabel.GAS  # Raw seal evidence, never a margin seed.
    first[chart] = PhaseLabel.GAS
    first[60:65, 5:25] = PhaseLabel.UNKNOWN
    absent = np.full_like(first, PhaseLabel.WATER)
    late = absent.copy()
    late[70:85, 98:113] = PhaseLabel.GAS
    frames = [
        _model(labels, probability=np.full(domain.shape, 0.3, dtype))
        for labels in (first, first, absent, late)
    ]
    dates = [START + pd.Timedelta(hours=h) for h in (0, 1, 3, 4)]
    args = (
        dates,
        image,
        domain,
        {1: 1, 2: 0.01},
        _protocol(positions, (80, 40)),
        RULES,
        SETTINGS,
    )
    incremental = list(
        gas_zone.gas_model_zones(
            frames, *args,  chart=chart, depth_m=depth
        )
    )
    assert any(zone.links for zone in incremental[:2])
    assert all(not zone.links for zone in incremental[2:])
    method = gas_zone._depth_margin

    def complete(contour, depth_m, path, pitch_mm, **unused):
        return method(contour, depth_m, path, pitch_mm)

    monkeypatch.setattr(gas_zone, "_depth_margin", complete)
    reference = list(
        gas_zone.gas_model_zones(
            frames, *args,  chart=chart, depth_m=depth
        )
    )
    for made, full in zip(incremental, reference, strict=True):
        np.testing.assert_array_equal(made.contour, full.contour)
        assert made.contour.tobytes() == full.contour.tobytes()
        assert not made.contour.flags.writeable
        assert made.diagnostics == full.diagnostics
        assert made.links.keys() == full.links.keys()
        for key in made.links:
            one, two = made.links[key], full.links[key]
            np.testing.assert_array_equal(one.path, two.path)
            np.testing.assert_array_equal(one.blend, two.blend)
            np.testing.assert_array_equal(one.path_coordinates, two.path_coordinates)
            for name in ("path", "blend", "path_coordinates"):
                left, right = getattr(one, name), getattr(two, name)
                assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
                assert not left.flags.writeable
            assert one.diagnostics == two.diagnostics
