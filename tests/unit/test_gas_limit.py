"""Causal gas geometry, metric filtering and allocation invariants."""

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

import darsia
from darsia.multiphase import gas_allocation as allocation
from darsia.multiphase import gas_zone
from darsia.multiphase.gas_model_cache import GasModelFrame
from darsia.multiphase.phase_rules import PhaseLabel, PhaseRule
from tests.unit.test_gas_allocation import RULES, SETTINGS, START


def _zones(*args, **kwargs):
    """A positive subpixel fixture depth rounds to no margin unless specified."""
    kwargs.setdefault("depth_m", np.full(args[3].shape, 0.0001))
    return gas_zone.gas_model_zones(*args, **kwargs)


def _scene(shape=(70, 90), pitch=(1.0, 1.0)):
    image = darsia.Image(
        np.ones(shape, int), dimensions=np.asarray(shape) * pitch / 1000
    )
    return image, darsia.grid_positions(image), np.ones(shape, bool)


def _protocol(positions, pixel=None):
    columns = ["rate_kg_s", "location_x", "location_y", "start", "end"]
    rows = (
        []
        if pixel is None
        else [
            [
                1,
                float(positions.x[pixel]),
                float(positions.y[pixel]),
                START,
                START + pd.Timedelta(hours=2),
            ]
        ]
    )
    return SimpleNamespace(df=pd.DataFrame(rows, columns=columns))


def _model(labels, probability=None, before=None, rules=None):
    return GasModelFrame(
        np.full(labels.shape, 0.3, np.float32) if probability is None else probability,
        labels.astype(np.uint8) if before is None else before.astype(np.uint8),
        labels.astype(np.uint8),
        np.zeros(labels.shape, np.uint8) if rules is None else rules.astype(np.uint8),
    )


def test_contour_preserves_observed_seal_gas_and_excludes_checker_water():
    image, positions, domain = _scene()
    labels = np.full(domain.shape, PhaseLabel.WATER, np.uint8)
    labels[15:25, 15:25] = PhaseLabel.GAS
    before = labels.copy()
    before[40:45, 40:45] = PhaseLabel.GAS
    labels[40:45, 40:45] = PhaseLabel.DISSOLVED
    rules = np.zeros_like(labels)
    rules[40:45, 40:45] = PhaseRule.GRAVITY
    late = labels.copy()
    late[50:60, 65:75] = PhaseLabel.GAS
    labels[4:8, 4:8] = PhaseLabel.GAS
    image.img[4:8, 4:8] = 2
    labels[4:8, 80:85] = PhaseLabel.GAS
    image.img[4:8, 80:85] = 5
    chart = np.zeros_like(domain)
    chart[55:65, 5:10] = True
    labels[chart] = PhaseLabel.GAS
    settings = replace(SETTINGS, closing_length_mm=0, contour_smoothing_mm=(0, 0))
    zone = list(
        _zones(
            [_model(labels, before=before, rules=rules), _model(late)],
            [START, START + pd.Timedelta(hours=3)],
            image,
            domain,
            {1: 1, 2: 0.01, 5: 1000},
            _protocol(positions),
            RULES,
            settings,
            positions=positions,
            chart=chart,
        )
    )[-1]
    assert zone.contour[15:25, 15:25].all()
    assert zone.contour[50:60, 65:75].all()
    assert not zone.contour[40:45, 40:45].any()
    assert not zone.contour[chart | (image.img == 5)].any()
    assert zone.contour[4:8, 4:8].all()
    assert not zone.contour.flags.writeable


@pytest.mark.parametrize("pitch", [(1.0, 1.0), (2.0, 0.5)])
def test_border_safe_metric_closing_equals_padded_disc_reference(pitch):
    mask = np.zeros((60, 80), bool)
    mask[:15, 10:25] = True
    mask[:15, 30:45] = True
    radius = 5
    reach = np.ceil(radius / np.asarray(pitch)).astype(int)
    rows, cols = np.mgrid[-reach[0] : reach[0] + 1, -reach[1] : reach[1] + 1]
    disc = (rows * pitch[0]) ** 2 + (cols * pitch[1]) ** 2 <= radius**2
    padded = np.pad(mask, 30)
    reference = ndimage.binary_closing(padded, disc)[30:-30, 30:-30]
    made = gas_zone._rounded(mask, pitch, radius, 0, 0)
    np.testing.assert_array_equal(made, reference)
    assert made[mask].all() and not made[20:].any()


def test_rounding_preserves_border_gas_without_a_bottom_wall():
    mask = np.zeros((100, 150), bool)
    mask[90:, 35:60] = True
    path = np.ones_like(mask)
    made = gas_zone._envelope(mask, path, (1, 1), SETTINGS)
    assert made[mask].all()
    assert not made[:, :20].any() and not made[:, 80:].any()
    assert not gas_zone._envelope(np.zeros_like(mask), path, (1, 1), SETTINGS).any()


def test_link_reaches_injector_inside_observed_co2_and_preserves_endpoints():
    image, positions, path = _scene()
    contour = np.zeros_like(path)
    contour[10:24, 38:53] = True
    observed = np.zeros_like(path)
    observed[10:61, 32:59] = True
    pixel = (60, 45)
    port = (positions.x[pixel], positions.y[pixel])
    probability = np.where(observed, 0.4, 0).astype(np.float32)
    line, _, report = gas_zone._injector_link(
        port,
        contour,
        observed,
        probability,
        path,
        positions,
        SETTINGS,
        current_co2=observed,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    assert report["status"] == "linked"
    assert line[pixel] and (line & contour).any()
    assert not (line & ~observed).any()
    assert cv2.connectedComponents(line.astype(np.uint8), connectivity=8)[0] == 2
    inside_port = (positions.x[15, 45], positions.y[15, 45])
    no_line, _, note = gas_zone._injector_link(
        inside_port,
        contour,
        observed,
        probability,
        path,
        positions,
        SETTINGS,
        current_co2=observed,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    assert no_line is None and note["status"] == "already_inside"
    path[35, :] = False
    no_line, _, note = gas_zone._injector_link(
        port,
        contour,
        observed,
        probability,
        path,
        positions,
        SETTINGS,
        current_co2=observed,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    assert no_line is None and note["status"] == "no_observed_co2_path"


@pytest.mark.parametrize("offset", [(0, 1), (1, 1)])
def test_porosity_anchors_links_without_changing_observed_co2(offset):
    _, positions, path = _scene()
    contour = np.zeros_like(path)
    contour[10:20, 40:50] = True
    observed = np.zeros_like(path)
    observed[10:51, 40:50] = True
    pixel = (50 + offset[0], 49 + offset[1])
    port = (positions.x[pixel], positions.y[pixel])
    probability = np.where(observed, 0.4, 0).astype(np.float32)
    original = observed.copy()
    line, _, note = gas_zone._injector_link(
        port,
        contour,
        observed,
        probability,
        path,
        positions,
        SETTINGS,
        current_co2=observed,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    assert note["status"] == "linked" and line[pixel]
    outside = line & ~observed
    assert outside.sum() == 1 and outside[pixel]
    np.testing.assert_array_equal(observed, original)
    path[pixel] = False
    assert (
        gas_zone._injector_link(
            port,
            contour,
            observed,
            probability,
            path,
            positions,
            SETTINGS,
            current_co2=observed,
            depth_m=np.full(np.shape(contour), 0.1),
        )[0]
        is None
    )
    path[pixel] = True
    observed[48:53, 47:53] = False
    bridged, _, note = gas_zone._injector_link(
        port,
        contour,
        observed,
        probability,
        path,
        positions,
        SETTINGS,
        current_co2=observed,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    assert note["status"] == "linked" and bridged[pixel]


@pytest.mark.parametrize("shape", [(1, 13), (13, 1), (23, 31)])
@pytest.mark.parametrize("pitch", [(1.0, 1.0), (2.0, 0.5)])
@pytest.mark.parametrize("radius", [0.5, 4.0, 40.0])
def test_half_disc_fft_matches_direct_morphology_including_borders(
    shape, pitch, radius
):
    rng = np.random.default_rng(5748)
    source = rng.random(shape) < 0.04
    source[-1, shape[1] // 2] = True
    reach = np.ceil(radius / np.asarray(pitch)).astype(int)
    rr = np.arange(-reach[0], reach[0] + 1)[:, None]
    cc = np.arange(-reach[1], reach[1] + 1)[None, :]
    kernel = (rr >= 0) & ((rr * pitch[0]) ** 2 + (cc * pitch[1]) ** 2 <= radius**2)
    expected = cv2.dilate(
        source.astype(np.uint8),
        kernel.astype(np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    np.testing.assert_array_equal(
        gas_zone._half_disc_dilation(source, radius, pitch), expected
    )


def test_slanted_edge_margin_is_union_of_source_half_discs_without_comb():
    contour = np.zeros((45, 55), bool)
    for row in range(8, 35):
        contour[row, 10 : 20 + row // 3] = True
    depth = np.full(contour.shape, 0.0042)
    depth[contour & (np.indices(contour.shape)[0] < 20)] = 0.0063
    pitch = (1.0, 0.5)
    expected = contour.copy()
    rr, cc = np.indices(contour.shape)
    for row, col in zip(*np.nonzero(contour)):
        radius = np.floor(depth[row, col] * 1000 / 0.5 + 0.5) * 0.5
        expected |= (rr <= row) & (
            ((rr - row) * pitch[0]) ** 2 + ((cc - col) * pitch[1]) ** 2 <= radius**2
        )
    made, record = gas_zone._depth_margin(contour, depth, np.ones_like(contour), pitch)
    np.testing.assert_array_equal(made, expected)
    assert record["depth_quantum_mm"] == 0.5
    assert not made[35:].any()


def test_model_zone_link_evidence_is_limited_to_its_injection():
    image, positions, domain = _scene()
    labels = np.full(domain.shape, PhaseLabel.WATER, np.uint8)
    labels[10:61, 32:59] = PhaseLabel.DISSOLVED
    labels[10:24, 38:53] = PhaseLabel.GAS
    settings = replace(SETTINGS, closing_length_mm=0, contour_smoothing_mm=(0, 0))
    arguments = (image, domain, {1: 1}, _protocol(positions, (60, 45)), RULES, settings)
    active = list(_zones([_model(labels)], [START], *arguments))[-1]
    late = list(_zones([_model(labels)], [START + pd.Timedelta(hours=3)], *arguments))[
        -1
    ]
    assert active.contour[60, 45] and len(active.links) == 1
    assert not late.contour[60, 45] and not late.links


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pitch,window", [((1, 1), 5), ((2, 0.5), 34.64101615137754)])
def test_fft_score_matches_direct_disc_filter(dtype, pitch, window):
    rng = np.random.default_rng(5717)
    values = rng.uniform(0, 1, (53, 71)).astype(dtype)
    scored = rng.random(values.shape) > 0.3
    values[~scored] = np.nan
    sigma = window / np.sqrt(12)
    reach = np.ceil(3 * sigma / np.asarray(pitch)).astype(int)
    rr, cc = np.mgrid[-reach[0] : reach[0] + 1, -reach[1] : reach[1] + 1]
    square = (rr * pitch[0]) ** 2 + (cc * pitch[1]) ** 2
    kernel = np.where(square <= (3 * sigma) ** 2, np.exp(-square / (2 * sigma**2)), 0)
    kernel = (kernel / kernel.sum()).astype(dtype)
    numerator = cv2.filter2D(
        np.where(scored, values, 0), -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    denominator = cv2.filter2D(
        scored.astype(dtype), -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    expected = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    made = allocation._round_score(scored, values, pitch, window)
    np.testing.assert_allclose(
        made,
        expected,
        rtol=2e-5 if dtype == np.float32 else 1e-12,
        atol=2e-7 if dtype == np.float32 else 1e-14,
    )
    assert made.dtype == dtype
    assert not allocation._round_score(
        scored, np.zeros_like(values), pitch, window
    ).any()


def test_support_crop_matches_the_full_grid_envelope_exactly():
    mask = np.zeros((500, 700), bool)
    mask[250:300, 290:330] = True
    mask[270:320, 345:375] = True
    mask[330, 390] = True
    path = np.ones_like(mask)
    path[280:283, 320:355] = False
    settings = replace(SETTINGS, contour_smoothing_mm=(3, 10), closing_length_mm=25)
    np.testing.assert_array_equal(
        gas_zone._envelope(mask, path, (1, 1), settings),
        gas_zone._smooth_envelope(mask, path, (1, 1), settings),
    )


def test_measured_depth_margin_grows_only_sideways_and_up():
    contour = np.zeros((41, 41), bool)
    contour[20, 20] = True
    depth = np.full(contour.shape, 0.004)
    made, record = gas_zone._depth_margin(contour, depth, np.ones_like(contour), (1, 1))
    rows, cols = np.mgrid[:41, :41]
    expected = ((rows - 20) ** 2 + (cols - 20) ** 2 <= 16) & (rows <= 20)
    np.testing.assert_array_equal(made, expected)
    assert record["source_depth_mm"] == {"min": 4, "median": 4, "max": 4}


def test_margin_uses_sources_local_depth_and_requires_known_depth():
    contour = np.zeros((41, 51), bool)
    contour[20, 10] = contour[20, 35] = True
    depth = np.full(contour.shape, 0.004)
    depth[20, 10], depth[20, 35] = 0.002, 0.006
    depth[16, 35] = np.nan
    made, record = gas_zone._depth_margin(contour, depth, np.ones_like(contour), (1, 1))
    assert made[18, 10] and not made[17, 10]
    assert made[14, 35] and not made[13, 35]
    assert not made[16, 35] and not made[21:].any()
    assert record["source_depth_mm"] == {"min": 2, "median": 4, "max": 6}
    depth[contour] = np.nan
    unchanged, _ = gas_zone._depth_margin(contour, depth, np.ones_like(contour), (1, 1))
    np.testing.assert_array_equal(unchanged, contour)


def test_repeated_frames_do_not_reapply_depth_margin_to_its_own_output():
    image, positions, domain = _scene()
    labels = np.zeros(domain.shape, np.uint8)
    labels[25:40, 35:50] = PhaseLabel.GAS
    settings = replace(SETTINGS, closing_length_mm=0, contour_smoothing_mm=(0, 0))
    zones = list(
        _zones(
            [_model(labels)] * 4,
            [START + pd.Timedelta(hours=h) for h in range(4)],
            image,
            domain,
            {1: 1},
            _protocol(positions),
            RULES,
            settings,
            depth_m=np.full(domain.shape, 0.005),
        )
    )
    for zone in zones[1:]:
        np.testing.assert_array_equal(zone.contour, zones[0].contour)
    assert zones[0].contour[20, 40] and not zones[0].contour[19, 40]
    assert not zones[0].contour[40:].any()


def test_observed_seal_gas_is_preserved_but_no_gas_is_added_on_seal(tmp_path):
    from tests.unit.test_gas_allocation import SHAPE, _frame, _run

    facies = np.ones(SHAPE, int)
    facies[30:33] = 2
    frame = _frame([np.s_[20:45, 20:60]], [np.s_[25:36, 35:45]])
    ordinary = _run(tmp_path, [frame], [0], [None], facies=facies)[0]
    expanded = _run(tmp_path, [frame], [0], [1.0], facies=facies)[0]
    seal = facies == 2
    assert ordinary[2][seal].any()
    np.testing.assert_array_equal(expanded[2][seal], ordinary[2][seal])


def test_observed_seal_pixels_cannot_change_smoothing_links_or_depth_margin():
    image, positions, domain = _scene((160, 160))
    image.img[:, 85:88] = 2
    labels = np.zeros(domain.shape, np.uint8)
    labels[50:75, 40:60] = PhaseLabel.GAS
    with_seal = labels.copy()
    with_seal[55:57, 85:87] = PhaseLabel.GAS
    seal_pixels = with_seal != labels
    arguments = (image, domain, {1: 1, 2: 0.01}, _protocol(positions), RULES, SETTINGS)
    ordinary = list(
        _zones(
            [_model(labels)] * 2,
            [START, START],
            *arguments,
            depth_m=np.full(domain.shape, 0.005),
        )
    )
    observed = list(
        _zones(
            [_model(with_seal), _model(labels)],
            [START, START],
            *arguments,
            depth_m=np.full(domain.shape, 0.005),
        )
    )
    for left, right in zip(ordinary, observed):
        np.testing.assert_array_equal(left.contour, right.contour & ~seal_pixels)
        assert right.contour[seal_pixels].all()
        assert left.diagnostics["depth_margin"] == right.diagnostics["depth_margin"]
    assert observed[0].diagnostics["model_gas_on_seals_pixels"] == 4
    assert observed[1].diagnostics["model_gas_on_seals_pixels"] == 0
    assert observed[1].diagnostics["model_gas_seen_on_seals_pixels"] == 4


def test_prefix_contour_is_causal_never_shrinks_and_admits_late_new_gas():
    image, positions, domain = _scene((120, 150))
    first = np.full(domain.shape, PhaseLabel.WATER, np.uint8)
    first[25:45, 25:45] = PhaseLabel.GAS
    absent = np.full_like(first, PhaseLabel.WATER)
    late = absent.copy()
    late[85:105, 105:125] = PhaseLabel.GAS
    dates = [START + pd.Timedelta(hours=h) for h in (0, 1, 3, 4)]
    args = (image, domain, {1: 1}, _protocol(positions), RULES, SETTINGS)
    prefix = list(_zones([_model(first)], dates[:1], *args))
    full = list(
        _zones(
            [_model(first), _model(absent), _model(late), _model(absent)], dates, *args
        )
    )
    np.testing.assert_array_equal(full[0].contour, prefix[0].contour)
    assert not full[0].contour[85:105, 105:125].any()
    for previous, current in zip(full, full[1:]):
        assert not (previous.contour & ~current.contour).any()
    np.testing.assert_array_equal(full[0].contour, full[1].contour)
    np.testing.assert_array_equal(full[2].contour, full[3].contour)
    assert full[2].contour[85:105, 105:125].all()


def test_active_link_is_retained_after_injection_without_future_evidence():
    image, positions, domain = _scene()
    labels = np.full(domain.shape, PhaseLabel.WATER, np.uint8)
    labels[10:24, 38:53] = PhaseLabel.GAS
    connected = labels.copy()
    connected[24:61, 32:59] = PhaseLabel.DISSOLVED
    settings = replace(SETTINGS, closing_length_mm=0, contour_smoothing_mm=(0, 0))
    zones = list(
        _zones(
            [_model(labels), _model(connected), _model(labels)],
            [START, START + pd.Timedelta(hours=1), START + pd.Timedelta(hours=3)],
            image,
            domain,
            {1: 1},
            _protocol(positions, (60, 45)),
            RULES,
            settings,
        )
    )
    assert zones[0].links and zones[0].contour[60, 45]
    first_only = list(
        _zones(
            [_model(labels)],
            [START],
            image,
            domain,
            {1: 1},
            _protocol(positions, (60, 45)),
            RULES,
            settings,
        )
    )[0]
    np.testing.assert_array_equal(zones[0].contour, first_only.contour)
    assert zones[1].links and zones[1].contour[60, 45]
    assert not zones[2].links and zones[2].contour[60, 45]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fft_normalisation_preserves_a_single_scored_probability(dtype):
    values = np.zeros((301, 401), dtype)
    scored = np.zeros_like(values, bool)
    scored[150, 200] = True
    values[scored] = 0.4
    score = allocation._round_score(scored, values, (0.5, 0.5), 10 * np.sqrt(12))
    rows, cols = np.mgrid[:301, :401]
    support = (rows - 150) ** 2 + (cols - 200) ** 2 <= 60**2
    np.testing.assert_allclose(
        score[support], 0.4, rtol=2e-5 if dtype == np.float32 else 1e-12
    )
    assert not score[~support].any()


def test_isolated_co2_noise_gets_no_added_gas_even_inside_fixed_contour():
    shape = (70, 90)
    gas = np.zeros(shape, bool)
    gas[30:40, 35:45] = True
    candidate = np.zeros_like(gas)
    candidate[25:45, 30:50] = True
    candidate[12, 20] = candidate[55, 70] = True
    candidate &= ~gas
    saturation = gas.astype(float) * 0.8
    allocation._add(
        saturation,
        25,
        gas,
        candidate,
        np.full(shape, 0.8),
        np.full(shape, 0.8),
        (gas | candidate, np.full(shape, 0.2)),
        gas,
        np.ones(shape),
        (1, 1),
        SETTINGS,
    )
    added = (saturation > 0) & ~gas
    assert not added[12, 20] and not added[55, 70]
    assert not (added & ~allocation._touching(gas | added, gas)).any()
    assert np.isclose(saturation[candidate].sum(), 25)


@pytest.mark.parametrize(
    "field,value",
    [
        ("closing_length_mm", -1),
        ("closing_length_mm", np.inf),
        ("contour_smoothing_mm", [1]),
        ("contour_blend_mm", [1, 2, np.nan]),
        ("link_smoothing_mm", [-1, 2]),
        ("link_width_mm", 0),
        ("fade_length_mm", np.inf),
    ],
)
def test_fixed_zone_settings_reject_invalid_geometry(field, value):
    with pytest.raises(ValueError):
        replace(SETTINGS, **{field: value})


def test_the_inward_fade_is_zero_at_the_limit_edge_and_outside():
    limit = np.zeros((21, 21), bool)
    limit[3:18, 3:18] = True
    weight = allocation._inward_weight(limit, (1, 1), 2)
    assert not weight[~limit].any()
    assert not weight[3].any() and not weight[:, 3].any()
    assert 0 < weight[4, 10] < weight[10, 10] < 1
    assert weight[4, 10] == -np.expm1(-0.5)


def test_one_high_probability_pixel_has_a_round_gaussian_score():
    values = np.zeros((41, 41), np.float64)
    values[20, 20] = 1
    score = allocation._round_score(np.ones_like(values, bool), values, (1, 1), 12)
    np.testing.assert_allclose(score, score.T, rtol=1e-12, atol=0)
    np.testing.assert_allclose(score[20, 25], score[23, 24], rtol=1e-12)
    assert score[20, 25] > score[25, 25]
    rows, cols = np.mgrid[:41, :41]
    outside = (rows - 20) ** 2 + (cols - 20) ** 2 > (3 * 12 / np.sqrt(12)) ** 2
    assert not score[outside].any()


def test_recorded_limit_is_current_co2_plus_the_explicit_hidden_bridge(tmp_path):
    from tests.unit.test_gas_allocation import _frame, _run

    frame = _frame([np.s_[15:40, 25:55]], [np.s_[23:30, 35:45]])
    records = []
    made = _run(tmp_path, [frame], [0], [1.0], record=lambda i, r: records.append(r))
    co2 = allocation._co2(made[0][0])
    bridge = made[0][1] == PhaseRule.GAS_BRIDGE
    assert 0 < records[0].limit_area_mm2 <= int(co2.sum()) + records[0].bridge_area_mm2
    assert not made[0][2][~(co2 | bridge)].any()
    assert (made[0][0][bridge] == PhaseLabel.WATER).all()
    assert records[0].disconnected_added_area_mm2 == 0


def test_added_gas_obeys_the_limit_even_in_a_connected_finger():
    from tests.unit.test_gas_allocation import SETTINGS

    shape = (25, 25)
    saturation = np.zeros(shape, np.float64)
    gas = np.zeros(shape, bool)
    gas[8:12, 10:15] = True
    saturation[gas] = 0.8
    limit = np.zeros(shape, bool)
    limit[2:13, 5:20] = True
    candidate = ~gas
    weight = allocation._inward_weight(limit, (1, 1), 2)
    allocation._add(
        saturation,
        20,
        gas,
        candidate,
        np.full(shape, 0.8),
        np.full(shape, 0.8),
        (np.ones(shape, bool), np.ones(shape, np.float64)),
        gas,
        weight,
        (1, 1),
        SETTINGS,
    )
    assert np.isclose(saturation[~gas].sum(), 20)
    assert not saturation[~limit].any()
    assert (saturation[gas] == 0.8).all()
    saturation[~gas] = 0
    capacity = float(np.sum(np.full(shape, 0.8)[candidate & (weight > 0)]))
    allocation._add(
        saturation,
        1e6,
        gas,
        candidate,
        np.full(shape, 0.8),
        np.full(shape, 0.8),
        (np.ones(shape, bool), np.ones(shape, np.float64)),
        gas,
        weight,
        (1, 1),
        SETTINGS,
    )
    assert np.isclose(saturation[~gas].sum(), capacity)


def test_the_volume_fits_or_is_capped_with_a_numeric_record(tmp_path):
    from tests.unit.test_gas_allocation import PORE, SWI, _frame, _run

    frame = _frame([np.s_[8:45, 12:68]], [np.s_[20:30, 30:50]])
    model = 200 * PORE * (1 - float(SWI))
    for target in (model + 2e-7, 1.0):
        records = []
        made = _run(
            tmp_path,
            [frame],
            [0],
            [target],
            record=lambda i, record: records.append((i, record)),
        )
        index, record = records[0]
        assert index == 0 and record.target_m3 == target
        assert record.limit_area_mm2 > 0
        assert record.placed_m3 == float(np.sum(PORE * made[0][2].astype(np.float64)))
        if target < 1:
            assert np.isclose(record.placed_m3, target, rtol=1e-7)
        else:
            assert record.unplaced_m3 > 0
            assert np.isclose(record.placed_m3 + record.unplaced_m3, target)


def test_added_volume_cannot_enlarge_the_later_model_gas_footprint(tmp_path):
    from tests.unit.test_gas_allocation import _frame, _run

    co2, gas = np.s_[10:48, 12:68], np.s_[25:34, 34:46]
    before = _frame([co2], [gas])
    after = _frame([co2], [np.s_[20:36, 22:58]], level=0.7)
    model_only = _run(tmp_path, [before, after], [0, 2], [None, None], end_h=1)
    overflow = _run(tmp_path, [before, after], [0, 2], [1.0, None], end_h=1)
    assert np.count_nonzero(overflow[0][2]) > np.count_nonzero(model_only[0][2])
    for actual, expected in zip(overflow[1], model_only[1]):
        np.testing.assert_array_equal(actual, expected)
