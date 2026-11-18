"""Ordered active-injector routes through known porous water and CO2."""

from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.multiphase import gas_zone
from darsia.multiphase.gas_model_cache import GasModelFrame
from darsia.multiphase.gas_parameters import GasAllocationSettings, parameters_at_depth
from darsia.multiphase.phase_rules import PhaseLabel


def scene(shape=(40, 50), pitch=(1.0, 1.0)):
    image = darsia.Image(
        np.ones(shape, int), dimensions=np.asarray(shape) * pitch / 1000
    )
    columns, rows = darsia.pixel_centres(image)
    x, y = np.broadcast_arrays(columns[None, :], rows[:, None])
    positions = SimpleNamespace(x=x, y=y, pitch_mm=tuple(pitch))
    return image, positions, np.ones(shape, bool)


def settings():
    return replace(
        parameters_at_depth(GasAllocationSettings(1, 1, 1, 1, 1, 1, 1, 1), 5),
        closing_length_mm=0,
        contour_smoothing_mm=(0, 0),
        contour_blend_mm=(4, 10, 4),
        link_smoothing_mm=(4, 8),
        link_width_mm=16,
        fade_length_mm=10,
    )


def port(positions, pixel):
    return float(positions.x[pixel]), float(positions.y[pixel])


def check_route(line, coordinates, path, start, targets):
    assert coordinates.dtype == np.int32 and coordinates.shape[1] == 2
    assert tuple(coordinates[0]) == start
    assert targets[tuple(coordinates[-1])]
    reconstructed = np.zeros_like(line)
    reconstructed[tuple(coordinates.T)] = True
    np.testing.assert_array_equal(reconstructed, line)
    assert not (line & ~path).any()
    assert np.all(np.max(np.abs(np.diff(coordinates, axis=0)), axis=1) == 1)


def test_ordered_coordinates_preserve_existing_cv2_raster_and_endpoints():
    random = np.random.default_rng(5754)
    for count in (1, 2, 7, 30):
        for _ in range(8):
            points = random.uniform(0, 29, (count, 2))
            coordinates = gas_zone._path_coordinates(points)
            raster = np.zeros((30, 30), bool)
            raster[tuple(coordinates.T)] = True
            np.testing.assert_array_equal(
                raster, gas_zone._raster_path(points, raster.shape)
            )
            np.testing.assert_array_equal(
                coordinates[[0, -1]], np.rint(points[[0, -1]])
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hidden_segment_is_straight_and_seen_cost_uses_probability_median(
    monkeypatch, dtype
):
    _, positions, path = scene()
    contour = np.zeros_like(path)
    contour[4:9, 22:28] = True
    observed = contour.copy()
    observed[8:25, 24:27] = True
    current = contour.copy()
    current[24, 25] = True
    start = (34, 25)
    path[12:16, 8:12] = False
    probability = np.linspace(0.1, 0.9, path.size, dtype=dtype).reshape(path.shape)
    original = probability.copy()
    costs = []
    native = gas_zone.MCP_Geometric

    def record(cost, **kwargs):
        costs.append(cost.copy())
        return native(cost, **kwargs)

    monkeypatch.setattr(gas_zone, "MCP_Geometric", record)
    line, coordinates, note = gas_zone._injector_link(
        port(positions, start),
        contour,
        observed,
        probability,
        path,
        positions.x, positions.y, positions.pitch_mm,
        settings(),
        current_co2=current,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    check_route(line, coordinates, path, start, contour & observed)
    assert note["status"] == "linked" and note["outside_observed_co2_pixels"] > 1
    meeting = (24, 25)
    assert note["meeting_pixel"] == list(meeting)
    hidden = gas_zone._path_coordinates(np.asarray([start, meeting]))
    np.testing.assert_array_equal(coordinates[: len(hidden)], hidden)
    assert observed[tuple(coordinates[len(hidden) - 1 :].T)].all()
    signal = gas_zone._gaussian(probability, 4, positions.pitch_mm)
    reference = np.median(signal[observed])
    assert note["p_ref"] == float(reference)
    box = (slice(4, 25), slice(22, 28))
    inside = observed[box]
    np.testing.assert_array_equal(
        costs[0][inside], 1.0 / (signal[box][inside] + reference)
    )
    assert np.isinf(costs[0][~inside]).all()
    np.testing.assert_array_equal(probability, original)


def test_hidden_segment_refuses_seal_even_when_detour_and_farther_target_exist():
    _, positions, path = scene()
    path[:32, 25] = False
    contour = np.zeros_like(path)
    contour[20, 35] = True
    contour[0, 0] = True
    start = (20, 12)
    line, coordinates, note = gas_zone._injector_link(
        port(positions, start),
        contour,
        contour,
        np.ones(path.shape),
        path,
        positions.x, positions.y, positions.pitch_mm,
        settings(),
        current_co2=contour,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    assert line is coordinates is None
    assert note["status"] == "straight_hidden_segment_blocked"
    assert note["meeting_pixel"] == [20, 35]


def test_seen_segment_can_follow_observed_co2_around_seal():
    _, positions, path = scene()
    path[:32, 25] = False
    contour = np.zeros_like(path)
    contour[20, 35] = True
    current = contour.copy()
    current[25, 12] = True
    start = (34, 12)
    line, coordinates, note = gas_zone._injector_link(
        port(positions, start),
        contour,
        path,
        np.ones(path.shape),
        path,
        positions.x, positions.y, positions.pitch_mm,
        settings(),
        current_co2=current,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    check_route(line, coordinates, path, start, contour)
    assert line[32:].any() and note["status"] == "linked"
    assert note["meeting_pixel"] == [25, 12]


@pytest.mark.parametrize("measured", [False, True])
def test_nearest_current_meeting_uses_physical_not_pixel_distance(measured):
    _, positions, path = scene(pitch=(3.0, 1.0))
    if measured:
        # A projective physical map preserves lines but varies pixel distances.
        denominator = 1 + positions.y / 0.1
        positions = SimpleNamespace(
            x=positions.x / denominator,
            y=positions.y / denominator,
            pitch_mm=positions.pitch_mm,
        )
    start = (20, 20)
    current = np.zeros_like(path)
    current[18, 20] = current[20, 24] = True
    anchor = port(positions, start)
    candidates = np.argwhere(current)
    distances = [
        (positions.x[tuple(pixel)] - anchor[0]) ** 2
        + (positions.y[tuple(pixel)] - anchor[1]) ** 2
        for pixel in candidates
    ]
    expected = candidates[np.argmin(distances)]
    assert expected.tolist() == ([18, 20] if measured else [20, 24])
    line, coordinates, note = gas_zone._injector_link(
        anchor,
        current,
        current,
        np.ones(path.shape),
        path,
        positions.x, positions.y, positions.pitch_mm,
        settings(),
        current_co2=current,
        depth_m=np.full(np.shape(current), 0.1),
    )
    check_route(line, coordinates, path, start, current)
    assert note["meeting_pixel"] == expected.tolist()


def test_nearest_current_meeting_does_not_jump_disconnected_seen_co2():
    _, positions, path = scene()
    contour = np.zeros_like(path)
    contour[5, 25] = True
    current = contour.copy()
    current[25, 25] = True
    line, coordinates, note = gas_zone._injector_link(
        port(positions, (34, 25)),
        contour,
        current,
        np.ones(path.shape),
        path,
        positions.x, positions.y, positions.pitch_mm,
        settings(),
        current_co2=current,
        depth_m=np.full(np.shape(contour), 0.1),
    )
    assert line is coordinates is None
    assert note["status"] == "no_observed_co2_path"
    assert note["meeting_pixel"] == [25, 25]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_probability_reference_does_not_invent_finite_cost(dtype):
    _, positions, path = scene()
    current = np.zeros_like(path)
    current[5:10, 25] = True
    configured = replace(settings(), link_smoothing_mm=(0, 0))
    line, coordinates, note = gas_zone._injector_link(
        port(positions, (34, 25)),
        current,
        current,
        np.zeros(path.shape, dtype),
        path,
        positions.x, positions.y, positions.pitch_mm,
        configured,
        current_co2=current,
        depth_m=np.full(np.shape(current), 0.1),
    )
    assert line is coordinates is None
    assert note["p_ref"] == 0 and note["status"] == "no_finite_path"


def test_zero_reference_keeps_positive_probability_cells_usable(monkeypatch):
    _, positions, path = scene()
    current = np.zeros_like(path)
    current[5:10, 25] = True
    probability = np.zeros(path.shape)
    probability[9, 25] = 0.5
    configured = replace(settings(), link_smoothing_mm=(0, 0))
    costs = []
    native = gas_zone.MCP_Geometric

    def record(cost, **kwargs):
        costs.append(cost.copy())
        return native(cost, **kwargs)

    monkeypatch.setattr(gas_zone, "MCP_Geometric", record)
    line, coordinates, note = gas_zone._injector_link(
        port(positions, (34, 25)),
        current,
        current,
        probability,
        path,
        positions.x, positions.y, positions.pitch_mm,
        configured,
        current_co2=current,
        depth_m=np.full(np.shape(current), 0.1),
    )
    check_route(line, coordinates, path, (34, 25), current)
    assert note["p_ref"] == 0
    assert costs[0][-1, 0] == 2 and np.isinf(costs[0][:-1]).all()


def test_water_anchor_inside_contour_still_routes_to_current_co2():
    _, positions, path = scene()
    current = np.zeros_like(path)
    current[5:9, 20:30] = True
    start = (25, 25)
    line, coordinates, note = gas_zone._injector_link(
        port(positions, start),
        path,
        path,
        np.ones(path.shape),
        path,
        positions.x, positions.y, positions.pitch_mm,
        settings(),
        current_co2=current,
        depth_m=np.full(np.shape(path), 0.1),
    )
    check_route(line, coordinates, path, start, current)
    assert note["status"] == "linked"
    current[start] = True
    line, coordinates, note = gas_zone._injector_link(
        port(positions, start),
        path,
        path,
        np.ones(path.shape),
        path,
        positions.x, positions.y, positions.pitch_mm,
        settings(),
        current_co2=current,
        depth_m=np.full(np.shape(path), 0.1),
    )
    assert line is coordinates is None and note["status"] == "already_inside"


def test_historical_co2_without_current_target_never_invents_a_meeting():
    _, positions, path = scene()
    contour = np.zeros_like(path)
    contour[5:9, 20:30] = True
    line, coordinates, note = gas_zone._injector_link(
        port(positions, (30, 25)),
        contour,
        path,
        np.ones(path.shape),
        path,
        positions.x, positions.y, positions.pitch_mm,
        settings(),
        current_co2=np.zeros_like(path),
        depth_m=np.full(np.shape(contour), 0.1),
    )
    assert line is coordinates is None and note["status"] == "no_current_co2_target"


def zones(labels, dates, start):
    image, positions, domain = scene(labels[0].shape)
    protocol = SimpleNamespace(
        df=pd.DataFrame(
            [
                {
                    "rate_kg_s": 1,
                    "location_x": port(positions, start)[0],
                    "location_y": port(positions, start)[1],
                    "start": dates[0],
                    "end": dates[0] + timedelta(hours=2),
                }
            ]
        )
    )
    frames = [
        GasModelFrame(
            np.full(domain.shape, 0.5, np.float32),
            frame.copy(),
            frame.copy(),
            np.zeros(domain.shape, np.uint8),
        )
        for frame in labels
    ]
    return list(
        gas_zone.gas_model_zones(
            frames,
            dates,
            image,
            domain,
            {1: 1},
            protocol,
            SimpleNamespace(free_water_facies=(5,), seal_permeability_fraction=0.05),
            settings(),
            
            depth_m=np.full(domain.shape, 0.0001),
        )
    )


def test_links_stop_at_injection_end_and_unknown_barrier_is_never_crossed():
    first = np.full((40, 50), PhaseLabel.WATER, np.uint8)
    first[4:9, 22:28] = PhaseLabel.GAS
    blocked = first.copy()
    blocked[20] = PhaseLabel.UNKNOWN
    dates = [datetime(2000, 1, 1) + timedelta(hours=h) for h in (0, 1, 3)]
    active, unknown, after = zones([first, blocked, first], dates, (34, 25))
    assert len(active.links) == 1
    link = next(iter(active.links.values()))
    assert not link.path_coordinates.flags.writeable
    assert tuple(link.path_coordinates[0]) == (34, 25)
    assert first[tuple(link.path_coordinates[-1])] == PhaseLabel.GAS
    assert not unknown.links and not after.links
    assert (
        next(iter(unknown.diagnostics["ports"].values()))["status"]
        == "straight_hidden_segment_blocked"
    )
    assert not (active.contour & ~after.contour).any()
