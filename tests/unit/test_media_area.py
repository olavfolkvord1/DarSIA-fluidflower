"""Shared photo-and-mass panels keep physical units, masks and clipping visible."""

from dataclasses import replace

import cv2
import numpy as np
import pytest
from matplotlib import colormaps

from darsia.multiphase.phase_rules import PhaseLabel
from darsia.presets.workflows.media.display_mean import mass_per_area, pooled_mean
from darsia.presets.workflows.media.films import (
    FilmSettings,
    grid_frame,
    pyramid_frame,
    write_films,
)
from darsia.presets.workflows.media.layouts import side_by_side_layout
from darsia.presets.workflows.media.panels import (
    PhotographSettings,
    colour_norm,
    image_tile,
    map_tile,
    outline_masks,
)
from darsia.presets.workflows.media.tiles import (
    Frame,
    Panel,
    SandGeometry,
    TileSettings,
    draw_tile,
    read_tile_index,
    save_tile,
    write_tile_index,
)
from tests.unit.test_media import STYLE


def test_area_density_preserves_unknown_positive_pore_and_integrated_mass():
    """The conversion and area-weighted pooling conserve the known grams."""
    density = np.array([[2.0, 4.0, np.nan, np.nan, 3.0, 2.0]])
    volume = np.array([[2e-6, 6e-6, 1e-6, 0.0, 1e-6, np.nan]])
    area = np.array([[1e-4, 2e-4, 1e-4, 1e-4, 0.0, 1e-4]])
    shown = mass_per_area(density, volume, area)
    np.testing.assert_allclose(shown[0, :2], [40, 120])
    assert np.isnan(shown[0, 2]) and shown[0, 3] == 0
    assert np.isnan(shown[0, 4:]).all()
    pooled = pooled_mean(shown[:, :2], area[:, :2], 2)
    assert pooled[0, 0] * area[0, :2].sum() == pytest.approx(
        np.sum(density[0, :2] * volume[0, :2] * 1000)
    )


def test_contour_uses_local_available_saturation_and_known_co2():
    """The threshold includes equality and excludes unknown labels or Swi."""
    phase = np.array([[2, 2, 2, 2, 1, 255]], dtype=np.uint8)
    saturation = np.array([[0.45, 0.449, 0.9, 1.0, 1.0, 1.0]])
    swi = np.array([[0.5, 0.5, 0.0, np.nan, 0.0, 0.0]])
    settings = PhotographSettings(
        tuple(map(tuple, np.eye(3))), 128, co2_outline=True, gas_contour=True
    )
    co2, gas = outline_masks(phase, saturation, swi, settings)
    np.testing.assert_array_equal(co2, [[1, 1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(gas, [[1, 0, 1, 0, 1, 0]])
    with pytest.raises(ValueError, match="Swi"):
        outline_masks(phase, saturation, None, settings)
    with pytest.raises(ValueError, match="phase"):
        outline_masks(None, saturation, swi, settings)


@pytest.mark.parametrize("box", [(0, 0, 80, 40), (10, 4, 60, 32)])
def test_full_and_half_resolution_contours_keep_the_same_physical_position(box):
    """Contours are traced before subsampling; both displays use grid coordinates."""
    photo = np.full((40, 80, 3), 0.25, np.float32)
    mask = np.zeros((40, 80), bool)
    mask[10:31, 20:61] = True
    x, y, width, height = box
    photo = photo[y : y + height, x : x + width]
    tiles = [
        image_tile(
            photo[::step, ::step],
            box,
            step,
            (320, 160),
            outlines=[(mask, "red")],
        )
        for step in (1, 2)
    ]
    bounds = []
    for tile in tiles:
        red = (tile[..., 0] > 180) & (tile[..., 1] < 100)
        rows, columns = np.nonzero(red)
        bounds.append((rows.min(), rows.max(), columns.min(), columns.max()))
    # Sampling shifts the photographed cell's centre by at most half a source
    # sample; outline positions themselves must stay within two rendered pixels.
    np.testing.assert_allclose(bounds[0], bounds[1], atol=2)


def test_sqrt_scale_keeps_physical_ticks_and_a_distinct_overrange_colour():
    """One quarter of the physical maximum is halfway through a square-root map."""
    norm = colour_norm(100, "sqrt")
    np.testing.assert_allclose(norm([0, 25, 100]), [0, 0.5, 1])
    assert norm(101) > 1
    shown = np.tile([25.0, 100.0, 101.0, np.nan], (20, 10))
    # Four broad bands give unambiguous rendered pixels away from boundaries.
    shown = np.repeat(shown[:, :4], 20, axis=1)
    tile = map_tile(
        shown,
        (0, 0, 80, 20),
        1,
        (320, 80),
        scale=100,
        label="Mass",
        unit="g/m²",
        style=STYLE,
        normalization="sqrt",
        show_over_range=True,
    )
    np.testing.assert_allclose(
        tile[40, 40], np.array(colormaps[STYLE.colour_map](0.5)[:3]) * 255, atol=1
    )
    np.testing.assert_array_equal(tile[40, 200], [255, 0, 255])


def _paired_tiles(tmp_path):
    shape = (40, 80)
    phase = np.full(shape, PhaseLabel.DISSOLVED, np.uint8)
    phase[10:25, 20:60] = PhaseLabel.GAS
    density = np.full(shape, 0.02)
    density[:, 40:] = 0.2
    sand = SandGeometry(
        np.ones(shape),
        [1],
        np.full(shape, 1e-6),
        np.full(shape, 0.01),
        np.full(shape, 1e-5),
        np.full(shape, 0.5),
    )
    frame = Frame(
        "one",
        2.5,
        2.5,
        shape,
        (0, 0, 80, 40),
        image=np.full((*shape, 3), 0.2),
        maps={"mass": density},
        unit="g/L",
        phase=phase,
        saturation=np.where(phase == PhaseLabel.GAS, 0.45, 0),
        sand=sand,
    )
    settings = TileSettings(400, 40, 10, mass_quantity="area", normalization="sqrt")
    photo = PhotographSettings(
        tuple(map(tuple, np.eye(3))), 128, co2_outline=True, gas_contour=True
    )
    panels = (
        Panel("image", "Corrected photo", "none"),
        Panel("mass", "Total CO₂ mass", "none"),
    )
    sets = []
    for n, panel in enumerate(panels):
        tile, record = draw_tile(
            panel, frame, settings, style=STYLE, photograph_settings=photo
        )
        folder = tmp_path / str(n)
        folder.mkdir()
        write_tile_index(folder, [save_tile(folder, frame, tile, record)])
        sets.append(read_tile_index(folder))
    return frame, panels, sets, settings


def test_pair_render_and_film_records_show_clipping_and_contours(tmp_path):
    """The actual two-panel render and encoded film retain units and clipping."""
    _frame, panels, sets, settings = _paired_tiles(tmp_path)
    film = FilmSettings(
        "pair", "side-by-side", panels, (960, 540), 2, "mp4v", "Stored scientific maps"
    )
    rendered, choices, note = pyramid_frame(
        "run",
        sets,
        2.5,
        1,
        1,
        film=film,
        tile_settings=settings,
        unit="g/L",
        style=STYLE,
    )
    assert "50.00% of known display cells above range" in note
    assert "CO₂ outline white" in note and "0.9 × (1 − Swi) red" in note
    assert choices[1][3]["unit"] == "g/m²"
    assert choices[1][3]["clipped_display_cells"] == 400
    layout = side_by_side_layout(film.size_px, 2)
    x, y, width, height = layout.cells[1]
    np.testing.assert_array_equal(
        rendered[y + height // 2, x + 3 * width // 4], [255, 0, 255]
    )
    cv2.imwrite(str(tmp_path / "two-panel-clipping.png"), rendered)
    out = tmp_path / "film"
    out.mkdir()
    (record,) = write_films(
        {"run": sets}, out, film=film, tile_settings=settings, unit="g/L", style=STYLE
    )
    assert record["frames"][0]["note"] == note
    saved = record["frames"][0]["panels"]
    assert saved[0]["display"]["co2_outline"]
    assert saved[1]["display"]["clipped_share"] == 0.5
    assert (out / record["figure"]).exists()
    with pytest.raises(ValueError, match="fixed scale"):
        write_films(
            {"run": sets},
            tmp_path / "other",
            film=film,
            tile_settings=replace(settings, colour_scale=20),
            unit="g/L",
            style=STYLE,
        )


def test_grid_keeps_each_runs_clipping_caption_and_record(tmp_path):
    """Per-run fractions stay with their cells instead of a repeated global note."""
    _frame, panels, sets, settings = _paired_tiles(tmp_path)
    film = FilmSettings(
        "grid", "grid", (panels[1],), (960, 540), 2, "mp4v", "Stored maps"
    )
    other = [dict(sets[1][0], clipped_display_cells=0, clipped_share=0.0)]
    _pixels, choices, note = grid_frame(
        {"first": sets[1], "second": other},
        2.5,
        1,
        1,
        film=film,
        tile_settings=settings,
        unit="g/L",
        style=STYLE,
    )
    assert note == "Stored maps"
    assert "50.00% of known display cells" in choices["first"][2]
    assert "0.00% of known display cells" in choices["second"][2]
    output = tmp_path / "grid"
    output.mkdir()
    (record,) = write_films(
        {"first": [sets[1]], "second": [other]},
        output,
        film=film,
        tile_settings=settings,
        unit="g/L",
        style=STYLE,
    )
    runs = record["frames"][0]["runs"]
    assert runs["first"]["display"]["clipped_share"] == 0.5
    assert runs["second"]["display"]["clipped_share"] == 0.0


def test_pair_marks_absent_photo_without_inventing_contours(tmp_path):
    """A missing first tile remains empty and has no claimed outline metadata."""
    _frame, panels, sets, settings = _paired_tiles(tmp_path)
    film = FilmSettings(
        "pair", "side-by-side", panels, (960, 540), 2, "mp4v", "Stored maps"
    )
    _pixels, choices, note = pyramid_frame(
        "run",
        [[], sets[1]],
        2.5,
        1,
        1,
        film=film,
        tile_settings=settings,
        unit="g/L",
        style=STYLE,
    )
    assert choices[0][0] is None and choices[0][1]
    assert "no image before" in choices[0][2]
    assert "CO₂ outline" not in note
    assert "50.00% of known display cells" in note


def test_unknown_map_has_no_invented_clipped_percentage(tmp_path):
    """No known data is recorded as unavailable, never as a zero clipped share."""
    frame, panels, _sets, settings = _paired_tiles(tmp_path)
    frame = replace(frame, maps={"mass": np.full(frame.shape, np.nan)})
    _, record = draw_tile(panels[1], frame, settings, style=STYLE)
    assert record["known_display_cells"] == 0 and record["clipped_share"] is None


@pytest.mark.parametrize("value", [float("nan"), 0, -1, float("inf")])
def test_fixed_range_requires_finite_positive_values(value):
    with pytest.raises(ValueError):
        TileSettings(100, 100, value)


def test_legacy_tile_settings_and_default_photo_remain_exact(tmp_path):
    """Explicit legacy defaults and omitted new options render identical bytes."""
    frame, panels, _sets, _settings = _paired_tiles(tmp_path)
    settings = TileSettings(400, 40, 1.7)
    left = draw_tile(panels[1], frame, settings, style=STYLE)
    right = draw_tile(
        panels[1],
        frame,
        replace(settings, normalization="linear", mass_quantity="pore-volume"),
        style=STYLE,
    )
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1] == right[1] == {"display_mean": "none"}
