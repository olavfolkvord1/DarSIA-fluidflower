"""The reviewed photo/area recipe keeps one transfer and one population scale."""

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from darsia.multiphase.phase_rules import PhaseLabel, PhaseRule
from darsia.presets.workflows.media import films
from darsia.presets.workflows.media.films import FilmSettings, pyramid_frame
from darsia.presets.workflows.media.layouts import paired_grid_layout
from darsia.presets.workflows.media.panels import (
    PhotographSettings,
    outline_masks,
    overlay_outlines,
)
from darsia.presets.workflows.media.tiles import (
    Frame,
    Panel,
    SandGeometry,
    TileSettings,
    draw_tile,
)
from darsia.presets.workflows.media.transfer import (
    area_map,
    divided_size,
    first_frame_limits,
    population_scale,
    probe_indices,
    resized_photo,
    stretched_photo,
)
from tests.unit.test_media import STYLE


@pytest.mark.parametrize("source, name", [("phase", "CO₂"), ("mass", "CO₂ mass")])
def test_outline_caption_preserves_source_and_mathematical_symbols(source, name):
    """Both display contracts identify their CO2 source without encoding damage."""
    panel = Panel(content="image", label="Corrected photograph", display_mean="none")
    row = dict(
        co2_outline=True,
        co2_source=source,
        co2_colour="white",
        gas_contour=True,
        gas_contour_fraction=0.9,
        gas_colour="red",
    )
    assert films._display_notes(panel, row) == [
        f"{name} outline white, gas ≥ 0.9 × (1 − Swi) red"
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("percentile", [0, 25, 50, 99, 99.9, 100])
def test_pooled_tail_matches_exact_numpy_across_runs(dtype, percentile):
    """Unequal positive populations, ties and invalids cannot become mean quantiles."""
    rng = np.random.default_rng(927)
    runs = [
        rng.uniform(-2, 30, size=(7, 13)).astype(dtype),
        rng.uniform(-1, 2, size=(43, 19)).astype(dtype),
        np.ones((5, 11), dtype=dtype),
    ]
    runs[0][0, :4] = [np.nan, np.inf, -np.inf, 0]
    copies = [a.copy() for a in runs]
    values = np.concatenate(
        [a[np.isfinite(a) & (a > 0)].astype(np.float64) for a in runs]
    )
    receipt = population_scale(
        iter(runs),
        maximum_samples=sum(a.size for a in runs) + 500,
        percentile=percentile,
    )
    assert receipt["colour_scale"] == np.percentile(values, percentile)
    assert receipt["positive_samples"] == values.size
    assert receipt["probe_frames"] == 3
    for before, after in zip(copies, runs):
        np.testing.assert_array_equal(before, after)


def test_pooled_scale_is_global_not_average_of_run_scales():
    result = population_scale(
        [np.ones(1000), np.full(1000, 100.0)], maximum_samples=2000
    )
    assert result["colour_scale"] == 100
    assert result["retained_samples"] <= result["buffer_capacity"] <= 8
    with pytest.raises(ValueError, match="no finite positive"):
        population_scale([np.array([0, -1, np.nan, np.inf])], maximum_samples=4)
    with pytest.raises(ValueError, match="exceed"):
        population_scale([np.ones(5)], maximum_samples=4)


def test_probe_indices_and_quarter_grid_preserve_logical_odd_height():
    assert probe_indices(79) == (9, 19, 39, 78)
    assert probe_indices(2) == (0, 1)
    assert divided_size((2982, 5472), 4) == (1368, 745)
    layout = paired_grid_layout((2856, 746), (1368, 745), 120)
    assert layout.cells == ((0, 0, 1368, 745), (1368, 0, 1368, 745))
    with pytest.raises(ValueError, match="even padding"):
        paired_grid_layout((2856, 744), (1368, 745), 120)


def test_first_frame_transfer_matches_producer_and_does_not_remeasure_later():
    image = np.linspace(0.03, 0.9, 20 * 32 * 3, dtype=np.float32).reshape(20, 32, 3)
    size = (8, 5)
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    limits = first_frame_limits(image, size)
    np.testing.assert_array_equal(limits, np.percentile(small, [0.5, 99.5]))
    expected = (
        np.clip((small - limits[0]) / (limits[1] - limits[0]), 0, 1) ** (1 / 2.2) * 255
    ).astype(np.uint8)
    np.testing.assert_array_equal(
        stretched_photo(image, size, limits, 2.2, 40), expected
    )
    later = image * 0.4
    assert stretched_photo(later, size, limits, 2.2, 40).mean() < expected.mean()
    invalid = image.copy()
    invalid[:4, :4] = np.nan
    np.testing.assert_array_equal(
        stretched_photo(invalid, size, limits, 2.2, 40)[0, 0], [40, 40, 40]
    )


def test_white_mass_contour_keeps_hidden_mass_and_excludes_unknowns():
    phase = np.array(
        [
            [
                PhaseLabel.WATER,
                PhaseLabel.DISSOLVED,
                PhaseLabel.UNKNOWN,
                PhaseLabel.GAS,
                PhaseLabel.WATER,
            ]
        ]
    )
    mass = np.array([[2.0, 0.0, 3.0, np.nan, 0]])
    rule = np.array([[PhaseRule.GAS_BRIDGE, 0, PhaseRule.GAS_BRIDGE, 0, 0]])
    settings = PhotographSettings(
        tuple(map(tuple, np.eye(3))),
        40,
        co2_outline=True,
        gas_contour=True,
        co2_source="mass",
    )
    white, red = outline_masks(
        phase,
        np.ones_like(mass),
        np.zeros_like(mass),
        settings,
        rules=rule,
        mass_area=mass,
    )
    np.testing.assert_array_equal(white, [[1, 0, 0, 0, 0]])
    np.testing.assert_array_equal(red, [[1, 1, 0, 1, 0]])
    with pytest.raises(ValueError, match="mass per area"):
        outline_masks(phase, None, None, replace(settings, gas_contour=False))


def test_producer_contours_reduced_occupancy_without_painting_fragmented_interiors():
    """Many full-resolution boundaries become ONE display-region boundary."""
    mask = np.zeros((64, 64), np.uint8)
    rows, columns = np.indices((32, 32))
    mask[16:48, 16:48] = (rows + columns) % 2 == 0
    empty = np.zeros((16, 16, 3), np.uint8)
    actual = overlay_outlines(
        empty.copy(), (0, 0, 64, 64), [(mask, "white")], 0.72, reduced_occupancy=True
    )
    reduced = (
        cv2.resize(mask.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA) > 0
    ).astype(np.uint8)
    contours, _ = cv2.findContours(reduced, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    expected = empty.copy()
    cv2.drawContours(expected, contours, -1, (255, 255, 255), 1)
    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(actual[..., 0]) == 28
    legacy = overlay_outlines(empty.copy(), (0, 0, 64, 64), [(mask, "white")], 0.72)
    assert np.count_nonzero(legacy[..., 0]) == 64


@pytest.mark.parametrize("pattern", ["rising", "falling", "ties"])
def test_batched_tail_counts_every_positive_before_cutoff_filter(pattern):
    values = np.arange(1, 200001, dtype=np.float64)
    if pattern == "falling":
        values = values[::-1]
    elif pattern == "ties":
        values = values // 1000 + 1
    samples = [values[:70000], np.array([np.nan, -1, np.inf]), values[70000:]]
    result = population_scale(samples, maximum_samples=250000)
    assert result["colour_scale"] == np.percentile(values, 99.9)
    assert result["positive_samples"] == values.size
    assert result["retained_samples"] <= result["buffer_capacity"]


def test_fractional_photo_resize_keeps_every_valid_row():
    image = np.ones((2982, 5472, 3), dtype=np.float32)
    assert np.isfinite(resized_photo(image, (1368, 746))).all()
    image[100, 100] = np.nan
    missing = ~np.isfinite(resized_photo(image, (1368, 746))).all(axis=-1)
    assert 0 < missing.sum() <= 4


def test_display_mask_respects_stored_reporting_domain_and_keeps_source():
    values = np.full((6, 8, 3), 0.5, dtype=np.float32)
    values[-2:] = [1, 0, 1]
    support = np.ones(values.shape[:2], dtype=bool)
    support[-2:] = False
    frame = Frame(
        "case",
        1,
        1,
        values.shape[:2],
        (0, 0, 8, 6),
        image=values,
        reporting_domain=support,
    )
    settings = PhotographSettings(restrict_to_reporting_domain=True)
    shown = frame.photo_samples(settings, step=2)
    assert np.isnan(shown[-1]).all()
    np.testing.assert_array_equal(shown[:-1], values[:4:2, ::2])
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(
        frame.photo_samples(
            replace(settings, restrict_to_reporting_domain=False), step=2
        ),
        values[::2, ::2],
    )


def test_area_resampling_keeps_unknown_separate_from_no_pore():
    values = np.array([[1, 1, np.nan, 0, 0]], dtype=np.float32)
    volume = np.array([[1, 1, 1, 0, np.nan]])
    pixels, shown, known = area_map(
        values,
        volume,
        (5, 1),
        10,
        colour_map="viridis",
        no_value="#abcdef",
        over_range="#ff00ff",
    )
    np.testing.assert_array_equal(pixels[0, 3], [40, 40, 40])
    np.testing.assert_array_equal(pixels[0, 2], [171, 205, 239])
    np.testing.assert_array_equal(pixels[0, 4], [171, 205, 239])
    np.testing.assert_array_equal(known, [[1, 1, 0, 0, 0]])


def test_actual_paired_render_keeps_pixels_padding_and_transfer_receipts():
    shape = (300, 600)
    image = np.linspace(0.01, 0.9, np.prod(shape) * 3, dtype=np.float32).reshape(
        *shape, 3
    )
    sand = SandGeometry(
        np.ones(shape),
        [1],
        np.full(shape, 1e-6),
        np.full(shape, 0.01),
        np.full(shape, 1e-4),
        np.zeros(shape),
    )
    frame = Frame(
        "frame",
        1.98,
        1.98,
        shape,
        (0, 0, shape[1], shape[0]),
        image=image,
        maps={"mass": np.full(shape, 0.1)},
        unit="g/L",
        phase=np.full(shape, PhaseLabel.DISSOLVED),
        saturation=np.zeros(shape),
        sand=sand,
    )
    settings = TileSettings(
        150, 150, 2, mass_quantity="area", normalization="sqrt", grid_divisor=4
    )
    photo = PhotographSettings(
        tuple(map(tuple, np.eye(3))), 40, transfer="first-frame", co2_source="mass"
    )
    limits = first_frame_limits(image, (150, 75))
    panels = (
        Panel("image", "Corrected photo", "none"),
        Panel("mass", "CO2 mass per area", "none"),
    )
    drawn = [
        draw_tile(
            p,
            frame,
            settings,
            style=STYLE,
            photograph_settings=photo,
            photograph_limits=limits,
        )
        for p in panels
    ]
    rows = [
        [
            dict(
                case_id="frame",
                elapsed_h=1.98,
                target_h=1.98,
                size=[150, 75],
                path=str(i),
                **r,
            )
        ]
        for i, (_pixels, r) in enumerate(drawn)
    ]
    film = FilmSettings(
        "mass", "paired-grid", panels, (420, 76), 6, "mp4v", "stored maps"
    )
    rendered, choices, note = pyramid_frame(
        "run",
        rows,
        1.98,
        1,
        1,
        film=film,
        tile_settings=settings,
        unit="g/L",
        style=STYLE,
        read=lambda path, size: drawn[int(path)][0][..., ::-1],
    )
    assert rendered.shape == (76, 420, 3)
    np.testing.assert_array_equal(rendered[-1], np.full((420, 3), 40))
    assert choices[0][3]["photograph_limits"] == list(limits)
    assert choices[1][3]["logical_panel_size"] == [150, 75]


def test_workflow_fixes_earliest_photo_transfer_and_records_actual_times(
    tmp_path, monkeypatch
):
    """Reversed authoring order cannot remeasure limits on a later acquisition."""
    from darsia.presets.workflows.media import steps
    from darsia.presets.workflows.media.tiles import read_tile_index

    shape = (40, 80)
    first = np.linspace(0.01, 0.9, 40 * 80 * 3, dtype=np.float32).reshape(*shape, 3)
    images = {"first": first, "later": first * 0.2}
    dates = {"first": 1.98, "later": 2.03}
    frames = [SimpleNamespace(session="s", name=name) for name in ("later", "first")]
    image = lambda data: SimpleNamespace(img=data)
    rig = SimpleNamespace(
        labels=image(np.ones(shape)),
        facies=image(np.ones(shape)),
        depth=image(np.full(shape, 0.01)),
        pixel_area=image(np.full(shape, 1e-4)),
    )
    geometry = SimpleNamespace(pore_volume_m3=np.full(shape, 1e-6), swi=np.zeros(shape))
    monkeypatch.setattr(
        steps,
        "run_grid",
        lambda value, material: SimpleNamespace(rig=rig, geometry=geometry),
    )
    monkeypatch.setattr(steps, "run_settings", lambda *args: {})
    monkeypatch.setattr(
        steps,
        "make_experiment",
        lambda *args: SimpleNamespace(
            get_datetime=lambda name: dates[name], time_since_start=lambda date: date
        ),
    )
    photo = PhotographSettings(
        tuple(map(tuple, np.eye(3))), 40, transfer="first-frame", co2_source="mass"
    )
    panels = (Panel("image", "Photo", "none"), Panel("mass", "Mass", "none"))
    film = FilmSettings(
        "mass", "paired-grid", panels, (160, 10), 6, "mp4v", "stored maps"
    )
    settings = {
        "style": STYLE,
        "tiles": TileSettings(
            20, 20, 2, mass_quantity="area", normalization="sqrt", grid_divisor=4
        ),
        "photograph": photo,
        "media": steps.MediaSettings((film,)),
        "material": None,
        "domains": SimpleNamespace(sand_labels=[1]),
    }

    def read(step, reader, frame=None):
        if step == "rig":
            return rig
        if step == "corrected-image":
            return image(images[frame.name])
        if step == "mass-maps":
            return SimpleNamespace(
                **{name: image(np.full(shape, 0.1)) for name in steps.MAPS}
            )
        if step == "phase-maps":
            return {
                "labels": np.full(shape, PhaseLabel.DISSOLVED),
                "saturation": np.zeros(shape),
                "rules": np.zeros(shape),
            }
        raise AssertionError(step)

    ctx = SimpleNamespace(
        task=SimpleNamespace(run="run", frames=frames),
        settings=settings,
        tables={"ensemble": {"target_h": {"run": {"s/first": 1.95, "s/later": 2.05}}}},
        read=read,
        photo=lambda run, frame: frame.name,
        publish=lambda writer: writer(tmp_path),
    )
    steps._film_tiles(ctx)
    rows = read_tile_index(tmp_path / "00")
    assert [row["elapsed_h"] for row in rows] == [1.98, 2.03]
    assert [row["target_h"] for row in rows] == [1.95, 2.05]
    expected = list(first_frame_limits(first, (20, 10)))
    assert all(row["photograph_limits"] == expected for row in rows)
    assert [row["case_id"] for row in rows] == ["s/first", "s/later"]
    pixels = [cv2.imread(row["path"]) for row in rows]
    assert pixels[1].mean() < pixels[0].mean()


def test_single_graph_is_authored_per_run_without_cross_run_time_alignment(
    tmp_path, monkeypatch
):
    """The separate graph route reuses the standard writer once for each run."""
    from darsia.presets.workflows.media import films

    panel = Panel("graph", "Whole-domain CO2 mass", "none")
    film = FilmSettings(
        "graph", "single", (panel,), (640, 480), 6, "mp4v", "stored domain mass"
    )
    calls = []

    def write(tiles, output, **kwargs):
        calls.append((tiles, kwargs["film"]))
        return {"name": kwargs["film"].name}

    monkeypatch.setattr(films, "write_grid_film", write)
    frames = {"first": [[{"elapsed_h": 1.98}]], "second": [[{"elapsed_h": 2.03}]]}
    result = films.write_films(
        frames,
        tmp_path,
        film=film,
        tile_settings=TileSettings(160, 160, 1),
        unit="g/L",
        style=STYLE,
    )
    assert [row["name"] for row in result] == ["graph_first", "graph_second"]
    assert calls[0][0] == {"first": [{"elapsed_h": 1.98}]}
    assert calls[1][0] == {"second": [{"elapsed_h": 2.03}]}


def test_single_graph_label_margin_survives_drawing_storage_and_film_adapter(
    tmp_path, monkeypatch
):
    """A separately authored graph must reserve space for its overlaid run name."""
    from contextlib import nullcontext

    from darsia.presets.workflows.media import films, steps

    panel = Panel("graph", "Whole-domain mass", "none")
    film = FilmSettings(
        "graph", "single", (panel,), (640, 480), 6, "mp4v", "stored maps"
    )
    media = steps.MediaSettings((film,))
    (drawing,) = steps.drawings(media)
    assert drawing["label_room"] is True
    rows = [{"label_room": True, "elapsed_h": 1.98}]
    films._drawn_as_chosen(panel, [rows], film)
    with pytest.raises(ValueError, match="label_room"):
        films._drawn_as_chosen(panel, [[{"label_room": False}]], film)
    called = []

    def write(tiles, output, **options):
        called.append(tiles)
        films._drawn_as_chosen(panel, [tiles["run"][0]], options["film"])
        return [{"name": "graph_run"}]

    monkeypatch.setattr(steps, "write_films", write)
    ctx = SimpleNamespace(
        task=SimpleNamespace(part="graph"),
        settings={"media": media, "tiles": TileSettings(160, 160, 1), "style": STYLE},
        selection=SimpleNamespace(frames={"run": []}),
        read=lambda *args, **kwargs: [(drawing, rows)],
        host_lock=lambda *args: nullcontext(),
        publish=lambda writer: writer(tmp_path),
    )
    steps._films(ctx)
    assert called == [{"run": [rows]}]
    assert (tmp_path / steps.RECORDS).is_file()


def test_real_paired_mp4_roundtrip_preserves_odd_logical_panel_and_even_padding(
    tmp_path,
):
    """A real mp4v encode/decode records logical75 rows and encoded76 rows."""
    from darsia.presets.workflows.media.films import write_films

    panels = (
        Panel("image", "CO2 mass white; gas red", "none"),
        Panel("mass", "CO2 mass per area", "none"),
    )
    settings = TileSettings(
        150, 150, 10, mass_quantity="area", normalization="sqrt", grid_divisor=4
    )
    sets = []
    for number in range(2):
        path = tmp_path / f"tile-{number}.png"
        image = np.full((75, 150, 3), 70 + 40 * number, np.uint8)
        image[-1, :] = (20, 180, 40)
        assert cv2.imwrite(str(path), image)
        row = dict(
            case_id="frame",
            elapsed_h=1.98,
            target_h=1.98,
            size=[150, 75],
            path=str(path),
            display_mean="none",
        )
        if number:
            row.update(
                mass_quantity="area",
                normalization="sqrt",
                colour_scale=10,
                over_range=STYLE.over_range,
                clipped_share=0,
                logical_panel_size=[150, 75],
            )
        sets.append([row])
    film = FilmSettings(
        "mass", "paired-grid", panels, (420, 76), 6, "mp4v", "stored maps"
    )
    output = tmp_path / "output"
    records = write_films(
        {"run": sets},
        output,
        film=film,
        tile_settings=settings,
        unit="g/L",
        style=STYLE,
    )
    assert records[0]["decoded_frames"] == 1 and records[0]["fps"] == 6
    assert records[0]["logical_panel_size"] == [150, 75]
    assert records[0]["logical_size"] == [420, 75]
    assert records[0]["encoded_size"] == [420, 76]
    saved = cv2.imread(str(output / records[0]["figure"]))
    np.testing.assert_array_equal(saved[-1], np.full((420, 3), 40))
    np.testing.assert_array_equal(saved[-2, :300], np.tile([20, 180, 40], (300, 1)))
