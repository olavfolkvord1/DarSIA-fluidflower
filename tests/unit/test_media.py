"""Unit tests of films and figures: display means, tiles, layouts and MP4 films."""

import tomllib

import cv2
import numpy as np
import pytest
from matplotlib import colormaps

from darsia.multiphase.phase_rules import PhaseLabel
from darsia.presets.workflows.media.display_mean import (
    block_mean,
    depth_scaled,
    display_step,
    pooled_mean,
    shown_values,
)
from darsia.presets.workflows.media.films import (
    DEPTH_NOTE,
    FilmSettings,
    film_targets,
    tile_at,
    write_films,
)
from darsia.presets.workflows.media.layouts import (
    SEPARATOR,
    compose_grid,
    compose_pyramid,
    grid_layout,
    held,
    pyramid_layout,
)
from darsia.presets.workflows.media.panels import (
    PhotographSettings,
    graph_tile,
    map_tile,
    photograph,
    shared_axes,
)
from darsia.presets.workflows.media.style import Curve, FigureStyle, bgr
from darsia.presets.workflows.media.tiles import (
    Frame,
    Panel,
    SandGeometry,
    TileSettings,
    draw_tile,
    graph_axes,
    read_tile_index,
    save_tile,
    write_tile_index,
)
from darsia.presets.workflows.media.video import MovieWriter

TABLES = tomllib.loads(
    """
    # Our production's figure style: the films' colours and mass curves, and the
    # ensemble figures' palette, resolution and rule for a logarithmic time axis.
    [style]
    dpi = 150
    log_span = 50.0
    surface = "#fcfcfb"
    ink = "#0b0b0b"
    secondary = "#52514e"
    muted = "#898781"
    grid = "#e1e0d9"
    axis = "#c3c2b7"
    washes = ["#ebeae4", "#dcdcd4"]
    series = "#2a78d6"
    frame = "#1e2226"
    empty = "#3e4246"
    no_value = "#e6edf2"
    colour_map = "viridis"
    marker = "#667085"
    curves = [
        {column = "total_mass_g", name = "Total", colour = "#175aad", line = "-"},
        {column = "aqueous_mass_g", name = "Dissolved", colour = "#169879", line = "-"},
        {column = "gas_mass_g", name = "Free gas", colour = "#dc398b", line = "-"},
        {column = "injected_mass_g", name = "Injected", colour = "black", line = "--"},
    ]

    # Our production's films: 1120 pixel wide tiles on a display grid of at most
    # 1280 cells, every map from 0 to 1.7 g/L, the graphs on axes shared by the runs.
    [tiles]
    width_px = 1120
    display_cells = 1280
    colour_scale = 1.7

    # Our production's films ran at 3 frames per second in H.264 ("avc1"), which
    # not every OpenCV build writes; the tests write MPEG-4.
    [video]
    fps = 3.0
    codec = "mp4v"
    """
)
STYLE = FigureStyle(
    **{
        **TABLES["style"],
        "washes": tuple(TABLES["style"]["washes"]),
        "curves": tuple(Curve(**curve) for curve in TABLES["style"]["curves"]),
    }
)
TILES = TileSettings(**TABLES["tiles"], graph_axes=None)
VIDEO = TABLES["video"]
SMALL = TileSettings(width_px=160, display_cells=40, colour_scale=1.7, graph_axes=None)
"""Small tiles for quick tests."""
UNKNOWN = PhaseLabel.UNKNOWN


def rgb(colour: str) -> tuple:
    """A colour as 8-bit (red, green, blue)."""
    return bgr(colour)[::-1]


def layers_frame():
    """Two touching sand layers (10 left, 20 right) of grainy CO2, gas below row 6,
    an unknown phase and a missing value; with pore volumes rising to the right."""
    shape = (10, 16)
    rows, columns = np.indices(shape)
    layers = np.where(columns < 8, 10, 20).astype(np.uint8)
    saturation = np.where(rows < 6, 0.0, 0.5)
    phase = np.where(saturation > 0, PhaseLabel.GAS, PhaseLabel.DISSOLVED)
    phase = phase.astype(np.uint8)
    phase[0, 0] = UNKNOWN
    values = np.where(layers == 10, 5.0, 100.0) + np.where((rows + columns) % 2, 2, -2)
    values[saturation > 0] += 300.0
    values[0, 1] = np.nan
    volume = 1.0 + columns / 10.0
    return values, phase, saturation, layers, volume


# -- display means ---------------------------------------------------------------


def test_each_shown_cell_holds_the_mean_of_its_known_values():
    """Edge blocks hold fewer cells; unknown phases and missing values do not count;
    the means keep the map's precision."""
    values = np.arange(35, dtype=np.float32).reshape(5, 7)
    values[0, 0] = np.nan
    phase = np.zeros((5, 7), np.uint8)
    phase[0, 1] = UNKNOWN
    values[3:, 6] = np.nan
    means = block_mean(shown_values(values, phase), 3)

    assert display_step((5, 7), 3) == 3
    assert means.shape == (2, 3) and means.dtype == np.float32
    assert means[0, 0] == np.float32(np.mean([2, 7, 8, 9, 14, 15, 16]))
    assert means[0, 2] == np.float32(np.mean([6, 13, 20]))
    assert means[1, 1] == np.float32(np.mean([24, 25, 26, 31, 32, 33]))
    assert np.isnan(means[1, 2])


def test_the_depth_mean_stays_inside_each_layer_and_its_water():
    """Two touching layers stay apart, gas and unknown pixels keep their values, and
    each layer keeps its pore-volume-weighted mass."""
    values, phase, saturation, layers, volume = layers_frame()
    before = values.copy()
    depth, area = np.full(values.shape, 0.006), np.full(values.shape, 1e-6)
    shown = shown_values(values, phase)
    mean, window, depth_mm = depth_scaled(
        shown, phase, saturation, layers, [10, 20], volume, depth, area
    )

    assert window == 7 and depth_mm == pytest.approx(6.0)
    assert np.isnan(mean[0, 0]) and np.isnan(mean[0, 1])
    np.testing.assert_array_equal(mean[saturation > 0], values[saturation > 0])
    for label in (10, 20):
        water = (layers == label) & (saturation <= 0) & np.isfinite(shown)
        kept = np.sum(mean[water] * volume[water])
        assert kept == pytest.approx(np.sum(values[water] * volume[water]), rel=1e-13)
        assert np.var(mean[water]) < np.var(values[water])
        if label == 10:
            assert np.max(mean[water]) < 10
        else:
            assert np.min(mean[water]) > 90
    np.testing.assert_array_equal(values, before)


def test_the_depth_mean_keeps_each_part_of_a_layer_apart():
    """Two parts of one sand, three pixels apart across a thin layer of another
    sand: within the disk's reach of each other, yet each keeps its own mean and
    its own pore-volume-weighted total."""
    rows, columns = np.indices((5, 12))
    layers = np.where((columns >= 5) & (columns <= 6), 7, 5)
    values = np.where(columns < 5, 10.0, np.where(columns > 6, 100.0, 50.0))
    values += np.where((rows + columns) % 2, 2.0, -2.0)
    volume = 1.0 + columns / 10.0
    phase = np.full(layers.shape, PhaseLabel.WATER, np.uint8)
    mean, window, _depth_mm = depth_scaled(
        values, phase, np.zeros(layers.shape), layers, [5, 7], volume, 0.007, 1e-6
    )

    assert window == 7
    for part, low, high in ((columns < 5, 6, 14), (columns > 6, 96, 104)):
        kept = np.sum(mean[part] * volume[part])
        assert kept == pytest.approx(np.sum(values[part] * volume[part]), rel=1e-13)
        assert low < mean[part].min() and mean[part].max() < high


def test_the_depth_mean_weighs_pore_volume_in_the_sand_alone():
    """Three water pixels of one layer in a row, beside a label that is not sand: the
    window is the median depth of the sand alone, the sand's mean is weighted by
    pore volume, and the other label keeps its values."""
    layers = np.array([[5, 5, 5, 9, 9, 9, 9]])
    values = np.array([[0.0, 0.0, 1.0, 7.0, 7.5, 8.0, 8.5]])
    volume = np.array([[1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0]])
    depth = np.array([[0.003, 0.003, 0.006, 0.009, 0.009, 0.009, 0.009]])
    phase = np.full(layers.shape, PhaseLabel.WATER, np.uint8)
    mean, window, _depth_mm = depth_scaled(
        values, phase, np.zeros(layers.shape), layers, [5], volume, depth, 1e-6
    )
    # The cross of a 3-pixel window: means 0, 4/6 and 4/5, scaled to keep the mass 4.
    weighted = np.array([0.0, 4 / 6, 4 / 5]) * 4 / (4 / 6 + 4 / 5 * 4)

    assert window == 3
    np.testing.assert_allclose(mean[0, :3], weighted, rtol=1e-12)
    np.testing.assert_array_equal(mean[0, 3:], values[0, 3:])


def test_shown_cells_of_the_depth_mean_are_weighted_by_pore_volume():
    """Each shown cell keeps the pore-volume-weighted mass of its known pixels."""
    values, phase, _saturation, _layers, volume = layers_frame()
    shown = shown_values(values, phase)
    pooled = pooled_mean(shown, volume, 3)
    known = np.isfinite(shown)
    weights = np.add.reduceat(
        np.add.reduceat(np.where(known, volume, 0), np.arange(0, 10, 3), axis=0),
        np.arange(0, 16, 3),
        axis=1,
    )

    assert pooled.shape == (4, 6)
    assert np.sum(pooled * weights) == pytest.approx(
        np.sum(shown[known] * volume[known]), rel=1e-13
    )
    assert pooled_mean(shown, volume, 1) is shown


# -- tiles -------------------------------------------------------------------------


def test_a_photograph_follows_the_colour_transfer_in_8_bits():
    """Linear light through the matrix and the sRGB transfer, rounded to 8 bits;
    pixels without a value are grey."""
    settings = PhotographSettings(
        matrix=((0.5, 0, 0), (0, 1, 0), (0, 0, 2)), invalid_grey=128
    )
    linear = np.array([[[0.002, 0.0, 0.2], [2.0, 0.5, np.nan]]])
    shown = photograph(linear, settings)

    srgb = 12.92 * 0.001, 1.055 * 0.4 ** (1 / 2.4) - 0.055
    expected = np.round(255 * np.array([srgb[0], 0.0, srgb[1]])) / 255
    np.testing.assert_allclose(shown[0, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(shown[0, 1], 128 / 255, rtol=1e-6)


def test_a_map_tile_shows_the_colour_scale_and_cells_without_a_value():
    """A value in the scale's colour map, a missing value in the no-value colour."""
    values = np.full((20, 40), 0.85)
    values[:, 20:] = np.nan
    tile = map_tile(
        values,
        (0, 0, 40, 20),
        1,
        (160, 80),
        scale=1.7,
        label="Total CO₂ mass",
        unit="g/L",
        style=STYLE,
    )
    middle = np.round(255 * np.array(colormaps["viridis"](0.5)[:3]))

    assert tile.shape == (80, 160, 3)
    np.testing.assert_allclose(tile[40, 40], middle, atol=1)
    np.testing.assert_allclose(tile[40, 120], rgb(STYLE.no_value), atol=1)


def mass_rows(count=5, hours=1.0):
    """A run's mass records, one per hour."""
    return [
        dict(
            case_id=f"r-{n}",
            elapsed_h=hours * n,
            total_mass_g=0.1 * n,
            aqueous_mass_g=0.08 * n,
            gas_mass_g=0.02 * n,
            injected_mass_g=0.1 * n,
        )
        for n in range(count)
    ]


def test_the_graph_marks_the_frame_on_shared_axes():
    """The frame's line moves with its time; the shared axes cover every run."""
    runs = [mass_rows(5), mass_rows(3, hours=2.0)]
    axes = shared_axes(runs, STYLE)
    lines = []
    for case in ("r-1", "r-3"):
        tile = graph_tile(runs[0], case, (320, 180), axes=axes, style=STYLE)
        near = np.abs(tile.astype(int) - rgb(STYLE.marker)).max(axis=2) < 30
        lines.append(int(np.argmax(near.sum(axis=0))))

    assert axes == ((0.0, 4 * 1.03), (0.0, 0.4 * 1.05))
    assert graph_axes(TILES, runs, STYLE) == axes
    fixed = TileSettings(**TABLES["tiles"], graph_axes=((0.0, 9.0), (0.0, 1.0)))
    assert graph_axes(fixed, runs, STYLE) == fixed.graph_axes
    assert lines[1] > lines[0] + 50


def test_each_panel_is_drawn_as_its_settings_choose(tmp_path):
    """A depth-scaled map differs from the plain one; a grid's graph keeps room at
    the top; each tile is saved, listed and read back."""
    values, phase, saturation, layers, volume = layers_frame()
    sand = SandGeometry(layers, [10, 20], volume, np.full((10, 16), 0.005), 1e-6)
    frame = Frame(
        case="r-1",
        elapsed_h=1.0,
        target_h=1.0,
        shape=(10, 16),
        box=(0, 0, 16, 10),
        maps={"total_mass": values / 100},
        unit="g/L",
        phase=phase,
        saturation=saturation,
        sand=sand,
    )
    (plain, as_plain), (depth, as_depth) = (
        draw_tile(
            Panel("total_mass", "Total CO₂ mass", mean), frame, SMALL, style=STYLE
        )
        for mean in ("none", "depth")
    )
    wide = TileSettings(400, 40, 1.7, None)
    (graph, _), (roomy, as_roomy) = (
        draw_tile(
            Panel("graph", "Mass graph", "none"),
            frame,
            wide,
            style=STYLE,
            rows=mass_rows(),
            axes=((0, 5), (0, 1)),
            label_room=room,
        )
        for room in (False, True)
    )
    rows = [save_tile(tmp_path, frame, depth, as_depth)]
    write_tile_index(tmp_path, rows)

    assert plain.shape == (100, 160, 3) and as_plain == dict(display_mean="none")
    assert not np.array_equal(plain, depth)
    assert as_depth == dict(display_mean="depth", window_px=5, window_mm=5.0)
    assert as_roomy == dict(label_room=True)
    drawn = [np.argmax((tile < 255).any(axis=(1, 2))) for tile in (graph, roomy)]
    assert drawn[1] > drawn[0] + 0.1 * graph.shape[0]
    (row,) = read_tile_index(tmp_path)
    assert (row["case_id"], row["target_h"], row["size"]) == ("r-1", 1.0, [160, 100])
    assert (row["display_mean"], row["window_px"]) == ("depth", 5)
    shown = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(shown, depth)


# -- layouts -----------------------------------------------------------------------


def solid(colour, size=(112, 61)):
    """A BGR tile of one colour."""
    tile = np.empty((size[1], size[0], 3), np.uint8)
    tile[...] = colour
    return tile


def centre(frame, cell):
    """The pixel at a cell's centre."""
    x, y, width, height = cell
    return frame[y + height // 2, x + width // 2]


def test_pyramid_and_grid_layouts():
    """A pyramid: one cell centred on top of two that fill the frame's width, all of
    one size. A grid: every run's cell, the largest that fit, beside a title column.
    Each tile lands in its cell; a held tile is greyed, a missing one empty; a graph
    below the top has its key in its corner."""
    pyramid = pyramid_layout((1920, 1080), 1120 / 610)
    top, left, right = pyramid.top, pyramid.left_cell, pyramid.right_cell

    assert top[2:] == left[2:] == right[2:] == (958, 522)
    assert left[0] + left[2] + SEPARATOR == right[0] and left[1] == right[1]
    assert left[0] == 1920 - right[0] - right[2] and left[1] + left[3] == 1080
    assert top[0] == (1920 - top[2]) // 2 and top[1] + top[3] + SEPARATOR == left[1]

    grid = grid_layout(37, 1120 / 610, (3840, 2160))
    cells = np.zeros((2160, 3840), int)
    for x, y, width, height in grid.cells:
        cells[y : y + height, x : x + width] += 1
    assert (grid.columns, grid.rows, grid.tile) == (6, 7, (563, 306))
    assert cells.max() == 1 and grid.side[0] >= max(x + w for x, _, w, _ in grid.cells)

    colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200)]
    small = pyramid_layout((640, 360), 112 / 61)
    frame = compose_pyramid(
        [
            (solid(colours[0]), False, ""),
            (solid(colours[1]), True, "held"),
            (None, True, ""),
        ],
        ["Mass graph", "Corrected image", "Total CO₂ mass (g/L)"],
        small,
        lines=["R1", "Elapsed 1.00 h"],
        style=STYLE,
        graph=0,
        scale=(1.7, "g/L"),
    )
    np.testing.assert_array_equal(centre(frame, small.top), colours[0])
    np.testing.assert_array_equal(
        centre(frame, small.left_cell), held(solid(colours[1]))[0, 0]
    )
    np.testing.assert_array_equal(centre(frame, small.right_cell), bgr(STYLE.empty))
    lower = compose_pyramid(
        [
            (solid(colours[1]), False, ""),
            (solid(colours[0]), False, ""),
            (None, True, ""),
        ],
        ["Corrected image", "Mass graph", "Total CO₂ mass (g/L)"],
        small,
        lines=["R1"],
        style=STYLE,
        graph=1,
        scale=None,
    )
    x, y, width, _height = small.left_cell
    corner = lower[y + 8 : y + 20, x + width - 40 : x + width - 8]
    assert (np.abs(corner.astype(int) - colours[0]).max(axis=2) > 60).mean() > 0.5

    runs = grid_layout(3, 112 / 61, (960, 360))
    frame = compose_grid(
        {
            "r1": (solid(colours[0]), False, ""),
            "r2": (None, True, ""),
            "r3": (solid(colours[2]), True, ""),
        },
        runs,
        lines=["Total CO₂ mass"],
        style=STYLE,
        graph=False,
        scale=(1.7, "g/L"),
    )
    np.testing.assert_array_equal(centre(frame, runs.cells[0]), colours[0])
    np.testing.assert_array_equal(centre(frame, runs.cells[1]), bgr(STYLE.empty))
    np.testing.assert_array_equal(
        centre(frame, runs.cells[2]), held(solid(colours[2]))[0, 0]
    )


# -- films -------------------------------------------------------------------------


class Lock:
    """A stand-in for a host lock that records when it is held."""

    def __init__(self):
        self.held = []

    def __enter__(self):
        self.held.append(True)

    def __exit__(self, *_exception):
        self.held.append(False)


def test_a_film_is_refused_unless_it_decodes_to_the_frames_written(tmp_path):
    """Three frames decode as three; a film that lost a frame is refused. The
    caller's lock is held while the encoder opens, and only then."""
    lock = Lock()
    writer = MovieWriter(tmp_path / "a.mp4", (320, 180), **VIDEO, lock=lock)
    assert lock.held == [True, False]
    for level in (0, 120, 240):
        writer.write(solid((level, level, level), (320, 180)))
    assert writer.close() == dict(decoded_frames=3, fps=3.0, size=[320, 180])

    lossy = MovieWriter(tmp_path / "b.mp4", (320, 180), **VIDEO)
    lossy.writer.write(solid((0, 0, 0), (320, 180)))
    lossy.count = 2
    with pytest.raises(ValueError, match="2 frames"):
        lossy.close()


def test_a_panel_without_a_tile_at_a_target_holds_its_last_earlier_tile():
    """The tile of the target; else the last earlier one, held; else none."""
    rows = [dict(target_h=1.0), dict(target_h=2.0), dict(target_h=None)]

    assert tile_at(rows, 2.0) == (rows[1], False)
    assert tile_at(rows, 3.5) == (rows[1], True)
    assert tile_at(rows, 0.5) == (None, True)
    assert film_targets([rows, [dict(target_h=0.5)]]) == [0.5, 1.0, 2.0]


def saved_tiles(folder, colour, targets, drawn):
    """Tiles of one colour saved at the targets, drawn as ``drawn`` says, as their
    read-back rows."""
    rows = [
        save_tile(
            folder,
            Frame(f"c-{n}", t + 0.01, t, (61, 112), (0, 0, 112, 61)),
            solid(colour)[..., ::-1],
            drawn,
        )
        for n, t in enumerate(targets)
    ]
    write_tile_index(folder, rows)
    return read_tile_index(folder)


def test_pyramid_and_grid_films_from_saved_tiles(tmp_path):
    """A run's pyramid and a grid of runs: one frame per target of any tile, a panel
    without a tile at a target held, the depth-scaled maps' window in the caption,
    the film decoded and its last frame saved, named by its setting and never over
    another film."""
    panels = (
        Panel("graph", "Mass graph", "none"),
        Panel("image", "Corrected image", "none"),
        Panel("total_mass", "Total CO₂ mass", "depth"),
    )
    depth = dict(display_mean="depth", window_px=7)
    tiles = [
        saved_tiles(tmp_path / name, colour, targets, drawn)
        for name, colour, targets, drawn in (
            ("graph", (90, 90, 90), (1.0, 2.0, 3.0), dict(label_room=False)),
            ("image", (40, 80, 120), (1.0, 3.0), {}),
            ("map", (120, 40, 80), (1.0, 2.0, 3.0), dict(depth, window_mm=6.04)),
            ("map10", (80, 120, 40), (1.0, 3.0), dict(depth, window_mm=6.26)),
        )
    ]
    options = dict(tile_settings=SMALL, unit="g/L", style=STYLE)
    film = FilmSettings("film", "pyramid", panels, (640, 360), **VIDEO, note="maps")
    (record,) = write_films({"r1": tiles[:3]}, tmp_path, film=film, **options)

    assert (record["movie"], record["figure"]) == ("film_r1.mp4", "film_r1.png")
    assert record["frames"][0]["note"] == "maps; " + DEPTH_NOTE.format("6.0 mm")
    assert [frame["target_h"] for frame in record["frames"]] == [1.0, 2.0, 3.0]
    assert [panel["held"] for panel in record["frames"][1]["panels"]] == [
        False,
        True,
        False,
    ]
    assert record["frames"][1]["panels"][1]["case_id"] == "c-0"
    assert record["decoded_frames"] == 3
    assert cv2.imread(str(tmp_path / record["figure"])).shape == (360, 640, 3)

    grid = FilmSettings("grid", "grid", panels[2:], (640, 360), **VIDEO, note="maps")
    runs = {"r10": [tiles[3]], "r9": [tiles[2]]}
    (record,) = write_films(runs, tmp_path, film=grid, **options)
    assert list(record["frames"][1]["runs"]) == ["r9", "r10"]
    assert record["frames"][1]["runs"]["r10"]["held"] is True
    assert record["frames"][1]["note"].endswith(DEPTH_NOTE.format("6.0 to 6.3 mm"))
    assert record["decoded_frames"] == 3 and record["movie"] == "grid.mp4"
    with pytest.raises(FileExistsError):
        write_films(runs, tmp_path, film=grid, **options)
