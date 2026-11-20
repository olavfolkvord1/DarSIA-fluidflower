"""The media workflow accounts for optional display arrays without native jobs."""

from dataclasses import replace

from darsia.presets.workflows.media.steps import FILM_TILES


def test_optional_views_reserve_scratch_without_changing_legacy_memory():
    """Admission includes geometry conversion and output-size-dependent overlays."""
    memory = FILM_TILES.memory
    tables = {"tiles": {"width_px": 400}, "photograph": {}}
    arguments = dict(
        itemsize=8,
        camera_pixels=3200,
        grid_pixels=3200,
        frames=79,
        grid_shape=(40, 80),
        tables=tables,
    )
    legacy = memory.bytes(**arguments)
    assert legacy == replace(memory, terms=()).bytes(**arguments)
    tables["tiles"]["mass_quantity"] = "area"
    area = memory.bytes(**arguments)
    assert area > legacy
    tables["photograph"]["co2_outline"] = True
    overlay = memory.bytes(**arguments)
    assert overlay > area
    tables["tiles"]["width_px"] = 800
    assert memory.bytes(**arguments) > overlay


def test_first_frame_and_mass_contours_have_explicit_scratch_reservations():
    """Extra transfer buffers and physical mass-mask conversion are admitted."""
    arguments = dict(
        itemsize=4, camera_pixels=3200, grid_pixels=3200, frames=2, grid_shape=(40, 80)
    )
    tables = {
        "tiles": {"width_px": 20, "grid_divisor": 4, "mass_quantity": "area"},
        "photograph": {"co2_outline": True, "co2_source": "mass"},
    }
    baseline = FILM_TILES.memory.bytes(**arguments, tables=tables)
    tables["photograph"]["transfer"] = "first-frame"
    measured = FILM_TILES.memory.bytes(**arguments, tables=tables)
    assert measured > baseline
