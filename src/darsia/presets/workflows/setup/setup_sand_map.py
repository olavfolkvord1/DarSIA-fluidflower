"""Set up the labels of a run from the series' sand-map drawing."""

import logging
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.sections import (
    list_required_sections,
    required_sections,
)
from darsia.presets.workflows.setup.illustrations import save_discrete_map_illustration
from darsia.presets.workflows.setup.sand_map import (
    SandMapDrawing,
    grid_homographies,
    rasterize_sand_map,
    read_moved_anchors,
)

logger = logging.getLogger(__name__)


@required_sections("labeling", "rig", "corrections")
def setup_sand_map(path: Path | list[Path], show: bool = False) -> None:
    """Place the sand-map drawing on the run's grid and save it as the labels.

    The drawing, with the run's adjustment if one is configured, is rasterized on
    the run's grid through the run's rig geometry
    (:func:`rasterize_sand_map`). The run's grid is the target grid of the rig's
    image mapping, ``[corrections.mapping]``, on which the rig expects labels and
    facies; without a rig geometry, the grid's own metric pixel centres place
    the drawing (:func:`rig_grid`). The labels are the layers of the drawing;
    the facies follow from them as for any labels (``setup_facies``).

    Parameters
    ----------
    path : Path | list[Path]
        Path to the configuration file(s), with [labeling.sand_map] and
        [corrections.mapping].
    show : bool
        Whether to show the labels.

    Raises
    ------
    ValueError
        If the rig has no image mapping, or pixels inside the rig's frame lie in
        no layer of the drawing.

    """
    logger.info("\033[92mSetting up the sand map...\033[0m")

    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check(*list_required_sections(setup_sand_map))
    assert config.labeling is not None and config.rig is not None
    sand_map = config.labeling.sand_map
    assert sand_map is not None and config.corrections is not None
    if config.corrections.mapping is None:
        raise ValueError(
            "The sand map is placed on the run grid of the rig's image mapping; "
            "configure [corrections.mapping]."
        )
    mapping = darsia.ImageMappingCorrection()
    mapping.load(config.corrections.mapping.path)
    shape = mapping.target_shape
    grid = darsia.Image(np.zeros(shape, np.uint8), **mapping.correct_metadata())

    drawing = SandMapDrawing.read(sand_map.drawing)
    if sand_map.adjustment:
        drawing = drawing.moved(read_moved_anchors(sand_map.adjustment))
    width, height = config.rig.width, config.rig.height
    _, rig_to_grid = grid_homographies(sand_map.grid_to_rig, grid)
    labels, inside = rasterize_sand_map(
        drawing,
        shape,
        np.asarray(rig_to_grid, dtype=float),
        width=width,
        height=height,
        edge_fill_mm=sand_map.edge_fill_mm,
    )
    uncovered = np.count_nonzero(inside & (labels == 0))
    if uncovered:
        raise ValueError(
            f"{uncovered} pixels inside the rig's frame lie in no layer of the "
            "drawing; extend its polygons to cover the frame."
        )

    image = darsia.Image(labels, **mapping.correct_metadata())
    image.save(config.labeling.labels)
    save_discrete_map_illustration(
        labels,
        config.labeling.labels.with_suffix(".jpg"),
        title="Labels",
        colorbar_label="Label id",
    )
    if show:
        image.show(title="Labels")
