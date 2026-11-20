"""Labeling configuration."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class SandMapConfig:
    """Labels from the series' sand-map drawing, placed on the run's grid.

    The run's grid is the target grid of the rig's image mapping
    (``[corrections.mapping]``), so it is not repeated here.
    """

    drawing: Path = field(
        default_factory=Path,
        metadata={
            "name": "Drawing",
            "help": (
                "Path to the series' sand-map drawing (.json): the layers as "
                "polygons on shared anchors in rig coordinates (metres, x to the "
                "right and y upwards from the rig's lower left inner corner)."
            ),
        },
    )
    """Path to the drawing."""
    adjustment: Path | None = field(
        default=None,
        metadata={
            "name": "Run adjustment",
            "help": (
                "Path to this run's adjustment (.json): the anchors it moves, for a "
                "run whose sand has settled. Leave empty for the common drawing."
            ),
        },
    )
    """Path to the run's moved anchors."""
    grid_to_rig: list[list[float]] | None = field(
        default=None,
        metadata={
            "name": "Rig geometry",
            "help": (
                "The run's measured homography from the grid's pixel coordinates "
                "(column, row; pixel centres at whole numbers) to rig coordinates "
                "(metres), as three rows of three numbers; the sand map is placed "
                "by its inverse. Leave empty for a grid in rig metres, e.g. the "
                "rig mapping's metric grid."
            ),
        },
    )
    """Homography from the grid's pixel coordinates to rig coordinates."""
    edge_fill_mm: float | None = field(
        default=None,
        metadata={
            "name": "Edge fill [mm]",
            "help": (
                "Pixels inside the rig's frame that no polygon covers take the "
                "nearest layer within this distance of the frame's edge; 0 for none."
            ),
            "placeholder": "e.g., 5.0",
        },
    )
    """Distance from the frame's edge within which uncovered pixels are filled."""

    def load(self, sec: dict) -> "SandMapConfig":
        """Load the sand-map configuration from the [labeling.sand_map] section."""
        self.drawing = _get_key(sec, "drawing", required=True, type_=Path)
        self.adjustment = _get_key(sec, "adjustment", required=False, type_=Path)
        if "rig_to_grid" in sec:
            raise ValueError(
                "[labeling.sand_map] rig_to_grid is replaced by grid_to_rig, its "
                "inverse: the run's homography from the grid's pixels to rig metres."
            )
        self.grid_to_rig = _get_key(sec, "grid_to_rig", required=False, type_=list)
        self.edge_fill_mm = _get_key(sec, "edge_fill_mm", required=True, type_=float)
        return self


@dataclass
class LabelingConfig:
    colored_image: Path = field(
        default_factory=Path,
        metadata={
            "name": "Colored image",
            "help": (
                """Path to the manually-colored/segmented reference image """
                """used to seed label detection."""
            ),
            "group": "Input",
        },
    )
    """Path to the segmented file."""
    sand_map: SandMapConfig | None = field(
        default=None,
        metadata={
            "name": "Sand map",
            "help": (
                "Labels from the series' sand-map drawing instead of a colored "
                "image."
            ),
        },
    )
    """Labels from a sand-map drawing (if any), in place of the colored image."""
    rtol: float = field(
        default=0.001,
        metadata={
            "name": "Significance tolerance",
            "help": (
                """Minimum label significance to keep, as a fraction of image area """
                """(default 0.001 = 0.1%)."""
            ),
            "group": "Options",
        },
    )
    """Minimum significance of labels to be kept (default: 0.1% of the image area)."""
    ensure_connectivity: bool = field(
        default=True,
        metadata={
            "name": "Ensure connectivity",
            "help": "Whether detected labels must be spatially connected regions.",
            "group": "Options",
        },
    )
    """Whether to ensure that labels are connected regions (default: True)."""
    unite_labels: list[list[int]] = field(
        default_factory=list,
        metadata={
            "name": "Unite labels",
            "help": (
                """Groups of label IDs to merge together. """
                """Each row is a comma/space-separated list of integers """
                """(e.g., "3, 5, 8" merges labels 3, 5, and 8 into one label). """
                """Leave empty if not needed."""
            ),
            "widget": "int_rows",
            "placeholder": "Comma/space-separated label IDs (e.g., 3, 5, 8)",
            "group": "Options",
        },
    )
    """List of lists of labels to be united."""
    water_label: int | None = field(
        default=None,
        metadata={
            "name": "Water",
            "help": (
                """Label ID identifying the water region (if any); """
                """reserved for future use."""
            ),
            "group": "Custom labels",
        },
    )
    """Label corresponding to water (if any)."""
    colorchecker_label: int | None = field(
        default=None,
        metadata={
            "name": "Colorchecker",
            "help": (
                """Label ID identifying the colorchecker region (if any); """
                """reserved for future use."""
            ),
            "group": "Custom labels",
        },
    )
    """Label corresponding to colorchecker (if any)."""
    labels: Path = field(
        default_factory=Path,
        metadata={"hidden": True},
    )
    """Path to the labels file."""

    def load(self, path: Path, results: Path | None = None) -> "LabelingConfig":
        """Load labeling config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "labeling")
        if "sand_map" in sec:
            self.sand_map = SandMapConfig().load(sec["sand_map"])
        self.colored_image = _get_key(
            sec,
            "colored_image",
            default=self.colored_image,
            required=self.sand_map is None,
            type_=Path,
        )
        self.unite_labels = _get_key(sec, "unite_labels", required=False, type_=list)
        self.water_label = _get_key(sec, "water_label", required=False, type_=int)
        self.colorchecker_label = _get_key(
            sec, "colorchecker_label", required=False, type_=int
        )
        self.labels = _get_key(sec, "labels", required=False, type_=Path)
        if not self.labels:
            assert results is not None
            self.labels = results / "setup" / "labels" / "labels.npz"
        self.rtol = _get_key(
            sec, "rtol", default=self.rtol, required=False, type_=float
        )
        self.ensure_connectivity = _get_key(
            sec,
            "ensure_connectivity",
            default=self.ensure_connectivity,
            required=False,
            type_=bool,
        )
        return self

    def error(self):
        raise ValueError("Use [labeling] in the config file to load labeling.")
