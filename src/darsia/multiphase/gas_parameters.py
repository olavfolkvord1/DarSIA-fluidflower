"""Gas geometry at the physical scale of a run's measured cell depth.

Only the front glass is observed. Features finer than one cell depth cannot
locate the gas across that depth. These dimensionless ratios define the method;
they are not a particular rig's lengths or user configuration defaults.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

CLOSING_DEPTHS = 2.0
ROUNDING_DEPTHS = FADE_DEPTHS = SCORE_DEPTHS = LINK_WIDTH_DEPTHS = 1.0
PATH_SMOOTHING_DEPTHS = 0.5
TRACE_SMOOTHING_DEPTHS = OUTLIER_DILATION_DEPTHS = OUTLIER_ROUNDING_DEPTHS = 1 / 3
OUTLIER_CLOSING_DEPTHS = 1.0


@dataclass(frozen=True)
class GasAllocationSettings:
    """The unchanged seed and peeling controls; geometry comes from rig depth."""

    seed_mm2: float
    persistent_seed_mm2: float
    seed_window_h: float
    fallback_seed_mm2: float
    lattice_mm: float
    cost_up_per_mm: float
    cost_side_per_mm: float
    cost_down_per_mm: float
    no_growth_after_injection: bool = field(default=False, kw_only=True)
    """Whether a frame at which no port injects (between injections and after
    the last) holds at most the free gas the run's frame before it placed:
    without a source, free gas only dissolves, so gone gas stays gone."""


@dataclass(frozen=True)
class GasParameters(GasAllocationSettings):
    """Internal resolved parameters, never a series configuration section."""

    depth_mm: float
    score_window_mm: float
    closing_length_mm: float
    fade_length_mm: float
    contour_smoothing_mm: tuple[float, float]
    contour_blend_mm: tuple[float, float, float]
    link_smoothing_mm: tuple[float, float]
    link_width_mm: float

    def __post_init__(self):
        positive = (
            self.depth_mm,
            self.score_window_mm,
            self.fade_length_mm,
            self.link_width_mm,
        )
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Gas depth and lengths must be finite and positive")
        if not np.isfinite(self.closing_length_mm) or self.closing_length_mm < 0:
            raise ValueError("Closing radius must be finite and nonnegative")
        for values, length in (
            (self.contour_smoothing_mm, 2),
            (self.contour_blend_mm, 3),
            (self.link_smoothing_mm, 2),
        ):
            if len(values) != length or any(
                not np.isfinite(v) or v < 0 for v in values
            ):
                raise ValueError("Geometry needs finite nonnegative lengths")

    def geometry_record(self):
        """Physical lengths, including actual Gaussian sigma rather than a window."""
        return {
            "depth_mm": self.depth_mm,
            "closing_radius_mm": self.closing_length_mm,
            "pre_rounding_margin_mm": self.contour_smoothing_mm[0],
            "contour_sigma_mm": self.contour_smoothing_mm[1],
            "fade_length_mm": self.fade_length_mm,
            "probability_sigma_mm": self.score_window_mm / np.sqrt(12),
            "link_width_mm": self.link_width_mm,
            "trace_sigma_mm": self.link_smoothing_mm[0],
            "path_sigma_mm": self.link_smoothing_mm[1],
            "outlier_dilation_mm": self.contour_blend_mm[0],
            "outlier_closing_radius_mm": self.contour_blend_mm[1],
            "outlier_sigma_mm": self.contour_blend_mm[2],
        }


def parameters_at_depth(settings, depth_mm):
    """Resolve one physical scale; also used with an upper depth bound for memory."""
    d = float(depth_mm)
    return GasParameters(
        **asdict(settings),
        depth_mm=d,
        score_window_mm=d * SCORE_DEPTHS * np.sqrt(12),
        closing_length_mm=d * CLOSING_DEPTHS,
        fade_length_mm=d * FADE_DEPTHS,
        contour_smoothing_mm=(0.0, d * ROUNDING_DEPTHS),
        contour_blend_mm=(
            d * OUTLIER_DILATION_DEPTHS,
            d * OUTLIER_CLOSING_DEPTHS,
            d * OUTLIER_ROUNDING_DEPTHS,
        ),
        link_smoothing_mm=(d * TRACE_SMOOTHING_DEPTHS, d * PATH_SMOOTHING_DEPTHS),
        link_width_mm=d * LINK_WIDTH_DEPTHS,
    )


def gas_parameters(settings, depth_m):
    """Use the median finite positive depth of this run's own rig grid.

    Private resolved parameters are accepted for isolated operator experiments.
    Public workflows construct only GasAllocationSettings from configuration.
    """
    if isinstance(settings, GasParameters):
        return settings
    if depth_m is None:
        raise ValueError("Gas geometry requires the measured rig depth map")
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("Gas geometry requires a two-dimensional rig depth map")
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if not valid.size:
        raise ValueError("Gas geometry requires finite positive measured rig depth")
    return parameters_at_depth(settings, float(np.median(valid)) * 1000)
