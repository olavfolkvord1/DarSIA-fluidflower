"""Compatibility wrappers for DarSIA preset calibration workflows.

DarSIA 2026.08 moved calibration entry points to a shared ``AnalysisContext``.
Keep FF-AC campaign code independent of that API transition while preserving
the exact DarSIA implementation selected by the active Python environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from darsia.presets.workflows.analysis.analysis_context import (
    prepare_analysis_context,
    select_image_paths,
)
try:
    from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
except ImportError:
    from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig

try:
    from darsia.presets.workflows.calibration.calibration_color_paths import (
        calibration_color_paths as _legacy_calibration_color_paths,
    )
except ImportError:
    _legacy_calibration_color_paths = None
    from darsia.presets.workflows.calibration.calibration_color_paths import (
        calibration_color_paths_from_context,
    )

try:
    from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
        calibration_color_to_mass_analysis as _legacy_calibration_color_to_mass,
    )
except ImportError:
    _legacy_calibration_color_to_mass = None
    from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
        calibration_color_to_mass_analysis_from_context,
    )


def _calibration_context(
    cls: type[Rig],
    path: Path | list[Path],
    *,
    stage: str,
):
    config = FluidFlowerConfig(path, require_results=False, require_data=True)
    if config.calibration is None:
        raise ValueError("DarSIA configuration has no [calibration] section.")
    if stage == "color":
        if config.calibration.color is None or config.calibration.color.color is None:
            raise ValueError("DarSIA configuration has no color calibration.")
        sub_config = config.calibration.color.color
    elif stage == "mass":
        if config.calibration.mass is None:
            raise ValueError("DarSIA configuration has no mass calibration.")
        sub_config = config.calibration.mass
    else:
        raise ValueError(f"Unsupported DarSIA calibration stage: {stage}")
    return prepare_analysis_context(
        cls=cls,
        path=path,
        all=False,
        require_color_to_mass=False,
        section="calibration",
        require_results=False,
        sub_config=sub_config,
    )


def calibration_color_paths(
    cls: type[Rig], path: Path | list[Path], show: bool = False
) -> None:
    """Run colour-path calibration across supported DarSIA workflow APIs."""

    if _legacy_calibration_color_paths is not None:
        _legacy_calibration_color_paths(cls, path, show=show)
        return
    calibration_color_paths_from_context(
        _calibration_context(cls, path, stage="color"), show=show
    )


def calibration_color_to_mass_analysis(
    cls: type[Rig],
    path: Path | list[Path],
    *,
    ref_path: Path | None = None,
    reset: bool = True,
    show: bool = False,
    default: bool = False,
    **kwargs: Any,
) -> None:
    """Run mass calibration across supported DarSIA workflow APIs."""

    if _legacy_calibration_color_to_mass is not None:
        _legacy_calibration_color_to_mass(
            cls,
            path,
            ref_path=ref_path,
            reset=reset,
            show=show,
            default=default,
            **kwargs,
        )
        return
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "The active DarSIA context API does not accept legacy options: "
            + unexpected
        )
    calibration_color_to_mass_analysis_from_context(
        _calibration_context(cls, path, stage="mass"),
        ref_path=ref_path,
        reset=reset,
        show=show,
        default=default,
    )


def calibration_color_paths_and_mass(
    cls: type[Rig],
    path: Path | list[Path],
    *,
    ref_path: Path | None = None,
    reset: bool = True,
    show: bool = False,
    default: bool = False,
) -> None:
    """Run both stages while sharing the heavy context on modern DarSIA."""

    if (
        _legacy_calibration_color_paths is not None
        and _legacy_calibration_color_to_mass is not None
    ):
        calibration_color_paths(cls, path, show=show)
        calibration_color_to_mass_analysis(
            cls,
            path,
            ref_path=ref_path,
            reset=reset,
            show=show,
            default=default,
        )
        return

    ctx = _calibration_context(cls, path, stage="color")
    calibration_color_paths_from_context(ctx, show=show)
    if ctx.config.calibration is None or ctx.config.calibration.mass is None:
        raise ValueError("DarSIA configuration has no mass calibration.")
    ctx.image_paths = select_image_paths(
        ctx.config,
        ctx.experiment,
        all=False,
        sub_config=ctx.config.calibration.mass,
        data_registry=ctx.config.data.registry,
    )
    calibration_color_to_mass_analysis_from_context(
        ctx,
        ref_path=ref_path,
        reset=reset,
        show=show,
        default=default,
    )
