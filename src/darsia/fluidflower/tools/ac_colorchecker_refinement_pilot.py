"""Recalibrate AC52 with robust per-frame ColorChecker normalization.

This pilot keeps the 25-frame Objective V2 calibration contract unchanged and
only replaces the optical preprocessing. The ColorChecker is auto-oriented,
the six neutral patches define a monotone per-channel response curve in linear
RGB, and a small robust affine residual correction is accepted only when it
improves the measured checker residual.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

from dataclasses import replace
from pathlib import Path
import sys
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools import ac_objective_v2_campaign as objective_v2  # noqa: E402
from darsia.fluidflower.tools import ac_spatial_calibration_campaign as engine  # noqa: E402


CAMPAIGN_ID = "colorchecker_refinement_pilot_20260826_5000"
DEFAULT_OUTPUT_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
RUNS = ("ac52",)
MAX_ITERS = 5000

COLORCHECKER_ENVIRONMENT = (
    ("FFAC_COLORCHECKER_REFINEMENT", "on"),
    ("FFAC_COLORCHECKER_AUTO_ORIENT", "on"),
    ("FFAC_COLORCHECKER_STRATEGY", "neutral-tone-affine"),
    ("FFAC_COLORCHECKER_REFINEMENT_STEPS", "2"),
    ("FFAC_COLORCHECKER_REFINEMENT_RIDGE", "0.01"),
    ("FFAC_COLORCHECKER_REFINEMENT_HUBER_DELTA", "0.025"),
    ("FFAC_COLORCHECKER_REFINEMENT_MIN_IMPROVEMENT", "0.005"),
    ("FFAC_COLORCHECKER_REFINEMENT_MAX_CONDITION", "4.0"),
    ("FFAC_COLORCHECKER_REFINEMENT_DIAGONAL_MIN", "0.75"),
    ("FFAC_COLORCHECKER_REFINEMENT_DIAGONAL_MAX", "1.35"),
    ("FFAC_COLORCHECKER_REFINEMENT_OFF_DIAGONAL_MAX_ABS", "0.20"),
    ("FFAC_COLORCHECKER_REFINEMENT_TRANSLATION_MAX_ABS", "0.06"),
    ("FFAC_COLORCHECKER_RESIDUAL_WARN_THRESHOLD", "0.015"),
)

_base_spec = objective_v2.SEARCH_SPECS[0]
SEARCH_SPECS = (
    replace(
        _base_spec,
        name="expanded25_physical_v2_colorchecker",
        variant=replace(
            _base_spec.variant,
            name="expanded25_physical_v2_colorchecker",
        ),
        environment_overrides=COLORCHECKER_ENVIRONMENT,
        rationale=(
            "Objective V2 and its 25-frame schedule with per-frame, run-native "
            "ColorChecker auto-orientation and guarded neutral-tone normalization."
        ),
    ),
)


def configure_engine(output_root: Path | None = None) -> None:
    resolved_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
    objective_v2.configure_engine()
    engine.CAMPAIGN_ID = CAMPAIGN_ID
    engine.DEFAULT_OUTPUT_ROOT = resolved_root
    engine.PILOT_RUNS = RUNS
    engine.RUNS = RUNS
    engine.ALL_RUNS = RUNS
    engine.MAX_ITERS = MAX_ITERS
    engine.OPTIMIZE_JOB_ORDER = "run-major"
    engine.WORKER_SCRIPT = Path(__file__).resolve()
    engine.SEARCH_SPECS = SEARCH_SPECS
    engine.SPECS_BY_NAME = {spec.name: spec for spec in SEARCH_SPECS}


def _default_optimize_track(argv: Sequence[str]) -> list[str]:
    resolved = list(argv)
    if "--tracks" not in resolved:
        resolved.extend(("--tracks", "optimize"))
    return resolved


def main(argv: Sequence[str] | None = None) -> None:
    configure_engine()
    arguments = _default_optimize_track(sys.argv[1:] if argv is None else argv)
    engine.main(arguments)


if __name__ == "__main__":
    main()
