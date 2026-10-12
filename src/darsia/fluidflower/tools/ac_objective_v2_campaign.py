"""Targeted Objective V2 calibration pilot.

This campaign keeps the selected Titration mass model and 25-frame schedule,
but changes the search contract: injection-sensitive temporal error, explicit
physical penalties, constrained TPE, RGB-supported spatial evidence, gain
regularization, staged gain release, validation-based finalist selection, and
an untouched audit schedule.
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

from darsia.fluidflower.tools import ac_final_method_selection as selection  # noqa: E402
from darsia.fluidflower.tools import ac_final_production_calibration as final_campaign  # noqa: E402
from darsia.fluidflower.tools import ac_injection_frame_density_pilot as density  # noqa: E402
from darsia.fluidflower.tools import ac_spatial_calibration_campaign as engine  # noqa: E402


CAMPAIGN_ID = "objective_v2_pilot_20260822_5000"
DEFAULT_OUTPUT_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
LATEST_SEED_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'expanded25_aligned_restored_20260820_5000'
RUNS = ("ac18", "ac22", "ac24", "ac27", "ac44", "ac52", "ac60", "ac61")
MAX_ITERS = 5000
RANDOM_WARMUPS = 150
HISTORICAL_SEED_COUNT = 16
# The expanded training schedule already contains 1.6 h and 2.2 h. Keep both
# validation partitions image-disjoint from the 25 calibration frames.
VALIDATION_TIMES_H = tuple(
    value for value in selection.HOLDOUT_TIMES_H[::2] if value != 2.2
)
AUDIT_TIMES_H = tuple(
    value for value in selection.HOLDOUT_TIMES_H[1::2] if value != 1.6
)


SEARCH_SPECS = (
    engine.SearchSpec(
        name="expanded25_physical_v2",
        variant=replace(
            final_campaign.VARIANT,
            name="expanded25_physical_v2",
        ),
        gain_weight=0.0,
        color_path_evidence="diagnostic",
        objective_policy="aligned-physical-v2",
        spatial_objective_weight=0.0,
        seed_group="expanded25_physical_v2",
        historical_seed_count=HISTORICAL_SEED_COUNT,
        objective_integral="window-balanced-scaled",
        additional_calibration_times_h=density.ADDITIONAL_INJECTION_TIMES_H,
        finalist_count=12,
        stage1_gain_fraction=0.25,
        seed_campaign_roots=(str(LATEST_SEED_ROOT),),
        validation_times_h=VALIDATION_TIMES_H,
        audit_times_h=AUDIT_TIMES_H,
        rationale=(
            "Objective V2 pilot: time-relative mass fit, full physical penalty "
            "alignment, constrained broad TPE, full-RGB spatial support, weak "
            "layer-gain prior, staged gain release, validation ranking, and an "
            "independent audit schedule."
        ),
    ),
)


def configure_engine() -> None:
    engine.CAMPAIGN_ID = CAMPAIGN_ID
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.PILOT_RUNS = RUNS
    engine.RUNS = RUNS
    engine.ALL_RUNS = RUNS
    engine.MAX_ITERS = MAX_ITERS
    engine.RANDOM_WARMUPS = RANDOM_WARMUPS
    engine.SPATIAL_OBJECTIVE_WEIGHT = 0.0
    engine.SPATIAL_TRACE_LEVEL = 0.01
    engine.SPATIAL_CORE_LEVEL = 0.02
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
