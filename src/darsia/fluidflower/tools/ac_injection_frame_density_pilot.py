"""Compare the current 17-frame calibration with eight extra injection frames.

The two cells use the same final Titration model, physical constraints,
staged spatial Optuna objective, warmups, random seed, and overflow targets.
Only the calibration-image schedule differs. Window-balanced mass fitting
keeps the denser injection window from dominating the other analysis windows.
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

from darsia.fluidflower.tools import ac_final_production_calibration as final_campaign  # noqa: E402
from darsia.fluidflower.tools import ac_spatial_calibration_campaign as engine  # noqa: E402


CAMPAIGN_ID = "injection_frame_density_pilot_20260818_5000"
DEFAULT_OUTPUT_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
RUNS = ("ac17", "ac18", "ac22", "ac52", "ac56", "ac60")
MAX_ITERS = 5000
RANDOM_WARMUPS = 150
SEED_GROUP = "injection_frame_density"

# Shared protocol landmarks: first injector ramp/body/shut-down followed by
# second injector ramp/body/shut-down. The existing schedule contributes two
# active-injection frames in each injection; these add four to each.
ADDITIONAL_INJECTION_TIMES_H = (
    5.0 / 60.0,
    21.0 / 60.0,
    33.0 / 60.0,
    54.0 / 60.0,
    63.0 / 60.0,
    96.0 / 60.0,
    132.0 / 60.0,
    141.0 / 60.0,
)


def _variant(name: str):
    return replace(final_campaign.VARIANT, name=name)


SEARCH_SPECS = (
    engine.SearchSpec(
        name="current17_staged_spatial",
        variant=_variant("current17_staged_spatial"),
        gain_weight=0.0,
        color_path_evidence="off",
        objective_policy="staged-spatial",
        spatial_objective_weight=0.0,
        seed_group=SEED_GROUP,
        objective_integral="window-balanced-scaled",
        rationale=(
            "Control: current 17 calibration frames with window-balanced mass "
            "fitting and staged spatial candidate selection."
        ),
    ),
    engine.SearchSpec(
        name="expanded25_staged_spatial",
        variant=_variant("expanded25_staged_spatial"),
        gain_weight=0.0,
        color_path_evidence="off",
        objective_policy="staged-spatial",
        spatial_objective_weight=0.0,
        seed_group=SEED_GROUP,
        objective_integral="window-balanced-scaled",
        additional_calibration_times_h=ADDITIONAL_INJECTION_TIMES_H,
        rationale=(
            "Treatment: the same method plus eight protocol-targeted active-"
            "injection frames, four in each injection."
        ),
    ),
)


def configure_engine() -> None:
    engine.CAMPAIGN_ID = CAMPAIGN_ID
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.PILOT_RUNS = RUNS
    engine.RUNS = RUNS
    engine.MAX_ITERS = MAX_ITERS
    engine.RANDOM_WARMUPS = RANDOM_WARMUPS
    engine.SPATIAL_OBJECTIVE_WEIGHT = 0.0
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
    arguments = _default_optimize_track(
        sys.argv[1:] if argv is None else argv
    )
    engine.main(arguments)


if __name__ == "__main__":
    main()
