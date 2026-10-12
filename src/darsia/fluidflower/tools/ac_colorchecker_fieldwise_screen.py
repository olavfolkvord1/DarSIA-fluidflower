"""Factorial ColorChecker screen with run-native rebuilt colour paths.

The screen separates two questions on AC18 and AC52:

1. Does BTB-sensitive field weighting improve the optical correction?
2. Does rejecting a frame with one badly corrected checker field remove the
   transient mass discontinuity without changing the calibration schedule?

Every optical profile gets an isolated, run-native colour-path embedding.
Production calibration artefacts below ``Z:/Albus/Results`` are never touched.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools import ac_colorchecker_refinement_colorpath_pilot as rebuilt  # noqa: E402
from darsia.fluidflower.tools import ac_colorchecker_refinement_pilot as first_pilot  # noqa: E402
from darsia.fluidflower.tools import ac_objective_v2_campaign as objective_v2  # noqa: E402
from darsia.fluidflower.tools import ac_spatial_calibration_campaign as engine  # noqa: E402


CAMPAIGN_ID = "colorchecker_fieldwise_screen_20260827_5000"
DEFAULT_OUTPUT_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
RUNS = ("ac18", "ac52")
MAX_ITERS = 5000
MAX_SWATCH_THRESHOLD = 0.0125
_base_spec = objective_v2.SEARCH_SPECS[0]


@dataclass(frozen=True)
class OpticalProfile:
    name: str
    btb_swatch_weight: float
    max_swatch_loss_weight: float
    rationale: str


OPTICAL_PROFILES = (
    OpticalProfile(
        name="uniform24",
        btb_swatch_weight=1.0,
        max_swatch_loss_weight=0.0,
        rationale="All 24 checker fields have equal influence, matching V1.",
    ),
    OpticalProfile(
        name="btb_fieldwise",
        btb_swatch_weight=3.0,
        max_swatch_loss_weight=0.30,
        rationale=(
            "Blue/cyan/green/yellow fields receive threefold weight and the "
            "worst individual field contributes to transform acceptance."
        ),
    ),
)


def _profile_environment(profile: OpticalProfile) -> dict[str, str]:
    environment = dict(first_pilot.COLORCHECKER_ENVIRONMENT)
    environment.update(
        {
            "FFAC_COLORCHECKER_BTB_SWATCH_WEIGHT": str(
                profile.btb_swatch_weight
            ),
            "FFAC_COLORCHECKER_MAX_SWATCH_LOSS_WEIGHT": str(
                profile.max_swatch_loss_weight
            ),
            # Embedding generation applies the correction but does not select
            # calibration frames, so this threshold is deliberately inert here.
            "FFAC_COLORCHECKER_MAX_SWATCH_WARN_THRESHOLD": "1.0",
        }
    )
    return environment


def embedding_root(output_root: Path, profile: str, run: str) -> Path:
    return output_root / "run_native_color_embedding" / profile / run


def overlay_path(output_root: Path, profile: str, run: str) -> Path:
    return output_root / "config" / f"{run}_{profile}_color_path_overlay.toml"


def overlay_template(output_root: Path, profile: str) -> str:
    return str(
        output_root
        / "config"
        / f"{{run}}_{profile}_color_path_overlay.toml"
    )


def _toml_filesystem_path(path: Path | str) -> str:
    """Return a TOML-safe path which also bypasses Windows MAX_PATH."""

    raw = str(path)
    if raw.startswith("\\\\?\\UNC\\"):
        return "//?/UNC/" + raw[8:].replace("\\", "/")
    if raw.startswith("\\\\"):
        return "//?/UNC/" + raw[2:].replace("\\", "/")
    return Path(raw).as_posix()


def write_overlay(output_root: Path, profile: str, run: str) -> Path:
    destination = overlay_path(output_root, profile, run).resolve()
    calibration_folder = _toml_filesystem_path(
        embedding_root(output_root, profile, run).resolve()
    )
    rebuilt._atomic_text(
        destination,
        "[color.path.relative_colorpath_labels]\n"
        f'calibration_folder = "{calibration_folder}"\n',
    )
    return destination


def _config_paths(output_root: Path, profile: str, run: str) -> list[Path]:
    config_dir = project_path('config_seg6/run_ac')
    paths = [
        config_dir.parent / "common.toml",
        config_dir / f"{run}.toml",
        config_dir.parent / "labels_basis.toml",
    ]
    stamp = config_dir / ".color_state" / f"{run}.txt"
    if (
        stamp.is_file()
        and stamp.read_text(encoding="utf-8-sig").strip().lower() == "on"
    ):
        paths.append(config_dir.parent / "coloron.toml")
    paths.append(write_overlay(output_root, profile, run))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing colour-path input(s): "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _embedding_manifest_path(
    output_root: Path, profile: str, run: str
) -> Path:
    return output_root / "color_path_preflight" / profile / f"{run}.json"


def _prepare_one_embedding(
    output_root: Path,
    profile: OpticalProfile,
    run: str,
    *,
    force: bool,
) -> dict[str, Any]:
    target = embedding_root(output_root, profile.name, run)
    manifest_path = _embedding_manifest_path(output_root, profile.name, run)
    if rebuilt._embedding_ready(target) and manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if target.exists() and any(target.iterdir()):
        if not force:
            raise RuntimeError(
                "Isolated colour-path folder is partial or unverified; use "
                f"--force after inspection: {target}"
            )
        archive = target.with_name(
            target.name + ".archived_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.move(str(target), str(archive))

    paths = _config_paths(output_root, profile.name, run)
    environment = _profile_environment(profile)
    with rebuilt._temporary_environment(environment):
        rebuilt.calibration_color_paths(
            rebuilt.RefinedColorPathRig,
            paths,
            show=False,
        )
        rebuilt.calibration_color_to_mass_analysis(
            rebuilt.RefinedColorPathRig,
            paths,
            reset=True,
            show=False,
            default=True,
        )
    if not rebuilt._embedding_ready(target):
        raise RuntimeError(
            f"Colour-path calibration did not complete for {profile.name}/{run}: "
            f"{target}"
        )

    metadata_paths = (
        target / "color_paths" / "from_labels" / "metadata.json",
        target / "color_to_mass" / "from_labels" / "metadata.json",
    )
    payload = {
        "schema": 1,
        "status": "complete",
        "run": run,
        "profile": profile.name,
        "profile_rationale": profile.rationale,
        "created_utc": rebuilt._utc_now(),
        "host": os.environ.get("COMPUTERNAME", "").upper(),
        "overlay": str(overlay_path(output_root, profile.name, run).resolve()),
        "calibration_root": str(target.resolve()),
        "production_results_untouched": True,
        "environment": environment,
        "config_paths": [str(path.resolve()) for path in paths],
        "metadata": [
            {"path": str(path), "sha256": rebuilt._sha256(path)}
            for path in metadata_paths
        ],
        "artifacts": rebuilt._artifact_inventory(target),
    }
    rebuilt._atomic_json(manifest_path, payload)
    return payload


def prepare_color_paths(
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    rows = [
        _prepare_one_embedding(
            output_root,
            profile,
            run,
            force=force,
        )
        for profile in OPTICAL_PROFILES
        for run in RUNS
    ]
    payload = {
        "schema": 1,
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "created_utc": rebuilt._utc_now(),
        "runs": list(RUNS),
        "profiles": [profile.name for profile in OPTICAL_PROFILES],
        "embeddings": rows,
    }
    rebuilt._atomic_json(output_root / "color_path_preflight.json", payload)
    return payload


def _spec_environment(
    output_root: Path,
    profile: OpticalProfile,
    *,
    use_frame_qc: bool,
) -> tuple[tuple[str, str], ...]:
    environment = _profile_environment(profile)
    environment.update(
        {
            "FFAC_COLOR_PATH_CONFIG_OVERLAY": overlay_template(
                output_root, profile.name
            ),
            "FFAC_FLAGGED_FRAME_POLICY": (
                "nearest-good" if use_frame_qc else "exact"
            ),
            "FFAC_FLAGGED_FRAME_WINDOW_FRAMES": "6",
            "FFAC_FLAGGED_FRAME_MAX_MINUTES": "30",
            "FFAC_COLORCHECKER_MAX_SWATCH_WARN_THRESHOLD": (
                str(MAX_SWATCH_THRESHOLD) if use_frame_qc else "1.0"
            ),
        }
    )
    return tuple(environment.items())


def _search_specs(output_root: Path) -> tuple[engine.SearchSpec, ...]:
    specs: list[engine.SearchSpec] = []
    for profile in OPTICAL_PROFILES:
        for use_frame_qc in (False, True):
            suffix = "nearest_good" if use_frame_qc else "exact"
            name = f"expanded25_physical_v2_{profile.name}_{suffix}"
            specs.append(
                replace(
                    _base_spec,
                    name=name,
                    variant=replace(_base_spec.variant, name=name),
                    seed_group="colorchecker_fieldwise_screen",
                    environment_overrides=_spec_environment(
                        output_root,
                        profile,
                        use_frame_qc=use_frame_qc,
                    ),
                    rationale=(
                        f"{profile.rationale} "
                        + (
                            "A frame is substituted only when its worst checker "
                            f"field exceeds {MAX_SWATCH_THRESHOLD:g}."
                            if use_frame_qc
                            else "Every requested calibration frame is retained."
                        )
                    ),
                )
            )
    return tuple(specs)


SEARCH_SPECS = _search_specs(DEFAULT_OUTPUT_ROOT)


def configure_engine(output_root: Path | None = None) -> None:
    global SEARCH_SPECS

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
    SEARCH_SPECS = _search_specs(resolved_root)
    engine.SEARCH_SPECS = SEARCH_SPECS
    engine.SPECS_BY_NAME = {spec.name: spec for spec in SEARCH_SPECS}


def _extract_output_root(arguments: Sequence[str]) -> Path:
    values = list(arguments)
    if "--output-root" not in values:
        return DEFAULT_OUTPUT_ROOT
    index = values.index("--output-root")
    if index + 1 >= len(values):
        raise SystemExit("--output-root requires a path")
    return Path(values[index + 1])


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    output_root = _extract_output_root(arguments).resolve()
    if arguments and arguments[0] == "prepare-color-paths":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("action", choices=("prepare-color-paths",))
        parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        parser.add_argument("--force", action="store_true")
        parsed = parser.parse_args(arguments)
        print(
            json.dumps(
                prepare_color_paths(parsed.output_root, force=parsed.force),
                indent=2,
            )
        )
        return

    action = arguments[0] if arguments else ""
    if action in {"prepare", "launch"}:
        prepare_color_paths(output_root, force=False)
    elif action == "worker":
        missing = [
            f"{profile.name}/{run}"
            for profile in OPTICAL_PROFILES
            for run in RUNS
            if not rebuilt._embedding_ready(
                embedding_root(output_root, profile.name, run)
            )
        ]
        if missing:
            raise SystemExit(
                "Run prepare-color-paths before starting workers; missing: "
                + ", ".join(missing)
            )
    configure_engine(output_root)
    if "--tracks" not in arguments:
        arguments.extend(("--tracks", "optimize"))
    engine.main(arguments)


if __name__ == "__main__":
    main()
