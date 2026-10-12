"""Prepared FF-AC 2x2 screen for optical correction and gas constraints.

The campaign isolates two decisions on five diagnostic runs:

* uniform 24-field ColorChecker normalization versus the same general
  normalization followed by a guarded, leave-one-field-out BTB residual;
* the established generic 24--48 h gas envelope versus run-specific
  overflow-first support with no generic late-gas rejection.

All colour paths and overflow targets are rebuilt below the campaign root.
Nothing below ``Z:/Albus/Results`` is modified. The script is prepared for a
later launch, but importing it or running ``commands`` starts no calibration.
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

from darsia.fluidflower.tools import ac_colorchecker_fieldwise_screen as fieldwise  # noqa: E402
from darsia.fluidflower.tools import ac_colorchecker_refinement_colorpath_pilot as rebuilt  # noqa: E402
from darsia.fluidflower.tools import ac_final_production_calibration as final_campaign  # noqa: E402
from darsia.fluidflower.tools import ac_objective_v2_campaign as objective_v2  # noqa: E402
from darsia.fluidflower.tools import ac_spatial_calibration_campaign as engine  # noqa: E402
from darsia.fluidflower.tools import auto_calibrate_color_to_mass as calibration  # noqa: E402
from darsia.fluidflower.tools import colorchecker_signal_refinement as signal_refinement  # noqa: E402
from darsia.fluidflower.tools.overflow_phase_constraint import (  # noqa: E402
    build_imputed_overflow_phase_targets,
    build_overflow_fraction_reference,
    load_overflow_phase_constraint,
    write_overflow_phase_targets,
)


CAMPAIGN_ID = "overflow_first_colorchecker_float32_localcache_20260828_5000"
DEFAULT_OUTPUT_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
RUNS = ("ac18", "ac52", "ac56", "ac60", "ac61")
MAX_ITERS = 5000
MAX_SWATCH_THRESHOLD = 0.0125
TARGET_MATCH_TOLERANCE_H = 0.12
OVERFLOW_TARGET_TIMES_H = tuple(
    sorted(
        {
            *final_campaign.EVALUATION_TIMES_H[:-1],
            *objective_v2.density.ADDITIONAL_INJECTION_TIMES_H,
            9.5,
            12.0,
        }
    )
)
_BASE_SPEC = objective_v2.SEARCH_SPECS[0]
_ORIGINAL_CONFIGURE_REFINEMENT = (
    calibration._configure_runtime_colorchecker_refinement
)


@dataclass(frozen=True)
class OpticalProfile:
    name: str
    use_signal_residual: bool
    rationale: str


OPTICAL_PROFILES = (
    OpticalProfile(
        name="uniform24",
        use_signal_residual=False,
        rationale=(
            "General per-image correction: six neutral fields determine tone "
            "response and all 24 fields have equal influence on the affine RGB map."
        ),
    ),
    OpticalProfile(
        name="general24_btb_residual_loo",
        use_signal_residual=True,
        rationale=(
            "The same equal-weight 24-field base correction followed by a "
            "bounded BTB-coordinate residual. The residual is accepted only "
            "when leave-one-field-out signal error improves without RGB regression."
        ),
    ),
)


def _switch_enabled(name: str) -> bool:
    value = os.environ.get(name, "off").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{name} must be on or off; got {value!r}")
    return value in {"1", "true", "yes", "on"}


def _signal_settings_from_environment() -> dict[str, float]:
    names = {
        "signal_refinement_ridge": "FFAC_COLORCHECKER_SIGNAL_RIDGE",
        "signal_refinement_huber_delta": "FFAC_COLORCHECKER_SIGNAL_HUBER_DELTA",
        "signal_minimum_improvement": "FFAC_COLORCHECKER_SIGNAL_MIN_IMPROVEMENT",
        "signal_minimum_loo_improvement": (
            "FFAC_COLORCHECKER_SIGNAL_MIN_LOO_IMPROVEMENT"
        ),
        "signal_btb_swatch_weight": "FFAC_COLORCHECKER_SIGNAL_BTB_WEIGHT",
    }
    return {
        setting: float(os.environ[environment])
        for setting, environment in names.items()
        if environment in os.environ
    }


def _configure_runtime_refinement(fluidflower: Any) -> dict[str, Any]:
    provenance = _ORIGINAL_CONFIGURE_REFINEMENT(fluidflower)
    if not _switch_enabled("FFAC_COLORCHECKER_SIGNAL_REFINEMENT"):
        return provenance
    if not provenance.get("active"):
        raise RuntimeError("BTB residual requires the general 24-field correction.")
    settings = _signal_settings_from_environment()
    configured: list[dict[str, Any]] = []
    for index, correction in enumerate(
        getattr(fluidflower, "color_corrections", None) or []
    ):
        if not getattr(correction, "active", False):
            continue
        configured.append(
            {
                "index": int(index),
                **signal_refinement.install(correction, settings=settings),
            }
        )
    if not configured:
        raise RuntimeError("BTB residual found no active ColorChecker correction.")
    refined_baseline = fluidflower.shape_corrected_baseline.copy()
    for correction in fluidflower.color_corrections:
        refined_baseline = correction(refined_baseline)
    fluidflower.baseline = refined_baseline
    return {
        **provenance,
        "signal_refinement": {
            "active": True,
            "baseline_rebuilt_with_complete_pipeline": True,
            "corrections": configured,
        },
    }


def install_runtime_hooks() -> None:
    """Use exactly the same optical pipeline for paths and trial images."""

    calibration._configure_runtime_colorchecker_refinement = (
        _configure_runtime_refinement
    )
    rebuilt._configure_runtime_colorchecker_refinement = (
        _configure_runtime_refinement
    )


def _profile_environment(profile: OpticalProfile) -> dict[str, str]:
    base_profile = fieldwise.OpticalProfile(
        name=profile.name,
        btb_swatch_weight=1.0,
        max_swatch_loss_weight=0.0,
        rationale=profile.rationale,
    )
    environment = fieldwise._profile_environment(base_profile)
    environment.update(
        {
            "FFAC_COLORCHECKER_SIGNAL_REFINEMENT": (
                "on" if profile.use_signal_residual else "off"
            ),
            "FFAC_COLORCHECKER_SIGNAL_RIDGE": "0.05",
            "FFAC_COLORCHECKER_SIGNAL_HUBER_DELTA": "0.02",
            "FFAC_COLORCHECKER_SIGNAL_MIN_IMPROVEMENT": "0.01",
            "FFAC_COLORCHECKER_SIGNAL_MIN_LOO_IMPROVEMENT": "0.02",
            "FFAC_COLORCHECKER_SIGNAL_BTB_WEIGHT": "3.0",
        }
    )
    return environment


def embedding_root(output_root: Path, profile: str, run: str) -> Path:
    return fieldwise.embedding_root(output_root, profile, run)


def overlay_template(output_root: Path, profile: str) -> str:
    return fieldwise.overlay_template(output_root, profile)


def _manifest_path(output_root: Path, profile: str, run: str) -> Path:
    return output_root / "color_path_preflight" / profile / f"{run}.json"


def _prepare_one_embedding(
    output_root: Path,
    profile: OpticalProfile,
    run: str,
    *,
    force: bool,
) -> dict[str, Any]:
    target = embedding_root(output_root, profile.name, run)
    manifest_path = _manifest_path(output_root, profile.name, run)
    if rebuilt._embedding_ready(target) and manifest_path.is_file() and not force:
        # Refresh the tiny overlay even when the expensive embedding is reused.
        # This keeps path-format fixes independent of embedding regeneration.
        fieldwise.write_overlay(output_root, profile.name, run)
        return json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if target.exists() and any(target.iterdir()):
        if not force:
            raise RuntimeError(
                "Optical embedding is partial or unverified; inspect it and "
                f"rerun with --force: {target}"
            )
        archive = target.with_name(
            target.name + ".archived_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.move(str(target), str(archive))

    paths = fieldwise._config_paths(output_root, profile.name, run)
    environment = _profile_environment(profile)
    with rebuilt._temporary_environment(environment):
        rebuilt.calibration_color_paths_and_mass(
            rebuilt.RefinedColorPathRig,
            paths,
            reset=True,
            show=False,
            default=True,
        )
    if not rebuilt._embedding_ready(target):
        raise RuntimeError(f"Optical embedding did not complete: {profile.name}/{run}")
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
        "overlay": str(
            fieldwise.overlay_path(output_root, profile.name, run).resolve()
        ),
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


def prepare_color_paths(output_root: Path, *, force: bool = False) -> dict[str, Any]:
    rows = [
        _prepare_one_embedding(output_root, profile, run, force=force)
        for profile in OPTICAL_PROFILES
        for run in RUNS
    ]
    payload = {
        "schema": 1,
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "created_utc": rebuilt._utc_now(),
        "profiles": [profile.name for profile in OPTICAL_PROFILES],
        "runs": list(RUNS),
        "embeddings": rows,
    }
    rebuilt._atomic_json(output_root / "color_path_preflight.json", payload)
    return payload


def _measured_overflow_payload(run: str) -> dict[str, Any]:
    """Include the final measured scale time as the post-coverage anchor."""

    payload = final_campaign._build_measured_overflow_payload(
        run,
        evaluation_times_h=OVERFLOW_TARGET_TIMES_H,
    )
    coverage_end_h = float(
        payload.get("derivation", {}).get("phase_balance_coverage_end_h", 0.0)
        or 0.0
    )
    if coverage_end_h > 0.0 and not any(
        abs(coverage_end_h - value) <= 1e-6
        for value in OVERFLOW_TARGET_TIMES_H
    ):
        payload = final_campaign._build_measured_overflow_payload(
            run,
            evaluation_times_h=(*OVERFLOW_TARGET_TIMES_H, coverage_end_h),
        )
    return payload


def _overflow_payload(run: str) -> tuple[dict[str, Any], str, float, list[str]]:
    if run not in final_campaign.FULLY_IMPUTED_OVERFLOW_RUNS:
        payload = _measured_overflow_payload(run)
        return payload, "measured-run-specific", final_campaign.DIRECT_OVERFLOW_WEIGHT, [run]

    donors = final_campaign.OVERFLOW_COHORT_DONORS[run]
    donor_payloads = {
        donor: _measured_overflow_payload(donor)
        for donor in donors
    }
    reference = build_overflow_fraction_reference(
        donor_payloads,
        minimum_donors=final_campaign.OVERFLOW_REFERENCE_MIN_DONORS,
        relative_sigma_floor=final_campaign.OVERFLOW_REFERENCE_SIGMA_FLOOR,
    )
    payload = build_imputed_overflow_phase_targets(
        run=run,
        injection_protocol_path=(
            project_path('protocols') / run / "injection_protocol.csv"
        ),
        reference=reference,
        reason=final_campaign._overflow_fallback_reason(run),
        constraint_mode="gas-upper-bound",
    )
    return (
        payload,
        "imputed-cohort-reference",
        final_campaign.IMPUTED_OVERFLOW_WEIGHT,
        list(donors),
    )


def prepare_overflow_targets(
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    target_root = output_root / "overflow_targets"
    manifest_path = output_root / "overflow_target_preflight.json"
    expected_paths = [target_root / f"{run}.json" for run in RUNS]
    if manifest_path.is_file() and all(path.is_file() for path in expected_paths):
        existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not force:
            expected_hashes = {
                str(row["run"]): str(row["sha256"])
                for row in existing.get("rows") or []
            }
            actual_hashes = {
                run: load_overflow_phase_constraint(
                    target_root / f"{run}.json", run=run
                ).sha256
                for run in RUNS
            }
            if expected_hashes != actual_hashes:
                raise RuntimeError(
                    "Existing overflow targets differ from their frozen preflight. "
                    "Inspect the campaign root and use --force to archive/rebuild."
                )
            return existing
    elif not force and (
        manifest_path.exists() or any(path.exists() for path in expected_paths)
    ):
        raise RuntimeError(
            "Overflow target preparation is partial. Inspect the campaign root "
            "and use --force to archive/rebuild it."
        )
    if force and (manifest_path.exists() or target_root.exists()):
        archive_root = output_root / (
            "overflow_targets.archived_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        archive_root.mkdir(parents=True, exist_ok=False)
        if target_root.exists():
            shutil.move(str(target_root), str(archive_root / "overflow_targets"))
        if manifest_path.exists():
            shutil.move(
                str(manifest_path),
                str(archive_root / manifest_path.name),
            )
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        payload, assignment, weight, donors = _overflow_payload(run)
        payload["constraint_mode"] = "gas-upper-bound"
        payload["match_tolerance_h"] = TARGET_MATCH_TOLERANCE_H
        source = payload.setdefault("source", {})
        source["assignment_mode"] = assignment
        source["campaign_id"] = CAMPAIGN_ID
        path = write_overflow_phase_targets(target_root / f"{run}.json", payload)
        constraint = load_overflow_phase_constraint(path, run=run)
        rows.append(
            {
                "run": run,
                "assignment_mode": assignment,
                "penalty_weight": weight,
                "donor_runs": donors,
                "target_count": len(constraint.targets),
                "target_times_h": [target.time_h for target in constraint.targets],
                "constraint_mode": constraint.constraint_mode,
                "match_tolerance_h": constraint.match_tolerance_h,
                "path": str(path.resolve()),
                "sha256": constraint.sha256,
            }
        )
    payload = {
        "schema": 1,
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "created_utc": rebuilt._utc_now(),
        "contract": (
            "Run-specific measured points are hard gas upper bounds; imputed "
            "points are uncertainty-aware soft upper priors."
        ),
        "rows": rows,
    }
    rebuilt._atomic_json(manifest_path, payload)
    return payload


def _run_environment_overrides() -> tuple[
    tuple[str, tuple[tuple[str, str], ...]], ...
]:
    rows = []
    for run in RUNS:
        measured = run not in final_campaign.FULLY_IMPUTED_OVERFLOW_RUNS
        rows.append(
            (
                run,
                (
                    (
                        "FFAC_OVERFLOW_PHASE_PENALTY_WEIGHT",
                        str(
                            final_campaign.DIRECT_OVERFLOW_WEIGHT
                            if measured
                            else final_campaign.IMPUTED_OVERFLOW_WEIGHT
                        ),
                    ),
                    (
                        "FFAC_OVERFLOW_ASSIGNMENT_MODE",
                        "measured-run-specific"
                        if measured
                        else "imputed-cohort-reference",
                    ),
                ),
            )
        )
    return tuple(rows)


def _spec_environment(
    output_root: Path,
    profile: OpticalProfile,
    *,
    overflow_first: bool,
) -> tuple[tuple[str, str], ...]:
    environment = _profile_environment(profile)
    environment.update(
        {
            "FFAC_COLOR_PATH_CONFIG_OVERLAY": overlay_template(
                output_root, profile.name
            ),
            "FFAC_FLAGGED_FRAME_POLICY": "nearest-good",
            "FFAC_FLAGGED_FRAME_WINDOW_FRAMES": "6",
            "FFAC_FLAGGED_FRAME_MAX_MINUTES": "30",
            "FFAC_COLORCHECKER_MAX_SWATCH_WARN_THRESHOLD": str(
                MAX_SWATCH_THRESHOLD
            ),
            "FFAC_OVERFLOW_PHASE_TARGETS": str(
                output_root / "overflow_targets" / "{run}.json"
            ),
            "FFAC_OVERFLOW_POST_COVERAGE_PENALTY_WEIGHT": (
                "inherit" if overflow_first else "0"
            ),
            "FFAC_OVERFLOW_POST_COVERAGE_SIGMA_MULTIPLIER": "2.0",
            "FFAC_OVERFLOW_POST_COVERAGE_FRACTION_TOLERANCE": "0.02",
        }
    )
    if overflow_first:
        environment["FFAC_LATE_GAS_PENALTY_WEIGHT"] = "0"
    else:
        environment.update(
            {
                "FFAC_LATE_GAS_START_H": "24",
                "FFAC_LATE_GAS_MAX_FRACTION": "0.05",
                "FFAC_LATE_GAS_DECAY_HALF_LIFE_H": "12",
                "FFAC_LATE_GAS_FLOOR_FRACTION": "0",
                "FFAC_LATE_GAS_PENALTY_WEIGHT": "200",
                "FFAC_LATE_GAS_CONSTRAINT_MODE": "gas-only",
            }
        )
    return tuple(environment.items())


def _search_specs(output_root: Path) -> tuple[engine.SearchSpec, ...]:
    specs: list[engine.SearchSpec] = []
    run_overrides = _run_environment_overrides()
    for profile in OPTICAL_PROFILES:
        for overflow_first in (False, True):
            gas_name = "overflow_first" if overflow_first else "hard_gas24"
            name = f"{profile.name}_{gas_name}"
            specs.append(
                replace(
                    _BASE_SPEC,
                    name=name,
                    variant=replace(_BASE_SPEC.variant, name=name),
                    objective_policy=(
                        "aligned-physical-v2-overflow-first"
                        if overflow_first
                        else "aligned-physical-v2"
                    ),
                    seed_group="overflow_first_colorchecker_2x2",
                    preprocessing_cache_group=profile.name,
                    environment_overrides=_spec_environment(
                        output_root,
                        profile,
                        overflow_first=overflow_first,
                    ),
                    run_environment_overrides=run_overrides,
                    rationale=(
                        profile.rationale
                        + " "
                        + (
                            "Run-specific overflow upper bounds dominate phase "
                            "support; after coverage a soft no-new-large-gas "
                            "term replaces the generic late-gas envelope."
                            if overflow_first
                            else "Control retaining the established 24 h, 5% "
                            "decaying late-gas envelope."
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


def _inputs_ready(output_root: Path) -> list[str]:
    missing = [
        f"colour:{profile.name}/{run}"
        for profile in OPTICAL_PROFILES
        for run in RUNS
        if not rebuilt._embedding_ready(
            embedding_root(output_root, profile.name, run)
        )
    ]
    missing.extend(
        f"overflow:{run}"
        for run in RUNS
        if not (output_root / "overflow_targets" / f"{run}.json").is_file()
    )
    return missing


def main(argv: Sequence[str] | None = None) -> None:
    install_runtime_hooks()
    arguments = list(sys.argv[1:] if argv is None else argv)
    output_root = _extract_output_root(arguments).resolve()
    if arguments and arguments[0] == "prepare-inputs":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("action", choices=("prepare-inputs",))
        parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        parser.add_argument("--force", action="store_true")
        parsed = parser.parse_args(arguments)
        payload = {
            "overflow": prepare_overflow_targets(
                parsed.output_root.resolve(), force=parsed.force
            ),
            "color_paths": prepare_color_paths(
                parsed.output_root.resolve(), force=parsed.force
            ),
        }
        print(json.dumps(payload, indent=2))
        return

    action = arguments[0] if arguments else ""
    if action in {"prepare", "launch"}:
        prepare_overflow_targets(output_root, force=False)
        prepare_color_paths(output_root, force=False)
    elif action == "worker":
        missing = _inputs_ready(output_root)
        if missing:
            raise SystemExit(
                "Run prepare-inputs before starting workers; missing: "
                + ", ".join(missing)
            )
    configure_engine(output_root)
    if "--tracks" not in arguments:
        arguments.extend(("--tracks", "optimize"))
    engine.main(arguments)


if __name__ == "__main__":
    main()
