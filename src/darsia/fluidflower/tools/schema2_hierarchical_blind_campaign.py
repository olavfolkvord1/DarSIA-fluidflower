#!/usr/bin/env python3
"""Blind disjoint-run pilot for balanced hierarchical phase models v1.

Candidate pairs differ only by the optional BTB chemistry feature.  Both use
the same corrected RGB, true clear-water reference, causal topology, training
regions, balancing, and two-stage partition:

    P(free gas) = P(visible CO2) * P(free gas | visible CO2)
    P(aqueous CO2) = max(P(visible CO2) - P(free gas), 0)
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.dual_phase_signal import DualPhaseSignalModel  # noqa: E402
from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer  # noqa: E402


CAMPAIGN_ID = "schema2_hierarchical_blind_v1_20260829"
MODEL_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'schema2_crossseries_pilot_review_20260829/validation_v1_balanced_hierarchical/models'
MODEL_ORDER = ("generic_hierarchical", "btb_hierarchical")
GAS_MODEL_PATHS = {
    "generic_hierarchical": MODEL_ROOT / "generic_clearwater" / "gas_within_co2.json",
    "btb_hierarchical": MODEL_ROOT / "btb_clearwater" / "gas_within_co2.json",
}
CO2_MODEL_PATHS = {
    "generic_hierarchical": MODEL_ROOT / "generic_clearwater" / "co2_presence.json",
    "btb_hierarchical": MODEL_ROOT / "btb_clearwater" / "co2_presence.json",
}
RUNS = {"ffac": ("ac52",), "ffum": ("run16",)}
OUTPUTS = {
    "ffac": configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}',
    "ffum": configured_tool_path('ffum', 'analysis_v2') / f'{CAMPAIGN_ID}',
}
BLIND_SEED = "schema2-hierarchical-disjoint-v1-20260829"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_model_pairs() -> tuple[
    dict[str, DualPhaseSignalModel],
    dict[str, DualPhaseSignalModel],
]:
    gas = {name: DualPhaseSignalModel.load(GAS_MODEL_PATHS[name]) for name in MODEL_ORDER}
    co2 = {name: DualPhaseSignalModel.load(CO2_MODEL_PATHS[name]) for name in MODEL_ORDER}
    for name in MODEL_ORDER:
        gas_target = gas[name].training_provenance.get("target_kind")
        co2_target = co2[name].training_provenance.get("target_kind")
        if gas_target != "gas_within_co2" or co2_target != "co2_presence":
            raise RuntimeError(
                f"{name} model targets differ: gas={gas_target}, co2={co2_target}"
            )
        if bool(gas[name].training_provenance.get("deployment_allowed")):
            raise RuntimeError("Pilot gas models must not claim deployment approval")
        if bool(co2[name].training_provenance.get("deployment_allowed")):
            raise RuntimeError("Pilot CO2 models must not claim deployment approval")
    return gas, co2


def _configure_transfer() -> None:
    transfer.CAMPAIGN_ID = CAMPAIGN_ID
    transfer.MODEL_ORDER = MODEL_ORDER
    transfer.MODEL_PATHS = dict(GAS_MODEL_PATHS)
    transfer.BLIND_SEED = BLIND_SEED


def _candidate_jobs(
    dataset: str,
    runs: Sequence[str],
    gas: Mapping[str, DualPhaseSignalModel],
    co2: Mapping[str, DualPhaseSignalModel],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for run in runs:
        for order, name in enumerate(MODEL_ORDER, start=1):
            digest = transfer._composite_candidate_digest(gas[name], co2[name])
            jobs.append(
                {
                    "job_id": f"{dataset}_{run}_{name}",
                    "run": run,
                    "role": name,
                    "family": "phase-transfer-schema2-hierarchical-v1",
                    "variant": f"{order:02d}_{name}_schema2",
                    "grid_order": order,
                    "candidate_id": f"{dataset}__{run}__{name}__{digest[:16]}",
                    "quality_flags": {
                        "dataset": dataset,
                        "external_transfer": dataset == "ffum",
                        "target_used_for_fitting": False,
                        "hierarchical_phase_partition_active": True,
                        "five_class_model_active": False,
                    },
                }
            )
    return jobs


def _manifest(
    dataset: str,
    root: Path,
    runs: Sequence[str],
    jobs: Sequence[Mapping[str, Any]],
    gas: Mapping[str, DualPhaseSignalModel],
    co2: Mapping[str, DualPhaseSignalModel],
) -> dict[str, Any]:
    return {
        "schema": 2,
        "campaign_id": CAMPAIGN_ID,
        "dataset": dataset,
        "selection_id": f"{CAMPAIGN_ID}_{dataset}",
        "created_utc": _utc_now(),
        "output_root": str(root),
        "runs": list(runs),
        "jobs": list(jobs),
        "design": {
            "comparison": "generic-clearwater-vs-btb-clearwater-hierarchical-v1",
            "training_runs": ["ac18", "run15"],
            "target_runs_disjoint_from_training": True,
            "target_used_for_refitting": False,
            "weighting": "run -> class -> frame -> region -> pixel",
            "validation": ["leave-one-run-out", "contiguous temporal frame blocks"],
            "hierarchical_partition": {
                "co2_presence": "water versus any visible CO2",
                "conditional_phase": "free-gas evidence versus aqueous within visible CO2",
                "gas": "P(CO2) * P(gas | CO2)",
                "aqueous": "max(P(CO2) - P(gas), 0)",
            },
            "interpretation_scope": "visible-2d-slice-only",
            "mass_calibration_active": False,
            "depth_used": False,
            "deployment_allowed": False,
        },
        "models": {
            name: {
                "gas_path": str(GAS_MODEL_PATHS[name]),
                "gas_model_sha256": transfer._model_digest(gas[name]),
                "co2_path": str(CO2_MODEL_PATHS[name]),
                "co2_model_sha256": transfer._model_digest(co2[name]),
                "candidate_digest": transfer._composite_candidate_digest(
                    gas[name], co2[name]
                ),
                "feature_contract": gas[name].feature_contract,
            }
            for name in MODEL_ORDER
        },
        "review": {
            "blind_seed": BLIND_SEED,
            "incremental_publication": True,
            "output_dir": str(root / "manual_pairwise_review"),
        },
    }


def run_campaign(
    dataset: str,
    *,
    output_root: Path | None = None,
    runs: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    transfer.optical_control.install_runtime_hooks()
    _configure_transfer()
    gas, co2 = _load_model_pairs()
    selected_runs = tuple(str(run).casefold() for run in (runs or RUNS[dataset]))
    if set(selected_runs) & {"ac18", "run15"}:
        raise RuntimeError("Training runs AC18/Run15 cannot enter the blind campaign")
    root = Path(output_root or OUTPUTS[dataset]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = _candidate_jobs(dataset, selected_runs, gas, co2)
    manifest = _manifest(dataset, root, selected_runs, jobs, gas, co2)
    transfer._atomic_json(root / "postflight_manifest.json", manifest)
    transfer._atomic_json(root / "campaign_manifest.json", manifest)
    jobs_by_run = {
        run: {
            str(job["role"]): job
            for job in jobs
            if str(job["run"]).casefold() == run
        }
        for run in selected_runs
    }
    state_path = root / "campaign_state.json"
    completed: list[dict[str, Any]] = []
    transfer._atomic_json(
        state_path,
        {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "dataset": dataset,
            "status": "running",
            "completed": [],
            "remaining": list(selected_runs),
            "started_utc": _utc_now(),
            "updated_utc": _utc_now(),
        },
    )
    for run in selected_runs:
        required = [
            transfer._candidate_path(root, jobs_by_run[run][name]) / "summary.json"
            for name in MODEL_ORDER
        ]
        native_manifest = (
            root / "manual_pairwise_review" / "native_frames" / run / "manifest.json"
        )
        if not force and native_manifest.is_file() and all(path.is_file() for path in required):
            completed.append({"run": run, "dataset": dataset, "status": "reused"})
        else:
            try:
                completed.append(
                    transfer._write_run(
                        dataset,
                        run,
                        root,
                        gas,
                        jobs_by_run[run],
                        co2,
                        CO2_MODEL_PATHS,
                        hierarchical_phase=True,
                    )
                )
            except transfer.NegativeControlPreflightError as exc:
                completed.append(
                    {
                        "run": run,
                        "dataset": dataset,
                        "status": "blocked_negative_control_preflight",
                        "report": str(exc.report_path),
                        "failures": [
                            {
                                "control": row["control"],
                                "model": row["model"],
                                "failures": row["failures"],
                            }
                            for row in exc.report.get("controls") or []
                            if not bool(row.get("passed"))
                        ],
                        "completed_utc": _utc_now(),
                    }
                )
        done = {row["run"] for row in completed}
        transfer._atomic_json(
            state_path,
            {
                "schema": 1,
                "campaign_id": CAMPAIGN_ID,
                "dataset": dataset,
                "status": "running",
                "completed": completed,
                "remaining": [run for run in selected_runs if run not in done],
                "updated_utc": _utc_now(),
            },
        )
    blocked = [
        row for row in completed if str(row.get("status", "")).startswith("blocked_")
    ]
    result = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "dataset": dataset,
        "status": "complete" if not blocked else "completed_with_blocked_runs",
        "completed": completed,
        "blocked": blocked,
        "remaining": [],
        "updated_utc": _utc_now(),
    }
    transfer._atomic_json(state_path, result)
    transfer._atomic_json(root / "campaign_terminal.json", result)
    return result


def run_preflight(
    dataset: str,
    *,
    output_root: Path | None = None,
    runs: Sequence[str] | None = None,
) -> dict[str, Any]:
    transfer.optical_control.install_runtime_hooks()
    _configure_transfer()
    gas, co2 = _load_model_pairs()
    selected_runs = tuple(str(run).casefold() for run in (runs or RUNS[dataset]))
    root = Path(output_root or OUTPUTS[dataset]).resolve()
    rows: list[dict[str, Any]] = []
    for run in selected_runs:
        with transfer._open_source(dataset, run, require_clearwater=True) as source:
            report = transfer._negative_control_preflight(
                gas,
                source,
                co2,
                hierarchical_phase=True,
            )
        path = root / "preflight" / run / "negative_controls.json"
        transfer._atomic_json(path, report)
        rows.append({"run": run, "passed": bool(report["passed"]), "report": str(path)})
    result = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "dataset": dataset,
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "runs": rows,
        "updated_utc": _utc_now(),
    }
    transfer._atomic_json(root / "negative_control_preflight_summary.json", result)
    return result


def _status(dataset: str, output_root: Path | None = None) -> dict[str, Any]:
    root = Path(output_root or OUTPUTS[dataset]).resolve()
    path = root / "campaign_state.json"
    if not path.is_file():
        return {"dataset": dataset, "status": "not_started", "root": str(root)}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "preflight", "status"), nargs="?", default="run")
    parser.add_argument("--dataset", choices=("ffac", "ffum", "all"), default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    datasets = ("ffac", "ffum") if args.dataset == "all" else (args.dataset,)
    results: dict[str, Any] = {}
    for dataset in datasets:
        if args.action == "status":
            results[dataset] = _status(dataset, args.output_root)
        elif args.action == "preflight":
            results[dataset] = run_preflight(
                dataset,
                output_root=args.output_root,
                runs=args.runs,
            )
        else:
            results[dataset] = run_campaign(
                dataset,
                output_root=args.output_root,
                runs=args.runs,
                force=args.force,
            )
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
