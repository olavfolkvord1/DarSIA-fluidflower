"""Physically re-rank and independently validate the final AC campaign.

The final production search stored 5000 Optuna trials for every run.  The
Optuna objective contains deliberately soft physical terms, so its numerical
winner is not automatically the production winner.  This script:

1. re-ranks every feasible history row with fixed physical diagnostics;
2. exactly replays the best candidates on the 17 calibration frames;
3. evaluates reproducible finalists on 17 untouched image times; and
4. writes an approved calibration only when both fixed gates pass.

Validation is distributed through small claim files on the shared output
folder.  CUDA and OpenCL workers can therefore draw from the same run pool.
No result below ``Z:\\Albus\\Results`` is read, changed, or deleted.
"""

from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools import ac_clearwater_phase_pilot as pilot  # noqa: E402
from darsia.fluidflower.tools import ac_final_method_selection as selection  # noqa: E402
from darsia.fluidflower.tools import ac_final_production_calibration as campaign  # noqa: E402


CAMPAIGN_ID = "final_production_physical_selection_20260815"
SOURCE_CAMPAIGN_ID = campaign.CAMPAIGN_ID
SOURCE_VARIANT = campaign.VARIANT_NAME
SOURCE_CAMPAIGN_ROOT = campaign.DEFAULT_LOGS_ROOT
SOURCE_ROOT = SOURCE_CAMPAIGN_ROOT / SOURCE_VARIANT
DEFAULT_OUTPUT_ROOT = SOURCE_CAMPAIGN_ROOT / "physical_selection_20260815"
RUNS = tuple(campaign.ALL_RUNS)

PRIORITY_RUNS = (
    "ac17",
    "ac18",
    "ac56",
    "ac20",
    "ac24",
    "ac60",
)
TERMINAL_STATUSES = {
    "passed",
    "holdout_failed",
    "insufficient_temporal_coverage",
    "no_history_candidates",
    "no_training_guardrail_candidates",
    "no_reproducible_candidates",
    "validation_error",
}

DIAGNOSTIC_FIELDS = (
    "normalized_mae",
    "maximum_absolute_error_fraction",
    "window_mae_I1",
    "window_mae_I2",
    "window_mae_early_redistribution",
    "window_mae_late_redistribution",
    "worst_window_mae",
    "ratio_4p1",
    "ratio_4p1_abs_bias",
    "late_mean_ratio",
    "late_mean_abs_bias",
    "terminal_ratio",
    "late_ratio_range",
    "post_injection_total_variation",
    "post_injection_aqueous_decrease_fraction",
    "post_injection_gas_increase_fraction",
    "maximum_gas_envelope_excess_fraction",
    "selection_score",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _history_path(run: str) -> Path:
    final = SOURCE_ROOT / run / f"auto_calibration_{run}.csv"
    temporary = SOURCE_ROOT / run / f"tmp_auto_calibration_{run}.csv"
    return final if final.is_file() else temporary


def _source_final_path(run: str) -> Path:
    return SOURCE_ROOT / run / f"final_full_scale_{run}.json"


def _source_final_hash(run: str) -> str | None:
    path = _source_final_path(run)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    params = payload.get("params")
    if not isinstance(params, Mapping) or not params:
        return None
    return selection._params_sha256(params)


def collect_candidates(run: str) -> list[selection.Candidate]:
    """Load, diagnose, and de-duplicate one final production history."""

    history = _history_path(run)
    if not history.is_file():
        return []
    final_hash = _source_final_hash(run)
    by_hash: dict[str, selection.Candidate] = {}
    with history.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if not selection._valid_history_row(row):
                continue
            params = selection._parse_mapping(row.get("params"))
            metrics = selection._parse_mapping(row.get("metrics"))
            if not params or not metrics:
                continue
            try:
                diagnostics = selection.candidate_diagnostics(metrics)
                history_iter = int(row["iter"])
                objective = float(row["objective"])
            except (KeyError, TypeError, ValueError):
                continue
            digest = selection._params_sha256(params)
            candidate = selection.Candidate(
                run=run,
                source=SOURCE_CAMPAIGN_ID,
                history_path=str(history),
                history_iter=history_iter,
                history_objective=objective,
                params=params,
                metrics=metrics,
                params_sha256=digest,
                diagnostics=diagnostics,
                guardrail_failures=selection.training_guardrail_failures(
                    diagnostics
                ),
                source_final=digest == final_hash,
            )
            previous = by_hash.get(digest)
            if previous is None or (
                candidate.diagnostics["selection_score"],
                candidate.history_objective,
                candidate.history_iter,
            ) < (
                previous.diagnostics["selection_score"],
                previous.history_objective,
                previous.history_iter,
            ):
                by_hash[digest] = candidate
            elif candidate.source_final:
                previous.source_final = True
    return sorted(
        by_hash.values(),
        key=lambda item: (
            not item.passes_guardrails,
            item.diagnostics["selection_score"],
            item.history_objective,
            item.history_iter,
        ),
    )


def _candidate_payload(candidate: selection.Candidate) -> dict[str, Any]:
    return {
        "schema": 1,
        "run": candidate.run,
        "objective_full_scale": candidate.history_objective,
        "metrics": candidate.metrics,
        "params": candidate.params,
        "source": "final_production_fixed_physics_reranking_candidate",
        "selection_candidate": {
            "created_utc": _utc_now(),
            "source_campaign": SOURCE_CAMPAIGN_ID,
            "source_variant": SOURCE_VARIANT,
            "history_path": candidate.history_path,
            "history_iter": candidate.history_iter,
            "history_objective": candidate.history_objective,
            "params_sha256": candidate.params_sha256,
            "source_final": candidate.source_final,
            "diagnostics": candidate.diagnostics,
            "guardrail_failures": candidate.guardrail_failures,
            "score_weights": selection.SCORE_WEIGHTS,
            "training_guardrails": selection.TRAINING_GUARDRAILS,
        },
    }


def rank_histories(output_root: Path, *, replay_top: int) -> dict[str, Any]:
    """Create a compact replay set from all 200,000 stored trials."""

    output_root.mkdir(parents=True, exist_ok=True)
    ranking_path = output_root / "candidate_ranking.csv"
    ranking_fields = (
        "run",
        "rank",
        "passes_guardrails",
        "guardrail_failures",
        "source_final",
        "history_iter",
        "history_objective",
        "params_sha256",
        *DIAGNOSTIC_FIELDS,
    )
    manifest_runs: dict[str, Any] = {}
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ranking_fields)
        writer.writeheader()
        for run in RUNS:
            candidates = collect_candidates(run)
            accepted = [item for item in candidates if item.passes_guardrails]
            for rank, candidate in enumerate(accepted, start=1):
                candidate.rank = rank

            replay_pool = list(
                (accepted if accepted else candidates)[: max(1, replay_top)]
            )
            replay_hashes = {item.params_sha256 for item in replay_pool}
            replay_pool.extend(
                item
                for item in candidates
                if item.source_final and item.params_sha256 not in replay_hashes
            )
            candidate_rows: list[dict[str, Any]] = []
            for index, candidate in enumerate(replay_pool, start=1):
                path = (
                    output_root
                    / "replay_candidates"
                    / run
                    / (
                        f"candidate_{index:02d}_"
                        f"{candidate.params_sha256[:12]}.json"
                    )
                )
                _atomic_json(path, _candidate_payload(candidate))
                candidate.candidate_path = str(path)
                candidate_rows.append(
                    {
                        "rank": candidate.rank,
                        "history_iter": candidate.history_iter,
                        "history_objective": candidate.history_objective,
                        "params_sha256": candidate.params_sha256,
                        "source_final": candidate.source_final,
                        "candidate_path": str(path),
                        "diagnostics": candidate.diagnostics,
                        "guardrail_failures": candidate.guardrail_failures,
                    }
                )

            for candidate in candidates:
                writer.writerow(
                    {
                        "run": run,
                        "rank": candidate.rank,
                        "passes_guardrails": candidate.passes_guardrails,
                        "guardrail_failures": ";".join(
                            candidate.guardrail_failures
                        ),
                        "source_final": candidate.source_final,
                        "history_iter": candidate.history_iter,
                        "history_objective": candidate.history_objective,
                        "params_sha256": candidate.params_sha256,
                        **candidate.diagnostics,
                    }
                )

            if not candidates:
                status = "no_history_candidates"
            elif not accepted:
                status = "no_training_guardrail_candidates"
            else:
                status = "ready"
            manifest_runs[run] = {
                "status": status,
                "history_path": str(_history_path(run)),
                "source_final_path": str(_source_final_path(run)),
                "unique_candidate_count": len(candidates),
                "guardrail_pass_count": len(accepted),
                "replay_candidates": candidate_rows,
            }
            _atomic_json(
                output_root / "queue" / "pending" / f"{run}.json",
                {
                    "schema": 1,
                    "run": run,
                    "status": status,
                    "created_utc": _utc_now(),
                },
            )
            print(
                f"{run}: {len(accepted)}/{len(candidates)} pass; "
                f"{len(candidate_rows)} replay candidates",
                flush=True,
            )
            del candidates, accepted, replay_pool
            gc.collect()

    manifest = {
        "schema": 1,
        "created_utc": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_variant": SOURCE_VARIANT,
        "source_root": str(SOURCE_ROOT),
        "runs": manifest_runs,
        "run_order": list(RUNS),
        "score_weights": selection.SCORE_WEIGHTS,
        "training_guardrails": selection.TRAINING_GUARDRAILS,
        "holdout_guardrails": selection.HOLDOUT_GUARDRAILS,
        "holdout_times_h": list(selection.HOLDOUT_TIMES_H),
        "ranking_csv": str(ranking_path),
        "replay_top": replay_top,
    }
    _atomic_json(output_root / "selection_manifest.json", manifest)
    return manifest


def _overflow_assignments() -> dict[str, dict[str, str]]:
    path = SOURCE_CAMPAIGN_ROOT / "overflow_assignment.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {
            str(row["run"]).lower(): dict(row) for row in csv.DictReader(stream)
        }


def _campaign_environment_values(run: str) -> dict[str, str]:
    assignments = _overflow_assignments()
    if run not in assignments:
        raise KeyError(f"Missing overflow assignment for {run}")
    assignment = assignments[run]
    target = SOURCE_CAMPAIGN_ROOT / "overflow_targets" / f"{run}.json"
    if not target.is_file():
        raise FileNotFoundError(target)
    environment = campaign._campaign_env(campaign.VARIANT)
    environment.update(
        {
            "FFAC_OVERFLOW_PHASE_TARGETS": str(target),
            "FFAC_OVERFLOW_PHASE_PENALTY_WEIGHT": str(
                assignment["penalty_weight"]
            ),
            "FFAC_OVERFLOW_ASSIGNMENT_MODE": str(
                assignment["assignment_mode"]
            ),
        }
    )
    return environment


@contextmanager
def _selection_environment(run: str) -> Iterator[Any]:
    environment = _campaign_environment_values(run)
    with pilot.campaign_environment(campaign.VARIANT, environment):
        yield campaign.VARIANT


def _close_context(context: Any) -> None:
    selection._close_context(context)


def _reproducible_candidates(
    replay: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in replay:
        objective = float(row.get("objective", math.inf))
        mass_delta = row.get("history_max_relative_mass_delta")
        objective_delta = row.get("history_objective_abs_delta")
        if (
            row.get("status") == "ok"
            and bool(row.get("feasible"))
            and not row.get("guardrail_failures")
            and mass_delta is not None
            and objective_delta is not None
            and float(mass_delta) <= 5e-5
            and float(objective_delta)
            <= max(1e-10, 5e-5 * abs(objective))
        ):
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["diagnostics"]["selection_score"],
            row["objective"],
            row["params_sha256"],
        )
    )
    return rows


def _candidate_output(
    *,
    output_root: Path,
    run: str,
    training: Mapping[str, Any],
    holdout: Mapping[str, Any],
    validation_passed: bool,
) -> Path:
    source_payload = json.loads(
        Path(str(training["candidate_path"])).read_text(encoding="utf-8-sig")
    )
    folder = "selected" if validation_passed else "provisional"
    destination = output_root / folder / f"final_full_scale_{run}.json"
    payload = {
        "schema": 3,
        "run": run,
        "objective_full_scale": training["objective"],
        "objective_terms": training["objective_terms"],
        "metrics": training["metrics"],
        "params": source_payload["params"],
        "source": "fixed_physics_reranking_exact_replay_and_17_point_holdout",
        "final_selection": {
            "created_utc": _utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "source_campaign_id": SOURCE_CAMPAIGN_ID,
            "source_variant": SOURCE_VARIANT,
            "validation_passed": validation_passed,
            "source_candidate": training["candidate_path"],
            "params_sha256": training["params_sha256"],
            "training_rank": training.get("rank"),
            "history_iter": training.get("history_iter"),
            "training_diagnostics": training["diagnostics"],
            "training_guardrail_failures": training["guardrail_failures"],
            "training_reproduction": {
                "maximum_relative_mass_delta": training[
                    "history_max_relative_mass_delta"
                ],
                "objective_absolute_delta": training[
                    "history_objective_abs_delta"
                ],
            },
            "holdout_times_h": list(selection.HOLDOUT_TIMES_H),
            "holdout_diagnostics": holdout.get("diagnostics", {}),
            "holdout_guardrail_failures": holdout.get(
                "guardrail_failures", ["missing_holdout"]
            ),
            "score_weights": selection.SCORE_WEIGHTS,
            "training_guardrails": selection.TRAINING_GUARDRAILS,
            "holdout_guardrails": selection.HOLDOUT_GUARDRAILS,
        },
    }
    _atomic_json(destination, payload)
    return destination


def _error_status(stage: str, error: BaseException) -> str:
    message = str(error).lower()
    temporal_markers = (
        "holdout time",
        "evaluation time",
        "outside",
        "tolerance",
        "duplicate image",
    )
    if stage.startswith("holdout") and any(
        marker in message for marker in temporal_markers
    ):
        return "insufficient_temporal_coverage"
    return "validation_error"


def validate_run(
    output_root: Path,
    *,
    run: str,
    backend: str,
    finalist_count: int,
    force: bool = False,
) -> dict[str, Any]:
    """Replay and hold out one run without mutating any other run state."""

    manifest_path = output_root / "selection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run rank first; missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if run not in manifest["runs"]:
        raise KeyError(f"Run {run} is absent from {manifest_path}")
    report_path = output_root / "validation" / f"{run}.json"
    if report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if report.get("status") in TERMINAL_STATUSES:
            return report

    run_manifest = manifest["runs"][run]
    replay_candidates = list(run_manifest.get("replay_candidates") or [])
    if not replay_candidates:
        report = {
            "schema": 1,
            "run": run,
            "status": run_manifest.get("status", "no_history_candidates"),
            "validation_passed": False,
            "backend": backend,
            "host": socket.gethostname(),
            "completed_utc": _utc_now(),
            "training_replay": [],
            "holdout": [],
        }
        _atomic_json(report_path, report)
        return report

    started = time.perf_counter()
    stage = "training_context"
    training_context = None
    holdout_context = None
    replay: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    try:
        with _selection_environment(run) as variant:
            checkpoint = output_root / "checkpoints" / f"{run}_training.json"
            if checkpoint.is_file() and not force:
                saved = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
                replay = list(saved.get("training_replay") or [])
            if not replay:
                stage = "training_context"
                training_context = selection._build_evaluation_context(
                    run, variant, times=None
                )
                stage = "training_evaluation"
                replay = selection._evaluate_candidates(
                    training_context,
                    replay_candidates,
                    backend=backend,
                    holdout=False,
                )
                _atomic_json(
                    checkpoint,
                    {
                        "schema": 1,
                        "run": run,
                        "backend": backend,
                        "host": socket.gethostname(),
                        "completed_utc": _utc_now(),
                        "training_replay": replay,
                    },
                )
            if training_context is not None:
                _close_context(training_context)
                training_context = None

            reproducible = _reproducible_candidates(replay)
            if not reproducible:
                report = {
                    "schema": 1,
                    "run": run,
                    "status": "no_reproducible_candidates",
                    "validation_passed": False,
                    "backend": backend,
                    "host": socket.gethostname(),
                    "completed_utc": _utc_now(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "training_replay": replay,
                    "holdout": [],
                }
                _atomic_json(report_path, report)
                return report
            finalists = reproducible[: max(1, finalist_count)]

            stage = "holdout_context"
            holdout_context = selection._build_evaluation_context(
                run, variant, times=selection.HOLDOUT_TIMES_H
            )
            selection._disable_holdout_objective_terms(holdout_context)
            stage = "holdout_evaluation"
            holdout = selection._evaluate_candidates(
                holdout_context,
                finalists,
                backend=backend,
                holdout=True,
            )
            _close_context(holdout_context)
            holdout_context = None

        holdout_by_hash = {row["params_sha256"]: row for row in holdout}
        selected_training: dict[str, Any] | None = None
        selected_holdout: dict[str, Any] | None = None
        for training in finalists:
            candidate_holdout = holdout_by_hash[training["params_sha256"]]
            if (
                candidate_holdout["status"] == "ok"
                and candidate_holdout["feasible"]
                and not candidate_holdout["guardrail_failures"]
            ):
                selected_training = training
                selected_holdout = candidate_holdout
                break

        validation_passed = selected_training is not None
        if not validation_passed:
            paired = [
                (training, holdout_by_hash[training["params_sha256"]])
                for training in finalists
                if training["params_sha256"] in holdout_by_hash
            ]
            selected_training, selected_holdout = min(
                paired,
                key=lambda pair: (
                    len(pair[1].get("guardrail_failures") or []),
                    pair[1].get("diagnostics", {}).get(
                        "selection_score", math.inf
                    ),
                    pair[0]["diagnostics"]["selection_score"],
                ),
            )
        assert selected_training is not None and selected_holdout is not None
        destination = _candidate_output(
            output_root=output_root,
            run=run,
            training=selected_training,
            holdout=selected_holdout,
            validation_passed=validation_passed,
        )
        report = {
            "schema": 1,
            "run": run,
            "status": "passed" if validation_passed else "holdout_failed",
            "validation_passed": validation_passed,
            "backend": backend,
            "host": socket.gethostname(),
            "completed_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "selected": str(destination) if validation_passed else None,
            "provisional": None if validation_passed else str(destination),
            "selected_params_sha256": selected_training["params_sha256"],
            "training_rank": selected_training.get("rank"),
            "history_iter": selected_training.get("history_iter"),
            "training_diagnostics": selected_training["diagnostics"],
            "holdout_diagnostics": selected_holdout["diagnostics"],
            "holdout_guardrail_failures": selected_holdout[
                "guardrail_failures"
            ],
            "training_replay": replay,
            "holdout": holdout,
        }
        _atomic_json(report_path, report)
        print(
            f"{run}: {report['status']} rank={report['training_rank']} "
            f"holdout_MAE="
            f"{report['holdout_diagnostics']['normalized_mae']:.3%}",
            flush=True,
        )
        return report
    except Exception as error:
        report = {
            "schema": 1,
            "run": run,
            "status": _error_status(stage, error),
            "validation_passed": False,
            "backend": backend,
            "host": socket.gethostname(),
            "completed_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "training_replay": replay,
            "holdout": holdout,
        }
        _atomic_json(report_path, report)
        print(f"{run}: {report['status']}: {error}", flush=True)
        return report
    finally:
        if training_context is not None:
            _close_context(training_context)
        if holdout_context is not None:
            _close_context(holdout_context)


def _ordered_runs(manifest: Mapping[str, Any]) -> list[str]:
    available = [str(run) for run in manifest.get("run_order", RUNS)]
    priority = [run for run in PRIORITY_RUNS if run in available]
    return priority + [run for run in available if run not in priority]


def _claim_run(
    output_root: Path,
    *,
    run: str,
    worker_id: str,
    stale_seconds: float,
) -> bool:
    report_path = output_root / "validation" / f"{run}.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if report.get("status") in TERMINAL_STATUSES:
            return False
    claim = output_root / "queue" / "claims" / f"{run}.json"
    claim.parent.mkdir(parents=True, exist_ok=True)
    if claim.exists() and stale_seconds > 0:
        age = time.time() - claim.stat().st_mtime
        if age > stale_seconds:
            stale = claim.with_name(
                f"{claim.stem}.stale_{int(time.time())}_{os.getpid()}.json"
            )
            try:
                os.replace(claim, stale)
            except OSError:
                pass
    payload = json.dumps(
        {
            "schema": 1,
            "run": run,
            "worker_id": worker_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_utc": _utc_now(),
        },
        indent=2,
    )
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
    return True


def worker_loop(
    output_root: Path,
    *,
    backend: str,
    finalist_count: int,
    poll_seconds: float,
    stale_seconds: float,
    once: bool,
) -> dict[str, Any]:
    manifest_path = output_root / "selection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run rank first; missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    worker_id = f"{socket.gethostname()}_{backend}_{os.getpid()}"
    processed: list[str] = []
    while True:
        claimed: str | None = None
        for run in _ordered_runs(manifest):
            if _claim_run(
                output_root,
                run=run,
                worker_id=worker_id,
                stale_seconds=stale_seconds,
            ):
                claimed = run
                break
        if claimed is not None:
            active = output_root / "queue" / "active" / f"{claimed}.json"
            _atomic_json(
                active,
                {
                    "schema": 1,
                    "run": claimed,
                    "worker_id": worker_id,
                    "backend": backend,
                    "started_utc": _utc_now(),
                },
            )
            print(f"{worker_id}: validating {claimed}", flush=True)
            validate_run(
                output_root,
                run=claimed,
                backend=backend,
                finalist_count=finalist_count,
            )
            active.unlink(missing_ok=True)
            processed.append(claimed)
            if once:
                break
            continue

        reports = {
            path.stem: json.loads(path.read_text(encoding="utf-8-sig"))
            for path in (output_root / "validation").glob("ac*.json")
        }
        terminal = {
            run
            for run, report in reports.items()
            if report.get("status") in TERMINAL_STATUSES
        }
        if set(manifest["runs"]).issubset(terminal):
            break
        if once:
            break
        time.sleep(max(0.2, poll_seconds))
    return {
        "schema": 1,
        "worker_id": worker_id,
        "backend": backend,
        "processed": processed,
        "completed_utc": _utc_now(),
    }


def finalize(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "selection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    reports: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for run in manifest["run_order"]:
        path = output_root / "validation" / f"{run}.json"
        report = (
            json.loads(path.read_text(encoding="utf-8-sig"))
            if path.is_file()
            else {"run": run, "status": "pending", "validation_passed": False}
        )
        reports[run] = report
        status = str(report.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
        training = report.get("training_diagnostics") or {}
        holdout = report.get("holdout_diagnostics") or {}
        rows.append(
            {
                "run": run,
                "status": status,
                "validation_passed": bool(report.get("validation_passed")),
                "host": report.get("host", ""),
                "backend": report.get("backend", ""),
                "training_rank": report.get("training_rank", ""),
                "history_iter": report.get("history_iter", ""),
                "training_selection_score": training.get(
                    "selection_score", ""
                ),
                "training_normalized_mae": training.get("normalized_mae", ""),
                "training_terminal_ratio": training.get("terminal_ratio", ""),
                "holdout_selection_score": holdout.get("selection_score", ""),
                "holdout_normalized_mae": holdout.get("normalized_mae", ""),
                "holdout_worst_window_mae": holdout.get(
                    "worst_window_mae", ""
                ),
                "holdout_late_mean_ratio": holdout.get("late_mean_ratio", ""),
                "holdout_terminal_ratio": holdout.get("terminal_ratio", ""),
                "holdout_guardrail_failures": ";".join(
                    report.get("holdout_guardrail_failures") or []
                ),
                "selected": report.get("selected") or "",
                "provisional": report.get("provisional") or "",
            }
        )
    summary_csv = output_root / "validation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": 1,
        "updated_utc": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_variant": SOURCE_VARIANT,
        "expected_runs": len(manifest["runs"]),
        "reported_runs": sum(
            count for status, count in counts.items() if status != "pending"
        ),
        "passed_runs": counts.get("passed", 0),
        "complete": sum(counts.values()) == len(manifest["runs"])
        and counts.get("pending", 0) == 0,
        "status_counts": counts,
        "holdout_times_h": list(selection.HOLDOUT_TIMES_H),
        "reports": reports,
        "summary_csv": str(summary_csv),
    }
    _atomic_json(output_root / "validation_summary.json", summary)
    return summary


def render_review_movie(
    output_root: Path,
    *,
    run: str,
    backend: str,
    diagnostic_workers: int,
) -> dict[str, Any]:
    """Render the exact selected calibration as images, contours, and movie."""

    import cv2
    import numpy as np

    from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
        apply_params,
        voxel_volume_with_depth_model,
    )
    from darsia.fluidflower.tools.calibration_review_movie import (
        render_contour_frame,
        render_mass_plot,
        write_calibration_movie,
        write_movie_provenance,
    )
    from darsia.fluidflower.tools.spatial_mass_evaluator import create_spatial_mass_evaluator

    selected = output_root / "selected" / f"final_full_scale_{run}.json"
    if not selected.is_file():
        raise FileNotFoundError(
            f"{run} has no holdout-approved calibration: {selected}"
        )
    review_input = (
        output_root
        / "review_input"
        / SOURCE_VARIANT
        / run
        / f"final_full_scale_{run}.json"
    )
    payload = json.loads(selected.read_text(encoding="utf-8-sig"))
    _atomic_json(review_input, payload)
    review_root = output_root / "spatial_review"
    run_output = review_root / SOURCE_VARIANT / run
    run_output.mkdir(parents=True, exist_ok=True)
    params = payload.get("params")
    if not isinstance(params, dict) or not params:
        raise RuntimeError(f"Selected candidate has no params: {selected}")
    context = None
    evaluator = None
    records: list[dict[str, float | int | str]] = []
    contour_frames: list[Any] = []
    parity_errors: list[float] = []
    expected_metrics = [
        (float(str(key).removesuffix("h")), metric)
        for key, metric in (payload.get("metrics") or {}).items()
    ]
    if not expected_metrics:
        raise RuntimeError(f"Selected candidate has no time-resolved metrics: {selected}")
    try:
        with _selection_environment(run) as variant:
            context = selection._build_evaluation_context(run, variant, times=None)
            apply_params(
                context.calibration,
                params,
                labels=context.signal_labels,
                np_module=np,
            )
            shape = tuple(np.asarray(context.calibration.labels.img).shape)
            voxel_volume = voxel_volume_with_depth_model(context, params, shape)
            evaluator = create_spatial_mass_evaluator(
                context.calibration,
                backend=backend,
                output_dtype="float32",
                cuda_color_interpretation=(
                    "on" if backend in {"auto", "cuda"} else "off"
                ),
            )
            for index, (image, injected, elapsed_h) in enumerate(context._loaded):
                spatial = evaluator.evaluate(image)
                mass = np.asarray(spatial.mass.img)
                mass_g = np.asarray(spatial.mass_g.img)
                mass_aq = np.asarray(spatial.mass_aq.img)
                total = float(np.sum(mass * voxel_volume, dtype=np.float64))
                gas = float(np.sum(mass_g * voxel_volume, dtype=np.float64))
                aqueous = float(np.sum(mass_aq * voxel_volume, dtype=np.float64))
                saturation_g = np.asarray(spatial.saturation_g.img)
                concentration_aq = np.asarray(spatial.concentration_aq.img)
                expected = min(
                    expected_metrics,
                    key=lambda row: abs(row[0] - float(elapsed_h)),
                )[1]
                for actual, name in (
                    (total, "total_full"),
                    (gas, "gaseous_full"),
                    (aqueous, "aqueous_full"),
                ):
                    reference = selection._metric_value(expected, name)
                    parity_errors.append(
                        abs(actual - reference) / max(abs(reference), 1e-12)
                    )
                record: dict[str, float | int | str] = {
                    "run": run,
                    "variant": SOURCE_VARIANT,
                    "frame_index": index,
                    "elapsed_h": float(elapsed_h),
                    "injected_mass_kg": float(injected),
                    "detected_total_mass_kg": total,
                    "detected_gas_mass_kg": gas,
                    "detected_aqueous_mass_kg": aqueous,
                }
                records.append(record)
                contour_frames.append(
                    render_contour_frame(
                        image=np.asarray(image.img),
                        concentration_aq=concentration_aq,
                        saturation_g=saturation_g,
                        run=run,
                        variant=f"{SOURCE_VARIANT} | holdout-approved",
                        frame_index=index,
                        frame_count=len(context._loaded),
                        elapsed_h=float(elapsed_h),
                        injected_mass_kg=float(injected),
                        detected_total_mass_kg=total,
                        detected_gas_mass_kg=gas,
                        detected_aqueous_mass_kg=aqueous,
                        gas_fill_alpha=0.0,
                    )
                )
    finally:
        if evaluator is not None:
            evaluator.close()
        if context is not None:
            _close_context(context)

    maximum_parity_error = max(parity_errors, default=math.inf)
    if not math.isfinite(maximum_parity_error) or maximum_parity_error > 5e-5:
        raise RuntimeError(
            f"{run}: spatial movie fields do not reproduce selected masses; "
            f"maximum relative error={maximum_parity_error:.3e}"
        )
    per_frame_csv = run_output / "calibration_review_per_frame.csv"
    with per_frame_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    mass_plot = render_mass_plot(
        records,
        run=run,
        variant=f"{SOURCE_VARIANT} | fixed reranking + untouched holdout",
        objective_full_scale=float(payload["objective_full_scale"]),
        mass_objective_full_scale=(
            float(payload["objective_terms"]["mass_objective"])
            if isinstance(payload.get("objective_terms"), Mapping)
            and payload["objective_terms"].get("mass_objective") is not None
            else None
        ),
    )
    mass_plot_path = run_output / "calibration_mass_fit.png"
    cv2.imwrite(str(mass_plot_path), cv2.cvtColor(mass_plot, cv2.COLOR_RGB2BGR))
    movie_path = run_output / "calibration_review.mp4"
    movie = write_calibration_movie(
        output=movie_path,
        plot_frame=mass_plot,
        contour_frames=contour_frames,
    )
    movie.update(
        {
            "run": run,
            "variant": SOURCE_VARIANT,
            "candidate": str(selected),
            "candidate_sha256": _sha256(selected),
            "params_sha256": selection._params_sha256(params),
            "mass_plot": str(mass_plot_path),
            "per_frame_csv": str(per_frame_csv),
            "maximum_spatial_backend_parity_relative_error": (
                maximum_parity_error
            ),
            "spatial_backend_parity_relative_tolerance": 5e-5,
            "diagnostic_workers_requested": max(1, int(diagnostic_workers)),
            "clearwater_reference_used": False,
        }
    )
    write_movie_provenance(run_output / "calibration_review.json", movie)
    summary = {
        "normalized_mae": float(
            sum(
                abs(
                    float(row["detected_total_mass_kg"])
                    - float(row["injected_mass_kg"])
                )
                for row in records
            )
            / (
                len(records)
                * max(float(row["injected_mass_kg"]) for row in records)
            )
        ),
        "maximum_spatial_backend_parity_relative_error": maximum_parity_error,
        "frame_count": len(records),
        "calibration_review_movie": movie,
    }
    result = {
        "schema": 1,
        "run": run,
        "backend": backend,
        "selected_candidate": str(selected),
        "selected_params_sha256": selection._params_sha256(payload["params"]),
        "movie": summary["calibration_review_movie"]["movie"],
        "summary": summary,
        "completed_utc": _utc_now(),
    }
    _atomic_json(review_root / SOURCE_VARIANT / run / "selection_review.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "rank",
            "validate-run",
            "worker",
            "status",
            "finalize",
            "movie",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run", choices=RUNS)
    parser.add_argument("--replay-top", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument(
        "--backend", choices=("cuda", "opencl", "prepared"), default="cuda"
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--claim-stale-seconds", type=float, default=7200.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--movie-backend",
        choices=("auto", "cpu", "cuda", "opencl"),
        default="cuda",
    )
    parser.add_argument("--diagnostic-workers", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "rank":
        result = rank_histories(args.output_root, replay_top=args.replay_top)
    elif args.action == "validate-run":
        if args.run is None:
            raise SystemExit("validate-run requires --run")
        result = validate_run(
            args.output_root,
            run=args.run,
            backend=args.backend,
            finalist_count=args.finalists,
            force=args.force,
        )
    elif args.action == "worker":
        result = worker_loop(
            args.output_root,
            backend=args.backend,
            finalist_count=args.finalists,
            poll_seconds=args.poll_seconds,
            stale_seconds=args.claim_stale_seconds,
            once=args.once,
        )
    elif args.action == "movie":
        if args.run is None:
            raise SystemExit("movie requires --run")
        result = render_review_movie(
            args.output_root,
            run=args.run,
            backend=args.movie_backend,
            diagnostic_workers=args.diagnostic_workers,
        )
    else:
        result = finalize(args.output_root)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
