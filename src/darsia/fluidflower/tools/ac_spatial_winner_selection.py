"""Select spatially credible FF-AC calibration winners from the final screen.

The calibration campaign deliberately produced a broad candidate archive.  This
module performs the decision step separately: mass and temporal behaviour are
hard constraints, non-dominated candidates are checked on untouched times, and
spatial agreement decides among candidates that remain physically admissible.
No production calibration is replaced by this workflow.
"""

from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools import ac_clearwater_phase_pilot as pilot  # noqa: E402
from darsia.fluidflower.tools import ac_spatial_calibration_campaign as screen  # noqa: E402
from darsia.fluidflower.tools import calibration_review_grid_movie as grid_movie  # noqa: E402


SELECTION_ID = "spatial_winner_selection_20260816"
DEFAULT_SOURCE_ROOT = screen.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_ROOT / "winner_selection"
RUNS = screen.RUNS
MATERIAL_SPATIAL_CHANGE_FRACTION = 0.05

FAMILY_SPECS = {
    "hierarchical": screen.SEARCH_SPECS[0],
    "shared": screen.SEARCH_SPECS[-1],
}
VARIANT_FAMILIES = {
    spec.name: (
        "shared"
        if spec.variant.signal_parameterization == "shared-shape"
        else "hierarchical"
    )
    for spec in screen.SEARCH_SPECS
}

# The margins represent practical equivalence in the mass-balance diagnostics.
# They are deliberately expressed relative to the best observed candidate for
# each run, instead of silently combining mass and image appearance in one sum.
TRAINING_CAP_RULES = {
    "selection_score": (0.0125, 0.20),
    "normalized_mae": (0.0100, 0.20),
    "worst_window_mae": (0.0300, 0.20),
    "maximum_absolute_error_fraction": (0.0500, 0.00),
}
HOLDOUT_CAP_RULES = {
    "selection_score": (0.0150, 0.20),
    "normalized_mae": (0.0125, 0.20),
    "worst_window_mae": (0.0400, 0.20),
    "maximum_absolute_error_fraction": (0.0600, 0.00),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    for attempt in range(screen.TRANSIENT_IO_RETRIES):
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            os.replace(temporary, path)
            return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt + 1 >= screen.TRANSIENT_IO_RETRIES:
                raise
            time.sleep(min(5.0, 0.25 * (2**attempt)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _diagnostic(row: Mapping[str, Any], name: str) -> float:
    return float((row.get("diagnostics") or {}).get(name, math.inf))


def _spatial_score(row: Mapping[str, Any]) -> float:
    explicit = row.get("training_spatial_normalized")
    if explicit is not None:
        return float(explicit)
    return max(
        0.0,
        float(row.get("spatial_rank_score", math.inf))
        - _diagnostic(row, "selection_score"),
    )


def _gain_regularization(row: Mapping[str, Any]) -> float:
    return float(
        (row.get("objective_terms") or {}).get(
            "layer_gain_regularization_z2", math.inf
        )
    )


def _base_valid(row: Mapping[str, Any]) -> bool:
    required = (
        "selection_score",
        "normalized_mae",
        "worst_window_mae",
        "maximum_absolute_error_fraction",
    )
    return (
        row.get("status") == "ok"
        and bool(row.get("feasible"))
        and not row.get("guardrail_failures")
        and all(_finite(_diagnostic(row, name)) for name in required)
        and _finite(_spatial_score(row))
        and _finite(_gain_regularization(row))
    )


def relative_caps(
    rows: Sequence[Mapping[str, Any]],
    rules: Mapping[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Return per-metric best values and practical-equivalence caps."""

    if not rows:
        raise ValueError("Cannot derive relative caps from an empty candidate set.")
    caps: dict[str, dict[str, float]] = {}
    for metric, (absolute_margin, relative_margin) in rules.items():
        best = min(_diagnostic(row, metric) for row in rows)
        margin = max(float(absolute_margin), float(relative_margin) * best)
        caps[metric] = {"best": best, "margin": margin, "cap": best + margin}
    return caps


def relative_cap_failures(
    row: Mapping[str, Any], caps: Mapping[str, Mapping[str, float]]
) -> list[str]:
    return [
        f"relative_{metric}"
        for metric, values in caps.items()
        if _diagnostic(row, metric) > float(values["cap"]) + 1e-12
    ]


def pareto_front(
    rows: Sequence[Mapping[str, Any]], objective_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Return rows not strictly dominated in all minimization objectives."""

    def values(row: Mapping[str, Any]) -> tuple[float, ...]:
        return tuple(float(row[name]) for name in objective_names)

    front: list[dict[str, Any]] = []
    for index, candidate in enumerate(rows):
        candidate_values = values(candidate)
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            other_values = values(other)
            no_worse = all(
                left <= right + 1e-12
                for left, right in zip(other_values, candidate_values, strict=True)
            )
            strictly_better = any(
                left < right - 1e-12
                for left, right in zip(other_values, candidate_values, strict=True)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(dict(candidate))
    return front


def _canonical_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not _base_valid(row),
        len(row.get("guardrail_failures") or []),
        _diagnostic(row, "selection_score"),
        _spatial_score(row),
        float(row.get("objective", math.inf)),
        str(row.get("source_variant", "")),
        int(row.get("sequence", 0)),
    )


def _enrich_row(
    raw: Mapping[str, Any], *, run: str, family: str, source_variant: str
) -> dict[str, Any]:
    row = dict(raw)
    digest = str(row.get("params_sha256", ""))
    row.update(
        {
            "candidate_id": f"{run}__{family}__{digest[:16]}",
            "run": run,
            "family": family,
            "source_variant": source_variant,
            "training_spatial_normalized": _spatial_score(row),
            "gain_regularization_z2": _gain_regularization(row),
        }
    )
    return row


def _load_run_candidates(source_root: Path, run: str) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    replay_path = source_root / "track_a_replay" / run / "candidate_replay.jsonl"
    raw_rows.extend(
        _enrich_row(
            row,
            run=run,
            family="hierarchical",
            source_variant="history_replay",
        )
        for row in _read_jsonl(replay_path)
    )
    for spec in screen.SEARCH_SPECS:
        trials_path = source_root / "track_b_optuna" / spec.name / run / "trials.jsonl"
        raw_rows.extend(
            _enrich_row(
                row,
                run=run,
                family=VARIANT_FAMILIES[spec.name],
                source_variant=spec.name,
            )
            for row in _read_jsonl(trials_path)
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        digest = str(row.get("params_sha256", ""))
        if digest:
            grouped.setdefault((str(row["family"]), digest), []).append(row)

    unique: list[dict[str, Any]] = []
    for (_family, _digest), duplicates in grouped.items():
        canonical = dict(min(duplicates, key=_canonical_key))
        canonical["provenance"] = [
            {
                "source_variant": row["source_variant"],
                "phase": row.get("phase"),
                "sequence": row.get("sequence"),
                "source": row.get("source") or {},
            }
            for row in sorted(duplicates, key=_canonical_key)
        ]
        unique.append(canonical)
    return unique


def _campaign_runs(source_root: Path) -> tuple[str, ...]:
    manifest_path = source_root / "campaign_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source campaign manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    runs = tuple(str(run).lower() for run in manifest.get("runs") or ())
    if not runs:
        raise RuntimeError(f"Source campaign manifest has no runs: {manifest_path}")
    return runs


def _control_hash(run: str) -> str:
    return screen._params_hash(screen._source_params(run))


def build_training_shortlist(
    source_root: Path, run: str
) -> dict[str, Any]:
    """Build a constrained three-objective Pareto shortlist for one run."""

    unique = _load_run_candidates(source_root, run)
    control_digest = _control_hash(run)
    control = next(
        (
            row
            for row in unique
            if row["family"] == "hierarchical"
            and row.get("params_sha256") == control_digest
        ),
        None,
    )
    if control is None:
        raise RuntimeError(f"Production control was not replayed for {run}.")

    base_valid = [row for row in unique if _base_valid(row)]
    caps = relative_caps(base_valid, TRAINING_CAP_RULES) if base_valid else {}
    for row in unique:
        if not base_valid:
            failures = ["no_training_candidate_passed_fixed_guardrails"]
        else:
            failures = relative_cap_failures(row, caps) if _base_valid(row) else []
        row["training_relative_failures"] = failures
        row["training_eligible"] = bool(base_valid) and _base_valid(row) and not failures
        row["training_mass_score"] = _diagnostic(row, "selection_score")

    eligible = [row for row in unique if row["training_eligible"]]
    front = pareto_front(
        eligible,
        (
            "training_mass_score",
            "training_spatial_normalized",
            "gain_regularization_z2",
        ),
    )
    front_ids = {row["candidate_id"] for row in front}

    selected_by_id = {row["candidate_id"]: dict(row) for row in front}
    selected_by_id.setdefault(control["candidate_id"], dict(control))
    shortlist = []
    for candidate in selected_by_id.values():
        roles = []
        if candidate["candidate_id"] in front_ids:
            roles.append("training_pareto")
        if candidate["candidate_id"] == control["candidate_id"]:
            roles.append("production_control")
        candidate["roles"] = roles
        shortlist.append(candidate)
    shortlist.sort(
        key=lambda row: (
            row["family"],
            row["training_mass_score"],
            row["training_spatial_normalized"],
            row["candidate_id"],
        )
    )
    return {
        "schema": 1,
        "selection_id": SELECTION_ID,
        "run": run,
        "created_utc": _utc_now(),
        "candidate_counts": {
            "loaded": len(unique),
            "base_valid": len(base_valid),
            "mass_constrained": len(eligible),
            "pareto": len(front),
            "holdout_shortlist": len(shortlist),
        },
        "selection_resolution": (
            "no_fixed_guardrail_candidate"
            if not base_valid
            else "no_candidate_within_joint_relative_caps"
            if not eligible
            else "standard"
        ),
        "training_cap_rules": TRAINING_CAP_RULES,
        "training_caps": caps,
        "pareto_objectives": [
            "training_mass_score",
            "training_spatial_normalized",
            "gain_regularization_z2",
        ],
        "production_control_id": control["candidate_id"],
        "candidates": shortlist,
    }


def prepare_selection(
    source_root: Path, output_root: Path, *, force: bool = False
) -> dict[str, Any]:
    manifest_path = output_root / "selection_manifest.json"
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    campaign_status = screen.campaign_status(source_root)
    if not campaign_status.get("complete") or campaign_status.get("counts", {}).get(
        "failed", 0
    ):
        raise RuntimeError(
            "Source spatial campaign must be complete without failed jobs before "
            "winner selection."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    runs = _campaign_runs(source_root)
    jobs: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for run in runs:
        summary = build_training_shortlist(source_root, run)
        _atomic_json(output_root / "training_shortlist" / f"{run}.json", summary)
        summaries[run] = summary["candidate_counts"]
        by_family: dict[str, list[str]] = {}
        for candidate in summary["candidates"]:
            by_family.setdefault(candidate["family"], []).append(
                candidate["candidate_id"]
            )
        for family, candidate_ids in sorted(by_family.items()):
            jobs.append(
                {
                    "job_id": f"holdout_{run}_{family}",
                    "run": run,
                    "family": family,
                    "candidate_ids": candidate_ids,
                }
            )

    manifest = {
        "schema": 1,
        "selection_id": SELECTION_ID,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "created_utc": _utc_now(),
        "runs": list(runs),
        "holdout_times_h": list(screen.selection.HOLDOUT_TIMES_H),
        "training_summary": summaries,
        "jobs": jobs,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _claim_job(root: Path, group: str, job_id: str, worker_id: str) -> bool:
    if (root / f"{group}_done" / f"{job_id}.json").is_file():
        return False
    if (root / f"{group}_failed" / f"{job_id}.json").is_file():
        return False
    claim = root / f"{group}_claims" / job_id
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError:
        return False
    _atomic_json(
        claim / "lease.json",
        {
            "schema": 1,
            "worker_id": worker_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_utc": _utc_now(),
        },
    )
    return True


def _all_terminal(root: Path, group: str, jobs: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        (root / f"{group}_done" / f"{job['job_id']}.json").is_file()
        or (root / f"{group}_failed" / f"{job['job_id']}.json").is_file()
        for job in jobs
    )


def _shortlist_candidates(
    output_root: Path, run: str, candidate_ids: Iterable[str]
) -> list[dict[str, Any]]:
    payload = json.loads(
        (output_root / "training_shortlist" / f"{run}.json").read_text(
            encoding="utf-8-sig"
        )
    )
    wanted = set(candidate_ids)
    candidates = [
        row for row in payload["candidates"] if row["candidate_id"] in wanted
    ]
    if len(candidates) != len(wanted):
        raise RuntimeError(f"Shortlist changed after manifest preparation for {run}.")
    return candidates


def _run_holdout_job(
    output_root: Path, job: Mapping[str, Any], *, backend: str
) -> dict[str, Any]:
    run = str(job["run"])
    family = str(job["family"])
    candidates = _shortlist_candidates(output_root, run, job["candidate_ids"])
    spec = FAMILY_SPECS[family]
    rows = screen._run_holdout(
        output_root / "raw_holdout",
        run=run,
        spec=spec,
        backend=backend,
        finalists=candidates,
    )
    by_hash = {
        str(row["source"]["training_params_sha256"]): row for row in rows
    }
    enriched = []
    for candidate in candidates:
        row = dict(by_hash[str(candidate["params_sha256"])])
        row["candidate_id"] = candidate["candidate_id"]
        row["run"] = run
        row["family"] = family
        row["holdout_spatial_normalized"] = _spatial_score(row)
        enriched.append(row)
    result = {
        "schema": 1,
        "selection_id": SELECTION_ID,
        "run": run,
        "family": family,
        "backend": backend,
        "requested_times_h": list(screen.selection.HOLDOUT_TIMES_H),
        "times_h": (
            list(enriched[0]["source"].get("resolved_holdout_times_h") or [])
            if enriched
            else []
        ),
        "evaluation_time_records": (
            list(enriched[0]["source"].get("evaluation_time_records") or [])
            if enriched
            else []
        ),
        "completed_utc": _utc_now(),
        "candidates": enriched,
    }
    _atomic_json(output_root / "holdout" / run / f"{family}.json", result)
    return result


def holdout_worker(
    output_root: Path, *, backend: str, once: bool = False
) -> dict[str, Any]:
    manifest = json.loads(
        (output_root / "selection_manifest.json").read_text(encoding="utf-8-sig")
    )
    jobs = list(manifest["jobs"])
    worker_id = f"{socket.gethostname()}_{backend}_{os.getpid()}"
    processed: list[str] = []
    while True:
        job = next(
            (
                item
                for item in jobs
                if _claim_job(
                    output_root, "holdout", str(item["job_id"]), worker_id
                )
            ),
            None,
        )
        if job is None:
            if _all_terminal(output_root, "holdout", jobs) or once:
                break
            time.sleep(3.0)
            continue
        job_id = str(job["job_id"])
        print(f"[{_utc_now()}] starting {job_id}", flush=True)
        try:
            result = _run_holdout_job(output_root, job, backend=backend)
            _atomic_json(
                output_root / "holdout_done" / f"{job_id}.json",
                {
                    "schema": 1,
                    "job": job,
                    "worker_id": worker_id,
                    "completed_utc": _utc_now(),
                    "candidate_count": len(result["candidates"]),
                },
            )
            processed.append(job_id)
            print(f"[{_utc_now()}] completed {job_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            _atomic_json(
                output_root / "holdout_failed" / f"{job_id}.json",
                {
                    "schema": 1,
                    "job": job,
                    "worker_id": worker_id,
                    "failed_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{_utc_now()}] FAILED {job_id}: {exc}", flush=True)
        if once:
            break
    return {"worker_id": worker_id, "processed": processed}


def _status_for_jobs(
    output_root: Path, group: str, jobs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    counts = {"pending": 0, "active": 0, "done": 0, "failed": 0}
    rows = []
    for job in jobs:
        job_id = str(job["job_id"])
        if (output_root / f"{group}_done" / f"{job_id}.json").is_file():
            state = "done"
        elif (output_root / f"{group}_failed" / f"{job_id}.json").is_file():
            state = "failed"
        elif (output_root / f"{group}_claims" / job_id).is_dir():
            state = "active"
        else:
            state = "pending"
        counts[state] += 1
        rows.append({**job, "state": state})
    return {
        "counts": counts,
        "complete": counts["done"] + counts["failed"] == len(jobs),
        "jobs": rows,
    }


def selection_status(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "selection_manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_prepared", "output_root": str(output_root)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    result = {
        "schema": 1,
        "selection_id": SELECTION_ID,
        "output_root": str(output_root),
        "holdout": _status_for_jobs(output_root, "holdout", manifest["jobs"]),
        "finalized": (output_root / "selection_summary.json").is_file(),
        "updated_utc": _utc_now(),
    }
    postflight_manifest = output_root / "postflight_manifest.json"
    if postflight_manifest.is_file():
        payload = json.loads(postflight_manifest.read_text(encoding="utf-8-sig"))
        result["postflight"] = _status_for_jobs(
            output_root, "postflight", payload["jobs"]
        )
        result["postflight_finalized"] = (
            output_root / "postflight" / "POSTFLIGHT_COMPLETE.json"
        ).is_file()
    return result


def _holdout_rows(output_root: Path, run: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((output_root / "holdout" / run).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in payload["candidates"]:
            rows[str(row["candidate_id"])] = row
    return rows


def choose_recommended(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose minimum robust spatial error after all mass gates pass."""

    eligible = [row for row in candidates if row.get("selection_eligible")]
    if not eligible:
        raise RuntimeError("No candidate passed the complete selection contract.")
    return dict(
        min(
            eligible,
            key=lambda row: (
                float(row["robust_spatial_score"]),
                float(row["robust_mass_score"]),
                float(row["gain_regularization_z2"]),
                str(row["candidate_id"]),
            ),
        )
    )


def choose_diagnostic_alternative(
    candidates: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Choose the strongest distinct candidate for blinded visual review."""

    reference_identity = (
        str(reference.get("family") or ""),
        str(reference.get("params_sha256") or ""),
    )
    distinct = [
        row
        for row in candidates
        if (
            str(row.get("family") or ""),
            str(row.get("params_sha256") or ""),
        )
        != reference_identity
    ]
    if not distinct:
        return None

    def finite_score(row: Mapping[str, Any], key: str) -> float:
        try:
            value = float(row.get(key, math.inf))
        except (TypeError, ValueError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    return dict(
        min(
            distinct,
            key=lambda row: (
                0 if row.get("selection_eligible") else 1,
                len(row.get("selection_failures") or []),
                finite_score(row, "robust_spatial_score"),
                finite_score(row, "robust_mass_score"),
                finite_score(row, "gain_regularization_z2"),
                str(row.get("candidate_id") or ""),
            ),
        )
    )


def classify_recommendation(
    candidates: Sequence[Mapping[str, Any]], control_id: str
) -> dict[str, Any]:
    """Choose conservatively and expose cases that require a new search."""

    control = next(
        dict(row) for row in candidates if row["candidate_id"] == control_id
    )
    eligible = [row for row in candidates if row.get("selection_eligible")]
    if not eligible:
        return {
            "status": "unresolved_no_candidate_passed",
            "reason": (
                "No candidate passed the complete training and holdout contract; "
                "retain the production control pending targeted review."
            ),
            "recommended": control,
            "proposed": control,
            "control": control,
            "spatial_change_fraction": 0.0,
        }

    proposed = choose_recommended(candidates)
    same_candidate = (
        proposed["params_sha256"] == control["params_sha256"]
        and proposed["family"] == control["family"]
    )
    spatial_change = (
        float(proposed["robust_spatial_score"])
        / max(float(control["robust_spatial_score"]), 1e-15)
        - 1.0
    )
    if same_candidate:
        status = "retain_control"
        recommended = control
        reason = "The production control is the best eligible spatial candidate."
    elif bool(control.get("selection_eligible")):
        if spatial_change <= -MATERIAL_SPATIAL_CHANGE_FRACTION:
            status = "clear_improvement_pending_visual_review"
            recommended = proposed
            reason = "Eligible alternative materially improves robust spatial error."
        else:
            status = "retain_control_marginal_change"
            recommended = control
            reason = (
                "Alternative spatial improvement is below the material-change "
                "threshold."
            )
    elif spatial_change <= MATERIAL_SPATIAL_CHANGE_FRACTION:
        status = "replacement_for_mass_failure_pending_visual_review"
        recommended = proposed
        reason = (
            "Control misses relative mass gates and the replacement is spatially "
            "non-inferior within tolerance."
        )
    else:
        status = "unresolved_mass_spatial_tradeoff"
        recommended = control
        reason = (
            "Mass-improving candidates materially worsen spatial behaviour; run "
            "requires a targeted spatial Optuna search."
        )
    return {
        "status": status,
        "reason": reason,
        "recommended": recommended,
        "proposed": proposed,
        "control": control,
        "spatial_change_fraction": spatial_change,
    }


def _candidate_payload(
    run: str, candidate: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    return {
        "schema": 1,
        "run": run,
        "variant": f"spatial_{role}_{candidate['family']}",
        "objective_full_scale": float(candidate.get("objective", math.nan)),
        "objective_terms": candidate.get("objective_terms") or {},
        "params": candidate["params"],
        "selection": {
            "selection_id": SELECTION_ID,
            "role": role,
            "candidate_id": candidate["candidate_id"],
            "params_sha256": candidate["params_sha256"],
            "family": candidate["family"],
            "source_variant": candidate.get("source_variant"),
            "training": {
                "diagnostics": candidate.get("diagnostics"),
                "spatial_normalized": candidate["training_spatial_normalized"],
                "relative_failures": candidate.get("training_relative_failures"),
            },
            "holdout": candidate.get("holdout"),
            "robust_mass_score": candidate.get("robust_mass_score"),
            "robust_spatial_score": candidate.get("robust_spatial_score"),
            "status": "pending_visual_review",
        },
    }


def finalize_selection(output_root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (output_root / "selection_manifest.json").read_text(encoding="utf-8-sig")
    )
    status = _status_for_jobs(output_root, "holdout", manifest["jobs"])
    if not status["complete"] or status["counts"]["failed"]:
        raise RuntimeError(f"Holdout is not complete and clean: {status['counts']}")

    reports: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    runs = tuple(str(run) for run in manifest["runs"])
    source_manifest = json.loads(
        (Path(manifest["source_root"]) / "campaign_manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )
    limited_temporal = set(
        source_manifest.get("limited_temporal_coverage_runs")
        or screen.LIMITED_TEMPORAL_COVERAGE_RUNS
    )
    for run in runs:
        training = json.loads(
            (output_root / "training_shortlist" / f"{run}.json").read_text(
                encoding="utf-8-sig"
            )
        )
        holdout_by_id = _holdout_rows(output_root, run)
        combined: list[dict[str, Any]] = []
        for raw in training["candidates"]:
            candidate = dict(raw)
            holdout = holdout_by_id.get(candidate["candidate_id"])
            if holdout is None:
                raise RuntimeError(f"Missing holdout for {candidate['candidate_id']}")
            candidate["holdout"] = holdout
            combined.append(candidate)

        holdout_base = [
            row["holdout"]
            for row in combined
            if row["holdout"].get("status") == "ok"
            and bool(row["holdout"].get("feasible"))
            and not row["holdout"].get("holdout_guardrail_failures")
        ]
        holdout_caps = (
            relative_caps(holdout_base, HOLDOUT_CAP_RULES) if holdout_base else {}
        )

        for candidate in combined:
            holdout = candidate["holdout"]
            if not holdout_base:
                holdout_relative = [
                    "no_holdout_candidate_passed_fixed_guardrails"
                ]
            elif holdout.get("status") == "ok" and bool(holdout.get("feasible")):
                holdout_relative = relative_cap_failures(holdout, holdout_caps)
            else:
                holdout_relative = ["invalid_holdout"]
            candidate["holdout_relative_failures"] = holdout_relative
            fixed_failures = list(holdout.get("holdout_guardrail_failures") or [])
            candidate["selection_failures"] = [
                *list(candidate.get("guardrail_failures") or []),
                *list(candidate.get("training_relative_failures") or []),
                *[f"holdout_{name}" for name in fixed_failures],
                *[f"holdout_{name}" for name in holdout_relative],
            ]
            candidate["selection_eligible"] = (
                bool(candidate.get("training_eligible"))
                and holdout.get("status") == "ok"
                and bool(holdout.get("feasible"))
                and not candidate["selection_failures"]
            )
            candidate["robust_mass_score"] = max(
                float(candidate["training_mass_score"]),
                _diagnostic(holdout, "selection_score"),
            )
            candidate["robust_spatial_score"] = max(
                float(candidate["training_spatial_normalized"]),
                float(holdout["holdout_spatial_normalized"]),
            )

        control_id = training["production_control_id"]
        decision = classify_recommendation(combined, control_id)
        recommended = decision["recommended"]
        proposed = decision["proposed"]
        control = decision["control"]
        eligible = [row for row in combined if row["selection_eligible"]]
        robust_front = pareto_front(
            eligible,
            (
                "robust_mass_score",
                "robust_spatial_score",
                "gain_regularization_z2",
            ),
        )
        robust_front_ids = [row["candidate_id"] for row in robust_front]
        recommendation_changed = (
            recommended["params_sha256"] != control["params_sha256"]
            or recommended["family"] != control["family"]
        )
        holdout_source = combined[0]["holdout"].get("source") or {}
        resolved_holdout_times = [
            float(value)
            for value in holdout_source.get("resolved_holdout_times_h") or []
        ]
        if not resolved_holdout_times:
            for key in (combined[0]["holdout"].get("metrics") or {}):
                try:
                    resolved_holdout_times.append(float(str(key).removesuffix("h")))
                except ValueError:
                    continue
        evaluation_time_records = list(
            holdout_source.get("evaluation_time_records") or []
        )
        report = {
            "schema": 1,
            "selection_id": SELECTION_ID,
            "run": run,
            "status": decision["status"],
            "decision_reason": decision["reason"],
            "temporal_coverage": (
                "limited_before_24h"
                if run in limited_temporal
                else "full_late_gas_window"
            ),
            "holdout_time_coverage": {
                "frame_count": len(resolved_holdout_times),
                "first_time_h": (
                    min(resolved_holdout_times) if resolved_holdout_times else None
                ),
                "last_time_h": (
                    max(resolved_holdout_times) if resolved_holdout_times else None
                ),
                "resolved_times_h": resolved_holdout_times,
                "dropped": [
                    row
                    for row in evaluation_time_records
                    if row.get("status") != "selected"
                ],
            },
            "policy": (
                "Fixed and relative mass/temporal gates first; minimum worst-case "
                "training/holdout spatial error second; robust mass, gain "
                "regularization and stable ID break ties."
            ),
            "training_caps": training["training_caps"],
            "holdout_caps": holdout_caps,
            "robust_pareto_ids": robust_front_ids,
            "recommended_id": recommended["candidate_id"],
            "proposed_id": proposed["candidate_id"],
            "control_id": control["candidate_id"],
            "recommendation_changed": recommendation_changed,
            "recommended": recommended,
            "proposed": proposed,
            "production_control": control,
            "candidates": combined,
        }
        reports[run] = report
        _atomic_json(output_root / "decisions" / f"{run}.json", report)
        _atomic_json(
            output_root / "recommended" / f"final_full_scale_{run}.json",
            _candidate_payload(run, recommended, role="recommended"),
        )
        _atomic_json(
            output_root / "controls" / f"final_full_scale_{run}.json",
            _candidate_payload(run, control, role="control"),
        )
        csv_rows.append(
            {
                "run": run,
                "temporal_coverage": report["temporal_coverage"],
                "holdout_frame_count": len(resolved_holdout_times),
                "holdout_last_time_h": (
                    max(resolved_holdout_times) if resolved_holdout_times else None
                ),
                "decision_status": decision["status"],
                "changed_from_control": recommendation_changed,
                "recommended_family": recommended["family"],
                "recommended_candidate_id": recommended["candidate_id"],
                "proposed_candidate_id": proposed["candidate_id"],
                "recommended_training_nmae": _diagnostic(
                    recommended, "normalized_mae"
                ),
                "control_training_nmae": _diagnostic(control, "normalized_mae"),
                "recommended_holdout_nmae": _diagnostic(
                    recommended["holdout"], "normalized_mae"
                ),
                "control_holdout_nmae": _diagnostic(
                    control["holdout"], "normalized_mae"
                ),
                "recommended_robust_spatial": recommended["robust_spatial_score"],
                "control_robust_spatial": control["robust_spatial_score"],
                "spatial_change_fraction": decision["spatial_change_fraction"],
                "recommended_robust_mass": recommended["robust_mass_score"],
                "control_robust_mass": control["robust_mass_score"],
                "recommended_gain_z2": recommended["gain_regularization_z2"],
                "control_gain_z2": control["gain_regularization_z2"],
                "eligible_candidates": len(eligible),
                "robust_pareto_candidates": len(robust_front),
            }
        )

    csv_path = output_root / "selection_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    result = {
        "schema": 1,
        "selection_id": SELECTION_ID,
        "completed_utc": _utc_now(),
        "status": "classified_pending_visual_review",
        "source_root": manifest["source_root"],
        "summary_csv": str(csv_path),
        "runs": reports,
    }
    _atomic_json(output_root / "selection_summary.json", result)
    return result


def prepare_postflight(output_root: Path, *, force: bool = False) -> dict[str, Any]:
    summary_path = output_root / "selection_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Finalize winner selection before preparing postflight.")
    manifest_path = output_root / "postflight_manifest.json"
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    jobs: list[dict[str, Any]] = []
    runs = tuple(str(run) for run in summary["runs"])
    for run in runs:
        report = summary["runs"][run]
        candidates = [("recommended", report["recommended"])]
        comparison = report.get("proposed") or report["recommended"]
        comparison_role = "proposed"
        if (
            comparison["candidate_id"] == report["recommended"]["candidate_id"]
            and report["recommended"]["candidate_id"]
            == report["production_control"]["candidate_id"]
        ):
            diagnostic = choose_diagnostic_alternative(
                report.get("candidates") or [],
                report["recommended"],
            )
            if diagnostic is not None:
                comparison = diagnostic
                comparison_role = "alternative"
        if (
            comparison["candidate_id"] != report["recommended"]["candidate_id"]
        ):
            candidates.insert(0, (comparison_role, comparison))
        if all(
            candidate["candidate_id"]
            != report["production_control"]["candidate_id"]
            for _role, candidate in candidates
        ):
            candidates.append(("control", report["production_control"]))
        for order, (role, candidate) in enumerate(candidates):
            variant_name = f"{order + 1:02d}_{role}_{candidate['family']}"
            payload = _candidate_payload(run, candidate, role=role)
            payload["variant"] = variant_name
            candidate_path = (
                output_root
                / "postflight_input"
                / variant_name
                / run
                / f"final_full_scale_{run}.json"
            )
            _atomic_json(candidate_path, payload)
            jobs.append(
                {
                    "job_id": f"postflight_{run}_{role}",
                    "run": run,
                    "role": role,
                    "family": candidate["family"],
                    "variant": variant_name,
                    "grid_order": order + 1,
                    "candidate_id": candidate["candidate_id"],
                }
            )
    manifest = {
        "schema": 1,
        "selection_id": SELECTION_ID,
        "created_utc": _utc_now(),
        "runs": list(runs),
        "jobs": jobs,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _run_postflight_job(
    output_root: Path,
    job: Mapping[str, Any],
    *,
    backend: str,
    diagnostic_workers: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from darsia.fluidflower.tools.calibration_review_movie import (
        render_contour_frame,
        render_mass_plot,
        write_calibration_movie,
        write_movie_provenance,
    )
    from darsia.fluidflower.tools.spatial_mass_evaluator import create_spatial_mass_evaluator

    run = str(job["run"])
    family = str(job["family"])
    spec = FAMILY_SPECS[family]
    variant = replace(spec.variant, name=str(job["variant"]))
    candidate_path = (
        output_root
        / "postflight_input"
        / variant.name
        / run
        / f"final_full_scale_{run}.json"
    )
    payload = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    params = payload.get("params")
    if not isinstance(params, dict) or not params:
        raise RuntimeError(f"Candidate has no parameters: {candidate_path}")

    context = None
    evaluator = None
    records: list[dict[str, float | int | str]] = []
    contour_frames: list[Any] = []
    try:
        environment = screen._spatial_environment_values(run, spec)
        with pilot.campaign_environment(variant, environment):
            context = screen.selection._build_evaluation_context(
                run, variant, times=None
            )
            screen.calibration.apply_params(
                context.calibration,
                params,
                labels=context.signal_labels,
                np_module=np,
            )
            shape = tuple(np.asarray(context.calibration.labels.img).shape)
            voxel_volume = screen.calibration.voxel_volume_with_depth_model(
                context, params, shape
            )
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
                record: dict[str, float | int | str] = {
                    "run": run,
                    "variant": variant.name,
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
                        concentration_aq=np.asarray(spatial.concentration_aq.img),
                        saturation_g=np.asarray(spatial.saturation_g.img),
                        run=run,
                        variant=f"{variant.name} | {job['role']}",
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
            screen._close_context(context)

    run_output = output_root / "postflight" / variant.name / run
    run_output.mkdir(parents=True, exist_ok=True)
    per_frame_csv = run_output / "calibration_review_per_frame.csv"
    with per_frame_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    mass_plot = render_mass_plot(
        records,
        run=run,
        variant=f"{variant.name} | constrained Pareto + untouched holdout",
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
            "variant": variant.name,
            "candidate": str(candidate_path),
            "params_sha256": payload["selection"]["params_sha256"],
            "mass_plot": str(mass_plot_path),
            "per_frame_csv": str(per_frame_csv),
            "diagnostic_workers_requested": max(1, int(diagnostic_workers)),
            "clearwater_reference_used": False,
        }
    )
    write_movie_provenance(run_output / "calibration_review.json", movie)
    maximum_injected = max(float(row["injected_mass_kg"]) for row in records)
    normalized_mae = sum(
        abs(
            float(row["detected_total_mass_kg"])
            - float(row["injected_mass_kg"])
        )
        for row in records
    ) / (len(records) * maximum_injected)
    summary = {
        "schema": 1,
        "run": run,
        "variant": variant.name,
        "objective_full_scale": float(payload["objective_full_scale"]),
        "normalized_mae": float(normalized_mae),
        "frame_count": len(records),
        "calibration_review_movie": movie,
        "selection_id": SELECTION_ID,
        "selection_role": job["role"],
        "selection_candidate_id": job["candidate_id"],
        "grid_order": job["grid_order"],
    }
    summary_path = output_root / "postflight" / variant.name / run / "summary.json"
    _atomic_json(summary_path, summary)
    return summary


def postflight_worker(
    output_root: Path,
    *,
    backend: str,
    diagnostic_workers: int,
    once: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(
        (output_root / "postflight_manifest.json").read_text(encoding="utf-8-sig")
    )
    jobs = list(manifest["jobs"])
    worker_id = f"{socket.gethostname()}_{backend}_{os.getpid()}"
    processed: list[str] = []
    while True:
        job = next(
            (
                item
                for item in jobs
                if _claim_job(
                    output_root, "postflight", str(item["job_id"]), worker_id
                )
            ),
            None,
        )
        if job is None:
            if _all_terminal(output_root, "postflight", jobs) or once:
                break
            time.sleep(3.0)
            continue
        job_id = str(job["job_id"])
        print(f"[{_utc_now()}] starting {job_id}", flush=True)
        try:
            summary = _run_postflight_job(
                output_root,
                job,
                backend=backend,
                diagnostic_workers=diagnostic_workers,
            )
            _atomic_json(
                output_root / "postflight_done" / f"{job_id}.json",
                {
                    "schema": 1,
                    "job": job,
                    "worker_id": worker_id,
                    "completed_utc": _utc_now(),
                    "movie": summary["calibration_review_movie"]["movie"],
                },
            )
            processed.append(job_id)
            print(f"[{_utc_now()}] completed {job_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            _atomic_json(
                output_root / "postflight_failed" / f"{job_id}.json",
                {
                    "schema": 1,
                    "job": job,
                    "worker_id": worker_id,
                    "failed_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{_utc_now()}] FAILED {job_id}: {exc}", flush=True)
        if once:
            break
    return {"worker_id": worker_id, "processed": processed}


def finalize_postflight(output_root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (output_root / "postflight_manifest.json").read_text(encoding="utf-8-sig")
    )
    status = _status_for_jobs(output_root, "postflight", manifest["jobs"])
    if not status["complete"] or status["counts"]["failed"]:
        raise RuntimeError(f"Postflight is not complete and clean: {status['counts']}")
    movies: dict[str, str] = {}
    for run in tuple(str(run) for run in manifest["runs"]):
        entries = grid_movie._discover(
            output_root / "postflight", run, sort_by="grid-order"
        )
        output = output_root / "postflight" / f"{run}_winner_control_grid.mp4"
        grid_movie.create_grid_movie(
            entries,
            output,
            columns=len(entries),
            output_width=3840 if len(entries) > 1 else 1920,
            output_height=1080,
        )
        movies[run] = str(output)
    result = {
        "schema": 1,
        "selection_id": SELECTION_ID,
        "completed_utc": _utc_now(),
        "movies": movies,
    }
    _atomic_json(output_root / "postflight" / "POSTFLIGHT_COMPLETE.json", result)
    return result


def _launch_workers(
    output_root: Path,
    *,
    action: str,
    workers: int,
    backend: str,
    diagnostic_workers: int,
) -> dict[str, Any]:
    local_logs = project_path('logs') / SELECTION_ID / socket.gethostname()
    local_logs.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    processes: list[dict[str, Any]] = []
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    for index in range(max(1, workers)):
        log_path = local_logs / f"{action}_{index}.log"
        stream = log_path.open("a", encoding="utf-8", buffering=1)
        command = [
            str(python),
            str(Path(__file__).resolve()),
            action,
            "--output-root",
            str(output_root),
            "--backend",
            backend,
        ]
        if action == "postflight-worker":
            command.extend(["--diagnostic-workers", str(diagnostic_workers)])
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=process_directory(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )
        stream.close()
        processes.append(
            {
                "index": index,
                "pid": process.pid,
                "action": action,
                "backend": backend,
                "log": str(log_path),
                "command": command,
            }
        )
    payload = {
        "schema": 1,
        "selection_id": SELECTION_ID,
        "host": socket.gethostname(),
        "started_utc": _utc_now(),
        "action": action,
        "processes": processes,
    }
    _atomic_json(local_logs / f"launcher_{action}.json", payload)
    _atomic_json(output_root / f"launcher_{action}_{socket.gethostname()}.json", payload)
    return payload


def stop_workers() -> dict[str, Any]:
    import psutil

    local_logs = project_path('logs') / SELECTION_ID / socket.gethostname()
    manifests = sorted(local_logs.glob("launcher_*.json"))
    stopped: list[int] = []
    missing: list[int] = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        for row in payload.get("processes") or []:
            pid = int(row["pid"])
            if not psutil.pid_exists(pid):
                missing.append(pid)
                continue
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for process in reversed(children):
                process.terminate()
            parent.terminate()
            _gone, alive = psutil.wait_procs([*children, parent], timeout=5.0)
            for process in alive:
                process.kill()
            stopped.append(pid)
    return {"status": "stopped", "stopped": stopped, "already_missing": missing}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "holdout-worker",
            "launch-holdout",
            "finalize",
            "prepare-postflight",
            "postflight-worker",
            "launch-postflight",
            "finalize-postflight",
            "status",
            "stop",
        ),
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", choices=("cuda", "prepared"), default="cuda")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--diagnostic-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        result = prepare_selection(
            args.source_root, args.output_root, force=args.force
        )
    elif args.action == "holdout-worker":
        result = holdout_worker(
            args.output_root, backend=args.backend, once=args.once
        )
    elif args.action == "launch-holdout":
        prepare_selection(args.source_root, args.output_root, force=False)
        result = _launch_workers(
            args.output_root,
            action="holdout-worker",
            workers=args.workers,
            backend=args.backend,
            diagnostic_workers=args.diagnostic_workers,
        )
    elif args.action == "finalize":
        result = finalize_selection(args.output_root)
    elif args.action == "prepare-postflight":
        result = prepare_postflight(args.output_root, force=args.force)
    elif args.action == "postflight-worker":
        result = postflight_worker(
            args.output_root,
            backend=args.backend,
            diagnostic_workers=args.diagnostic_workers,
            once=args.once,
        )
    elif args.action == "launch-postflight":
        prepare_postflight(args.output_root, force=False)
        result = _launch_workers(
            args.output_root,
            action="postflight-worker",
            workers=args.workers,
            backend=args.backend,
            diagnostic_workers=args.diagnostic_workers,
        )
    elif args.action == "finalize-postflight":
        result = finalize_postflight(args.output_root)
    elif args.action == "stop":
        result = stop_workers()
    else:
        result = selection_status(args.output_root)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
