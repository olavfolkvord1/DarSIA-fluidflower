"""Prepare and predict every discoverable experiment in one model's series.

The selected model stays frozen. Each experiment has its own prepared images,
physical context, transfer evaluation and independently verified review result.
The status file is stable across invocations so an interrupted series can resume.
"""
from __future__ import annotations
from darsia.fluidflower.series_paths import tool_arguments

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


from darsia.fluidflower.execution import default_execution_cache_root
from darsia.fluidflower.phase_validation import FRAME_TIMING_PREFIX, _atomic_json, _model_identity
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools import phase_transfer_review as transfer


EVENT_PREFIX = "PHASE_SERIES_BATCH_EVENT "
SERIES = ("ffac", "ffum", "ffbm")


def _slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-")[:72]


def batch_status_path(model: str | Path, series: str, output_parent: str | Path) -> Path:
    """Return the same status location for one exact model and series."""
    if series not in SERIES:
        raise ValueError(f"Unknown experiment series: {series}")
    model_id, model_sha = _model_identity(Path(model).resolve())
    return (Path(output_parent).resolve() /
            f"series-batch-{series}-{_slug(model_id)}-{model_sha[:16]}" / "status.json")


def _event(stage: str, status_path: Path, *, run: str = "", index: int = 0,
           total: int = 0, counts: dict | None = None, **extra: object) -> None:
    print(EVENT_PREFIX + json.dumps(dict(stage=stage, status_path=str(status_path),
        run=run, index=index, total=total, counts=counts or {}, **extra),
        ensure_ascii=False, allow_nan=False), flush=True)


@contextmanager
def _batch_lock(status_path: Path):
    """An OS-held lock releases even when an unattended process is killed."""
    lock_path = status_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("A series batch for this model is already running") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("A series batch for this model is already running") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _counts(rows: dict[str, dict]) -> dict[str, int]:
    states = [row.get("state") for row in rows.values()]
    reviewed = sum(row.get("state") == "complete" and row.get("review_status") == "reviewed"
                   for row in rows.values())
    return dict(completed=states.count("complete"), failed=states.count("failed"),
        pending=len(states) - states.count("complete") - states.count("failed"),
        generated=sum(bool(row.get("generated_this_run")) for row in rows.values()),
        skipped=sum(bool(row.get("skipped_this_run")) for row in rows.values()),
        needs_preparation=sum(row.get("preparation_status") == "needs_preparation" for row in rows.values()),
        manually_reviewed=reviewed, needs_manual_review=states.count("complete") - reviewed)


def _write_status(status_path: Path, status: dict) -> None:
    status["updated_utc"] = utc_now()
    status["counts"] = _counts(status["rows"])
    status["runs"] = status["rows"]
    status["total"] = len(status["rows"])
    for name in ("completed", "failed", "pending"):
        status[name] = status["counts"][name]
    # Studio polls this file on Windows. A reader or antivirus scanner can
    # briefly hold the destination open and make the atomic replace fail.
    for attempt in range(20):
        try:
            _atomic_json(status_path, status)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def _prepared_target_root(model_id: str, model_sha: str, series: str, run: str) -> Path:
    parent = default_execution_cache_root() / "phase_transfer_targets"
    return parent / f"series-batch-{series}-{_slug(model_id)}-{model_sha[:12]}-{run}"


def _fresh_output(base: Path) -> Path:
    if not base.exists():
        return base
    return base.with_name(f"{base.name}-retry-{uuid.uuid4().hex[:8]}")


def _prepared_manifest(path: Path, series: str, run: str) -> Path | None:
    manifest = path / "postflight_manifest.json"
    if not manifest.is_file():
        return None
    try:
        report = _read(path / "preparation_report.json")
        matches = [row for row in transfer.target_sources(manifest, series) if row["run"] == run]
        if report.get("status") == "prepared_target_inputs_ready" and len(matches) == 1:
            return manifest
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _ensure_reference_alignment(manifest_path: Path, series: str, run: str) -> None:
    """Measure the display-only baseline alignment this target never measured.

    prepare_ffum_phase_target publishes corrected review frames without measuring how far
    the rig moved between the pre-injection baseline and each later photograph, so the
    review window cannot place the aligned reference and keeps its switch off. The
    measurement is kept beside the reference photograph, never inside the hashed native
    manifest, so an evaluation that already exists stays verifiable. A target that cannot
    be measured keeps the switch off, exactly as it does today.
    """
    try:
        from darsia.fluidflower.tools.phase_reference_registration import ensure

        job = next((row for row in _read(manifest_path).get("jobs", [])
                    if str(row.get("run") or "").casefold() == run.casefold()), None)
        if job is None:
            return
        root = Path(job.get("analysis_source_review_root") or job["review_output_dir"])
        ensure(root, run, progress=lambda text: print(text, flush=True))
    except Exception as error:  # noqa: BLE001 - display only; never fails a batch
        print(f"{run.upper()}: reference alignment unavailable: "
              f"{type(error).__name__}: {error}", flush=True)


def _prepare_target(model_id: str, model_sha: str, series: str, run: str) -> Path:
    base = _prepared_target_root(model_id, model_sha, series, run)
    existing = _prepared_manifest(base, series, run)
    if existing:
        _ensure_reference_alignment(existing, series, run)
        return existing
    # A previous failure may leave partial input. Keep it for inspection and
    # retry in a fresh folder instead of modifying any earlier campaign.
    output = _fresh_output(base)
    if series == "ffbm":
        from darsia.fluidflower.tools.prepare_bm_phase_target import prepare
        prepare(run, output)
    else:
        from darsia.fluidflower.tools.prepare_ffum_phase_target import prepare
        prepare(run, output, dataset=series)
    verified = _prepared_manifest(output, series, run)
    if not verified:
        raise ValueError(f"{run.upper()}: image preparation did not publish a complete target")
    _ensure_reference_alignment(verified, series, run)
    return verified


def _preferred_row(rows: list[dict], run: str, *, source_manifest: Path | None = None) -> dict | None:
    matches = [row for row in rows if row.get("run") == run
        and row.get("preparation_status", "ready") == "ready"]
    if source_manifest is not None:
        matches = [row for row in matches if Path(row["source_manifest"]).resolve() == source_manifest.resolve()]
    if not matches:
        return None
    modes = {"registered-target": 0, "prepared-target": 0, "optical-only": 1}
    def modified(row: dict) -> float:
        try:
            return Path(row["source_manifest"]).stat().st_mtime
        except (OSError, TypeError, ValueError):
            return 0.
    return min(matches, key=lambda row: (modes.get(row.get("physics_mode"), 2),
        -modified(row),
        not bool((row.get("completed_evaluations") or {}).get(row.get("physics_mode"))),
        str(row.get("source_manifest"))))


def _reusable_prepared_row(rows: list[dict], series: str, run: str) -> dict | None:
    """Use only complete, unambiguous full-physics sources accepted by discovery."""
    usable = []
    for row in rows:
        if (row.get("dataset") != series or row.get("run") != run
                or row.get("preparation_status") != "ready"
                or row.get("physics_mode") not in {"registered-target", "prepared-target"}
                or not row.get("source_manifest") or not row.get("native_manifest_sha256")):
            continue
        try:
            targets = [target for target in transfer.target_sources(row["source_manifest"], series)
                       if target["run"] == run]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        # prepare_transfer_review requires exactly one native revision in the
        # selected campaign. Check it before treating an old source as reusable.
        if {target["native_manifest_sha256"] for target in targets} == {row["native_manifest_sha256"]}:
            usable.append(row)
    return _preferred_row(usable, run)


def _resumable_plan(evaluation_root: str | Path | None, evaluation_parent: Path,
                    model: Path, series: str, row: dict) -> dict | None:
    """Recover an unfinished evaluation only when every frozen input still matches."""
    if not evaluation_root:
        return None
    try:
        root = Path(evaluation_root).resolve()
        if root.parent != evaluation_parent.resolve() or (root / "transfer_result.json").exists():
            return None
        plan = _read(root / "transfer_review.json")
        model_id, model_sha = _model_identity(model)
        if (Path(plan["evaluation_root"]).resolve() != root
                or Path(plan["source_model"]).resolve() != model.resolve()
                or (plan["source_model_id"], plan["source_model_sha256"],
                    plan["dataset"], plan["run"], plan["physics_mode"], plan["review_mode"])
                != (model_id, model_sha, series, row["run"], row["physics_mode"], "visible")
                or Path(plan["source_manifest"]).resolve() != Path(row["source_manifest"]).resolve()
                or Path(plan["target_native_manifest"]).resolve() != Path(row["native_manifest"]).resolve()):
            return None
        native_path = str(Path(row["native_manifest"]).resolve())
        if plan["protected_sha256"].get(native_path) != row["native_manifest_sha256"]:
            return None
        transfer.verify_hashes(plan["protected_sha256"])
        transfer.verify_hashes(plan["evaluation_sha256"])
        if (transfer._model_identity(root) != (plan["evaluation_model_id"], plan["evaluation_model_sha256"])
                or transfer.physics_runtime_identity(transfer.load_physics_profile(root))
                != plan["physics_runtime"]):
            return None
        return plan
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _verify_result(path: Path, model: Path, series: str, row: dict) -> dict:
    """Verify a reusable result against its original model and current images."""
    path = path.resolve()
    result = _read(path)
    root = path.parent
    plan = _read(root / "transfer_review.json")
    binding = _read(root / "validation" / "validation_binding.json")
    model_id, model_sha = _model_identity(model)
    if (plan["source_model_id"], plan["source_model_sha256"], plan["dataset"], plan["run"],
            plan["physics_mode"]) != (model_id, model_sha, series, row["run"], row["physics_mode"]):
        raise ValueError("Saved evaluation has another model, series, run or physics mode")
    native_path = str(Path(plan["target_native_manifest"]).resolve())
    if (plan["protected_sha256"].get(native_path) != row["native_manifest_sha256"]
            or Path(plan["source_manifest"]).resolve() != Path(row["source_manifest"]).resolve()):
        raise ValueError("Saved evaluation belongs to another target image revision")
    saved_root = result.get("evaluation_root")
    if (result.get("review_mode") != "visible" or not isinstance(saved_root, str)
            or Path(saved_root).resolve() != root
            or result.get("target_native_manifest_sha256") != row["native_manifest_sha256"]):
        raise ValueError("Saved result is not a visible review for this target")
    transfer.verify_hashes(plan["protected_sha256"])
    transfer.validate_review_binding(plan, binding)
    receipt = _read(root / "validation" / "transfer_materialization.json")
    if receipt.get("evaluation_model_sha256") != plan["evaluation_model_sha256"] or receipt.get("run") != row["run"]:
        raise ValueError("Saved materialization receipt identifies another target")
    for name, digest in receipt["generated_sha256"].items():
        asset = (root / "validation" / name).resolve()
        if not asset.is_relative_to((root / "validation").resolve()) or sha256_file(asset) != digest:
            raise ValueError(f"Saved phase-map asset changed: {name}")
    return result


def _make_result(plan: dict, binding: dict, model: Path, row: dict, *, reviewer: str,
                 save_root: Path, cache_root: Path) -> dict:
    options = dict(reviewer=reviewer, save_root=save_root, cache_root=cache_root)
    command = transfer.visible_review_command(plan, binding, **options)
    result = dict(stage="ready", evaluation_root=plan["evaluation_root"], binding=binding,
        completed_utc=utc_now(), model_root=str(model), model_id=plan["source_model_id"],
        source_model_sha256=plan["source_model_sha256"], target_run=plan["run"],
        target_native_manifest_sha256=row["native_manifest_sha256"], command=command,
        correction_command=None, reopen_command=command, review_mode="visible",
        training_seen=plan["training_seen"], label=plan["label"], physics_mode=plan["physics_mode"],
        **{name: plan[name] for name in ("optical_training_status", "calibration_evidence_status",
           "exposure_label", "calibration_history")})
    result_path = Path(plan["evaluation_root"]) / "transfer_result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    transfer.write(result_path, result)
    return result


def select_requested_runs(runs, requested_runs=None, excluded_runs=()):
    requested = None if requested_runs is None else {str(r).casefold() for r in requested_runs}
    if requested is not None and requested-set(runs):
        raise ValueError("Requested experiments are missing: "+", ".join(sorted(requested-set(runs))))
    excluded={str(r).casefold() for r in excluded_runs}
    return [r for r in runs if (requested is None or r in requested) and r not in excluded]


def _reprepare_selection(reprepare_runs, runs: list[str], raw_runs: set[str]) -> set[str]:
    """Names of experiments that must be prepared again from their own raw target.

    Both names are checked loudly: a typo, or an experiment without raw photographs, would
    otherwise silently leave the broken evaluation in place.
    """
    reprepare = {str(run).casefold() for run in reprepare_runs or ()}
    outside = sorted(reprepare - set(runs))
    if outside:
        raise ValueError("Re-preparation names experiments outside this selection: "
                         + ", ".join(outside))
    without_raw = sorted(reprepare - set(raw_runs))
    if without_raw:
        raise ValueError("Re-preparation needs raw photographs, missing for: "
                         + ", ".join(without_raw))
    return reprepare


def run_batch(model: Path, series: str, output_parent: Path, *, reviewer: str,
              save_root: Path, cache_root: Path, backend: str = "auto",
              review_roots: list[Path] | None = None,
              source_manifests: list[Path] | None = None, dry_run: bool = False,
              only_run: str | None = None, worker_token: str | None = None,
              requested_runs=None, excluded_runs=(), reprepare_runs=()) -> dict:
    model, output_parent = Path(model).resolve(), Path(output_parent).resolve()
    evaluation_parent = default_execution_cache_root().resolve() / "phase-transfer-review"
    if series not in SERIES or backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("Choose a supported series and inference backend")
    if transfer.model_training_scope(model)[0] != series:
        raise ValueError("The selected frozen model is not a single-series model for this series")
    model_id, model_sha = _model_identity(model)
    status_path = batch_status_path(model, series, output_parent)
    discovery = transfer.transfer_discovery(model, source_manifests or (), review_roots or ())
    prepared = [row for row in discovery["sources"] if row.get("preparation_status") == "ready"]
    raw = transfer.discover_raw_series_runs(series)
    raw_runs = {row["run"] for row in raw}
    runs = sorted({row["run"] for row in [*raw, *prepared]},
        key=lambda run: (int(re.search(r"\d+", run).group()) if re.search(r"\d+", run) else 10**9, run))
    runs = select_requested_runs(runs, requested_runs, excluded_runs)
    if only_run is not None:
        only_run = only_run.casefold()
        if only_run not in runs:
            raise ValueError(f"{only_run}: run is not in the selected series inventory")
        runs = [only_run]
    if not runs:
        raise ValueError("No same-series experiments were found in raw or prepared sources")
    reprepare = _reprepare_selection(reprepare_runs, runs, raw_runs)
    if dry_run:
        counts = dict(completed=0, failed=0, pending=len(runs), generated=0, skipped=0,
            manually_reviewed=0, needs_manual_review=0,
            needs_preparation=sum(run in reprepare
                                  or (_reusable_prepared_row(prepared, series, run) if run in raw_runs
                                      else _preferred_row(prepared, run)) is None
                                  for run in runs))
        _event("started", status_path, total=len(runs), counts=counts, dry_run=True)
        _event("complete", status_path, index=len(runs), total=len(runs), counts=counts,
            dry_run=True, success=True, planned_runs=runs,
            raw_inventory_available=Path(transfer.RAW_SERIES_ROOTS[series]).is_dir())
        return dict(dry_run=True, runs=runs, counts=counts, status_path=str(status_path))
    with _batch_lock(status_path):
        previous = _read(status_path) if status_path.is_file() else {}
        if previous and (previous.get("model_id"), previous.get("model_sha256"), previous.get("series")) != (model_id, model_sha, series):
            raise ValueError("Existing batch status belongs to another model or series")
        prior_rows = previous.get("rows") or {}
        rows = {run: dict(prior_rows.get(run) or {}) for run in runs}
        for run in runs:
            chosen = (None if run in reprepare
                      else _reusable_prepared_row(prepared, series, run) if run in raw_runs
                      else _preferred_row(prepared, run))
            rows[run].update(state="pending", preparation_status="ready" if chosen else "needs_preparation")
            rows[run].pop("generated_this_run", None)
            rows[run].pop("skipped_this_run", None)
            rows[run].pop("resumed_this_run", None)
        status = dict(format="phase-series-batch-v1", model_root=str(model), model_id=model_id,
            model_sha256=model_sha, series=series, output_parent=str(output_parent),
            evaluation_parent=str(evaluation_parent), reprepare_runs=sorted(reprepare),
            status_path=str(status_path), pid=os.getpid(), state="running", stage="started",
            backend=backend, worker_token=worker_token,
            started_utc=previous.get("started_utc") or utc_now(), invocation_started_utc=utc_now(),
            raw_inventory_available=Path(transfer.RAW_SERIES_ROOTS[series]).is_dir(), rows=rows)
        _write_status(status_path, status)
        _event("started", status_path, total=len(runs), counts=status["counts"], model_id=model_id,
            model_sha256=model_sha)
        for index, run in enumerate(runs, 1):
            item = rows[run]
            item.update(state="running", started_utc=utc_now(), error=None)
            status["stage"] = "run_started"
            status["current_run"] = run
            _write_status(status_path, status)
            _event("run_started", status_path, run=run, index=index, total=len(runs), counts=status["counts"])
            try:
                if run in raw_runs:
                    # A named experiment ignores reusable sources so its own target is rebuilt.
                    row = None if run in reprepare else _reusable_prepared_row(prepared, series, run)
                    if row is not None:
                        item["source_policy"] = "existing_prepared_source"
                    else:
                        item["source_policy"] = "series_preparer"
                        preparation_started = time.perf_counter()
                        manifest = _prepare_target(model_id, model_sha, series, run)
                        # Raw photograph -> registered, corrected native frame, for the whole run.
                        item["image_preparation_s"] = round(time.perf_counter() - preparation_started, 1)
                        rows_now = transfer.transfer_discovery(model, [manifest, *(source_manifests or ())],
                            review_roots or ())["sources"]
                        row = _preferred_row(rows_now, run, source_manifest=manifest)
                        if row is None:
                            raise ValueError(f"{run.upper()}: prepared images were not discoverable")
                        if row["physics_mode"] == "optical-only":
                            raise ValueError(f"{run.upper()}: own prepared physical context was not accepted by the model")
                else:
                    item["source_policy"] = "prepared_only_fallback"
                    row = _preferred_row(prepared, run)
                    if row is None:
                        raise ValueError(f"{run.upper()}: no prepared source remains available")
                item.update(preparation_status="ready", source_manifest=row["source_manifest"],
                    native_manifest_sha256=row["native_manifest_sha256"], physics_mode=row["physics_mode"],
                    frame_count=row.get("frame_count"), review_status=row.get("review_status"))
                result_path = Path(str(item.get("result_path") or ""))
                result = None
                if item.get("result_path") and result_path.is_file():
                    try:
                        result = _verify_result(result_path, model, series, row)
                    except (OSError, ValueError, KeyError, TypeError):
                        result = None
                if result is None:
                    prior = (row.get("completed_evaluations") or {}).get(row["physics_mode"])
                    if prior and prior.get("evaluation_root"):
                        candidate = Path(prior["evaluation_root"]) / "transfer_result.json"
                        try:
                            result = _verify_result(candidate, model, series, row)
                            result_path = candidate
                        except (OSError, ValueError, KeyError, TypeError):
                            result = None
                if run in reprepare:
                    # Verification asks only whether the saved result still matches its own
                    # inputs, so an experiment already rebuilt from its own target verifies
                    # against itself. Naming it must mean "produce it again", as the flag says.
                    result = None
                if result is not None:
                    item.update(state="complete", result_path=str(result_path),
                        evaluation_root=result["evaluation_root"], skipped_this_run=True,
                        completed_utc=utc_now())
                    _write_status(status_path, status)
                    _event("run_skipped", status_path, run=run, index=index, total=len(runs),
                        counts=status["counts"], physics_mode=row["physics_mode"],
                        result_path=str(result_path), reason="verified_existing_result")
                    continue
                plan = (None if run in reprepare else
                        _resumable_plan(item.get("evaluation_root"), evaluation_parent,
                                        model, series, row))
                if plan is None:
                    plan = transfer.prepare_transfer_review(model, run, row["source_manifest"],
                        evaluation_parent, physics_mode=row["physics_mode"], review_mode="visible")
                else:
                    item["resumed_this_run"] = True
                item["evaluation_root"] = plan["evaluation_root"]
                _write_status(status_path, status)
                def progress(message: str) -> None:
                    text = str(message)
                    if text.startswith(FRAME_TIMING_PREFIX):
                        # Measured seconds per image; Studio shows them beside the
                        # progress text, never instead of it.
                        try:
                            item["frame_timing"] = json.loads(text[len(FRAME_TIMING_PREFIX):])
                        except ValueError:
                            return
                    else:
                        item["progress_message"] = text
                    _write_status(status_path, status)
                progress.accepts_frame_timing = True
                binding = transfer.materialize_transfer_review(plan, backend=backend, progress=progress)
                result = _make_result(plan, binding, model, row, reviewer=reviewer,
                    save_root=save_root, cache_root=cache_root)
                result_path = Path(result["evaluation_root"]) / "transfer_result.json"
                _verify_result(result_path, model, series, row)
                item.update(state="complete", result_path=str(result_path),
                    generated_this_run=True, completed_utc=utc_now())
                _write_status(status_path, status)
                _event("run_completed", status_path, run=run, index=index, total=len(runs),
                    counts=status["counts"], physics_mode=row["physics_mode"], result_path=str(result_path))
            except Exception as error:  # noqa: BLE001 - independent runs must continue
                item.update(state="failed", error=f"{type(error).__name__}: {error}", failed_utc=utc_now())
                _write_status(status_path, status)
                _event("run_failed", status_path, run=run, index=index, total=len(runs),
                    counts=status["counts"], error=item["error"])
        status["state"] = "complete"
        status["stage"] = "complete"
        status["current_run"] = None
        status["completed_utc"] = utc_now()
        status["success"] = _counts(rows)["failed"] == 0 and _counts(rows)["pending"] == 0
        _write_status(status_path, status)
        _event("complete", status_path, index=len(runs), total=len(runs), counts=status["counts"],
            success=status["counts"]["failed"] == 0 and status["counts"]["pending"] == 0,
            raw_inventory_available=status["raw_inventory_available"])
        return status


def _available_memory_gib() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except ImportError:
        return None


def _worker_rss_gib(process) -> float:
    """Memory already charged to available RAM, including a Windows launcher child."""
    try:
        import psutil
        parent = psutil.Process(process.pid)
        return sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)]) / 2**30
    except Exception:  # Missing/exited process: reserve the full allowance.
        return 0.0


def _worker_command(script: Path, model: Path, series: str, run: str, worker_parent: Path,
                    *, reviewer: str, save_root: Path, cache_root: Path, backend: str,
                    review_roots: list[Path], source_manifests: list[Path],
                    worker_token: str, reprepare: bool = False) -> list[str]:
    command = [sys.executable, *tool_arguments(script), "--model", str(model), "--series", series,
               "--only-run", run, "--output-parent", str(worker_parent),
               "--reviewer", reviewer, "--save-root", str(save_root),
               "--cache-root", str(cache_root), "--backend", backend,
               "--worker-token", worker_token]
    command += [argument for root in review_roots for argument in ("--review-root", str(root))]
    command += [argument for path in source_manifests
                for argument in ("--source-manifest", str(path))]
    if reprepare:
        # This experiment ignores every reusable prepared source and starts from its own target.
        command += ["--reprepare-runs", run]
    return command


class _ExistingWorker:
    """Monitor a per-run process that outlived a killed batch coordinator."""

    def __init__(self, process):
        self._process = process
        self.pid = process.pid

    def poll(self):
        return None if self._process.is_running() else 0


def _adopt_running_worker(path: Path, *, model_id: str, model_sha: str,
                          series: str, run: str):
    if not path.is_file():
        return None
    saved = _read(path)
    if saved.get("state") != "running":
        return None
    if (saved.get("model_id"), saved.get("model_sha256"), saved.get("series")) != (
            model_id, model_sha, series):
        raise RuntimeError(f"{run}: running worker status has another frozen model")
    token, pid = saved.get("worker_token"), saved.get("pid")
    if not isinstance(token, str) or not token or not isinstance(pid, int):
        raise RuntimeError(f"{run}: cannot safely identify previous worker")
    try:
        import psutil
        process = psutil.Process(pid)
        if not process.is_running():
            return None
        args = process.cmdline()
        if (not any(Path(arg).name == "phase_series_batch.py" for arg in args)
                or "--only-run" not in args
                or args[args.index("--only-run") + 1].casefold() != run):
            raise RuntimeError(f"{run}: previous PID now belongs to another process")
        started = datetime.fromisoformat(saved["invocation_started_utc"].replace("Z", "+00:00")).timestamp()
        if abs(process.create_time() - started) > 120:
            raise RuntimeError(f"{run}: previous PID creation time does not match worker status")
        return _ExistingWorker(process), saved
    except psutil.NoSuchProcess:
        return None
    except (psutil.AccessDenied, OSError, ValueError, KeyError, IndexError) as error:
        raise RuntimeError(f"{run}: previous worker could not be verified: {error}") from error


def run_parallel_batch(model: Path, series: str, output_parent: Path, *, reviewer: str,
                       save_root: Path, cache_root: Path, backend: str = "auto",
                       review_roots: list[Path] | None = None,
                       source_manifests: list[Path] | None = None,
                       dry_run: bool = False, max_workers: int = 2, gpu_workers: int = 0,
                       worker_memory_gib: float = 24., free_headroom_gib: float = 48.,
                       requested_runs=None, excluded_runs=(), frame_broker=None,
                       reprepare_runs=()) -> dict:
    """Schedule independent runs; each worker uses the serial, verified path.

    No parallelism occurs within a run, so temporal physics is unchanged.
    The aggregate status retains the schema consumed by Phase Studio. Worker
    status/logs are isolated per run and can be reused after interruption.
    """
    if dry_run:
        return run_batch(model, series, output_parent, reviewer=reviewer,
            save_root=save_root, cache_root=cache_root, backend=backend,
            review_roots=review_roots, source_manifests=source_manifests, dry_run=True,
            requested_runs=requested_runs, excluded_runs=excluded_runs,
            reprepare_runs=reprepare_runs)
    if not 0 <= max_workers <= 14 or gpu_workers not in (0, 1) or max_workers + gpu_workers < 1:
        raise ValueError("Choose 0-14 CPU workers and 0-1 GPU workers, with at least one worker")
    if backend not in {"auto", "cpu", "cuda"} or series not in SERIES:
        raise ValueError("Choose a supported series and backend")
    if backend == "cuda" and max_workers:
        raise ValueError("Explicit CUDA backend cannot be combined with CPU workers")
    if backend == "cpu" and gpu_workers:
        raise ValueError("Explicit CPU backend cannot be combined with a GPU worker")
    # The legacy per-run scheduler reserves a large allowance because each
    # worker owns its full-resolution frame pipeline.  A shared frame broker
    # owns those allocations instead and admits individual frame tasks against
    # its live RSS/headroom budget.  Applying the legacy 16-GiB floor here
    # makes the broker-backed path fail before it can start (the broker uses an
    # 8-GiB coordinator allowance by design).
    minimum_worker_memory = 4. if frame_broker is not None else 16.
    if worker_memory_gib < minimum_worker_memory or free_headroom_gib < 8:
        raise ValueError("Memory reservations are too small for native full-resolution runs")
    model, output_parent = Path(model).resolve(), Path(output_parent).resolve()
    if transfer.model_training_scope(model)[0] != series:
        raise ValueError("The frozen model is not for this single series")
    model_id, model_sha = _model_identity(model)
    status_path = batch_status_path(model, series, output_parent)
    discovery = transfer.transfer_discovery(model, source_manifests or (), review_roots or ())
    prepared = [row for row in discovery["sources"] if row.get("preparation_status") == "ready"]
    raw = transfer.discover_raw_series_runs(series)
    runs = sorted({row["run"] for row in [*raw, *prepared]},
        key=lambda run: (int(re.search(r"\d+", run).group()) if re.search(r"\d+", run) else 10**9, run))
    runs = select_requested_runs(runs, requested_runs, excluded_runs)
    if not runs and excluded_runs:
        return run_batch(model,series,output_parent,reviewer=reviewer,save_root=save_root,
            cache_root=cache_root,backend=backend,review_roots=review_roots,source_manifests=source_manifests,
            requested_runs=requested_runs,excluded_runs=excluded_runs,reprepare_runs=reprepare_runs)
    if not runs and not excluded_runs:
        raise ValueError("No same-series experiments were found")
    reprepare = _reprepare_selection(reprepare_runs, runs, {row["run"] for row in raw})
    with _batch_lock(status_path):
        previous = _read(status_path) if status_path.is_file() else {}
        if previous and (previous.get("model_id"), previous.get("model_sha256"),
                         previous.get("series")) != (model_id, model_sha, series):
            raise ValueError("Existing batch status belongs to another model or series")
        prior_rows = previous.get("rows") or {}
        rows = {run: dict(prior_rows.get(run) or {}) for run in runs}
        raw_runs = {row["run"] for row in raw}
        for run in runs:
            chosen = (None if run in reprepare
                      else _reusable_prepared_row(prepared, series, run) if run in raw_runs
                      else _preferred_row(prepared, run))
            rows[run].update(state="pending", preparation_status="ready" if chosen else "needs_preparation")
            for field in ("generated_this_run", "skipped_this_run", "resumed_this_run"):
                rows[run].pop(field, None)
        status = dict(format="phase-series-batch-v1", model_root=str(model), model_id=model_id,
            model_sha256=model_sha, series=series, output_parent=str(output_parent),
            evaluation_parent=str(default_execution_cache_root().resolve() / "phase-transfer-review"),
            status_path=str(status_path), pid=os.getpid(), state="running", stage="started",
            started_utc=previous.get("started_utc") or utc_now(), invocation_started_utc=utc_now(),
            raw_inventory_available=Path(transfer.RAW_SERIES_ROOTS[series]).is_dir(),
            parallel_workers=max_workers + gpu_workers, cpu_workers=max_workers,
            gpu_workers=gpu_workers,
            backend="mixed" if max_workers and gpu_workers else "cuda" if gpu_workers else "cpu",
            rows=rows)
        if frame_broker is not None:
            status.update(parallel_workers=max(1, frame_broker.cpu_workers) + frame_broker.gpu_workers,
                cpu_workers=frame_broker.cpu_workers, gpu_workers=frame_broker.gpu_workers,
                backend="shared-frame-workers", frame_workers=frame_broker.snapshot())
        _write_status(status_path, status)
        _event("started", status_path, total=len(runs), counts=status["counts"],
               model_id=model_id, model_sha256=model_sha)
        active: dict[str, dict] = {}
        queue = list(runs)
        # A killed coordinator must not start a second copy of a still-running
        # experiment. Adopt only an exact, verifiable child; otherwise abort.
        for run in list(queue):
            child_path = batch_status_path(model, series, status_path.parent / "workers" / run)
            adopted = _adopt_running_worker(child_path, model_id=model_id,
                                             model_sha=model_sha, series=series, run=run)
            if adopted is None:
                continue
            process, child = adopted
            queue.remove(run)
            active[run] = dict(process=process, status_path=child_path,
                backend=child["backend"], log_path=status_path.parent / "worker-logs" / f"{run}.log",
                token=child["worker_token"], adopted=True)
            rows[run].update((child.get("rows") or {}).get(run) or {})
            rows[run]["worker_pid"] = process.pid
            rows[run]["worker_backend"] = child["backend"]
        if active:
            status["current_runs"] = list(active)
            status["current_run"] = next(iter(active))
            _write_status(status_path, status)
        try:
            while queue or active:
                if frame_broker is not None:
                    max_workers = min(4, max(1, frame_broker.cpu_workers // 2))
                # The reservation covers a worker before its allocations show
                # up in system memory. If monitoring is unavailable, serialize.
                while queue and len(active) < max_workers + gpu_workers:
                    active_cpu = sum(worker["backend"] == "cpu" for worker in active.values())
                    active_gpu = sum(worker["backend"] == "cuda" for worker in active.values())
                    if active_gpu < gpu_workers:
                        worker_backend = "cuda"
                    elif active_cpu < max_workers:
                        worker_backend = "cpu"
                    else:
                        break
                    available = _available_memory_gib()
                    if available is None and active:
                        break
                    # Available RAM already excludes resident worker memory.
                    # Reserve only the still-unallocated part of each worker's
                    # allowance, rather than counting its RSS twice.
                    reserved = sum(max(0., worker_memory_gib - _worker_rss_gib(worker["process"]))
                                   for worker in active.values())
                    if available is not None and (available - reserved
                                                  < worker_memory_gib + free_headroom_gib):
                        if active:
                            break
                        raise MemoryError("Insufficient free RAM for one full-resolution worker")
                    run = queue.pop(0)
                    worker_parent = status_path.parent / "workers" / run
                    worker_status = batch_status_path(model, series, worker_parent)
                    worker_token = uuid.uuid4().hex
                    command = _worker_command(Path(__file__).resolve(), model, series, run,
                        worker_parent, reviewer=reviewer, save_root=save_root, cache_root=cache_root,
                        backend="auto" if frame_broker is not None else worker_backend,
                        review_roots=review_roots or [],
                        source_manifests=source_manifests or [], worker_token=worker_token,
                        reprepare=run in reprepare)
                    log_path = status_path.parent / "worker-logs" / f"{run}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    env.update(PHASE_BATCH_WORKER="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
                    if frame_broker is not None:
                        env.update(frame_broker.connection_env)
                    with log_path.open("ab") as log:
                        process = subprocess.Popen(command, cwd=str(process_directory()),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
                    active[run] = dict(process=process, status_path=worker_status,
                                       backend=worker_backend, log_path=log_path,
                                       token=worker_token,
                                       adopted=False)
                    rows[run].update(state="running", started_utc=utc_now(), error=None,
                                     worker_pid=process.pid, worker_backend=worker_backend)
                    status["stage"] = "run_started"
                    status["current_run"] = next(iter(active))
                    status["current_runs"] = list(active)
                    _write_status(status_path, status)
                    _event("run_started", status_path, run=run, index=runs.index(run) + 1,
                           total=len(runs), counts=status["counts"], backend=worker_backend)
                changed = False
                for run, worker in list(active.items()):
                    process = worker["process"]
                    child_status = worker["status_path"]
                    child = None
                    if child_status.is_file():
                        try:
                            candidate = _read(child_status)
                            if candidate.get("worker_token") == worker["token"]:
                                child = candidate
                        except (OSError, ValueError, TypeError):
                            pass
                    if child and run in (child.get("rows") or {}):
                        update = child["rows"][run]
                        if update != rows[run]:
                            rows[run].update(update)
                            rows[run]["worker_pid"] = process.pid
                            rows[run]["worker_backend"] = worker["backend"]
                            changed = True
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                    if child and child.get("state") == "running" and child.get("pid") != process.pid:
                        # Windows virtual-environment launchers may exit while
                        # the actual Python worker continues under another PID.
                        adopted = _adopt_running_worker(child_status, model_id=model_id,
                                                         model_sha=model_sha, series=series, run=run)
                        if adopted is not None:
                            worker["process"] = adopted[0]
                            worker["adopted"] = True
                            rows[run]["worker_pid"] = adopted[0].pid
                            changed = True
                            continue
                    if not child or child.get("state") != "complete" or rows[run].get("state") not in {"complete", "failed"}:
                        rows[run].update(state="failed", failed_utc=utc_now(),
                            error=f"Worker exited {exit_code} without a complete per-run status; see {worker['log_path']}")
                    if frame_broker is not None and rows[run].get("evaluation_root"):
                        frame_broker.cancel_owner(rows[run]["evaluation_root"])
                    del active[run]
                    changed = True
                    _event("run_completed" if rows[run]["state"] == "complete" else "run_failed",
                           status_path, run=run, index=runs.index(run) + 1, total=len(runs),
                           counts=_counts(rows), backend=worker["backend"])
                if frame_broker is not None:
                    control_path = status_path.with_name("worker_control.json")
                    if control_path.is_file():
                        try:
                            control = _read(control_path)
                            if (control.get("pid") == os.getpid()
                                    and control.get("invocation_started_utc") == status["invocation_started_utc"]):
                                frame_broker.set_workers(int(control["cpu_workers"]),
                                    int(control.get("gpu_workers", frame_broker.gpu_workers)))
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            status["worker_control_error"] = str(error)
                    status["frame_workers"] = frame_broker.snapshot()
                    status["cpu_workers"] = frame_broker.cpu_workers
                    status["gpu_workers"] = frame_broker.gpu_workers
                    status["parallel_workers"] = max(1, frame_broker.cpu_workers) + frame_broker.gpu_workers
                    changed = True
                if changed:
                    status["current_runs"] = list(active)
                    status["current_run"] = next(iter(active), None)
                    status["stage"] = "run_progress" if active else "run_completed"
                    _write_status(status_path, status)
                if active:
                    time.sleep(1)
        except BaseException:
            for worker in active.values():
                if not worker.get("adopted"):
                    worker["process"].terminate()
            for worker in active.values():
                if worker.get("adopted"):
                    continue
                try:
                    worker["process"].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker["process"].kill()
            status.update(state="interrupted", stage="interrupted", current_runs=[], current_run=None)
            _write_status(status_path, status)
            raise
        status.update(state="complete", stage="complete", current_runs=[], current_run=None,
                      completed_utc=utc_now(), success=_counts(rows)["failed"] == 0)
        _write_status(status_path, status)
        _event("complete", status_path, index=len(runs), total=len(runs), counts=status["counts"],
               success=status["success"], raw_inventory_available=status["raw_inventory_available"])
        return status


def run_frame_parallel_batch(model, series, output_parent, *, max_workers=2, gpu_workers=0, **options):
    """Share CPU/GPU frame workers across a bounded number of ordered runs."""
    from darsia.fluidflower.tools.phase_frame_workers import serve_frame_workers
    if not 0 <= max_workers <= 14 or gpu_workers not in (0, 1) or max_workers + gpu_workers < 1:
        raise ValueError("Choose 0-14 CPU workers and 0-1 GPU workers")
    with serve_frame_workers(max_workers, gpu_workers) as broker:
        # These are state holders, not dedicated optical compute slots. Their
        # ordered physics/reference work is memory-heavy, so keep it bounded
        # while the shared broker handles the many cheap optical tasks.
        coordinators = min(2, max(1, max_workers // 6))
        return run_parallel_batch(model, series, output_parent,
            max_workers=coordinators, gpu_workers=0, worker_memory_gib=8.,
            free_headroom_gib=broker.headroom, frame_broker=broker, **options)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--series", required=True, choices=SERIES)
    parser.add_argument("--output-parent", required=True, type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--save-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--backend", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--review-root", action="append", default=[], type=Path)
    parser.add_argument("--source-manifest", action="append", default=[], type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", nargs="+", help="Prepare only these selected experiments")
    parser.add_argument("--exclude-runs", nargs="*", default=[], help="Runs already prepared for this exact model")
    parser.add_argument("--reprepare-runs", nargs="+", default=[],
                        help="Prepare these experiments again from their own raw target, "
                             "ignoring any reusable prepared source and any saved evaluation")
    parser.add_argument("--only-run", help="Internal: process one independently locked experiment")
    parser.add_argument("--parallel-workers", type=int,
                        help="Legacy CPU-only worker count (1-14); 1 uses the selected backend")
    parser.add_argument("--cpu-workers", type=int,
                        help="Independent CPU experiments at once (0-14)")
    parser.add_argument("--gpu-workers", type=int,
                        help="Independent GPU experiments at once (0-1)")
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.dry_run and not all((args.reviewer, args.save_root, args.cache_root)):
        parser.error("Prediction requires --reviewer, --save-root and --cache-root")
    explicit_workers = args.cpu_workers is not None or args.gpu_workers is not None
    if explicit_workers and args.parallel_workers is not None:
        parser.error("Use either --parallel-workers or --cpu-workers/--gpu-workers")
    cpu_workers = (args.cpu_workers or 0) if explicit_workers else (args.parallel_workers or 1)
    gpu_workers = (args.gpu_workers or 0) if explicit_workers else 0
    if explicit_workers and not (0 <= cpu_workers <= 14 and 0 <= gpu_workers <= 1
                                 and cpu_workers + gpu_workers >= 1):
        parser.error("Choose 0-14 CPU workers and 0-1 GPU workers, at least one total")
    if not explicit_workers and not 1 <= cpu_workers <= 14:
        parser.error("--parallel-workers must be between 1 and 14")
    if args.only_run and (explicit_workers or cpu_workers != 1):
        parser.error("--only-run cannot schedule further workers")
    if os.environ.get("PHASE_BATCH_WORKER"):
        # Avoid a four-worker batch silently becoming a 64-thread batch.
        try:
            import cv2
            cv2.setNumThreads(1)
        except ImportError:
            pass
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
    batch_fn = (run_frame_parallel_batch if not args.dry_run and
                (explicit_workers or cpu_workers > 1) else run_batch)
    options = dict(reviewer=args.reviewer or "", save_root=args.save_root or Path("."),
                   cache_root=args.cache_root or Path("."), backend=args.backend,
                   review_roots=args.review_root, source_manifests=args.source_manifest,
                   dry_run=args.dry_run, requested_runs=args.runs, excluded_runs=args.exclude_runs,
                   reprepare_runs=args.reprepare_runs)
    if batch_fn is run_frame_parallel_batch:
        options["max_workers"] = cpu_workers
        options["gpu_workers"] = gpu_workers
    else:
        options["only_run"] = args.only_run
        options["worker_token"] = args.worker_token
    batch_fn(args.model, args.series, args.output_parent, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
