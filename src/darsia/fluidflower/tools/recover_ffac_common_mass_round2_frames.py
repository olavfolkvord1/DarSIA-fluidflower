"""Explicit independent-frame recovery using the unchanged frozen R2 driver.

Only named frames are computed. Existing scientific receipts are preserved;
the full source-bound run inventory is checked after those frames finish.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

CONTRACT = "ffac-common-mass-round2-explicit-frame-recovery-v1"
_WORKER = {}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def asset(path):
    return dict(path=str(path), sha256=sha(path), size_bytes=Path(path).stat().st_size)


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def checked_receipt(path, metadata, frozen_sha):
    value = read(path)
    wanted = dict(run=metadata["run"], frame_index=metadata["frame_index"], elapsed_h=metadata["elapsed_h"],
        source_sha256=metadata["source_sha256"], phase_sha256=metadata["labels"]["sha256"],
        pixel_area_sha256=metadata["area_map"]["sha256"], calibration_freeze_sha256=frozen_sha)
    if value.get("complete") is not True or any(value.get(key) != item for key, item in wanted.items()):
        raise ValueError(f"Saved frame is incomplete or differently source-bound: {path}")
    return value


def init_worker(job, ops, frozen_sha, helper_sha):
    if sha(__file__) != helper_sha:
        raise ValueError("Recovery helper changed between processes")
    from darsia.fluidflower.tools import run_ffac_common_mass_round2 as driver
    job = Path(job); campaign_path = job / "campaign_binding.json"; campaign = read(campaign_path)
    for name in ("input_plan", "round1_freeze"):
        if sha(campaign[name]["path"]) != campaign[name]["sha256"]:
            raise ValueError(f"Campaign source changed: {name}")
    frozen_path = job / "calibration/freeze/frozen_calibration.json"
    if sha(frozen_path) != frozen_sha:
        raise ValueError("The existing frozen calibration changed")
    driver.verify_worker_code(campaign_path)
    driver.init_worker(campaign["input_plan"]["path"], campaign["round1_freeze"]["path"], "cpu",
        campaign.get("l1_reference"), str(campaign_path))
    freeze = driver.output.FrozenEvidence.verify(frozen_path)
    _WORKER.update(driver=driver, job=job, ops=Path(ops), freeze=freeze,
        campaign=campaign, campaign_sha=sha(campaign_path), anchors={}, helper_sha=helper_sha)


def compute_frame(frame_index, run):
    """The same prepare/anchor/evaluate/save sequence as driver.produce_run."""
    driver = _WORKER["driver"]; freeze = _WORKER["freeze"]
    freeze.check_frame_run(run)
    rows = driver.rows_for_run(run)
    selected = [row for row in rows if row["frame_index"] == frame_index]
    if len(selected) != 1:
        raise ValueError("One exact source frame is required")
    metadata = selected[0]
    folder = _WORKER["job"] / "production" / run / f"frame_{frame_index:03d}"
    source_receipt = folder / "receipt.json"
    started = time.perf_counter(); computed = not source_receipt.is_file()
    if computed:
        if run not in _WORKER["anchors"]:
            anchor_row = driver.output.nearest(rows, 2.25)
            frame, response = driver.prepare(anchor_row, freeze)
            _WORKER["anchors"][run] = (driver.engine().anchor_from_response(response), anchor_row)
            del frame, response; gc.collect()
        anchor, anchor_row = _WORKER["anchors"][run]
        frame, response = driver.prepare(metadata, freeze)
        result = driver.engine().evaluate_round2_frame(response, freeze.record["parameters"], anchor, fields=True)
        driver.output.save_frame(folder, result, driver.scientific_metadata(metadata, frame), freeze)
        del frame, response, result; gc.collect()
    else:
        anchor_row = driver.output.nearest(rows, 2.25)
    checked_receipt(source_receipt, metadata, freeze.sha256)
    receipt = dict(contract=CONTRACT, complete=True, run=run, frame_index=frame_index,
        elapsed_h=metadata["elapsed_h"], source_sha256=metadata["source_sha256"],
        phase_sha256=metadata["labels"]["sha256"], pixel_area_sha256=metadata["area_map"]["sha256"],
        scientific_receipt=asset(source_receipt), frozen_sha256=freeze.sha256,
        campaign_binding_sha256=_WORKER["campaign_sha"], helper_sha256=_WORKER["helper_sha"],
        anchor_source_binding={key: anchor_row[key] for key in ("run", "frame_index", "elapsed_h", "source_sha256", "labels", "area_map")},
        scientific_computation_performed=computed, existing_receipt_reused=not computed,
        backend="cpu", threads_per_worker=1, original_driver_sequence=True,
        new_fit=False, new_freeze=False, new_phase_inference=False, wall_seconds=time.perf_counter() - started)
    destination = _WORKER["ops"] / "frames" / run / f"frame_{frame_index:03d}" / "receipt.json"
    write_new(destination, receipt)
    return asset(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True); parser.add_argument("--ops", type=Path, required=True)
    parser.add_argument("--run", required=True); parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6); parser.add_argument("--frozen-sha256", required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 6 or len(args.frames) != len(set(args.frames)) or len(args.frozen_sha256) != 64:
        parser.error("Use one to six workers, unique explicit frames and the exact existing freeze SHA")
    if (args.ops / "completion.json").exists() or (args.ops / "recovery_started.json").exists():
        raise FileExistsError("This explicit recovery has already been launched; no blind retry")
    helper_sha = sha(__file__)
    init_worker(str(args.job), str(args.ops), args.frozen_sha256, helper_sha)
    source_rows = _WORKER["driver"].rows_for_run(args.run)
    if not set(args.frames) <= {row["frame_index"] for row in source_rows}:
        raise ValueError("An explicit frame is outside the frozen run's source index")
    write_new(args.ops / "recovery_started.json", dict(contract=CONTRACT, pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), run=args.run, frames=args.frames,
        workers=args.workers, threads_per_worker=1, helper=asset(__file__),
        campaign=asset(args.job / "campaign_binding.json"), frozen_sha256=args.frozen_sha256, automatic_retry=False))
    completed = []; failures = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
            initargs=(str(args.job), str(args.ops), args.frozen_sha256, helper_sha)) as pool:
        futures = {pool.submit(compute_frame, frame, args.run): frame for frame in args.frames}
        for future in as_completed(futures):
            try:
                result = future.result(); completed.append(result)
                print(json.dumps(dict(completed=result)), flush=True)
            except Exception as exc:
                failure = dict(frame_index=futures[future], error=repr(exc)); failures.append(failure)
                print(json.dumps(dict(failure=failure)), flush=True)
    full = []
    if not failures:
        for metadata in source_rows:
            path = args.job / "production" / args.run / f"frame_{metadata['frame_index']:03d}" / "receipt.json"
            checked_receipt(path, metadata, args.frozen_sha256); full.append(asset(path))
    write_new(args.ops / "completion.json", dict(contract=CONTRACT, complete=not failures, run=args.run,
        named_frames=args.frames, frozen_sha256=args.frozen_sha256, campaign=asset(args.job / "campaign_binding.json"),
        helper=asset(__file__), recovered_frame_receipts=completed, failures=failures,
        full_run_frame_receipts=full, full_run_frame_count=len(full), workers=args.workers,
        threads_per_worker=1, scientific_code_changed=False, new_fit=False, new_freeze=False,
        original_driver_sequence=True, automatic_retry=False))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
