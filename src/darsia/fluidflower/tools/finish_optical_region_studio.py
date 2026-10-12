"""Finish fresh review media while native run packages are being prepared."""
import os

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
import cv2

from darsia.fluidflower.tools.phase_optical_cleanup_preview import read
from darsia.fluidflower.tools.phase_reference_maps import _asset
from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness, RECEIPT
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import utc_now


def write(path, value):
    from darsia.fluidflower.tools.render_concentration_movie import write as publish
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    publish(path, value)


def media(root, run, threads=1, *, eager=False):
    from darsia.fluidflower.tools.phase_reference_maps import _worker_lock
    with _worker_lock(Path(root) / 'validation/presentation_locks' / (run + '.lock')):
        return _media(root, run, threads, eager=eager)


def _media(root, run, threads=1, *, eager=False):
    cv2.setNumThreads(threads)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=threads)
    root = Path(root)
    package = root / "validation/run_packages" / run
    destination = root / "validation/presentation" / run
    manifest = read(package / "postflight_manifest.json")
    job = manifest["jobs"][0]
    quality = dict(job["quality_flags"])
    from darsia.fluidflower.tools.phase_transfer_materialization import generate_review_media, generate_review_metadata
    identifier, digest = _model_identity(root)
    plan = dict(run=run, dataset=job['dataset'], source_model_id=identifier, source_model_sha256=digest,
        training_seen=quality["training_seen"], evidence_role="trained_optical_region_candidate",
        review_mode="visible")
    receipt = destination / "media.json"
    if receipt.is_file():
        record = read(receipt)
        recorded = record['job']
        fields = ('summary', 'per_frame_csv') if recorded.get('presentation_status') == 'deferred' else ('summary', 'mass_plot', 'contour_movie', 'per_frame_csv')
        for field in fields:
            path = Path(recorded[field])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Completed media receipt has a missing/empty {field}: {path}')
        if eager and recorded.get('presentation_status') == 'deferred':
            from darsia.fluidflower.tools.calibration_pairwise_review import _candidate_from_job
            from darsia.fluidflower.tools.phase_transfer_materialization import ensure_candidate_presentation
            candidate = _candidate_from_job(package, recorded)
            for kind in ('plot', 'video'):
                ensure_candidate_presentation(candidate, kind)
            record['optional_media_generated'] = True
            write(receipt, record)
        return str(receipt)
    if destination.exists():
        # Interrupted encodes have no completion receipt. Preserve them outside
        # the destination so restarting never mistakes them for valid media.
        import uuid
        archive = root / 'interrupted_presentation' / (run + '-' + uuid.uuid4().hex)
        archive.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(archive)
    generated = (generate_review_media if eager else generate_review_metadata)(root, destination, manifest, plan)
    job["quality_flags"] = quality
    candidate = Path(job["candidate_folder"])
    summary = read(candidate / "summary.json")
    for key in ("include_in_training", "requires_explicit_training_selection"):
        summary.pop(key, None)
    summary.update(quality)
    write(candidate / "summary.json", summary)
    for field, filename in (("summary", "summary.json"), ("mass_plot", "calibration_mass_fit.png"),
                            ("contour_movie", "calibration_review.mp4"), ("per_frame_csv", "calibration_review_per_frame.csv")):
        job[field] = str(candidate / filename)
    write(receipt, dict(job=job, media=generated, ready_utc=utc_now()))
    print(f"{run.upper()}: ordinary-Studio timeline ready" + (" with video and diagnostics" if eager else "; video and diagnostics deferred"), flush=True)
    return str(receipt)


def finish(root, *, cpus=None, memory_gb=None, eager=False):
    from darsia.fluidflower.tools.phase_reference_maps import _worker_lock
    root = Path(root)
    with _worker_lock(root / 'validation/presentation_locks/finish.lock'):
        return _finish(root, cpus=cpus, memory_gb=memory_gb, eager=eager)


def _finish(root, *, cpus=None, memory_gb=None, eager=False):
    if (Path(root) / 'validation/registered_base_frozen.json').exists():
        raise ValueError('Registered validation base is immutable; use resumed Studio preparation to publish a new prepared set')
    from darsia.fluidflower.phase_execution_resources import available_resources
    available_cpus, available_memory = available_resources()
    cpus = cpus if cpus is not None else int(os.environ.get("PHASE_CPU_LIMIT", available_cpus))
    memory_gb = memory_gb if memory_gb is not None else float(os.environ.get("PHASE_MEMORY_GB", available_memory / 1024**3))
    if cpus < 1 or memory_gb <= 0:
        raise ValueError("CPU and memory budgets must be positive")
    workers = min(2, cpus, max(1, int(memory_gb // 2)))
    threads = max(1, cpus // workers)
    runs = [j["run"] for j in read(root / "review_source/postflight_manifest.json")["jobs"]]
    submitted = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while len(submitted) < len(runs):
            status_path = root / 'studio_preparation.json'
            if status_path.is_file() and read(status_path).get('status') == 'failed':
                raise RuntimeError('Phase-map preparation failed; stopping presentation wait')
            for future in submitted.values():
                if future.done():
                    future.result()  # Fail promptly, without waiting for all remaining runs.
            for run in runs:
                if run in submitted:
                    continue
                package = root / "validation/run_packages" / run
                if not (package / 'validation_binding.json').is_file() or not (package / 'postflight_manifest.json').is_file():
                    continue
                require_reference_readiness(package, allow_pending=not eager)
                submitted[run] = pool.submit(media, root, run, threads, eager=eager)
            if len(submitted) < len(runs):
                time.sleep(2)
        receipts = {run: read(Path(future.result())) for run, future in submitted.items()}
    while not (root / "studio_preparation.json").is_file() or read(root / "studio_preparation.json").get("status") != "ready":
        if (root / 'studio_preparation.json').is_file() and read(root / 'studio_preparation.json').get('status') == 'failed':
            raise RuntimeError('Phase-map preparation failed; stopping presentation wait')
        time.sleep(2)
    # Only presentation paths change. Model/map/native identities stay bound to
    # the standard validator. Reference pixels and their receipts are retained.
    for run, record in receipts.items():
        package = root / "validation/run_packages" / run
        manifest = read(package / "postflight_manifest.json")
        require_reference_readiness(package, allow_pending=not eager)
        manifest["jobs"] = [record["job"]]
        write(package / "postflight_manifest.json", manifest)
        if (package / RECEIPT).is_file():
            readiness = read(package / RECEIPT)
            readiness["campaign"] = _asset(package / "postflight_manifest.json")
            write(package / RECEIPT, readiness)
        require_reference_readiness(package, allow_pending=not eager)
    destination = root / "validation"
    manifest = read(destination / "postflight_manifest.json")
    require_reference_readiness(destination, allow_pending=not eager)
    manifest["jobs"] = [receipts[run]["job"] for run in runs]
    write(destination / "postflight_manifest.json", manifest)
    if (destination / RECEIPT).is_file():
        readiness = read(destination / RECEIPT)
        readiness["campaign"] = _asset(destination / "postflight_manifest.json")
        write(destination / RECEIPT, readiness)
    require_reference_readiness(destination, allow_pending=not eager)
    status = read(root / "studio_preparation.json")
    status.update(presentation_ready=True, presentation_status="ready" if eager else "deferred", presentation_completed_utc=utc_now())
    write(root / "studio_preparation.json", status)
    print(f"Ordinary Studio candidate ready: {len(runs)} runs, fresh maps and presentation", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--eager-presentation", action="store_true", help="Explicitly render video and diagnostics in advance")
    args = parser.parse_args()
    finish(args.root, cpus=args.cpus, memory_gb=args.memory_gb, eager=args.eager_presentation)
