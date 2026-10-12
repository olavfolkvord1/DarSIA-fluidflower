"""Resume-safe display-file rendering, never inference or scientific writes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
import json
import os
from threading import Lock, Event
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .. import phase_rig_raw_display as display
from ..phase_rig_display_prerender import CONTRACT, source_binding, pointer_path, _token
from ..project import sha256_file, utc_now
from ..review_storage import resolve_review_path
from ..phase_rig_display_acceleration import gpu_seconds, backend_description

_VOLUME_LOCK = Lock()
_VOLUME_TOTAL = 0
_VOLUME_LIMIT = 50_000_000_000
_STOP = Event()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, data):
    global _VOLUME_TOTAL
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing display derivative differs: {path}")
    else:
        with _VOLUME_LOCK:
            if _VOLUME_TOTAL+len(data) > _VOLUME_LIMIT:
                _STOP.set()
                raise ValueError("Display derivatives would exceed 50 GB; ask Olav before continuing")
            _VOLUME_TOTAL += len(data)
        with path.open("xb") as stream:
            stream.write(data)
    return dict(path=str(path), sha256=sha256(data).hexdigest())


def _json(path, value):
    return _write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())


def _difference(a, b):
    difference = np.abs(a.astype(np.int16)-b.astype(np.int16))
    return dict(max_abs_per_channel=difference.max(axis=(0, 1)).tolist(),
                mean_abs_per_channel=difference.mean(axis=(0, 1)).tolist(),
                differing_pixels=int(np.any(difference, axis=-1).sum()))


def render_record(review_root, run, frame_index, *, reference, output, backend="auto", parity=False):
    key = "reference" if reference else f"frame_{frame_index:03d}"
    folder = Path(output) / run / key
    state, record, binding = display._native_frame_assets_cached(review_root, run, frame_index, reference=reference)
    result = {}
    for mode in ("change", "raw_detail"):
        bound = source_binding(state, record, binding, mode=mode, reference=reference)
        receipt_path = folder / (mode+".receipt.json")
        if receipt_path.is_file():
            receipt = _read(receipt_path)
            if receipt.get("contract") != CONTRACT or receipt.get("source_binding") != bound:
                raise ValueError(f"Existing display receipt binds another source: {receipt_path}")
            for name in ("display", "masks"):
                if sha256_file(resolve_review_path(receipt[name]["path"])) != receipt[name]["sha256"]:
                    raise ValueError(f"Existing display derivative changed: {receipt_path}")
            result[mode] = dict(binding_sha256=_token(bound), display=receipt["display"], masks=receipt["masks"],
                                receipt=dict(path=str(receipt_path), sha256=sha256_file(receipt_path)))
            continue
        started = time.perf_counter()
        rgb, masks, audit = display.load_rig_raw_display_rgb(review_root, run, frame_index, mode=mode,
                           reference=reference, use_prerender=False, backend=backend, cache_signal=True)
        parity_result = None
        if parity:
            cpu_rgb, cpu_masks, _ = display.load_rig_raw_display_rgb(review_root, run, frame_index, mode=mode,
                                 reference=reference, use_prerender=False, backend="cpu", cache_signal=True)
            parity_result = _difference(rgb, cpu_rgb)
            if any(parity_result["max_abs_per_channel"]) or any(not np.array_equal(masks[name], cpu_masks[name]) for name in masks):
                raise ValueError(f"CUDA/CPU display disagreement in {run}/{key}/{mode}: {parity_result}")
        encoded = BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(encoded, format="JPEG", quality=100, subsampling=0)
        image_asset = _write(folder / (mode+".jpg"), encoded.getvalue())
        compressed = BytesIO()
        np.savez_compressed(compressed, shape_hw=np.array(rgb.shape[:2], np.int32),
                            **{name: np.packbits(value.reshape(-1), bitorder="little") for name, value in masks.items()})
        mask_asset = _write(folder / (mode+".masks.npz"), compressed.getvalue())
        with Image.open(BytesIO(encoded.getvalue())) as photograph:
            decoded = np.asarray(photograph.convert("RGB"), dtype=np.uint8)
        receipt = dict(contract=CONTRACT, source_binding=bound, live_audit=audit,
                       display=image_asset, masks=mask_asset,
                       encoder=dict(format="JPEG", quality=100, subsampling=0),
                       rendered_pixels_sha256=sha256(rgb.tobytes()).hexdigest(),
                       jpeg_vs_live_pixels=_difference(decoded, rgb), cuda_vs_cpu_pixels=parity_result,
                       elapsed_seconds=time.perf_counter()-started, display_only=True,
                       maps_changed=False, scientific_sidecars_changed=False, labels_changed=False)
        receipt_asset = _json(receipt_path, receipt)
        result[mode] = dict(binding_sha256=_token(bound), display=image_asset, masks=mask_asset, receipt=receipt_asset)
    return key, result


def render_run(review_root, run, *, output, backend, parity_frames=()):
    state, _, _ = display._native_frame_assets_cached(review_root, run, reference=True)
    count = len(state["native"]["frame_sources"])
    records = {}
    for reference, index in [(True, None), *[(False, i) for i in range(count)]]:
        if _STOP.is_set():
            raise RuntimeError("Display rendering stopped after another worker failed")
        key, rendered = render_record(review_root, run, index, reference=reference, output=output, backend=backend,
                                     parity=(run, index) in parity_frames)
        records[key] = rendered
        print(json.dumps(dict(run=run, record=key, status="rendered")), flush=True)
    return _json(Path(output)/run/"manifest.json", dict(contract=CONTRACT, run=run, frame_count=count, records=records))


def studio_timings(candidate, *, output):
    """Measure the real Qt photograph loader over 20 changes/run/session."""
    from ..phase_v5_display_asset import load_display_image
    candidate = resolve_review_path(candidate).resolve()
    root = candidate/"source_alias"
    result = dict(contract="ffac-p1-final-studio-display-timings-v1", candidate_id=candidate.name,
                  created_utc=utc_now(), runs={}, display_only=True,
                  measurement="wall time to decode/copy full Qt photograph; actual Studio photo-worker adapter")
    display.clear_display_session_cache()
    for run in ("ac17", "ac58"):
        manifest = _read(root/"native_frames"/run/"manifest.json")
        count = len(manifest["frame_sources"])
        samples = {"change": [], "lift1": []}
        for number in range(20):
            index = (number+1) % count
            # Alternate ordering within the same session and source timeline.
            for mode in (("lift1", "change") if number % 2 == 0 else ("change", "lift1")):
                started = time.perf_counter()
                image = load_display_image(root, run, index, mode=mode)
                elapsed = time.perf_counter()-started
                if image.isNull():
                    raise ValueError("Studio photo worker returned no image")
                if mode == "change" and not getattr(image, "_display_support_audit", {}).get("pre_rendered"):
                    raise ValueError("Studio timing reached live fallback instead of the new bound display file")
                samples[mode].append(dict(frame_index=index, elapsed_seconds=elapsed))
        medians = {mode: float(np.median([row["elapsed_seconds"] for row in rows])) for mode, rows in samples.items()}
        ratio = medians["change"] / medians["lift1"]
        result["runs"][run] = dict(samples=samples, median_seconds=medians, change_to_lift1_ratio=ratio,
                                    acceptance_ratio_limit=1.5, passed=ratio <= 1.5)
    result["passed"] = all(row["passed"] for row in result["runs"].values())
    return _json(Path(output)/"studio_timing_acceptance.json", result)


def main(argv=None):
    global _VOLUME_TOTAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--backend", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--measure-studio", action="store_true")
    parser.add_argument("--timings-only", action="store_true")
    args = parser.parse_args(argv)
    candidate = resolve_review_path(args.candidate).resolve()
    if not (candidate/"p1_final_plan.json").is_file():
        raise ValueError("Pre-rendering requires the new P1-final candidate")
    preparation = _read(candidate/"studio_preparation.json")
    if preparation.get("status") != "ready" or preparation.get("runs_done") != 43:
        raise ValueError("§1 candidate QA and registration must complete before display rendering")
    if args.timings_only:
        result = studio_timings(candidate, output=args.output)
        print(json.dumps(dict(studio_timing_report=result)), flush=True)
        return result
    if (args.output/"manifest.json").is_file():
        existing = _read(args.output/"manifest.json")
        if existing.get("contract") != CONTRACT or existing.get("candidate_id") != candidate.name:
            raise ValueError("Existing completed display batch binds another candidate")
        asset = dict(path=str(args.output/"manifest.json"), sha256=sha256_file(args.output/"manifest.json"))
        if args.publish:
            _json(pointer_path(candidate), dict(contract=CONTRACT, candidate_root=str(candidate), manifest=asset,
                                               display_only=True, default_candidate_changed=False))
        print(json.dumps(dict(status="already_complete", manifest=asset)), flush=True)
        return existing
    review_root = resolve_review_path(args.review_root or candidate/"source_alias").resolve()
    runs = sorted(path.name for path in (review_root/"native_frames").iterdir() if (path/"manifest.json").is_file())
    if args.runs:
        runs = [run.casefold() for run in args.runs]
    if args.publish and len(runs) != 43:
        raise ValueError("A partial display batch cannot be published as complete")
    args.output.mkdir(parents=True, exist_ok=True)
    _STOP.clear()
    _VOLUME_TOTAL = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    total_frames = sum(len(_read(review_root/"native_frames"/run/"manifest.json")["frame_sources"])+1 for run in runs)
    # A sample must be measured before the entire batch is allocated. This
    # conservative guard includes JPEGs, masks, receipts and manifest overhead.
    cv2.setNumThreads(1)
    started, cpu_start, gpu_start = time.perf_counter(), time.process_time(), gpu_seconds()
    sample_key, sample = render_record(review_root, runs[0], 0, reference=False, output=args.output,
                                      backend=args.backend, parity=True)
    sample_bytes = sum(Path(item[name]["path"]).stat().st_size for item in sample.values()
                       for name in ("display", "masks", "receipt"))
    estimated = int(sample_bytes*total_frames*1.25)
    if estimated > 50_000_000_000:
        raise ValueError(f"Estimated display volume {estimated/1e9:.2f} GB exceeds 50 GB; ask Olav before continuing")
    print(json.dumps(dict(estimated_total_bytes=estimated, workers=args.workers, backend=args.backend)), flush=True)
    parity_frames = {(run, index) for run, index in (("ac17", 5), ("ac17", 6), ("ac17", 20),
                     ("ac19", 13), ("ac28", 13), ("ac40", 13), ("ac51", 19))}
    manifests = {}
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="rig-display-render") as workers:
        jobs = {workers.submit(render_run, review_root, run, output=args.output, backend=args.backend,
                               parity_frames=parity_frames): run for run in runs}
        try:
            for future in as_completed(jobs):
                manifests[jobs[future]] = future.result()
        except Exception:
            _STOP.set()
            for future in jobs:
                future.cancel()
            raise
    actual_bytes = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    manifest = dict(contract=CONTRACT, candidate_id=candidate.name, candidate_root=str(candidate),
                    candidate_seal=dict(path=str(candidate/"rig_camera_candidate_receipt.json"),
                                        sha256=sha256_file(candidate/"rig_camera_candidate_receipt.json")),
                    runs=manifests, frame_and_reference_count=total_frames, display_only=True,
                    estimated_total_bytes=estimated, total_bytes=actual_bytes,
                    wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_start,
                    gpu_compute_seconds=gpu_seconds()-gpu_start, backend=backend_description(),
                    gpu_time_measurement="sum of CUDA event spans for RAW transfer and integer chroma median; RAW span includes synchronized download",
                    cpu_time_measurement="process CPU time across all renderer threads, including initial size sample and parity checks",
                    maps_changed=False, labels_changed=False, sessions_changed=False)
    manifest_asset = _json(args.output/"manifest.json", manifest)
    if args.publish:
        _json(pointer_path(candidate), dict(contract=CONTRACT, candidate_root=str(candidate), manifest=manifest_asset,
                                           display_only=True, default_candidate_changed=False))
    if args.measure_studio:
        studio_timings(candidate, output=args.output)
    print(json.dumps(dict(status="complete", manifest=manifest_asset, total_bytes=actual_bytes)), flush=True)
    return manifest


if __name__ == "__main__":
    main()
