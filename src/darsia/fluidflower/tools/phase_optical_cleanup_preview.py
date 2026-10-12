"""Exact native optical-cache adapter for the experimental cleanup controls."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from PIL import Image

from darsia.fluidflower import phase_optical_cleanup as cleanup

_LOCK = threading.RLock()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{uuid.uuid4().hex}.json")
    try:
        temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        for attempt in range(12):
            try:
                os.replace(temp, path)
                break
            except PermissionError as exc:
                # Windows SMB can deny a rename while Studio has the previous
                # status file open for reading. Keep the old complete JSON in
                # place and retry the atomic replacement after that handle closes.
                if (os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32, 33)
                        or attempt == 11):
                    raise
                time.sleep(min(0.03 * (attempt + 1), 0.25))
    finally:
        temp.unlink(missing_ok=True)


def select_archive(evaluation, index):
    """Never pick an arbitrary latest cache when a frame has ambiguous inputs."""
    matches = []
    for path in (Path(evaluation) / "optical_frame_cache").glob("*.json"):
        if path.name.endswith(".running.json"):
            continue
        receipt = read(path)
        if receipt.get("frame") == index and path.with_suffix(".npz").is_file():
            if receipt.get("key") != path.stem:
                raise ValueError("Optical cache receipt identity differs from archive")
            matches.append(path.with_suffix(".npz"))
    if len(matches) != 1:
        raise ValueError(f"Frame {index + 1}: expected one cached optical score field, found {len(matches)}. Prepare the phase maps first.")
    return matches[0]


def verify_archive_key(archive, plan, native, index):
    """Reconstruct the worker key from actual model, photos and inference code.

    A matching raw mask alone is insufficient: distinct continuous scores can
    yield the same thresholded mask. Do not silently accept such a cache.
    """
    from darsia.fluidflower.tools.phase_reference_maps import _sha
    from darsia.fluidflower import phase_validation, phase_features, phase_learning
    scripts = Path(__file__).parent
    sources = [scripts / "phase_frame_workers.py", Path(phase_validation.__file__),
               Path(phase_features.__file__), Path(phase_learning.__file__),
               scripts / "phase_inference_benchmark_backends.py", scripts / "materialize_phase_iteration.py"]
    code_sha = hashlib.sha256("".join(_sha(p) for p in sources).encode()).hexdigest()
    photo = Path(plan["native_manifest"]["path"]).parent / f"frame_{index:03d}.jpg"
    previous = photo.with_name(f"frame_{index - 1:03d}.jpg") if index else None
    # The original validator permits only frame 0 as an independent control.
    # Old review packages do not all preserve that flag, so verify either valid
    # frame-1 history identity, not an arbitrary previous photograph.
    previous_options = [previous, None] if index == 1 else [previous]
    backend = (plan.get("inference_provenance") or {}).get("requested_backend")
    for baseline in archive.parent.glob("baseline-*.npy"):
        for previous in previous_options:
            task = dict(model=str(Path(plan["model_root"])), model_sha=plan["model_sha256"],
                baseline=str(baseline), current=str(photo), previous=str(previous) if previous else None,
                current_sha=_sha(photo), previous_sha=_sha(previous) if previous else None,
                frame=index, box=list(native["content_box_xywh"]),
                shape=[native["content_box_xywh"][3], native["content_box_xywh"][2]],
                radiometry_profile=None, code_sha=code_sha,
                required_backend="cpu" if backend == "cpu" else None)
            if digest(task) == archive.stem:
                values = np.load(baseline, allow_pickle=False, mmap_mode="r")
                if hashlib.sha256(values.tobytes()).hexdigest() != baseline.stem.removeprefix("baseline-"):
                    raise ValueError("Optical cache baseline changed")
                return task
    raise ValueError("Optical cache input identity differs from the current model, photos or inference version. Prepare phase maps again.")


@lru_cache(maxsize=1)
def load(root, run, candidate_json, index):
    """One full-resolution frame in RAM; cache is serialized by resolve()."""
    from darsia.fluidflower.tools.phase_reference_maps import build_plan, _sha
    from darsia.fluidflower.tools.phase_mass_overnight_data import decode_labels
    from darsia.fluidflower import phase_runtime
    candidate = json.loads(candidate_json)
    plan = build_plan(None, Path(root), run, candidate)
    evaluation = Path(plan["model_root"])
    transfer = read(evaluation / "transfer_review.json")
    if (transfer.get("evaluation_model_sha256") != plan["model_sha256"]
            or transfer.get("run", "").casefold() != run.casefold()
            or _sha(transfer["target_native_manifest"]) != plan["native_manifest"]["sha256"]):
        raise ValueError("Optical cache does not belong to the displayed model and native timeline")
    archive = select_archive(evaluation, index)
    native = read(plan["native_manifest"]["path"])
    task = verify_archive_key(archive, plan, native, index)
    x, y, w, h = native["content_box_xywh"]
    raw_path = (Path(candidate["raw_phase_map_output_dir"]) / "native_phase_maps" / run /
                ("A_" + candidate["optical_model_sha256"][:16]) / f"frame_{index:03d}.png")
    raw = decode_labels(raw_path, (x, y, w, h))
    with np.load(archive, allow_pickle=False) as data:
        heads = data["heads"]
    if heads.shape != (h, w, 2) or not np.isfinite(heads).all() or (heads < 0).any() or (heads > 1).any():
        raise ValueError("Invalid native optical score field")
    expected = np.where(heads[..., 0] >= .5, np.where(heads[..., 1] >= .6, 2, 1), 0).astype(np.uint8)
    if not np.array_equal(expected, raw):
        raise ValueError("Cached scores do not reproduce the archived raw optical map")
    del expected
    presence = np.ascontiguousarray(heads[..., 0])
    del heads
    profile = phase_runtime.load_physics_profile(evaluation)
    context, _ = phase_runtime.load_run_context(evaluation, profile, plan["dataset"], run,
                       native_manifest_sha256=plan["native_manifest"]["sha256"])
    domain = np.asarray(context.domain, bool).copy()
    spacing = (abs(float(context.y_m[1] - context.y_m[0])), abs(float(context.x_m[1] - context.x_m[0])))
    if domain.shape != raw.shape:
        raise ValueError("Native optical and physical grids differ")
    photo = Path(plan["native_manifest"]["path"]).parent / f"frame_{index:03d}.jpg"
    identity = dict(format=cleanup.VERSION, dataset=plan["dataset"], run=run, frame_index=index,
        source_name=native["frame_sources"][index].get("source_name"), model_id=plan["model_id"],
        model_sha256=plan["model_sha256"], candidate_id=plan["candidate_id"],
        native_manifest_sha256=plan["native_manifest"]["sha256"], photo_sha256=_sha(photo),
        raw_map_sha256=_sha(raw_path), scores_sha256=_sha(archive), score_archive=str(archive),
        raw_map_verified_pixel_exact=True, worker_task=task, spacing_m=spacing,
        algorithm_sha256=_sha(cleanup.__file__), adapter_sha256=_sha(__file__))
    return presence, raw, domain, spacing, native, identity


def resolve(root, run, candidate_json, index, settings, mode, *, cancelled=lambda: False, cache_root=None):
    """Native-size render; obsolete slider requests never run behind newer work."""
    if mode not in {"processed", "raw", "changes"}:
        raise ValueError("Unknown cleanup view")
    from darsia.fluidflower.phase_rendering import AQUEOUS_RGBA, GAS_RGBA, phase_display_cache_root
    with _LOCK:
        if cancelled():
            raise RuntimeError("Superseded optical preview")
        presence, raw, domain, spacing, native, identity = load(str(root), run, candidate_json, index)
        if cancelled():
            raise RuntimeError("Superseded optical preview")
        request = dict(identity=identity, settings=asdict(settings), mode=mode)
        folder = Path(cache_root or phase_display_cache_root().parent / "optical-cleanup") / digest(request)
        target = folder / "overlay.png"
        if target.is_file() and (folder / "receipt.json").is_file():
            return target
        if mode == "raw":
            labels, report = raw, {}
        else:
            labels, report = cleanup.apply(presence, raw, domain, spacing, settings)
        if cancelled():
            raise RuntimeError("Superseded optical preview")
        x, y, w, h = native["content_box_xywh"]
        cw, ch = native["canvas_wh"]
        rgba = np.zeros((ch, cw, 4), np.uint8)
        crop = rgba[y:y+h, x:x+w]
        if mode == "changes":
            crop[(labels > 0) & (raw == 0)] = (0, 230, 255, 255)
            crop[(labels == 0) & (raw > 0)] = (255, 150, 0, 255)
        else:
            crop[labels == 1] = AQUEOUS_RGBA
            crop[labels == 2] = GAS_RGBA
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f".{uuid.uuid4().hex}.png"
        try:
            Image.fromarray(rgba).save(temporary, compress_level=1)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        write(folder / "receipt.json", dict(**request, report=report, production_updated=False,
              interpretation="Experimental optical support; gas held fixed within domain. No reference support or learned completion. No concentration calibration."))
        return target


def save_profile(series, settings, receipt, *, project_root=None):
    from darsia.fluidflower.phase_review_retractions import resolve_project_root
    if receipt["identity"]["dataset"].casefold() != series.casefold():
        raise ValueError("Preview and destination series differ")
    if receipt.get("settings") != asdict(settings):
        raise ValueError("Settings differ from the rendered preview")
    folder = resolve_project_root(series, project_root) / "config" / "optical_cleanup"
    profile = dict(format=cleanup.VERSION, series=series, settings=asdict(settings),
                   created_utc=datetime.now(timezone.utc).isoformat(), status="experimental_preview",
                   evidence=receipt, production_updated=False)
    path = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8] + ".json")
    write(path, profile)
    write(folder / "latest.json", profile)
    return path


def load_profile(series, *, project_root=None):
    from darsia.fluidflower.phase_review_retractions import resolve_project_root
    path = resolve_project_root(series, project_root) / "config" / "optical_cleanup" / "latest.json"
    profile = read(path)
    if profile.get("format") != cleanup.VERSION or profile.get("series") != series:
        raise ValueError("Incompatible optical cleanup profile")
    return cleanup.Settings(**profile["settings"]), path
