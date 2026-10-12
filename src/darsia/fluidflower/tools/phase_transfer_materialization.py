"""Generate target-bound transfer predictions and fresh review media; never fit."""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import csv
import json
import os
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower import phase_validation
from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
from darsia.fluidflower.phase_runtime import (
    load_physics_profile, load_run_context, physics_runtime_identity,
)
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools.materialize_phase_iteration import CheckedCuda
from darsia.fluidflower.tools.phase_inference_benchmark_backends import CudaFeaturePredictor
from darsia.fluidflower.tools.phase_inference_benchmark_backends import partition
from darsia.fluidflower.tools.phase_injection_references import injection_end_references, resolve_reference_asset
from darsia.fluidflower.tools.phase_reference_single_pass import Endpoint, SinglePassReferences, sha256 as reference_sha256


def _read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def _verify_plan(plan):
    for name in ("protected_sha256", "evaluation_sha256"):
        for source, digest in plan[name].items():
            if not Path(source).is_file() or sha256_file(Path(source)) != digest:
                raise ValueError(f"Evaluation input changed: {source}")
    root = Path(plan["evaluation_root"]).resolve()
    if phase_validation._model_identity(root) != (plan["evaluation_model_id"], plan["evaluation_model_sha256"]):
        raise ValueError("Transfer model identity changed")
    if physics_runtime_identity(load_physics_profile(root)) != plan["physics_runtime"]:
        raise ValueError("Transfer physical runtime changed")
    jobs = phase_validation._source_jobs(root / "review_source")
    if len(jobs) != 1 or (jobs[0]["run"], jobs[0]["dataset"]) != (plan["run"], plan["dataset"]):
        raise ValueError("Transfer must contain only its selected target experiment")
    if plan["include_in_training"] is not False:
        raise ValueError("Transfer predictions cannot opt evaluation evidence into training")
    return root


def _cuda_device():
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("No CUDA device is available")
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def select_backend(model, requested, *, verify_numerics=False):
    """Fall back only during capability discovery, never after a failed prediction."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Transfer backend must be auto, cpu or cuda")
    if requested == "cpu":
        return dict(requested_backend=requested, backend="cpu-reference-tiled", fallback_reason=None)
    try:
        CudaFeaturePredictor.checked_contract(model)
    except ValueError as error:
        reason = str(error)
    else:
        try:
            device = _cuda_device()
        except Exception as error:
            # Device probing is deliberately isolated from model/data validation.
            reason = f"CUDA unavailable: {type(error).__name__}: {error}"
        else:
            return dict(requested_backend=requested, backend="cuda-float32-cached-features-checked"
                        if verify_numerics else "cuda-float32-cached-features",
                        fallback_reason=None, device=device)
    if requested == "cuda":
        raise ValueError(f"Explicit CUDA transfer is unsupported: {reason}")
    return dict(requested_backend=requested, backend="cpu-reference-tiled", fallback_reason=reason)


def _exposure(plan):
    return {"dataset": plan["dataset"], "training_seen": plan["training_seen"],
        "training_seen_review": plan["training_seen"], "independent_holdout": False,
        "independent_validation": False, "human_review_required": True,
        "evidence_role": plan["evidence_role"], "include_in_training": False,
        "requires_explicit_training_selection": True,
        **{key: plan.get(key) for key in ("optical_training_status", "calibration_evidence_status",
            "calibration_history", "exposure_label", "physics_mode", "physics_status", "review_mode")}}


def _area_domain(model_root, job, native):
    _, _, width, height = native["content_box_xywh"]
    profile = load_physics_profile(model_root)
    if profile is not None:
        entry = profile["runs"][f"{job['dataset']}/{job['run']}"]
        context, _ = load_run_context(model_root, profile, job["dataset"], job["run"],
            native_manifest_sha256=entry["native_manifest_sha256"])
        domain = np.asarray(context.domain, dtype=bool)
        source = str(model_root / entry["path"])
    elif (native.get("domain") or {}).get("path"):
        source = Path(native["domain"]["path"])
        if not source.is_absolute():
            # Radiometry display assets may live elsewhere; domain paths belong
            # to the target's original native-review root.
            source = Path(job.get("analysis_source_review_root") or job["review_output_dir"]) / source
        source = resolve_review_path(source)
        with np.load(source, allow_pickle=False) as archive:
            domain = np.asarray(archive["domain"], dtype=bool)
        source = str(source.resolve())
    else:
        return np.ones((height, width), dtype=bool), dict(kind="image-crop", pixels=width * height,
            axis_label="Visible image crop area [%]", source=None)
    if domain.shape != (height, width) or not domain.any():
        raise ValueError("Transfer area domain must be nonempty and match the target crop")
    return domain, dict(kind="porous-domain", pixels=int(domain.sum()),
        axis_label="Visible porous area [%]", source=source)


def _plot_area(path, times, counts, area, run):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(14, 5))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    if area.get("kind") in {"frame-native-porous-domain",
                              "frame-native-observation-domain"}:
        denominator = np.asarray(area.get("per_frame_pixels"), dtype=np.float64)
        if (denominator.shape != (len(counts),) or not np.isfinite(denominator).all()
                or np.any(denominator <= 0)):
            raise ValueError("Per-frame phase area denominator is incomplete")
    else:
        denominator = int(area["pixels"])
        if denominator <= 0:
            raise ValueError("Porous area denominator must be positive")
    values = [np.asarray([row[key] for row in counts]) * 100 / denominator
              for key in ("aqueous", "gas")]
    ax.stackplot(times, *values, colors=["#31d9a0", "#f24c9d"], labels=["Aqueous", "Free gas"])
    ax.set(xlabel="Time since first injection [h]", ylabel=area["axis_label"],
           title=f"{run.upper()} predicted 2D phase area - not concentration or mass")
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(path)


DEFERRED_PRESENTATION_FORMAT = "phase-review-presentation-on-demand-v1"


def generate_review_metadata(model_root, output_root, manifest, plan, progress=None):
    """Publish the native timeline needed by Studio without encoding display media.

    Numerical area counts from the phase report are carried into the summary.
    Legacy packages without those counts calculate them only if a plot is requested.
    """
    media = []
    for job in manifest["jobs"]:
        run = job["run"]
        if (run, job["dataset"]) != (plan["run"], plan["dataset"]):
            raise ValueError("Review metadata contains another experiment")
        native_path = Path(job["review_output_dir"]) / "native_frames" / run / "manifest.json"
        native = _read(native_path)
        frames = native.get("frame_sources", [])
        frame_count = int(native.get("frame_count", 0))
        if frame_count < 1 or len(frames) != frame_count:
            raise ValueError("Review metadata requires a complete native timeline")
        times = [float(row["elapsed_h"]) for row in frames]
        if not np.isfinite(times).all():
            raise ValueError("Review timeline contains nonfinite elapsed times")
        candidate = Path(output_root) / "postflight" / run
        candidate.mkdir(parents=True, exist_ok=False)
        report_path = Path(job["phase_map_output_dir"]).parent / "physics_reports" / f"{job['dataset']}_{run}.json"
        report = _read(report_path) if report_path.is_file() else {}
        reports = report.get("frames", [])
        indexed = {int(row["frame_index"]): row for row in reports}
        counts = [indexed.get(index, {}).get("phase_area_counts") for index in range(frame_count)]
        if not all(isinstance(row, dict) and all(k in row for k in ("aqueous", "gas", "mixed")) for row in counts):
            counts = []
        area = report.get("phase_area_denominator")
        if area is None and report.get("phase_area_domain_pixels"):
            kind = report.get("phase_area_domain_kind", "porous-domain")
            area = dict(kind=kind, pixels=int(report["phase_area_domain_pixels"]),
                        axis_label="Visible image crop area [%]" if kind == "image-crop" else "Visible porous area [%]",
                        source=str(report_path.resolve()))
        descriptor = {
            "format": DEFERRED_PRESENTATION_FORMAT, "model_root": str(Path(model_root).resolve()),
            "native_manifest": str(native_path.resolve()), "native_manifest_sha256": sha256_file(native_path),
            "phase_map_output_dir": str(Path(job["phase_map_output_dir"]).resolve()),
            "run": run, "dataset": job["dataset"], "candidate_id": job["candidate_id"],
            "params_sha256": job["params_sha256"], "source_model_id": plan["source_model_id"],
            "source_model_sha256": plan["source_model_sha256"], "frame_count": frame_count,
            "frame_counts": counts, "area_denominator": area,
        }
        fields = ["elapsed_h"] + (["aqueous_pixels", "gas_pixels", "mixed_pixels"] if counts else [])
        with (candidate / "calibration_review_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, elapsed in enumerate(times):
                writer.writerow({"elapsed_h": elapsed, **({key+"_pixels": counts[index][key]
                    for key in ("aqueous", "gas", "mixed")} if counts else {})})
        phase_validation._atomic_json(candidate / "summary.json", {
            "selection_candidate_id": job["candidate_id"], "source_model_id": plan["source_model_id"],
            "source_model_sha256": plan["source_model_sha256"], "params_sha256": job["params_sha256"],
            "review_plot_title": f"{run.upper()} predicted 2D phase areas; no mass analysis",
            "training_performed": False, "presentation_status": "deferred", "deferred_presentation": descriptor,
            "review_media_generated": False, "calibration_review_movie": {
                "params_sha256": job["params_sha256"], "fps": 2, "calibration_frame_count": frame_count,
                "plot_seconds": 0, "image_seconds_per_calibration_frame": .5}, **_exposure(plan)})
        job.update(candidate_folder=str(candidate), quality_flags=_exposure(plan), presentation_status="deferred")
        for field in ("calibration_review_movie", "review_movie", "movie_path", "review_plot", "plot_path"):
            job.pop(field, None)
        media.append(dict(run=run, frames=frame_count, candidate_folder=str(candidate),
                          frame_counts=counts, elapsed_h=times, presentation_status="deferred"))
        if progress:
            progress(f"{run.upper()}: native timeline ready; video and diagnostics are generated on request")
    return media


def validate_deferred_presentation(summary, *, run, candidate_id, params_sha256,
                                   review_output_dir, phase_map_output_dir, per_frame_csv):
    """Accept absent display media only for an explicit, identity-bound native package."""
    descriptor = summary.get("deferred_presentation") or {}
    if summary.get("presentation_status") != "deferred" or descriptor.get("format") != DEFERRED_PRESENTATION_FORMAT:
        raise ValueError("Missing valid deferred presentation metadata")
    for key, expected in (("run", run), ("candidate_id", candidate_id), ("params_sha256", params_sha256)):
        if descriptor.get(key) != expected:
            raise ValueError(f"Deferred presentation identifies another {key}")
    native_path = resolve_review_path(review_output_dir) / "native_frames" / run / "manifest.json"
    if (resolve_review_path(descriptor["native_manifest"]) != native_path.resolve()
            or sha256_file(native_path) != descriptor["native_manifest_sha256"]
            or resolve_review_path(descriptor["phase_map_output_dir"]) != resolve_review_path(phase_map_output_dir)):
        raise ValueError("Deferred presentation native identity changed")
    native = _read(native_path)
    count = int(descriptor["frame_count"])
    with Path(per_frame_csv).open(encoding="utf-8-sig", newline="") as stream:
        times = [float(row["elapsed_h"]) for row in csv.DictReader(stream)]
    expected = [float(row["elapsed_h"]) for row in native["frame_sources"]]
    if count < 1 or len(expected) != count or int(native["frame_count"]) != count or times != expected:
        raise ValueError("Deferred presentation has another or incomplete timeline")
    if not np.isfinite(times).all():
        raise ValueError("Deferred presentation has a nonfinite timeline")
    if int((summary.get("calibration_review_movie") or {}).get("calibration_frame_count", 0)) != count:
        raise ValueError("Deferred presentation frame count changed")
    return descriptor, native, times


def ensure_candidate_presentation(candidate, kind="plot", progress=None):
    """Build one explicitly requested display asset, leaving scientific data untouched."""
    if kind not in {"plot", "video"}:
        raise ValueError("Presentation kind must be plot or video")
    from darsia.fluidflower.tools.phase_reference_maps import _worker_lock
    target = Path(candidate.mass_plot if kind == "plot" else candidate.contour_movie)
    summary_path = Path(candidate.summary_path)
    summary = _read(summary_path)
    if not summary.get("deferred_presentation"):
        if not target.is_file():
            raise FileNotFoundError(target)
        return target
    descriptor, native, times = validate_deferred_presentation(summary, run=candidate.run,
        candidate_id=candidate.candidate_id, params_sha256=candidate.params_sha256,
        review_output_dir=candidate.review_output_dir, phase_map_output_dir=candidate.phase_map_output_dir,
        per_frame_csv=candidate.per_frame_csv)
    identity = sha256_file(summary_path)
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    with _worker_lock(target.with_suffix(target.suffix + ".lock")):
        if receipt_path.is_file():
            receipt = _read(receipt_path)
            if (receipt.get("summary_sha256") != identity or not target.is_file()
                    or receipt.get("sha256") != sha256_file(target)):
                raise ValueError("Generated presentation asset changed")
            return target
        temporary = target.with_name(target.stem + ".building-" + uuid.uuid4().hex + target.suffix)
        try:
            if progress:
                progress(f"Preparing {candidate.run.upper()} {'area diagnostics' if kind == 'plot' else 'video'}")
            job = {"run": candidate.run, "dataset": candidate.dataset,
                   "review_output_dir": str(candidate.review_output_dir)}
            if kind == "plot":
                counts, area = descriptor.get("frame_counts"), descriptor.get("area_denominator")
                domain = None
                if not area or not counts:
                    domain, resolved_area = _area_domain(Path(descriptor["model_root"]), job, native)
                    area = area or resolved_area
                if not counts:
                    counts = []
                    x, y, width, height = native["content_box_xywh"]
                    for index in range(len(times)):
                        if progress:
                            progress(f"Reading phase areas {index+1}/{len(times)}")
                        with Image.open(_presentation_layer(candidate, index)) as image:
                            if image.size != tuple(native["canvas_wh"]):
                                raise ValueError("Phase-map geometry changed")
                            rgba = np.asarray(image.convert("RGBA"))[y:y+height, x:x+width]
                        visible = (rgba[..., 3] > 0) & domain
                        counts.append({phase: int((visible & np.all(rgba[..., :3] == color[:3], axis=-1)).sum())
                                       for phase, color in phase_validation.PHASE_FILL_COLORS.items()})
                _plot_area(temporary, times, counts, area, candidate.run)
            else:
                movie = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 2., (1280, 720))
                try:
                    if not movie.isOpened():
                        raise RuntimeError("Cannot create prediction video")
                    for index in range(len(times)):
                        if progress:
                            progress(f"Encoding video {index+1}/{len(times)}")
                        photo_path = Path(candidate.review_output_dir) / "native_frames" / candidate.run / f"frame_{index:03d}.png"
                        if not photo_path.is_file():
                            photo_path = photo_path.with_suffix(".jpg")
                        with Image.open(_presentation_layer(candidate, index)) as layer, Image.open(photo_path) as photo:
                            if layer.size != tuple(native["canvas_wh"]) or photo.size != layer.size:
                                raise ValueError("Prediction layer and photo have different native geometry")
                            image = Image.alpha_composite(photo.convert("RGBA"), layer.convert("RGBA")).convert("RGB")
                            image.thumbnail((1280, 720))
                            canvas = Image.new("RGB", (1280, 720))
                            canvas.paste(image, ((1280-image.width)//2, (720-image.height)//2))
                            movie.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
                finally:
                    movie.release()
            if not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError("Presentation produced no output")
            if progress:
                progress("Publishing requested presentation")
            if sha256_file(summary_path) != identity:
                raise ValueError("Candidate changed while preparing presentation")
            temporary.replace(target)
            phase_validation._atomic_json(receipt_path, {"summary_sha256": identity, "sha256": sha256_file(target)})
        finally:
            temporary.unlink(missing_ok=True)
    return target


def _presentation_layer(candidate, index):
    return Path(candidate.phase_map_output_dir) / "native_phase_maps" / candidate.run / (
        "A_" + candidate.params_sha256[:16]) / f"frame_{index:03d}.png"


def generate_review_media(model_root, output_root, manifest, plan, progress=None):
    """Replace every inherited movie/plot with media made from these predictions."""
    media = []
    for job in manifest["jobs"]:
        run = job["run"]
        if (run, job["dataset"]) != (plan["run"], plan["dataset"]):
            raise ValueError("Generated review media contains another experiment")
        display = Path(job["review_output_dir"])
        native = _read(display / "native_frames" / run / "manifest.json")
        frame_count = int(native["frame_count"])
        rows = native.get("frame_sources", [])
        if frame_count < 1 or len(rows) != frame_count:
            raise ValueError("Transfer media requires a complete native timeline")
        times = [float(row["elapsed_h"]) for row in rows]
        if not np.isfinite(times).all():
            raise ValueError("Transfer timeline contains nonfinite elapsed times")
        domain, area = _area_domain(model_root, job, native)
        x, y, width, height = native["content_box_xywh"]
        canvas_width, canvas_height = native["canvas_wh"]
        candidate = output_root / "postflight" / run
        candidate.mkdir(parents=True, exist_ok=False)
        counts = []
        movie = cv2.VideoWriter(str(candidate / "calibration_review.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 2., (1280, 720))
        try:
            if not movie.isOpened():
                raise RuntimeError("Cannot create transfer prediction movie")
            for index in range(frame_count):
                layer_path = Path(job["phase_map_output_dir"]) / "native_phase_maps" / run / (
                    f"A_{job['params_sha256'][:16]}") / f"frame_{index:03d}.png"
                photo_path = display / "native_frames" / run / f"frame_{index:03d}.png"
                if not photo_path.is_file():
                    photo_path = photo_path.with_suffix(".jpg")
                with Image.open(layer_path) as raw_layer, Image.open(photo_path) as photo:
                    layer = raw_layer.convert("RGBA")
                    if layer.size != (canvas_width, canvas_height) or photo.size != layer.size:
                        raise ValueError("Prediction layer and target photo have different native geometry")
                    rgba = np.asarray(layer)[y:y+height, x:x+width]
                    visible = (rgba[..., 3] > 0) & domain
                    counts.append({phase: int((visible & np.all(rgba[..., :3] == color[:3], axis=-1)).sum())
                        for phase, color in phase_validation.PHASE_FILL_COLORS.items()})
                    image = Image.alpha_composite(photo.convert("RGBA"), layer).convert("RGB")
                    image.thumbnail((1280, 720))
                    canvas = Image.new("RGB", (1280, 720))
                    canvas.paste(image, ((1280-image.width)//2, (720-image.height)//2))
                    movie.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        finally:
            movie.release()
        movie_path = candidate / "calibration_review.mp4"
        if not movie_path.is_file() or movie_path.stat().st_size == 0:
            raise RuntimeError("Transfer movie encoder produced no output")
        # The compatibility filename is required by the viewer; content is area,
        # and both plot labels and summary explicitly exclude mass/concentration.
        _plot_area(candidate / "calibration_mass_fit.png", times, counts, area, run)
        with (candidate / "calibration_review_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["elapsed_h", "aqueous_pixels", "gas_pixels", "mixed_pixels"])
            writer.writeheader()
            for elapsed, count in zip(times, counts):
                writer.writerow({"elapsed_h": elapsed, **{key+"_pixels": value for key, value in count.items()}})
        phase_validation._atomic_json(candidate / "summary.json", {
            "selection_candidate_id": job["candidate_id"], "source_model_id": plan["source_model_id"],
            "source_model_sha256": plan["source_model_sha256"],
            "review_plot_title": f"{run.upper()} predicted 2D phase areas; no mass analysis",
            "area_denominator": area, "training_performed": False,
            "calibration_review_movie": {"params_sha256": job["params_sha256"], "fps": 2,
                "calibration_frame_count": frame_count, "plot_seconds": 0, "image_seconds_per_calibration_frame": .5},
            **_exposure(plan)})
        job["candidate_folder"] = str(candidate)
        job["quality_flags"] = _exposure(plan)
        # Existing source jobs can have explicit legacy media links as well.
        for field in ("calibration_review_movie", "review_movie", "movie_path", "review_plot", "plot_path"):
            job.pop(field, None)
        media.append(dict(run=run, frames=frame_count, area_denominator=area,
            candidate_folder=str(candidate), frame_counts=counts, elapsed_h=times))
        if progress:
            progress(f"{run.upper()}: new phase-area plot and {frame_count}-frame prediction movie ready")
    return media


def generate_production_metadata(output_root, manifest, plan):
    """Bind mass inputs to this timeline without inheriting old review media."""
    frame_count = 0
    for job in manifest["jobs"]:
        run = job["run"]
        if (run, job["dataset"]) != (plan["run"], plan["dataset"]):
            raise ValueError("Production output contains another experiment")
        native = _read(Path(job["review_output_dir"]) / "native_frames" / run / "manifest.json")
        frame_count = int(native["frame_count"])
        rows = native.get("frame_sources", [])
        if frame_count < 1 or len(rows) != frame_count:
            raise ValueError("Production output requires a complete native timeline")
        times = [float(row["elapsed_h"]) for row in rows]
        if not np.isfinite(times).all():
            raise ValueError("Production timeline contains nonfinite elapsed times")
        for index in range(frame_count):
            layer = Path(job["phase_map_output_dir"]) / "native_phase_maps" / run / (
                f"A_{job['params_sha256'][:16]}") / f"frame_{index:03d}.png"
            with Image.open(layer) as image:
                if image.size != tuple(native["canvas_wh"]):
                    raise ValueError("Production map and target have different native geometry")
        candidate = output_root / "postflight" / run
        candidate.mkdir(parents=True, exist_ok=False)
        with (candidate / "calibration_review_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["elapsed_h"])
            writer.writeheader()
            writer.writerows({"elapsed_h": value} for value in times)
        phase_validation._atomic_json(candidate / "summary.json", {
            "selection_candidate_id": job["candidate_id"], "params_sha256": job["params_sha256"],
            "source_model_id": plan["source_model_id"], "source_model_sha256": plan["source_model_sha256"],
            "output_mode": "production", "frame_count": frame_count, "training_performed": False,
            "review_media_generated": False, **_exposure(plan)})
        job.update(candidate_folder=str(candidate), quality_flags=_exposure(plan), output_mode="production")
        for field in ("calibration_review_movie", "review_movie", "movie_path", "review_plot", "plot_path"):
            job.pop(field, None)
    return frame_count


def _relocate(value, old, new):
    if isinstance(value, dict):
        return {key: _relocate(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item, old, new) for item in value]
    if isinstance(value, str):
        for prefix, replacement in ((str(old), str(new)), (old.as_posix(), new.as_posix())):
            if value == prefix or any(value.startswith(prefix+separator) for separator in ("/", "\\")):
                return replacement + value[len(prefix):]
    return value


def _validate_parity(report, frame_count):
    audit = report.get("verify_numerics", str(report.get("backend", "")).endswith("-checked"))
    if str(report.get("frame_execution") or "").startswith("shared-optical-frame-workers-v"):
        frames = [frame for run in report["runs"] for frame in run["frames"]]
        if sorted(frame["frame"] for frame in frames) != list(range(frame_count)):
            raise ValueError("Shared frame execution is incomplete or duplicated")
        if not audit:
            return dict(status="not-requested", checked_frames=[], separate_backend_comparison=False)
        checked = set()
        gpu_count = 0
        for frame in frames:
            if frame["backend"] == "cpu":
                continue  # Exact predict_image_tiled reference implementation.
            if frame["backend"] != "cuda" or not frame.get("cuda_validation"):
                raise ValueError("Shared GPU frame lacks its CPU comparison")
            gpu_count += 1
            check = frame["cuda_validation"]
            heads = check.get("heads") or {}
            if not {"co2_presence_probability", "gas_probability_conditional_on_co2"} <= heads.keys():
                raise ValueError("Shared GPU comparison is missing fitted heads")
            for values in heads.values():
                if values["q99_abs_error"] > .001 or values["disagreement"] > .0001:
                    raise ValueError("Shared GPU comparison failed")
            checked.add(check["frame"])
        return dict(status="passed", cpu_frames=frame_count-gpu_count, gpu_frames=gpu_count,
            checked_frames=sorted(checked), q99_tolerance=.001, label_disagreement_tolerance=.0001,
            policy="First GPU frame per frozen model/baseline/code, plus GPU frames 0, 1 and 10")
    if report["backend"] == "cpu-reference-tiled":
        return dict(status="reference-backend", reference="phase_validation.predict_image_tiled",
                    checked_frames=[], separate_backend_comparison=False)
    if len(report["runs"]) != 1 or len(report["runs"][0]["frames"]) != frame_count:
        raise ValueError("CUDA frame report does not cover the target timeline")
    if not audit:
        return dict(status="not-requested", checked_frames=[], separate_backend_comparison=False)
    checks = report["runs"][0]["parity"]
    if {row["frame"] for row in checks} != {index for index in (0, 1, 10) if index < frame_count}:
        raise ValueError("CUDA transfer lacks its required CPU reference checks")
    for row in checks:
        if set(row["heads"]) != {"co2_presence_probability", "gas_probability_conditional_on_co2"}:
            raise ValueError("CUDA reference check is missing a fitted head")
        for values in row["heads"].values():
            if not all(np.isfinite(value) for value in values.values()) or (
                    values["q99_abs_error"] > .001 or values["disagreement"] > .0001):
                raise ValueError("CUDA transfer prediction parity failed")
    return dict(status="passed", checked_frames=[row["frame"] for row in checks],
                q99_tolerance=.001, label_disagreement_tolerance=.0001,
                applies_to="optical scores; physical rules use the same CPU runtime")


def materialize_transfer_predictions(plan, progress=None, backend="auto", *, output_mode="review",
                                     eager_presentation=False, verify_numerics=False):
    """Materialize one frozen target, optionally omitting Studio presentation."""
    from darsia.fluidflower.tools.phase_reference_readiness import (
        RECEIPT, mark_references_pending, prepare_validation_references,
    )
    if output_mode not in ("review", "production"):
        raise ValueError("Phase output mode must be review or production")
    root = _verify_plan(plan)
    destination = root / "validation"
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("Transfer backend must be auto, cpu or cuda")
    if destination.exists():
        receipt = _read(destination / "transfer_materialization.json")
        if (receipt["evaluation_model_sha256"] != plan["evaluation_model_sha256"] or
                receipt["run"] != plan["run"]):
            raise ValueError("Existing transfer output belongs to another evaluation")
        if receipt.get("output_mode", "review") != output_mode:
            raise ValueError("Existing transfer output used another output mode; prepare a new evaluation")
        selected = "cuda" if receipt["backend"].startswith("cuda") else "cpu"
        if backend != "auto" and backend != selected:
            raise ValueError("Existing transfer output used another backend; prepare a new evaluation")
        if verify_numerics and not receipt.get("verify_numerics", receipt["backend"].endswith("-checked")):
            raise ValueError("Existing transfer output was not numerically audited; prepare a new evaluation")
        for name, digest in receipt["generated_sha256"].items():
            path = (destination / name).resolve()
            if not path.is_relative_to(destination) or not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Generated transfer asset changed: {name}")
        if eager_presentation and output_mode == "review":
            from darsia.fluidflower.tools.calibration_pairwise_review import _candidate_from_job
            for job in _read(destination / "postflight_manifest.json")["jobs"]:
                candidate = _candidate_from_job(destination, job)
                for kind in ("plot", "video"):
                    ensure_candidate_presentation(candidate, kind, progress)
        prepare_validation_references(destination, progress)
        return _read(destination / "validation_binding.json")
    model = HierarchicalPhaseModel.load(root, adapter="generic")
    choice = select_backend(model, backend, verify_numerics=verify_numerics)
    if progress:
        progress(f"Transfer inference: {choice['backend']}" + (
            f"; CPU fallback: {choice['fallback_reason']}" if choice["fallback_reason"] else ""))
    paths = [Path(__file__).resolve(), tool_path("materialize_phase_iteration.py").resolve(),
        tool_path("phase_inference_benchmark_backends.py").resolve(),
        Path(phase_validation.__file__).resolve(),
        tool_path("phase_movie_facies_cache.py").resolve(),
        tool_path("phase_reference_single_pass.py").resolve(),
        tool_path("phase_decision_execution_cache.py").resolve(),
        tool_path("phase_decision_pipeline.py").resolve()]
    code_hashes = {str(path): sha256_file(path) for path in paths}
    provenance = dict(choice, materializer_sha256=code_hashes[str(paths[0])],
        checked_cuda_sha256=code_hashes[str(paths[1])], backend_sha256=code_hashes[str(paths[2])],
        phase_validation_sha256=code_hashes[str(paths[3])],
        execution_cache_sha256=code_hashes[str(paths[4])],
        single_pass_reference_sha256=code_hashes[str(paths[5])],
        decision_execution_cache_sha256=code_hashes[str(paths[6])],
        decision_pipeline_sha256=code_hashes[str(paths[7])],
        verify_numerics=bool(verify_numerics),
        cpu_reference_frames=[0, 1, 10] if verify_numerics and choice["backend"].startswith("cuda") else [])
    if verify_numerics:
        provenance.update(q99_tolerance=.001, label_disagreement_tolerance=.0001)
    report = dict(created_utc=utc_now(), runs=[], output_mode=output_mode, verify_numerics=bool(verify_numerics),
        **choice, **_exposure(plan),
        source_model_id=plan["source_model_id"], source_model_sha256=plan["source_model_sha256"],
        evaluation_model_id=plan["evaluation_model_id"], evaluation_model_sha256=plan["evaluation_model_sha256"],
        target_run=plan["run"], target_baseline=plan["target_baseline"], physics_runtime=plan["physics_runtime"],
        training_performed=False, inference_provenance=provenance, code_sha256=code_hashes)
    work = root / f"validation-building-{uuid.uuid4().hex}"
    cache_handles = []
    def install_execution_cache(context):
        from darsia.fluidflower.tools.phase_movie_facies_cache import (
            install_facies_cache, install_material_cache, install_support_cache,
        )
        # Bit-identical by construction and self-checked against the frozen function at run time.
        from darsia.fluidflower.tools.phase_decision_execution_cache import install_reference_evidence_cache
        for installer in (install_facies_cache, install_material_cache, install_support_cache,
                          install_reference_evidence_cache):
            cache_handles.append(installer(context))

    decision_executors = []
    def decision_executor_factory(physics, predictor):
        # Identical values: only the waiting between a photograph's final decision and the next
        # photograph's history disappears. Self-checked against the frozen method at run time.
        from darsia.fluidflower.tools.phase_decision_pipeline import create
        executor = create(physics, slot=getattr(predictor, "decision_slot", None))
        if executor is not None:
            decision_executors.append(executor)
        return executor

    factory = None
    if choice["backend"].startswith("cuda"):
        def factory(selected, baseline):
            predictor = CheckedCuda(selected, baseline, report, verify_numerics=verify_numerics)
            predictor.run_report["run"] = plan["run"]
            return predictor
    from darsia.fluidflower.tools.phase_frame_workers import ENV as FRAME_ENV, ParallelFramePredictor, VERSION as FRAME_VERSION
    if os.environ.get(FRAME_ENV) and not verify_numerics:
        provenance["frame_execution"] = FRAME_VERSION
        provenance["cpu_reference_frames"] = []
        provenance["cpu_reference_policy"] = "Numerical audit not requested"
        report["frame_execution"] = FRAME_VERSION
        def factory(selected, baseline):
            return ParallelFramePredictor(root, baseline, report,
                "cuda" if choice["backend"].startswith("cuda") else "cpu",
                root / "optical_frame_cache")
    def reference_factory(*, run_id, dataset, review_root, native_manifest,
                          destination, predictor, model, baseline, canvas_shape,
                          content_box, pipeline_sha256, model_sha256):
        # A separately calibrated radiometry profile requires the original
        # reference worker until its display/input parity has been tested.
        from darsia.fluidflower.phase_training.radiometry import model_profile
        if model_profile(model) is not None:
            return None
        selected = injection_end_references({"run": run_id, "dataset": dataset}, review_root)
        # A missing acquisition must not prevent the experiment's phase maps
        # from being generated. The readiness receipt records that limitation.
        shared_frames = isinstance(predictor, ParallelFramePredictor)
        if not shared_frames and any(not row.get("available") for row in selected):
            return None
        baseline_photo = Path(review_root) / "native_references" / run_id / "pre_injection.jpg"
        baseline_meta = native_manifest.get("reference") or {}
        baseline_time = baseline_meta.get("elapsed_h")
        if baseline_meta.get("timeline_frame_index") is not None:
            baseline_time = native_manifest["frame_sources"][int(baseline_meta["timeline_frame_index"])]["elapsed_h"]
        baseline_time = float(baseline_time) if baseline_time is not None else -1e-6
        endpoints = ([Endpoint("baseline", baseline_time, baseline_photo, reference_sha256(baseline_photo))]
                     if shared_frames else [])
        for row in selected:
            if not row.get("available"):
                continue
            if row.get("frame_index") is not None:
                continue  # The existing reviewed phase map is already exact.
            if not row.get("available") or row.get("elapsed_h") is None:
                raise ValueError(f"Unprepared injection-end reference: {run_id}/{row['id']}")
            photo = resolve_reference_asset(row, review_root, run_id)
            endpoints.append(Endpoint(str(row["id"]), float(row["elapsed_h"]),
                                      photo, reference_sha256(photo)))
        if not endpoints:
            return None
        if progress is not None:
            progress(f"{run_id.upper()}: evaluating {len(endpoints)} injection-end references in one pass")
        def predict_reference(current, previous, reference_id):
            if predictor is None or isinstance(predictor, ParallelFramePredictor):
                return phase_validation.predict_image_tiled(model, current, baseline,
                                                            previous_rgb=previous)
            if not isinstance(predictor, CheckedCuda):
                raise ValueError("Single-pass references require a CUDA predictor")
            if verify_numerics:
                reference_report = {"runs": []}
                result = predictor.predict_reference(current, previous, reference_report)
                report.setdefault("reference_runs", []).extend(reference_report["runs"])
                return result
            gpu = predictor.gpu
            saved_previous, saved_key = gpu.previous, gpu.previous_key
            try:
                scores = gpu.predict(current, previous,
                                     current_key=("reference", reference_id),
                                     previous_key=predictor.frame - 1)
            finally:
                gpu.previous, gpu.previous_key = saved_previous, saved_key
            return partition(np.stack([
                scores["co2_presence_probability"],
                scores["gas_probability_conditional_on_co2"],
            ], axis=-1))
        observer = SinglePassReferences(endpoints, run=run_id, destination=destination,
            canvas_shape=canvas_shape, content_box=content_box,
            pipeline_sha256=pipeline_sha256, model_sha256=model_sha256,
            predict_reference=predict_reference, output_mode=output_mode)
        if shared_frames:
            observer.initial_reference_time = baseline_time
        return observer
    try:
        try:
            from darsia.fluidflower.tools import phase_memory_defaults
            with phase_memory_defaults.scoped() as memory_defaults:      # a small machine keeps the old memory profile
                report["memory_defaults"] = memory_defaults
                binding = phase_validation.materialize_phase_validation_campaign(root, root / "review_source", work,
                    progress=progress, predictor_factory=factory, inference_provenance=provenance,
                    execution_cache_factory=install_execution_cache,
                    reference_factory=reference_factory, output_mode=output_mode,
                    decision_executor_factory=decision_executor_factory)
        finally:
            for handle in reversed(cache_handles):
                handle.close()
            report["execution_caches"] = [handle.report() for handle in cache_handles
                                          if hasattr(handle, "report")]
            for executor in decision_executors:
                executor.close()                    # no-op after a normal run; releases threads after a failure
            report["decision_execution"] = [executor.report() for executor in decision_executors]
        if (binding["model_sha256"] != plan["evaluation_model_sha256"] or
                set(binding["selected_candidates"]) != {plan["run"]} or
                binding.get("physics_runtime") != plan["physics_runtime"]):
            raise ValueError("Materialized transfer changed model, target or physical runtime")
        manifest = _read(work / "postflight_manifest.json")
        if output_mode == "review":
            report["media"] = (generate_review_media if eager_presentation else generate_review_metadata)(root, work, manifest, plan, progress)
            frame_count = report["media"][0]["frames"]
        else:
            report["media"] = []
            frame_count = generate_production_metadata(work, manifest, plan)
        report["prediction_parity"] = _validate_parity(report, frame_count)
        manifest.update(**_exposure(plan), training_performed=False, output_mode=output_mode)
        phase_validation._atomic_json(work / "postflight_manifest.json", manifest)
        report["completed_utc"] = utc_now()
        phase_validation._atomic_json(work / "inference_report.json", report)
        snapshot = work / "code_snapshot"
        snapshot.mkdir()
        for path in paths:
            shutil.copy2(path, snapshot / path.name)
            if sha256_file(path) != code_hashes[str(path)] or sha256_file(snapshot / path.name) != code_hashes[str(path)]:
                raise ValueError("Inference source changed during transfer materialization")
        for path in work.rglob("*.json"):
            phase_validation._atomic_json(path, _relocate(_read(path), work, destination))
        _verify_plan(plan)
        mark_references_pending(work)
        phase_validation._atomic_json(work / "transfer_materialization.json", {
            "format": "phase-transfer-materialization-v1", "evaluation_model_sha256": plan["evaluation_model_sha256"],
            "run": plan["run"], "backend": choice["backend"], "output_mode": output_mode, "completed_utc": utc_now(),
            "verify_numerics": bool(verify_numerics),
            "generated_sha256": {path.relative_to(work).as_posix(): sha256_file(path)
                for path in work.rglob("*") if path.is_file() and path.name != RECEIPT}})
        work.rename(destination)
    except Exception as error:
        if work.exists():
            phase_validation._atomic_json(work / "transfer_failure.json", dict(
                error=f"{type(error).__name__}: {error}", completed=False, **report))
        raise
    prepare_validation_references(destination, progress)
    return _read(destination / "validation_binding.json")
