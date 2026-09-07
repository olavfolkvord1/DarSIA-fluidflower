"""Materialize model-bound phase-review assets from corrected source frames.

The source review campaign is treated as immutable evidence.  Generated phase
maps, active-learning proposals, and a compact compatibility manifest are
written to a model-specific cache that can be opened by the established
FluidFlower review application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_features import SPATIAL_CONTEXT_FRACTION
from darsia.fluidflower.phase_learning import (
    HierarchicalPhaseModel,
    propose_uncertain_phase_regions,
)
from darsia.fluidflower.project import sha256_file, utc_now


PHASE_VALIDATION_CAMPAIGN_FORMAT = "darsia-phase-validation-campaign-v1"
PHASE_REVIEW_PROPOSAL_SCHEMA = 2
PHASE_FILL_COLORS = {
    "aqueous": (49, 217, 160, 255),
    "mixed": (255, 191, 71, 255),
    "gas": (242, 76, 157, 255),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected one JSON object in {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _model_identity(model_root: Path) -> tuple[str, str]:
    summary = _read_json(model_root / "summary.json")
    model_id = str(summary.get("model_id") or model_root.name).strip()
    digest = hashlib.sha256()
    for path in (
        model_root / "summary.json",
        model_root / "models" / "generic" / "co2_presence_signal.json",
        model_root / "models" / "generic" / "gas_within_co2_signal.json",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return model_id, digest.hexdigest()


def _source_jobs(source_root: Path) -> list[dict[str, Any]]:
    manifest = _read_json(source_root / "postflight_manifest.json")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in manifest.get("jobs") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        run_id = str(row.get("run") or "").strip().casefold()
        if run_id:
            grouped.setdefault(run_id, []).append(row)
    selected: list[dict[str, Any]] = []
    for run_id, rows in sorted(grouped.items()):
        rows.sort(
            key=lambda row: (
                "physics" not in str(row.get("role") or "").casefold(),
                "reviewed" not in str(row.get("role") or "").casefold(),
                int(row.get("grid_order") or 999),
            )
        )
        selected.append({**rows[0], "run": run_id})
    if not selected:
        raise ValueError(f"No review candidates were found in {source_root}")
    return selected


def _candidate_paths(source_root: Path, job: Mapping[str, Any]) -> tuple[Path, Path]:
    run_id = str(job["run"]).casefold()
    variant = str(job.get("variant") or "")
    candidate = (
        _resolve(source_root, job["candidate_folder"])
        if job.get("candidate_folder")
        else (source_root / "postflight" / variant / run_id).resolve()
    )
    review_root = (
        _resolve(source_root, job["review_output_dir"])
        if job.get("review_output_dir")
        else (source_root / "manual_pairwise_review").resolve()
    )
    return candidate, review_root


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def predict_image_tiled(
    model: HierarchicalPhaseModel,
    image_rgb: np.ndarray,
    baseline_rgb: np.ndarray,
    *,
    previous_rgb: np.ndarray | None = None,
    clearwater_rgb: np.ndarray | None = None,
    tile_height: int = 384,
    texture_sigma: float = 2.0,
    radiometry_profile: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Apply one optical model in bounded-memory horizontal tiles."""

    current = np.asarray(image_rgb)
    baseline = np.asarray(baseline_rgb)
    if current.shape != baseline.shape or current.ndim != 3:
        raise ValueError("Current and baseline images must be matching RGB arrays")
    height, width = current.shape[:2]
    spatial_sigma = max(
        float(texture_sigma), SPATIAL_CONTEXT_FRACTION * float(min(height, width))
    )
    overlap = max(16, int(math.ceil(6.0 * spatial_sigma)))
    outputs: dict[str, np.ndarray] = {}
    for start in range(0, height, max(32, int(tile_height))):
        stop = min(height, start + max(32, int(tile_height)))
        extended_start = max(0, start - overlap)
        extended_stop = min(height, stop + overlap)
        selection = slice(extended_start, extended_stop)
        predicted = model.predict_image(
            current[selection],
            baseline[selection],
            previous_rgb=(
                None if previous_rgb is None else np.asarray(previous_rgb)[selection]
            ),
            clearwater_rgb=(
                None if clearwater_rgb is None else np.asarray(clearwater_rgb)[selection]
            ),
            texture_sigma=texture_sigma,
            spatial_context_sigma=spatial_sigma,
            **({"radiometry_profile": radiometry_profile} if radiometry_profile is not None else {}),
        )
        local = slice(start - extended_start, stop - extended_start)
        for name, values in predicted.items():
            output = outputs.setdefault(name, np.empty((height, width), dtype=np.float32))
            output[start:stop] = np.asarray(values[local], dtype=np.float32)
    return outputs


def _phase_masks(predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    co2 = np.asarray(predictions["co2_presence_probability"], dtype=np.float32)
    gas_share = np.asarray(
        predictions["gas_probability_conditional_on_co2"], dtype=np.float32
    )
    active = co2 >= np.float32(0.5)
    aqueous = active & (gas_share < np.float32(0.4))
    gas = active & (gas_share >= np.float32(0.6))
    return {"aqueous": aqueous, "mixed": active & ~(aqueous | gas), "gas": gas}


def _write_phase_maps(
    root: Path,
    run_id: str,
    fingerprint: str,
    frame_index: int,
    canvas_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
    predictions: Mapping[str, np.ndarray],
) -> None:
    masks = _phase_masks(predictions)
    canvas_height, canvas_width = canvas_shape
    x, y, width, height = content_box
    folder = root / "native_phase_maps" / run_id / f"A_{fingerprint[:16]}"
    combined = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    for phase, mask in masks.items():
        if mask.shape != (height, width):
            raise ValueError(
                f"Phase-map shape {mask.shape} differs from content box {(height, width)}"
            )
        local = np.zeros((height, width, 4), dtype=np.uint8)
        local[mask] = PHASE_FILL_COLORS[phase]
        combined[y : y + height, x : x + width][mask] = PHASE_FILL_COLORS[phase]
        path = folder / phase / f"frame_{frame_index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            _place_local(local, canvas_shape, x, y), "RGBA"
        ).save(path)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined, "RGBA").save(folder / f"frame_{frame_index:03d}.png")


def _place_local(
    local: np.ndarray, canvas_shape: tuple[int, int], x: int, y: int
) -> np.ndarray:
    canvas = np.zeros((*canvas_shape, 4), dtype=np.uint8)
    height, width = local.shape[:2]
    canvas[y : y + height, x : x + width] = local
    return canvas


def _canvas_proposals(
    predictions: Mapping[str, np.ndarray],
    *,
    run_id: str,
    frame_index: int,
    elapsed_h: float,
    model_id: str,
    candidate_id: str,
    fingerprint: str,
    canvas_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    raw = propose_uncertain_phase_regions(
        predictions,
        run_id=run_id,
        frame_index=frame_index,
        elapsed_h=elapsed_h,
        model_id=model_id,
        count=16,
    )
    canvas_height, canvas_width = canvas_shape
    content_x, content_y, content_width, content_height = content_box
    output: list[dict[str, Any]] = []
    for row in raw:
        local_x = float(row["x"])
        local_y = float(row["y"])
        local_width = float(row["width"])
        local_height = float(row["height"])
        canvas_x = (content_x + local_x * content_width) / canvas_width
        canvas_y = (content_y + local_y * content_height) / canvas_height
        canvas_w = local_width * content_width / canvas_width
        canvas_h = local_height * content_height / canvas_height
        output.append(
            {
                **row,
                "candidate": "A",
                "candidate_id": candidate_id,
                "params_sha256": fingerprint,
                "analysis_domain": "registered_visible_camera_plane_2d",
                "coordinate_space": "normalized_native_review_canvas_16x9",
                "shape": "box",
                "x": canvas_x,
                "y": canvas_y,
                "width": canvas_w,
                "height": canvas_h,
                "points": [
                    [canvas_x, canvas_y],
                    [canvas_x + canvas_w, canvas_y],
                    [canvas_x + canvas_w, canvas_y + canvas_h],
                    [canvas_x, canvas_y + canvas_h],
                ],
                "focus_box": {
                    "x": max(0.0, canvas_x - canvas_w),
                    "y": max(0.0, canvas_y - canvas_h),
                    "width": min(1.0, 3.0 * canvas_w),
                    "height": min(1.0, 3.0 * canvas_h),
                },
            }
        )
    return output


def materialize_phase_validation_campaign(
    model_root: Path,
    source_campaign_root: Path,
    output_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build one immutable-model validation campaign and return its binding."""

    model_root = Path(model_root).expanduser().resolve()
    source_root = Path(source_campaign_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    model_id, model_sha256 = _model_identity(model_root)
    model = HierarchicalPhaseModel.load(model_root, adapter="generic")
    from darsia.fluidflower.phase_training.radiometry import model_profile, RunRadiometry, runtime_identity
    profile = model_profile(model)
    radiometry_runtime = None if profile is None else runtime_identity()
    source_manifest_sha256 = sha256_file(source_root / "postflight_manifest.json")
    binding_path = destination / "validation_binding.json"
    if binding_path.is_file():
        cached = _read_json(binding_path)
        if (
            cached.get("model_sha256") == model_sha256
            and cached.get("source_manifest_sha256") == source_manifest_sha256
            and (destination / "postflight_manifest.json").is_file()
            and cached.get("radiometry_runtime") == radiometry_runtime
        ):
            return cached

    jobs: list[dict[str, Any]] = []
    selected_candidates: dict[str, str] = {}
    for source_job in _source_jobs(source_root):
        run_id = str(source_job["run"]).casefold()
        dataset = str(source_job.get("dataset") or "").casefold()
        if not dataset:
            dataset = "ffac" if run_id.startswith("ac") else "ffum"
        candidate_root, review_root = _candidate_paths(source_root, source_job)
        review_root = Path(str(source_job.get("analysis_source_review_root") or review_root))
        from darsia.fluidflower.phase_training.radiometry import analysis_review_root
        review_root = analysis_review_root(review_root, run_id)
        native_manifest = _read_json(review_root / "native_frames" / run_id / "manifest.json")
        frame_count = int(native_manifest.get("frame_count") or 0)
        canvas_width, canvas_height = (
            int(value) for value in native_manifest.get("canvas_wh") or []
        )
        content_x, content_y, content_width, content_height = (
            int(value) for value in native_manifest.get("content_box_xywh") or []
        )
        reference_path = review_root / "native_references" / run_id / "pre_injection.jpg"
        baseline_canvas = _load_rgb(reference_path)
        baseline = baseline_canvas[
            content_y : content_y + content_height,
            content_x : content_x + content_width,
        ]
        radiometry = None
        display_root = review_root
        if profile is not None:
            radiometry = RunRadiometry(
                profile, series_id=dataset, run_id=run_id, baseline_path=reference_path,
                content_box=(content_x, content_y, content_width, content_height),
                cache_root=destination / "radiometry_cache" / dataset / run_id,
                source_manifest=review_root / "native_frames" / run_id / "manifest.json",
            )
            baseline = radiometry.load(reference_path, "pre_injection")
            display_root = destination / "review_assets"
            corrected_canvas = baseline_canvas.copy()
            corrected_canvas[content_y:content_y + content_height, content_x:content_x + content_width] = np.rint(baseline * 255).astype(np.uint8)
            for folder in ("native_references", "reference_frames"):
                path = display_root / folder / run_id / "pre_injection.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(corrected_canvas).save(path)
        candidate_id = f"{model_id}__{dataset}__{run_id}"
        all_proposals: list[dict[str, Any]] = []
        previous: np.ndarray | None = None
        frame_sources = [
            dict(row)
            for row in native_manifest.get("frame_sources") or []
            if isinstance(row, Mapping)
        ]
        for frame_index in range(frame_count):
            if progress is not None:
                progress(f"{run_id.upper()} frame {frame_index + 1}/{frame_count}")
            frame_path = review_root / "native_frames" / run_id / f"frame_{frame_index:03d}.jpg"
            frame_canvas = _load_rgb(frame_path)
            current = frame_canvas[
                content_y : content_y + content_height,
                content_x : content_x + content_width,
            ]
            if radiometry is not None:
                current = radiometry.load(frame_path, f"frame_{frame_index:03d}")
                corrected_canvas = frame_canvas.copy()
                corrected_canvas[content_y:content_y + content_height, content_x:content_x + content_width] = np.rint(current * 255).astype(np.uint8)
                path = display_root / "native_frames" / run_id / f"frame_{frame_index:03d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(corrected_canvas).save(path)
            predictions = predict_image_tiled(
                model,
                current,
                baseline,
                previous_rgb=previous,
                radiometry_profile=profile,
            )
            _write_phase_maps(
                destination / "review_assets",
                run_id,
                model_sha256,
                frame_index,
                (canvas_height, canvas_width),
                (content_x, content_y, content_width, content_height),
                predictions,
            )
            elapsed_h = (
                float(frame_sources[frame_index].get("elapsed_h") or 0.0)
                if frame_index < len(frame_sources)
                else float(frame_index)
            )
            all_proposals.extend(
                _canvas_proposals(
                    predictions,
                    run_id=run_id,
                    frame_index=frame_index,
                    elapsed_h=elapsed_h,
                    model_id=model_id,
                    candidate_id=candidate_id,
                    fingerprint=model_sha256,
                    canvas_shape=(canvas_height, canvas_width),
                    content_box=(content_x, content_y, content_width, content_height),
                )
            )
            previous = current

        if radiometry is not None:
            _atomic_json(display_root / "native_frames" / run_id / "manifest.json", {
                **native_manifest,
                "radiometry_profile": profile,
                "analysis_source_review_root": str(review_root),
                "display_encoding": "8-bit lossless PNG; inference uses float32",
            })
            _atomic_json(display_root / "radiometry" / dataset / f"{run_id}.json", radiometry.report())

        proposal_path = (
            destination
            / "review_assets"
            / "phase_quiz"
            / run_id
            / f"A_{model_sha256[:16]}_q{PHASE_REVIEW_PROPOSAL_SCHEMA}"
            / "proposals.json"
        )
        _atomic_json(
            proposal_path,
            {
                "schema": PHASE_REVIEW_PROPOSAL_SCHEMA,
                "candidate_id": candidate_id,
                "model_id": model_id,
                "initial_proposal_count": min(24, len(all_proposals)),
                "proposal_count": len(all_proposals),
                "proposals": all_proposals,
            },
        )
        selected_candidates[run_id] = candidate_id
        jobs.append(
            {
                **source_job,
                "run": run_id,
                "dataset": dataset,
                "role": "selected_model",
                "variant": "selected_model_validation",
                "grid_order": 1,
                "candidate_id": candidate_id,
                "candidate_folder": str(candidate_root),
                "review_output_dir": str(display_root),
                "analysis_source_review_root": str(review_root),
                "radiometry_profile_sha256": None if profile is None else profile["sha256"],
                "phase_map_output_dir": str(destination / "review_assets"),
                "params_sha256": model_sha256,
            }
        )

    campaign_id = f"phase-validation-{model_id}"
    manifest = {
        "schema": 1,
        "format": PHASE_VALIDATION_CAMPAIGN_FORMAT,
        "campaign_id": campaign_id,
        "created_utc": utc_now(),
        "source_campaign_root": str(source_root),
        "source_manifest_sha256": source_manifest_sha256,
        "model_id": model_id,
        "model_root": str(model_root),
        "model_sha256": model_sha256,
        "models": {
            "selected_model": {
                "co2_path": str(
                    model_root / "models" / "generic" / "co2_presence_signal.json"
                ),
                "gas_path": str(
                    model_root / "models" / "generic" / "gas_within_co2_signal.json"
                ),
            }
        },
        "jobs": jobs,
    }
    _atomic_json(destination / "postflight_manifest.json", manifest)
    binding = {
        "radiometry_runtime": radiometry_runtime,
        "radiometry_profile": profile,
        "schema": 1,
        "format": PHASE_VALIDATION_CAMPAIGN_FORMAT,
        "campaign_id": campaign_id,
        "model_id": model_id,
        "model_root": str(model_root),
        "model_sha256": model_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "ffac_root": str(destination),
        "ffum_root": str(destination),
        "combined_root": str(destination),
        "selected_candidates": selected_candidates,
        "roles": ["selected_model"],
        "run_count": len(selected_candidates),
        "completed_utc": utc_now(),
    }
    _atomic_json(binding_path, binding)
    return binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binding = materialize_phase_validation_campaign(
        args.model_root,
        args.source_campaign_root,
        args.output_root,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(binding), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PHASE_VALIDATION_CAMPAIGN_FORMAT",
    "materialize_phase_validation_campaign",
    "predict_image_tiled",
]
