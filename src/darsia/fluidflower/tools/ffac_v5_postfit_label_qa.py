"""Source-exact FF_AC V5 native-map evidence QA; never selects a model.

The 4048 fitted labels are diagnostic. Independently supplied, disjoint later
labels are reported separately. Historical revisions without their own V5 map
are explicitly unscored; a same-run current map is never substituted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from darsia.fluidflower.phase_training import trainer
from darsia.fluidflower.phase_training import logistic
from darsia.fluidflower.phase_training.v5_linear_input import model_v5_settings
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import train_ffac_two_arm_optical_region as frozen
from darsia.fluidflower.tools import train_ffac_v5_raw_candidate as v5
from darsia.fluidflower.tools import train_phase_model_further as optical
from darsia.fluidflower.tools.phase_training_review_comparison import decode_phase


CONTRACT = "ffac-v5-postfit-source-exact-label-qa-v1"


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _original_manifest_sha(native: dict, own_sha: str) -> str:
    for key in ("v5_photometry", "two_arm_photometry"):
        source = (native.get(key) or {}).get("source_manifest_sha256")
        if source:
            return str(source).lower()
    return own_sha.lower()


def _source_frame_identity(source, region) -> tuple[str, str, str, dict]:
    native_path = Path(source.asset_manifest_path)
    native_sha = sha256_file(native_path)
    native = _read(native_path)
    index = int(region.frame_index)
    frames = native.get("frame_sources") or []
    _require(0 <= index < len(frames), f"Label frame absent: {region.region_id}")
    row = frames[index]
    _require(int(row["frame_index"]) == index
             and abs(float(row["elapsed_h"]) - float(region.elapsed_h)) <= 1e-5,
             f"Label time/frame differs from source: {region.region_id}")
    jpeg_sha = row.get("source_sha256") or row.get("source_raw_sha256")
    _require(isinstance(jpeg_sha, str) and len(jpeg_sha) == 64,
             f"Original camera JPEG not bound: {region.region_id}")
    return (region.run.casefold(), _original_manifest_sha(native, native_sha),
            jpeg_sha.lower(), native)


def _native_map_index(model: Path) -> dict:
    post = model / "validation" / "postflight_manifest.json"
    binding = model / "validation" / "validation_binding.json"
    _require(post.is_file() and binding.is_file(), "Model has no complete native validation")
    document, bound = _read(post), _read(binding)
    for key in ("model_id", "model_sha256", "physics_runtime"):
        _require(document.get(key) == bound.get(key), f"Model binding differs: {key}")
    jobs = document.get("jobs") or []
    _require(len(jobs) == 43, "Expected 43 canonical FF_AC runs")
    settings = model_v5_settings(model)
    _require(settings is not None, "Selected model has no V5 RAW input marker")
    receipt_path = Path(settings["materialization_receipt_path"])
    _require(sha256_file(receipt_path) == settings["materialization_receipt_sha256"],
             "V5 source materialization receipt changed")
    canonical = _read(receipt_path).get("canonical_mapping") or []
    _require(len(canonical) == 43, "V5 canonical receipt is incomplete")
    expected_native = {row["run"].casefold(): row["materialized_manifest_sha256"]
                       for row in canonical}
    _require(len(expected_native) == 43, "V5 canonical receipt is ambiguous")
    # Includes one combined and three separate full-canvas RGBA PNGs/frame.
    structure = v5.verify_native_maps(jobs)
    frames = {}
    for job in jobs:
        _root, path, native = optical.native_source(job)
        _require(sha256_file(path) == expected_native.get(str(job["run"]).casefold()),
                 f"V5 candidate source revision changed: {job['run']}")
        source_sha = (native.get("v5_photometry") or {}).get("source_manifest_sha256")
        _require(source_sha and native["v5_photometry"].get("same_frame_order_timestamps_and_native_canvas") is True,
                 f"Candidate lacks original V5 source identity: {job['run']}")
        count = int(native["frame_count"])
        _require(len(native.get("frame_sources") or []) == count,
                 f"Candidate frame count differs: {job['run']}")
        for index, row in enumerate(native["frame_sources"]):
            _require(int(row["frame_index"]) == index, "Candidate native timeline changed")
            original_jpeg_sha = row.get("source_sha256") or row.get("source_raw_sha256")
            _require(original_jpeg_sha, f"Original JPEG SHA absent: {job['run']}/{index}")
            key = (str(job["run"]).casefold(), source_sha.lower(), original_jpeg_sha.lower())
            _require(key not in frames, f"Ambiguous acquisition: {key}")
            folder = (Path(job["phase_map_output_dir"]) / "native_phase_maps" /
                      key[0] / ("A_" + job["params_sha256"][:16]))
            frames[key] = dict(map_path=str(folder / f"frame_{index:03d}.png"),
                frame_index=index, elapsed_h=float(row["elapsed_h"]),
                content_box=native["content_box_xywh"], canvas_wh=native["canvas_wh"],
                native_manifest_path=str(path), native_manifest_sha256=sha256_file(path))
    _require(len(frames) == 864, "Candidate has fewer than 864 distinct acquisitions")
    return dict(frames=frames, structure=structure,
                model_id=document["model_id"], model_sha256=document["model_sha256"],
                postflight_sha256=sha256_file(post), binding_sha256=sha256_file(binding))


def _crop_mask(annotation: dict, canvas_wh, content_box) -> tuple[slice, slice, np.ndarray]:
    """Use exact annotation_mask rounding while allocating only its bounding ROI."""
    canvas_width, canvas_height = map(float, canvas_wh)
    content_x, content_y, width, height = map(int, content_box)
    points = annotation.get("points") or []
    if not points:
        x, y = float(annotation["x"]), float(annotation["y"])
        box_width, box_height = float(annotation["width"]), float(annotation["height"])
        points = [(x, y), (x + box_width, y),
                  (x + box_width, y + box_height), (x, y + box_height)]
    polygon = np.asarray([(float(p[0]) * canvas_width - content_x,
                           float(p[1]) * canvas_height - content_y)
                          for p in points], np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    integer = np.rint(polygon).astype(np.int32)
    left, top = np.min(integer, axis=0)
    right, bottom = np.max(integer, axis=0)
    mask = np.zeros((bottom - top + 1, right - left + 1), np.uint8)
    cv2.fillPoly(mask, [integer - [left, top]], color=1)
    return slice(top, bottom + 1), slice(left, right + 1), mask.astype(bool)


def _obligations(phase: str) -> dict[str, str]:
    # "all" applies only to a whole-region claim. Mixed or uncertain positive
    # labels assert presence somewhere, not full-box gas/aqueous coverage.
    return {
        logistic.PHASE_WATER: {"gas": "none", "aqueous": "none"},
        logistic.PHASE_AQUEOUS: {"gas": "none", "aqueous": "all"},
        logistic.PHASE_GAS: {"gas": "all", "aqueous": "none"},
        logistic.PHASE_MIXED_GAS_AQUEOUS: {"gas": "some", "aqueous": "some"},
        logistic.PHASE_MIXED_AQUEOUS_WATER: {"gas": "none", "aqueous": "some"},
        logistic.PHASE_MIXED_ALL: {"gas": "some", "aqueous": "some"},
        logistic.PHASE_CO2_PRESENT_REGION: {"co2": "some"},
        logistic.PHASE_CO2_ABSENT_REGION: {"gas": "none", "aqueous": "none"},
        logistic.PHASE_GAS_PRESENT_REGION: {"gas": "some"},
        logistic.PHASE_AQUEOUS_PRESENT_REGION: {"aqueous": "some"},
        logistic.PHASE_GAS_ABSENT_REGION: {"gas": "none"},
        logistic.PHASE_AQUEOUS_ABSENT_REGION: {"aqueous": "none"},
        logistic.PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: {"gas": "none", "aqueous": "some"},
        logistic.PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: {},
    }.get(phase, {})


def _assess(region, source, frame: dict, labels: np.ndarray) -> dict:
    ys, xs, mask = _crop_mask(region.payload, frame["canvas_wh"], frame["content_box"])
    size = int(np.count_nonzero(mask))
    _require(size > 0, f"Empty native label polygon: {region.region_id}")
    values = labels[ys, xs][mask]
    counts = np.bincount(values, minlength=4)
    present = dict(gas=int(counts[2] + counts[3]),
                   aqueous=int(counts[1] + counts[3]),
                   co2=int(counts[1] + counts[2] + counts[3]))
    checks = {}
    for name, quantifier in _obligations(region.phase_class).items():
        pixels = present[name]
        checks[name] = dict(claim=quantifier, predicted_pixels=pixels,
            predicted_fraction=pixels / size,
            satisfied=(pixels == size if quantifier == "all" else
                       pixels == 0 if quantifier == "none" else pixels > 0))
    return dict(annotation_id=region.region_id, run=region.run,
        frame_index=region.frame_index, elapsed_h=region.elapsed_h,
        phase_class=region.phase_class, source=region.source,
        source_manifest_sha256=sha256_file(source.asset_manifest_path),
        original_source_manifest_sha256=_original_manifest_sha(
            _read(source.asset_manifest_path), sha256_file(source.asset_manifest_path)),
        candidate_native_manifest_sha256=frame["native_manifest_sha256"],
        map_path=frame["map_path"], map_sha256=frame["map_sha256"],
        polygon_pixels=size, phase_pixel_counts=counts.tolist(), checks=checks)


def _regions(request_path: Path):
    request = _read(request_path)
    pairs = frozen._frozen_pairs(request)
    sources = {}
    _, regions, _ = trainer._load_sources_and_regions(
        pairs, preserve_native_revisions=True, region_sources=sources)
    _require(set(sources) == {r.region_id for r in regions},
             "Region-source binding is incomplete")
    return regions, sources


def _score_cohort(regions, sources, frame_index: dict) -> tuple[list, list]:
    scored, unscored = [], []
    grouped = defaultdict(list)
    for region in regions:
        source = sources[region.region_id]
        run, original_sha, original_jpeg_sha, _native = _source_frame_identity(source, region)
        key = (run, original_sha, original_jpeg_sha)
        frame = frame_index.get(key)
        if frame is None:
            unscored.append(dict(annotation_id=region.region_id, run=run,
                source_manifest_sha256=sha256_file(source.asset_manifest_path),
                original_source_manifest_sha256=original_sha,
                original_camera_jpeg_sha256=original_jpeg_sha,
                reason="No candidate map for this exact original acquisition/revision"))
            continue
        _require(frame["frame_index"] == region.frame_index
                 and abs(frame["elapsed_h"] - region.elapsed_h) <= 1e-5,
                 f"Exact-acquisition label frame/time mismatch: {region.region_id}")
        grouped[frame["map_path"]].append((region, source, frame))
    # Bound resident memory to one native 16-Mpixel map rather than retaining
    # up to 864 decoded maps (roughly 14 GB) across 4048 labels.
    for path, group in grouped.items():
        frame = group[0][2]
        labels = decode_phase(path, frame["content_box"])
        frame["map_sha256"] = sha256_file(Path(path))
        scored.extend(_assess(region, source, bound, labels)
                      for region, source, bound in group)
    return scored, unscored


def _summarize(scored: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in scored:
        for name, check in row["checks"].items():
            grouped[(name, check["claim"])].append((row["run"], row["polygon_pixels"], check))
    result = {}
    for (name, claim), rows in sorted(grouped.items()):
        fractions = [x[2]["predicted_fraction"] for x in rows]
        pixel_numer = sum(x[1] * x[2]["predicted_fraction"] for x in rows)
        pixels = sum(x[1] for x in rows)
        result[f"{name}/{claim}"] = dict(regions=len(rows), runs=len({x[0] for x in rows}),
            satisfied=sum(bool(x[2]["satisfied"]) for x in rows),
            polygon_macro_predicted_fraction=float(np.mean(fractions)),
            pixel_weighted_predicted_fraction=float(pixel_numer / pixels),
            interpretation=("whole-region pixel recall" if claim == "all" else
                            "whole-region false-area fraction" if claim == "none" else
                            "presence-only; pixel coverage is descriptive"))
    return result


def run(model: Path, *, independent_request: Path | None = None) -> dict:
    indexed = _native_map_index(model)
    regions, sources = _regions(model / "training_request.json")
    _require(len(regions) == 4048, "Fitted V5 cohort no longer has 4048 effective labels")
    training, historical = _score_cohort(regions, sources, indexed["frames"])
    independent, independent_unscored = [], []
    if independent_request is not None:
        more, more_sources = _regions(independent_request)
        _require(not ({r.region_id for r in regions} & {r.region_id for r in more}),
                 "Independent review request overlaps fitted annotation IDs")
        independent, independent_unscored = _score_cohort(
            more, more_sources, indexed["frames"])
    return dict(contract=CONTRACT, model_path=str(model),
        model_id=indexed["model_id"], model_sha256=indexed["model_sha256"],
        postflight_sha256=indexed["postflight_sha256"],
        validation_binding_sha256=indexed["binding_sha256"],
        native_map_structure=indexed["structure"],
        fitted_request_sha256=sha256_file(model / "training_request.json"),
        independent_request_path=None if independent_request is None else str(independent_request),
        independent_request_sha256=None if independent_request is None else sha256_file(independent_request),
        training_seen=dict(total=len(regions), scored=len(training),
            historical_unscored=len(historical), summary=_summarize(training),
            regions=training, unscored=historical),
        independent_postfit=dict(supplied=independent_request is not None,
            scored=len(independent), unscored=len(independent_unscored),
            summary=_summarize(independent), regions=independent,
            missing=independent_unscored),
        independent_scientific_acceptance=False,
        note="Training-seen labels are diagnostic only; historical labels need their exact revision's own maps.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--independent-request", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.model, independent_request=args.independent_request)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8")
    print(args.out, sha256_file(args.out),
          "training-seen", result["training_seen"]["scored"],
          "independent", result["independent_postfit"]["scored"])


if __name__ == "__main__":
    main()
