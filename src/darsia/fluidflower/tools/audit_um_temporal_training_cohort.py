"""Audit source-aligned dense FF_UM labels against frozen optical probabilities.

This is a read-only preparation step for a possible learned regional CO2 head.
It does not fit a model, infer a phase map, or interpret issue/mixed boxes as
dense truth. Cache keys come from the published model's native run receipts;
every sampled cache payload is SHA-verified before use.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


FORMAT = "ffum-source-aligned-optical-weakness-audit-v1"
CLASSES = {
    "aqueous": "aqueous_positive",
    "gas": "gas_co2_positive",
    "current_experiment_co2_artifact": "carryover_negative",
    "water": "water_negative",
}
WEAK_CUTOFF = 0.5


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def dense_cohort(comparison: dict, evidence: dict) -> dict[tuple[str, int], list[dict]]:
    """Keep only explicitly dense classes in previously source-aligned reviews."""
    labels = comparison["labels"]
    if labels.get("independent_validation") is not False or labels.get("training_seen") is not True:
        raise ValueError("Expected training-seen, source-aligned comparison labels")
    if labels["summary"]["aligned_regions"] != len(labels["observations"]):
        raise ValueError("Aligned region count changed")
    by_id = {item["region_id"]: item for item in evidence["regions"]}
    if len(by_id) != len(evidence["regions"]):
        raise ValueError("Duplicate frozen region IDs")
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    seen = set()
    for row in labels["observations"]:
        region_id = row["region_id"]
        if region_id in seen:
            raise ValueError(f"Duplicate aligned observation: {region_id}")
        seen.add(region_id)
        item = by_id.get(region_id)
        if item is None or item["run"] != row["run"] or item["phase_class"] != row["phase_class"]:
            raise ValueError(f"Frozen evidence disagrees with aligned observation: {region_id}")
        role = CLASSES.get(row["phase_class"])
        if role is None:
            continue
        frame = int(row["studio_frame_1based"]) - 1
        if frame < 0:
            raise ValueError("Invalid Studio frame number")
        grouped[(row["run"], frame)].append(dict(region_id=region_id,
            phase_class=row["phase_class"], role=role, payload=item["payload"]))
    return grouped


def region_crop(payload: dict, canvas_wh: list[int], box: list[int]) -> tuple[tuple[int, ...], np.ndarray]:
    """Rasterize exactly as the native QA comparison does, including crop offset."""
    canvas_w, canvas_h = map(float, canvas_wh)
    bx, by, width, height = map(int, box)
    points = payload.get("points") or []
    if not points:
        x, y = float(payload["x"]), float(payload["y"])
        w, h = float(payload["width"]), float(payload["height"])
        points = ((x, y), (x+w, y), (x+w, y+h), (x, y+h))
    polygon = np.asarray([(float(x)*canvas_w-bx, float(y)*canvas_h-by)
                          for x, y in points], dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width-1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height-1)
    polygon = np.rint(polygon).astype(np.int32)
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0) + 1
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty region polygon")
    mask = np.zeros((int(y1-y0), int(x1-x0)), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon-(x0, y0)], 1)
    if not mask.any():
        raise ValueError("Empty region raster")
    return (int(x0), int(y0), int(x1), int(y1)), mask.astype(bool)


def probability_stats(values: np.ndarray) -> dict:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite optical probabilities")
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Optical probability outside [0,1]")
    return dict(pixels=int(values.size), weak_pixels=int(np.count_nonzero(values < WEAK_CUTOFF)),
                weak_fraction=float(np.mean(values < WEAK_CUTOFF)),
                p10=float(np.quantile(values, .10)), median=float(np.median(values)),
                p90=float(np.quantile(values, .90)))


def _verified_heads(cache_root: Path, key: str, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
        raise ValueError("Invalid optical cache key")
    entry = cache_root / key
    receipt_path, payload = entry / "receipt.json", entry / "heads.npy"
    receipt = _read(receipt_path)
    content = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (receipt["receipt_sha256"] != _json_sha(content)
            or receipt["key"] != key
            or receipt["identity_sha256"] != _json_sha(receipt["identity"])
            or receipt["format"] != "exact-optical-head-cache-v1"
            or receipt["shape"] != [*shape, 2]
            or receipt["dtype"] != "float32"
            or _sha(payload) != receipt["payload_sha256"]):
        raise ValueError(f"Frozen optical cache mismatch: {key}")
    array = np.load(payload, mmap_mode="r", allow_pickle=False)
    if array.shape != (*shape, 2) or array.dtype != np.float32:
        raise ValueError("Frozen optical head shape/dtype mismatch")
    return array, receipt


def _native_manifest(model: Path, run: str, expected_sha: str) -> dict:
    postflight = _read(model / "validation" / "run_packages" / run / "postflight_manifest.json")
    jobs = postflight["jobs"]
    if len(jobs) != 1 or jobs[0]["run"] != run:
        raise ValueError(f"Unexpected native validation job for {run}")
    manifest_path = Path(jobs[0]["review_output_dir"]) / "native_frames" / run / "manifest.json"
    if _sha(manifest_path) != expected_sha:
        raise ValueError(f"Native source manifest changed for {run}")
    return _read(manifest_path)


def audit(model: Path, comparison_path: Path, cache_root: Path) -> dict:
    comparison, evidence = _read(comparison_path), _read(model / "evidence.json")
    receipt = _read(model / "native_source_receipt.json")
    request_path = Path(comparison["labels"]["frozen_request"])
    if (_sha(request_path) != comparison["labels"]["frozen_request_sha256"]
            or receipt["series"] != "ffum" or receipt["runs"] != 13
            or receipt["frames"] != 261):
        raise ValueError("Frozen FF_UM cohort/source request mismatch")
    cohort = dense_cohort(comparison, evidence)
    area_frames = {(run, int(row["candidate_index_0based"])): row
                   for run, info in comparison["area"]["runs"].items()
                   for row in info["frames"]}
    rows, cache_sha = [], {}
    manifests = {}
    backends = {}
    for run, frame in sorted(cohort):
        if run not in manifests:
            manifests[run] = _native_manifest(model, run, receipt["images"][run]["manifest_sha256"])
            backend = _read(model / "validation" / "run_packages" / run / "backend_checks.json")
            if len(backend["runs"]) != 1:
                raise ValueError(f"Unexpected optical backend report for {run}")
            backends[run] = {int(item["frame"]): item["optical_cache_key"]
                             for item in backend["runs"][0]["frames"]}
        manifest = manifests[run]
        source = manifest["frame_sources"][frame]
        paired = area_frames[(run, frame)]
        if (source["source_name"] != paired["source_name"]
                or source["source_sha256"] != paired["source_sha256"]
                or source["native_jpeg_sha256"] != paired["native_jpeg_sha256"]
                or source["native_jpeg_sha256"] != receipt["images"][run]["frame_sha256"][frame]
                or abs(float(source["elapsed_h"])-float(paired["elapsed_h"])) >= 1e-6):
            raise ValueError(f"Native source/QA frame mismatch: {run} {frame}")
        box = list(map(int, manifest["content_box_xywh"]))
        shape = (box[3], box[2])
        key = backends[run][frame]
        heads, cache_receipt = _verified_heads(cache_root, key, shape)
        cache_sha[key] = cache_receipt["payload_sha256"]
        for item in cohort[(run, frame)]:
            if item["payload"].get("coordinate_space") != "normalized_native_review_canvas_16x9":
                raise ValueError(f"Unexpected annotation coordinate space: {item['region_id']}")
            (x0, y0, x1, y1), mask = region_crop(item["payload"], manifest["canvas_wh"], box)
            pixels = heads[y0:y1, x0:x1, 0][mask]
            rows.append(dict(region_id=item["region_id"], run=run,
                studio_frame_1based=frame+1, elapsed_h=float(source["elapsed_h"]),
                phase_class=item["phase_class"], role=item["role"],
                source_sha256=source["source_sha256"],
                native_jpeg_sha256=source["native_jpeg_sha256"],
                optical_cache_key=key, optical_cache_payload_sha256=cache_receipt["payload_sha256"],
                optical_co2_presence=probability_stats(pixels)))
        del heads
    counts = Counter(row["role"] for row in rows)
    if counts != Counter(aqueous_positive=496, gas_co2_positive=32,
                         carryover_negative=45, water_negative=72):
        raise ValueError(f"Aligned dense class inventory differs: {counts}")
    by_run = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_run[row["run"]][row["role"]].append(row)
    summary = {}
    for run, roles in sorted(by_run.items()):
        summary[run] = {}
        for role, items in sorted(roles.items()):
            pixel_count = sum(r["optical_co2_presence"]["pixels"] for r in items)
            weak_count = sum(r["optical_co2_presence"]["weak_pixels"] for r in items)
            summary[run][role] = dict(regions=len(items), pixels=pixel_count,
                                      weak_pixels=weak_count,
                                      weak_fraction=weak_count/pixel_count)
    return dict(format=FORMAT, model=str(model), comparison=str(comparison_path),
        comparison_sha256=_sha(comparison_path),
        frozen_request_sha256=comparison["labels"]["frozen_request_sha256"],
        native_source_manifest_sha256=receipt["source_manifest_sha256"],
        optical_cache_root=str(cache_root), verified_cache_payloads=cache_sha,
        training_seen=True, independent_validation=False, pixel_weak_cutoff=WEAK_CUTOFF,
        interpretation=("Only aqueous CO2 is dense aqueous positive over the whole polygon. "
            "Current-experiment CO2 artifact is CO2 negative over the whole polygon. "
            "Dense gas is CO2-presence positive but remains a separate safety stratum. "
            "Water is CO2 negative. Mixed and model-specific issue boxes are excluded; "
            "a model-specific incorrect-free-gas report is not a whole-box gas-negative label."),
        counts=dict(counts), by_run=summary, regions=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.model, args.comparison, args.cache_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(format=result["format"], counts=result["counts"],
                          by_run=result["by_run"], verified_cache_frames=len(result["verified_cache_payloads"])),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
