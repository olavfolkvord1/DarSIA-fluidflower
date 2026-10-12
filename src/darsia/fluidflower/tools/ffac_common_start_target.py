"""Build and validate immutable FF_AC common-start colour targets.

This module consumes only frozen pre-injection baseline observations, the
registered facies grid and the existing ColorChecker gauge. It never reads
phase labels or model outputs and never writes into a model directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


ROOT = Path(os.environ.get("FFAC_COMMON_START_WORKSPACE",
                           r"C:\Users\olav_\Documents\FF Albus"))
FEATURE_ROOT = ROOT / "outputs" / "ffac-common-start-preflight-20260928"
INVENTORY_ROOT = ROOT / "outputs" / "ffac-two-arm-chart-inventory-20260927" / "runs"
CHART_PATH = ROOT / "outputs" / "ffac-global-chart-target-20260928" / "robust16_common_target.json"
PUBLISH_ROOT = ROOT / "outputs" / "ffac-common-start-target-20260928"
KNOWN_CHEMISTRY_EXCLUSIONS = frozenset({"ac20", "ac24", "ac27", "ac29", "ac51"})
KINDS = frozenset({"ac39", "synthetic_median38"})
CELL_RE = re.compile(r"f(?P<facies>\d+)-y(?P<row>\d+)-x(?P<column>\d+)$")
X_EDGES = (0.16, 0.38, 0.60, 0.82)
Y_EDGES = (0.20, 0.42, 0.64, 0.86)
STEP = 8
FACIES = (1, 2, 5, 7, 8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def checked_rgb(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    rgb = np.asarray(value, dtype=np.float64)
    if rgb.shape != shape or not np.isfinite(rgb).all() or not ((rgb >= 0) & (rgb <= 1)).all():
        raise ValueError(f"{label}: expected finite RGB array of shape {shape} in [0,1]")
    return rgb


def cell_rect_rc(cell_id: str, shape_hw: tuple[int, int]) -> list[list[int]]:
    """Return the exact half-open full-resolution rectangle used by scan.py."""
    match = CELL_RE.fullmatch(cell_id)
    if match is None:
        raise ValueError(f"Invalid cell id: {cell_id}")
    facies, yi, xi = (int(match.group(name)) for name in ("facies", "row", "column"))
    if facies not in FACIES or yi not in range(len(Y_EDGES) - 1) or xi not in range(len(X_EDGES) - 1):
        raise ValueError(f"Cell is outside the approved facies/interior grid: {cell_id}")
    h, w = shape_hw
    hs, ws = (h + STEP - 1) // STEP, (w + STEP - 1) // STEP
    r0, r1 = int(hs * Y_EDGES[yi]) * STEP, min(int(hs * Y_EDGES[yi + 1]) * STEP, h)
    c0, c1 = int(ws * X_EDGES[xi]) * STEP, min(int(ws * X_EDGES[xi + 1]) * STEP, w)
    if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
        raise ValueError(f"Empty or out-of-bounds cell: {cell_id}")
    return [[r0, c0], [r1, c1]]


def _expected_rows(feature_root: Path, inventory_root: Path,
                   verify_source_bytes: bool) -> tuple[list[dict], list[dict], dict]:
    paths = sorted((feature_root / "runs").glob("ac*.json"))
    if len(paths) != 43:
        raise ValueError(f"Expected 43 exact run baseline observations, found {len(paths)}")
    rows: list[dict] = []
    receipts: list[dict] = []
    context_path: Path | None = None
    grid_hw: tuple[int, int] | None = None
    keys: list[str] | None = None
    for path in paths:
        feature = read_json(path)
        run = feature.get("run")
        if run != path.stem or feature.get("format") != "ffac-common-start-baseline-features-v1":
            raise ValueError(f"Incorrect baseline feature receipt: {path}")
        inventory_path = inventory_root / f"{run}.json"
        inventory = read_json(inventory_path)
        if inventory.get("run") != run or inventory.get("format") != "ffac-shape-corrected-baseline-chart-row-v1":
            raise ValueError(f"Incorrect frozen inventory receipt: {inventory_path}")
        if feature.get("inventory_receipt_path") != str(inventory_path.resolve()):
            raise ValueError(f"Feature/inventory binding differs for {run}")
        if (feature.get("raw_baseline_path") != inventory.get("raw_baseline_path")
                or feature.get("raw_baseline_sha256") != inventory.get("raw_baseline_sha256")):
            raise ValueError(f"Raw baseline identity differs for {run}")
        source_path = Path(inventory["raw_baseline_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if verify_source_bytes and sha256_file(source_path) != inventory["raw_baseline_sha256"]:
            raise ValueError(f"Raw baseline SHA-256 changed for {run}")
        shape = tuple(inventory["registered_shape_hw"])
        if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError(f"Invalid registered shape for {run}")
        if grid_hw is None:
            grid_hw = shape
        elif shape != grid_hw:
            raise ValueError(f"Registered grid differs for {run}")
        chart = checked_rgb(inventory["swatches_rgb"], (4, 6, 3), f"{run} chart")
        observed = checked_rgb(feature["swatches_rgb"], (4, 6, 3), f"{run} feature chart")
        if not np.array_equal(chart, observed):
            raise ValueError(f"Frozen ColorChecker observation changed for {run}")
        if feature.get("sample_step") != STEP or tuple(feature.get("facies", [])) != FACIES:
            raise ValueError(f"Scan sampling contract changed for {run}")
        this_context = Path(feature["facies_context_path"])
        if context_path is None:
            context_path = this_context
        elif this_context != context_path:
            raise ValueError(f"Facies context differs for {run}")
        cells = feature.get("cells")
        if not isinstance(cells, dict):
            raise ValueError(f"Missing cells for {run}")
        chosen = [key for key in cells if "-y3-" not in key]
        if len(chosen) != 27 or any(CELL_RE.fullmatch(key) is None for key in chosen):
            raise ValueError(f"Expected 27 interior cells for {run}")
        if keys is None:
            keys = chosen
        elif chosen != keys:
            raise ValueError(f"Cell layout differs for {run}")
        for key in chosen:
            checked_rgb(cells[key]["median_rgb"], (3,), f"{run}/{key}")
            if not (isinstance(cells[key].get("n"), int) and cells[key]["n"] >= 100):
                raise ValueError(f"Insufficient sampled pixels for {run}/{key}")
            cell_rect_rc(key, shape)
        if int(feature["chart_clipped_patch_count"]) != int(np.any(chart >= 0.995, axis=2).sum()):
            raise ValueError(f"Chart clipping count differs for {run}")
        receipts.append({
            "run": run,
            "raw_baseline_path": str(source_path),
            "raw_baseline_sha256": inventory["raw_baseline_sha256"],
            "native_manifest_sha256": inventory["native_manifest_sha256"],
            "inventory_path": str(inventory_path.resolve()),
            "inventory_sha256": sha256_file(inventory_path),
            "feature_path": str(path.resolve()),
            "feature_sha256": sha256_file(path),
            "chart_clipped_patch_count": feature["chart_clipped_patch_count"],
        })
        rows.append(feature)
    assert context_path is not None and grid_hw is not None and keys is not None
    if not context_path.is_file():
        raise FileNotFoundError(context_path)
    with np.load(context_path) as context:
        if context["facies"].shape != grid_hw or context["domain"].shape != grid_hw:
            raise ValueError("Registered facies/domain context shape differs")
    context_receipt = {
        "path": str(context_path), "sha256": sha256_file(context_path),
        "registered_shape_hw": list(grid_hw), "cell_ids": keys,
    }
    return rows, receipts, context_receipt


def _target_rgb(rows: list[dict], keys: list[str], kind: str) -> np.ndarray:
    by_run = {row["run"]: row for row in rows}
    if kind == "ac39":
        if "ac39" not in by_run:
            raise ValueError("AC39 baseline is missing")
        source = [by_run["ac39"]]
    elif kind == "synthetic_median38":
        source = [row for row in rows if row["run"] not in KNOWN_CHEMISTRY_EXCLUSIONS]
        if len(source) != 38:
            raise ValueError("Synthetic cohort must contain 38 chemically eligible runs")
    else:
        raise ValueError(f"Unknown target kind: {kind}")
    values = np.stack([[row["cells"][key]["median_rgb"] for key in keys] for row in source])
    return np.median(values, axis=0)


def _candidate_runs(rows: list[dict], keys: list[str]) -> list[str]:
    eligible = []
    for row in rows:
        if row["run"] in KNOWN_CHEMISTRY_EXCLUSIONS or row["chart_clipped_patch_count"]:
            continue
        rgb = np.asarray([row["cells"][key]["median_rgb"] for key in keys])
        if np.all(rgb[:, 2] / np.maximum(rgb[:, 1], 0.001) >= 1.4):
            eligible.append(row["run"])
    return eligible


def build_candidate(kind: str, *, feature_root: Path = FEATURE_ROOT,
                    inventory_root: Path = INVENTORY_ROOT,
                    chart_path: Path = CHART_PATH,
                    verify_source_bytes: bool = True) -> dict:
    if kind not in KINDS:
        raise ValueError(f"Unsupported candidate kind: {kind}")
    feature_root, inventory_root, chart_path = map(Path, (feature_root, inventory_root, chart_path))
    rows, receipts, context = _expected_rows(feature_root, inventory_root, verify_source_bytes)
    keys = context["cell_ids"]
    candidates = _candidate_runs(rows, keys)
    if "ac39" not in candidates:
        raise ValueError("AC39 no longer meets source quality rules")
    chart_document = read_json(chart_path)
    if chart_document.get("format") != "ffac-common-colorchecker-target-v1":
        raise ValueError("Unexpected frozen ColorChecker gauge format")
    chart_rgb = checked_rgb(chart_document.get("swatches_rgb"), (4, 6, 3), "frozen chart")
    cohort = [row["run"] for row in rows if row["run"] not in KNOWN_CHEMISTRY_EXCLUSIONS]
    observed_charts = np.stack([row["swatches_rgb"] for row in rows if row["run"] in cohort])
    common_patches = np.where(np.all(observed_charts < 0.995, axis=(0, 3)).reshape(-1))[0].tolist()
    if len(common_patches) != 17:
        raise ValueError("Common usable ColorChecker patch set changed")
    sand = _target_rgb(rows, keys, kind)
    shape = tuple(context["registered_shape_hw"])
    cells = []
    for key, rgb in zip(keys, sand):
        match = CELL_RE.fullmatch(key)
        assert match is not None
        cells.append({
            "id": key, "facies": int(match.group("facies")),
            "rect_rc": cell_rect_rc(key, shape),
            "rgb_median": rgb.tolist(),
        })
    analysis_path = feature_root / "result.json"
    analysis = read_json(analysis_path)
    if (analysis.get("runs") != 43 or analysis.get("source_cohort") != cohort
            or analysis.get("source_candidate_runs") != candidates
            or analysis.get("matched_sand_cells") != keys):
        raise ValueError("Frozen preflight ranking no longer matches source observations")
    result = {
        "format": "ffac-common-start-target-v1",
        "series": "ffac",
        "kind": kind,
        "reference_run": "ac39" if kind == "ac39" else None,
        "shape_hw": list(shape),
        "facies_context_path": context["path"],
        "facies_context_sha256": context["sha256"],
        "chart_target_rgb": chart_rgb.tolist(),
        "cells": cells,
        "geometry": {
            "coordinate_system": "registered pre-injection baseline pixels",
            "registered_shape_hw": list(shape),
            "interior_fraction_xyxy": [0.16, 0.20, 0.82, 0.86],
            "sampling_step": STEP,
            "facies_erosion_kernel_sampled_px": 3,
            "cell_sampling": "image[::8,::8] intersected with 3x3-eroded facies, domain and run valid mask, then median RGB",
            "facies_context": context,
        },
        "selection": {
            "source_cohort": cohort,
            "chemistry_excluded_as_sources": sorted(KNOWN_CHEMISTRY_EXCLUSIONS),
            "eligible_run_references": candidates,
            "common_usable_chart_patch_indices": common_patches,
            "raw_chart_saturation_cutoff": 0.995,
            "source_blue_ratio_min": 1.4,
            "ac39_medoid_joint_mean_normalized": analysis["ranking"][0]["mean_cohort"],
            "synthetic_frozen_chart_joint_mean_normalized": analysis[
                "synthetic_sand_plus_frozen_robust16_chart_mean_cohort_normalized"
            ],
        },
        "provenance": {
            "raw_baseline_receipts": receipts,
            "chart_gauge_path": str(chart_path.resolve()),
            "chart_gauge_sha256": sha256_file(chart_path),
            "preflight_result_path": str(analysis_path.resolve()),
            "preflight_result_sha256": sha256_file(analysis_path),
            "scan_code_path": str((feature_root / "scan.py").resolve()),
            "scan_code_sha256": sha256_file(feature_root / "scan.py"),
            "analysis_code_path": str((feature_root / "analyze.py").resolve()),
            "analysis_code_sha256": sha256_file(feature_root / "analyze.py"),
            "builder_code_path": str(Path(__file__).resolve()),
            "builder_code_sha256": sha256_file(Path(__file__)),
            "uses_phase_labels_or_model_maps": False,
        },
        "status": "candidate; full-frame and temporal QA required",
    }
    result["target_id"] = "ffac-common-start-v1:" + canonical_sha256(result)[:24]
    return result


def validate_candidate(target: dict, *, feature_root: Path = FEATURE_ROOT,
                       inventory_root: Path = INVENTORY_ROOT,
                       chart_path: Path = CHART_PATH,
                       verify_source_bytes: bool = True) -> dict:
    if target.get("format") != "ffac-common-start-target-v1" or target.get("series") != "ffac":
        raise ValueError("Not an FF_AC common-start target")
    supplied_id = target.get("target_id")
    without_id = {key: value for key, value in target.items() if key != "target_id"}
    if supplied_id != "ffac-common-start-v1:" + canonical_sha256(without_id)[:24]:
        raise ValueError("Target content digest mismatch")
    expected = build_candidate(target["kind"], feature_root=feature_root,
                               inventory_root=inventory_root, chart_path=chart_path,
                               verify_source_bytes=verify_source_bytes)
    if target != expected:
        raise ValueError("Target differs from its frozen sources or sampling contract")
    return {"target_id": supplied_id, "source_runs_verified": 43,
            "source_bytes_verified": bool(verify_source_bytes),
            "chart_gauge_sha256": target["provenance"]["chart_gauge_sha256"]}


def publish_candidate(kind: str, *, out_dir: Path = PUBLISH_ROOT,
                      verify_source_bytes: bool = True) -> Path:
    target = build_candidate(kind, verify_source_bytes=verify_source_bytes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kind}.json"
    serialized = json.dumps(target, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"Immutable target already exists with different bytes: {path}")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    validate_candidate(read_json(path), verify_source_bytes=verify_source_bytes)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--kind", choices=sorted(KINDS), default="ac39")
    parser.add_argument("--path", type=Path, help="Candidate JSON to validate")
    parser.add_argument("--skip-source-byte-check", action="store_true",
                        help="Fast structural check only; production publication verifies source bytes")
    args = parser.parse_args()
    verify = not args.skip_source_byte_check
    if args.command == "build":
        path = publish_candidate(args.kind, verify_source_bytes=verify)
        print(json.dumps({"path": str(path), "sha256": sha256_file(path),
                          "target_id": read_json(path)["target_id"]}, indent=2))
    else:
        path = args.path or PUBLISH_ROOT / f"{args.kind}.json"
        print(json.dumps(validate_candidate(read_json(path), verify_source_bytes=verify), indent=2))


if __name__ == "__main__":
    main()
