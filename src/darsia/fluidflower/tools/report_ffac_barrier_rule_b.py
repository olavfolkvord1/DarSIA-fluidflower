"""Independent, source-bound Rule B QA; never writes to a candidate."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.phase_v5_source_alias import resolve_original_native_source

SPEC_SHA = "41fa22e6bbf91b10086eaf9eb5a198039d4453c1687370338221ca010322a3f8"
CONTRACT = "ffac-gas-barrier-rule-b-independent-report-v1"
FRAME_CONTRACT = "ffac-gas-barrier-rule-b-candidate-v1"
QA_CONTRACT = "ffac-gas-barrier-rule-b-qa-v1"
SNAPSHOTS = (
    ("round1_ac39_ac50_ac31_20261001T1112Z", {"ac39", "ac50", "ac31"}, 871),
    ("round2_test_ac48_ac53_ac35_20261001T1825Z", {"ac48", "ac53", "ac35"}, 176),
    ("round3_q1_ac46_ac59_20261002T2256Z", {"ac46", "ac59"}, 338),
)
GAS_CATEGORIES = {"confirmed_only_free_gas", "confirmed_mostly_free_gas"}
FOUR = ndimage.generate_binary_structure(2, 1)


def _read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _atomic_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Independent report refuses to overwrite an existing result")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _below_normal():
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 5)
    cv2.setNumThreads(1)


def _asset(path, digest=None):
    path = resolve_review_path(path).resolve()
    digest = digest or sha256_file(path)
    return {"path": str(path), "sha256": digest}


def _verify_asset(asset):
    path = resolve_review_path(asset["path"])
    if sha256_file(path) != asset["sha256"]:
        raise ValueError(f"Immutable source/file hash differs: {path}")
    return path


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return ([int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
            if len(xs) else None)


def reconstruct_rule_b(labels, components, sources, barrier):
    """An independent SciPy implementation of the literal connected-path rule.

    Scientific components and CO2 touches are evaluated on the inherited union.
    This neither calls the production operator nor filters gas a second time.
    """
    if (labels.ndim != 2 or labels.dtype != np.uint8 or
            not np.isin(labels, [0, 1, 2, 255]).all() or
            components.shape != labels.shape or not np.issubdtype(components.dtype, np.integer) or
            (components < 0).any() or barrier.dtype != np.bool_ or barrier.shape != labels.shape or
            np.any(barrier & (components > 0))):
        raise ValueError("Invalid categorical, open-component or barrier grid")
    ids = set(int(i) for i in np.unique(components) if i > 0)
    sources = set(int(i) for i in sources)
    if not sources or not sources <= ids:
        raise ValueError("Missing injector component or undeclared source component")
    co2 = (labels == 1) | (labels == 2)
    connected, count = ndimage.label(co2, structure=FOUR)
    sizes = np.bincount(connected.ravel(), minlength=count+1)
    # Only the CO2 cells touching multiple open regions can bridge them.
    touches = {}
    for region in sorted(ids):
        for cc in np.unique(connected[(components == region) & co2]):
            touches.setdefault(int(cc), set()).add(region)
    bridges = {cc: touched for cc, touched in touches.items() if len(touched) > 1}
    reachable, triggered = set(sources), set()
    while True:
        expanded = reachable | set().union(*(touched for touched in bridges.values()
                                             if touched & reachable), set())
        if expanded == reachable:
            break
        new = expanded - reachable
        triggered |= {cc for cc, touched in bridges.items() if touched & reachable and touched & new}
        reachable = expanded
    breached = sorted(reachable - sources)
    logs = []
    slices = ndimage.find_objects(connected)
    for cc, touched in sorted(bridges.items()):
        if cc in triggered:
            sl = slices[cc-1]
            logs.append(dict(co2_component_id=cc, co2_component_pixels=int(sizes[cc]),
                open_component_ids=sorted(touched),
                bbox_xywh=[int(sl[1].start), int(sl[0].start),
                           int(sl[1].stop-sl[1].start), int(sl[0].stop-sl[0].start)]))
    remove = (labels == 2) & (components > 0) & ~np.isin(components, sorted(reachable))
    result = labels.copy()
    result[remove] = 1
    return result, remove, dict(breached_components=breached,
        bridging_co2_components=logs, source_component_ids=sorted(sources),
        reachable_components=sorted(reachable), co2_components=count)


def rasterize_label_roi(label, native_shape, canvas_wh, content_box):
    """Rasterise in the label's own native review canvas, without a warp."""
    if (label["coordinate_space"] != "normalized_native_review_canvas_16x9" or
            label["shape"] != "freehand"):
        raise ValueError("Unsupported gas-label coordinate contract")
    points = np.asarray(json.loads(label["points_json"]), dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or
            not np.isfinite(points).all() or np.any((points < 0) | (points > 1))):
        raise ValueError("Invalid sealed gas polygon")
    x, y, w, h = [int(v) for v in content_box]
    if tuple(native_shape) != (h, w):
        raise ValueError("Gas label canvas and native analysis dimensions differ")
    # Match the existing native annotation contract: compute canvas coordinates
    # then cast to float32, clip each vertex and round, without image warping.
    polygon = (points * np.asarray(canvas_wh) - [x, y]).astype(np.float32)
    if (np.all(polygon[:, 0] < 0) or np.all(polygon[:, 0] > w-1) or
            np.all(polygon[:, 1] < 0) or np.all(polygon[:, 1] > h-1)):
        raise ValueError("Sealed gas label lies outside its source image")
    polygon[:, 0] = np.clip(polygon[:, 0], 0, w-1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, h-1)
    points = np.rint(polygon).astype(np.int32)
    x0, y0 = max(0, int(points[:, 0].min())), max(0, int(points[:, 1].min()))
    x1, y1 = min(w, int(points[:, 0].max())+1), min(h, int(points[:, 1].max())+1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Sealed gas label lies outside its source image")
    mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    cv2.fillPoly(mask, [points - [x0, y0]], 1)
    if not mask.any():
        raise ValueError("Sealed gas label rasterises to no native pixels")
    return (slice(y0, y1), slice(x0, x1)), mask.astype(bool)


def load_sealed_labels(snapshot_root, parent):
    """Bind all 1,385 independent gas polygons to their exact source geometry."""
    snapshots, grouped, seen, native_cache, postflight_cache = [], {}, set(), {}, {}
    parent = resolve_review_path(parent)
    for name, selected_runs, expected_count in SNAPSHOTS:
        folder = resolve_review_path(snapshot_root)/name
        manifest_path = folder/"SNAPSHOT_MANIFEST.json"
        manifest = _read(manifest_path)
        bindings = [_asset(manifest_path)]
        for filename, bound in manifest["files"].items():
            asset = _asset(folder/filename, bound["sha256"])
            _verify_asset(asset)
            if (folder/filename).stat().st_size != bound["bytes"]:
                raise ValueError("Sealed snapshot byte count differs")
            bindings.append(asset)
        session = _read(folder/"review_session.json")
        postpath = resolve_review_path(session["combined_root"])/"postflight_manifest.json"
        postkey = str(postpath.resolve())
        if postkey not in postflight_cache:
            postflight_cache[postkey] = (_read(postpath), _asset(postpath))
        post, post_asset = postflight_cache[postkey]
        jobs = {job["run"]: job for job in post["jobs"]}
        count = 0
        with (folder/"pairwise_review_annotations.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for original_row in rows:
            run = original_row["run"].casefold()
            if run not in selected_runs or original_row["category"] not in GAS_CATEGORIES:
                continue
            row = dict(original_row)
            key = (run, row["annotation_id"])
            if key in seen:
                raise ValueError("Gas snapshot labels overlap between declared rounds")
            seen.add(key)
            job = jobs[run]
            if (row["candidate_id"] != session["selected_candidates"][run] or
                    row["candidate_id"] != job["candidate_id"] or
                    row["params_sha256"] != job["params_sha256"]):
                raise ValueError("Gas label is attributed to another candidate/parameter source")
            if run not in native_cache:
                alias_root = parent/"source_alias"
                _root, oldpath, native = resolve_original_native_source(alias_root, run)
                expected_path = resolve_review_path(job["analysis_source_review_root"])/"native_frames"/run/"manifest.json"
                if oldpath.resolve() != expected_path.resolve():
                    raise ValueError("Parent source alias differs from the sealed label's own native source")
                old_sha = sha256_file(oldpath)
                if job["frame_geometry_manifest_sha256"] != old_sha:
                    raise ValueError("Saved label job changed its native geometry manifest")
                native_cache[run] = (native, _asset(oldpath, old_sha))
            native, native_asset = native_cache[run]
            index = int(row["frame_index"])
            frame = next((f for f in native["frame_sources"] if f["frame_index"] == index), None)
            if frame is None or abs(float(row["elapsed_h"])-frame["elapsed_h"]) > 1e-10:
                raise ValueError("Gas label belongs to a different frame or time")
            for field, target in (("analysis_image_width", native["analysis_image_shape_hw"][1]),
                                  ("analysis_image_height", native["analysis_image_shape_hw"][0]),
                                  ("review_canvas_width", native["canvas_wh"][0]),
                                  ("review_canvas_height", native["canvas_wh"][1])):
                if row.get(field) and int(float(row[field])) != target:
                    raise ValueError("Gas label declares a different source canvas")
            row.update(round=name, frame_index=index, elapsed_h=float(row["elapsed_h"]),
                native_shape_hw=native["analysis_image_shape_hw"], canvas_wh=native["canvas_wh"],
                content_box_xywh=native["content_box_xywh"],
                original_native_manifest=native_asset,
                source_frame_sha256=_canonical_sha(frame))
            rasterize_label_roi(row, row["native_shape_hw"], row["canvas_wh"], row["content_box_xywh"])
            grouped.setdefault((run, index), []).append(row)
            count += 1
        if count != expected_count:
            raise ValueError(f"Sealed {name} gas-label count differs: {count}")
        snapshots.append(dict(round=name, gas_labels=count, files=bindings,
                              original_comparison_manifest=post_asset))
    if sum(len(v) for v in grouped.values()) != 1385:
        raise ValueError("Independent gas QA requires exactly 1,385 sealed labels")
    return grouped, dict(contract="ffac-rule-b-sealed-gas-label-source-binding-v1",
        gas_labels=1385, label_frames=len(grouped), snapshots=snapshots,
        source_native_manifests={run: asset for run, (_native, asset) in native_cache.items()},
        coordinates="Original per-frame native review canvas; no label warp or revision rebase")


def _bridge_signature(logs):
    """Ignore implementation numbering, retain bridge size and touched regions."""
    return sorted((int(log["co2_component_pixels"]), tuple(sorted(log["open_component_ids"])))
                  for log in logs)


def _read_labels(path):
    with Image.open(resolve_review_path(path)) as image:
        if image.mode != "L":
            raise ValueError("Authoritative categorical map must be native uint8 L")
        return np.asarray(image, dtype=np.uint8).copy()


def verify_frame(row, labels=()):
    """Verify actual pixels against the literal rule, source and sealed labels."""
    before, after = _read_labels(row["parent_labels_map_path"]), _read_labels(row["map_path"])
    with np.load(resolve_review_path(row["components_path"]), allow_pickle=False) as bundle:
        components, barrier = bundle["open_components"].copy(), bundle["barrier_mask"].copy()
        reference_domain = bundle["reference_domain"].copy()
    if reference_domain.dtype != np.bool_ or reference_domain.shape != before.shape:
        raise ValueError("Rule B domain grid differs from its categorical source")
    expected, remove, independent = reconstruct_rule_b(before, components, row["source_component_ids"], barrier)
    if after.shape != before.shape or not np.array_equal(after, expected):
        raise ValueError("Actual Rule B map differs from independent connected-path reconstruction")
    co2_before, co2_after = np.isin(before, [1, 2]), np.isin(after, [1, 2])
    if not np.array_equal(co2_before, co2_after) or np.any((before != after) & ~((before == 2) & (after == 1))):
        raise ValueError("Rule B changes a CO2 pixel or anything except gas to aqueous")
    if (row["co2_union_exact"] is not True or row["only_gas_to_aqueous"] is not True or
            independent["breached_components"] != sorted(row["breached_components"]) or
            _bridge_signature(independent["bridging_co2_components"]) != _bridge_signature(row["bridging_co2_components"])):
        raise ValueError("Producer Rule B audit differs from independent literal reconstruction")
    with Image.open(resolve_review_path(row["parent_map_path"])) as image:
        old_rgba = np.asarray(image.convert("RGBA")).copy()
    with Image.open(resolve_review_path(row["studio_map_path"])) as image:
        new_rgba = np.asarray(image.convert("RGBA")).copy()
    x, y, w, h = row["content_box_xywh"]
    if (before.shape != (h, w) or old_rgba.shape != new_rgba.shape or
            old_rgba.shape[:2] != tuple(reversed(row["canvas_wh"]))):
        raise ValueError("Categorical/Studio canvas or content geometry differs")
    old, new = old_rgba[y:y+h, x:x+w], new_rgba[y:y+h, x:x+w]
    if not np.array_equal(old[~remove], new[~remove]):
        raise ValueError("Studio overlay changed outside gas-to-aqueous pixels")
    from darsia.fluidflower.phase_validation import PHASE_FILL_COLORS
    aqueous = np.asarray(PHASE_FILL_COLORS["aqueous"], dtype=np.uint8)
    gas = np.asarray(PHASE_FILL_COLORS["gas"], dtype=np.uint8)
    for native_labels, rgba in ((before, old), (after, new)):
        for phase, colour in ((1, aqueous), (2, gas)):
            if np.any(rgba[native_labels == phase] != colour):
                raise ValueError("Studio overlay does not represent authoritative gas/aqueous labels")
    # Canvas margins must also remain unchanged, including transparent 255.
    changed_canvas = np.any(old_rgba != new_rgba, axis=-1)
    changed_canvas[y:y+h, x:x+w] = False
    if changed_canvas.any():
        raise ValueError("Studio canvas margins changed under Rule B")
    pixel_area = float(row["pixel_area_mm2"])
    if not np.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError("Invalid nominal area scale")
    old_count, new_count, removed = int((before == 2).sum()), int((after == 2).sum()), int(remove.sum())
    for key, value in (("before_gas_pixels", old_count), ("after_gas_pixels", new_count), ("removed_gas_pixels", removed)):
        if row[key] != value:
            raise ValueError("Declared Rule B gas counts differ from actual native maps")
    union_sha = hashlib.sha256(np.packbits(co2_before, bitorder="little").tobytes()).hexdigest()
    if (row["co2_sha256_before"] != union_sha or row["co2_sha256_after"] != union_sha or
            row["before_co2_pixels"] != int(co2_before.sum()) or row["after_co2_pixels"] != int(co2_after.sum())):
        raise ValueError("Declared Rule B CO2 pixel/bit binding differs from actual maps")
    for key, count in (("before_gas_cm2", old_count), ("after_gas_cm2", new_count), ("removed_gas_cm2", removed)):
        if row[key] != count*pixel_area/100:
            raise ValueError("Declared gas area differs from inherited nominal scale")
    result = dict(run=row["run"], frame_index=int(row["frame_index"]), elapsed_h=row["elapsed_h"],
        pixel_area_mm2=pixel_area, before_gas_pixels=old_count, after_gas_pixels=new_count,
        removed_gas_pixels=removed, before_gas_cm2=old_count*pixel_area/100,
        after_gas_cm2=new_count*pixel_area/100, removed_gas_cm2=removed*pixel_area/100,
        removed_gas_percent=100*removed/old_count if old_count else 0.,
        co2_union_exact=True, only_gas_to_aqueous=True,
        co2_pixels=int(co2_before.sum()), co2_cm2=float(co2_before.sum()*pixel_area/100),
        breached_components=independent["breached_components"],
        bridging_co2_components=independent["bridging_co2_components"], label_results=[])
    kept_gas = after == 2
    reachable = independent["reachable_components"]
    breakdown = dict(barrier=barrier, source=np.isin(components, row["source_component_ids"]),
        breached=np.isin(components, independent["breached_components"]),
        unmapped=(components == 0) & ~barrier)
    for name, mask in breakdown.items():
        remaining = kept_gas & mask
        count = int(remaining.sum())
        result[f"remaining_{name}_pixels"] = count
        result[f"remaining_{name}_cm2"] = count*pixel_area/100
        result[f"remaining_{name}_bbox_xywh"] = _bbox(remaining)
    if sum(result[f"remaining_{name}_pixels"] for name in breakdown) != new_count:
        raise ValueError("Remaining gas location categories do not partition all gas")
    unconnected = co2_after & (components > 0) & ~np.isin(components, reachable)
    result["unconnected_co2_pixels"] = int(unconnected.sum())
    result["unconnected_co2_cm2"] = float(unconnected.sum()*pixel_area/100)
    for label in labels:
        if (tuple(label["native_shape_hw"]) != before.shape or label["canvas_wh"] != row["canvas_wh"] or
                label["content_box_xywh"] != row["content_box_xywh"] or
                label["frame_index"] != row["frame_index"] or label["elapsed_h"] != row["elapsed_h"]):
            raise ValueError("Sealed label differs from the actual candidate source canvas/frame")
        sl, mask = rasterize_label_roi(label, before.shape, row["canvas_wh"], row["content_box_xywh"])
        removed_label = int(np.count_nonzero(remove[sl] & mask))
        result["label_results"].append(dict(round=label["round"], run=label["run"],
            frame_index=label["frame_index"], elapsed_h=label["elapsed_h"],
            annotation_id=label["annotation_id"], category=label["category"],
            source_native_manifest_sha256=label["original_native_manifest"]["sha256"],
            source_frame_sha256=label["source_frame_sha256"], label_native_pixels=int(mask.sum()),
            before_gas_pixels=int(np.count_nonzero((before[sl] == 2) & mask)),
            after_gas_pixels=int(np.count_nonzero((after[sl] == 2) & mask)),
            removed_gas_pixels=removed_label))
    return result


def _csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def _complete_inventory(rows):
    keys = {(r["run"], int(r["frame_index"])) for r in rows}
    if len(keys) != len(rows) or len(keys) != 864 or len({run for run, _ in keys}) != 43:
        raise ValueError("Independent report requires exactly 43 runs / 864 distinct frames")
    return keys


def pixel_verifier_identity(path):
    """Allow a resource-only resume only when scientific verification is identical."""
    document = ast.parse(resolve_review_path(path).read_text(encoding="utf-8-sig"))
    functions = {"_canonical_sha", "_verify_asset", "_bbox", "reconstruct_rule_b",
        "rasterize_label_roi", "load_sealed_labels", "_bridge_signature", "_read_labels",
        "verify_frame", "_frame_binding", "_receipt_files", "_file_states",
        "verify_completed_receipt", "validate_cached_frame", "validate_sealed_coverage"}
    constants = {"SPEC_SHA", "CONTRACT", "FRAME_CONTRACT", "QA_CONTRACT", "SNAPSHOTS", "GAS_CATEGORIES", "FOUR"}
    selected = {}
    for node in document.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            selected[node.name] = ast.dump(node, include_attributes=False)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    selected[target.id] = ast.dump(node, include_attributes=False)
    if set(selected) != functions | constants:
        raise ValueError("Report entrypoint lacks the complete scientific verifier contract")
    return _canonical_sha(selected)


def load_resume_cache(output, checkpoint_path, plan, plan_sha, label_binding, verifier_path):
    """Reuse sealed completed checks, with original-code and byte-level lineage."""
    output = resolve_review_path(output).resolve()
    checkpoint = _read(checkpoint_path)
    prior = checkpoint["previous_entrypoint"]
    _verify_asset(prior)
    old_stream = _read(output/"stream_binding.json")
    if (checkpoint.get("contract") != "ffac-rule-b-verifier-resource-resume-v1" or
            resolve_review_path(checkpoint["output"]).resolve() != output or
            checkpoint["plan_sha256"] != plan_sha or
            checkpoint["pipeline_sha256"] != plan["pipeline_sha256"] or
            old_stream["plan_sha256"] != plan_sha or
            old_stream["pipeline_sha256"] != plan["pipeline_sha256"] or
            old_stream["verifier_code_sha256"] != prior["sha256"] or
            sha256_file(output/"stream_binding.json") != checkpoint["original_stream_binding_sha256"] or
            sha256_file(output/"sealed_label_source_binding.json") != checkpoint["label_source_binding_sha256"] or
            _read(output/"sealed_label_source_binding.json") != label_binding or
            pixel_verifier_identity(prior["path"]) != pixel_verifier_identity(verifier_path)):
        raise ValueError("Worker-scaling resume changed original plan, labels or scientific verification")
    inventory = {(r["run"], r["frame_index"]): r for r in plan["source_inventory"]}
    declared = checkpoint["cache_files"]
    existing = {str(p.resolve()) for p in (output/"frame_verification_cache").glob("*/frame_*.json")}
    if existing != {str(resolve_review_path(r["path"]).resolve()) for r in declared}:
        raise ValueError("Completed cache differs from the stopped verifier's exact checkpoint")
    cached = {}
    for asset in declared:
        path = _verify_asset(asset)
        entry = _read(path); key = entry["run"], entry["frame_index"]
        if (key not in inventory or key in cached or key != (asset["run"], asset["frame_index"]) or
                entry.get("contract") != CONTRACT or entry["pipeline_sha256"] != plan["pipeline_sha256"] or
                (entry["qa"]["run"], entry["qa"]["frame_index"]) != key or
                _canonical_sha(entry["qa"]) != entry["qa_canonical_sha256"] or
                (entry["result"]["run"], entry["result"]["frame_index"]) != key):
            raise ValueError("Resume cache has a foreign, duplicated or altered completed frame")
        cached[key] = entry
    return cached, dict(checkpoint=_asset(checkpoint_path), prior_entrypoint=prior,
        pixel_verifier_identity=pixel_verifier_identity(verifier_path), reused_completed_frames=len(cached))


def nearest_cohort(rows, cohort):
    if len(cohort) != len(set(cohort)) or len(cohort) != 37 or not set(cohort) <= {r["run"] for r in rows}:
        raise ValueError("Cohort must be 37 distinct source-configured candidate runs")
    snapshots, sums = [], []
    for target in (2.25, 5., 12., 24.):
        group = []
        for run in sorted(cohort):
            nearest = min((r for r in rows if r["run"] == run),
                          key=lambda r: (abs(r["elapsed_h"]-target), r["frame_index"]))
            row = {k: v for k, v in nearest.items() if k != "label_results"}
            group.append(dict(target_h=target, actual_h=nearest["elapsed_h"],
                time_offset_h=nearest["elapsed_h"]-target, **row))
        snapshots.extend(group)
        before, after = (sum(r[f"{kind}_gas_cm2"] for r in group) for kind in ("before", "after"))
        sums.append(dict(target_h=target, runs=37, min_actual_h=min(r["actual_h"] for r in group),
            max_actual_h=max(r["actual_h"] for r in group),
            max_abs_time_offset_h=max(abs(r["time_offset_h"]) for r in group),
            sum_before_gas_cm2=before, sum_after_gas_cm2=after,
            sum_removed_gas_cm2=before-after, removed_gas_percent=100*(before-after)/before if before else 0.,
            mean_before_gas_cm2=before/37, mean_after_gas_cm2=after/37,
            mean_unconnected_co2_cm2=sum(r["unconnected_co2_cm2"] for r in group)/37))
    return snapshots, sums


def _frame_binding(source, pipeline):
    return dict(run=source["run"], frame_index=int(source["frame_index"]),
        pipeline_sha256=pipeline, components=source["components"],
        parent_labels_map=source["parent_labels_map"], parent_phase_map=source["parent_phase_map"],
        native_manifest_sha256=source["native_manifest_sha256"],
        barrier_derivation_sha256=source["barrier_derivation_sha256"])


def _receipt_files(receipt):
    qa, binding = receipt["qa"], receipt["binding"]
    fields = dict(map=receipt["labels_map"], studio_map=receipt["maps"]["combined"],
        parent_map=binding["parent_phase_map"], parent_labels_map=binding["parent_labels_map"],
        components=binding["components"])
    for prefix, asset in fields.items():
        if (qa[prefix+"_sha256"] != asset["sha256"] or
                resolve_review_path(qa[prefix+"_path"]).resolve() != resolve_review_path(asset["path"]).resolve()):
            raise ValueError("Rule B frame QA differs from its sealed asset declarations")
    assets = [*fields.values(), *receipt["maps"].values(), receipt["labels_archive"],
              dict(path=qa["native_manifest_path"], sha256=qa["native_manifest_sha256"])]
    files = {}
    for asset in assets:
        path, digest = str(resolve_review_path(asset["path"]).resolve()), asset["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or (path in files and files[path] != digest):
            raise ValueError("Rule B receipt has ambiguous asset identities")
        files[path] = digest
    return [dict(path=path, sha256=digest) for path, digest in sorted(files.items())]


def _file_states(files):
    states = []
    for asset in files:
        path = _verify_asset(asset)
        stat = path.stat()
        states.append(dict(asset, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return states


def verify_completed_receipt(path, source, pipeline, labels):
    receipt_path = resolve_review_path(path)
    receipt_sha = sha256_file(receipt_path)
    receipt = _read(receipt_path)
    if (receipt.get("contract") != FRAME_CONTRACT or receipt.get("complete") is not True or
            receipt.get("binding") != _frame_binding(source, pipeline) or
            any(receipt.get(key) is not False for key in
                ("training", "model_inference", "component_filtering", "photo_pixels_changed"))):
        raise ValueError("Only exact-plan, completed, map-only Rule B receipts can be verified")
    qa = receipt["qa"]
    if any(qa[key] != source[key] for key in ("run", "frame_index", "elapsed_h", "pixel_area_mm2",
                "content_box_xywh", "canvas_wh", "native_manifest_sha256", "source_component_ids",
                "barrier_derivation_sha256")):
        raise ValueError("Rule B frame QA changed its inherited frame/source/scale geometry")
    files = _receipt_files(receipt)
    states = _file_states(files)
    for label in labels:
        _verify_asset(label["original_native_manifest"])
    result = verify_frame(qa, labels)
    # Bytes were checked before decoding. A concurrent change in size/mtime is
    # fatal; sealing rechecks all SHA values without another pixel decode.
    for asset in states:
        stat = resolve_review_path(asset["path"]).stat()
        if stat.st_size != asset["size"] or stat.st_mtime_ns != asset["mtime_ns"]:
            raise ValueError("Rule B immutable pixel/source asset changed during verification")
    if sha256_file(receipt_path) != receipt_sha:
        raise ValueError("Atomic Rule B receipt changed during independent verification")
    return dict(contract=CONTRACT, run=source["run"], frame_index=source["frame_index"],
        receipt_path=str(receipt_path), receipt_sha256=receipt_sha,
        pipeline_sha256=pipeline, qa=qa, qa_canonical_sha256=_canonical_sha(qa),
        files=files, verified_file_states=states, result=result)


def validate_cached_frame(entry):
    receipt_path = resolve_review_path(entry["receipt_path"])
    receipt = _read(receipt_path)
    if (entry.get("contract") != CONTRACT or sha256_file(receipt_path) != entry["receipt_sha256"] or
            receipt.get("complete") is not True or receipt["qa"] != entry["qa"] or
            _canonical_sha(receipt["qa"]) != entry["qa_canonical_sha256"] or
            _receipt_files(receipt) != entry["files"]):
        raise ValueError("Independent cache differs from its final Rule B frame receipt")
    for asset in entry["files"]:
        _verify_asset(asset)
    return entry["result"]


def validate_sealed_coverage(qa, sealed, cached, expected):
    if (qa.get("contract") != QA_CONTRACT or qa.get("all_pass") is not True or
            sealed.get("contract") != FRAME_CONTRACT or sealed.get("complete") is not True or
            qa.get("spec_sha256") != SPEC_SHA or
            qa.get("model_identity") != sealed.get("model_identity") or
            qa.get("run_count") != 43 or qa.get("frame_count") != 864):
        raise ValueError("Independent report requires the successful exact-spec 43/864 candidate seal")
    rows, frames = qa.get("rows") or [], sealed.get("frames") or []
    if _complete_inventory(rows) != expected or set(cached) != expected or len(frames) != 864:
        raise ValueError("Independent report cannot publish missing or foreign frames")
    proofs = {(r["run"], int(r["frame_index"])): r for r in frames}
    if set(proofs) != expected:
        raise ValueError("Rule B production seal has duplicated or missing receipt bindings")
    for row in rows:
        key = row["run"], int(row["frame_index"])
        entry, proof = cached[key], proofs[key]
        if (row != entry["qa"] or _canonical_sha(row) != entry["qa_canonical_sha256"] or
                proof["receipt_sha256"] != entry["receipt_sha256"] or
                resolve_review_path(proof["receipt_path"]).resolve() != resolve_review_path(entry["receipt_path"]).resolve()):
            raise ValueError("Sealed QA differs from the independently verified atomic frame")


def _producer_state(exit_path, pid, sealed):
    if exit_path is None:
        return sealed
    path = resolve_review_path(exit_path)
    if not path.is_file():
        if pid is not None:
            import psutil
            if not psutil.pid_exists(pid):
                raise RuntimeError("Producer exited without success; report remains unpublished")
        return False
    receipt = _read(path)
    if receipt.get("succeeded") is not True or receipt.get("returncode") != 0:
        raise RuntimeError("Rule B producer failed; independent report remains unpublished")
    if not sealed:
        raise RuntimeError("Producer succeeded without the complete Rule B seal")
    return True


def _load_cohort(model):
    path = resolve_review_path(model)/"gas_barrier_rule_b/barrier_derivation.json"
    derivation = _read(path)
    # Derivation is source-derived by the independent geometry worker. Require
    # its exact configured cohort, never infer membership from available maps.
    cohort = derivation.get("cap_cohort_runs") or derivation.get("cohort_runs")
    if cohort is None:
        cohort = (derivation.get("cohort") or {}).get("included_runs")
    if cohort is None:
        raise ValueError("Barrier derivation has not bound its exact source-configured cohort")
    cohort = [str(run).casefold() for run in cohort]
    if len(cohort) != 37 or len(set(cohort)) != 37:
        raise ValueError("Barrier derivation does not contain the required distinct 37-run cohort")
    sources = derivation.get("cap_sources") or []
    if (len(sources) != 37 or {row["run"] for row in sources} != set(cohort) or
            len({row["run"] for row in sources}) != 37):
        raise ValueError("Rule B cohort differs from its exact 37 source cap artifacts")
    for source in sources:
        _verify_asset(source["cap_artifact"])
    for key in ("spec", "physics_overrides", "protocol_profile", "series_common", "series_facies_table",
                "reference_facies", "source_derivation_script", "source_physics_reader"):
        _verify_asset(derivation[key])
    if derivation["spec"]["sha256"] != SPEC_SHA:
        raise ValueError("Barrier cohort derivation binds another specification")
    return cohort, _asset(path), derivation


def make_ac28_figures(model, output, cached):
    """Read-only photographic figures, with native barrier boundaries."""
    files = {}
    for index, target in ((8, 2.25), (16, 12.)):
        entry = cached[("ac28", index)]
        row = entry["qa"]
        native_path = resolve_review_path(row["native_manifest_path"])
        native = _read(native_path)
        frame = next(f for f in native["frame_sources"] if f["frame_index"] == index)
        photo_path = resolve_review_path(frame["frame_path"])
        if sha256_file(photo_path) != frame["frame_sha256"]:
            raise ValueError("AC28 figure photograph differs from its inherited scientific display source")
        with Image.open(photo_path) as image:
            photo = image.convert("RGB")
        x, y, w, h = row["content_box_xywh"]
        if photo.size == (w, h):
            base = photo
        elif photo.size == tuple(row["canvas_wh"]):
            base = photo.crop((x, y, x+w, y+h))
        else:
            raise ValueError("AC28 photographic figure geometry differs")
        with np.load(resolve_review_path(row["components_path"]), allow_pickle=False) as bundle:
            barrier = bundle["barrier_mask"].copy()
        outline = barrier ^ ndimage.binary_erosion(barrier, structure=FOUR)
        outline = ndimage.binary_dilation(outline, iterations=2)
        panels = []
        for name, key in (("Before Rule B", "parent_map_path"), ("After Rule B", "studio_map_path")):
            with Image.open(resolve_review_path(row[key])) as image:
                overlay = image.convert("RGBA").crop((x, y, x+w, y+h))
            panel = np.asarray(Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")).copy()
            panel[outline] = [255, 255, 255]
            panel = Image.fromarray(panel)
            # Display figure only; original inference/source images are untouched.
            panel.thumbnail((1800, 1000), Image.Resampling.LANCZOS)
            labelled = Image.new("RGB", (panel.width, panel.height+104), (15, 19, 23))
            labelled.paste(panel, (0, 104))
            try:
                font = ImageFont.truetype("arial.ttf", size=30)
            except OSError:
                font = ImageFont.load_default(size=26)
            ImageDraw.Draw(labelled).text((15, 12), f"AC28 frame {index} (Studio {index+1}), {row['elapsed_h']:.6f} h — {name}\nWhite: barrier boundary; gas pink, aqueous green", fill="white", font=font)
            panels.append(labelled)
        sheet = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), (15, 19, 23))
        offset = 0
        for panel in panels:
            sheet.paste(panel, (offset, 0)); offset += panel.width
        path = Path(output)/f"ac28_frame_{index:03d}_before_after_barrier.png"
        if path.exists():
            raise ValueError("Independent photographic report refuses to overwrite a figure")
        sheet.save(path)
        files[path.name] = dict(path=str(path), sha256=sha256_file(path),
            target_h=target, actual_h=row["elapsed_h"], time_offset_h=row["elapsed_h"]-target,
            photograph=_asset(photo_path), components=_asset(row["components_path"], row["components_sha256"]))
    return files


def _write_report(model, output, qa, cached, label_binding, streaming=None):
    rows = sorted((item["result"] for item in cached.values()), key=lambda r: (r["run"], r["frame_index"]))
    _complete_inventory(rows)
    cohort, cohort_asset, derivation = _load_cohort(model)
    snapshots, aggregates = nearest_cohort(rows, cohort)
    label_rows = [label for row in rows for label in row["label_results"]]
    if len(label_rows) != 1385 or len({(r["run"], r["annotation_id"]) for r in label_rows}) != 1385:
        raise ValueError("Completed report does not cover all sealed gas labels exactly once")
    removed_inside = sum(r["removed_gas_pixels"] for r in label_rows)
    if removed_inside:
        # This reports a concrete scientific failure without silently hiding it.
        # Registration can make its review policy decision from this sealed QA.
        all_pass = False
    else:
        all_pass = True
    flat_rows = [{k: v for k, v in row.items() if k != "label_results"} for row in rows]
    run_summaries = []
    for run in sorted({row["run"] for row in rows}):
        group = [row for row in rows if row["run"] == run]
        run_summaries.append(dict(run=run, frame_count=len(group),
            first_h=min(r["elapsed_h"] for r in group), last_h=max(r["elapsed_h"] for r in group),
            mean_before_gas_cm2=sum(r["before_gas_cm2"] for r in group)/len(group),
            mean_after_gas_cm2=sum(r["after_gas_cm2"] for r in group)/len(group),
            max_before_gas_cm2=max(r["before_gas_cm2"] for r in group),
            max_after_gas_cm2=max(r["after_gas_cm2"] for r in group),
            removed_gas_pixels_sum=sum(r["removed_gas_pixels"] for r in group),
            breached_frames=sum(bool(r["breached_components"]) for r in group)))
    late = [row for row in flat_rows if row["run"] in {"ac46", "ac59"} and row["frame_index"] >= 16]
    bridges = []
    for row in rows:
        if row["breached_components"]:
            for bridge in row["bridging_co2_components"]:
                bridges.append(dict(run=row["run"], frame_index=row["frame_index"], elapsed_h=row["elapsed_h"],
                    breached_components=row["breached_components"], **bridge))
    _csv(Path(output)/"all_864_frames_before_after.csv", flat_rows)
    _csv(Path(output)/"all_43_runs_before_after.csv", run_summaries)
    _csv(Path(output)/"sealed_1385_gas_labels.csv", label_rows)
    _csv(Path(output)/"ac46_ac59_frames_16_onward.csv", late)
    _csv(Path(output)/"cohort_37_nearest_2p25_5_12_24h.csv", snapshots)
    _csv(Path(output)/"cohort_37_aggregate_2p25_5_12_24h.csv", aggregates)
    _atomic_new_json(Path(output)/"breakthrough_log.json", dict(contract=CONTRACT,
        breached_frames=sum(bool(row["breached_components"]) for row in rows), bridges=bridges))
    figures = make_ac28_figures(model, output, cached)
    summary = dict(contract=CONTRACT, created_utc=utc_now(), candidate=str(model),
        model_identity=qa["model_identity"], spec_sha256=SPEC_SHA, runs=43, frames=864,
        co2_union_exact_frames=864, only_gas_to_aqueous_frames=864, all_pass=all_pass,
        gas_label_count=1385, changed_gas_labels=sum(r["removed_gas_pixels"] > 0 for r in label_rows),
        removed_pixels_inside_gas_labels_sum=removed_inside,
        label_overlap_note="Per-label removed counts may overlap; the expected value is zero in every polygon.",
        breached_frames=sum(bool(row["breached_components"]) for row in rows),
        bridge_component_count=len(bridges), cohort_runs=cohort, cohort_source=cohort_asset,
        cohort_provenance_note="Membership is frozen in the Rule B barrier derivation receipt and its exact 37 cap-source artifacts; exclusions are explicit in that receipt.",
        cohort_aggregates=aggregates, ac46_ac59_late=late,
        ac28_frames=[cached[("ac28", i)]["result"] for i in (8, 16)],
        label_source_binding=label_binding, figures=figures,
        nominal_scale_only=True, mass_analysis=False, model_training=False,
        statistics_note="Areas use inherited nominal pixel scale. Cohort snapshots are nearest actual saved times with offsets; no interpolation.",
        known_limits=["FF_UM is not processed; its out-of-domain seal requires a separate capillary breakthrough test.",
            "Unconnected CO2 above the upper barrier remains aqueous; this rule does not establish that it is real CO2.",
            "Remaining gas in the source/breached regions or barriers is untouched, including possible late false positives below the seal."],
        qa_source=_asset(Path(model)/"production_qa/barrier_rule_b_qa.json"),
        files={p.name: sha256_file(p) for p in Path(output).glob("*.csv")})
    if streaming:
        summary["streaming_verification"] = streaming
    _atomic_new_json(Path(output)/"summary.json", summary)
    return summary


def follow_candidate(model, output, snapshot_root, *, workers=4, poll_seconds=10., max_wait_seconds=21600.,
                     producer_exit_receipt=None, producer_pid=None, resume_checkpoint=None):
    if not 1 <= workers <= 24 or not 0 < poll_seconds <= 60 or max_wait_seconds <= 0:
        raise ValueError("Invalid bounded stream verification resource settings")
    _below_normal()
    model, output = resolve_review_path(model).resolve(), resolve_review_path(output).resolve()
    plan_path = model/"barrier_rule_b_candidate_plan.json"
    plan, plan_sha = _read(plan_path), sha256_file(plan_path)
    expected = _complete_inventory(plan["source_inventory"])
    if plan["binding"].get("spec_sha256") != SPEC_SHA:
        raise ValueError("Rule B candidate plan binds another specification")
    labels, label_binding = load_sealed_labels(snapshot_root, plan["binding"]["parent"])
    stream_binding = dict(contract=CONTRACT, candidate=str(model),
        plan_sha256=plan_sha, pipeline_sha256=plan["pipeline_sha256"],
        verifier_code_sha256=sha256_file(__file__), workers=workers, started_utc=utc_now(),
        producer_exit_receipt=str(producer_exit_receipt) if producer_exit_receipt else None,
        producer_pid=producer_pid)
    if resume_checkpoint is None:
        output.mkdir(parents=True, exist_ok=False)
        _atomic_new_json(output/"sealed_label_source_binding.json", label_binding)
        _atomic_new_json(output/"stream_binding.json", stream_binding)
        cached, resume_lineage = {}, None
    else:
        if (output/"summary.json").exists():
            raise ValueError("Cannot resume an already completed independent report")
        cached, resume_lineage = load_resume_cache(output, resume_checkpoint, plan, plan_sha, label_binding, __file__)
        _atomic_new_json(output/f"stream_resume_{os.getpid()}.json", dict(stream_binding, resource_resume=resume_lineage))
    inventory = {(r["run"], r["frame_index"]): r for r in plan["source_inventory"]}
    pending, available = {}, {}
    start, last_log, next_scan = time.monotonic(), 0., 0.
    seal_path, qa_path = model/"barrier_rule_b_candidate_receipt.json", model/"production_qa/barrier_rule_b_qa.json"
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_below_normal) as pool:
        while True:
            now = time.monotonic()
            if now-start > max_wait_seconds:
                raise TimeoutError("Rule B stream expired without complete successful coverage")
            ready = _producer_state(producer_exit_receipt, producer_pid, seal_path.is_file())
            if now >= next_scan:
                for path in sorted((model/"barrier_rule_b_frames").glob("*/frame_*.json")):
                    key = path.parent.name, int(path.stem.removeprefix("frame_"))
                    if key not in expected:
                        raise ValueError("Rule B receipt lies outside the frozen frame inventory")
                    if key not in cached and key not in pending.values():
                        available[key] = path
                next_scan = now+poll_seconds
            while available and len(pending) < workers*2:
                key = min(available); path = available.pop(key)
                future = pool.submit(verify_completed_receipt, str(path), inventory[key],
                    plan["pipeline_sha256"], labels.get(key, []))
                pending[future] = key
            if pending:
                done, _ = wait(pending, timeout=min(1., poll_seconds), return_when=FIRST_COMPLETED)
                for future in done:
                    key = pending.pop(future); item = future.result()
                    if (item["run"], item["frame_index"]) != key:
                        raise ValueError("Independent Rule B worker returned another frame")
                    _atomic_new_json(output/"frame_verification_cache"/key[0]/f"frame_{key[1]:03d}.json", item)
                    cached[key] = item
            elif not ready:
                time.sleep(min(poll_seconds, max(.1, next_scan-time.monotonic())))
            if time.monotonic()-last_log >= 30.:
                print(json.dumps(dict(stage="rule_b_independent_pixels", verified_frames=len(cached),
                    pending_frames=len(pending), expected_frames=864, producer_succeeded=ready,
                    elapsed_seconds=round(time.monotonic()-start, 1))), flush=True)
                last_log = time.monotonic()
            if len(cached) == 864 and not pending and ready:
                if sha256_file(plan_path) != plan_sha:
                    raise ValueError("Immutable Rule B source plan changed during verification")
                qa, seal = _read(qa_path), _read(seal_path)
                if seal.get("source_plan_sha256") != plan_sha or seal.get("qa") != _asset(qa_path):
                    raise ValueError("Rule B production seal differs from its exact plan/QA bytes")
                validate_sealed_coverage(qa, seal, cached, expected)
                print(json.dumps(dict(stage="rule_b_final_sha_only", verified_frames=864,
                    pixel_frames_decoded_again=0)), flush=True)
                results = list(pool.map(validate_cached_frame, cached.values(), chunksize=1))
                if any(r != item["result"] for r, item in zip(results, cached.values())):
                    raise ValueError("Final Rule B cache integrity differs")
                # Recheck the tiny sealed label/source inventory after the fit.
                final_labels, final_binding = load_sealed_labels(snapshot_root, plan["binding"]["parent"])
                if final_binding != label_binding or sum(map(len, final_labels.values())) != 1385:
                    raise ValueError("Sealed gas evidence changed while maps were verified")
                result = _write_report(model, output, qa, cached, label_binding, streaming=dict(
                    pixel_frames_verified_once=864, cached_rows_matched_exact_sealed_qa=864,
                    receipt_and_file_sha_revalidated=864, workers=workers,
                    plan_sha256=plan_sha, candidate_seal=_asset(seal_path),
                    producer_success_required=producer_exit_receipt is not None,
                    producer_exit_receipt=_asset(producer_exit_receipt) if producer_exit_receipt else None,
                    verifier_code_sha256=sha256_file(__file__), resource_resume=resume_lineage))
                print(json.dumps(dict(stage="rule_b_independent_report_complete", all_pass=result["all_pass"],
                    summary=str(output/"summary.json"), summary_sha256=sha256_file(output/"summary.json"))), flush=True)
                return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--follow-candidate", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.)
    parser.add_argument("--max-wait-seconds", type=float, default=21600.)
    parser.add_argument("--producer-exit-receipt", type=Path)
    parser.add_argument("--producer-pid", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    # The same stream works for an already sealed candidate and never reruns
    # inference or writes evidence into a production candidate.
    result = follow_candidate(args.model, args.output, args.snapshots, workers=args.workers,
        poll_seconds=args.poll_seconds, max_wait_seconds=args.max_wait_seconds,
        producer_exit_receipt=args.producer_exit_receipt, producer_pid=args.producer_pid,
        resume_checkpoint=args.resume_checkpoint)
    print(json.dumps(dict(all_pass=result["all_pass"], frames=result["frames"],
        gas_label_count=result["gas_label_count"], breached_frames=result["breached_frames"])), flush=True)
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

