"""Read-only source-bound QA of the G3-T2 / flagged floor-contact candidate.

Round 2--4 annotations are never read. Export frame keys come only from sealed
count inventories; round 2 is an explicit all-frames superset for its three runs.
The sealed scorer, operated separately by Claude, remains the acceptance test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from darsia.fluidflower.phase_ffac_floor_contact import CONTRACT as FLOOR_CONTRACT

CONTRACT = "ffac-g3-t2-floor-contact-qa-v1"
BLIND = frozenset(("ac30", "ac49", "ac54"))
ZONE_ARRAY_SHA = "76db153124af785ff826ea67e8da9c4e42bf3e2d3d16f4ad4958a01a21a68b8d"
ZONE_POLICY = {
    "l1": "Claude's frozen AC28 zones==4; last facies 1/2 barrier above in the same column",
    "cap_band": "0--40 mm below that barrier using Claude's fixed reference PX_MM; transported nearest to each frame",
    "plume": "deeper than 40 mm, through reference row 2897+15",
    "below_injector": "L1 zone at reference row greater than 2897+15",
    "caps_away_L1": "open reference zones 2,3,5, 0--40 mm below the preceding facies 1/2 barrier; explicitly a fixed zone proxy",
    "reference_pixel_area_mm2": 0.027053408950374867,
    "area_measurement": "approved bound variable per-pixel area; the old constant is used only to reconstruct Claude's frozen zone definition",
    "injector_verified_from_photograph": False,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def asset(path):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verified(descriptor):
    path = Path(descriptor["path"])
    if digest(path) != descriptor["sha256"]:
        raise ValueError(f"QA source SHA mismatch: {path}")
    return path


def labels(descriptor):
    path = verified(descriptor)
    if path.suffix.casefold() == ".npy":
        result = np.load(path, allow_pickle=False)
    else:
        result = np.asarray(Image.open(path))
    if result.ndim != 2 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("QA categorical phase array has an invalid representation")
    return result


def area_map(descriptor):
    path = verified(descriptor)
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        with value:
            result = value[descriptor.get("array_key", "pixel_area_mm2")].copy()
    else:
        result = value
    if result.ndim != 2 or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("QA requires the approved positive measured area grid")
    return result


def fill_mask(descriptor, shape, source_sha, frame_index):
    if (descriptor.get("contract") != FLOOR_CONTRACT or descriptor.get("source_sha256") != source_sha
            or descriptor.get("frame_index") != frame_index or tuple(descriptor.get("shape_hw", ())) != shape):
        raise ValueError("Floor flag does not bind the QA frame source and canvas")
    x = np.load(verified(descriptor), allow_pickle=False)
    if descriptor.get("storage") == "packbits-npy":
        if x.dtype != np.uint8 or x.ndim != 1 or x.size != (shape[0]*shape[1]+7)//8:
            raise ValueError("Floor flag packbits storage differs")
        x = np.unpackbits(x, count=shape[0]*shape[1]).reshape(shape).astype(bool)
    if x.dtype != bool or x.shape != shape:
        raise ValueError("Floor flag differs from the categorical label canvas")
    return x


def public_rows(rows):
    return [row for row in rows if str(row["run"]).casefold() not in BLIND]


def floor_qa(old, new, flag, area, elapsed_h):
    if old.shape != new.shape or flag.shape != old.shape or area.shape != old.shape:
        raise ValueError("Floor QA arrays differ in grid")
    before = (old == 1) | (old == 2)
    after = (new == 1) | (new == 2)
    under_gas = 0
    invalid_trigger = 0
    discontinuous = 0
    for col in np.flatnonzero(flag.any(0)):
        ys = np.flatnonzero(flag[:, col])
        if len(ys) != ys[-1]-ys[0]+1:
            discontinuous += 1
        trigger = ys[0]-1
        if trigger < 0 or new[trigger, col] != 1:
            invalid_trigger += 1
            if trigger >= 0 and new[trigger, col] == 2:
                under_gas += len(ys)
    return {
        "filled_pixels": int(flag.sum()), "filled_area_cm2": float(area[flag].sum(dtype=np.float64)) / 100,
        "filled_pixels_not_aqueous": int((flag & (new != 1)).sum()),
        "filled_pixels_already_co2_in_base": int((flag & before).sum()),
        "preinjection_fill_pixels": int(flag.sum()) if elapsed_h <= 0 else 0,
        "filled_under_gas_pixels": under_gas,
        "invalid_aqueous_trigger_columns": invalid_trigger,
        "discontinuous_fill_columns": discontinuous,
        "co2_union_changes_outside_floor_contact_fill": int(((before != after) & ~flag).sum()),
        "co2_union_removed_pixels": int((before & ~after).sum()),
        "co2_union_added_without_floor_flag_pixels": int((~before & after & ~flag).sum()),
    }


def build_reference_zones(facies_path, zones_path):
    with np.load(facies_path, allow_pickle=False) as data:
        fac = data["facies"]
    with np.load(zones_path, allow_pickle=False) as data:
        z = data["zones"]
    if fac.shape != z.shape or hashlib.sha256(z.tobytes()).hexdigest() != ZONE_ARRAY_SHA:
        raise ValueError("Claude's frozen reference zone array has changed")
    rows = np.broadcast_to(np.arange(fac.shape[0], dtype=np.int32)[:, None], fac.shape)
    last = np.maximum.accumulate(np.where(np.isin(fac, (1, 2)), rows, -1), axis=0)
    depth = (rows-last).astype(np.float32) * np.float32(ZONE_POLICY["reference_pixel_area_mm2"] ** .5)
    m = np.zeros(fac.shape, np.uint8)
    l1 = z == 4
    m[l1 & (last >= 0) & (depth <= 40)] = 1
    m[l1 & (depth > 40) & (rows <= 2897+15)] = 2
    m[l1 & (rows > 2897+15)] = 3
    m[np.isin(z, (2, 3, 5)) & (last >= 0) & (depth >= 0) & (depth <= 40)] = 4
    return m


def native_zones(reference, row, shape):
    geometry = row["binding"]["source"]["rig_geometry"]
    H = np.asarray(geometry["H_ac28_to_frame_native"], np.float64)
    ox, oy = row["rig_domain_revision"]["native_origin_xy"]
    canvas = np.asarray([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], np.float64) @ H
    return cv2.warpPerspective(reference, canvas, shape[::-1], flags=cv2.INTER_NEAREST, borderValue=0)


def measured_gas(old, new, weights, zones):
    result = {}
    for name, code in (("l1_cap_band", 1), ("l1_plume", 2), ("below_l1", 3), ("caps_away_l1", 4)):
        selection = zones == code
        before = float(weights[selection & (old == 2)].sum(dtype=np.float64)) / 100
        after = float(weights[selection & (new == 2)].sum(dtype=np.float64)) / 100
        result.update({f"gas_{name}_old_cm2": before, f"gas_{name}_new_cm2": after,
                       f"gas_{name}_delta_cm2": after-before})
        if code == 4:
            relative = (after-before)/before if before > 0 else None
            result["caps_away_l1_relative_change"] = relative
            result["caps_away_l1_below_5_percent"] = abs(relative) < .05 if relative is not None else after == 0
    for value, name in ((1, "aqueous"), (2, "gas")):
        result[f"{name}_old_cm2"] = float(weights[old == value].sum(dtype=np.float64))/100
        result[f"{name}_new_cm2"] = float(weights[new == value].sum(dtype=np.float64))/100
    return result


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"QA never overwrites an existing report: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def frame_receipts(candidate, receipt_dir):
    folder = Path(candidate) / receipt_dir
    paths = sorted(folder.glob("*/frame_*.json"))
    if not paths:
        raise FileNotFoundError(f"Candidate has no completed per-frame receipts: {folder}")
    result = {}
    for path in paths:
        row = read_json(path)
        if not row.get("complete"):
            raise ValueError(f"Incomplete candidate frame: {path}")
        run = str(row.get("run") or row.get("binding", {}).get("source", {}).get("run") or path.parent.name).casefold()
        index = int(row["frame_index"])
        if (run, index) in result:
            raise ValueError("Duplicated candidate frame identity")
        result[run, index] = (path, row)
    return result


def count_inventory(validation_root, available):
    """Read count manifests only; never open held-out annotations or decisions."""
    snapshots = Path(validation_root) / "snapshots"
    selected = []
    keys = set()
    for round_name, pattern, count_key in (
        ("round2", "round2_*", "test_label_counts"),
        ("round3", "round3_*", "round3_counts"),
        ("round4", "round4b_*", "round4_counts"),
    ):
        folders = sorted(snapshots.glob(pattern))
        if len(folders) != 1:
            raise ValueError(f"Expected one sealed {round_name} inventory, found {len(folders)}")
        manifest_path = folders[0] / "SNAPSHOT_MANIFEST.json"
        manifest = read_json(manifest_path)
        row = {"round": round_name, "manifest": asset(manifest_path), "annotations_read": False,
               "annotation_seal": manifest["files"]["pairwise_review_annotations.csv"]}
        counts = manifest[count_key]
        if round_name == "round2":
            runs = {str(run).casefold() for run in counts}
            round_keys = {key for key in available if key[0] in runs}
            row["selection"] = "all saved candidate frames for the sealed three runs; count manifest does not list frame keys; declared superset"
        else:
            round_keys = set()
            for text in counts:
                match = re.match(r"(ac\d+)\s+f(\d+)", text, re.I)
                if not match:
                    raise ValueError("Sealed count key has no exact run/frame identity")
                round_keys.add((match.group(1).casefold(), int(match.group(2))))
            row["selection"] = "exact run/frame keys in sealed count inventory; annotations not read"
        if round_keys & {(run, i) for run, i in round_keys if run in BLIND}:
            raise PermissionError("A withheld bottom blind run must not enter the regression export")
        missing = round_keys - set(available)
        if missing:
            raise ValueError(f"Candidate missing sealed regression frames: {sorted(missing)}")
        row["frames"] = [{"run": run.upper(), "frame_index": index} for run, index in sorted(round_keys)]
        selected.append(row)
        keys |= round_keys
    return selected, keys


def inspect_run(task):
    cv2.setNumThreads(1)
    run, entries, reference_zones, diagnostic_root, output, job_binding, resume, base_candidate = task
    frame_rows = []
    for path, row in entries:
        index = int(row["frame_index"])
        outpath = Path(output) / "frame_qa" / run / f"frame_{index:03d}.json"
        if outpath.exists():
            if not resume:
                raise FileExistsError(outpath)
            cached = read_json(outpath)
            if cached.get("job_binding_sha256") != job_binding or cached["provenance"]["candidate_frame_receipt"] != asset(path):
                raise ValueError("Resume QA frame binding changed")
            for name in ("base_labels", "candidate_labels", "floor_contact_fill", "measured_area", "base_frame_receipt"):
                verified(cached["provenance"][name])
            frame_rows.append(cached)
            continue
        old = labels(row["before_labels"])
        new = labels(row["labels_map"])
        parent_path = Path(base_candidate)/"rig_camera_frames"/run.upper()/f"frame_{index:03d}.json"
        parent = read_json(parent_path)
        if (parent["source_sha256"] != row["source_sha256"] or float(parent["elapsed_h"]) != float(row["elapsed_h"])
                or parent["labels_map"]["sha256"] != row["before_labels"]["sha256"]
                or parent["rig_domain_revision"] != row["rig_domain_revision"]
                or parent["binding"]["source"] != row["binding"]["source"]):
            raise ValueError("Candidate sources, geometry, measured area or before-map differ from the exact seam-corrected base")
        area = area_map(row["rig_domain_revision"]["area_map"])
        mask = fill_mask(row["floor_contact_fill"], new.shape, row["source_sha256"], index)
        elapsed = float(row["elapsed_h"])
        q = floor_qa(old, new, mask, area, elapsed)
        if any(q[k] for k in ("filled_pixels_not_aqueous", "filled_pixels_already_co2_in_base", "preinjection_fill_pixels",
                              "filled_under_gas_pixels", "invalid_aqueous_trigger_columns", "discontinuous_fill_columns",
                              "co2_union_changes_outside_floor_contact_fill", "co2_union_removed_pixels")):
            raise ValueError(f"Candidate floor/unchanged-union QA failed: {run} frame {index}: {q}")
        saved = row["floor_rule"]
        if q["filled_pixels"] != saved["filled_pixels"] or not np.isclose(q["filled_area_cm2"], saved["filled_area_cm2"], atol=1e-8, rtol=0):
            raise ValueError("Numerical floor producer and independent measured QA disagree")
        q.update({k: saved[k] for k in ("observable_floor_columns", "filled_observable_floor_column_share", "filled_supported_floor_column_share")})
        metrics = measured_gas(old, new, area, native_zones(reference_zones, row, new.shape))
        result = dict(contract=CONTRACT, job_binding_sha256=job_binding, run=run.upper(), frame_index=index,
            source_sha256=row["source_sha256"], elapsed_h=elapsed, floor=q, gas=metrics,
            coordinate_binding={"phase_shape_hw": row["rig_domain_revision"]["phase_shape_hw"],
                "native_origin_xy": row["rig_domain_revision"]["native_origin_xy"],
                "photographic_canvas_wh": row["rig_domain_revision"]["canvas_wh"],
                "H_ac28_to_frame_native": row["binding"]["source"]["rig_geometry"]["H_ac28_to_frame_native"]},
            provenance={"candidate_frame_receipt": asset(path), "base_labels": row["before_labels"],
                        "base_frame_receipt": asset(parent_path),
                        "candidate_labels": row["labels_map"], "floor_contact_fill": row["floor_contact_fill"],
                        "measured_area": row["rig_domain_revision"]["area_map"]})
        _write_json(outpath, result)
        frame_rows.append(result)
    baseline = min((r for r in frame_rows if r["elapsed_h"] < 0), key=lambda r: abs(r["elapsed_h"]), default=None)
    if baseline is None:
        raise ValueError("Candidate has no own source-bound pre-injection frame")
    diagnostic_path = Path(diagnostic_root) / "frame_receipts" / run / f"frame_{baseline['frame_index']:03d}.json"
    old_diagnostic = read_json(diagnostic_path)
    old_source = old_diagnostic["frame_report"]["provenance"]["source_sha256"]
    if old_source != baseline["source_sha256"]:
        raise ValueError("Pre-injection colour diagnostic belongs to another source version")
    if old_diagnostic["frame_report"]["provenance"]["measured_area"]["sha256"] != baseline["provenance"]["measured_area"]["sha256"]:
        raise ValueError("Pre-injection colour diagnostic uses another physical area revision")
    band = old_diagnostic["preinject_run_report"]
    run_row = {"run": run.upper(), "preinject_frame_index": baseline["frame_index"],
        "preinject_elapsed_h": baseline["elapsed_h"], "source_sha256": baseline["source_sha256"],
        "description": "indicator colour present before injection; fixed 20 mm photographic proxy; not measured CO2 concentration",
        "proxy": band["carryover_proxy"]["20"], "diagnostic_frame_receipt": asset(diagnostic_path),
        "mass_eligible": False, "bottom_mass_allowed": False,
        "future_balance_policy": "subtract pre-injection indicator band from injected-CO2 balance; no mass computed"}
    return frame_rows, run_row


def export_scoring_frames(output, rows, keys, inventories):
    folder = Path(output) / "regression_export"
    exported = []
    for row in rows:
        key = row["run"].casefold(), row["frame_index"]
        if key not in keys:
            continue
        target = folder / key[0] / f"frame_{key[1]:03d}"
        target.mkdir(parents=True, exist_ok=True)
        copies = {}
        for label, descriptor in (("new_labels", row["provenance"]["candidate_labels"]),
                                  ("old_labels", row["provenance"]["base_labels"]),
                                  ("floor_contact_fill", row["provenance"]["floor_contact_fill"])):
            source = verified(descriptor)
            dest = target / (label + source.suffix)
            if dest.exists():
                if digest(dest) != descriptor["sha256"]:
                    raise ValueError("Existing regression export bytes differ")
            else:
                shutil.copyfile(source, dest)
            if digest(dest) != descriptor["sha256"]:
                raise IOError("Regression export copy failed SHA verification")
            copies[label] = asset(dest)
        value = dict(contract="ffac-g3-t2-floor-regression-export-v1", run=row["run"], frame_index=key[1],
            elapsed_h=row["elapsed_h"], source_sha256=row["source_sha256"], files=copies,
            coordinate_binding=row["coordinate_binding"], production_frame_receipt=row["provenance"]["candidate_frame_receipt"],
            measured_area=row["provenance"]["measured_area"], annotations_read=False,
            scoring="Claude's separately sealed scorer; conditional on fixed CO2 head/geometry",
            fixed_acceptance={"recall_minimum_relative_to_old": -.03, "false_gas_maximum_relative_to_old": .01})
        receipt = target / "receipt.json"
        if not receipt.exists():
            _write_json(receipt, value)
        elif read_json(receipt) != value:
            raise ValueError("Regression export receipt changed")
        exported.append(asset(receipt))
    return {"inventories": inventories, "exported_frame_count": len(exported), "frames": exported,
            "annotations_read": False, "scorer_status": "not run; Claude must score with presealed scorer"}


def _flatten(rows):
    return [{k: row[k] for k in ("run", "frame_index", "source_sha256", "elapsed_h")}
            | row["floor"] | row["gas"] for row in rows]


def _nearest(rows, targets):
    selected = []
    for run in sorted({r["run"] for r in rows}):
        group = [r for r in rows if r["run"] == run and r["elapsed_h"] > 0]
        for target in targets:
            row = min(group, key=lambda r: abs(r["elapsed_h"]-target))
            selected.append({"run": run, "target_elapsed_h": target, "frame_index": row["frame_index"],
                             "actual_elapsed_h": row["elapsed_h"], "time_distance_seconds": abs(row["elapsed_h"]-target)*3600,
                             **row["gas"]})
    return selected


def make_figures(output, candidate, rows, receipt_dir):
    """Exact saved photograph and old/new overlays; crop/resize only for report."""
    from darsia.fluidflower.phase_v5_display_asset import load_display_image
    from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
    from darsia.fluidflower.phase_floor_contact_display import hatch_floor_contact_rgba
    figures = []
    for run, targets in (("AC50", (6.0,)), ("AC28", (2.25, 4.0)), ("AC47", (2.25, 4.0))):
        group = [r for r in rows if r["run"] == run]
        for target in targets:
            result = min(group, key=lambda r: abs(r["elapsed_h"]-target))
            index = result["frame_index"]
            row = read_json(Path(candidate) / receipt_dir / run / f"frame_{index:03d}.json")
            photo = load_display_image(Path(candidate)/"source_alias", run.casefold(), index, mode="natural")
            photo = pad_rig_photograph(photo, row["rig_domain_revision"])
            from PySide6.QtGui import QImage
            photo = photo.convertToFormat(QImage.Format.Format_RGB888)
            rgb = np.frombuffer(photo.constBits(), np.uint8).reshape(photo.height(), photo.bytesPerLine())[:, :photo.width()*3].reshape(photo.height(), photo.width(), 3).copy()
            old = labels(row["before_labels"])
            new = labels(row["labels_map"])
            mask = fill_mask(row["floor_contact_fill"], new.shape, row["source_sha256"], index)
            if rgb.shape[:2] != new.shape:
                raise ValueError("Figure photograph and saved maps have different native grids")
            if run == "AC50":
                crop = [0, max(0, new.shape[0]-1000), new.shape[1]//2+200, new.shape[0]]
            else:
                H = np.asarray(row["binding"]["source"]["rig_geometry"]["H_ac28_to_frame_native"], float)
                point = H @ [2786, 2897, 1]
                point = point[:2]/point[2] - np.asarray(row["rig_domain_revision"]["native_origin_xy"])
                x, y = point
                crop = [max(0, int(x-1100)), max(0, int(y-1100)), min(new.shape[1], int(x+1100)), min(new.shape[0], int(y+700))]
            panels = []
            for title, phase, flag in (("Photograph", None, None), ("Old overlay", old, None), ("New overlay", new, mask)):
                pixels = rgb.copy()
                if phase is not None:
                    rgba = np.zeros((*phase.shape, 4), np.uint8)
                    rgba[phase == 1] = [70, 220, 110, 128]
                    rgba[phase == 2] = [230, 75, 140, 128]
                    rgba = hatch_floor_contact_rgba(rgba, flag)
                    alpha = rgba[..., 3:4].astype(np.float32)/255
                    pixels = np.rint(pixels*(1-alpha)+rgba[..., :3]*alpha).astype(np.uint8)
                panel = Image.fromarray(pixels).crop(tuple(crop))
                panel.thumbnail((900, 700), Image.Resampling.LANCZOS)
                labelled = Image.new("RGB", (panel.width, panel.height+45), "#152029")
                labelled.paste(panel, (0, 45))
                ImageDraw.Draw(labelled).text((10, 12), f"{run} {result['elapsed_h']:.4f} h — {title}", fill="white")
                panels.append(labelled)
            figure = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "#152029")
            x = 0
            for panel in panels:
                figure.paste(panel, (x, 0)); x += panel.width
            target_path = Path(output)/"figures"/f"{run}_frame_{index:03d}_old_new.png"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                raise FileExistsError(target_path)
            figure.save(target_path)
            figures.append(dict(run=run, frame_index=index, target_elapsed_h=target, actual_elapsed_h=result["elapsed_h"],
                image=asset(target_path), native_crop_xyxy=crop, labels_resampled=False,
                photograph_display="Natural; report thumbnail only, model arrays unchanged"))
    return figures


def run(args):
    candidate, output = Path(args.candidate), Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError("Choose a new report directory or resume the exact report binding")
    output.mkdir(parents=True, exist_ok=True)
    available = frame_receipts(candidate, args.receipt_dir)
    if not args.allow_partial and (len(available) != 864 or len({k[0] for k in available}) != 43):
        raise ValueError("Full QA requires exactly 43 runs / 864 unique completed frames")
    zones = build_reference_zones(args.facies_map, args.zone_map)
    inventories, test_keys = count_inventory(args.validation_root, available)
    inputs = {"candidate": str(candidate), "receipt_dir": args.receipt_dir, "base_candidate": str(args.base_candidate),
              "facies_map": asset(args.facies_map), "zone_map": asset(args.zone_map), "zone_policy": ZONE_POLICY,
              "bottom_diagnostic_receipt": asset(Path(args.diagnostic_root)/"report_receipt.json"),
              "sealed_count_inventories": inventories, "contract": CONTRACT}
    binding = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    by_run = {}
    for (run_id, _), value in available.items():
        by_run.setdefault(run_id, []).append(value)
    workers = max(1, min(24, args.workers, len(by_run)))
    try:
        import psutil
        workers = min(workers, max(1, int(psutil.virtual_memory().available // (2 << 30))))
    except ImportError:
        workers = min(workers, 2)
    tasks = [(run_id, entries, zones, args.diagnostic_root, output, binding, args.resume, args.base_candidate) for run_id, entries in by_run.items()]
    frames, runs = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(inspect_run, task): task[0] for task in tasks}
        for future in as_completed(pending):
            frame_rows, run_row = future.result()
            frames.extend(frame_rows); runs.append(run_row)
            print(json.dumps({"run": pending[future], "completed_runs": len(runs), "completed_frames": len(frames), "workers": workers}), flush=True)
    frames.sort(key=lambda row: (row["run"], row["frame_index"]))
    runs.sort(key=lambda row: row["run"])
    public_frames, public_runs = public_rows(frames), public_rows(runs)
    export = export_scoring_frames(output, frames, test_keys, inventories)
    figures = make_figures(output, candidate, public_frames, args.receipt_dir) if not args.no_figures else []
    report_paths = {}
    for name, frows, rrows in (("internal", frames, runs), ("public", public_frames, public_runs)):
        frame_csv, run_csv = output/f"floor_gas_frames_{name}.csv", output/f"preinject_band_{name}.csv"
        if not frame_csv.exists(): _csv(frame_csv, _flatten(frows))
        if not run_csv.exists(): _csv(run_csv, [{k: r[k] for k in ("run", "preinject_frame_index", "preinject_elapsed_h", "source_sha256")}
            | {"description": r["description"]} | r["proxy"] for r in rrows])
        detail = {"contract": CONTRACT, "frame_count": len(frows), "run_count": len(rrows), "frames": frows, "preinject_bands": rrows,
                  "bottom_mass_allowed": False, "mass_eligible": False, "zone_policy": ZONE_POLICY}
        detail_path = output/f"qa_{name}.json"
        if not detail_path.exists(): _write_json(detail_path, detail)
        report_paths[name] = dict(json=asset(detail_path), frame_csv=asset(frame_csv), run_csv=asset(run_csv),
            frame_count=len(frows), run_count=len(rrows))
    targets = _nearest(public_frames, (1.0, 1.25, 2.25, 4.0, 6.0))
    target_csv = output/"gas_target_times_public.csv"
    if not target_csv.exists(): _csv(target_csv, targets)
    sums = {name: sum(row["floor"][name] for row in public_frames) for name in ("filled_pixels", "filled_area_cm2", "preinjection_fill_pixels", "filled_under_gas_pixels", "co2_union_changes_outside_floor_contact_fill")}
    receipt = {"contract": CONTRACT, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_binding_sha256": binding, "inputs": inputs, "reports": report_paths, "public_floor_qa_totals": sums,
        "target_time_report": asset(target_csv), "regression_export": export, "figures": figures,
        "bottom_mass_allowed": False, "mass_eligible": False, "default_candidate_changed": False,
        "regression_acceptance": "pending Claude's sealed scorer; not scientifically accepted for mass",
        "blind_policy": "AC30/AC49/AC54 excluded from all public rows, figures and numerical aggregates"}
    receipt_path = output/"report_receipt.json"
    if not receipt_path.exists(): _write_json(receipt_path, receipt)
    print(json.dumps({"complete": True, "receipt": str(receipt_path), "public_runs": len(public_runs), "public_frames": len(public_frames)}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "base-candidate", "output", "diagnostic-root", "validation-root", "facies-map", "zone-map"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--receipt-dir", default="rig_camera_frames")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
