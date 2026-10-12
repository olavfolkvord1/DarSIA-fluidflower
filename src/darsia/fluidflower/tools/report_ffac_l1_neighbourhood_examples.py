"""Authorized promoted B-T2 in-sample check and public L1 display examples.

This is not the sealed regression or blind scorer. Only the SHA-whitelisted
AC30/49/54 training snapshot is read; later labels and AC26/55 maps are never
opened. Photographs and overlays are report artifacts, never scientific input.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
from darsia.fluidflower.phase_l1_neighbourhood_gas import apply_l1_neighbourhood, l1_neighbourhood_zone

CONTRACT = "ffac-l1-neighbourhood-in-sample-examples-v1"
BASE_ID = "phase-model-20261005T121928Z-g3-old-domain-seam-corrected-ffac"
PROMOTED_RUNS = frozenset(("ac30", "ac49", "ac54"))
BT2_CSV_SHA256 = "4b08f5e1a75b01dab349dfd2f9826d3fe4791ad7bc6e39cae24b2a962de22680"
BT2_SESSION_SHA256 = "2988273a8d68e52c81b7ad94efc14c8457fb4960edf45b9199e8466bcf488f07"
BT2_MANIFEST_SHA256 = "f1b5020599ef5da2419c0237023b0b55ebd2f43e4bc64b9d8e97386812f92c76"
CW, CH = 6304, 3546
PHASE_SHAPE = (3572, 6304)
MM_PER_PX = .16
GAS_CATEGORY = "confirmed_only_free_gas"
AQ_CATEGORY = "confirmed_only_aqueous_co2"
VARIANTS = ("base", "unrestricted_one_pass", "production_excluded_floor")
CLAUDE_IN_SAMPLE = {.3: dict(recall_l1=.876, false_gas_l1=.057),
                    .5: dict(recall_l1=.687, false_gas_l1=.038)}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(path):
    return dict(path=str(Path(path).resolve()), sha256=_digest(path))


def _checked(path, expected):
    path = Path(path)
    if _digest(path) != expected:
        raise ValueError("Promoted B-T2 snapshot SHA differs from the authorized immutable input")
    return path


def _write(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def require_promoted_run(run):
    if str(run).casefold() not in PROMOTED_RUNS:
        raise PermissionError("This report opens only authorized promoted AC30/49/54 data")


def load_promoted_rows(csv_path, base_candidate):
    """Only one exact already-promoted CSV and its sealed session metadata."""
    base = Path(base_candidate).resolve()
    if base.name != BASE_ID:
        raise ValueError("In-sample check requires the frozen old-domain-seam G3 base, never T3")
    csv_path = Path(csv_path)
    if csv_path.name != "blind_annotations_ac30_ac49_ac54.csv":
        raise PermissionError("Only the explicitly promoted B-T2 CSV may be opened")
    csv_path = _checked(csv_path, BT2_CSV_SHA256)
    session_path = _checked(csv_path.parent / "session_copy/review_session.json", BT2_SESSION_SHA256)
    manifest_path = _checked(csv_path.parent / "MANIFEST.json", BT2_MANIFEST_SHA256)
    session, manifest = _read(session_path), _read(manifest_path)
    if (session.get("target_model_id") != BASE_ID or
            manifest.get("contract") != "claude-ffac-blind-bt2-label-snapshot-v1" or
            manifest.get("blind_annotations_sha256") != BT2_CSV_SHA256):
        raise ValueError("Promoted B-T2 snapshot belongs to another base candidate")
    validation_path = Path(session["combined_root"]) / "validation_binding.json"
    validation = _read(validation_path)
    if (validation.get("model_id") != BASE_ID or Path(validation["model_root"]).resolve() != base):
        raise ValueError("Promoted B-T2 validation binding belongs to another base")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (len({r["annotation_id"] for r in rows}) != 321 or
            Counter(r["category"] for r in rows) != {GAS_CATEGORY: 140, AQ_CATEGORY: 181} or
            Counter(r["run"] for r in rows) != {"ac30": 103, "ac49": 106, "ac54": 112}):
        raise ValueError("Promoted B-T2 unique/class/run inventory differs")
    native = {run: _read(base / "source_alias/native_frames" / run / "manifest.json") for run in PROMOTED_RUNS}
    native_sha = {run: _digest(base / "source_alias/native_frames" / run / "manifest.json") for run in PROMOTED_RUNS}
    for row in rows:
        require_promoted_run(row["run"])
        run, index = row["run"], int(row["frame_index"])
        if index not in (5, 8, 11, 13):
            raise ValueError("Promoted B-T2 frame is outside the fixed training snapshot")
        source = native[run]["frame_sources"][index]
        validate_box_binding(row, session, validation, source)
        row["_source_sha256"] = source["source_sha256"]
        row["_native_manifest_sha256"] = native_sha[run]
    return rows, dict(annotation_csv=_asset(csv_path), session=_asset(session_path),
        snapshot_manifest=_asset(manifest_path), validation_binding=_asset(validation_path),
        base_native_sources={run: _asset(base / "source_alias/native_frames" / run / "manifest.json")
                             for run in sorted(PROMOTED_RUNS)})


def validate_box_binding(row, session, validation, source):
    require_promoted_run(row["run"])
    run, index, elapsed = row["run"], int(row["frame_index"]), float(row["elapsed_h"])
    if (row["candidate_id"] != session["selected_candidates"][run] or
            row["candidate_id"] != validation["selected_candidates"][run] or
            row["params_sha256"] != validation["pipeline_sha256"] or
            source.get("frame_index") != index or
            not math.isclose(elapsed, float(source["elapsed_h"]), rel_tol=0, abs_tol=1e-8) or
            row["coordinate_space"] != "normalized_native_review_canvas_16x9" or
            row["shape"] != "rectangle" or row["contours_visible"].casefold() != "false"):
        raise ValueError("Promoted B-T2 box lost its candidate/source/time/coordinate binding")
    for name in ("x", "y", "width", "height"):
        value = float(row[name])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Promoted B-T2 normalized coordinate is invalid")


def polygon_pixels(row):
    """Literal sealed scorer convention: CH3546, round then cv2.fillPoly."""
    if row["shape"] == "rectangle":
        x, y, width, height = (float(row[key]) for key in ("x", "y", "width", "height"))
        points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    else:
        points = json.loads(row.get("points_json") or "[]")
        points = [(p["x"], p["y"]) if isinstance(p, dict) else p[:2] for p in points]
    points = np.asarray([(x * CW, y * CH) for x, y in points], np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Promoted region lacks a finite polygon")
    return np.round(points).astype(np.int32)


def box_raster(row, shape):
    """Crop allocation only; raster coordinates equal the full-grid fillPoly."""
    points = polygon_pixels(row)
    x0, y0 = np.maximum(points.min(0), 0)
    x1, y1 = np.minimum(points.max(0) + 1, (shape[1], shape[0]))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0), np.zeros((0, 0), bool)
    bounds = tuple(map(int, (x0, y0, x1, y1)))
    image = np.zeros((int(y1 - y0), int(x1 - x0)), np.uint8)
    cv2.fillPoly(image, [points - (x0, y0)], 1)
    return bounds, image.astype(bool)


def floor_rows(labels):
    domain = labels != 255
    reverse = domain[::-1]
    return np.where(reverse.any(0), labels.shape[0] - 1 - reverse.argmax(0), -1).astype(float)


def score_box(row, maps, l1_xy, *, floor_row=None):
    require_promoted_run(row["run"])
    base = maps["base"]
    if any(value.shape != base.shape or value.dtype != np.uint8 for value in maps.values()):
        raise ValueError("In-sample map shape/type differs")
    bounds, drawn = box_raster(row, base.shape)
    x0, y0, x1, y1 = bounds
    selected = drawn & (base[y0:y1, x0:x1] != 255)
    count = int(selected.sum())
    result = dict(run=row["run"], frame_index=int(row["frame_index"]), annotation_id=row["annotation_id"],
                  category=row["category"], px_drawn=int(drawn.sum()), px_in_domain=count)
    if not count:
        result["selection"] = "no_in_domain_pixels"
        return result
    ys, xs = np.nonzero(selected); ys = ys + y0; xs = xs + x0
    floor_row = floor_rows(base) if floor_row is None else floor_row
    floor = float(np.median(floor_row[np.unique(xs)]))
    bottom_distance, top_distance = floor - ys.max(), floor - ys.min()
    floor_box = bool(bottom_distance <= 5. / MM_PER_PX and top_distance <= 25. / MM_PER_PX)
    cx, cy = float(xs.mean()), float(ys.mean())
    distance = float(np.hypot(cx - l1_xy[0], cy - l1_xy[1]))
    result.update(cx=cx, cy=cy, bottom_to_floor_px=float(bottom_distance), top_to_floor_px=float(top_distance),
                  floor_box=floor_box, dist_l1_px=distance,
                  selection="gas_150mm" if row["category"] == GAS_CATEGORY and distance <= 150. / MM_PER_PX
                  else "aqueous_nonfloor_60mm" if row["category"] == AQ_CATEGORY and not floor_box
                  and distance <= 60. / MM_PER_PX else "outside_metric_scope")
    for name in VARIANTS:
        part = maps[name][y0:y1, x0:x1][selected]
        result[name + "_gas_fraction"] = float(np.mean(part == 2))
        result[name + "_co2_fraction"] = float(np.mean((part == 1) | (part == 2)))
    return result


def aggregate_boxes(rows, *, gas_fraction_threshold=.3):
    if type(gas_fraction_threshold) not in (int, float) or gas_fraction_threshold not in CLAUDE_IN_SAMPLE:
        raise ValueError("In-sample report requires the approved .3 or .5 threshold")
    result = {}
    for kind, metric in (("gas_150mm", "recall_l1"), ("aqueous_nonfloor_60mm", "false_gas_l1")):
        selected = [row for row in rows if row["selection"] == kind]
        result[metric] = dict(box_count=len(selected), **{
            name: float(np.mean([row[name + "_gas_fraction"] for row in selected])) if selected else None
            for name in VARIANTS})
        if selected:
            values = result[metric]
            values["exclusions_delta"] = values["production_excluded_floor"] - values["unrestricted_one_pass"]
    reference = CLAUDE_IN_SAMPLE[gas_fraction_threshold]
    result["claude_in_sample_reference"] = dict(**reference, gas_fraction_threshold=gas_fraction_threshold,
        excludes_barriers_and_rule_b=False, independent_validation=False)
    for metric, expected in reference.items():
        value = result[metric]["unrestricted_one_pass"]
        result[metric]["unrestricted_difference_from_claude"] = None if value is None else value - expected
    return result


def _labels(asset):
    with Image.open(bound_path(asset)) as image:
        labels = np.array(image)
    if labels.dtype != np.uint8 or labels.shape != PHASE_SHAPE or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("Stored maps differ from the fixed uint8 categorical phase grid")
    return labels


def _flag(asset, source, shape):
    if (asset.get("source_sha256") != source["source_sha256"] or asset.get("frame_index") != source["frame_index"] or
            tuple(asset.get("shape_hw", ())) != shape or asset.get("storage") != "packbits-npy"):
        raise ValueError("Report flag lost source/frame/grid identity")
    value = np.load(bound_path(asset), allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != ((math.prod(shape) + 7) // 8,):
        raise ValueError("Invalid stored report mask")
    return np.unpackbits(value, count=math.prod(shape)).reshape(shape).astype(bool)


def load_frame(candidate, base_candidate, run, index):
    require_promoted_run(run)
    from . import run_ffac_l1_neighbourhood_candidate as runner
    root, plan, _sources = runner.candidate_state(candidate)
    rule_settings = runner.plan_rule_settings(plan)
    if Path(plan['binding']['parent']).resolve()!=Path(base_candidate).resolve():
        raise ValueError("Examples must use the candidate's exact frozen G3 base, never the .3 revision")
    paths = [Path(root) / "rig_camera_frames" / run / f"frame_{index:03d}.json" for root in (base_candidate, candidate)]
    base, new = map(_read, paths)
    source = base["binding"]["source"]
    if (base.get("complete") is not True or new.get("complete") is not True or
            new.get("contract") != "ffac-g3-l1-neighbourhood-floor-candidate-v1" or
            new.get("binding", {}).get("source") != source or
            new["before_labels"] != base["labels_map"] or new["rig_domain_revision"] != base["rig_domain_revision"] or
            new["source_sha256"] != base["source_sha256"] or new["elapsed_h"] != base["elapsed_h"] or
            source["run"] != run or source["frame_index"] != index):
        raise ValueError("Examples candidate and frozen base lost their exact source/time/geometry/map binding")
    if bound_path(new["binding"]["parent_receipt"]) != paths[0].resolve():
        raise ValueError("Examples candidate receipt is not bound to this exact frozen base frame")
    if (new['binding']['pipeline_sha256']!=plan['pipeline_sha256']
            or new.get('gas_fraction_threshold',.3)!=rule_settings['gas_fraction_threshold']
            or new.get('spec_sha256',runner.SPEC_SHA256)!=rule_settings['spec_sha256']):
        raise ValueError("Examples frame differs from its immutable threshold/spec/pipeline")
    before, after = _labels(base["labels_map"]), _labels(new["labels_map"])
    zone = l1_neighbourhood_zone(before.shape, source["rig_geometry"]["H_ac28_to_frame_native"], source["elapsed_h"])
    if not np.allclose(new["l1_point_xy"], zone.point_xy, rtol=0, atol=1e-8):
        raise ValueError("Candidate L1 point differs from the frozen source homography")
    unrestricted = apply_l1_neighbourhood(before,
        H_ac28_to_frame_native=source["rig_geometry"]["H_ac28_to_frame_native"], elapsed_hours=source["elapsed_h"],
        barrier_mask=np.zeros(before.shape, bool), rule_b_forbidden_mask=np.zeros(before.shape, bool),
        gas_fraction_threshold=rule_settings['gas_fraction_threshold']).labels
    gas = _flag(new["l1_neighbourhood_gas"], source, before.shape)
    fill = _flag(new["floor_contact_fill"], source, before.shape)
    expected = before.copy(); expected[gas] = 2; expected[fill] = 1
    if (not np.array_equal(expected, after) or np.any(gas & (before != 1)) or np.any(fill & (before != 0)) or
            not np.array_equal(after == 255, before == 255)):
        raise ValueError("Example maps have differences outside the two certified interpretation flags")
    return dict(maps=dict(base=before, unrestricted_one_pass=unrestricted, production_excluded_floor=after),
        base=base, new=new, gas_flag=gas, floor_flag=fill, point_xy=zone.point_xy,
        rule_settings=rule_settings,
        provenance=dict(base_frame_receipt=_asset(paths[0]), candidate_frame_receipt=_asset(paths[1]),
                        candidate_plan=runner.artifact(root/runner.PLAN), rule_settings=rule_settings,
                        source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"]))


def score_frame(task):
    cv2.setNumThreads(1)
    started = time.process_time()
    candidate, base, run, index, boxes = task
    frame = load_frame(candidate, base, run, index)
    for row in boxes:
        if (row["_source_sha256"] != frame["provenance"]["source_sha256"] or
                row["params_sha256"] != frame["base"]["binding"]["pipeline_sha256"] or
                not math.isclose(float(row["elapsed_h"]), frame["base"]["elapsed_h"], rel_tol=0, abs_tol=1e-8)):
            raise ValueError("Promoted box belongs to another exact source frame/time/base pipeline")
    floor = floor_rows(frame["maps"]["base"])
    rows = [score_box(row, frame["maps"], frame["point_xy"], floor_row=floor) for row in boxes]
    return rows, dict(run=run, frame_index=index, **frame["provenance"], worker_cpu_s=time.process_time() - started)


def figure_image(rgb, base, new, gas_flag, floor_flag, point_xy, *, run, elapsed_h, gas_fraction_threshold=.3):
    require_promoted_run(run)
    from darsia.fluidflower.phase_floor_contact_display import hatch_floor_contact_rgba
    from darsia.fluidflower.phase_l1_neighbourhood_display import hatch_l1_neighbourhood_rgba
    if rgb.dtype != np.uint8 or rgb.shape != (*base.shape, 3) or new.shape != base.shape:
        raise ValueError("Photograph and overlays have different phase-grid coordinates")
    x, y = point_xy
    crop = (max(0, int(x - 1100)), max(0, int(y - 1100)), min(new.shape[1], int(x + 1100)), min(new.shape[0], int(y + 700)))
    panels = []
    for title, phase in (("Photograph", None), ("Base G3 overlay", base), ("Neighbourhood + floor overlay", new)):
        pixels = rgb.copy()
        if phase is not None:
            rgba = np.zeros((*phase.shape, 4), np.uint8)
            rgba[phase == 1] = [70, 220, 110, 128]; rgba[phase == 2] = [230, 75, 140, 128]
            if phase is new:
                rgba = hatch_l1_neighbourhood_rgba(rgba, gas_flag)
                rgba = hatch_floor_contact_rgba(rgba, floor_flag)
            alpha = rgba[..., 3:4].astype(np.float32) / 255.
            pixels = np.rint(pixels * (1 - alpha) + rgba[..., :3] * alpha).astype(np.uint8)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.ellipse((x - 937.5, y - 937.5, x + 937.5, y + 937.5), outline="#ffe26a", width=5)
        draw.line((x - 12, y, x + 12, y), fill="#ffe26a", width=3)
        draw.line((x, y - 12, x, y + 12), fill="#ffe26a", width=3)
        image = image.crop(crop); image.thumbnail((1000, 700), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (image.width, image.height + 68), "#152029"); panel.paste(image, (0, 68))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 8), f"{run.upper()} {elapsed_h:.4f} h - {title}", fill="white")
        if phase is new:
            draw.text((10, 31), f"L1 gas fraction >= {gas_fraction_threshold:g}; purple vertical gas / teal diagonal floor", fill="white")
        panels.append(panel)
    figure = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "#152029")
    offset = 0
    for panel in panels: figure.paste(panel, (offset, 0)); offset += panel.width
    return figure, crop


def make_figures(output, candidate, base_candidate):
    from darsia.fluidflower.phase_v5_display_asset import load_display_image
    from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
    from PySide6.QtGui import QImage
    records = []
    for run in ("ac30", "ac49"):
        require_promoted_run(run)
        native = _read(Path(base_candidate) / "source_alias/native_frames" / run / "manifest.json")
        for target in (2.25, 4.):
            selected = min(native["frame_sources"], key=lambda row: abs(float(row["elapsed_h"]) - target))
            index = int(selected["frame_index"])
            frame = load_frame(candidate, base_candidate, run, index)
            photo = pad_rig_photograph(load_display_image(Path(candidate) / "source_alias", run, index, mode="natural"),
                                       frame["new"]["rig_domain_revision"]).convertToFormat(QImage.Format.Format_RGB888)
            rgb = np.frombuffer(photo.constBits(), np.uint8).reshape(photo.height(), photo.bytesPerLine())[:, :photo.width() * 3]
            rgb = rgb.reshape(photo.height(), photo.width(), 3).copy()
            image, crop = figure_image(rgb, frame["maps"]["base"], frame["maps"]["production_excluded_floor"],
                frame["gas_flag"], frame["floor_flag"], frame["point_xy"], run=run, elapsed_h=selected["elapsed_h"],
                gas_fraction_threshold=frame['rule_settings']['gas_fraction_threshold'])
            path = Path(output) / "figures" / f"{run}_frame_{index:03d}_L1_{target:.2f}h.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists(): raise FileExistsError(path)
            image.save(path)
            records.append(dict(run=run, frame_index=index, target_elapsed_h=target, actual_elapsed_h=selected["elapsed_h"],
                image=_asset(path), native_crop_xyxy=list(crop), **frame["provenance"],
                photograph="source-bound Natural; unchanged coordinates; report display only"))
    return records


def run(args):
    started, cpu = time.perf_counter(), time.process_time()
    candidate, base, output = map(lambda value: Path(value).resolve(), (args.candidate, args.base_candidate, args.output))
    if output.exists() or output == candidate or output == base or output.is_relative_to(base) or output.is_relative_to(candidate):
        raise ValueError("Use a new separate immutable report directory")
    from . import run_ffac_l1_neighbourhood_candidate as runner
    root, plan, _sources = runner.candidate_state(candidate)
    rule_settings = runner.plan_rule_settings(plan)
    threshold = rule_settings['gas_fraction_threshold']
    if Path(plan['binding']['parent']).resolve()!=base:
        raise ValueError("Examples require the exact original frozen base rather than .3 candidate maps")
    rows, snapshot = load_promoted_rows(args.bt2_csv, base)
    groups = defaultdict(list)
    for row in rows: groups[(row["run"], int(row["frame_index"]))].append(row)
    tasks = [(candidate, base, run, index, boxes) for (run, index), boxes in sorted(groups.items())]
    if args.workers <= 0: raise ValueError("Report CPU workers must be positive")
    scored, provenance = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        for boxes, frame in pool.map(score_frame, tasks):
            scored.extend(boxes); provenance.append(frame)
    if len(scored) != 321: raise ValueError("In-sample check lost an authorized promoted box")
    if any(frame['rule_settings']!=rule_settings for frame in provenance):
        raise ValueError("In-sample frames changed immutable rule settings during scoring")
    output.mkdir(parents=True, exist_ok=False)
    _csv(output / "bt2_promoted_in_sample_boxes.csv", scored)
    figures = make_figures(output, candidate, base) if not args.no_figures else []
    report = dict(contract=CONTRACT, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        candidate=str(candidate), base_candidate=str(base), code=_asset(__file__), promoted_snapshot=snapshot,
        candidate_plan=runner.artifact(root/runner.PLAN),rule_settings=rule_settings,
        box_count=321, pooled=aggregate_boxes(scored,gas_fraction_threshold=threshold),
        per_run={run: aggregate_boxes([r for r in scored if r["run"] == run],gas_fraction_threshold=threshold)
                 for run in sorted(PROMOTED_RUNS)},
        source_frames=provenance, figures=figures,
        raster_definition="np.round(normalized polygon ×[6304,3546]); cv2.fillPoly; clip label!=255; equal-box mean",
        floor_definition="median floor rows on box columns; bottom<=5/.16 and top<=25/.16 pixels",
        metric_scope="gas boxes centred within150mm of L1; only-aqueous nonfloor boxes within60mm",
        independent_validation=False, acceptance_claimed=False, in_sample=True,
        promoted_training_snapshot_read=True, later_regression_labels_read=False, new_blind_labels_read=False,
        hidden_runs_data_read=False, sealed_scorer_imported=False, sealed_scorer_run=False,
        scientific_sidecars_read=False, mass_eligible=False, bottom_mass_allowed=False,
        default_candidate_changed=False, models_changed=False,
        cost=dict(report_parent_cpu_s=time.process_time() - cpu,
            report_worker_cpu_s=sum(frame["worker_cpu_s"] for frame in provenance),
            wall_s=time.perf_counter() - started, workers=args.workers))
    _write(output / "in_sample_examples_report.json", report)
    receipt = dict(contract=CONTRACT, complete=True, independent_validation=False, acceptance_claimed=False,
        candidate=str(candidate), base_candidate=str(base), report=_asset(output / "in_sample_examples_report.json"),
        box_csv=_asset(output / "bt2_promoted_in_sample_boxes.csv"), figures=figures, pooled=report["pooled"],
        promoted_snapshot=snapshot, sealed_scorer_run=False, hidden_runs_data_read=False)
    receipt['rule_settings']=rule_settings
    _write(output / "examples_receipt.json", receipt)
    print(json.dumps(dict(complete=True, report_receipt=str(output / "examples_receipt.json"), pooled=report["pooled"])), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "base-candidate", "bt2-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-figures", action="store_true")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
