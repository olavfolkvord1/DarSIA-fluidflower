"""Supplemental P1-final figures for the maximum combined A+B removal.

Saved labels and source-bound area maps are read, never modified. The earlier
top-only figures remain untouched; this report makes the broader interpretation
of section 4d explicit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import report_ffac_p1_final as original

CONTRACT = "ffac-p1-final-combined-removal-figures-supplement-v1"
AREA_FIELDS = ("removed_floor_cm2", "removed_top_aqueous_cm2", "removed_top_gas_cm2")
TOP_COLOUR = (255, 205, 35)
FLOOR_COLOUR = (40, 220, 255)


def select_maximum(rows):
    """Select by measured combined area, resolving equal areas by first frame."""
    if not rows:
        raise ValueError("No saved per-frame QA rows")
    keys = [(r["run"].casefold(), int(r["frame_index"])) for r in rows]
    if len(set(keys)) != len(keys) or len({r for r, _ in keys}) != 1:
        raise ValueError("Selection needs unique frame rows from one run")
    for row in rows:
        values = [float(row[k]) for k in AREA_FIELDS]
        if not np.isfinite(values).all() or min(values) < 0 or row.get("all_pass") is not True:
            raise ValueError("Invalid or failed saved area QA")
    return max(rows, key=lambda r: (sum(float(r[k]) for k in AREA_FIELDS), -int(r["frame_index"])))


def actual_removals(before, base, after, floor_flags, top_flags):
    """Exclude unchanged provenance flags from the actual floor-removal outline."""
    shape = before.shape
    if any(a.shape != shape for a in (base, after, floor_flags, top_flags)):
        raise ValueError("Removal arrays do not have one geometry")
    floor = floor_flags & (before != base)
    step_a = before.copy()
    step_a[floor_flags] = base[floor_flags]
    if np.any(floor & top_flags) or not np.array_equal(before != after, floor | top_flags):
        raise ValueError("Saved final label changes differ from the two actual removal masks")
    if np.any(after[top_flags] != 0):
        raise ValueError("Top flags are not removed CO2 pixels")
    return floor, top_flags, step_a


def outline(mask):
    boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # Only the photograph receives a wider outline for visibility in thumbnails.
    return cv2.dilate(boundary.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)


def render_run(parent, candidate, selected, output):
    started, cpu_started = time.perf_counter(), time.process_time()
    cv2.setNumThreads(1)
    run, index = selected["run"].casefold(), int(selected["frame_index"])
    manifest_path, manifest = original.native_manifest(parent, run)
    input_asset = manifest["ffac_rig_camera_sources"]["input_manifest"]
    inputs = original.read(original.bound(input_asset, base=manifest_path.parent))
    source = manifest["frame_sources"][index]
    final_path = original.frame_path(candidate, run, index)
    parent_path = original.frame_path(parent, run, index)
    final, previous = original.read(final_path), original.read(parent_path)
    for row in (source, final, previous):
        if row["source_sha256"] != selected["source_sha256"] or float(row["elapsed_h"]) != float(selected["elapsed_h"]):
            raise ValueError("Selected figure source/time binding differs from saved QA")
    if final.get("complete") is not True or final.get("bottom_mass_allowed") is not False:
        raise ValueError("Final frame is incomplete or bottom mass is enabled")
    if any(float(final["qa"][k]) != float(selected[k]) for k in AREA_FIELDS):
        raise ValueError("Selected saved QA differs from the exact final frame receipt")
    before = np.asarray(Image.open(original.bound(previous["labels_map"])))
    base = np.asarray(Image.open(original.bound(final["base_labels"])))
    after = np.asarray(Image.open(original.bound(final["labels_map"])))
    floor_flags = original.unpack_mask(final["floor_contact_fill"], before.shape)
    top_flags = original.unpack_mask(final["top_surface_rule_masks"], before.shape)
    floor, top, step_a = actual_removals(before, base, after, floor_flags, top_flags)
    domain, area = original.arrays(final["rig_domain_revision"])
    actual = {
        "removed_floor_cm2": float(area[floor].sum(dtype=np.float64)) / 100.,
        "removed_top_aqueous_cm2": float(area[top & (step_a == 1)].sum(dtype=np.float64)) / 100.,
        "removed_top_gas_cm2": float(area[top & (step_a == 2)].sum(dtype=np.float64)) / 100.,
    }
    for name, value in actual.items():
        if not np.isclose(value, float(selected[name]), atol=1e-8, rtol=0):
            raise ValueError("Measured selected removal area differs from the sealed QA")
    photo_asset = inputs["frames"][index]["photographs"]["lift1"]
    photo, valid = original.photo_canvas(photo_asset, before.shape, manifest)
    # Extend the photographed bottom from the same recorded RAW assets. This is
    # a display-only construction; unsupported pixels retain the original marker.
    occupied = domain.any(axis=0)
    bottom = domain.shape[0] - 1 - np.argmax(domain[::-1], axis=0)[occupied]
    y0 = max(0, int(np.percentile(bottom, 10)) - 360)
    y1 = min(domain.shape[0], int(np.percentile(bottom, 90)) + 30)
    raw, support, _ = original.raw_roi(source, manifest, y0, y1)
    modes, _ = original.render_display_modes(raw, domain[y0:y1])
    raw_display = np.rint(modes["lift1"] * 255).astype(np.uint8)
    extend = ~valid[y0:y1] & support
    photo[y0:y1][extend] = raw_display[extend]
    photo[outline(floor)] = FLOOR_COLOUR
    photo[outline(top)] = TOP_COLOUR
    image = Image.fromarray(photo)
    image.thumbnail((1200, 720), Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", (image.width, image.height + 90), "#181b20")
    draw = ImageDraw.Draw(sheet)
    total = sum(actual.values())
    draw.text((12, 10), f"{run.upper()} frame {index} t={selected['elapsed_h']:.3f} h; combined A+B removal {total:.3f} cm2", fill="white")
    draw.text((12, 32), f"Cyan: floor fill removed, {actual['removed_floor_cm2']:.3f} cm2", fill=FLOOR_COLOUR)
    draw.text((12, 54), f"Yellow: top-surface CO2 removed, {actual['removed_top_aqueous_cm2'] + actual['removed_top_gas_cm2']:.3f} cm2", fill=TOP_COLOUR)
    sheet.paste(image, (0, 90))
    output_path = Path(output) / f"{run.upper()}_frame_{index:03d}_combined_removed_outline.png"
    if output_path.exists():
        raise FileExistsError(str(output_path))
    sheet.save(output_path)
    return {
        "run": run.upper(), "frame_index": index, "elapsed_h": float(selected["elapsed_h"]),
        "source_sha256": selected["source_sha256"], **actual, "combined_removed_cm2": total,
        "removed_floor_pixels": int(floor.sum()), "removed_top_pixels": int(top.sum()),
        "selection": "Maximum measured removed_floor_cm2 + removed_top_aqueous_cm2 + removed_top_gas_cm2; ties use earliest frame index",
        "image": original.artifact(output_path),
        "provenance": {"parent_receipt": original.artifact(parent_path), "final_receipt": original.artifact(final_path),
            "source_manifest": original.artifact(manifest_path), "input_manifest": input_asset, "photograph": photo_asset,
            "floor_flags": final["floor_contact_fill"], "top_flags": final["top_surface_rule_masks"],
            "approved_area_map": final["rig_domain_revision"]["area_map"], "base_labels": final["base_labels"]},
        "bottom_mass_allowed": False, "scientific_assets_changed": False, "labels_changed": False,
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
    }


def make_montage(output, rows):
    cellw, cellh, columns = 700, 480, 6
    sheet = Image.new("RGB", (columns * cellw, 100 + ((len(rows) + columns - 1) // columns) * cellh), "#181b20")
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 12), "P1-final section 4d supplement: maximum COMBINED A+B removal per run, approved measured cm2", fill="white")
    draw.text((15, 36), "Cyan: actual floor-fill removal (A). Yellow: top-surface component removal (B). Photo lift x4.3.", fill="white")
    draw.text((15, 60), "Display outlines only; no maps/domain/labels changed. The bottom strip remains blocked for mass.", fill="#c2cedb")
    for n, row in enumerate(rows):
        image = Image.open(row["image"]["path"]).convert("RGB")
        image.thumbnail((cellw - 10, cellh - 25), Image.Resampling.LANCZOS)
        x, y = (n % columns) * cellw, 100 + (n // columns) * cellh
        sheet.paste(image, (x + 5, y + 5))
    path = Path(output) / "maximum_combined_removed_all_runs.png"
    sheet.save(path)
    return original.artifact(path)


def run(parent, candidate, output, workers=4):
    started, cpu_started = time.perf_counter(), time.process_time()
    if workers < 1 or workers > 4:
        raise ValueError("This supplement is limited to four readers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seal_path = Path(candidate) / "rig_camera_candidate_receipt.json"
    seal = original.read(seal_path)
    qa_path = Path(candidate) / "production_qa" / "p1_final_qa.json"
    qa = original.read(qa_path)
    if seal.get("complete") is not True or seal.get("contract") != "ffac-g3-t4c-p1-final-candidate-v1" or qa.get("technical_all_pass") is not True:
        raise ValueError("The registered candidate is not sealed and technically complete")
    if qa["run_count"] != 43 or qa["frame_count"] != 864 or len(qa["rows"]) != 864:
        raise ValueError("Expected the exact 43-run / 864-frame final QA")
    grouped = {}
    for row in qa["rows"]:
        grouped.setdefault(row["run"].casefold(), []).append(row)
    selected = [select_maximum(grouped[name]) for name in sorted(grouped)]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=original._init_worker) as pool:
        pending = [pool.submit(render_run, parent, candidate, row, str(output)) for row in selected]
        for future in as_completed(pending):
            row = future.result()
            results.append(row)
            print(f"{row['run']} frame {row['frame_index']}: combined {row['combined_removed_cm2']:.3f} cm2", flush=True)
    results.sort(key=lambda row: row["run"])
    montage = make_montage(output, results)
    csv_path = output / "maximum_combined_removal_by_run.csv"
    fields = ("run", "frame_index", "elapsed_h", "source_sha256", *AREA_FIELDS, "combined_removed_cm2", "removed_floor_pixels", "removed_top_pixels")
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)
    report = {
        "contract": CONTRACT, "complete": True, "run_count": len(results), "selection_frame_count": 864,
        "spec_sha256": original.SPEC_SHA256,
        "interpretation": "Additional literal section 4d interpretation: largest total removal from A plus B, not the existing top-only maximum. Existing reports and figures are preserved.",
        "selection": "Maximum measured floor + top aqueous + top gas cm2 in each run; earliest frame index breaks ties",
        "outline": {"cyan": "Actual changed pixels from floor-contact fill restoration", "yellow": "CO2 pixels removed by top-surface rule", "display_only_dilation_px": 7},
        "candidate_seal": original.artifact(seal_path), "source_qa": original.artifact(qa_path),
        "report_code": original.artifact(__file__), "shared_read_only_helpers": original.artifact(original.__file__),
        "per_run": results, "montage": montage, "selection_csv": original.artifact(csv_path),
        "bottom_measurement_limit": "Existing section 2 full-band areas remain unknown; this supplement does not change or extend their measured support",
        "scientific_assets_changed": False, "labels_changed": False, "bottom_mass_allowed": False,
        "cost": {"wall_seconds": time.perf_counter() - started,
            "cpu_seconds": sum(row["cpu_seconds"] for row in results) + time.process_time() - cpu_started,
            "gpu_seconds": 0., "backend": "cpu-report-only", "readers": workers},
    }
    original.write_new(output / "combined_removal_supplement.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    run(args.parent, args.candidate, args.output, args.workers)


if __name__ == "__main__":
    main()
