"""Section 3 tables and floor comparisons from independently audited saved maps.

No scientific pixel transformation, inference, training or label/scorer access.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from ..phase_rig_display_prerender import pointer_path
from . import qa_ffac_p1_final3_saved as audit
from .refit_ffac_g3_downstream import utc_now

CONTRACT = "ffac-p1-final3-section3-report-v1"
FIGURES = (("ac44", 4), ("ac45", 5), ("ac56", 11), ("ac58", 14))
TARGETS_H = (1., 2.25, 4., 12., 24.)


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def new_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nearest_frame(rows, target_h):
    audit.require(bool(rows), "No exact camera-time frames")
    return min(rows, key=lambda row:(abs(float(row["elapsed_h"])-target_h), int(row["frame_index"])))


def tables(rows):
    """Aggregate complete per-frame independent measurements without rescoring."""
    groups = defaultdict(list)
    per_frame = []
    keys = ("run", "frame_index", "elapsed_h", "source_sha256", "light_epoch_id",
        "reverted_pixels", "reverted_nominal_cm2", "reverted_measured_cm2",
        "retained_reason_a_pixels", "retained_reason_a_nominal_cm2", "retained_reason_a_measured_cm2")
    for row in sorted(rows, key=lambda r:(r["run"], r["frame_index"])):
        item = {key:row[key] for key in keys}
        item["run"] = item["run"].upper()
        per_frame.append(item)
        groups[row["run"]].append(item)
    per_run = []
    sums = ("reverted_pixels", "reverted_nominal_cm2", "reverted_measured_cm2",
        "retained_reason_a_pixels", "retained_reason_a_nominal_cm2", "retained_reason_a_measured_cm2")
    for run, frames in sorted(groups.items()):
        per_run.append(dict(run=run.upper(), frames=len(frames),
            frames_with_reason_b=sum(r["reverted_pixels"] > 0 for r in frames),
            **{key:sum(r[key] for r in frames) for key in sums}))
    medians, selection = [], []
    for target in TARGETS_H:
        chosen = [nearest_frame(frames, target) for frames in groups.values()]
        medians.append(dict(target_h=target, run_count=len(chosen),
            median_reverted_pixels=float(np.median([r["reverted_pixels"] for r in chosen])),
            median_reverted_nominal_cm2=float(np.median([r["reverted_nominal_cm2"] for r in chosen])),
            median_reverted_measured_cm2=float(np.median([r["reverted_measured_cm2"] for r in chosen])),
            min_selected_elapsed_h=min(r["elapsed_h"] for r in chosen),
            max_selected_elapsed_h=max(r["elapsed_h"] for r in chosen),
            selection="closest exact camera-time frame; earliest frame index resolves ties"))
        selection.extend(dict(target_h=target, **row) for row in chosen)
    return per_frame, per_run, medians, selection


def parent_photo(parent, run, index):
    pointer = audit.read(pointer_path(parent))
    manifest = json.loads(audit.asset_bytes(pointer["manifest"]).decode("utf-8-sig"))
    local = json.loads(audit.asset_bytes(manifest["runs"][run]).decode("utf-8-sig"))
    item = local["records"][f"frame_{index:03d}"]["raw_detail"]
    data = audit.asset_bytes(item["display"])
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.array(image.convert("RGB"))
    return rgb, item["display"]


def phase_overlay(photo, labels):
    """Identical display blending for both panels, never a scientific output."""
    result = photo.copy()
    for category, colour in ((1, (50, 205, 125)), (2, (231, 100, 153))):
        mask = labels == category
        result[mask] = np.rint(.5*photo[mask] + .5*np.asarray(colour)).astype(np.uint8)
    return result


def render_pair(parent, candidate, run, index, output):
    parent_path = Path(parent)/"rig_camera_frames"/run/f"frame_{index:03d}.json"
    final_path = Path(candidate)/"rig_camera_frames"/run/f"frame_{index:03d}.json"
    old, new = audit.read(parent_path), audit.read(final_path)
    audit.require(new["binding"]["parent_receipt"] == artifact(parent_path)
        and new["binding"]["source"] == old["binding"]["source"], "Figure has another exact parent source")
    before = audit.labels(old["labels_map"])
    after = audit.labels(new["labels_map"], before.shape)
    rgb, photo = parent_photo(parent, run, index)
    audit.require(rgb.shape[:2] == before.shape, "Reused photograph grid differs")
    source = new["binding"]["source"]
    removed = audit.packed(new["contact_fill_b_reverted_masks"], source, before.shape,
                           contract=audit.PRODUCTION_CONTRACT)
    with np.load(io.BytesIO(audit.asset_bytes(new["rig_domain_revision"]["classes"])), allow_pickle=False) as z:
        domain = z["rig_domain"].copy()
    occupied = domain.any(axis=0)
    audit.require(occupied.any(), "Figure has no physical domain")
    floor = domain.shape[0] - 1 - domain[::-1].argmax(axis=0)
    floor_y = int(np.median(floor[occupied]))
    # Display crop only: keep the entire photographed floor width. A separate
    # 1:1 close-up centres on the removed pixels; neither crop enters any rule.
    x0, x1 = int(np.flatnonzero(occupied)[0]), int(np.flatnonzero(occupied)[-1]) + 1
    y0, y1 = max(0, floor_y-300), min(before.shape[0], floor_y+60)
    cy, cx = np.nonzero(removed)
    centre = int(np.median(cx)) if len(cx) else (x0+x1)//2
    crop = [max(x0, centre-1000), y0, min(x1, centre+1000), y1]
    x0, y0, x1, y1 = crop
    panels = [Image.fromarray(phase_overlay(rgb, value)[y0:y1, x0:x1]) for value in (before, after)]
    width, height = panels[0].size
    sheet = Image.new("RGB", (2*width, height+86), "#171a20")
    draw = ImageDraw.Draw(sheet)
    for number, (panel, title) in enumerate(zip(panels, ("P1 final 2", "P1 final 3: reason (b) removed"))):
        draw.text((number*width+12, 10), f"{run.upper()} frame_{index:03d} | camera t={new['elapsed_h']:.6f} h | {title}", fill="white")
        sheet.paste(panel, (number*width, 86))
    draw.text((12, 34), "Same saved RAW-detail photograph and 50% phase opacity; green aqueous, pink gas.", fill="#c7d4df")
    draw.text((12, 56), "No photographic correction or inference; floor strip remains blocked for mass.", fill="#c7d4df")
    path = Path(output)/f"{run.upper()}_frame_{index:03d}_floor_final2_final3.png"
    audit.require(not path.exists(), "Figure already exists")
    sheet.save(path)
    preview = sheet.copy()
    preview.thumbnail((2400, 700), Image.Resampling.LANCZOS)
    preview_path = path.with_name(path.stem+"_preview.png")
    audit.require(not preview_path.exists(), "Figure preview already exists")
    preview.save(preview_path)
    return dict(run=run.upper(), frame_index=index, elapsed_h=new["elapsed_h"],
        source_sha256=new["source_sha256"], figure=artifact(path), preview=artifact(preview_path),
        source_photograph=photo, parent_receipt=artifact(parent_path), final_receipt=artifact(final_path),
        display_crop_xyxy=crop, reverted_pixels=int(removed.sum()),
        crop_selection="display-only 2000 px close-up centred on reason-b pixels; floor median -300/+60 px",
        index_convention="zero-based frame_NNN, unchanged source/camera binding")


def run_report(parent, candidate, audit_output, output):
    started, cpu = time.perf_counter(), time.process_time()
    parent, candidate, audit_output, output = map(lambda p:Path(p).resolve(), (parent, candidate, audit_output, output))
    audit.require(not output.is_relative_to(candidate) and not output.is_relative_to(parent),
                  "Report must be outside both candidates")
    summary_path = audit_output/"final_identity_summary.json"
    summary = audit.read(summary_path)
    seal = audit.read(candidate/"rig_camera_candidate_receipt.json")
    plan = audit.read(candidate/"p1_final3_plan.json")
    audit.require(summary["contract"] == audit.CONTRACT and summary["complete"] is True
        and summary["all_pass"] is True and summary["audited_frames"] == 864 and summary["audited_runs"] == 43,
        "Report requires complete independent saved QA")
    audit.require(summary["binding"]["candidate"] == str(candidate)
        and summary["binding"]["pipeline_sha256"] == seal["pipeline_sha256"] == plan["pipeline_sha256"]
        and summary["binding"]["source_plan_sha256"] == artifact(candidate/"p1_final3_plan.json")["sha256"]
        and summary["binding"]["spec_sha256"] == audit.SPEC_SHA256, "Audit binding differs")
    audit.require(seal["contract"] == audit.PRODUCTION_CONTRACT and seal["complete"] is True
        and (seal["run_count"], seal["frame_count"]) == (43, 864)
        and seal["bottom_mass_allowed"] is False and seal["mass_eligible"] is False,
        "Report candidate is not sealed review-only 43/864")
    audit.require(plan["binding"]["inputs"]["parent_seal"] == artifact(parent/"rig_camera_candidate_receipt.json"),
                  "Report parent is not the source-bound final2 parent")
    rows = []
    for item in seal["frames"]:
        source = item
        receipt_path = candidate/"rig_camera_frames"/source["run"]/f"frame_{source['frame_index']:03d}.json"
        row = audit.read(audit_output/"frames"/source["run"]/f"frame_{source['frame_index']:03d}.json")
        audit.require(row["contract"] == audit.CONTRACT and row["all_pass"] is True
            and row["pipeline_sha256"] == plan["pipeline_sha256"]
            and row["receipt_sha256"] == item["receipt"]["sha256"] == artifact(receipt_path)["sha256"],
            "Report frame audit or sealed receipt changed")
        rows.append(row)
    audit.require(len(rows) == 864 and len({(r["run"], r["frame_index"]) for r in rows}) == 864,
                  "Report inventory is not unique 864")
    output.mkdir(parents=True, exist_ok=False)
    per_frame, per_run, medians, selection = tables(rows)
    paths = dict(per_frame=output/"contact_fill_b_removed_per_frame.csv",
        per_run=output/"contact_fill_b_removed_per_run.csv",
        target_medians=output/"contact_fill_b_removed_target_medians.csv",
        exact_time_selection=output/"contact_fill_b_removed_exact_time_selection.csv")
    for key, values in (("per_frame", per_frame), ("per_run", per_run), ("target_medians", medians), ("exact_time_selection", selection)):
        new_csv(paths[key], values)
    figures = [render_pair(parent, candidate, run, index, output) for run, index in FIGURES]
    completion_path = candidate/"completion.json"
    completion = audit.read(completion_path) if completion_path.is_file() else None
    identities = {key:summary[key] for key in ("audited_frames", "audited_runs", "reverted_pixels",
        "reverted_nominal_cm2", "reverted_measured_cm2", "retained_reason_a_pixels",
        "retained_reason_a_nominal_cm2", "retained_reason_a_measured_cm2", "gas_bit_identical_frames",
        "domain_category_bit_identical_frames", "net_changed_pixels_exactly_one_step_frames",
        "overlapping_final_changed_pixels")}
    document = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), spec_sha256=audit.SPEC_SHA256,
        candidate=artifact(candidate/"rig_camera_candidate_receipt.json"),
        parent=artifact(parent/"rig_camera_candidate_receipt.json"), independent_qa=artifact(summary_path),
        identities=identities, nominal_pixel_area_mm2=audit.NOMINAL_AREA_MM2,
        measured_area="unchanged per-pixel area maps, approved 900 mm rig width; Olav measurement 2026-10-05",
        tables={key:artifact(path) for key, path in paths.items()}, target_medians=medians,
        figures=figures, figure_note="Explicit frame_NNN indices used; AC44 frame_004 has camera time 0.997289722 h, despite spec rationale calling it 0.75 h.",
        regression_export=dict(layout="production_labels/<run>/frame_NNN.png", verified_frames=864,
            independent_sealed_scorer_run=False),
        cost=dict(production_completion=completion, independent_audit_wall_seconds=summary["audit_wall_seconds"],
            independent_audit_cpu_seconds=summary["audit_cpu_seconds"], report_wall_seconds=time.perf_counter()-started,
            report_cpu_seconds=time.process_time()-cpu, backend="cpu-read-only"),
        photographic_assets_reused=True, training=False, inference=False, annotations_read=False,
        labels_changed=False, bottom_mass_allowed=False, mass_eligible=False, default_changed=False,
        claude_independent_rule_and_floor_box_checks_pending=True, scientific_acceptance=False)
    new_json(output/"section3_report.json", document)
    lines = ["# FF_AC P1 final 3 — section 3 QA", "",
        "All 43 runs / 864 frames passed independent saved-file identities. No training or inference.", "",
        f"- Reason (b) reverted: **{summary['reverted_pixels']:,} pixels**, **{summary['reverted_nominal_cm2']:.6f} cm² nominal**, **{summary['reverted_measured_cm2']:.6f} cm² measured**, summed over frames.",
        f"- Retained reason (a): {summary['retained_reason_a_pixels']:,} pixels; {summary['retained_reason_a_nominal_cm2']:.6f} cm² nominal / {summary['retained_reason_a_measured_cm2']:.6f} cm² measured.",
        "- Every changed pixel is exactly a saved reason (b) pixel, aqueous 1 → 0. All other labels, gas and domain categories are bit-identical to P1 final 2.",
        "- Every net change from P1-final lies in exactly one A/B/C(a)/D mask. Masks A/B/D, domain and area descriptors, source/time/epoch binding, scientific images, heads and δ are unchanged.",
        "- Regression export: production_labels/<run>/frame_NNN.png, all 864 frames verified.", "",
        "| Target h | Runs | Median reverted px | Nominal cm² | Measured cm² |",
        "|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['target_h']:g} | {r['run_count']} | {r['median_reverted_pixels']:g} | {r['median_reverted_nominal_cm2']:.6f} | {r['median_reverted_measured_cm2']:.6f} |" for r in medians)
    lines += ["", "Selection uses each run's closest exact camera time; earliest frame index resolves ties. Exact selected sources/times are in the selection CSV.", "",
        "## Floor comparisons", "", "The same saved RAW-detail photographs and phase opacity are used in both panels. Explicit zero-based frame indices were retained; the AC44 frame_004 caption reports its actual camera time (~0.997 h), rather than the rationale's 0.75 h.", ""]
    for figure in figures:
        lines += [f"![{figure['run']} frame {figure['frame_index']} floor comparison]({figure['preview']['path']})", ""]
    lines += ["## Cost and scope", "", f"Independent byte audit: {summary['audit_wall_seconds']:.1f} s wall / {summary['audit_cpu_seconds']:.1f} s process CPU; {summary['workers']} readers. Production and report costs are recorded in section3_report.json. Backend: CPU; no GPU required.", "",
        "P1 final 2 is untouched. This is a new non-default Studio review candidate; the bottom strip remains blocked for mass. Claude's independent recomputation and floor-box check remain pending, and scientific acceptance is false.", ""]
    with (output/"README.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_report(args.parent, args.candidate, args.audit_output, args.output)), flush=True)


if __name__ == "__main__":
    main()
