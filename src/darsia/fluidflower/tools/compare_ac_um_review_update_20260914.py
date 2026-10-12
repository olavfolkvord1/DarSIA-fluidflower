"""Read saved AC/UM predictions and write evidence diagnostics and photo crops.

No training, inference, model edits, annotation edits, or contour cleanup occurs.
All evidence scores are on current effective training regions, not a holdout.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np


from darsia.fluidflower.phase_evidence_audit import disagreement_reasons, verified_review_binding
from darsia.fluidflower.phase_runtime import load_physics_profile
from darsia.fluidflower.phase_training.logistic import regional_phase_dominance
from darsia.fluidflower.tools.build_phase_disagreement_review import native_domain
from darsia.fluidflower.tools.phase_coupled_constraint_comparison import annotation_window, assessment, _evidence_key
from darsia.fluidflower.tools.phase_training_geometry_revision import _native, read, sha
from darsia.fluidflower.tools.phase_training_review_comparison import (
    _completed, _figure, _photo, _regions, decode_phase, fragmentation, fractions,
    mil_fraction_reference, phase_path, write,
)

DEFAULT_ROOT = local_cache_root() / 'phase_diagnostics/ac_um_review_update_20260914'
FORMAT = "ac-um-current-training-evidence-saved-map-comparison-v1"


def checked(path, hashes, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Bound diagnostic source changed: {path}")
    previous = hashes.get(str(path))
    if previous is not None and previous != digest:
        raise ValueError(f"Diagnostic source changed during comparison: {path}")
    hashes[str(path)] = digest
    return digest


def verify_training_request(state, model, hashes):
    """The standard coupled finish legitimately adds training-only settings."""
    path = Path(model) / "training_request.json"
    actual = checked(path, hashes)
    expected = state.get("request_sha256")
    if expected is None or expected == actual:
        return dict(path=str(path), sha256=actual, preparation_sha256=expected)
    from darsia.fluidflower.tools.train_coupled_aqueous_revision import SEED
    seed_path = Path(model) / SEED / "training_request.json"
    checked(seed_path, hashes, expected)
    old, new = read(seed_path), read(path)
    if not new.get("settings", {}).get("coupled_aqueous_refinement"):
        raise ValueError("Training request changed without standard coupled refinement")
    allowed = {"coupled_aqueous_refinement", "coupled_max_iters", "aqueous_absence_strength",
               "evaluation_mode", "generalization_validated", "deployment_allowed", "scientific_acceptance"}
    old = {**old, "settings": {k: v for k, v in old["settings"].items() if k not in allowed}}
    new = {**new, "settings": {k: v for k, v in new["settings"].items() if k not in allowed}}
    if old != new:
        raise ValueError("Training request changed beyond standard coupled settings")
    return dict(path=str(path), sha256=actual, preparation_sha256=expected,
                preparation_snapshot=str(seed_path), allowed_change="standard coupled refinement settings only")


def completed_jobs(model, hashes):
    """Validate immutable model/physical identity before accepting saved maps."""
    binding = verified_review_binding(model)
    manifest_path = Path(model) / "validation/postflight_manifest.json"
    manifest = read(manifest_path)
    for path in (manifest_path, manifest_path.with_name("validation_binding.json")):
        checked(path, hashes)
    jobs = {}
    for job in manifest["jobs"]:
        run = str(job["run"]).lower()
        if (run in jobs or binding["selected_candidates"].get(run) != job.get("candidate_id")
                or job.get("optical_model_sha256") != binding["model_sha256"]
                or job.get("physics_runtime") != binding["physics_runtime"]):
            raise ValueError(f"Saved map job differs from its selected model: {run}")
        jobs[run] = job
    return jobs, binding


def parent_jobs(parent, model, hashes):
    """Extend parent maps only from exact collections frozen for this fit.

    AC60's old maps were evaluated using the parent's optical model and its own
    prepared physical context. The source collection records that distinction.
    Never choose a collection merely because it is now the latest in an index.
    """
    from darsia.fluidflower.tools.phase_review_collections import validated_collection
    jobs, binding = completed_jobs(parent, hashes)
    provenance = {run: dict(kind="parent_validation", candidate_id=job["candidate_id"])
                  for run, job in jobs.items()}
    prep_path = Path(model) / "continuation_preparation.json"
    checked(prep_path, hashes)
    for index, source in enumerate(read(prep_path)["captured_sources"]):
        if source["kind"] != "collection_review":
            continue
        snapshot = Path(model) / "continuation_evidence/originals" / str(index) / "manifest.json"
        checked(snapshot, hashes, source["manifest_sha256"])
        manifest = read(snapshot)
        if manifest.get("source_model_id") != binding["model_id"]:
            continue
        if manifest.get("source_model_sha256") != binding["model_sha256"]:
            raise ValueError("Captured collection names different parent model weights")
        live = Path(source["combined_manifest"])
        checked(live, hashes, source["manifest_sha256"])
        validated_collection(live, parent, full=False)
        receipt = Path(manifest["collection_receipt"])
        checked(receipt, hashes, manifest["collection_receipt_sha256"])
        for row in read(receipt)["sources"]:
            for role in ("manifest", "binding", "plan", "materialization"):
                if role in row:
                    checked(row[role]["path"], hashes, row[role]["sha256"])
            for job in row["jobs"]:
                run = str(job["run"]).lower()
                if run in jobs:
                    if jobs[run] != job:
                        raise ValueError(f"Conflicting exact parent prediction bindings: {run}")
                    continue
                if row["kind"] != "extra":
                    raise ValueError("Unexpected non-extra collection prediction")
                materialization = read(row["materialization"]["path"])
                expected = materialization["generated_sha256"]
                validation = Path(row["materialization"]["path"]).parent
                jobs[run] = job
                provenance[run] = dict(kind="parent_optical_model_prepared_target", candidate_id=job["candidate_id"],
                    collection_manifest=str(live), collection_receipt=str(receipt),
                    source_model_id=binding["model_id"], source_model_sha256=binding["model_sha256"],
                    target_physics_runtime=job["physics_runtime"], validation_root=str(validation),
                    generated_sha256=expected)
    return jobs, binding, provenance


def matching_frame(old_native, new_native, index):
    """Match the source image/time explicitly; equal frame indices are insufficient."""
    for key in ("canvas_wh", "content_box_xywh", "geometry_revision_id", "correction_contract"):
        if old_native.get(key) != new_native.get(key):
            raise ValueError(f"Saved prediction geometry/photographic correction differs: {key}")
    current = new_native["frame_sources"][index]
    matches = [i for i, frame in enumerate(old_native["frame_sources"])
               if frame.get("source_name") == current.get("source_name")
               and abs(float(frame["elapsed_h"])-float(current["elapsed_h"])) <= 1e-7]
    if len(matches) != 1:
        raise ValueError("No unique parent map for the same source image and time")
    return matches[0]


def retention(old, new, domain):
    """Same-pixel retention is descriptive, not a claim those pixels were correct."""
    result = {}
    for phase, ids in (("gas", (2, 3)), ("aqueous", (1, 3)), ("co2", (1, 2, 3))):
        before, after = np.isin(old, ids) & domain, np.isin(new, ids) & domain
        n_old, n_new = int(before.sum()), int(after.sum())
        retained = int(np.count_nonzero(before & after))
        result[phase] = dict(old_pixels=n_old, new_pixels=n_new, retained_pixels=retained,
            removed_pixels=n_old-retained, added_pixels=n_new-retained,
            retained_old_fraction=retained/n_old if n_old else None,
            new_to_old_area_ratio=n_new/n_old if n_old else None)
    return result


def constraints(metrics, region):
    reasons = disagreement_reasons(dict(**{"class": region.phase_class},
        co2_fraction=metrics["co2_fraction"], old_gas_fraction=metrics["gas_evidence_fraction"],
        new_gas_fraction=metrics["gas_evidence_fraction"], aqueous_fraction=metrics["aqueous_evidence_fraction"],
        dominant_phase=regional_phase_dominance(region)), threshold=0.)
    return dict(satisfied=not reasons, reasons=[dict(reason=r, score=s, text=t) for r, s, t in reasons])


def aggregate(rows):
    result = dict(regions=len(rows), old_constraints_satisfied=0, new_constraints_satisfied=0,
                  improved_constraints=0, regressed_constraints=0, phases={})
    for side in ("old", "new"):
        result[side+"_constraints_satisfied"] = sum(row[side+"_constraint"]["satisfied"] for row in rows)
    result["improved_constraints"] = sum(not r["old_constraint"]["satisfied"] and r["new_constraint"]["satisfied"] for r in rows)
    result["regressed_constraints"] = sum(r["old_constraint"]["satisfied"] and not r["new_constraint"]["satisfied"] for r in rows)
    for phase in ("gas", "aqueous"):
        before = np.asarray([row["old"][phase+"_evidence_fraction"] for row in rows])
        after = np.asarray([row["new"][phase+"_evidence_fraction"] for row in rows])
        result["phases"][phase] = dict(old_mean_fraction=float(before.mean()) if len(rows) else None,
            new_mean_fraction=float(after.mean()) if len(rows) else None,
            old_any_regions=int((before > 0).sum()), new_any_regions=int((after > 0).sum()),
            old_zero_regions=int((before == 0).sum()), new_zero_regions=int((after == 0).sum()),
            lost_all_presence_regions=int(((before > 0) & (after == 0)).sum()),
            gained_presence_regions=int(((before == 0) & (after > 0)).sum()),
            decreased_coverage_regions=int((after < before).sum()), increased_coverage_regions=int((after > before).sum()))
    return result


def render_report(path, result):
    e = lambda value: html.escape(str(value), quote=True)
    sections = ["<!doctype html><meta charset='utf-8'><title>AC/UM phase-map comparison</title>",
        "<style>body{font:16px system-ui;max-width:1300px;margin:36px auto;padding:0 20px;color:#17242c;background:#f5f7f8}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd4d8;padding:9px;text-align:left}figure{margin:24px 0;background:white;padding:12px}img{width:100%}.note{padding:16px;background:#fff1cb}code{overflow-wrap:anywhere}</style>",
        f"<h1>{e(result['series'].upper())} · saved phase-map comparison</h1>",
        "<p class='note'>This compares predictions against evidence used for training. It is not independent validation. Presence labels require a phase somewhere in the region; pure labels supervise the whole area. Missing coverage is not a passing result.</p>",
        f"<p>Parent: <code>{e(result.get('parent_model_id', result['parent']))}</code><br>New: <code>{e(result.get('model_id', result['model']))}</code></p>",
        "<ul>"+"".join(f"<li>{e(line)}</li>" for line in result["summary"])+"</ul>"]
    if result.get("groups"):
        sections.append("<h2>Current training constraints</h2><p>Counts are regions, not independent observations. Overlapping boxes remain separate. Strict categorical agreement uses the recorded label and dominance constraints; it does not reconstruct the probability loss.</p><table><tr><th>Label</th><th>Compared</th><th>Old satisfied</th><th>New satisfied</th><th>Old → new mean gas</th><th>Old → new mean aqueous</th></tr>")
        for label, group in result["groups"].items():
            phase = group["phases"]
            sections.append(f"<tr><td>{e(label)}</td><td>{group['regions']}</td><td>{group['old_constraints_satisfied']}</td><td>{group['new_constraints_satisfied']}</td><td>{100*phase['gas']['old_mean_fraction']:.2f}% → {100*phase['gas']['new_mean_fraction']:.2f}%</td><td>{100*phase['aqueous']['old_mean_fraction']:.2f}% → {100*phase['aqueous']['new_mean_fraction']:.2f}%</td></tr>")
        sections.append("</table>")
    if result.get("late_frames"):
        sections.append("<h2>Late frames</h2><p>Components, boundary pixels and same-pixel retention describe the saved maps. Real fingers and separate plumes also affect fragmentation; lower values alone do not establish better predictions.</p><table><tr><th>Run / time</th><th>CO₂ components old → new</th><th>CO₂ boundary pixels old → new</th><th>Old gas retained</th><th>Old aqueous retained</th></tr>")
        for row in result["late_frames"]:
            pct = lambda v: "n/a (no old pixels)" if v is None else f"{100*v:.2f}%"
            sections.append(f"<tr><td>{e(row['run'])} / {row['elapsed_h']:.2f} h</td><td>{row['old_fragmentation']['co2_components']} → {row['new_fragmentation']['co2_components']}</td><td>{row['old_fragmentation']['co2_boundary_pixels']} → {row['new_fragmentation']['co2_boundary_pixels']}</td><td>{pct(row['retention']['gas']['retained_old_fraction'])}</td><td>{pct(row['retention']['aqueous']['retained_old_fraction'])}</td></tr>")
        sections.append("</table>")
    sections.append("<h2>Photographs and actual saved predictions</h2><p>Maps use their recorded categorical colors over the same photograph. No contour cleanup or photographic repaint is applied. Crops include largest regressions, improvements, residual errors and gas/aqueous anchor losses.</p>")
    for fig in result["figures"]:
        relative = Path(fig["path"]).relative_to(path.parent).as_posix()
        sections.append(f"<figure><a href='{e(relative)}'><img loading='lazy' src='{e(relative)}' alt='{e(fig['caption'])}'></a><figcaption>{e(fig['caption'])}</figcaption></figure>")
    sections.append("<p><a href='saved_map_comparison.json'>Full region metrics, coverage and source fingerprints</a></p>")
    path.write_text("\n".join(sections), encoding="utf-8")


def compare_series(campaign_root, series, *, model=None, figures=True):
    campaign_root = Path(campaign_root).resolve()
    state_path = campaign_root / series / "state.json"
    state = read(state_path)
    parent, model = Path(state["parent"]).resolve(), Path(model or state["output_root"]).resolve()
    outdir = campaign_root / series
    if any(outdir == protected or outdir.is_relative_to(protected) for protected in (parent, model)):
        raise ValueError("Diagnostic output must stay outside model bundles")
    result = dict(format=FORMAT, series=series, status="pending", parent=str(parent), model=str(model),
        independent_validation=False, source_files_modified=False, figures=[], regions=[], late_frames=[],
        unavailable_regions=[], unavailable_late_frames=[], summary=[],
        scope="All effective current frozen training regions, including saved disagreement revisions; both maps assessed against the current human label.")
    output = outdir / "saved_map_comparison.json"
    if not _completed(model):
        result["summary"] = ["Waiting for completed native phase maps; no comparison has run."]
        write(output, result)
        render_report(outdir / "comparison_report.html", result)
        return result
    hashes = {}
    checked(state_path, hashes)
    result["training_request_binding"] = verify_training_request(state, model, hashes)
    old_jobs, old_binding, old_provenance = parent_jobs(parent, model, hashes)
    new_jobs, new_binding = completed_jobs(model, hashes)
    if state.get("model_id") != new_binding["model_id"]:
        raise ValueError("Campaign state and selected new saved model differ")
    result.update(parent_model_id=old_binding["model_id"], model_id=new_binding["model_id"])
    old_regions, _ = _regions(parent)
    regions, request = _regions(model)
    if any(region.dataset != series for region in regions):
        raise ValueError("Current evidence includes another experimental series")
    for root in (parent, model):
        checked(root / "training_request.json", hashes)
        for bundle in read(root / "training_request.json")["bundles"]:
            for role in ("decisions", "combined_manifest"):
                checked(bundle[role]["snapshot_path"], hashes, bundle[role]["sha256"])
    previous = {r.region_id: r for r in old_regions}
    changed_ids = {r.region_id for r in regions if r.region_id not in previous or _evidence_key(r) != _evidence_key(previous[r.region_id])}
    revised_ids = {r.region_id for r in regions if r.region_id in changed_ids and r.payload.get("evidence_audit")}
    positive_fraction, fraction_source = mil_fraction_reference(model)
    result["mil_coverage_reference"] = dict(fraction=positive_fraction, source=fraction_source,
        probability_loss_evaluated=False, interpretation="Descriptive categorical coverage only; regional presence imposes no minimum phase-area fraction.")
    result["selected_evidence"] = dict(total=len(regions), new_or_changed=len(changed_ids), new_disagreement_revisions=len(revised_ids),
        classes=dict(Counter(r.phase_class for r in regions)))
    grouped = defaultdict(list)
    for region in regions:
        grouped[(region.run, region.frame_index)].append(region)
    profile = load_physics_profile(model)
    metadata = {}
    def skip(region, reason, **details):
        result["unavailable_regions"].append(dict(region_id=region.region_id, run=region.run,
            frame_index=region.frame_index, phase_class=region.phase_class, new_or_changed=region.region_id in changed_ids,
            reason=reason, **details))
    for run in sorted({r.run for r in regions} | set(new_jobs)):
        if run not in old_jobs or run not in new_jobs:
            for region in regions:
                if region.run == run:
                    skip(region, "run_has_no_paired_saved_predictions")
            continue
        old_job, new_job = old_jobs[run], new_jobs[run]
        old_root, old_path = _native(old_job)
        new_root, new_path = _native(new_job)
        old_native, new_native = read(old_path), read(new_path)
        for path in (old_path, new_path):
            checked(path, hashes)
        domain = np.asarray(native_domain(model, profile, new_job, new_native, new_path, hashes), dtype=bool)
        late = set(range(max(0, int(new_native["frame_count"])-3), int(new_native["frame_count"])))
        wanted = sorted(late | {index for r, index in grouped if r == run})
        metadata[run] = (old_job, new_job, old_root, new_root, old_native, new_native)
        for index in wanted:
            members = grouped[(run, index)]
            try:
                old_index = matching_frame(old_native, new_native, index)
            except (ValueError, IndexError) as exc:
                for region in members:
                    skip(region, "no_exact_paired_frame", detail=str(exc))
                if index in late:
                    result["unavailable_late_frames"].append(dict(run=run, frame_index=index, reason=str(exc)))
                continue
            old_map, new_map = phase_path(old_job, old_index), phase_path(new_job, index)
            missing = [str(p) for p in (old_map, new_map) if not p.is_file()]
            if missing:
                for region in members:
                    skip(region, "saved_map_missing", paths=missing)
                if index in late:
                    result["unavailable_late_frames"].append(dict(run=run, frame_index=index, reason="saved_map_missing", paths=missing))
                continue
            provenance = old_provenance[run]
            if provenance["kind"] == "parent_optical_model_prepared_target":
                relative = old_map.resolve().relative_to(Path(provenance["validation_root"]).resolve()).as_posix()
                expected = provenance["generated_sha256"].get(relative)
                if expected is None:
                    # Older Windows receipts sometimes record native separators.
                    expected = provenance["generated_sha256"].get(relative.replace("/", "\\"))
                if expected is None:
                    raise ValueError("Parent target map is absent from its materialization receipt")
                checked(old_map, hashes, expected)
            old_photo = old_root / "native_frames" / run / f"frame_{old_index:03d}.jpg"
            new_photo = new_root / "native_frames" / run / f"frame_{index:03d}.jpg"
            if checked(old_photo, hashes) != checked(new_photo, hashes):
                raise ValueError(f"Compared photographs differ at {run}/{index}; geometry is not interchangeable")
            print(f"Comparing {series.upper()} {run} frame {index+1}/{new_native['frame_count']} ({len(members)} training regions)", flush=True)
            old_labels = decode_phase(old_map, old_native["content_box_xywh"], hashes)
            new_labels = decode_phase(new_map, new_native["content_box_xywh"], hashes)
            if old_labels.shape != new_labels.shape or domain.shape != new_labels.shape:
                raise ValueError("Paired predictions and physical domain have different shapes")
            frame = dict(run=run, frame_index=index, parent_frame_index=old_index,
                elapsed_h=float(new_native["frame_sources"][index]["elapsed_h"]), old_map=str(old_map), new_map=str(new_map),
                parent_candidate_id=old_job["candidate_id"], new_candidate_id=new_job["candidate_id"])
            for region in members:
                if abs(region.elapsed_h-frame["elapsed_h"]) > 1e-5:
                    skip(region, "region_time_does_not_match_saved_frame")
                    continue
                bounds, mask = annotation_window(region.payload, canvas_wh=new_native["canvas_wh"], content_box=new_native["content_box_xywh"])
                x0, y0, x1, y1 = bounds
                all_pixels = int(mask.sum())
                mask &= domain[y0:y1, x0:x1]
                if not mask.any():
                    skip(region, "region_has_no_registered_domain_pixels")
                    continue
                before = fractions(old_labels[y0:y1, x0:x1], mask)
                after = fractions(new_labels[y0:y1, x0:x1], mask)
                result["regions"].append(dict(**frame, region_id=region.region_id, phase_class=region.phase_class,
                    new_or_changed=region.region_id in changed_ids, new_disagreement_revision=region.region_id in revised_ids,
                    evidence_audit=region.payload.get("evidence_audit"), old=before, new=after,
                    excluded_outside_domain_pixels=all_pixels-int(mask.sum()),
                    old_constraint=constraints(before, region), new_constraint=constraints(after, region),
                    assessment=assessment(before, after, region.phase_class, positive_fraction)))
            if index in late:
                row = dict(**frame, old=fractions(old_labels, domain), new=fractions(new_labels, domain),
                    old_fragmentation=fragmentation(old_labels, domain), new_fragmentation=fragmentation(new_labels, domain),
                    retention=retention(old_labels, new_labels, domain), phase_fragmentation={})
                for phase, codes in (("gas", (2, 3)), ("aqueous", (1, 3))):
                    row["phase_fragmentation"][phase] = {side: fragmentation(np.isin(labels, codes).astype(np.uint8), domain)
                        for side, labels in (("old", old_labels), ("new", new_labels))}
                if figures:
                    photo = _photo(new_photo, new_native["content_box_xywh"], hashes)
                    path = outdir / "comparison_figures" / f"{run}_late_{index:03d}.png"
                    _figure(path, photo, old_labels, new_labels, title=f"{series.upper()} · {run.upper()} · {frame['elapsed_h']:.2f} t")
                    result["figures"].append(dict(path=str(path), run=run, frame_index=index, kind="late_full_frame",
                        caption=f"{run.upper()} · {frame['elapsed_h']:.2f} h · same photograph and exact saved maps"))
                    row["figure"] = str(path)
                result["late_frames"].append(row)
            del old_labels, new_labels
    measured = {row["region_id"] for row in result["regions"]}
    result["coverage"] = dict(total_effective_regions=len(regions), compared_regions=len(measured),
        complete=len(measured) == len(regions), new_or_changed_total=len(changed_ids),
        new_or_changed_compared=len(measured & changed_ids), new_disagreement_revisions_total=len(revised_ids),
        new_disagreement_revisions_compared=len(measured & revised_ids),
        skipped_reason_counts=dict(Counter(r["reason"] for r in result["unavailable_regions"])),
        unmapped_runs=sorted({r["run"] for r in result["unavailable_regions"] if r["reason"] == "run_has_no_paired_saved_predictions"}))
    result["groups"] = {label: aggregate([r for r in result["regions"] if r["phase_class"] == label])
                        for label in sorted({r["phase_class"] for r in result["regions"]})}
    result["all_evidence"] = aggregate(result["regions"])
    result["new_evidence"] = aggregate([r for r in result["regions"] if r["new_or_changed"]])
    result["reviewed_revisions"] = aggregate([r for r in result["regions"] if r["new_disagreement_revision"]])
    if figures:
        make_crops(result, regions, metadata, hashes, outdir)
    result["parent_map_sources"] = {run: {k: v for k, v in row.items() if k != "generated_sha256"} for run, row in old_provenance.items()}
    cov = result["coverage"]
    result["summary"] = [f"Compared {cov['compared_regions']}/{cov['total_effective_regions']} effective current training regions; {cov['new_or_changed_compared']}/{cov['new_or_changed_total']} new or changed regions, including {cov['new_disagreement_revisions_compared']}/{cov['new_disagreement_revisions_total']} saved disagreement revisions.",
        f"Unmapped training runs: {', '.join(cov['unmapped_runs']) or 'none'}. Skipped regions total: {len(result['unavailable_regions'])}. Coverage complete: {cov['complete']}.",
        f"Strict current categorical constraints satisfied: {result['all_evidence']['old_constraints_satisfied']} → {result['all_evidence']['new_constraints_satisfied']} of {len(measured)} compared regions.",
        f"Among new/changed evidence: {result['new_evidence']['old_constraints_satisfied']} → {result['new_evidence']['new_constraints_satisfied']} of {result['new_evidence']['regions']} regions.",
        "Gas/aqueous retention and late-frame fragmentation are descriptive diagnostics; scores on fitted evidence do not measure independent generalization."]
    result.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=hashes, flags=dict(Counter(flag for r in result["regions"] for flag in r["assessment"]["flags"])))
    write(output, result)
    render_report(outdir / "comparison_report.html", result)
    return result


def make_crops(result, regions, metadata, hashes, outdir):
    lookup = {region.region_id: region for region in regions}
    chosen = {}
    for run in metadata:
        rows = [row for row in result["regions"] if row["run"] == run]
        orderings = [sorted(rows, key=lambda r: (-len(r["new_constraint"]["reasons"]), r["region_id"])),
            sorted(rows, key=lambda r: (-r["assessment"]["label_consistent_change_score"], r["region_id"])),
            sorted(rows, key=lambda r: (r["assessment"]["label_consistent_change_score"], r["region_id"]))]
        for phase in ("gas", "aqueous"):
            anchors = [r for r in rows if r["phase_class"] == phase]
            orderings.append(sorted(anchors, key=lambda r: (r["new"][phase+"_evidence_fraction"]-r["old"][phase+"_evidence_fraction"], r["region_id"])))
        for ordered in orderings:
            for row in ordered[:1]:
                chosen[row["region_id"]] = row
        # Ensure fresh corrections have at least one inspectable crop per run.
        fresh = [r for r in rows if r["new_or_changed"]]
        if fresh:
            row = min(fresh, key=lambda r: (r["assessment"]["label_consistent_change_score"], r["region_id"]))
            chosen[row["region_id"]] = row
    for row in chosen.values():
        run, index = row["run"], row["frame_index"]
        _, _, _, new_root, old_native, native = metadata[run]
        old = decode_phase(row["old_map"], old_native["content_box_xywh"], hashes)
        new = decode_phase(row["new_map"], native["content_box_xywh"], hashes)
        bounds, mask = annotation_window(lookup[row["region_id"]].payload, canvas_wh=native["canvas_wh"], content_box=native["content_box_xywh"])
        x0, y0, x1, y1 = bounds
        margin = max(24, round(max(x1-x0, y1-y0)*.20))
        left, top = max(0, x0-margin), max(0, y0-margin)
        right, bottom = min(new.shape[1], x1+margin), min(new.shape[0], y1+margin)
        outline = np.zeros((bottom-top, right-left), bool)
        outline[y0-top:y1-top, x0-left:x1-left] = mask
        photo = _photo(new_root/"native_frames"/run/f"frame_{index:03d}.jpg", native["content_box_xywh"], hashes)
        path = outdir / "comparison_figures" / f"{run}_region_{row['region_id'].split(':')[-1][:12]}.png"
        _figure(path, photo[top:bottom, left:right], old[top:bottom, left:right], new[top:bottom, left:right], outline=outline,
            title=f"{run.upper()} · {row['elapsed_h']:.2f} t · {row['phase_class']}\nGass: {100*row['old']['gas_evidence_fraction']:.2f}% → {100*row['new']['gas_evidence_fraction']:.2f}% · innløst: {100*row['old']['aqueous_evidence_fraction']:.2f}% → {100*row['new']['aqueous_evidence_fraction']:.2f}%")
        row["figure"] = str(path)
        result["figures"].append(dict(path=str(path), run=run, frame_index=index, region_id=row["region_id"], kind="training_region_crop",
            caption=f"{run.upper()} · {row['phase_class']} · {'new/reviewed' if row['new_or_changed'] else 'inherited'} training evidence; white outline is the human region"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--series", choices=("ffac", "ffum"), nargs="+", default=["ffac", "ffum"])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    for series in args.series:
        result = compare_series(args.root, series, figures=not args.no_figures)
        print(json.dumps(dict(series=series, status=result["status"], summary=result["summary"])), flush=True)


if __name__ == "__main__":
    main()
