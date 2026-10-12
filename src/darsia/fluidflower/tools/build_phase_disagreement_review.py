"""Automatically rank current saved annotations against completed native maps.

This reads saved categorical predictions only. Evidence is frozen as detached
copies; no inference, training, model changes or live annotation locks occur.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np


from darsia.fluidflower.phase_evidence_audit import build_queue, read, verified_review_binding
from darsia.fluidflower.phase_annotation_conflicts import audit_frame_conflicts
from darsia.fluidflower.phase_runtime import load_physics_profile, load_run_context
from darsia.fluidflower.phase_training import trainer as t
from darsia.fluidflower.phase_training.logistic import regional_phase_dominance
from darsia.fluidflower.phase_training.scope import series_id
from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
from darsia.fluidflower.phase_workflow import discover_phase_review_sessions
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools import train_phase_model_further as continuation
from darsia.fluidflower.tools.phase_training_geometry_revision import _native
from darsia.fluidflower.tools.phase_training_review_comparison import decode_phase, phase_path, write

FORMAT = "saved-phase-annotation-disagreement-v1"


def region_histogram(labels, annotation, *, canvas_wh, content_box, domain=None):
    """Exact production polygon rasterization restricted to its bounding ROI."""
    cw, ch = map(float, canvas_wh)
    cx, cy, width, height = map(int, content_box)
    points = annotation.get("points") or []
    if not points:
        x, y, w, h = (float(annotation[k]) for k in ("x", "y", "width", "height"))
        points = ((x, y), (x+w, y), (x+w, y+h), (x, y+h))
    vertices = np.asarray([(float(x)*cw-cx, float(y)*ch-cy) for x, y in points], np.float32)
    if vertices.ndim != 2 or vertices.shape[1:] != (2,) or not np.isfinite(vertices).all():
        raise ValueError("Invalid saved annotation geometry")
    vertices[:, 0] = np.clip(vertices[:, 0], 0, width-1)
    vertices[:, 1] = np.clip(vertices[:, 1], 0, height-1)
    polygon = np.rint(vertices).astype(np.int32)
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)+1
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.fillPoly(mask, [polygon-[x0, y0]], 1)
    selected = mask.astype(bool)
    if labels.shape != (height, width) or (domain is not None and domain.shape != labels.shape):
        raise ValueError("Annotation, map and domain use different native grids")
    if domain is not None:
        selected &= domain[y0:y1, x0:x1]
    counts = np.bincount(labels[y0:y1, x0:x1][selected], minlength=4)
    size = int(counts.sum())
    if not size:
        return None
    return dict(pixels=size, water_pixels=int(counts[0]), aqueous_pixels=int(counts[1]+counts[3]),
        gas_pixels=int(counts[2]+counts[3]), mixed_pixels=int(counts[3]),
        co2_fraction=float((size-counts[0])/size), aqueous_fraction=float((counts[1]+counts[3])/size),
        gas_fraction=float((counts[2]+counts[3])/size))


def native_domain(model, profile, job, native, native_path, hashes):
    """Use the experiment's own registered domain, including composite extras."""
    run, dataset = job["run"], series_id(job["dataset"])
    if profile and f"{dataset}/{run}" in profile["runs"]:
        context, entry = load_run_context(model, profile, dataset, run,
            native_manifest_sha256=sha256_file(native_path))
        path = model / entry["path"]
        hashes[str(path)] = sha256_file(path)
        return context.domain
    record = native.get("domain") or job.get("domain_mask")
    if not isinstance(record, dict) or not record.get("path"):
        raise ValueError("No registered domain is available for this experiment")
    path = Path(record["path"])
    if not path.is_absolute():
        path = native_path.parent / path
    path = resolve_review_path(path)
    digest = sha256_file(path)
    if record.get("sha256") != digest:
        raise ValueError("Registered native domain changed or lacks its binding")
    with np.load(path, allow_pickle=False) as archive:
        domain = np.asarray(archive["domain"], bool)
    if domain.shape != tuple(native["content_box_xywh"][3:1:-1]):
        raise ValueError("Registered domain does not match the native content grid")
    hashes[str(path)] = digest
    return domain


def evaluate_saved_regions(model, source, sources, regions, *, progress=print, region_originals=None):
    """Evaluate each mapped frame once; explicitly inventory every unscored region."""
    jobs = {str(j["run"]).lower(): j for j in _source_jobs(source)}
    originals = {s.run: read(s.asset_manifest_path) for s in sources}
    profile = load_physics_profile(model)
    grouped, natives, domains, hashes, skipped = defaultdict(list), {}, {}, {}, []
    def skip(region, reason, **details):
        skipped.append(dict(region_id=region.region_id, run=region.run, reason=reason, **details))
    for region in regions:
        if region.run not in jobs:
            skip(region, "no_saved_predictions_for_run")
            continue
        if region.run not in natives:
            _, path = _native(jobs[region.run])
            natives[region.run] = (path, read(path))
            hashes[str(path)] = sha256_file(path)
            try:
                domains[region.run] = native_domain(model, profile, jobs[region.run], natives[region.run][1], path, hashes)
            except (OSError, ValueError, KeyError) as exc:
                domains[region.run] = str(exc)
        if isinstance(domains[region.run], str):
            skip(region, "unavailable_registered_domain", detail=domains[region.run])
            continue
        try:
            original = originals[region.run] if region_originals is None else region_originals[region.region_id]
            from darsia.fluidflower.phase_evidence_audit import align_native_region
            align = t._align_region_timeline if region_originals is None else align_native_region
            aligned = align(region, original, natives[region.run][1])
        except (ValueError, KeyError, IndexError) as exc:
            skip(region, "annotation_timeline_not_mapped", detail=str(exc))
            continue
        grouped[(region.run, aligned.frame_index)].append((region, aligned))
    rows, frame_counts = [], Counter()
    strict_conflicts, partial_overlaps = [], []
    for (run, index), members in sorted(grouped.items()):
        native = natives[run][1]
        from darsia.fluidflower.phase_review_basis import optical_path, uses_optical
        path = optical_path(jobs[run], index) if uses_optical(jobs[run]) else phase_path(jobs[run], index)
        if not path.is_file():
            for region, _ in members:
                skip(region, "saved_phase_map_missing", frame=index, path=str(path))
            continue
        progress(f"Checking saved predictions: {run.upper()} frame {index+1}/{native['frame_count']} ({len(members)} annotations)", flush=True)
        labels = decode_phase(path, native["content_box_xywh"], hashes)
        if uses_optical(jobs[run]):
            # Studio's established two-colour display maps legacy amber to
            # aqueous (conditional gas score below 0.6), not to both phases.
            labels[labels == 3] = 1
        frame_counts[run] += 1
        compatibility = audit_frame_conflicts([aligned for _, aligned in members],
            canvas_wh=native["canvas_wh"], content_box=native["content_box_xywh"], domain=domains[run])
        strict_conflicts.extend(dict(run=run, frame=index, **item) for item in compatibility["strict"])
        partial_overlaps.extend(dict(run=run, frame=index, **item) for item in compatibility["partial"])
        per_region_conflicts = defaultdict(list)
        for conflict in compatibility["strict"]:
            for identifier in conflict["region_ids"]:
                per_region_conflicts[identifier].append(conflict)
        for region, aligned in members:
            measured = region_histogram(labels, aligned.payload, canvas_wh=native["canvas_wh"],
                content_box=native["content_box_xywh"], domain=domains[run])
            if measured is None:
                skip(region, "annotation_has_no_registered_domain_pixels", frame=index)
                continue
            rows.append(dict(region_id=region.region_id, run=run, frame=index, elapsed_h=aligned.elapsed_h,
                **{"class": region.phase_class}, dominant_phase=regional_phase_dominance(region), new=measured,
                annotation_conflicts=per_region_conflicts[region.region_id]))
        del labels
    return rows, hashes, dict(total_effective_regions=len(regions), evaluated_regions=len(rows),
        scored_regions=len(rows), missing_maps=len({r["path"] for r in skipped if r["reason"] == "saved_phase_map_missing"}),
        mapped_runs=sorted(frame_counts), evaluated_frames_per_run=dict(frame_counts),
        unmapped_runs=sorted({r["run"] for r in skipped if r["reason"] == "no_saved_predictions_for_run"}),
        skipped_regions=skipped, skipped_reason_counts=dict(Counter(r["reason"] for r in skipped)),
        strict_annotation_conflicts=strict_conflicts, partial_annotation_overlaps=partial_overlaps,
        complete=len(rows) == len(regions))


def build(model, output, *, review_root=None, threshold=.75):
    model, output = Path(model).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output queue; existing reviews remain immutable")
    assets = output.with_name(output.stem + "_assets")
    assets.mkdir(parents=True, exist_ok=False)
    request = read(model / "training_request.json")
    if request.get("evaluation_only") or request.get("settings", {}).get("evaluation_only"):
        raise ValueError("Select the fitted source model, not an evaluation-only transfer result")
    datasets = {series_id(value) for value in read(model / "summary.json")["inventory"]["datasets"]}
    if len(datasets) != 1:
        raise ValueError("Automatic review requires one selected experimental series")
    binding = verified_review_binding(model, review_root)
    source = Path(binding["combined_root"])
    print("Freezing inherited evidence and current saved annotations; live reviews remain editable.", flush=True)
    regional = request.get("artifact_kind") == "optical-region-training-candidate"
    native_options = {"preserve_native_revisions": True} if regional else {}
    _, pairs, regions, audit = continuation.capture_inputs(model, next(iter(datasets)), assets / "evidence_capture",
                                                           discover_phase_review_sessions(), **native_options)
    sources, checked_regions, conflicts = t._load_sources_and_regions([t.ReviewBundle(*pair) for pair in pairs], **native_options)
    write(assets / "capture_audit.json", audit)
    if checked_regions != regions or conflicts != audit.get("conflicts", []):
        raise ValueError("The detached cumulative evidence did not validate")
    evaluation_options = {}
    if regional:
        from darsia.fluidflower.tools.train_phase_optical_regions import groups
        original_groups = groups(dict(audit=audit, pairs=pairs, regions=[asdict(r) for r in regions]))
        evaluation_options["region_originals"] = {record["region_id"]: group["native"]
            for group in original_groups for _, record in group["records"]}
    rows, hashes, coverage = evaluate_saved_regions(model, source, sources, regions, **evaluation_options)
    policy = audit.get("training_label_policy") or {}
    coverage["candidate_issue_only_annotations"] = dict(
        count=policy.get("excluded_candidate_issue_count", 0),
        by_category=policy.get("excluded_candidate_issues_by_category", {}),
        records=policy.get("candidate_issue_evaluation_records", []),
        scored_as_full_region_phase_absence=False,
        evaluation_scope="trigger_candidate_only")
    if regional:
        hashes.update({str(group['native_path']): sha256_file(group['native_path']) for group in original_groups})
        evidence_path = model / 'evidence.json'
        learned = read(model / 'regional_model.json')
        if sha256_file(evidence_path) != learned['evidence_sha256']:
            raise ValueError('Regional training evidence changed')
        hashes[str(evidence_path)] = sha256_file(evidence_path)
        inherited_deferred = read(evidence_path)['audit'].get('deferred_revision_conflicts', [])
        coverage['inherited_deferred_revision_conflicts'] = inherited_deferred
        pending = {f"{r['run']}:manual:{r['id']}" for r in inherited_deferred if r['source'] == 'annotations'}
        for row in rows:
            if row['region_id'] in pending:
                row['annotation_conflicts'].append(dict(region_ids=[row['region_id']], conflict_peers=[],
                    reason='Historical revisions of this annotation were ambiguous during training; confirm or correct the saved classification.'))
    # The trainer reports competing labels at identical geometry while retaining
    # both source records in the frozen evidence. Keep that unresolved evidence
    # visible even when its experiment has no maps in this validation campaign.
    coverage["inherited_geometry_conflicts"] = conflicts
    coverage["deferred_revision_conflicts"] = audit.get("deferred_revision_conflicts", [])
    model_id, model_identity = _model_identity(model)
    frozen = [[dict(path=str(path), sha256=sha256_file(path)) for path in pair] for pair in pairs]
    report_path = assets / "evaluation.json"
    write(report_path, dict(format=FORMAT, completed_utc=utc_now(), model_root=str(model), model_id=model_id,
        model_identity=model_identity, review_root=str(source), new_manifest_sha256=sha256_file(source / "postflight_manifest.json"),
        frozen_evidence=frozen, prediction_sources=hashes, coverage=coverage, regions=rows,
        source_capture_audit=dict(path=str(assets / "capture_audit.json"), sha256=sha256_file(assets / "capture_audit.json")),
        scope="Optical maps without context completion; regional constraint violations and low-coverage review prompts, not label-error probabilities or independent accuracy",
        phase_review_basis="baseline-relative-optical-regions-v1" if regional else "studio-optical-without-context-v1",
        inference_performed=False, training_performed=False, live_reviews_modified=False))
    result = build_queue(report_path, output, threshold=threshold, verified_binding=binding)
    print(f"Registered {len(result['entries'])} disagreement regions in {output}", flush=True)
    if not coverage["complete"]:
        print(f"Explicitly unscored regions: {len(coverage['skipped_regions'])}; reasons: {coverage['skipped_reason_counts']}", flush=True)
    if conflicts:
        runs = ", ".join(sorted({row["run"].upper() for row in conflicts}))
        print(f"Unresolved saved-label conflicts: {len(conflicts)} ({runs}); both source annotations are preserved. "
              "Resolve these in their original saved reviews.", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--threshold", type=float, default=.75)
    args = parser.parse_args()
    build(args.model, args.output, review_root=args.review_root, threshold=args.threshold)
