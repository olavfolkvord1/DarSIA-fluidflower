"""Report saved Policy A predictions; stop at the first coarse Studio failure.

No inference, annotation/scorer read, concentration, or mass computation occurs.
Each changed pixel is stored in a compact spatial NPZ, including unknown NaNs.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np

from darsia.fluidflower import phase_ffac_selected_policy_a_qa as qa
from darsia.fluidflower.phase_ffac_selected_phase import area_time_series
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, utc_now

CONTRACT = "ffac-selected-policy-a-saved-report-v1"


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    columns = sorted({key for row in rows for key in row})
    if columns:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    with Path(path).open("x", encoding="utf-8", newline="") as target:
        target.write(stream.getvalue())
    return qa.asset(path)


def transported_zones(reference, phase):
    revision = phase["rig_domain_revision"]
    geometry = phase["full_extended_scientific"]["binding"]["geometry"]
    shift = np.eye(3); shift[:2, 2] = -np.asarray(revision["native_origin_xy"], float)
    h = shift @ np.asarray(geometry["H_ac28_to_frame_native"], float)
    # +1 distinguishes real reference zone 0 from missing reference support.
    values = cv2.warpPerspective(reference.astype(np.int16)+1, h,
        tuple(revision["phase_shape_hw"])[::-1], flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return values.astype(np.int16)-1


def frame_report(context, key, reader, destination, *, zones, injector=None, backend="auto"):
    source, phase_descriptor, phase, parent_descriptor, parent, before, current, area = qa.anchor_evidence(context, key, reader)
    comparison = qa.coarse_studio_gate(before, current, area)
    metric = qa.approved_canvas_to_mm(phase["rig_domain_revision"])
    targets = None if injector is None or metric is None else qa.metric_injectors(injector, phase["rig_domain_revision"])
    spatial, detail = qa.change_spatial_details(before, current, area,
        zones=transported_zones(zones, phase), canvas_to_mm=metric, injectors_mm=targets)
    geometry = phase["full_extended_scientific"]["binding"]["geometry"]
    # Optional GPU reduction is descriptive only. The CPU gate is authoritative.
    from darsia.fluidflower.phase_ffac_common_mass_round2_backend import reduce_weighted_codes
    _, _, reduction = reduce_weighted_codes(detail["zone"], detail["pixel_area_mm2"], backend=backend)
    run, index = key
    folder = Path(destination)/"frames"/run/f"frame_{index:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    with (folder/"changed_pixels.npz").open("xb") as stream:
        np.savez_compressed(stream, **detail)
    record = dict(contract=qa.CONTRACT, complete=True, created_utc=utc_now(), run=run,
        original_studio_frame_index=index, source_work_index=source["work_index"],
        source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
        phase_receipt=phase_descriptor, parent_receipt=parent_descriptor,
        current_labels=phase["labels_map"], parent_labels=parent["labels_map"],
        pixel_area=phase["rig_domain_revision"]["area_map"], comparison=comparison,
        spatial=spatial, changed_pixels=qa.asset(folder/"changed_pixels.npz"),
        injector_status="unknown" if injector is None else "source-bound protocol proxy; not photo-verified",
        reduction_backend=reduction, geometry_scope="approved measured projective metric chain; physical-context depth is not imputed",
        geometry_caveats=dict(unobserved_wall_uncertainty=geometry.get("unobserved_wall_uncertainty"),
            edge_gate_verified_on_own_photograph=geometry.get("edge_gate_verified_on_own_photograph"),
            boundary_evidence=geometry.get("boundary_evidence"),
            vertical_scale_independently_verified=phase["rig_domain_revision"]["physical_scale_revision"].get("vertical_scale_independently_verified")),
        maps_changed=False, labels_changed=False, annotations_read=False, mass_computed=False)
    atomic_json(folder/"receipt.json", record)
    return record, qa.asset(folder/"receipt.json")


def saved_areas(context, reader, parent_studio_rows):
    rows, missing = [], []
    originals = {run: [row for row in parent_studio_rows if row["run"] == run] for run in qa.RUNS}
    for run, inventory in sorted(context["inventories"].items()):
        for source in inventory["sources"]:
            path = context["root"]/"selected_phase_frames"/run/(source["source_sha256"]+".json")
            if not path.is_file():
                missing.append(dict(run=run, source_sha256=source["source_sha256"], selected=bool(source.get("selected")),
                    baseline=bool(source.get("baseline")), studio_anchor=bool(source.get("studio_anchor"))))
                continue
            descriptor = qa.asset(path)
            phase = reader.record(descriptor)
            qa.verify_phase_source(phase, source, context["plan"]["pipeline_sha256"])
            area = phase["qa"]["phase_area"]
            row = dict(run=run, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
                selected=bool(source.get("selected")), baseline=bool(source.get("baseline")),
                studio_anchor=bool(source.get("studio_anchor")), selected_index=source.get("selected_index"),
                studio_frame_index=source.get("studio_frame_index"), source_work_index=source["work_index"],
                phase_receipt_sha256=descriptor["sha256"],
                **{name: area[name] for name in ("co2_cm2", "gas_cm2", "aqueous_cm2", "co2_pixels", "gas_pixels", "aqueous_pixels")})
            rows.append(row)
    selected_steps = []
    for run in qa.RUNS:
        selected_steps.extend(area_time_series([row for row in rows if row["run"] == run and
            (row["selected"] or row["baseline"])], originals[run]))
    return rows, selected_steps, missing


def build(plan_path, root, identity_gate, destination, *, injector_reference=None, backend="auto"):
    started = time.perf_counter()
    reader = qa.EvidenceReader()
    identity_descriptor = qa.asset(identity_gate)
    # Admission happens before any prediction or image is read, or output made.
    context = qa.policy_context(plan_path, root, identity_descriptor, reader)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    zone_descriptor = context["plan"]["parent_chain"]["fixed_zones"]
    zones = reader.array(zone_descriptor, "zones", cache=True)
    injector_descriptor = None if injector_reference is None else qa.asset(injector_reference)
    injector = None if injector_descriptor is None else reader.record(injector_descriptor)
    receipts, flat_rows, failed = [], [], None
    for key in sorted(context["anchors"]):
        record, descriptor = frame_report(context, key, reader, destination,
            zones=zones, injector=injector, backend=backend)
        receipts.append(descriptor)
        row = dict(run=record["run"], original_studio_frame_index=key[1], source_work_index=record["source_work_index"],
            source_sha256=record["source_sha256"], elapsed_h=record["elapsed_h"],
            categorical_changed_pixels=record["comparison"]["categorical_changed_pixels"],
            support_changed_pixels=record["comparison"]["support_changed_pixels"],
            all_pass=record["comparison"]["all_pass"], receipt_sha256=descriptor["sha256"])
        for phase in ("co2", "gas"):
            row.update({phase+"_"+name: value for name, value in record["comparison"][phase].items()})
        flat_rows.append(row)
        print(json.dumps(dict(event="policy_a_saved_frame_qa", run=key[0], studio_frame_index=key[1],
            completed=len(receipts), expected=qa.EXPECTED_ANCHORS, all_pass=row["all_pass"])), flush=True)
        if not row["all_pass"]:
            failed = descriptor
            break
    complete = len(receipts) == qa.EXPECTED_ANCHORS and failed is None
    # No further phase reads after the first coarse failure.
    parent_studio_rows = [dict(run=row["run"], elapsed_h=row["elapsed_h"],
        co2_cm2=row["co2_parent_cm2"], gas_cm2=row["gas_parent_cm2"]) for row in flat_rows]
    rows, steps, missing = saved_areas(context, reader, parent_studio_rows) if complete else ([], [], [])
    files = dict(studio_qa=write_csv(destination/"studio_qa.csv", flat_rows),
        phase_areas=write_csv(destination/"phase_areas.csv", rows),
        selected_area_steps=write_csv(destination/"selected_area_steps.csv", steps))
    code = {name: qa.asset(path) for name, path in {"report": __file__, "qa": qa.__file__}.items()}
    report = dict(contract=CONTRACT, created_utc=utc_now(), complete=complete,
        all_pass=complete, first_failed_frame=failed, plan=context["plan_descriptor"], identity_gate=identity_descriptor,
        pipeline_sha256=context["plan"]["pipeline_sha256"], spec_sha256=qa.SPEC_SHA256,
        runs=list(qa.RUNS), studio_frames=len(receipts), expected_studio_frames=qa.EXPECTED_ANCHORS,
        frame_receipts=receipts, missing=missing,
        selected_timeline_complete=complete and not missing, files=files, fixed_zones=zone_descriptor,
        injector_reference=injector_descriptor, injector_position_verified=False,
        code=code, seconds=time.perf_counter()-started,
        parameters_changed=False, maps_changed=False, annotations_read=False, mass_computed=False,
        bottom_mass_allowed=False, default_changed=False, claude_scorer_gate_pending=True,
        round3_allowed=False, registered_by_this_report=False)
    atomic_json(destination/"report.json", report)
    text = ("Policy A saved-prediction QA\n\n"
        + ("All 782 coarse Studio checks passed. Claude's sealed scorer check is still pending.\n\n" if complete else
           "Stopped at the first coarse Studio failure; remaining maps were not read by this report.\n\n")
        + "CO2 XOR <= max(1% parent CO2 area, 0.5 cm2); gas XOR <= max(2% parent gas area, 0.5 cm2). "
        "Acceptance uses CPU float64 sums of the approved per-pixel area. GPU reductions, when available, are descriptive only.\n\n"
        "Every changed categorical pixel is in its frame's changed_pixels.npz, with parent/current category, zone, measured area "
        "and distances in mm. Missing geometry or injector locations are NaN; aggregate distances are null when unavailable. "
        "L1/L2 supplied from the existing nominal protocol proxy are explicitly not photo-verified. "
        "Plume-edge distance uses the parent CO2 boundary pixel centres.\n\n"
        "Time-series flags compare an adjacent selected area jump with the largest original Studio step bracketing it; "
        "they are descriptive and do not change the acceptance gate. No human labels or scorers were read. "
        "No map, model, session or mass product was changed. Round3 remains blocked until Claude's separate scorer check passes.\n")
    with (destination/"README.txt").open("x", encoding="utf-8") as stream:
        stream.write(text)
    return qa.asset(destination/"report.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "root", "identity-gate", "destination"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--injector-reference")
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args(argv)
    descriptor = build(args.plan, args.root, args.identity_gate, args.destination,
        injector_reference=args.injector_reference, backend=args.backend)
    print(json.dumps(descriptor)); return 0 if qa.EvidenceReader().record(descriptor)["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
