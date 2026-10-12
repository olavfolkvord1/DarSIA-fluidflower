"""Package a completed selected-camera campaign for non-default Studio review.

Only metadata and exact small frozen-model copies are written. Maps, RAW,
photographs, labels and sessions are never rewritten or re-inferred.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from darsia.fluidflower import phase_ffac_common_mass_inputs as assets
from darsia.fluidflower import phase_ffac_selected_review as review
from darsia.fluidflower.phase_validation import _model_identity
from . import build_ffac_round3_selected_plan as bridge
from .phase_reference_readiness import defer_validation_references

CONTRACT = "ffac-selected-studio-packaging-v1"
SEAL = "selected_studio_receipt.json"
WARNING = ("Selected-camera P1 final 3 review; the fixed Studio replay passed. "
    "The timeline contains the pre-injection baseline plus accepted selected captures. "
    "Maps and photographic assets are source-bound; bottom-strip mass remains blocked. "
    "Mass analysis, deployment and automatic activation are not approved.")


def _write(path,value):
    assets.write_new_json(path,value)
    return assets.asset(path)


def _copy_exact(descriptor,destination,reader):
    content = reader.path(descriptor).read_bytes()
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(content)
    result = assets.asset(destination)
    if result["sha256"] != descriptor["sha256"]:
        raise ValueError("Copying a frozen model changed its bytes")
    return result


def _display_row(entry,phase,display_descriptor,display,*,index,reference=False):
    source = entry["source_binding"]
    photographs = display.get("photographs",{})
    if (display.get("contract") != review.DISPLAY_CONTRACT or display.get("complete") is not True
            or display.get("binding",{}).get("phase_receipt") != entry["receipt"]
            or display["binding"].get("scientific_receipt") != entry["scientific_receipt"]
            or display["binding"].get("pipeline_sha256") != phase["binding"]["pipeline_sha256"]
            or display.get("run") != entry["run"] or display.get("source_sha256") != entry["source_sha256"]
            or display.get("elapsed_h") != entry["elapsed_h"] or display.get("frame_index") != entry["frame_index"]
            or display.get("phase_shape_hw") != phase["rig_domain_revision"]["phase_shape_hw"]
            or display.get("native_origin_xy") != phase["rig_domain_revision"]["native_origin_xy"]
            or set(photographs) != set(review.MODES) or display.get("display_only") is not True
            or display.get("scientific_inputs_changed") is not False or display.get("maps_changed") is not False):
        raise ValueError("Selected display receipt changes source, time, frozen pipeline or full-rig grid")
    for descriptor in photographs.values():
        if not descriptor.get("path") or len(descriptor.get("sha256","")) != 64 or not Path(descriptor["path"]).is_file():
            raise ValueError("A selected photographic view is missing its bound file")
    # Producer receipts bind the image hashes. Large files are SHA-verified on
    # first access by the session reader, rather than re-hashed during packaging.
    result = dict(frame_index=None if reference else index,source_work_index=entry["frame_index"],
        elapsed_h=entry["elapsed_h"],source_sha256=entry["source_sha256"],
        source_path=source["source_path"],source_name=Path(source["source_path"]).name,
        source_label_identity=dict(dataset="ffac",run=entry["run"],camera_source_sha256=entry["source_sha256"]),
        source_binding=source,selection_index=entry.get("selection_index"),
        baseline=bool(source.get("baseline")),target_hours=entry.get("target_hours",[]),
        phase_receipt=entry["receipt"],scientific_receipt=entry["scientific_receipt"],display_receipt=display_descriptor,
        maps=phase["maps"],labels_map=phase["labels_map"],photographs=photographs,
        rig_domain_revision=phase["rig_domain_revision"],operator_masks=phase["operator_masks"],
        full_extended_scientific=phase["full_extended_scientific"],
        baseline_extended_scientific=phase["baseline_extended_scientific"],
        phase_shape_hw=display["phase_shape_hw"],native_origin_xy=display["native_origin_xy"],
        frame_path=photographs["natural"]["path"],frame_sha256=photographs["natural"]["sha256"],
        native_jpeg_sha256=photographs["natural"]["sha256"],
        v5_raw_detail_jpeg_path=photographs["raw_detail"]["path"],v5_raw_detail_jpeg_sha256=photographs["raw_detail"]["sha256"],
        v5_display_modes={mode:dict(jpeg_path=photographs[mode]["path"],jpeg_sha256=photographs[mode]["sha256"])
            for mode in ("natural","lift1","lift2")},
        bottom_mass_allowed=False,mass_eligible=False,inference_authorized=False)
    if reference:
        result.update(kind="pre_injection_reference",included_in_timeline=True,timeline_frame_index=0)
    return result


def prepare(root,phase_plan_descriptor,gate_descriptor,*,display_root=None,reader=None):
    root = Path(root)
    reader = reader or assets.VerifiedAssetReader()
    gate = reader.json(gate_descriptor)
    if (gate.get("contract") != bridge.PHASE_COMPLETION_CONTRACT or gate.get("complete") is not True
            or gate.get("all_39_complete") is not True or gate.get("studio_time_parity_passed") is not True):
        raise ValueError("Studio packaging requires the actual completed strict phase campaign")
    plan = reader.json(phase_plan_descriptor)
    inspection = bridge.inspect_selection(phase_plan_descriptor,root,plan["scientific_source_root"],reader)
    if inspection["missing"] or inspection["missing_anchors"]:
        raise ValueError("Selected source production is incomplete")
    bridge.verify_completion_gate(gate,phase_plan_descriptor,inspection,reader)
    complete = reader.json(gate["selected_completion_seal"])
    display_root = Path(display_root or root/"display_renderings")
    entries = complete["entries"]+complete["baselines"]
    by_run = {}
    for entry in entries:
        digest = entry["source_sha256"]
        descriptor = assets.asset(display_root/entry["run"]/digest/"receipt.json")
        phase,display = reader.json(entry["receipt"]),reader.json(descriptor)
        row = _display_row(entry,phase,descriptor,display,index=None)
        by_run.setdefault(entry["run"],[]).append(row)
    if set(by_run) != set(gate["runs"]):
        raise ValueError("Photographic rendering does not cover the completed run set")
    # All source/display checks finish before any new package or readiness flag.
    protected = (root/"source_alias",root/"review_source",root/"selected_studio",
        root/review.PROFILE_FILE,root/"summary.json",root/"training_request.json",root/"studio_preparation.json",root/SEAL)
    if any(path.exists() for path in protected):
        raise FileExistsError("Selected Studio publication writes only new paths")
    parent = Path(plan["parent_chain"]["parent"])
    parent_summary = reader.json(assets.asset(parent/"summary.json"))
    parent_profile = reader.json(assets.asset(parent/"phase_rig_camera.json"))
    source_manifest = reader.json(assets.asset(parent/"review_source/postflight_manifest.json"))
    parent_validation = reader.json(assets.asset(parent/"validation/postflight_manifest.json"))
    source_jobs = {row["run"]:row for row in source_manifest["jobs"]}
    validation_jobs = {row["run"]:row for row in parent_validation["jobs"]}
    frozen_copies = {}
    for relative in ("models/generic/co2_presence_signal.json","models/generic/gas_within_co2_signal.json"):
        frozen_copies[relative] = _copy_exact(assets.asset(parent/relative),root/relative,reader)
    summary = dict(parent_summary,model_id=root.name,campaign_id=root.name,
        selected_camera_review=dict(contract=CONTRACT,pipeline_sha256=plan["pipeline_sha256"],
            accepted_phase_gate=gate_descriptor,selected_frames=gate["selected_frames"],baseline_frames=gate["baseline_frames"]),
        display_name="P1 final 3 · Selected camera grid · Studio review",
        run_count=len(by_run),frame_count=len(entries),warning=WARNING,training_performed=False,mass_eligible=False,scientific_acceptance=False,
        deployment_allowed=False,automatic_activation=False)
    _write(root/"summary.json",summary)
    model_identity = list(_model_identity(root))
    _write(root/"training_request.json",dict(format="ffac-selected-frozen-phase-review-request-v1",
        model_id=root.name,model_identity=model_identity,phase_plan=phase_plan_descriptor,
        inherited_training_request=assets.asset(parent/"training_request.json"),
        accepted_phase_gate=gate_descriptor,training=False,labels_read=False,labels_changed=False))
    runs,sources = {},[]
    for run,rows in sorted(by_run.items()):
        base = [row for row in rows if row["baseline"]]
        selected = sorted((row for row in rows if not row["baseline"]),key=lambda row:(row["elapsed_h"],row["source_sha256"]))
        if len(base)!=1 or len({row["source_sha256"] for row in rows}) != len(rows):
            raise ValueError("Studio run needs exactly one baseline and unique selected acquisitions")
        ordered = [base[0],*selected]
        for index,row in enumerate(ordered):row["frame_index"]=index
        reference = dict(ordered[0],frame_index=None,kind="pre_injection_reference",included_in_timeline=True,timeline_frame_index=0)
        old_native = reader.json(plan["runs"][run]["parent_native_manifest"])
        grid = ordered[0]["phase_shape_hw"]
        inputs = dict(contract=review.INPUT_CONTRACT,complete=True,run=run,frames=ordered,reference=reference,
            accepted_phase_gate=gate_descriptor,run_manifest=gate["run_manifests"][run],phase_plan=phase_plan_descriptor,
            pipeline_sha256=plan["pipeline_sha256"],labels_source_owned=True,labels_changed=False,
            timeline_index_is_not_source_index=True,scientific_arrays_modified=False)
        input_descriptor = _write(root/"source_alias/source_bindings"/(run+".json"),inputs)
        native = dict(schema=1,dataset="ffac",run=run,kind="selected_camera_full_rig_review_frames",
            frame_count=len(ordered),selected_frame_count=len(selected),baseline_frame_count=1,
            canvas_wh=grid[::-1],phase_shape_hw=grid,native_origin_xy=ordered[0]["native_origin_xy"],
            analysis_image_shape_hw=[2979,5472],content_box_xywh=[416,496,5472,2979],
            full_analysis_resolution=True,additional_spatial_resampling=False,
            frame_sources=ordered,reference=reference,domain=None,
            phase_topology=old_native.get("phase_topology"),injection_points=[],
            injection_point_display_status="not republished until own-frame placement is supplied",
            v5_photometry=dict(format="ffac-v5-native-source-materialization-v1",selected_review_only=True),
            **{review.SOURCE_KEY:dict(contract=review.SOURCE_CONTRACT,input_manifest=input_descriptor,
                accepted_phase_gate=gate_descriptor,inference_authorized=False,labels_source_owned=True)})
        native_descriptor = _write(root/"source_alias/native_frames"/run/"manifest.json",native)
        runs[run] = dict(native_manifest=native_descriptor,input_manifest=input_descriptor,
            frame_count=len(ordered),selected_frame_count=len(selected),baseline_frame_count=1)
        job = deepcopy(source_jobs[run])
        job.pop("phase_transfer_context",None)
        job.update(review_output_dir=str(root/"source_alias"),analysis_source_review_root=str(root/"source_alias"),
            native_manifest_sha256=native_descriptor["sha256"],frame_count=len(ordered),
            selected_camera_review=True,source_labels_bound_to_camera_sha256=True)
        _write(root/"review_source"/run/"postflight_manifest.json",dict(source_manifest,jobs=[job]))
        sources.append(job)
    source_descriptor = _write(root/"review_source/postflight_manifest.json",dict(source_manifest,jobs=sources,
        run_count=len(runs),frame_count=len(entries),selected_frame_count=gate["selected_frames"],baseline_frame_count=gate["baseline_frames"]))
    runtime = dict(contract=review.RUNTIME_CONTRACT,complete=True,pipeline_sha256=plan["pipeline_sha256"],
        accepted_phase_gate=gate_descriptor,run_count=len(runs),selected_frame_count=gate["selected_frames"],
        baseline_frame_count=gate["baseline_frames"],timeline_frame_count=sum(row["frame_count"] for row in runs.values()),runs=runs)
    runtime_descriptor = _write(root/"selected_studio/runtime_manifest.json",runtime)
    profile = dict(format=review.PROFILE_FORMAT,pipeline_sha256=plan["pipeline_sha256"],model_identity=model_identity,
        accepted_phase_gate=gate_descriptor,runtime_manifest=runtime_descriptor,
        frozen_heads=plan["heads"],inherited_public_profile=parent_profile,
        publication_scope="studio_validation_only",mass_eligible=False,bottom_mass_allowed=False,
        scientific_acceptance=False,deployment_allowed=False,automatic_activation=False,training=False,labels_changed=False)
    profile_descriptor = _write(root/review.PROFILE_FILE,profile)
    physics_runtime = review.runtime_identity(profile)
    packages,jobs=[],[]
    completed_utc=datetime.now(timezone.utc).isoformat()
    for run,state in sorted(runs.items()):
        package = root/"selected_studio/validation/run_packages"/run
        job = deepcopy(validation_jobs[run])
        job.pop("phase_transfer_context",None)
        job.update(candidate_id=root.name+"__ffac__"+run+"__selected_"+plan["pipeline_sha256"][:12],
            candidate_folder=str(root),review_output_dir=str(root/"source_alias"),analysis_source_review_root=str(root/"source_alias"),
            phase_map_output_dir=str(root/"validation/run_packages"/run/"review_assets"),
            params_sha256=plan["pipeline_sha256"],optical_model_sha256=model_identity[1],frame_count=state["frame_count"],
            native_manifest_sha256=state["native_manifest"]["sha256"],physics_runtime=physics_runtime,
            selected_camera_review=True,output_mode="review",phase_model_only=True,
            raw_phase_map_output_dir=None,numeric_phase_output_dir=None,presence_diagnostics_dir=None,
            presentation_status="ready",proposals_status="deferred")
        job["quality_flags"] = dict(selected_camera_review=True,human_review_required=True,
            mass_analysis_allowed=False,scientific_acceptance=False,bottom_mass_allowed=False)
        for key in ("summary","mass_plot","contour_movie","per_frame_csv"):job[key]=None
        common = dict(schema=1,format="darsia-phase-validation-campaign-v1",campaign_id="phase-validation-"+root.name,
            model_id=model_identity[0],model_sha256=model_identity[1],model_root=str(root),
            source_manifest_sha256=assets.asset(root/"review_source"/run/"postflight_manifest.json")["sha256"],
            pipeline_sha256=plan["pipeline_sha256"],physics_runtime=physics_runtime,physics_profile=profile,
            output_mode="review",numeric_outputs=False,phase_display_policy=review.SOURCE_CONTRACT,
            completed_utc=completed_utc)
        _write(package/"postflight_manifest.json",dict(common,jobs=[job],source_campaign_root=str(root/"review_source"/run)))
        _write(package/"validation_binding.json",dict(common,combined_root=str(package),ffac_root=str(package),ffum_root=str(package),
            selected_candidates={run:job["candidate_id"]},run_count=1))
        defer_validation_references(package)
        packages.append(package);jobs.append(job)
    snapshot = root/"selected_studio/validation/prepared_sets"/assets.asset(root/review.PROFILE_FILE)["sha256"][:20]
    common.update(source_manifest_sha256=source_descriptor["sha256"])
    _write(snapshot/"postflight_manifest.json",dict(common,jobs=jobs,source_campaign_root=str(root/"review_source")))
    _write(snapshot/"validation_binding.json",dict(common,combined_root=str(snapshot),ffac_root=str(snapshot),ffum_root=str(snapshot),
        selected_candidates={job["run"]:job["candidate_id"] for job in jobs},run_count=len(runs)))
    defer_validation_references(snapshot)
    preparation = dict(status="ready",runs=len(runs),runs_done=len(runs),frame_count=runtime["timeline_frame_count"],
        selected_frame_count=gate["selected_frames"],baseline_frame_count=gate["baseline_frames"],
        presentation_ready=True,partial=False,model_identity=model_identity,validation=str(snapshot),
        settings_file=None,default_activation_performed=False,selected_camera_review=True)
    preparation_descriptor = _write(root/"studio_preparation.json",preparation)
    receipt = dict(contract=CONTRACT,complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
        model_identity=model_identity,pipeline_sha256=plan["pipeline_sha256"],accepted_phase_gate=gate_descriptor,
        profile=profile_descriptor,runtime_manifest=runtime_descriptor,studio_preparation=preparation_descriptor,
        source_manifest=source_descriptor,validation_manifest=assets.asset(snapshot/"postflight_manifest.json"),
        validation_binding=assets.asset(snapshot/"validation_binding.json"),frozen_model_copies=frozen_copies,
        run_count=len(runs),selected_frame_count=gate["selected_frames"],baseline_frame_count=gate["baseline_frames"],
        timeline_frame_count=runtime["timeline_frame_count"],runs=runs,
        maps_changed=False,scientific_inputs_changed=False,labels_changed=False,sessions_changed=False,
        bottom_mass_allowed=False,mass_eligible=False,automatic_activation=False,registration_performed=False,
        photographic_hash_policy="Producer receipts bind bytes; verify once on first session access")
    return _write(root/SEAL,receipt)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--phase-plan",required=True)
    parser.add_argument("--accepted-gate",required=True)
    parser.add_argument("--display-root")
    args=parser.parse_args(argv)
    print(json.dumps(prepare(args.candidate,assets.asset(args.phase_plan),assets.asset(args.accepted_gate),display_root=args.display_root)))


if __name__=="__main__":main()
