"""Explicitly register a fully verified FF_AC rig/camera candidate for review."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import uuid

from PIL import Image

from darsia.fluidflower import phase_ffac_rig_camera_metadata as metadata
from darsia.fluidflower.phase_ffac_rig_camera_runtime import SPEC_SHA256,bound_path,load_native_source,read
from darsia.fluidflower.phase_g3_production import CO2_SHA256,G3_SHA256,REFIT_SHA256
from darsia.fluidflower.phase_validation import _model_identity,_source_jobs
from darsia.fluidflower.project import sha256_file,utc_now
from darsia.fluidflower.review_storage import resolve_review_path
from .register_track1_studio_candidate import FORMAT

KIND="approved_g3_rig_camera"
WARNING=("Studio review candidate: approved G3 gas · Barrier rule B · Rig domain · Camera clock. "
         "Extended rig strips include explicitly flagged nearest-pixel fills. "
         "'Modelled' means finite inference support; added-edge light-field forecasts are not "
         "independently radiometrically validated. Absolute rig width and physical scale remain "
         "provisional. Classification review only; mass analysis and deployment are not approved.")
INVALID="QA FAILED: rig/camera Studio registration has changed or is incomplete; do not use for mass analysis or deployment."
FILES={"model_summary_sha256":"summary.json","model_request_sha256":"training_request.json",
    "native_source_manifest_sha256":"review_source/postflight_manifest.json",
    "validation_binding_sha256":"validation/validation_binding.json",
    "postflight_manifest_sha256":"validation/postflight_manifest.json",
    "production_receipt_sha256":"rig_camera_candidate_receipt.json",
    "source_plan_sha256":"rig_camera_candidate_plan.json","phase_profile_sha256":"phase_rig_camera.json",
    "revision_manifest_sha256":"rig_camera_revision/manifest.json",
    "studio_preparation_sha256":"studio_preparation.json","qa_evidence_sha256":"production_qa/rig_camera_qa.json",
    "g3_model_sha256":"models/g3/gas_head.json","regional_model_sha256":"regional_model.json",
    "co2_model_sha256":"models/generic/co2_presence_signal.json"}


def validate_qa(report):
    metadata.require(report.get("contract")==metadata.QA_CONTRACT and report.get("all_pass") is True
        and report.get("spec_sha256")==SPEC_SHA256 and report.get("configuration")==metadata.CONFIGURATION
        and report.get("run_count")==43 and report.get("frame_count")==864
        and report.get("g3_sha256")==G3_SHA256 and report.get("regional_refit_sha256")==REFIT_SHA256
        and report.get("co2_sha256")==CO2_SHA256 and report.get("labels_changed") is False,
        "Rig/camera QA is incomplete or changes the approved chain")
    rows=report.get("rows") or []; keys={(row.get("run"),row.get("frame_index")) for row in rows}
    grid=metadata.validate_phase_grid(report.get("phase_grid",metadata.phase_grid()))
    metadata.require(len(rows)==len(keys)==864 and len({run for run,_ in keys})==43,
                     "Rig/camera QA must cover 43 runs and 864 unique frames")
    for row in rows:
        error=row.get("timeline_camera_error_h")
        metadata.require(isinstance(error,(int,float)) and math.isfinite(error) and abs(error)<=.05,
                         "Camera-clock QA exceeds 0.05 hours")
        metadata.require(type(row.get("source_changed")) is bool and
            (row["source_changed"] or (row.get("interior_changed_gt30mm_pixels")==0
             and row.get("parent_valid_bits_preserved") is True)),
            "Unchanged source altered its frozen interior or calibrated finite pixels")
        metadata.require(row.get("phase_grid")==grid,"QA mixes native photos with extended phase grids")
        for key in ("source_sha256","native_manifest_sha256","geometry_receipt_sha256",
                    "area_map_sha256","classes_sha256","map_sha256","studio_map_sha256"):
            metadata.require(isinstance(row.get(key),str) and metadata._SHA.fullmatch(row[key]),
                             "Rig/camera QA lacks an exact source/output binding")
    return report


def _reports(model):
    report=validate_qa(read(model/FILES["qa_evidence_sha256"]))
    acceptance=read(bound_path(report["geometry_acceptance"],base=model))
    metadata.validate_geometry_acceptance(acceptance,geometry_sha256=report["geometry_manifest_sha256"],spec_sha256=SPEC_SHA256)
    for name in ("rig_check_measurement","relative_geometry_certificate"):
        bound_path(acceptance[name],base=model)
    labels=read(bound_path(report["source_label_audit"],base=model))
    metadata.require(labels.get("labels_unchanged") is True and labels.get("labels_reindexed_or_rewritten")==0,
                     "AC31 source-owned labels were moved or rewritten")
    return report


def inspect_candidate(model):
    model=resolve_review_path(model).resolve();identity=_model_identity(model)
    summary=read(model/"summary.json");approved=summary.get("approved_g3_production") or {}
    metadata.require(identity[0]==model.name==summary.get("model_id")==summary.get("campaign_id")
        and summary.get("mass_eligible") is False and summary.get("scientific_acceptance") is False
        and (approved.get("rig_domain_revision") or {}).get("format")==metadata.REVISION_CONTRACT
        and approved.get("configuration")==metadata.CONFIGURATION,"Rig/camera candidate identity or review scope changed")
    for name,expected in (("models/g3/gas_head.json",G3_SHA256),("regional_model.json",REFIT_SHA256),
                          ("models/generic/co2_presence_signal.json",CO2_SHA256)):
        metadata.require(sha256_file(model/name)==expected,"Rig/camera candidate changed a frozen head")
    qa=_reports(model)
    metadata.require(qa.get("model_identity")==list(identity),"Rig/camera QA binds another model")
    seal=read(model/"rig_camera_candidate_receipt.json")
    metadata.require(seal.get("complete") is True and seal.get("run_count")==43 and seal.get("frame_count")==864
        and seal.get("model_identity")==list(identity) and seal.get("spec_sha256")==SPEC_SHA256,
        "Rig/camera candidate is not completely sealed")
    parent_registration=read(bound_path(seal["parent_registration"],base=model))
    metadata.require(parent_registration.get("registration_kind")=="approved_g3_barrier_rule_b"
        and [parent_registration.get("model_id"),parent_registration.get("model_sha256")]==seal.get("parent_model_identity"),
        "Rig/camera revision lost its exact Rule B parent")
    prep=read(model/"studio_preparation.json")
    metadata.require(prep.get("status")=="ready" and prep.get("runs")==prep.get("runs_done")==43
        and prep.get("presentation_ready") is True and prep.get("partial") is False
        and prep.get("model_identity")==list(identity),"Rig/camera Studio preparation is incomplete")
    from darsia.fluidflower.phase_evidence_audit import verified_review_binding
    binding=verified_review_binding(model,saved_comparison=True)
    metadata.require(binding.get("run_count")==43 and binding.get("output_mode")=="review", "Rig/camera campaign is not review-only")
    jobs=_source_jobs(model/"validation");metadata.require(len(jobs)==43,"Rig/camera candidate lacks source runs")
    qa_rows={(row["run"],row["frame_index"]):row for row in qa["rows"]};maps={};sources={}
    for job in jobs:
        run=job["run"];native_path=resolve_review_path(job["review_output_dir"])/"native_frames"/run/"manifest.json"
        metadata.require(native_path.resolve().is_relative_to(model),"New source metadata must live in the new candidate")
        state=load_native_source(job["review_output_dir"],run)
        metadata.require(state["inputs"]["parent_native_manifest"]["sha256"]==
            parent_registration["native_run_manifest_sha256"][run],"Source revision does not inherit this exact Rule B native run")
        native=state["native"];sources[run]=sha256_file(native_path)
        metadata.require(job.get("dataset")=="ffac" and (job.get("quality_flags") or {}).get("mass_analysis_allowed") is False,
                         "Rig/camera source job enables another series or mass analysis")
        for record,source in zip(native["frame_sources"],state["inputs"]["frames"]):
            index=record["frame_index"];row=qa_rows[(run,index)]
            metadata.require(row["source_sha256"]==record["source_sha256"] and row["native_manifest_sha256"]==sources[run]
                and row["source_changed"]==source["source_changed"]
                and row["geometry_receipt_sha256"]==source["geometry_receipt"]["sha256"]
                and row["area_map_sha256"]==record["rig_domain_revision"]["area_map"]["sha256"]
                and row["classes_sha256"]==record["rig_domain_revision"]["classes"]["sha256"],
                "Rig/camera QA binds different sources, geometry, weights or fill classes")
            metadata.require(row["phase_grid"]==source["phase_grid"]==native[metadata.SOURCE_KEY]["phase_grid"],
                             "Rig/camera saved maps declare another source phase grid")
            for path_key,sha_key,mode in (("map_path","map_sha256","L"),("studio_map_path","studio_map_sha256","RGBA")):
                path=bound_path(dict(path=row[path_key],sha256=row[sha_key]))
                metadata.require(path.is_relative_to(model),"Rig/camera map escaped the new candidate")
                with Image.open(path) as image:
                    metadata.require(image.mode==mode and list(image.size)==row["phase_grid"]["shape_hw"][::-1],
                                     "Rig/camera map has another grid/encoding")
                    image.verify()
            maps[f"{run}/{index:03d}"]=row["studio_map_sha256"]
    metadata.require(len(maps)==864 and len(sources)==43,"Rig/camera map/source coverage is incomplete")
    return dict(format=FORMAT,registration_kind=KIND,created_utc=utc_now(),model_root=str(model),
        model_id=identity[0],model_sha256=identity[1],series="ffac",publication_scope="studio_validation_only",
        qa_status="passed_rig_camera",qa_warning=WARNING,configuration=metadata.CONFIGURATION,spec_sha256=SPEC_SHA256,
        run_count=43,frame_count=864,map_sha256=maps,native_run_manifest_sha256=sources,
        **{key:sha256_file(model/name) for key,name in FILES.items()},
        scientific_acceptance=False,deployment_allowed=False,mass_eligible=False,automatic_activation=False,
        optical_training_performed=False,labels_changed=False)


def registered_warning(model,value):
    try:
        model=resolve_review_path(model).resolve()
        metadata.require(value.get("format")==FORMAT and value.get("registration_kind")==KIND
            and resolve_review_path(value["model_root"]).samefile(model)
            and (value.get("model_id"),value.get("model_sha256"))==_model_identity(model)
            and value.get("publication_scope")=="studio_validation_only" and value.get("qa_status")=="passed_rig_camera"
            and value.get("spec_sha256")==SPEC_SHA256 and value.get("run_count")==43 and value.get("frame_count")==864
            and len(value.get("map_sha256") or {})==864 and len(value.get("native_run_manifest_sha256") or {})==43
            and all(value.get(key) is False for key in ("scientific_acceptance","deployment_allowed","mass_eligible","automatic_activation")),
            "Rig/camera registration changed")
        for key,name in FILES.items():
            metadata.require(sha256_file(model/name)==value.get(key),"Registered rig/camera evidence changed")
        _reports(model)
        return WARNING
    except (OSError,ValueError,KeyError,TypeError):
        return INVALID


def register(model,*,results=None):
    from .register_track1_studio_candidate import registration_path
    value=inspect_candidate(model);target=registration_path(value["model_id"],results=results)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary=target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x",encoding="utf-8") as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
