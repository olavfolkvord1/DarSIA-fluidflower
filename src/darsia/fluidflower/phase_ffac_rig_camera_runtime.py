"""Strict opt-in readers for source-bound camera/rig review artifacts.

This revision has its own profile and source contract. It never weakens a
legacy top/B alias, guesses a preceding source, or authorizes generic inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import phase_ffac_rig_camera_metadata as metadata
from .phase_training.v5_linear_input import RunV5LinearInput
from .project import sha256_file
from .review_storage import resolve_review_path

SPEC_SHA256="518c5b5bcd9386dc833e315098c1d762ea702c3c2fd3e0b7ad57c45a0677c24b"
PROFILE_FILE="phase_rig_camera.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bound_path(asset,*,base=None,verify=True):
    metadata.asset(asset)
    path=Path(asset["path"])
    path=resolve_review_path(path if path.is_absolute() else Path(base or ".")/path).resolve()
    if verify and sha256_file(path)!=asset["sha256"]:
        raise ValueError("Rig/camera bound source artifact changed")
    if not verify and not path.is_file():
        raise ValueError("Rig/camera bound source artifact is missing")
    return path


def _geometry(row,record,*,base,reference=False):
    receipt=read(bound_path(row["geometry_receipt"],base=base))
    source_sha=receipt.get("camera_jpeg_sha256",receipt.get("source_sha256"))
    metadata.require(source_sha==record["source_sha256"],"Geometry belongs to another camera source")
    h=receipt.get("H_ac28_to_frame_native",receipt.get("H_ac28_to_reference_native"))
    metadata.require(metadata.homography(h)==metadata.homography(row["H_ac28_to_frame_native"]),
                     "Frame uses a homography borrowed from another source")
    if not reference:
        metadata.require(metadata.homography(receipt["H_v5_B_to_frame_native"])
                         ==metadata.homography(row["H_v5_B_to_frame_native"]),"Frozen B-to-source geometry changed")
    if row.get("source_changed"):
        metadata.require(receipt.get("new_geometry_measured_from_own_source") is True
            and receipt.get("camera_arw_sha256")==record.get("v5_raw_sha256")
            and (receipt.get("source_sand_registration") or {}).get("geometry_pass") is True,
            "New source lacks its own source-exact photographic geometry proof")
    return receipt


def _scientific_receipt(record):
    full=record["full_native_scientific"]
    receipt=read(bound_path(full["receipt"]))
    metadata.require(receipt.get("contract")==metadata.SCIENTIFIC_CONTRACT
        and receipt.get("source_sha256")==record["source_sha256"]
        and receipt.get("log_linear_rgb_sha256")==full["log_linear_rgb"]["sha256"]
        and receipt.get("valid_mask_sha256")==full["valid_mask"]["sha256"]
        and receipt.get("parent_cache")==full["parent_cache"] and receipt.get("frozen_photometry") is True,
        "Full-native scientific correction certificate differs from its source/output/parent cache")
    if full["parent_cache"] is not None:
        metadata.require(receipt.get("parent_valid_bits_preserved") is True,
                         "Expanded scientific source lacks preserved-parent-bit certification")
    return receipt


def load_native_source(review_root,run,manifest=None,*,verify_geometry=True):
    root=resolve_review_path(review_root).resolve();run=str(run).casefold()
    native_path=root/"native_frames"/run/"manifest.json"
    native=read(native_path) if manifest is None else manifest
    marker=native.get(metadata.SOURCE_KEY)
    metadata.require(isinstance(marker,dict) and marker.get("contract")==metadata.SOURCE_CONTRACT
        and marker.get("inference_authorized") is False,
        "Invalid explicit rig/camera native source contract")
    metadata.validate_phase_grid(marker.get("phase_grid"))
    metadata.require(not any(key in native for key in (
        "v5_native_source_alias","v5_track1_photo_geometry","v5_frame_geometry")),
        "Rig/camera timeline cannot pretend to be an old mask-only alias")
    input_path=bound_path(marker["input_manifest"],base=native_path.parent)
    inputs=read(input_path)
    metadata.require(inputs.get("contract")==metadata.INPUT_CONTRACT and inputs.get("spec_sha256")==SPEC_SHA256,
                     "Rig/camera source certificate binds another specification")
    parent=read(bound_path(inputs["parent_native_manifest"],base=input_path.parent))
    camera=read(bound_path(inputs["camera_timeline_run"],base=input_path.parent))
    frames=inputs.get("frames") or []
    expected=metadata.build_native_metadata(parent,camera,native["frame_sources"],frames,inputs["reference"],
                                           input_manifest=marker["input_manifest"])
    metadata.require(native==expected and native.get("run")==run,
                     "Native source metadata differs from its frozen source/camera contract")
    if verify_geometry:
        for row,record in zip(frames,native["frame_sources"]):
            _geometry(row,record,base=input_path.parent)
            _scientific_receipt(record)
        _geometry(inputs["reference"],native["reference"],base=input_path.parent,reference=True)
        _scientific_receipt(native["reference"])
    return dict(root=root,native_path=native_path,native=native,inputs=inputs,input_path=input_path)


def native_frame_assets(review_root,run,frame_index=None,*,reference=False,manifest=None):
    state=load_native_source(review_root,run,manifest,verify_geometry=False)
    if reference:
        record=state["native"]["reference"];row=state["inputs"]["reference"]
    else:
        rows=state["native"]["frame_sources"]
        metadata.require(type(frame_index) is int and 0<=frame_index<len(rows),"Rig/camera frame is outside its timeline")
        record=rows[frame_index];row=state["inputs"]["frames"][frame_index]
    _geometry(row,record,base=state["input_path"].parent,reference=reference)
    _scientific_receipt(record)
    return state,record,row


def resolve_photograph(review_root,run,frame_index=None,*,mode="natural",reference=False,verify=True):
    state,_record,row=native_frame_assets(review_root,run,frame_index,reference=reference)
    metadata.require(mode in row["photographs"],"Unknown rig/camera photographic mode")
    return bound_path(row["photographs"][mode],base=state["input_path"].parent,verify=verify)


def prepared_photographs(review_root,run,manifest=None):
    """Availability hashes only small certificates; selected reads hash JPEGs."""
    state=load_native_source(review_root,run,manifest)
    for row in [state["inputs"]["reference"],*state["inputs"]["frames"]]:
        for mode in ("natural","lift1","lift2","change","raw_detail","original"):
            bound_path(row["photographs"][mode],base=state["input_path"].parent,verify=False)
    return bound_path(state["inputs"]["reference"]["photographs"]["natural"],
                      base=state["input_path"].parent,verify=False)


class RunRigCameraLinearInput(RunV5LinearInput):
    """Reuse the exact float decoder with the separate per-source certificates."""
    def __init__(self,*,native_manifest_path,run_id,**kwargs):
        native=Path(native_manifest_path)
        self.rig_source=load_native_source(native.parent.parent.parent,run_id)
        super().__init__(native_manifest_path=native_manifest_path,run_id=run_id,**kwargs)

    def _bound_source(self,photo_path,frame_id):
        reference=frame_id=="pre_injection"
        if reference:
            row=self.rig_source["inputs"]["reference"];record=self.reference
        else:
            if frame_id not in self.frames:
                raise ValueError("Unknown source-bound rig/camera scientific frame")
            record=self.frames[frame_id]
            row=self.rig_source["inputs"]["frames"][record["frame_index"]]
        expected=bound_path(row["photographs"]["natural"],base=self.rig_source["input_path"].parent)
        metadata.require(resolve_review_path(photo_path).resolve()==expected,"Scientific frame uses another photograph")
        full=record["full_native_scientific"]
        _scientific_receipt(record)
        sidecar=bound_path(full["log_linear_rgb"],verify=False)
        mask=bound_path(full["valid_mask"],verify=False)
        return record,row["photographs"]["natural"]["sha256"],sidecar,full["log_linear_rgb"]["sha256"],mask,full["valid_mask"]["sha256"]

    def _bound_extended_ring(self,record,photo):
        pad=record.get("v5_extended_pad_px");origin=record.get("v5_extended_origin_xy")
        metadata.require(type(pad) is int and 1<=pad<=256 and origin==[-pad,-pad],"Invalid source-bound rig/camera border ring")
        path=bound_path(dict(path=record["v5_extended_ring_path"],sha256=record["v5_extended_ring_sha256"]),verify=False)
        return path,record["v5_extended_ring_sha256"],pad,origin


def load_profile(root):
    candidate=resolve_review_path(root).resolve();path=candidate/PROFILE_FILE
    profile=read(path)
    metadata.require(profile.get("format")==metadata.PROFILE_FORMAT and profile.get("spec_sha256")==SPEC_SHA256
        and profile.get("configuration")==metadata.CONFIGURATION
        and profile.get("publication_scope")=="studio_validation_only"
        and all(profile.get(key) is False for key in (
            "mass_eligible","scientific_acceptance","deployment_allowed","automatic_activation","training","labels_changed")),
        "Rig/camera profile is not an exact frozen-head review revision")
    marker=profile.get("rig_domain_revision") or {}
    metadata.require(marker.get("format")==metadata.REVISION_CONTRACT,
                     "Rig/camera profile conflates its photographic and extended phase grids")
    phase_grid=metadata.validate_phase_grid(marker.get("phase_grid"))
    manifest_path=bound_path(marker["manifest"],base=candidate)
    metadata.require(manifest_path.is_relative_to(candidate),"Rig/camera runtime manifest escaped its candidate")
    manifest=read(manifest_path)
    metadata.validate_campaign_bindings(manifest["sources"])
    metadata.require(manifest.get("contract")==metadata.REVISION_CONTRACT and manifest.get("spec_sha256")==SPEC_SHA256,
                     "Rig/camera runtime source manifest changed")
    metadata.require(manifest.get("phase_grid",phase_grid)==phase_grid,
                     "Runtime manifest and profile declare different phase rasters")
    for source in manifest["sources"]:
        for row in source["frames"]:
            metadata.require(row.get("phase_grid")==phase_grid,
                             "Runtime source declares another phase raster")
    acceptance=read(bound_path(manifest["geometry_acceptance"],base=manifest_path.parent))
    metadata.validate_geometry_acceptance(acceptance,geometry_sha256=manifest["geometry_manifest"]["sha256"],spec_sha256=SPEC_SHA256)
    for name in ("rig_check_measurement","relative_geometry_certificate"):
        bound_path(acceptance[name],base=manifest_path.parent)
    bound_path(manifest["geometry_manifest"],base=manifest_path.parent)
    if "g3_t4b_floor_revision" in profile:
        metadata.require(not {"g3_t4_floor_revision", "l1_neighbourhood_floor_revision",
                              "g3_t3_floor_revision", "g3_t2_l1zone_revision",
                              "g3_t2_floor_revision"}.intersection(profile),
                         "A T4b candidate cannot also declare another gas revision")
        from .tools.run_ffac_t4b_floor_candidate import validate_runtime_heads
        validate_runtime_heads(candidate, profile["g3_t4b_floor_revision"], profile["frozen_heads"])
    elif "g3_t4_floor_revision" in profile:
        metadata.require(not {"l1_neighbourhood_floor_revision", "g3_t3_floor_revision",
                              "g3_t2_l1zone_revision", "g3_t2_floor_revision"}.intersection(profile),
                         "A T4 candidate cannot also declare another gas revision")
        from .tools.run_ffac_t4_floor_candidate import validate_runtime_heads
        validate_runtime_heads(candidate, profile["g3_t4_floor_revision"], profile["frozen_heads"])
    elif "l1_neighbourhood_floor_revision" in profile:
        metadata.require(not {"g3_t3_floor_revision", "g3_t2_l1zone_revision", "g3_t2_floor_revision"}.intersection(profile),
                         "An L1-neighbourhood candidate cannot also declare a trained T2/T3 revision")
        from .tools.run_ffac_l1_neighbourhood_candidate import validate_runtime_heads
        validate_runtime_heads(candidate, profile["l1_neighbourhood_floor_revision"], profile["frozen_heads"])
    elif "g3_t3_floor_revision" in profile:
        metadata.require(not {"g3_t2_l1zone_revision", "g3_t2_floor_revision"}.intersection(profile),
                         "A T3 candidate cannot also declare a T2 revision")
        from .tools.run_ffac_t3_floor_candidate import validate_runtime_heads
        validate_runtime_heads(candidate, profile["g3_t3_floor_revision"], profile["frozen_heads"])
    elif "g3_t2_l1zone_revision" in profile:
        metadata.require("g3_t2_floor_revision" not in profile,
                         "A combined L1 candidate cannot also declare a whole-frame T2 revision")
        from .tools.run_ffac_t2_l1zone_candidate import validate_runtime_heads
        validate_runtime_heads(candidate, profile["g3_t2_l1zone_revision"], profile["frozen_heads"])
    elif "g3_t2_floor_revision" in profile:
        from .tools.run_ffac_t2_floor_candidate import validate_runtime_heads
        validate_runtime_heads(candidate, profile["g3_t2_floor_revision"], profile["frozen_heads"])
    else:
        from .phase_g3_production import CO2_SHA256,G3_SHA256,REFIT_SHA256
        for key,expected in (("co2",CO2_SHA256),("g3",G3_SHA256),("regional",REFIT_SHA256)):
            metadata.require(profile["frozen_heads"][key]["sha256"]==expected,"Rig/camera revision changed an approved head")
            bound_path(profile["frozen_heads"][key],base=candidate)
    return dict(profile,_rig_camera_runtime=dict(root=str(candidate),manifest=manifest))


def runtime_identity(profile):
    payload=dict(profile=metadata.public_profile(profile),reader_sha256=sha256_file(Path(__file__)),
                 metadata_sha256=sha256_file(Path(metadata.__file__)),numpy=np.__version__)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def require_dedicated_inference():
    raise ValueError("Rig/camera saved review does not authorize generic inference; use its exact frozen revision runner")
