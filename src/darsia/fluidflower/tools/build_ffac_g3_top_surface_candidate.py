"""Build a new saved-map candidate by clipping only above the photo surface.

The accepted geometry supplies registered, source-bound removal masks. No
optical inference, gas decision, component filtering, training or photo render
is performed. The old candidate and its original source remain immutable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import io
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.tools.build_ffac_g3_approved_candidate import (
    CONFIGURATION, SPEC_SHA256, _write_bytes, assemble_saved_candidate, mask_sha256, write_phase_layers,
)
from darsia.fluidflower.tools.refit_ffac_g3_downstream import (
    JobLease, ProgressLog, atomic_json, below_normal_priority, canonical_sha256,
    copy_file_verified, preflight_space, read_json, sha256_file, utc_now, worker_initializer,
)

FORMAT = "ffac-photo-sand-top-revision-v1"
ALIAS_FORMAT = "ffac-photo-sand-top-source-alias-v1"
CONTRACT = "ffac-g3-top-surface-candidate-v1"
QA_CONTRACT = "ffac-g3-top-surface-qa-v1"


def _asset(row, name):
    item = row.get(name)
    if isinstance(item, dict):
        return dict(path=str(item["path"]), sha256=item["sha256"])
    return dict(path=str(row[name + "_path"]), sha256=row[name + "_sha256"])


def _bound_asset(item):
    path = Path(item["path"]).resolve()
    if sha256_file(path) != item["sha256"]:
        raise ValueError("Surface revision bound source changed")
    return path


def _npy(path, value):
    encoded = io.BytesIO()
    np.save(encoded, np.asarray(value), allow_pickle=False)
    return _write_bytes(Path(path), encoded.getvalue())


def read_surface_mask(asset, *, original=None, shape=None):
    """Validate the registered removal-only bundle, without inventing support."""
    with np.load(_bound_asset(asset), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("top_keep", "photo_sand_original", "photo_sand_revised")}
    target = tuple(shape) if shape is not None else arrays["top_keep"].shape
    if any(value.dtype != np.bool_ or value.shape != target or value.ndim != 2 for value in arrays.values()):
        raise ValueError("Surface mask bundle differs from its source-native boolean grid")
    keep = arrays["top_keep"]
    if not keep.size or not keep[-1].all() or np.any(keep[:-1] & ~keep[1:]):
        raise ValueError("Surface keep mask must be a false-prefix/true-suffix in every column")
    if not np.array_equal(arrays["photo_sand_revised"], arrays["photo_sand_original"] & keep):
        raise ValueError("Surface revision changed photo sand below the accepted top boundary")
    if original is not None and not np.array_equal(arrays["photo_sand_original"], original):
        raise ValueError("Surface mask bundle does not retain the exact original photo-sand source")
    return arrays


def _copy_scientific_lineage(parent, root):
    for directory in ("models", "input_snapshot", "parent_provenance", "registered_context"):
        for source in (parent / directory).rglob("*"):
            if source.is_file():
                copy_file_verified(source, root / source.relative_to(parent))
    for name in ("training_request.json", "expected_review_runs.json", "evidence.json", "extraction.json",
                 "regional_training_protocol.json", "region_inventory.csv", "regional_model.json",
                 "approved_source_spec.md", "approved_selection_results.md"):
        if (parent / name).is_file():
            copy_file_verified(parent / name, root / name)
    for source in (parent / "review_source").rglob("*.json"):
        if source.name != "postflight_manifest.json":
            copy_file_verified(source, root / source.relative_to(parent))


def _record_source_mask(root, record, geometry, *, run, index, shape, reference=False):
    original_path = Path(record["v5_photo_sand_mask_path"])
    if sha256_file(original_path) != record["v5_photo_sand_mask_sha256"]:
        raise ValueError("Original photograph-only sand mask changed")
    original = np.load(original_path, allow_pickle=False)
    source = _asset(geometry, "mask")
    arrays = read_surface_mask(source, original=original, shape=shape)
    stem = "pre_injection" if reference else f"frame_{index:03d}"
    folder = root / "source_alias" / ("native_references" if reference else "native_frames") / run
    revised = _npy(folder / (stem + ".photo_sand_mask.npy"), arrays["photo_sand_revised"])
    bundle = root / "photo_sand_top_revision/masks" / run / (stem + ".npz")
    copy_file_verified(_bound_asset(source), bundle)
    registration = _asset(geometry,"registration_receipt")
    registration_path = root / "photo_sand_top_revision/registrations" / run / (stem + ".json")
    copy_file_verified(_bound_asset(registration),registration_path)
    record["v5_photo_sand_mask_path"], record["v5_photo_sand_mask_sha256"] = revised["path"], revised["sha256"]
    return dict(geometry, mask=dict(path=str(bundle), sha256=source["sha256"]),
                revised_photo_mask=revised,reference=reference,
                registration_receipt=dict(path=str(registration_path),sha256=registration["sha256"]))


def _stage_source_run(root,job,rows,reference_geometry,copied_geometry,accepted_sha):
    root=Path(root)
    run=job["run"]
    original_review=Path(job["review_output_dir"])
    native_path=original_review/"native_frames"/run/"manifest.json"
    native_sha=sha256_file(native_path)
    native=read_json(native_path)
    shape=native["analysis_image_shape_hw"]
    shadow=deepcopy(native)
    group=[]
    indexed={int(row["frame_index"]):row for row in rows}
    if set(indexed)!=set(range(len(shadow["frame_sources"]))):
        raise ValueError("Surface geometry does not cover the exact original run timeline")
    for index,record in enumerate(shadow["frame_sources"]):
        row=indexed[index]
        source_native=_asset(row,"native_manifest")
        if source_native["sha256"]!=native_sha or not Path(source_native["path"]).samefile(native_path):
            raise ValueError("Surface geometry identifies another sealed native image source")
        if float(row["elapsed_h"])!=float(record["elapsed_h"]):
            raise ValueError("Surface geometry identifies another exact native frame time")
        updated=_record_source_mask(root,record,row,run=run,index=index,shape=shape)
        parent_map=Path(job["phase_map_output_dir"])/"native_phase_maps"/run/("A_"+job["params_sha256"][:16])/f"frame_{index:03d}.png"
        labels_map=Path(job["candidate_folder"])/"production_labels"/run/f"frame_{index:03d}.png"
        updated.update(parent_phase_map=dict(path=str(parent_map),sha256=sha256_file(parent_map)),
                       parent_labels_map=dict(path=str(labels_map),sha256=sha256_file(labels_map)),
                       content_box_xywh=native["content_box_xywh"],canvas_wh=native["canvas_wh"])
        group.append(updated)
    ref_native=_asset(reference_geometry,"native_manifest")
    if ref_native["sha256"]!=native_sha or not Path(ref_native["path"]).samefile(native_path):
        raise ValueError("Reference surface geometry identifies another original native source")
    reference=_record_source_mask(root,shadow["reference"],reference_geometry,run=run,index=None,shape=shape,reference=True)
    shadow["v5_native_source_alias"]=dict(contract=ALIAS_FORMAT,original_review_root=str(original_review),
        original_native_manifest=dict(path=str(native_path),sha256=native_sha),
        geometry_manifest=dict(path=str(copied_geometry),sha256=accepted_sha))
    shadow_path=root/"source_alias/native_frames"/run/"manifest.json"
    atomic_json(shadow_path,shadow)
    digest=sha256_file(shadow_path)
    for updated in [*group,reference]:
        updated.update(shadow_native_manifest_sha256=digest,shadow_native_manifest_path=str(shadow_path))
    return dict(run=run,frames=group,reference=reference,shadow_native_manifest_sha256=digest)


def _shadow_source_profile(effective, shadow_sha):
    """Change source identities, retaining every scientific geometry binding."""
    profile = {key:deepcopy(value) for key,value in effective.items() if not key.startswith("_")}
    for run, digest in shadow_sha.items():
        entry = profile["runs"]["ffac/" + run]
        entry["native_manifest_sha256"] = digest
        for key in ("v5_track1_photo_geometry", "v5_frame_geometry"):
            provenance = (entry.get("provenance") or {}).get(key)
            if provenance is not None:
                provenance["native_manifest_sha256"] = digest
    return profile


def prepare_surface_candidate(parent: Path, output: Path, geometry_manifest: Path,
                              accepted_geometry_sha256: str, *, resume=False,workers=8):
    from darsia.fluidflower.tools.register_track1_studio_candidate import registration_path, registration_warning
    from darsia.fluidflower.phase_optical_regions_runtime import load_profile
    parent, root, geometry_path = map(lambda value: Path(value).resolve(), (parent, output, geometry_manifest))
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("Surface correction requires a new detached candidate")
    if sha256_file(geometry_path) != accepted_geometry_sha256:
        raise ValueError("Geometry differs from the explicitly accepted immutable boundary bundle")
    if not registration_warning(parent).startswith("Studio review candidate: approved G3"):
        raise ValueError("Surface correction requires an intact approved G3 parent")
    parent_identity = list(_model_identity(parent))
    parent_registration = registration_path(parent_identity[0])
    geometry = read_json(geometry_path)
    if geometry.get("contract", geometry.get("format")) != FORMAT:
        raise ValueError("Unexpected photograph-surface geometry manifest")
    manifest = read_json(parent / "validation/postflight_manifest.json")
    jobs = {job["run"]:job for job in manifest["jobs"]}
    rows = {(row["run"], int(row["frame_index"])):row for row in geometry["frames"]}
    refs = {row["run"]:row for row in geometry.get("references", [])}
    if (len(jobs) != 43 or len(rows) != 864 or len(geometry["frames"]) != 864
            or len(geometry.get("references", [])) != 43 or set(refs) != set(jobs)):
        raise ValueError("Surface geometry must cover the exact43/864 and each separately registered reference")
    binding = dict(contract=CONTRACT, parent=str(parent), candidate=str(root), parent_model_identity=parent_identity,
        parent_registration=dict(path=str(parent_registration), sha256=sha256_file(parent_registration)),
        geometry_manifest=dict(path=str(geometry_path), sha256=accepted_geometry_sha256),
        configuration=CONFIGURATION, spec_sha256=SPEC_SHA256, g3_sha256=G3_SHA256,
        regional_refit_sha256=REFIT_SHA256, co2_sha256=CO2_SHA256, code_sha256=sha256_file(__file__),
        cpu_only=True, training=False, model_inference=False, photo_pixels_changed=False,
        scientific_acceptance=False, mass_eligible=False)
    plan_path = root / "top_surface_candidate_plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if not resume or plan["binding"] != binding or plan["model_identity"] != list(_model_identity(root)):
            raise ValueError("Surface candidate resume differs from its immutable plan")
        return plan
    preparation_path=root/"surface_preparation_binding.json"
    if root.exists():
        if not resume or not preparation_path.is_file() or read_json(preparation_path)["binding"]!=binding:
            raise ValueError("Use a new absent surface candidate, or resume its exact preparation binding")
        preparation=read_json(preparation_path)
    else:
        if resume:
            raise ValueError("Cannot resume a surface candidate without its preparation binding")
        preflight_space(root,32*1024**3)
        root.mkdir(parents=True,exist_ok=False)
        preparation=dict(contract=CONTRACT,binding=binding,created_utc=utc_now())
        atomic_json(preparation_path,preparation)
    for name, digest in (("models/g3/gas_head.json",G3_SHA256), ("regional_model.json",REFIT_SHA256),
                         ("models/generic/co2_presence_signal.json",CO2_SHA256)):
        if sha256_file(parent / name) != digest:
            raise ValueError("Surface correction cannot change or replace approved model heads")
    log=ProgressLog(root)
    log.event("surface_source_staging_started",total_runs=43,workers=min(8,workers))
    _copy_scientific_lineage(parent, root)
    copied_geometry = root / "photo_sand_top_revision/source_geometry_manifest.json"
    copy_file_verified(geometry_path, copied_geometry)
    prepared_rows,prepared_refs,shadow_sha=[],[],{}
    started=time.monotonic()
    tasks=[(str(root),job,[row for (source_run,_),row in rows.items() if source_run==run],refs[run],
            str(copied_geometry),accepted_geometry_sha256) for run,job in sorted(jobs.items())]
    def consume(staged):
        prepared_rows.extend(staged["frames"])
        prepared_refs.append(staged["reference"])
        shadow_sha[staged["run"]]=staged["shadow_native_manifest_sha256"]
        done=len(shadow_sha)
        elapsed=time.monotonic()-started
        log.event("surface_source_run",run=staged["run"],completed_runs=done,total_runs=43,
                  elapsed_seconds=elapsed,eta_seconds=elapsed*(43-done)/done)
    if workers==1:
        for task in tasks:
            consume(_stage_source_run(*task))
    else:
        with ProcessPoolExecutor(max_workers=min(8,workers),mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initializer,initargs=(1,)) as pool:
            for future in as_completed([pool.submit(_stage_source_run,*task) for task in tasks]):
                consume(future.result())
    prepared_rows.sort(key=lambda row:(row["run"],row["frame_index"]))
    prepared_refs.sort(key=lambda row:row["run"])
    revision_path = root / "photo_sand_top_revision/manifest.json"
    atomic_json(revision_path, dict(contract=FORMAT, source_geometry_manifest_sha256=accepted_geometry_sha256,
        parent_model_identity=parent_identity, frames=prepared_rows, references=prepared_refs,
        operation="clip-only-after-approved-g3", outside_phase_label=255, outside_phase_display="transparent"))
    effective = load_profile(parent)
    profile = _shadow_source_profile(effective, shadow_sha)
    profile["photo_sand_top_revision"] = dict(format=FORMAT,
        manifest=dict(path="photo_sand_top_revision/manifest.json",sha256=sha256_file(revision_path)),
        operation="clip-only-after-approved-g3",outside_phase_label=255,outside_phase_display="transparent")
    atomic_json(root / "phase_optical_regions.json",profile)
    summary = read_json(parent / "summary.json")
    summary.update(model_id=root.name,campaign_id=root.name,created_utc=preparation["created_utc"],
        display_name="FF_AC · G3 gas · corrected photographic sand surface",
        inherited_training_candidate_id=parent.name,training_performed_in_this_revision=False,
        raw_inference_performed=False,scientific_acceptance=False,mass_eligible=False)
    summary["approved_g3_production"]["domain_revision"] = dict(format=FORMAT,
        parent_model_identity=parent_identity,geometry_manifest_sha256=accepted_geometry_sha256,
        mask_manifest_sha256=sha256_file(revision_path),operation="clip-only-after-approved-g3")
    summary["domain_revision"] = summary["approved_g3_production"]["domain_revision"]
    atomic_json(root / "summary.json",summary)
    for source in [parent / "review_source/postflight_manifest.json",
                   *(parent / "review_source" / run / "postflight_manifest.json" for run in jobs)]:
        source_manifest = read_json(source)
        for job in source_manifest["jobs"]:
            job["review_output_dir"] = str(root / "source_alias")
            job["analysis_source_review_root"] = str(root / "source_alias")
        atomic_json(root / source.relative_to(parent),source_manifest)
    identity = list(_model_identity(root))
    pipeline = canonical_sha256(dict(binding=binding,model_identity=identity,profile_sha256=sha256_file(root / "phase_optical_regions.json")))
    plan = dict(created_utc=preparation["created_utc"],binding=binding,job_sha256=canonical_sha256(binding),model_identity=identity,
        pipeline_sha256=pipeline,source_inventory=prepared_rows,mask_manifest_sha256=sha256_file(revision_path))
    atomic_json(plan_path,plan)
    return plan


def clip_surface_frame(candidate: str, row: dict, pipeline: str):
    from darsia.fluidflower.tools.train_ffac_gas_flatfield_variants import _read_frozen_map
    root = Path(candidate)
    run,index = row["run"],int(row["frame_index"])
    path = root / "top_surface_frames" / run / f"frame_{index:03d}.json"
    binding = dict(run=run,frame_index=index,pipeline_sha256=pipeline,mask=row["mask"],
                   parent_phase_map=row["parent_phase_map"],parent_labels_map=row["parent_labels_map"],
                   shadow_native_manifest_sha256=row["shadow_native_manifest_sha256"])
    if path.exists():
        receipt = read_json(path)
        if receipt["binding"] != binding or receipt.get("complete") is not True:
            raise ValueError("Surface frame resume differs from its completed binding")
        for asset in [*receipt["maps"].values(),receipt["labels_map"],receipt["labels_archive"],
                      binding["mask"],binding["parent_phase_map"],binding["parent_labels_map"],row["revised_photo_mask"]]:
            _bound_asset(asset)
        return dict(run=run,frame_index=index,receipt_path=str(path),receipt_sha256=sha256_file(path))
    before,rgba = _read_frozen_map(_bound_asset(row["parent_phase_map"]),row["content_box_xywh"])
    with Image.open(_bound_asset(row["parent_labels_map"])) as image:
        parent_labels = np.asarray(image).copy()
    if not np.array_equal(parent_labels,before):
        raise ValueError("Parent native proof differs from its saved Studio phase decisions")
    masks = read_surface_mask(row["mask"],shape=before.shape)
    keep,photo = masks["top_keep"],masks["photo_sand_revised"]
    from darsia.fluidflower.phase_photo_sand_top_runtime import clip_approved_top_domain
    after,_ = clip_approved_top_domain(before,keep,masks["photo_sand_original"],photo)
    x,y,width,height = row["content_box_xywh"]
    display = np.zeros_like(rgba)
    display[y:y+height,x:x+width] = rgba[y:y+height,x:x+width]
    local = display[y:y+height,x:x+width]
    local[~keep] = 0
    cyan_suppression = keep & (before==255) & ~photo
    local[cyan_suppression] = 0
    # The renderer changes only labels1/2. Retained phase decisions keep their
    # palette even outside photo sand; unresolved cyan has explicit support.
    maps = write_phase_layers(root / "validation/run_packages" / run / "review_assets",run,index,pipeline,
                              display,after,row["content_box_xywh"])
    encoded = io.BytesIO()
    Image.fromarray(after,"L").save(encoded,format="PNG",compress_level=1)
    proof = _write_bytes(root / "production_labels" / run / f"frame_{index:03d}.png",encoded.getvalue())
    encoded = io.BytesIO()
    np.savez_compressed(encoded,labels=after)
    archive = _write_bytes(root / "production_labels" / run / f"frame_{index:03d}.npz",encoded.getvalue())
    with Image.open(proof["path"]) as image:
        saved = np.asarray(image).copy()
    with Image.open(maps["combined"]["path"]) as image:
        saved_rgba = np.asarray(image).copy()[y:y+height,x:x+width]
    if not np.array_equal(saved,after) or np.any(saved_rgba[~keep,...,3]) or np.any(saved[~keep]!=255):
        raise ValueError("Saved surface clipping differs from its native labels or transparent boundary")
    before_rgba=rgba[y:y+height,x:x+width]
    changed_rgba=np.any(saved_rgba!=before_rgba,axis=-1)
    if np.any(keep & changed_rgba & ~cyan_suppression):
        raise ValueError("Saved surface clipping changed retained Studio phase colours")
    parent_root = Path(row["parent_phase_map"]["path"])
    while not (parent_root / "summary.json").exists():
        parent_root = parent_root.parent
        if parent_root == parent_root.parent:
            raise ValueError("Parent surface frame has no immutable model root")
    parent_receipt = read_json(parent_root / "production_frames" / run / f"frame_{index:03d}.json")
    area = float(parent_receipt["production_audit"]["pixel_area_mm2"])
    before_co2,after_co2 = np.isin(before,(1,2)),np.isin(saved,(1,2))
    before_gas,after_gas = before==2,saved==2
    qa = dict(run=run,frame_index=index,elapsed_h=float(row["elapsed_h"]),pixel_area_mm2=area,
        before_co2_pixels=int(before_co2.sum()),after_co2_pixels=int(after_co2.sum()),
        before_gas_pixels=int(before_gas.sum()),after_gas_pixels=int(after_gas.sum()),
        inside_co2_changed_pixels=int(np.count_nonzero(keep & (before_co2!=after_co2))),
        inside_gas_changed_pixels=int(np.count_nonzero(keep & (before_gas!=after_gas))),
        inside_labels_changed_pixels=int(np.count_nonzero(keep & (before!=saved))),
        inside_rgba_changed_pixels=int(np.count_nonzero(keep & changed_rgba)),
        inside_unresolved_cyan_suppressed_pixels=int(np.count_nonzero(cyan_suppression & changed_rgba)),
        inside_rgba_exact_except_unresolved_cyan=True,
        above_co2_after=int(np.count_nonzero(~keep & after_co2)),above_gas_after=int(np.count_nonzero(~keep & after_gas)),
        above_phase_label255_all=bool(np.all(saved[~keep]==255)),above_rgba_transparent_all=bool(not np.any(saved_rgba[~keep,...,3])),
        inside_co2_sha256_before=mask_sha256(before_co2 & keep),inside_co2_sha256_after=mask_sha256(after_co2 & keep),
        inside_gas_sha256_before=mask_sha256(before_gas & keep),inside_gas_sha256_after=mask_sha256(after_gas & keep),
        removed_top_pixels=int(np.count_nonzero(~keep)),
        photo_sand_removed_pixels=int(np.count_nonzero(masks["photo_sand_original"] & ~keep)),
        revised_photo_sand_exact=True,
        photographic_pixel_refs_unchanged=True,native_manifest_path=row["shadow_native_manifest_path"],
        native_manifest_sha256=row["shadow_native_manifest_sha256"],parent_native_manifest_sha256=_asset(row,"native_manifest")["sha256"],
        map_path=proof["path"],map_sha256=proof["sha256"],studio_map_path=maps["combined"]["path"],studio_map_sha256=maps["combined"]["sha256"],
        parent_map_path=row["parent_phase_map"]["path"],parent_map_sha256=row["parent_phase_map"]["sha256"],
        parent_labels_map_path=row["parent_labels_map"]["path"],parent_labels_map_sha256=row["parent_labels_map"]["sha256"],
        mask_path=row["mask"]["path"],mask_sha256=row["mask"]["sha256"],
        photo_sand_mask_path=row["revised_photo_mask"]["path"],photo_sand_mask_sha256=row["revised_photo_mask"]["sha256"],
        content_box_xywh=row["content_box_xywh"],canvas_wh=row["canvas_wh"])
    for phase in ("co2","gas"):
        for time in ("before","after"):
            qa[f"{time}_{phase}_area_mm2"] = qa[f"{time}_{phase}_pixels"]*area
    if any(qa[key] for key in ("inside_co2_changed_pixels","inside_gas_changed_pixels","inside_labels_changed_pixels","above_co2_after","above_gas_after")):
        raise ValueError("Surface clipping changed native labels below the accepted boundary")
    receipt = dict(contract=CONTRACT,complete=True,binding=binding,maps=maps,labels_map=proof,labels_archive=archive,qa=qa,
                   model_inference=False,training=False,component_filtering=False,photo_pixels_changed=False)
    atomic_json(path,receipt)
    return dict(run=run,frame_index=index,receipt_path=str(path),receipt_sha256=sha256_file(path))


def bind_surface_packages(root,plan):
    from darsia.fluidflower.phase_runtime import load_physics_profile,physics_runtime_identity
    from darsia.fluidflower.tools.phase_reference_readiness import defer_validation_references
    parent = Path(plan["binding"]["parent"])
    loaded_profile = load_physics_profile(root)
    runtime = physics_runtime_identity(loaded_profile)
    profile = {key:deepcopy(value) for key,value in loaded_profile.items() if not key.startswith("_")}
    jobs = read_json(parent / "validation/postflight_manifest.json")["jobs"]
    packages = []
    for old in jobs:
        run = old["run"]
        package = root / "validation/run_packages" / run
        job = deepcopy(old)
        job.update(candidate_id=root.name+"__ffac__"+run+"__surface_"+plan["pipeline_sha256"][:12],
            candidate_folder=str(root),review_output_dir=str(root / "source_alias"),analysis_source_review_root=str(root / "source_alias"),
            phase_map_output_dir=str(package / "review_assets"),params_sha256=plan["pipeline_sha256"],
            optical_model_sha256=plan["model_identity"][1],physics_runtime=runtime,output_mode="review",
            phase_model_only=True,raw_phase_map_output_dir=None,numeric_phase_output_dir=None,presence_diagnostics_dir=None)
        job["quality_flags"] = dict(job["quality_flags"],approved_g3_top_surface=True,mass_analysis_allowed=False,scientific_acceptance=False)
        source_sha = sha256_file(root / "review_source" / run / "postflight_manifest.json")
        common = dict(schema=1,format="darsia-phase-validation-campaign-v1",campaign_id="phase-validation-"+root.name,
            model_id=plan["model_identity"][0],model_sha256=plan["model_identity"][1],model_root=str(root),
            source_manifest_sha256=source_sha,pipeline_sha256=plan["pipeline_sha256"],physics_runtime=runtime,physics_profile=profile,
            output_mode="review",numeric_outputs=False,phase_display_policy="approved-g3-photo-surface-clip-v1")
        atomic_json(package / "postflight_manifest.json",dict(common,jobs=[job],source_campaign_root=str(root / "review_source" / run)))
        atomic_json(package / "validation_binding.json",dict(common,combined_root=str(package),ffac_root=str(package),ffum_root=str(package),
            selected_candidates={run:job["candidate_id"]},run_count=1,completed_utc=plan["created_utc"]))
        defer_validation_references(package)
        packages.append(package)
    return packages


def seal_surface_candidate(root,plan,completed):
    expected = {(row["run"],int(row["frame_index"])) for row in plan["source_inventory"]}
    if len(completed)!=864 or {(row["run"],row["frame_index"]) for row in completed}!=expected:
        raise ValueError("Surface candidate cannot publish partial or overlapping frame coverage")
    receipts=[]
    for item in sorted(completed,key=lambda value:(value["run"],value["frame_index"])):
        receipt = read_json(item["receipt_path"])
        if sha256_file(item["receipt_path"])!=item["receipt_sha256"] or receipt.get("complete") is not True:
            raise ValueError("Surface clipping receipt changed")
        for asset in [*receipt["maps"].values(),receipt["labels_map"],receipt["labels_archive"]]:
            _bound_asset(asset)
        receipts.append(receipt)
    header = dict(configuration=CONFIGURATION,spec_sha256=SPEC_SHA256,g3_sha256=G3_SHA256,
        regional_refit_sha256=REFIT_SHA256,co2_sha256=CO2_SHA256,parent_model_identity=plan["binding"]["parent_model_identity"],
        model_identity=plan["model_identity"],mask_manifest_sha256=plan["mask_manifest_sha256"],run_count=43,frame_count=864)
    qa_path = root / "production_qa/top_surface_qa.json"
    atomic_json(qa_path,dict(contract=QA_CONTRACT,**header,all_pass=True,rows=[value["qa"] for value in receipts]))
    packages = bind_surface_packages(root,plan)
    assemble_saved_candidate(root,packages,plan)
    sealed = dict(contract=CONTRACT,**header,complete=True,qa=dict(path=str(qa_path),sha256=sha256_file(qa_path)),
        parent_registration=plan["binding"]["parent_registration"],source_plan_sha256=sha256_file(root / "top_surface_candidate_plan.json"),
        phase_profile_sha256=sha256_file(root / "phase_optical_regions.json"),studio_preparation_sha256=sha256_file(root / "studio_preparation.json"),
        frames=sorted(completed,key=lambda value:(value["run"],value["frame_index"])),
        scientific_acceptance=False,mass_eligible=False,automatic_activation=False,model_inference=False,training=False)
    atomic_json(root / "top_surface_candidate_receipt.json",sealed)
    return sealed


def build_surface_candidate(parent,output,geometry_manifest,accepted_geometry_sha256,*,workers=8,threads=1,resume=False,pilot=False):
    if not 1<=workers<=12 or threads not in (1,2) or workers*threads>max(1,(os.cpu_count() or 1)-4):
        raise ValueError("Surface candidate CPU allocation must leave cores free")
    root=Path(output).resolve()
    below_normal_priority()
    plan=prepare_surface_candidate(parent,root,geometry_manifest,accepted_geometry_sha256,resume=resume,workers=min(8,workers))
    rows=plan["source_inventory"]
    if pilot:
        rows=[row for row in rows if row["run"]=="ac28" and row["frame_index"] in (0,8)]
    with JobLease(root,plan["job_sha256"]):
        below_normal_priority()
        log=ProgressLog(root)
        completed=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initializer,initargs=(threads,)) as pool:
            pending={pool.submit(clip_surface_frame,str(root),row,plan["pipeline_sha256"]):row for row in rows}
            for future in as_completed(pending):
                result=future.result()
                assigned=pending[future]
                if (result["run"],result["frame_index"])!=(assigned["run"],assigned["frame_index"]):
                    raise ValueError("Surface worker returned another frame")
                completed.append(result)
                log.event("surface_frame",run=result["run"],frame_index=result["frame_index"],completed_frames=len(completed),total_frames=len(rows))
        if pilot:
            receipt=dict(contract=CONTRACT,pilot=True,complete=True,model_identity=plan["model_identity"],
                frames=sorted(completed,key=lambda value:(value["run"],value["frame_index"])))
            atomic_json(root / "top_surface_pilot_receipt.json",receipt)
            return receipt
        return seal_surface_candidate(root,plan,completed)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("parent","output","geometry-manifest"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--accepted-geometry-sha256",required=True)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--pilot",action="store_true")
    args=parser.parse_args(argv)
    build_surface_candidate(args.parent,args.output,args.geometry_manifest,args.accepted_geometry_sha256,
        workers=args.workers,threads=args.threads,resume=args.resume,pilot=args.pilot)


if __name__=="__main__":
    main()
