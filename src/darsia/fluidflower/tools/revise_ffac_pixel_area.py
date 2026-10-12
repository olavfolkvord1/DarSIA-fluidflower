"""Create a measured-width, area-only FF_AC revision of sealed saved maps.

No inference, model fitting or annotation writes. All phase PNG copies retain
their producer's exact SHA. Paths and measurements are explicit CLI inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from darsia.fluidflower.project import sha256_file

CONTRACT = "ffac-measured-pixel-area-revision-v1"
KIND = "approved_g3_area_revision"
WARNING = ("Studio review candidate: approved G3; measured rig width; physical scale approved. "
           "Bottom strip blocked for mass; AC30/49/54 withheld pending blind check. "
           "Not approved for mass analysis or deployment.")
HIDDEN = frozenset(("ac30", "ac49", "ac54"))
FILES = {
    "model_summary_sha256": "summary.json", "model_request_sha256": "training_request.json",
    "phase_profile_sha256": "phase_rig_camera.json",
    "native_source_manifest_sha256": "review_source/postflight_manifest.json",
    "validation_binding_sha256": "validation/validation_binding.json",
    "postflight_manifest_sha256": "validation/postflight_manifest.json",
    "production_receipt_sha256": "physical_scale_receipt.json",
    "source_plan_sha256": "physical_scale_plan.json",
    "bottom_policy_sha256": "rig_bottom_review_policy.json",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def asset(path):
    return dict(path=str(path), sha256=sha256_file(path))


def checked(value):
    path = Path(value["path"])
    if sha256_file(path) != value["sha256"]:
        raise ValueError(f"Source artifact changed: {path}")
    return path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def scale_area_fields(value, factor, *, active=False):
    """Scale metric areas; historical fixed areas, pixel counts and ratios stay literal."""
    if isinstance(value, dict):
        return {key: (deepcopy(item) if key == "old_fixed_cm2" else
                      scale_area_fields(item, factor, active=active or "cm2" in key))
                for key, item in value.items()}
    if isinstance(value, list):
        return [scale_area_fields(item, factor, active=active) for item in value]
    if active and type(value) in (float, int):
        return value * factor
    return deepcopy(value)


def scale_descriptor(old, new_area, measurement):
    result = deepcopy(old)
    result["area_map"] = new_area
    result["physical_scale_approved"] = True
    result["physical_scale_revision"] = deepcopy(measurement)
    result["mass_eligible"] = False
    params = scale_area_fields(result.get("parameters", {}), measurement["area_factor"])
    if "H_frame_native_to_metric_pixel" in params:
        matrix = np.asarray(params["H_frame_native_to_metric_pixel"], dtype=np.float64)
        matrix[:2] *= measurement["linear_factor"]
        params["H_frame_native_to_metric_pixel"] = matrix.tolist()
    params["physical_scale_approved"] = True
    params["physical_scale_revision"] = deepcopy(measurement)
    result["parameters"] = params
    return result


def revise_area(source, destination, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Area scale must be positive and finite")
    source_path = checked(source)
    with np.load(source_path, allow_pickle=False) as archive:
        if set(archive.files) != {"pixel_area_mm2"}:
            raise ValueError("Area source has unexpected arrays")
        original = archive["pixel_area_mm2"]
    if original.dtype != np.float32 or original.ndim != 2 or not np.isfinite(original).all() or (original <= 0).any():
        raise ValueError("Area source must be a positive finite float32 raster")
    corrected = (original.astype(np.float64) * factor).astype(np.float32)
    expected = original.astype(np.float64) * factor
    relative = float(np.max(np.abs(corrected.astype(np.float64) / expected - 1)))
    if relative > 1e-7:
        raise ValueError("Scaled area exceeds float32 rounding tolerance")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, pixel_area_mm2=corrected)
    result = dict(asset(destination), array_key="pixel_area_mm2")
    return result, dict(source=source, shape_hw=list(original.shape), dtype="float32",
                       maximum_relative_rounding_error=relative, all_pixels_scaled=True,
                       minimum_before_mm2=float(original.min()), maximum_before_mm2=float(original.max()),
                       minimum_after_mm2=float(corrected.min()), maximum_after_mm2=float(corrected.max()))


def copy_exact(source, destination):
    source_path = checked(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=4 << 20)
    result = asset(destination)
    if result["sha256"] != source["sha256"]:
        raise ValueError("Byte-preserved copy failed SHA verification")
    return result


def worker_init():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def revise_run(parent, root, run, job, frame_assets, measurement, pipeline):
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    parent, root = Path(parent), Path(root)
    source_root = Path(job["review_output_dir"])
    native_path = source_root / "native_frames" / run / "manifest.json"
    native = read(native_path)
    inputs = read(checked(native[meta.SOURCE_KEY]["input_manifest"]))
    if len(native["frame_sources"]) != len(inputs["frames"]) or len(frame_assets) != len(inputs["frames"]):
        raise ValueError("Frame receipts and exact sources differ in count")
    frames, rows = [], []
    for index, (record, binding, old_frame) in enumerate(zip(native["frame_sources"], inputs["frames"], frame_assets)):
        old_receipt = read(checked(old_frame["receipt"]))
        if (record["frame_index"] != index or old_frame["frame_index"] != index
                or old_receipt["binding"]["source"]["source_sha256"] != record["source_sha256"]):
            raise ValueError("Source frame receipt no longer matches its camera source")
        descriptor = old_receipt["rig_domain_revision"]
        if descriptor != record["rig_domain_revision"] or descriptor != binding["rig_domain_revision"]:
            raise ValueError("Area revision differs between producer and native source")
        new_area, numerical = revise_area(descriptor["area_map"], root / "rig_camera_revision/context" / run /
                                          f"frame_{index:03d}.area.npz", measurement["area_factor"])
        new_descriptor = scale_descriptor(descriptor, new_area, measurement)
        record["rig_domain_revision"] = deepcopy(new_descriptor)
        binding["rig_domain_revision"] = deepcopy(new_descriptor)
        maps = {}
        for mode, old_map in old_receipt["maps"].items():
            relative = Path(old_map["path"]).relative_to(parent)
            parts = ["A_" + pipeline[:16] if part.startswith("A_") and len(part) == 18 else part
                     for part in relative.parts]
            maps[mode] = copy_exact(old_map, root.joinpath(*parts))
        labels = copy_exact(old_receipt["labels_map"], root / "production_labels" / run / f"frame_{index:03d}.png")
        qa = scale_area_fields(old_receipt["qa"], measurement["area_factor"])
        qa.update(area_map_sha256=new_area["sha256"], map_path=labels["path"], map_sha256=labels["sha256"],
                  studio_map_path=maps["combined"]["path"], studio_map_sha256=maps["combined"]["sha256"],
                  physical_scale_approved=True)
        new_receipt = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), run=run, frame_index=index,
            source_sha256=record["source_sha256"], elapsed_h=record["elapsed_h"],
            source_producer_receipt=old_frame["receipt"], maps=maps, labels_map=labels,
            rig_domain_revision=new_descriptor, qa=qa, numerical_scale_qa=numerical,
            measurement=measurement, maps_byte_identical=True, labels_changed=False,
            inference_performed=False, training=False, mass_eligible=False, bottom_mass_allowed=False)
        destination = root / "rig_camera_frames" / run / f"frame_{index:03d}.json"
        write(destination, new_receipt)
        frames.append(dict(run=run, frame_index=index, receipt=asset(destination)))
        rows.append(qa)
    old_ref = native["reference"]["rig_domain_revision"]
    ref_area, _ = revise_area(old_ref["area_map"], root / "rig_camera_revision/context" / run / "reference.area.npz",
                              measurement["area_factor"])
    reference = scale_descriptor(old_ref, ref_area, measurement)
    native["reference"]["rig_domain_revision"] = reference
    inputs["reference"]["rig_domain_revision"] = deepcopy(reference)
    input_path = root / "source_alias/source_bindings" / (run + ".json")
    write(input_path, inputs)
    rebuilt = meta.build_native_metadata(read(checked(inputs["parent_native_manifest"])),
        read(checked(inputs["camera_timeline_run"])), native["frame_sources"], inputs["frames"], inputs["reference"],
        input_manifest=asset(input_path))
    destination = root / "source_alias/native_frames" / run / "manifest.json"
    write(destination, rebuilt)
    native_asset = asset(destination)
    for row in rows:
        row["native_manifest_sha256"] = native_asset["sha256"]
    result = dict(run=run, complete=True, native_manifest=native_asset, input_manifest=asset(input_path),
                  frames=inputs["frames"], reference=inputs["reference"], frame_count=len(frames),
                  receipts=frames, qa_rows=rows)
    write(root / "area_revision_runs" / (run + ".json"), result)
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def report_rows(root, rows, old_rows, measurement):
    old_lookup = {(r["run"], r["frame_index"]): r for r in old_rows}
    phases, strips = [], []
    for row in rows:
        if row["run"] in HIDDEN:
            continue
        before = old_lookup[(row["run"], row["frame_index"])]
        common = {key: row[key] for key in ("run", "frame_index", "elapsed_h", "source_sha256")}
        phases.append(dict(common, physical_scale_approved=True, bottom_mass_allowed=False,
            **{f"{phase}_{which}_cm2": value["area_report"]["new_cm2"][phase]
               for phase in ("co2", "gas", "aqueous", "resolved")
               for which, value in (("before", before), ("after", row))}))
        for side, data in row["added_strip_by_side"]["sides"].items():
            original = before["added_strip_by_side"]["sides"][side]
            strips.append(dict(common, side=side, bottom_mass_allowed=False,
                light_field_group="AC17-38 extrapolated" if int(row["run"][2:]) <= 38 else "AC39-61 observed",
                **{f"{phase}_{which}_cm2": value[phase + "_cm2"] for phase in ("co2", "gas", "aqueous")
                   for which, value in (("before", original), ("after", data))}))
    folder = root / "area_reports"
    folder.mkdir()
    for name, values in (("phase_areas_public.csv", phases), ("strip_areas_public.csv", strips)):
        with (folder / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    write(folder / "report_receipt.json", dict(contract=CONTRACT, measurement=measurement,
        public_runs=40, excluded_blind_runs=sorted(HIDDEN), bottom_mass_allowed=False,
        timing_and_first_nonzero_unchanged=True, cohort_cv_and_lamp_ratios_unchanged_by_uniform_factor=True,
        historical_fixed_area_report_preserved=True, outputs=[asset(folder / name) for name in
            ("phase_areas_public.csv", "strip_areas_public.csv")]))


def build(parent, root, measurement, *, workers=24, vertical_report=None):
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    from darsia.fluidflower.phase_ffac_rig_camera_runtime import load_profile
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
    from darsia.fluidflower.tools.register_track1_studio_candidate import registration_path, registration_warning
    from darsia.fluidflower.tools.run_ffac_rig_camera_inference import _copy_heads_and_request
    from darsia.fluidflower.tools.build_ffac_g3_approved_candidate import assemble_saved_candidate
    from darsia.fluidflower.tools.phase_reference_readiness import defer_validation_references
    parent, root = Path(parent).resolve(), Path(root).resolve()
    if root.exists() or root == parent or root.is_relative_to(parent):
        raise FileExistsError("Area revisions require a new sibling candidate directory")
    parent_reg_path = registration_path(parent.name)
    parent_reg = read(parent_reg_path)
    if parent_reg.get("registration_kind") != "approved_g3_rig_camera" or "QA FAILED" in registration_warning(parent):
        raise ValueError("Area revision requires its intact completed rig/camera parent")
    parent_seal = read(parent / "rig_camera_candidate_receipt.json")
    if not parent_seal["complete"] or (parent_seal["run_count"], parent_seal["frame_count"]) != (43, 864):
        raise ValueError("Parent is incomplete")
    old_qa = read(checked(parent_seal["qa"]))
    jobs = _source_jobs(parent / "validation")
    if len(jobs) != 43:
        raise ValueError("Parent requires all43 source runs")
    root.mkdir(parents=True)
    write(root / "physical_scale_measurement.json", measurement)
    binding = dict(contract=CONTRACT, parent_root=str(parent), parent_identity=list(_model_identity(parent)),
        parent_registration=asset(parent_reg_path), parent_seal=asset(parent / "rig_camera_candidate_receipt.json"),
        measurement=asset(root / "physical_scale_measurement.json"), code=asset(Path(__file__)),
        output_root=str(root), cpu_only=True, inference=False, training=False,
        maps_changed=False, labels_changed=False, bottom_mass_allowed=False, blind_hold_runs=sorted(HIDDEN))
    if vertical_report:
        binding["vertical_check"] = copy_exact(asset(vertical_report), root / "vertical_scale_check.json")
    pipeline = digest(binding)
    _copy_heads_and_request(parent, root)
    summary = read(parent / "summary.json")
    summary.update(model_id=root.name, campaign_id=root.name, created_utc=utc_now(),
        display_name="FF_AC · G3 gas · Rig domain · Camera clock · Measured width 900 mm",
        inherited_area_candidate_id=parent.name, physical_scale_approved=True,
        physical_scale_revision=asset(root / "physical_scale_measurement.json"),
        regional_training_performed=False, optical_training_performed=False,
        training_performed_in_this_revision=False, inference_only_rerun=False,
        inference_performed_in_this_revision=False, raw_inference_performed=False,
        area_only_revision=True, mass_eligible=False, scientific_acceptance=False)
    write(root / "summary.json", summary)
    policy = read(parent / "rig_bottom_review_policy.json")
    policy.update(candidate_id=root.name)
    write(root / "rig_bottom_review_policy.json", policy)
    frames_by_run = {job["run"]: sorted([row for row in parent_seal["frames"] if row["run"] == job["run"]],
                                      key=lambda row: row["frame_index"]) for job in jobs}
    with ProcessPoolExecutor(max_workers=min(workers, 24, len(jobs)),
                             mp_context=multiprocessing.get_context("spawn"), initializer=worker_init) as pool:
        futures = {pool.submit(revise_run, str(parent), str(root), job["run"], job,
            frames_by_run[job["run"]], measurement, pipeline): job["run"] for job in jobs}
        sources = []
        for future in as_completed(futures):
            sources.append(future.result())
            print(json.dumps(dict(event="area_run_complete", completed_runs=len(sources), total_runs=43)), flush=True)
    sources.sort(key=lambda row: row["run"])
    # Keep geometric photographic evidence exact; only its absolute scale status is superseded.
    original_profile = read(parent / "phase_rig_camera.json")
    original_runtime = read(checked(original_profile["rig_domain_revision"]["manifest"]))
    original_acceptance = read(checked(original_runtime["geometry_acceptance"]))
    acceptance = deepcopy(original_acceptance)
    acceptance.update(physical_scale_approved=True, width_measurement=asset(root / "physical_scale_measurement.json"),
        inherited_geometry_acceptance=original_runtime["geometry_acceptance"], vertical_scale_independently_verified=False,
        absolute_scale_supersedes_old_unmeasured_qualifications=True)
    write(root / "rig_camera_revision/geometry_acceptance.json", acceptance)
    runtime_manifest = dict(original_runtime, sources=[{k: v for k, v in source.items() if k not in ("receipts", "qa_rows")}
        for source in sources], geometry_acceptance=asset(root / "rig_camera_revision/geometry_acceptance.json"),
        physical_scale_revision=asset(root / "physical_scale_measurement.json"))
    runtime_path = root / "rig_camera_revision/runtime_manifest.json"
    write(runtime_path, runtime_manifest)
    profile = deepcopy(original_profile)
    profile["rig_domain_revision"]["manifest"] = asset(runtime_path)
    profile["physical_scale_revision"] = asset(root / "physical_scale_measurement.json")
    profile["physical_scale_approved"] = True
    write(root / "phase_rig_camera.json", profile)
    identity = list(_model_identity(root))
    plan = dict(binding=binding, model_identity=identity, pipeline_sha256=pipeline, created_utc=utc_now())
    write(root / "physical_scale_plan.json", plan)
    new_sources = {row["run"]: row for row in sources}
    source_manifest = read(parent / "review_source/postflight_manifest.json")
    source_jobs = []
    for old in source_manifest["jobs"]:
        job = deepcopy(old); run = job["run"]
        job.update(review_output_dir=str(root / "source_alias"), analysis_source_review_root=str(root / "source_alias"),
                   native_manifest_sha256=new_sources[run]["native_manifest"]["sha256"])
        one = read(parent / "review_source" / run / "postflight_manifest.json")
        write(root / "review_source" / run / "postflight_manifest.json", dict(one, jobs=[job]))
        source_jobs.append(job)
    write(root / "review_source/postflight_manifest.json", dict(source_manifest, jobs=source_jobs))
    loaded = load_physics_profile(root)
    runtime, public = physics_runtime_identity(loaded), meta.public_profile(loaded)
    packages = []
    for old in jobs:
        run = old["run"]; package = root / "validation/run_packages" / run
        job = deepcopy(old)
        job.update(candidate_id=root.name + "__ffac__" + run + "__scale_" + pipeline[:12],
            candidate_folder=str(root), review_output_dir=str(root / "source_alias"),
            analysis_source_review_root=str(root / "source_alias"), phase_map_output_dir=str(package / "review_assets"),
            params_sha256=pipeline, optical_model_sha256=identity[1], physics_runtime=runtime,
            native_manifest_sha256=new_sources[run]["native_manifest"]["sha256"], physical_scale_qa_pass=True,
            physical_scale_warning="Width measured by Olav; 490 mm height not independently verified")
        job["quality_flags"] = dict(job.get("quality_flags") or {}, physical_scale_approved=True,
            approved_g3_downstream=True, mass_analysis_allowed=False, scientific_acceptance=False)
        common = dict(schema=1, format="darsia-phase-validation-campaign-v1", campaign_id="phase-validation-" + root.name,
            model_id=identity[0], model_sha256=identity[1], model_root=str(root),
            source_manifest_sha256=sha256_file(root / "review_source" / run / "postflight_manifest.json"),
            pipeline_sha256=pipeline, physics_runtime=runtime, physics_profile=public,
            output_mode="review", numeric_outputs=False, phase_display_policy=meta.REVISION_CONTRACT)
        write(package / "postflight_manifest.json", dict(common, jobs=[job], source_campaign_root=str(root / "review_source" / run)))
        write(package / "validation_binding.json", dict(common, combined_root=str(package), ffac_root=str(package),
            ffum_root=str(package), selected_candidates={run:job["candidate_id"]}, run_count=1, completed_utc=utc_now()))
        defer_validation_references(package); packages.append(package)
    assemble_saved_candidate(root, packages, plan)
    rows = [row for source in sources for row in source["qa_rows"]]
    frame_receipts = [row for source in sources for row in source["receipts"]]
    report_rows(root, rows, old_qa["rows"], measurement)
    write(root / "production_qa/area_revision_qa.json", dict(contract=CONTRACT, all_pass=True,
        run_count=43, frame_count=864, rows=rows, model_identity=identity, physical_scale_approved=True,
        maps_byte_identical=True, labels_changed=False, mass_eligible=False, bottom_mass_allowed=False))
    seal = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), model_identity=identity,
        pipeline_sha256=pipeline, source_plan_sha256=sha256_file(root / "physical_scale_plan.json"),
        run_count=43, frame_count=864, frames=frame_receipts, qa=asset(root / "production_qa/area_revision_qa.json"),
        measurement=asset(root / "physical_scale_measurement.json"), physical_scale_approved=True,
        parent_seal=binding["parent_seal"], parent_registration=binding["parent_registration"],
        maps_byte_identical=True, labels_changed=False, inference_performed=False, training=False,
        bottom_mass_allowed=False, mass_eligible=False, blind_hold_runs=sorted(HIDDEN))
    write(root / "physical_scale_receipt.json", seal)
    value = inspect_candidate(root)
    target = registration_path(root.name)
    write(target, value)
    print(json.dumps(dict(event="area_revision_complete", candidate=str(root), registration=str(target), frames=864)), flush=True)
    return seal


def inspect_candidate(root):
    from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
    from darsia.fluidflower.phase_evidence_audit import verified_review_binding
    from darsia.fluidflower.tools.register_track1_studio_candidate import FORMAT, registration_warning
    root = Path(root)
    seal, plan = read(root / "physical_scale_receipt.json"), read(root / "physical_scale_plan.json")
    identity = list(_model_identity(root))
    if (not seal.get("complete") or seal.get("model_identity") != identity or plan["model_identity"] != identity
            or seal["source_plan_sha256"] != sha256_file(root / "physical_scale_plan.json")
            or (seal.get("run_count"), seal.get("frame_count")) != (43, 864)
            or seal.get("physical_scale_approved") is not True or seal.get("maps_byte_identical") is not True
            or seal.get("bottom_mass_allowed") is not False or seal.get("mass_eligible") is not False
            or set(seal.get("blind_hold_runs", [])) != HIDDEN):
        raise ValueError("Area revision seal is invalid")
    parent_reg = read(checked(seal["parent_registration"]))
    parent = Path(parent_reg["model_root"])
    if "QA FAILED" in registration_warning(parent):
        raise ValueError("Parent registration changed")
    policy = read(root / "rig_bottom_review_policy.json")
    if policy != dict(read(parent / "rig_bottom_review_policy.json"), candidate_id=root.name):
        raise ValueError("Area revision changed the blind/mass policy")
    from darsia.fluidflower.phase_g3_production import G3_SHA256, REFIT_SHA256, CO2_SHA256
    for name, expected in (("models/g3/gas_head.json", G3_SHA256), ("regional_model.json", REFIT_SHA256),
                           ("models/generic/co2_presence_signal.json", CO2_SHA256)):
        if sha256_file(root / name) != expected:
            raise ValueError("Area revision changed a frozen model")
    verified_review_binding(root, saved_comparison=True)
    jobs = _source_jobs(root / "validation")
    if len(jobs) != 43:
        raise ValueError("Area revision lacks all43 review runs")
    qa = read(checked(seal["qa"]))
    if len(qa["rows"]) != 864 or len(seal["frames"]) != 864:
        raise ValueError("Area revision lacks all864 frames")
    old_seal = read(checked(seal["parent_seal"]))
    old_frames = {(r["run"], r["frame_index"]):r for r in old_seal["frames"]}
    new_frames = {(r["run"], r["frame_index"]):r for r in seal["frames"]}
    if len(new_frames) != 864 or set(old_frames) != set(new_frames):
        raise ValueError("Area revision frame keys differ from parent")
    measurement = read(checked(seal["measurement"]))
    for key, frame in new_frames.items():
        revised = read(checked(frame["receipt"]))
        old = read(checked(old_frames[key]["receipt"]))
        if (revised["source_producer_receipt"] != old_frames[key]["receipt"]
                or revised["source_sha256"] != old["binding"]["source"]["source_sha256"]
                or revised["labels_map"]["sha256"] != old["labels_map"]["sha256"]
                or {k:v["sha256"] for k,v in revised["maps"].items()} != {k:v["sha256"] for k,v in old["maps"].items()}
                or revised["rig_domain_revision"]["physical_scale_approved"] is not True
                or revised["measurement"] != measurement
                or revised["inference_performed"] is not False):
            raise ValueError("Area revision changed maps or exact source lineage")
    return dict(format=FORMAT, registration_kind=KIND,
        created_utc=utc_now(), model_root=str(root), model_id=identity[0], model_sha256=identity[1],
        series="ffac", publication_scope="studio_validation_only", qa_status="passed_measured_pixel_area",
        run_count=43, frame_count=864, physical_scale_approved=True, bottom_mass_allowed=False,
        map_sha256={f"{r['run']}/{r['frame_index']:03d}":r["studio_map_sha256"] for r in qa["rows"]},
        native_run_manifest_sha256={j["run"]:j["native_manifest_sha256"] for j in jobs},
        scientific_acceptance=False, deployment_allowed=False, mass_eligible=False, automatic_activation=False,
        labels_changed=False, optical_training_performed=False,
        **{key:sha256_file(root / name) for key,name in FILES.items()})


def registered_warning(root, value):
    try:
        from darsia.fluidflower.phase_validation import _model_identity
        from darsia.fluidflower.tools.register_track1_studio_candidate import FORMAT
        root = Path(root)
        if (value.get("format") != FORMAT or value.get("registration_kind") != KIND
                or [value.get("model_id"), value.get("model_sha256")] != list(_model_identity(root))
                or (value.get("run_count"),value.get("frame_count")) != (43,864)
                or len(value.get("map_sha256",{})) != 864 or len(value.get("native_run_manifest_sha256",{})) != 43
                or value.get("physical_scale_approved") is not True or value.get("bottom_mass_allowed") is not False
                or any(value.get(k) is not False for k in ("mass_eligible", "deployment_allowed", "scientific_acceptance", "automatic_activation"))):
            raise ValueError("Area registration changed")
        for key, name in FILES.items():
            if sha256_file(root / name) != value.get(key):
                raise ValueError("Registered area metadata changed")
        seal = read(root / "physical_scale_receipt.json")
        checked(seal["parent_registration"]); checked(seal["parent_seal"]); checked(seal["measurement"])
        if not seal["complete"] or (seal["run_count"], seal["frame_count"]) != (43,864):
            raise ValueError("Area seal incomplete")
        checked(seal["qa"])
        return WARNING
    except (OSError, ValueError, KeyError, TypeError):
        return "QA FAILED: measured-area revision binding changed or is incomplete; do not use for mass analysis."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-factor", type=float, required=True)
    parser.add_argument("--area-factor", type=float, required=True)
    parser.add_argument("--width-mm", type=float, required=True)
    parser.add_argument("--measured-by", required=True)
    parser.add_argument("--measurement-date", required=True)
    parser.add_argument("--vertical-report", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if (args.workers < 1 or not np.isfinite([args.linear_factor,args.area_factor,args.width_mm]).all()
            or min(args.linear_factor,args.area_factor,args.width_mm) <= 0
            or not np.isclose(args.linear_factor**2,args.area_factor,rtol=2e-6)):
        raise ValueError("Invalid measured width or inconsistent linear/area factors")
    datetime.fromisoformat(args.measurement_date)
    measurement = dict(contract=CONTRACT, source="Olav's measurement" if args.measured_by == "Olav" else args.measured_by+"'s measurement",
        measured_by=args.measured_by, measurement_date=args.measurement_date, inner_width_mm=args.width_mm,
        linear_factor=args.linear_factor, area_factor=args.area_factor, physical_scale_approved=True,
        approval_basis="user-measured inner metal wall width; uniform isotropic scale",
        inner_height_config_mm=490., vertical_scale_independently_verified=False,
        nominal_historical_pixel_area_mm2_preserved=.027053, mass_eligible=False, bottom_mass_allowed=False)
    build(args.parent,args.output,measurement,workers=args.workers,vertical_report=args.vertical_report)


if __name__ == "__main__":
    main()
