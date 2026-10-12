"""Build a detached P1-final3 saved-label revision, without model execution.

Sealing creates the review packages. Registration is deliberately a separate
operation so an independent saved-file audit can run before publication.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import csv
import importlib
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower import phase_ffac_p1_final3 as rules
from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
from .refit_ffac_g3_downstream import (
    JobLease, atomic_json, canonical_sha256, copy_file_verified, read_json,
    sha256_file, utc_now,
)
from .run_ffac_rig_camera_candidate import artifact
from .build_ffac_p1_final_candidate import _mask, _overlay, _png, _write

CONTRACT = rules.CONTRACT
SPEC_SHA256 = rules.SPEC_SHA256
PLAN = "p1_final3_plan.json"
SEAL = "rig_camera_candidate_receipt.json"
QA = "production_qa/p1_final3_qa.json"
RUN_DIRECTORY = "p1_final3_runs"
MASK_DIRECTORIES = dict(A="below_sand_floor_masks", B="floor_connectivity_masks",
                       C="floor_contact_fill_v2", D="gas_envelope_masks")
OVERLAYS = dict(A="below_sand_floor", B="removed_floor_connectivity",
                C="floor_contact_fill_v2", D="removed_gas_envelope")
REMOVED_OVERLAY = "removed_contact_fill_b"
UNCHANGED = "source_time_raw_geometry_area_heads_delta_unchanged"


def code_binding():
    names = ("darsia.fluidflower.tools.build_ffac_p1_final3_candidate",
             "darsia.fluidflower.phase_ffac_p1_final3",
             "darsia.fluidflower.tools.build_ffac_p1_final_candidate",
             "darsia.fluidflower.phase_ffac_rig_camera_metadata")
    return {name: sha256_file(importlib.import_module(name).__file__) for name in names}


def prepare_candidate(parent, output, spec, *, resume=False):
    from darsia.fluidflower.phase_ffac_p1_final2_review import p1_final2_saved_review
    from darsia.fluidflower.phase_validation import _model_identity
    from .register_track1_studio_candidate import registration_path
    parent, root, spec = (Path(value).resolve() for value in (parent, output, spec))
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("P1-final3 requires a detached new candidate")
    if sha256_file(spec) != SPEC_SHA256:
        raise ValueError("Approved P1-final3 specification changed")
    checked = p1_final2_saved_review(parent)
    if checked["placement"] != "P1-final2":
        raise ValueError("Only the admitted complete P1-final2 parent is allowed")
    seal = read_json(parent / SEAL)
    if seal.get("complete") is not True or (seal.get("run_count"), seal.get("frame_count")) != (43, 864):
        raise ValueError("P1-final2 parent is not complete 43/864")
    inputs = dict(parent_seal=artifact(parent / SEAL),
                  parent_registration=artifact(registration_path(parent.name)),
                  spec=artifact(spec), code=code_binding(), rules=rules.RULE,
                  training=False, inference=False, annotations_read=False,
                  mass_eligible=False, bottom_mass_allowed=False, backend="cpu")
    binding = dict(parent=str(parent), heads=seal["heads"], inputs=inputs)
    if (root / PLAN).exists():
        plan = read_json(root / PLAN)
        if not resume or plan["binding"] != binding:
            raise ValueError("Resume changes the immutable P1-final3 binding")
        return plan
    if root.exists() or resume:
        raise ValueError("Use a new output folder, or exact --resume")
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("models", "regional_models"):
        for path in (parent / folder).rglob("*"):
            if path.is_file():
                copy_file_verified(path, root / path.relative_to(parent))
    for name in ("regional_model.json", "training_request.json", "expected_review_runs.json",
                 "physical_scale_measurement.json", "vertical_scale_check.json"):
        if (parent / name).is_file():
            copy_file_verified(parent / name, root / name)
    summary = read_json(parent / "summary.json")
    created = utc_now()
    date = created[:10].split("-")
    summary.update(model_id=root.name, campaign_id=root.name, created_utc=created,
        display_name=("FF_AC · G3-T4c P1 final 3 · Sand floor · Floor connectivity · "
                      "Contact fill ≤ 1.5 mm · Gas injection envelope · "
                      f"{date[2]}.{date[1]}.{date[0]}"),
        inherited_base_id=parent.name, training_performed_in_this_revision=False,
        optical_training_performed=False, regional_training_performed=False,
        co2_training_performed=False, mass_eligible=False, scientific_acceptance=False,
        automatic_activation=False, bottom_mass_allowed=False, blind_hold_runs=[],
        p1_final=False, p1_final2=False, p1_final3=True, p1_final3_rules=rules.RULE)
    atomic_json(root / "summary.json", summary)
    sources = []
    epochs = {}
    def read_parent_frame(frame):
        return frame, read_json(bound_path(frame["receipt"]))
    with ThreadPoolExecutor(max_workers=24, thread_name_prefix="p1-final3-bind") as pool:
        saved_frames = list(pool.map(read_parent_frame, seal["frames"]))
    for frame, saved in saved_frames:
        source = saved["binding"]["source"]
        if (saved.get("contract") != "ffac-g3-t4c-p1-final2-candidate-v1"
                or saved.get("complete") is not True
                or any(saved.get(key) != source.get(key)
                       for key in ("run", "frame_index", "elapsed_h", "source_sha256"))):
            raise ValueError("P1-final2 sealed frame lost its immutable source binding")
        sources.append({key: source[key] for key in ("run", "frame_index", "elapsed_h", "source_sha256")}
                       | dict(parent_receipt=frame["receipt"]))
        epochs.setdefault(source["run"], saved["light_epoch_id"])
        if epochs[source["run"]] != saved["light_epoch_id"]:
            raise ValueError("Parent run light epoch changed")
    sources.sort(key=lambda item: (item["run"], item["frame_index"]))
    if len(sources) != 864 or len({(row["run"], row["frame_index"]) for row in sources}) != 864 or len(epochs) != 43:
        raise ValueError("Parent source inventory is incomplete")
    identity = list(_model_identity(root))
    plan = dict(contract=CONTRACT, created_utc=created, binding=binding,
        source_inventory=sources, run_to_epoch=epochs, model_identity=identity,
        job_sha256=canonical_sha256(binding),
        pipeline_sha256=canonical_sha256(dict(binding=binding, model_identity=identity)),
        output_root=str(root), placement="P1-final3", delta=seal["delta"], backend="cpu")
    atomic_json(root / PLAN, plan)
    profile = read_json(parent / "phase_rig_camera.json")
    for key in list(profile):
        if key.endswith("_revision") and key != "rig_domain_revision":
            profile.pop(key)
    profile["parent_model_identity"] = list(_model_identity(parent))
    profile["p1_final3_revision"] = dict(contract=CONTRACT, plan=artifact(root / PLAN))
    atomic_json(root / "profile_template.json", profile)
    copy_file_verified(spec, root / "approved_p1_final3_spec.md")
    for name in inputs["code"]:
        path = Path(importlib.import_module(name).__file__)
        copy_file_verified(path, root / "executed_code" / (name + ".py"))
    return plan


def _packed(root, directory, source, mask):
    stream = io.BytesIO()
    np.save(stream, np.packbits(mask), allow_pickle=False)
    return dict(_write(root / directory / source["run"] / f"frame_{source['frame_index']:03d}.npy", stream.getvalue()),
        contract=CONTRACT, storage="packbits-npy", shape_hw=list(mask.shape),
        frame_index=source["frame_index"], source_sha256=source["source_sha256"])


def _previous(root, directory, source, labels):
    stream = io.BytesIO()
    np.savez_compressed(stream, previous_labels=labels)
    return dict(_write(root / directory / source["run"] / f"frame_{source['frame_index']:03d}.npz", stream.getvalue()),
        contract=CONTRACT, storage="npz-array", array_key="previous_labels",
        shape_hw=list(labels.shape), frame_index=source["frame_index"],
        source_sha256=source["source_sha256"])


def _area(asset):
    path = bound_path(asset)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        return archive[asset.get("array_key", "pixel_area_mm2")]


def _save_maps(root, plan, source, labels, parent, *, unchanged=False):
    """Render only changed display layers; gas and mixed bytes are reused."""
    from darsia.fluidflower.phase_rendering import AQUEOUS_RGBA, GAS_RGBA
    run, index = source["run"], source["frame_index"]
    folder = root / "validation/run_packages" / run / "review_assets/native_phase_maps" / run / ("A_" + plan["pipeline_sha256"][:16])
    maps = {}
    if not unchanged:
        rgba = np.zeros((*labels.shape, 4), np.uint8)
        rgba[labels == 1] = AQUEOUS_RGBA
        rgba[labels == 2] = GAS_RGBA
        for name in ("combined", "aqueous"):
            if name == "aqueous":
                rgba[labels == 2] = 0
            stream = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(stream, format="PNG", compress_level=1)
            destination = folder / f"frame_{index:03d}.png" if name == "combined" else folder / name / f"frame_{index:03d}.png"
            maps[name] = _write(destination, stream.getvalue())
        del rgba
    for name in (("combined", "aqueous", "gas", "mixed") if unchanged else ("gas", "mixed")):
        destination = folder / f"frame_{index:03d}.png" if name == "combined" else folder / name / f"frame_{index:03d}.png"
        copy_file_verified(bound_path(parent["maps"][name]), destination)
        maps[name] = artifact(destination)
        if maps[name]["sha256"] != parent["maps"][name]["sha256"]:
            raise ValueError("Unchanged gas/mixed display bytes changed")
    label_path = root / "production_labels" / run / f"frame_{index:03d}.png"
    if unchanged:
        copy_file_verified(bound_path(parent["labels_map"]), label_path)
        label_asset = artifact(label_path)
    else:
        stream = io.BytesIO()
        Image.fromarray(labels, "L").save(stream, format="PNG", compress_level=1)
        label_asset = _write(label_path, stream.getvalue())
    if not np.array_equal(_png(label_asset), labels):
        raise ValueError("Saved P1-final3 categories differ from the computed labels")
    return maps, label_asset


def process_frame(root, plan, source, *, resume=False):
    wall, cpu = time.perf_counter(), time.thread_time()
    parent = read_json(bound_path(source["parent_receipt"]))
    full = parent["binding"]["source"]
    if any(source[key] != full.get(key) for key in ("run", "frame_index", "source_sha256", "elapsed_h")):
        raise ValueError("Source inventory changed")
    path = root / "rig_camera_frames" / source["run"] / f"frame_{source['frame_index']:03d}.json"
    if path.exists():
        saved = read_json(path)
        if (not resume or saved.get("contract") != CONTRACT or saved.get("complete") is not True
                or saved["binding"] != dict(pipeline_sha256=plan["pipeline_sha256"], source=full, parent_receipt=source["parent_receipt"])
                or saved["qa"].get("all_pass") is not True):
            raise ValueError("Existing frame is not an exact immutable resume")
        for asset in (saved["labels_map"], *saved["maps"].values(), saved["floor_contact_fill_v2"],
                      saved["contact_fill_b_reverted_masks"], saved["reverted_previous_labels"],
                      *saved["removed_display_layers"].values(), *saved["previous_labels"].values()):
            bound_path(asset)
        return dict(run=source["run"], frame_index=source["frame_index"], receipt=artifact(path), qa=saved["qa"])
    before = _png(parent["labels_map"])
    a = _mask(parent["fill_reason_masks"]["a"], full, before.shape)
    b = _mask(parent["fill_reason_masks"]["b"], full, before.shape)
    contact = _mask(parent["floor_contact_fill_v2"], full, before.shape)
    result = rules.remove_contact_fill_b(before, a, b, contact)
    old_parent = _png(parent["before_labels"])
    memberships = a.astype(np.uint8)
    flags = {}
    for stage in "ABD":
        flags[stage] = _mask(parent[MASK_DIRECTORIES[stage]], full, before.shape)
        memberships += flags[stage]
    changed = result != before
    changed_from_original = result != old_parent
    tests = dict(changed_exact_reason_b=np.array_equal(changed, b),
                 changed_one_to_zero=bool(np.all(before[changed] == 1) and np.all(result[changed] == 0)),
                 gas_bit_identical=np.array_equal(before == 2, result == 2),
                 domain_bit_identical=np.array_equal(before == 255, result == 255),
                 other_pixels_identical=np.array_equal(before[~b], result[~b]),
                 original_changed_exactly_one_step=bool(np.all(memberships[changed_from_original] == 1)))
    if not all(tests.values()):
        raise ValueError(f"P1-final3 identity failed: {source['run']}/{source['frame_index']}: {tests}")
    area = _area(parent["rig_domain_revision"]["area_map"])
    if area.shape != before.shape or not np.isfinite(area[b]).all() or np.any(area[b] <= 0):
        raise ValueError("Reverted pixels lack valid unchanged measured area")
    qa = dict(all_pass=True, **tests, run=source["run"], frame_index=source["frame_index"],
        source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
        light_epoch_id=parent["light_epoch_id"], reverted_pixels=int(b.sum()),
        reverted_nominal_cm2=float(b.sum()) * float(parent["rig_domain_revision"]["fixed_area_mm2"]) / 100,
        reverted_measured_cm2=float(area[b].sum(dtype=np.float64)) / 100,
        changed_from_p1_final_pixels=int(changed_from_original.sum()),
        nominal_pixel_area_mm2=float(parent["rig_domain_revision"]["fixed_area_mm2"]), backend="cpu")
    del area, old_parent, memberships, changed_from_original, flags, contact
    maps, label_asset = _save_maps(root, plan, source, result, parent, unchanged=not bool(b.any()))
    contact_asset = _packed(root, "floor_contact_fill_v2", full, a)
    reverted_asset = _packed(root, "contact_fill_b_reverted_masks", full, b)
    previous_b = np.zeros(before.shape, np.uint8)
    previous_b[b] = before[b]
    previous_asset = _previous(root, "reverted_previous_labels", full, previous_b)
    del previous_b
    previous = deepcopy(parent["previous_labels"])
    previous["C"] = _previous(root, "previous_labels/C", full, np.zeros(before.shape, np.uint8))
    previous["contact_fill_b_reverted"] = previous_asset
    overlays = deepcopy(parent["removed_display_layers"])
    overlays[OVERLAYS["C"]] = _overlay(root, source["run"], source["frame_index"], plan["pipeline_sha256"], a,
                                       (84, 156, 242, 255), OVERLAYS["C"])
    overlays[REMOVED_OVERLAY] = _overlay(root, source["run"], source["frame_index"], plan["pipeline_sha256"], b,
                                        (226, 190, 83, 255), REMOVED_OVERLAY)
    qa.update(cpu_seconds=time.thread_time() - cpu, gpu_seconds=0.0, wall_seconds=time.perf_counter() - wall)
    receipt = dict(contract=CONTRACT, complete=True, created_utc=utc_now(),
        binding=dict(pipeline_sha256=plan["pipeline_sha256"], source=full, parent_receipt=source["parent_receipt"]),
        **{key: parent[key] for key in ("run", "frame_index", "source_sha256", "elapsed_h", "delta", "light_epoch_id",
            "rig_domain_revision", "full_extended_scientific", "baseline_extended_scientific")},
        placement="P1-final3", before_labels=parent["labels_map"], p1_final_before_labels=parent["before_labels"],
        labels_map=label_asset, maps=maps, previous_labels=previous, reverted_previous_labels=previous_asset,
        removed_display_layers=overlays, fill_reason_masks=deepcopy(parent["fill_reason_masks"]),
        contact_fill_b_reverted_masks=reverted_asset, floor_contact_fill_v2=contact_asset,
        **{MASK_DIRECTORIES[key]: parent[MASK_DIRECTORIES[key]] for key in "ABD"},
        qa=qa, backend="cpu", labels_changed=False, production_labels_modified=True, annotations_read=False,
        training=False, training_performed=False, inference_performed=False, co2_inference_performed=False,
        new_gas_inference_performed=False, mass_eligible=False, bottom_mass_allowed=False,
        parent_flags_reused_bit_identically=[MASK_DIRECTORIES[key] for key in "ABD"])
    atomic_json(path, receipt)
    return dict(run=source["run"], frame_index=source["frame_index"], receipt=artifact(path), qa=qa)


def assemble_run(root, plan, run, frames):
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    parent = Path(plan["binding"]["parent"])
    native = read_json(parent / "source_alias/native_frames" / run / "manifest.json")
    inputs = read_json(bound_path(native[meta.SOURCE_KEY]["input_manifest"]))
    by_index = {frame["frame_index"]: frame for frame in frames}
    if len(by_index) != len(native["frame_sources"]):
        raise ValueError("Incomplete P1-final3 run")
    for record, row in zip(native["frame_sources"], inputs["frames"]):
        saved = read_json(bound_path(by_index[record["frame_index"]]["receipt"]))
        if saved["source_sha256"] != record["source_sha256"] or row["source_sha256"] != record["source_sha256"]:
            raise ValueError("Exact camera source order changed")
        for value in (record, row):
            for name in (*MASK_DIRECTORIES.values(), "contact_fill_b_reverted_masks", "reverted_previous_labels"):
                value[name] = saved[name]
    input_path = root / "source_alias/source_bindings" / (run + ".json")
    atomic_json(input_path, inputs)
    native = meta.build_native_metadata(read_json(bound_path(inputs["parent_native_manifest"])),
        read_json(bound_path(inputs["camera_timeline_run"])), native["frame_sources"], inputs["frames"],
        inputs["reference"], input_manifest=artifact(input_path))
    native_path = root / "source_alias/native_frames" / run / "manifest.json"
    atomic_json(native_path, native)
    result = dict(run=run, complete=True, native_manifest=artifact(native_path), input_manifest=artifact(input_path),
        frames=inputs["frames"], reference=inputs["reference"], frame_count=len(frames),
        receipts=[{key: value for key, value in frame.items() if key != "qa"} for frame in frames],
        pipeline_sha256=plan["pipeline_sha256"])
    atomic_json(root / RUN_DIRECTORY / (run + ".json"), result)
    return result


def bind_final3_packages(root, plan, source_runs):
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    from .phase_reference_readiness import defer_validation_references
    parent = Path(plan["binding"]["parent"])
    old_source = read_json(parent / "review_source/postflight_manifest.json")
    sources = {row["run"]: row for row in source_runs}
    jobs = []
    for old in old_source["jobs"]:
        run = old["run"]
        job = dict(old, review_output_dir=str(root / "source_alias"), analysis_source_review_root=str(root / "source_alias"),
                   native_manifest_sha256=sources[run]["native_manifest"]["sha256"], frame_count=sources[run]["frame_count"])
        original = read_json(parent / "review_source" / run / "postflight_manifest.json")
        atomic_json(root / "review_source" / run / "postflight_manifest.json", dict(original, jobs=[job]))
        jobs.append(job)
    atomic_json(root / "review_source/postflight_manifest.json", dict(old_source, jobs=jobs))
    loaded = load_physics_profile(root)
    runtime, profile = physics_runtime_identity(loaded), meta.public_profile(loaded)
    packages = []
    for old in read_json(parent / "validation/postflight_manifest.json")["jobs"]:
        run = old["run"]
        package = root / "validation/run_packages" / run
        job = deepcopy(old)
        job.update(candidate_id=root.name + "__ffac__" + run + "__rigcamera_" + plan["pipeline_sha256"][:12],
            candidate_folder=str(root), review_output_dir=str(root / "source_alias"), analysis_source_review_root=str(root / "source_alias"),
            phase_map_output_dir=str(package / "review_assets"), params_sha256=plan["pipeline_sha256"],
            optical_model_sha256=plan["model_identity"][1], physics_runtime=runtime, output_mode="review", phase_model_only=True,
            raw_phase_map_output_dir=None, numeric_phase_output_dir=None, presence_diagnostics_dir=None,
            native_manifest_sha256=sources[run]["native_manifest"]["sha256"], presentation_status="ready", proposals_status="deferred")
        job["quality_flags"] = dict(job.get("quality_flags") or {}, approved_g3_rig_camera=True, approved_g3_downstream=True,
            human_review_required=True, mass_analysis_allowed=False, scientific_acceptance=False,
            p1_final3_revision=True, floor_contact_rule=False, bottom_mass_allowed=False)
        for key in ("p1_final_revision", "p1_final2_revision", "p1_final_removed_top", "p1_final_removed_floor"):
            job["quality_flags"].pop(key, None)
        for key in ("summary", "mass_plot", "contour_movie", "per_frame_csv"):
            job[key] = None
        common = dict(schema=1, format="darsia-phase-validation-campaign-v1", campaign_id="phase-validation-" + root.name,
            model_id=plan["model_identity"][0], model_sha256=plan["model_identity"][1], model_root=str(root),
            source_manifest_sha256=sha256_file(root / "review_source" / run / "postflight_manifest.json"),
            pipeline_sha256=plan["pipeline_sha256"], physics_runtime=runtime, physics_profile=profile,
            output_mode="review", numeric_outputs=False, phase_display_policy=meta.REVISION_CONTRACT)
        atomic_json(package / "postflight_manifest.json", dict(common, jobs=[job], source_campaign_root=str(root / "review_source" / run)))
        atomic_json(package / "validation_binding.json", dict(common, combined_root=str(package), ffac_root=str(package), ffum_root=str(package),
            selected_candidates={run: job["candidate_id"]}, run_count=1, completed_utc=plan["created_utc"]))
        defer_validation_references(package)
        packages.append(package)
    return packages


def seal_candidate(root, plan, frames):
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    from .build_ffac_g3_approved_candidate import assemble_saved_candidate
    if len(frames) != 864 or {(row["run"], row["frame_index"]) for row in frames} != {
            (row["run"], row["frame_index"]) for row in plan["source_inventory"]}:
        raise ValueError("A candidate requires exact complete 43/864 source coverage")
    source_runs = [assemble_run(root, plan, run, [row for row in frames if row["run"] == run]) for run in sorted(plan["run_to_epoch"])]
    meta.validate_campaign_bindings(source_runs)
    profile = read_json(root / "profile_template.json")
    runtime = read_json(bound_path(profile["rig_domain_revision"]["manifest"]))
    runtime["sources"] = [{key: value for key, value in row.items() if key != "receipts"} for row in source_runs]
    path = root / "rig_camera_revision/runtime_manifest.json"
    atomic_json(path, runtime)
    profile["rig_domain_revision"]["manifest"] = artifact(path)
    atomic_json(root / "phase_rig_camera.json", profile)
    packages = bind_final3_packages(root, plan, source_runs)
    assemble_saved_candidate(root, packages, plan)
    rows = [row["qa"] for row in frames]
    if not all(row["all_pass"] for row in rows):
        raise ValueError("Producer pixel identities failed")
    qa = dict(contract=CONTRACT, technical_all_pass=True, run_count=43, frame_count=864, rules=rules.RULE,
        backend="cpu", rows=rows, **{UNCHANGED: True}, domain_revision="pooled_sand_floor_v1", annotations_read=False,
        training=False, inference=False, cost=dict(cpu_seconds=sum(row["cpu_seconds"] for row in rows), gpu_seconds=0.0))
    atomic_json(root / QA, qa)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader()
    writer.writerows(rows)
    _write(root / "production_qa/p1_final3_steps.csv", stream.getvalue().encode("utf-8"))
    seal = dict(contract=CONTRACT, complete=True, model_identity=plan["model_identity"], pipeline_sha256=plan["pipeline_sha256"],
        source_plan_sha256=sha256_file(root / PLAN), run_count=43, frame_count=864,
        frames=[{key: value for key, value in row.items() if key != "qa"} for row in frames],
        parent_seal=plan["binding"]["inputs"]["parent_seal"], heads=plan["binding"]["heads"], delta=plan["delta"],
        placement="P1-final3", qa=artifact(root / QA), physical_scale_approved=True, **{UNCHANGED: True},
        domain_revision="pooled_sand_floor_v1", labels_changed=False, mass_eligible=False, bottom_mass_allowed=False,
        blind_hold_runs=[], training_performed=False, inference_performed=False, automatic_activation=False,
        scientific_acceptance=False, independent_saved_qa_required_before_registration=True)
    atomic_json(root / SEAL, seal)
    return dict(seal=artifact(root / SEAL), qa=artifact(root / QA), registration_performed=False)


def build(root, plan, *, workers=24, resume=False, indices=None):
    import psutil
    if not 1 <= workers <= 24:
        raise ValueError("CPU workers must be 1–24")
    if psutil.virtual_memory().available < (workers * .75 + 4) * 1024 ** 3:
        raise ValueError("Insufficient RAM for the requested bounded saved-label workers")
    root = Path(root)
    if code_binding() != plan["binding"]["inputs"]["code"]:
        raise ValueError("Running code differs from the immutable saved-label snapshot")
    cv2.setNumThreads(1)
    wall, cpu = time.perf_counter(), time.process_time()
    sources = [row for row in plan["source_inventory"] if indices is None or (row["run"], row["frame_index"]) in indices]
    iterator = iter(sources)
    completed = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p1-final3-saved") as pool:
        pending = {pool.submit(process_frame, root, plan, source, resume=resume): source for source in [next(iterator, None) for _ in range(workers)] if source}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                source = pending.pop(future)
                completed.append(future.result())
                following = next(iterator, None)
                if following:
                    pending[pool.submit(process_frame, root, plan, following, resume=resume)] = following
            if len(completed) % 12 < len(done) or not pending:
                progress = dict(stage="saved_reason_b_removal", done=len(completed), total=len(sources),
                    backend="cpu", workers=workers, internal_threads=1, elapsed_seconds=time.perf_counter() - wall,
                    cpu_seconds=time.process_time() - cpu, last_run=source["run"], updated_utc=utc_now())
                atomic_json(root / "progress" / f"snapshot_{time.time_ns()}.json", progress)
                print(json.dumps(progress), flush=True)
    completed.sort(key=lambda row: (row["run"], row["frame_index"]))
    if indices is not None:
        return completed
    result = seal_candidate(root, plan, completed)
    atomic_json(root / "completion.json", dict(contract=CONTRACT, complete=True, **result,
        wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu,
        backend="cpu", workers=workers, internal_threads=1, inference_performed=False, training_performed=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "spec"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    with JobLease(args.output.parent / (args.output.name + "-operation"), SPEC_SHA256):
        plan = prepare_candidate(args.parent, args.output, args.spec, resume=args.resume)
        indices = {("ac44", 4), ("ac45", 5), ("ac56", 11)} if args.probe else None
        result = build(args.output, plan, workers=args.workers, resume=args.resume, indices=indices)
        print(json.dumps(dict(complete=not args.probe, result=result), default=str), flush=True)


if __name__ == "__main__":
    main()
