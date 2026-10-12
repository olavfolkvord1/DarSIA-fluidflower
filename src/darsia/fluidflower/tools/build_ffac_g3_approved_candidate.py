"""Build the approved G3 Studio review candidate from immutable exported layers.

No RAW inference, extraction or fitting is performed. SHA-bound available
frames can be processed while the producer continues its other exports.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower.phase_g3_production import (
    APPROVED_SETTINGS, CO2_SHA256, G3_SHA256, REFIT_SHA256, VERSION,
    apply_g3_configuration,
)
from darsia.fluidflower.phase_validation import PHASE_FILL_COLORS, _model_identity
from darsia.fluidflower.tools.refit_ffac_g3_downstream import (
    JobLease, ProgressLog, atomic_json, atomic_text, below_normal_priority,
    canonical_sha256, copy_file_verified, copy_tree_verified, ordered_stages,
    preflight_space, read_json, sha256_file, utc_now, validate_inventory, worker_initializer,
)


CONTRACT = "ffac-g3-approved-candidate-from-exported-layers-v1"
PARITY_CONTRACT = "ffac-g3-approved-labelled-parity-v1"
UNION_CONTRACT = "ffac-g3-approved-co2-union-identity-v1"
SPEC_SHA256 = "9d773b61c370dfa2e7f1116c4b858ea143ef8545581847810c52a96bf19f1bc9"
CONFIGURATION = dict(regional_threshold=.6, pixel_threshold=.7, minimum_area_mm2=2.5, connectivity=4)
DISPLAY_MODES = ("natural", "lift1", "lift2")


def mask_sha256(mask) -> str:
    value = np.asarray(mask, bool)
    return hashlib.sha256(np.packbits(value.reshape(-1), bitorder="little").tobytes()).hexdigest()


def independent_literal_gas(g3, regional, co2, pixel_area_mm2) -> np.ndarray:
    """Independent QA implementation using SciPy rather than production CC code."""
    from scipy.ndimage import label
    support = np.asarray(co2, bool)
    values = np.asarray(regional, np.float32)
    if values.shape != support.shape:
        values = cv2.resize(values, support.shape[::-1], interpolation=cv2.INTER_LINEAR)
    selected = support & (np.asarray(g3) >= .7) & (values >= .6)
    components, count = label(selected, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if count:
        sizes = np.bincount(components.reshape(-1))
        small = sizes.astype(np.float64) * float(pixel_area_mm2) < 2.5
        small[0] = False
        selected &= ~small[components]
    return selected


def parity_row(labels, arrays, metadata, *, run, frame_index) -> dict:
    co2 = np.asarray(arrays["co2"], bool)
    area = float(metadata["pixel_area_mm2"])
    reference = independent_literal_gas(arrays["g3_pixel"], arrays["regional_refit"], co2, area)
    actual = np.asarray(labels) == 2
    pixels = int(np.count_nonzero(co2))
    disagreements = int(np.count_nonzero(co2 & (reference != actual)))
    agreement = 1. if pixels == 0 else 1. - disagreements / pixels
    actual_gas, literal_gas = int(actual.sum()), int(reference.sum())
    return dict(run=run, frame_index=frame_index, elapsed_h=metadata["elapsed_h"],
                co2_pixels=pixels, agreement_inside_co2=agreement, disagreements=disagreements,
                production_gas_pixels=actual_gas, literal_gas_pixels=literal_gas,
                production_gas_area_mm2=actual_gas * area, literal_gas_area_mm2=literal_gas * area,
                pixel_area_mm2=area, passed=agreement >= .999)


def _write_bytes(path: Path, value: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(value).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise ValueError("Existing approved candidate artifact differs; never overwrite it")
    else:
        temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name == "nt":
                temporary.rename(path)
            else:
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return dict(path=str(path), sha256=digest)


def write_phase_layers(root: Path, run: str, index: int, pipeline: str,
                       frozen_rgba, labels, content_box) -> dict:
    x, y, width, height = map(int, content_box)
    original = np.asarray(frozen_rgba, np.uint8)
    if np.asarray(labels).shape != (height, width):
        raise ValueError("Production labels do not occupy the exact source-native content box")
    combined = original.copy()
    local = combined[y:y + height, x:x + width]
    for name, label_value in (("aqueous", 1), ("gas", 2)):
        local[labels == label_value] = PHASE_FILL_COLORS[name]
    folder = root / "native_phase_maps" / run / ("A_" + pipeline[:16])
    result = {}
    # Saved combined overlay retains all frozen canvas and unresolved pixels.
    layers = {"combined": combined}
    for name, label_value in (("aqueous", 1), ("gas", 2), ("mixed", 3)):
        rgba = np.zeros_like(original)
        if label_value != 3:
            region = rgba[y:y + height, x:x + width]
            region[labels == label_value] = PHASE_FILL_COLORS[name]
        layers[name] = rgba
    for name, rgba in layers.items():
        path = folder / f"frame_{index:03d}.png" if name == "combined" else folder / name / f"frame_{index:03d}.png"
        encoded = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(encoded, format="PNG", compress_level=1)
        result[name] = _write_bytes(path, encoded.getvalue())
    return result


def _frame_receipt_path(root: Path, run: str, index: int) -> Path:
    return root / "production_frames" / run / f"frame_{index:03d}.json"


def verify_frame_receipt(path: Path, candidate: Path, *, binding: dict | None = None) -> dict:
    receipt = read_json(path)
    if receipt.get("contract") != CONTRACT or receipt.get("complete") is not True:
        raise ValueError("Production frame receipt is incomplete")
    if binding is not None and receipt["binding"] != binding:
        raise ValueError("Completed production frame binds another source or approved configuration")
    for item in [*receipt["maps"].values(), receipt["labels_archive"], receipt["labels_map"], receipt["runtime_frame_binding"]]:
        artifact = Path(item["path"]).resolve()
        if not artifact.is_relative_to(candidate.resolve()) or sha256_file(artifact) != item["sha256"]:
            raise ValueError("Production candidate output changed or escaped its namespace")
    return receipt


def render_frame(candidate: str, export_root: str, row: dict, pipeline: str,
                 threads: int = 1, runtime_export_root: str | None = None) -> dict:
    from darsia.fluidflower.phase_g3_probability_layers import load_exported_layers
    from darsia.fluidflower.tools.train_ffac_gas_flatfield_variants import _read_frozen_map
    cv2.setNumThreads(threads)
    root = Path(candidate)
    run, index = row["run"], int(row["frame_index"])
    source_receipt = Path(export_root) / "layers" / run / f"frame_{index:03d}" / "receipt.json"
    runtime_receipt = Path(runtime_export_root or export_root) / "layers" / run / f"frame_{index:03d}" / "receipt.json"
    exported = read_json(source_receipt)
    inputs = exported["binding"]
    metadata = exported["metadata"]
    if (exported.get("complete") is not True or inputs["provisional"] is not False
            or inputs["g3_model_sha256"] != G3_SHA256 or inputs["regional_refit_sha256"] != REFIT_SHA256
            or inputs["native_manifest_sha256"] != row["native_manifest_sha256"]
            or (inputs["run"], int(inputs["frame_index"])) != (run, index)):
        raise ValueError("Frame export is not the exact completed approved G3/refit source")
    binding = dict(run=run, frame_index=index, pipeline_sha256=pipeline,
                   probability_receipt_sha256=sha256_file(source_receipt),
                   runtime_probability_receipt_path=str(runtime_receipt),
                   native_manifest_sha256=row["native_manifest_sha256"],
                   frozen_map_sha256=inputs["frozen_map_sha256"],
                   g3_sha256=G3_SHA256, regional_refit_sha256=REFIT_SHA256,
                   co2_sha256=CO2_SHA256, configuration=CONFIGURATION)
    receipt_path = _frame_receipt_path(root, run, index)
    if receipt_path.exists():
        verify_frame_receipt(receipt_path, root, binding=binding)
        return dict(run=run, frame_index=index, receipt_path=str(receipt_path),
                    receipt_sha256=sha256_file(receipt_path), resumed=True)
    arrays, exported = load_exported_layers(source_receipt)
    frozen_map = Path(metadata["frozen_map_path"])
    if sha256_file(frozen_map) != inputs["frozen_map_sha256"]:
        raise ValueError("Frozen parent categorical map changed after probability export")
    native_path = Path(row["native_manifest_path"])
    if sha256_file(native_path) != row["native_manifest_sha256"]:
        raise ValueError("Native source manifest changed after the production plan")
    native = read_json(native_path)
    content = native["content_box_xywh"]
    frozen, rgba = _read_frozen_map(frozen_map, content)
    co2 = np.isin(frozen, (1, 2))
    if not np.array_equal(co2, arrays["co2"]):
        raise ValueError("Exported CO2 union differs from the immutable parent categorical map")
    labels, audit = apply_g3_configuration(frozen, arrays["g3_pixel"], arrays["regional_refit"],
                                         metadata["pixel_area_mm2"])
    production_co2 = np.isin(labels, (1, 2))
    if not np.array_equal(co2, production_co2) or not np.array_equal(labels[~co2], frozen[~co2]):
        raise ValueError("Approved G3 partition changed the frozen CO2 union or unsupported labels")
    assets = native["frame_sources"][index].get("v5_display_modes") or {}
    if set(assets) != set(DISPLAY_MODES):
        raise ValueError("Native source lacks the exact three parent photographic display modes")
    for item in assets.values():
        if sha256_file(item["jpeg_path"]) != item["jpeg_sha256"]:
            raise ValueError("Inherited photographic display bytes changed")
    maps = write_phase_layers(root / "validation/run_packages" / run / "review_assets",
                              run, index, pipeline, rgba, labels, content)
    encoded = io.BytesIO()
    np.savez_compressed(encoded, labels=labels)
    archive = _write_bytes(root / "production_labels" / run / f"frame_{index:03d}.npz", encoded.getvalue())
    encoded = io.BytesIO()
    Image.fromarray(labels, "L").save(encoded, format="PNG", compress_level=1)
    proof = _write_bytes(root / "production_labels" / run / f"frame_{index:03d}.png", encoded.getvalue())
    with Image.open(proof["path"]) as image:
        persisted = np.asarray(image).copy()
    if not np.array_equal(persisted, labels):
        raise ValueError("Persisted categorical proof differs from production labels")
    production_co2 = np.isin(persisted, (1, 2))
    if inputs["labelled"]:
        studio_labels, _ = _read_frozen_map(Path(maps["combined"]["path"]), content)
        if not np.array_equal(studio_labels, persisted):
            raise ValueError("Persisted Studio palette differs from native categorical proof")
    runtime_path = root / "approved_g3_frame_bindings" / run / f"frame_{index:03d}.json"
    runtime = dict(format=VERSION, run=run, frame_index=index, elapsed_h=metadata["elapsed_h"],
                   native_manifest_sha256=row["native_manifest_sha256"], content_box_xywh=content,
                   probability_receipt=dict(path=str(runtime_receipt), sha256=binding["probability_receipt_sha256"]),
                   probability_folder=str(runtime_receipt.parent),
                   frozen_map=dict(path=str(frozen_map), sha256=inputs["frozen_map_sha256"]))
    atomic_json(runtime_path, runtime)
    union = dict(run=run, frame_index=index, exact=True, map_path=proof["path"],
                 map_sha256=proof["sha256"], studio_map_path=maps["combined"]["path"],
                 studio_map_sha256=maps["combined"]["sha256"], content_box_xywh=content,
                 canvas_wh=[rgba.shape[1], rgba.shape[0]], export_co2_sha256=mask_sha256(arrays["co2"]),
                 production_co2_sha256=mask_sha256(production_co2), frozen_co2_sha256=mask_sha256(co2),
                 export_co2_file_sha256=exported["files"]["co2.packbits.npy"]["sha256"],
                 frozen_map_sha256=inputs["frozen_map_sha256"], unsupported_exact=True)
    parity = parity_row(persisted, arrays, metadata, run=run, frame_index=index) if inputs["labelled"] else None
    if parity is not None:
        parity.update(native_map_sha256=proof["sha256"], studio_map_sha256=maps["combined"]["sha256"],
                      persisted_studio_matches_native=True)
    if parity is not None and not parity["passed"]:
        raise ValueError("Production map differs from independent literal reconstruction by more than0.1percent")
    receipt = dict(contract=CONTRACT, complete=True, binding=binding, elapsed_h=metadata["elapsed_h"],
                   maps=maps, labels_archive=archive, labels_map=proof,
                   runtime_frame_binding=dict(path=str(runtime_path), sha256=sha256_file(runtime_path)),
                   photographic_display_modes=assets, display_only=True, raw_inference_performed=False,
                   production_audit=audit, union=union, labelled_parity=parity)
    atomic_json(receipt_path, receipt)
    verify_frame_receipt(receipt_path, root, binding=binding)
    return dict(run=run, frame_index=index, receipt_path=str(receipt_path),
                receipt_sha256=sha256_file(receipt_path), resumed=False)


def _copy_lineage(parent: Path, output: Path) -> dict:
    paths = [parent / name for name in ("training_request.json", "expected_review_runs.json",
             "evidence.json", "extraction.json", "regional_training_protocol.json", "region_inventory.csv")
             if (parent / name).is_file()]
    for name in ("models", "input_snapshot", "parent_provenance", "registered_context"):
        directory = parent / name
        if not directory.is_dir():
            raise ValueError("Frozen candidate lacks scientific lineage: " + name)
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    paths.extend(path for path in (parent / "review_source").rglob("*.json"))
    preserved = {}
    for path in paths:
        relative = path.relative_to(parent)
        copy_file_verified(path, output / relative)
        preserved[relative.as_posix()] = sha256_file(path)
    return preserved


def prepare_candidate(parent: Path, experiment: Path, export_root: Path, output: Path,
                      spec: Path, selection: Path, *, resume: bool = False,
                      runtime_export_root: Path | None = None) -> dict:
    from darsia.fluidflower.phase_g3_probability_layers import inventory_frames
    from darsia.fluidflower.phase_optical_regions_runtime import load_profile
    from darsia.fluidflower import phase_g3_production
    parent, experiment, export_root, output, spec, selection = [Path(path).resolve() for path in
        (parent, experiment, export_root, output, spec, selection)]
    runtime_export_root = Path(runtime_export_root or export_root).resolve()
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (parent, experiment, export_root)):
        raise ValueError("Approved candidate must have a new detached output namespace")
    if sha256_file(spec) != SPEC_SHA256:
        raise ValueError("The approved production spec differs from the specified updated revision")
    refit = export_root / "regional_refit/regional_model.json"
    g3 = experiment / "variants/G3/gas_head.json"
    co2 = parent / "models/generic/co2_presence_signal.json"
    if (sha256_file(refit), sha256_file(g3), sha256_file(co2)) != (REFIT_SHA256, G3_SHA256, CO2_SHA256):
        raise ValueError("Approved production heads differ from the frozen G3/refit/CO2 heads")
    refit_receipt = read_json(export_root / "regional_refit/regional_refit_receipt.json")
    if refit_receipt.get("converged") is not True:
        raise ValueError("The approved regional gas refit is not converged")
    original, updated = read_json(parent / "regional_model.json"), read_json(refit)
    unchanged_before, unchanged_after = deepcopy(original), deepcopy(updated)
    unchanged_before["heads"].pop("gas_within_co2")
    unchanged_after["heads"].pop("gas_within_co2")
    if unchanged_before != unchanged_after:
        raise ValueError("Refit changed a non-gas regional head or another regional setting")
    indexed = validate_inventory(inventory_frames(parent, experiment))
    material = dict(contract=CONTRACT, parent=str(parent), experiment=str(experiment),
                    export_root=str(export_root), candidate=str(output), configuration=CONFIGURATION,
                    runtime_export_root=str(runtime_export_root),
                    spec_path=str(spec), spec_sha256=SPEC_SHA256,
                    selection_results_path=str(selection), selection_results_sha256=sha256_file(selection),
                    g3_sha256=G3_SHA256, regional_refit_sha256=REFIT_SHA256, co2_sha256=CO2_SHA256,
                    export_job_request_sha256=sha256_file(export_root / "job_request.json"),
                    source_inventory=[indexed[key] for key in sorted(indexed)],
                    code_sha256=dict(driver=sha256_file(Path(__file__)), production=sha256_file(Path(phase_g3_production.__file__))),
                    cpu_only=True, raw_inference=False, retraining=False, labels_changed=False,
                    mass_eligible=False, scientific_acceptance=False)
    job_sha = canonical_sha256(material)
    plan_path = output / "approved_g3_candidate_plan.json"
    if plan_path.exists():
        if not resume:
            raise ValueError("Existing approved candidate requires explicit --resume")
        plan = read_json(plan_path)
        if plan["binding"] != material or plan["job_sha256"] != job_sha:
            raise ValueError("Resume differs from the immutable approved candidate plan")
        if plan["model_identity"] != list(_model_identity(output)):
            raise ValueError("Approved candidate model identity changed")
        return plan
    if resume or output.exists():
        raise ValueError("Use a new absent candidate, or --resume with its complete immutable plan")
    preflight_space(output, 16 * 1024 ** 3)
    output.mkdir(parents=True, exist_ok=False)
    preserved = _copy_lineage(parent, output)
    copy_file_verified(refit, output / "regional_model.json")
    copy_file_verified(g3, output / "models/g3/gas_head.json")
    copy_file_verified(spec, output / "approved_source_spec.md")
    copy_file_verified(selection, output / "approved_selection_results.md")
    effective = load_profile(parent)
    profile = {key: deepcopy(value) for key, value in effective.items()
               if not key.startswith("_") and key != "v5_track1_incremental_inference"}
    profile["model"] = dict(profile["model"], path="regional_model.json", sha256=REFIT_SHA256)
    profile["approved_g3_gas"] = dict(format=VERSION, settings=asdict(APPROVED_SETTINGS),
        g3=dict(path="models/g3/gas_head.json", sha256=G3_SHA256),
        regional=dict(path="regional_model.json", sha256=REFIT_SHA256),
        co2=dict(path="models/generic/co2_presence_signal.json", sha256=CO2_SHA256),
        frame_bindings_directory="approved_g3_frame_bindings")
    atomic_json(output / "phase_optical_regions.json", profile)
    summary = read_json(parent / "summary.json")
    summary.update(model_id=output.name, campaign_id=output.name, created_utc=utc_now(),
        display_name="FF_AC · G3 / regional refit / t0.7 / 2.5mm² · 02.10.2026",
        inherited_training_candidate_id=parent.name, training_performed_in_this_revision=False,
        raw_inference_performed=False, approved_g3_production=dict(configuration=CONFIGURATION,
            g3_sha256=G3_SHA256, regional_refit_sha256=REFIT_SHA256, co2_sha256=CO2_SHA256,
            spec_sha256=SPEC_SHA256, selection_results_sha256=sha256_file(selection),
            production_code_sha256=material["code_sha256"], standalone_g3=True,
            generic_gas_bundle_role="immutable compatibility artifact; approved gas decisions use standalone G3"),
        scientific_acceptance=False, mass_eligible=False)
    atomic_json(output / "summary.json", summary)
    identity = list(_model_identity(output))
    pipeline = canonical_sha256(dict(model_identity=identity, approval=material,
                                     profile_sha256=sha256_file(output / "phase_optical_regions.json")))
    plan = dict(created_utc=utc_now(), binding=material, job_sha256=job_sha,
                model_identity=identity, pipeline_sha256=pipeline, preserved_artifacts=preserved)
    atomic_json(plan_path, plan)
    return plan


def bind_run_packages(root: Path, plan: dict) -> list[Path]:
    """Build fresh saved-map packages, retaining exact parent photo sources."""
    from darsia.fluidflower.tools.phase_reference_readiness import defer_validation_references
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    physics_profile = load_physics_profile(root)
    physics_runtime = physics_runtime_identity(physics_profile)
    parent = Path(plan["binding"]["parent"])
    source_jobs = read_json(root / "review_source/postflight_manifest.json")["jobs"]
    sources = {job["run"]: job for job in source_jobs}
    packages = []
    for run in sorted(sources):
        destination = root / "validation/run_packages" / run
        destination.mkdir(parents=True, exist_ok=True)
        old_manifest = read_json(parent / "validation/run_packages" / run / "postflight_manifest.json")
        job = deepcopy(old_manifest["jobs"][0])
        candidate_id = root.name + "__ffac__" + run + "__g3approved_" + plan["pipeline_sha256"][:12]
        job.update(candidate_id=candidate_id, candidate_folder=str(root),
                   phase_map_output_dir=str(destination / "review_assets"),
                   params_sha256=plan["pipeline_sha256"], optical_model_sha256=plan["model_identity"][1],
                   raw_phase_map_output_dir=None, presence_diagnostics_dir=None,
                   numeric_phase_output_dir=None, physics_runtime=physics_runtime, output_mode="review",
                   phase_support_format="baseline-relative-optical-regions-v1", phase_model_only=True,
                   presentation_status="ready", proposals_status="deferred")
        job["quality_flags"] = dict(job.get("quality_flags") or {}, human_review_required=True,
            mass_analysis_allowed=False, approved_g3_downstream=True, scientific_acceptance=False)
        for key in ("summary", "mass_plot", "contour_movie", "per_frame_csv"):
            job[key] = None
        source_sha = sha256_file(root / "review_source" / run / "postflight_manifest.json")
        common = dict(schema=1, format="darsia-phase-validation-campaign-v1",
                      campaign_id="phase-validation-" + root.name, model_id=plan["model_identity"][0],
                      model_root=str(root), model_sha256=plan["model_identity"][1],
                      source_manifest_sha256=source_sha, pipeline_sha256=plan["pipeline_sha256"],
                      physics_runtime=physics_runtime, physics_profile=physics_profile,
                      phase_display_policy="approved-g3-refit-t070-a2p5-v1",
                      output_mode="review", numeric_outputs=False)
        atomic_json(destination / "postflight_manifest.json", dict(common, jobs=[job],
                    source_campaign_root=str(root / "review_source" / run)))
        atomic_json(destination / "validation_binding.json", dict(common, ffac_root=str(destination),
                    ffum_root=str(destination), combined_root=str(destination),
                    selected_candidates={run: candidate_id}, roles=["selected_model"], run_count=1,
                    completed_utc=plan["created_utc"]))
        defer_validation_references(destination)
        packages.append(destination)
    return packages


def assemble_saved_candidate(root: Path, packages: list[Path], plan: dict) -> None:
    """Use the ordinary prepared-set metadata without activating any default."""
    from darsia.fluidflower.tools.phase_reference_readiness import defer_validation_references
    source_sha = sha256_file(root / "review_source/postflight_manifest.json")
    material = source_sha + "".join(sha256_file(package / filename) for package in packages
        for filename in ("postflight_manifest.json", "validation_binding.json", "phase_reference_readiness.json"))
    snapshot = root / "validation/prepared_sets" / canonical_sha256(material)[:20]
    manifests = [read_json(path / "postflight_manifest.json") for path in packages]
    bindings = [read_json(path / "validation_binding.json") for path in packages]
    manifest = dict(manifests[0], jobs=[job for value in manifests for job in value["jobs"]],
                    source_campaign_root=str(root / "review_source"), source_manifest_sha256=source_sha)
    binding = dict(bindings[0], ffac_root=str(snapshot), ffum_root=str(snapshot), combined_root=str(snapshot),
                   selected_candidates={key:value for value in bindings for key,value in value["selected_candidates"].items()},
                   source_manifest_sha256=source_sha, run_count=len(packages))
    for destination in (snapshot, root / "validation"):
        atomic_json(destination / "postflight_manifest.json", manifest)
        atomic_json(destination / "validation_binding.json", binding)
        defer_validation_references(destination)
    atomic_json(root / "studio_preparation.json", dict(status="ready", runs=len(packages), runs_done=len(packages),
        presentation_ready=True, partial=False, model_identity=plan["model_identity"], validation=str(snapshot),
        settings_file=None, default_activation_performed=False))


def verify_runtime_exports(root: Path, runtime_root: Path, receipts: list[dict]) -> None:
    """Verify the permanent probability mirror without decoding its arrays."""
    for frame in receipts:
        runtime = read_json(frame["runtime_frame_binding"]["path"])
        record = runtime["probability_receipt"]
        expected = runtime_root / "layers" / frame["binding"]["run"] / f"frame_{frame['binding']['frame_index']:03d}" / "receipt.json"
        receipt_path = Path(record["path"])
        folder = Path(runtime["probability_folder"])
        if receipt_path != expected or folder != expected.parent or sha256_file(receipt_path) != record["sha256"]:
            raise ValueError("Permanent probability receipt differs from the immutable frame binding")
        exported = read_json(receipt_path)
        if exported.get("complete") is not True or exported["binding"]["provisional"] is not False:
            raise ValueError("Permanent probability mirror is incomplete or provisional")
        for name, artifact in exported["files"].items():
            if Path(name).name != name or sha256_file(folder / name) != artifact["sha256"]:
                raise ValueError("Permanent probability mirror artifact differs from its export receipt")


def seal_candidate(root: Path, plan: dict, completed: list[dict]) -> dict:
    sources = plan["binding"]["source_inventory"]
    required = {(row["run"], row["frame_index"]) for row in sources}
    if len(completed) != 864 or {(row["run"], row["frame_index"]) for row in completed} != required:
        raise ValueError("A Studio candidate requires complete exact43/864 frame coverage")
    receipts = [verify_frame_receipt(Path(row["receipt_path"]), root) for row in sorted(completed,
                  key=lambda item: (item["run"], item["frame_index"]))]
    verify_runtime_exports(root, Path(plan["binding"]["runtime_export_root"]), receipts)
    common = dict(configuration=CONFIGURATION, spec_sha256=SPEC_SHA256, g3_sha256=G3_SHA256,
                  regional_refit_sha256=REFIT_SHA256, co2_sha256=CO2_SHA256,
                  model_id=plan["model_identity"][0], model_sha256=plan["model_identity"][1])
    parity = [value["labelled_parity"] for value in receipts if value["labelled_parity"] is not None]
    first, _ = ordered_stages({(row["run"], row["frame_index"]):row for row in sources})
    if (len(parity) != 35 or {(row["run"], row["frame_index"]) for row in parity}
            != {(row["run"], row["frame_index"]) for row in first} or not all(row["passed"] for row in parity)):
        raise ValueError("Candidate lacks the35 independent labelled-frame parity checks")
    parity_path = root / "production_qa/parity_labelled_frames.json"
    union_path = root / "production_qa/co2_union_identity.json"
    atomic_json(parity_path, dict(contract=PARITY_CONTRACT, **common, frame_count=35, minimum_agreement=.999,
                                 all_pass=True, rows=parity, reference_method="independent-scipy-4-connected-literal-export-reconstruction"))
    unions = [value["union"] for value in receipts]
    if not all(row["exact"] and row["export_co2_sha256"] == row["production_co2_sha256"] == row["frozen_co2_sha256"] for row in unions):
        raise ValueError("At least one production frame changed the frozen CO2 union")
    atomic_json(union_path, dict(contract=UNION_CONTRACT, **common, run_count=43, frame_count=864,
                                all_pass=True, rows=unions, mask_hash_definition="SHA256(little-endian flattened packbits payload)"))
    packages = bind_run_packages(root, plan)
    assemble_saved_candidate(root, packages, plan)
    receipt = dict(contract=CONTRACT, **common, complete=True, model_identity=plan["model_identity"],
        pipeline_sha256=plan["pipeline_sha256"], run_count=43, frame_count=864,
        labelled_parity=dict(path=str(parity_path), sha256=sha256_file(parity_path)),
        co2_union_identity=dict(path=str(union_path), sha256=sha256_file(union_path)),
        frames=[{key:row[key] for key in ("run", "frame_index", "receipt_path", "receipt_sha256")}
                for row in sorted(completed, key=lambda item: (item["run"], item["frame_index"]))],
        source_plan_sha256=sha256_file(root / "approved_g3_candidate_plan.json"),
        phase_profile_sha256=sha256_file(root / "phase_optical_regions.json"),
        studio_preparation_sha256=sha256_file(root / "studio_preparation.json"),
        cpu_only=True, raw_inference_performed=False, retraining_performed=False,
        photographic_modes_reused=list(DISPLAY_MODES), scientific_acceptance=False,
        mass_eligible=False, automatic_activation=False)
    atomic_json(root / "production_candidate_receipt.json", receipt)
    return receipt


def build_candidate(*, parent: Path, experiment: Path, export_root: Path, output: Path,
                    spec: Path, selection: Path, workers: int = 2, threads: int = 1,
                    resume: bool = False, available_only: bool = False,
                    pilot: bool = False, poll_seconds: float = 30., nas_output: Path | None = None,
                    runtime_export_root: Path | None = None) -> dict:
    if not 1 <= workers <= 12 or threads not in (1, 2) or workers * threads > max(1, (os.cpu_count() or 1) - 4):
        raise ValueError("Approved candidate CPU allocation is bounded and must leave cores free")
    import psutil
    if psutil.virtual_memory().available < (workers * 2 + 4) * 1024 ** 3:
        raise ValueError("Insufficient available RAM for bounded native-map workers")
    if not 1 <= poll_seconds <= 60:
        raise ValueError("Producer polling interval must be1–60seconds")
    root, export_root = Path(output).resolve(), Path(export_root).resolve()
    plan = prepare_candidate(parent, experiment, export_root, root, spec, selection, resume=resume,
                             runtime_export_root=runtime_export_root)
    rows = plan["binding"]["source_inventory"]
    if pilot:
        rows = [row for row in rows if row["run"] == "ac28" and row["frame_index"] in (0, 8)]
    completed, done = [], set()
    with JobLease(root, plan["job_sha256"]):
        below_normal_priority()
        log = ProgressLog(root)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=worker_initializer, initargs=(threads,)) as pool:
            while len(done) != len(rows):
                available = [row for row in rows if (row["run"], row["frame_index"]) not in done
                    and (export_root / "layers" / row["run"] / f"frame_{row['frame_index']:03d}" / "receipt.json").is_file()]
                futures = {pool.submit(render_frame, str(root), str(export_root), row,
                    plan["pipeline_sha256"], threads, plan["binding"]["runtime_export_root"]):row for row in available}
                for future in as_completed(futures):
                    result = future.result()
                    key = (result["run"], result["frame_index"])
                    expected = futures[future]
                    if key != (expected["run"], expected["frame_index"]):
                        raise ValueError("Production worker returned another assigned frame")
                    completed.append(result)
                    done.add(key)
                    log.event("approved_candidate_frame", run=key[0], frame_index=key[1],
                              completed_frames=len(done), total_frames=len(rows))
                if len(done) != len(rows):
                    if available_only or pilot:
                        return dict(complete=False, completed_frames=len(done), total_frames=len(rows), candidate=str(root))
                    if (export_root / "completion_receipt.json").is_file():
                        raise ValueError("Completed producer is missing required probability receipts")
                    log.event("waiting_for_exports", completed_frames=len(done), total_frames=len(rows))
                    time.sleep(poll_seconds)
        if pilot:
            result = dict(contract=CONTRACT, pilot=True, complete=True,
                          frames=[{key:row[key] for key in ("run", "frame_index", "receipt_path", "receipt_sha256")}
                                  for row in sorted(completed, key=lambda item: (item["run"], item["frame_index"]))],
                          model_identity=plan["model_identity"], source_plan_sha256=sha256_file(root / "approved_g3_candidate_plan.json"))
            atomic_json(root / "production_pilot_receipt.json", result)
            return result
        sealed = seal_candidate(root, plan, completed)
        log.event("approved_candidate_complete", run_count=43, frame_count=864)
        if nas_output is not None:
            copy_tree_verified(root, Path(nas_output).resolve(), log=log)
        return sealed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("parent", "experiment", "export-root", "output", "spec", "selection"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--nas-output", type=Path)
    parser.add_argument("--runtime-export-root", type=Path)
    args = parser.parse_args(argv)
    build_candidate(parent=args.parent, experiment=args.experiment, export_root=args.export_root,
        output=args.output, spec=args.spec, selection=args.selection, workers=args.workers, threads=args.threads,
        resume=args.resume, available_only=args.available_only, pilot=args.pilot,
        poll_seconds=args.poll_seconds, nas_output=args.nas_output, runtime_export_root=args.runtime_export_root)


if __name__ == "__main__":
    main()
