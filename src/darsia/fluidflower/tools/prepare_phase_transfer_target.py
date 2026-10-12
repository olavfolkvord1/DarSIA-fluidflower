"""Prepare label-free FF-AC inputs from a registered target rig and raw timeline.

This does not load a phase model, colour-path embedding, mass fit, or reviewed
labels. The existing campaign contributes only its explicitly listed raw frames.
All corrections and geometry come from the target's configured serialized rig.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root
from darsia.fluidflower.storage_paths import resolve_storage_path

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tomllib

import numpy as np

REPO = series_root("ffac")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def canonical(path):
    return resolve_storage_path(path).resolve()


def coordinate_arrays(facies, metadata, baseline_shape):
    """Use registered physical metadata, never an assumed pixel-to-metre ratio."""
    if (facies.shape != tuple(baseline_shape[:2]) or facies.ndim != 2
            or not np.issubdtype(facies.dtype, np.integer)):
        raise ValueError("Categorical facies and corrected baseline grids differ")
    if metadata.get("space_dim") != 2 or metadata.get("indexing") != "ij":
        raise ValueError("Unsupported registered rig coordinate system")
    height, width = np.asarray(metadata["dimensions"], dtype=float)
    ox, oy = np.asarray(metadata["origin"], dtype=float)
    if not np.isfinite([height, width, ox, oy]).all() or min(height, width) <= 0:
        raise ValueError("Invalid registered dimensions or origin")
    h, w = facies.shape
    return ox + (np.arange(w) + .5) * width / w, oy - (np.arange(h) + .5) * height / h


def checked_timeline(run, native, experiment, baseline_path, allowed_folders):
    """Retain exact source selection while checking it against target protocols."""
    if native.get("run", "").casefold() != run:
        raise ValueError("Native source belongs to another experiment")
    rows = native.get("frame_sources", [])
    if len(rows) != native.get("frame_count") or not rows:
        raise ValueError("Incomplete source timeline")
    allowed = {canonical(folder) for folder in allowed_folders}
    baseline_path = canonical(baseline_path)
    baseline_date = experiment.get_datetime(baseline_path)
    baseline_time = experiment.time_since_start(baseline_date)
    if baseline_path.parent not in allowed or baseline_time >= 0:
        raise ValueError("Configured baseline is not an own-target pre-injection image")
    result = [dict(frame_index=0, elapsed_h=baseline_time, acquired_elapsed_h=baseline_time,
        acquired_at=baseline_date.isoformat(), source_name=baseline_path.name,
        source_path=str(baseline_path), selection_status="pre_injection_negative_control")]
    seen = {baseline_path}
    previous = 0.
    for index, row in enumerate(rows):
        path = canonical(row["source_path"])
        if row.get("frame_index") != index or path.parent not in allowed or path in seen:
            raise ValueError("Wrong-run, duplicate or unordered raw frame")
        date = experiment.get_datetime(path)
        elapsed = float(experiment.time_since_start(date))
        if (not np.isfinite(elapsed) or elapsed <= previous
                or abs(elapsed - float(row["elapsed_h"])) > 1e-7
                or (row.get("acquired_at") and datetime.fromisoformat(row["acquired_at"]) != date)):
            raise ValueError(f"Source timeline disagrees with target protocol: {path.name}")
        result.append(dict(frame_index=index + 1, elapsed_h=elapsed, acquired_at=date.isoformat(),
            source_name=path.name, source_path=str(path), selection_status="exact",
            source_timeline_frame_index=index))
        seen.add(path)
        previous = elapsed
    return result


def _snapshot(path, destination, records):
    path, destination = Path(path), Path(destination)
    before = digest(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    if digest(destination) != before or digest(path) != before:
        raise RuntimeError(f"Source changed while freezing: {path}")
    records[str(path)] = dict(sha256=before, snapshot=str(destination))
    return destination


def prepare(run, source_native, output, *, config_path=None):
    import cv2
    import darsia
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as shared
    from darsia.fluidflower.tools.calibration_review_movie import render_contour_frame
    from darsia.fluidflower.tools.phase_injection_points import project_injection_points
    from darsia.fluidflower.phase_plume import PlumeContext

    if not re.fullmatch(r"ac\d+", run):
        raise ValueError("This preparer requires an explicitly configured FF-AC run")
    output, source_native = Path(output).resolve(), Path(source_native).resolve()
    config_path = Path(config_path) if config_path is not None else project_path('config_seg6/run_ac') / f"{run}.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("id", {}).get("run_id", "").casefold() != f"albus {run}":
        raise ValueError("Target configuration identity mismatch")
    from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
    from darsia.fluidflower.series_paths import config_files
    native_config = FluidFlowerConfig([series_path("ffac", "common_config"), config_path], require_data=True, require_results=False)
    folders = [canonical(p) for p in config["data"]["folders"]]
    baseline_name = config["data"].get("baseline")
    baselines = [folder / baseline_name for folder in folders if baseline_name and (folder / baseline_name).is_file()]
    if len(baselines) != 1:
        raise ValueError("An explicit unambiguous existing target baseline path is required")
    baseline_path = baselines[0]
    rig_source = canonical(native_config.rig.path)
    if canonical(read(rig_source / "meta_data.json")["baseline_path"]) != baseline_path:
        raise ValueError("Target rig baseline differs from registered configuration")
    folders = [canonical(p) for p in config["data"]["folders"]]
    pattern = rf"/albus/raw data/{re.escape(run)}/"
    if any(not re.search(pattern, str(p).replace("\\", "/").casefold() + "/") for p in folders):
        raise ValueError("Configured raw folders do not identify the target experiment")
    native = read(source_native)
    protocol = config["protocols"]
    if {canonical(p) for p in protocol["imaging"]} != set(folders):
        raise ValueError("Target imaging protocol must bind every raw folder exactly")
    for source in (rig_source, source_native.parent, package_root()):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Target output must be separate from original sources")
    if output.exists():
        raise FileExistsError("Use a fresh target directory; previous attempts are preserved")

    # These are the existing per-image uniform24 settings; no colour-path or
    # phase calibration is constructed. A private rig copy prevents any writes
    # through legacy loader helpers from reaching the registered source.
    records = {}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    _snapshot(config_path, snapshot / "config" / config_path.name, records)
    common_path = project_path('config_seg6/common.toml')
    _snapshot(common_path, snapshot / "config/common.toml", records)
    _snapshot(project_path('data/facies.csv'), snapshot / "config/facies.csv", records)
    stamp = config_path.parent / ".color_state" / f"{run}.txt"
    if stamp.is_file():
        _snapshot(stamp, snapshot / "config" / f"{run}_color_state.txt", records)
    _snapshot(source_native, snapshot / "source_timeline.json", records)
    for name in ("scripts/prepare_phase_transfer_target.py", "scripts/schema2_phase_transfer_campaign.py",
                 "scripts/ac_overflow_first_colorchecker_screen.py", "scripts/ac_colorchecker_fieldwise_screen.py",
                 "scripts/ac_colorchecker_refinement_colorpath_pilot.py", "scripts/auto_calibrate_color_to_mass.py",
                 "scripts/calibration_pairwise_review.py", "scripts/calibration_review_movie.py",
                 "external/darsia/src/darsia/presets/workflows/rig.py",
                 "external/darsia/src/darsia/corrections/color/colorcorrection.py"):
        path = source_path(name)
        if path.is_file():
            _snapshot(path, snapshot / "code" / name, records)
    rig_root = snapshot / "rig"
    for path in sorted(rig_source.iterdir()):
        if path.is_file() and path.suffix in {".npz", ".json"}:
            _snapshot(path, rig_root / path.name, records)
    recovery_path = rig_root / "FFAC_COLORON_RECOVERY_MANIFEST.json"
    if recovery_path.is_file():
        recovery = read(recovery_path)
        if recovery.get("run") != run:
            raise ValueError("Recovery manifest identifies another run")
        for name, expected in recovery.get("files", {}).items():
            if digest(rig_root / name) != expected["sha256"]:
                raise ValueError(f"Registered recovered rig checksum differs: {name}")
    protocol_paths = {}
    for relative in [protocol["injection"], protocol["pressure_temperature"], *protocol["imaging"].values()]:
        if not isinstance(relative, str):
            raise ValueError("This preparation supports explicit CSV protocols")
        original = (REPO / relative).resolve()
        protocol_paths[relative] = _snapshot(original, snapshot / "protocols" / original.name, records)
    experiment = darsia.ProtocolledExperiment(
        data=[baseline_path, *[canonical(row["source_path"]) for row in native["frame_sources"]]],
        imaging_protocol={canonical(folder): protocol_paths[p] for folder, p in protocol["imaging"].items()},
        injection_protocol=protocol_paths[protocol["injection"]],
        pressure_temperature_protocol=protocol_paths[protocol["pressure_temperature"]])
    frames = checked_timeline(run, native, experiment, baseline_path, folders)
    for row in frames:
        row["source_sha256"] = digest(row["source_path"])
    stages = shared._injection_stages(experiment)
    environment = shared.optical_control._profile_environment(shared._profile())
    cv2.setNumThreads(4)
    shared.optical_control.install_runtime_hooks()
    print(f"{run}: frozen inputs; loading own rig; {len(frames)} frames", flush=True)
    with shared.optical_control.rebuilt._temporary_environment(environment):
        rig = shared.optical_control.rebuilt.RefinedColorPathRig.load(rig_root)
        rig.load_experiment(experiment)
        baseline = np.asarray(rig.baseline.img, dtype=np.float32)[..., :3]
        if not np.isfinite(baseline).all():
            raise ValueError("Non-finite corrected baseline")
        with np.load(rig_root / "facies.npz", allow_pickle=True) as archive:
            facies, metadata = archive["array"], archive["metadata"].item()
        x_m, y_m = coordinate_arrays(facies, metadata, baseline.shape)
        if (not np.allclose(metadata["dimensions"], rig.baseline.dimensions)
                or not np.allclose(metadata["origin"], rig.baseline.origin)
                or rig.baseline.indexing != "ij"):
            raise ValueError("Baseline and facies have different registered physical grids")
        with np.load(rig_root / "labels.npz", allow_pickle=False) as archive:
            labels = archive["array"]
        mapping = tomllib.loads(common_path.read_text(encoding="utf-8-sig"))["facies"]
        if labels.shape != facies.shape:
            raise ValueError("Label and facies grids differ")
        covered = np.zeros(facies.shape, dtype=bool)
        mapping_audit = []
        for key, entry in mapping.items():
            if not key.isdecimal():
                continue
            for label in entry["labels"]:
                selected = labels == label
                if not selected.any():
                    continue
                values, counts = np.unique(facies[selected], return_counts=True)
                dominant = int(values[np.argmax(counts)])
                if dominant != int(key):
                    raise ValueError("Target facies have a different categorical identity from the common mapping")
                mapping_audit.append(dict(label=int(label),expected_facies=int(key),dominant_facies=dominant,
                    pixels=int(counts.sum()),legacy_boundary_mismatch_pixels=int(counts[values != int(key)].sum())))
                covered |= selected
        if not covered.all():
            raise ValueError("Target labels are missing from the common facies mapping")
        with np.load(rig_root / "permeability.npz", allow_pickle=False) as archive:
            permeability = np.asarray(archive["array"], dtype=np.float32)
        # Match the frozen FF-AC physical-context convention: only free water
        # (facies 0) is excluded; image-porosity seams never become hard holes.
        domain = facies != 0
        PlumeContext(domain, facies, x_m, y_m, stages, permeability)
        review_root = output / "manual_pairwise_review"
        context_path = output / "target_context" / f"{run}.npz"
        context_path.parent.mkdir(parents=True)
        np.savez_compressed(context_path, domain=domain, facies=facies, x_m=x_m, y_m=y_m, permeability=permeability)
        width, height = shared.review._native_review_dimensions(baseline.shape)
        box = shared.review._native_content_box(baseline.shape, width, height)
        registrations = []
        baseline_diagnostics = shared._correction_diagnostics(rig)
        for row in frames:
            index, path = row["frame_index"], Path(row["source_path"])
            image = baseline if index == 0 else np.asarray(rig.read_image(path, dtype=np.float32).img, dtype=np.float32)[..., :3]
            if image.shape != baseline.shape or not np.isfinite(image).all():
                raise ValueError(f"Corrected image differs from the baseline grid: {path}")
            row["corrected_array_sha256"] = shared._array_sha256(image)
            row["correction_diagnostics"] = baseline_diagnostics if index == 0 else shared._correction_diagnostics(rig)
            if index:
                dx, dy, response, region = shared.review._estimate_review_translation(baseline, image, header_height=0)
            else:
                dx, dy, response, region = 0., 0., 1., "identical_baseline"
            registrations.append(dict(frame_index=index, dx_px=float(dx), dy_px=float(dy), response=float(response),
                region=region, source="native_corrected_images", dx_canvas_fraction=float(dx)/width, dy_canvas_fraction=float(dy)/height))
            zeros = np.zeros(image.shape[:2], dtype=np.float32)
            canvas = render_contour_frame(image=image, concentration_aq=zeros, saturation_g=zeros,
                run=run, variant="Prepared target input", frame_index=index, frame_count=len(frames),
                elapsed_h=row["elapsed_h"], injected_mass_kg=0., detected_total_mass_kg=0.,
                detected_gas_mass_kg=0., detected_aqueous_mass_kg=0., width=width, height=height,
                gas_fill_alpha=0., show_aqueous=False, show_gas=False, preserve_native_resolution=True,
                header_title=f"{run.upper()} | frame {index + 1}/{len(frames)} | t={row['elapsed_h']:.3f} h | corrected target input",
                header_mass="Target rig and protocol only | no phase predictions or fitted mass calibration", status_text="")
            target = review_root / "native_frames" / run / f"frame_{index:03d}.jpg"
            shared._write_jpeg(target, canvas)
            if index == 0:
                reference = review_root / "native_references" / run / "pre_injection.jpg"
                reference.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, reference)
            row["native_jpeg_sha256"] = digest(target)
            if digest(path) != row["source_sha256"]:
                raise RuntimeError("Raw input changed during preparation")
            print(f"{run}: corrected native frame {index + 1}/{len(frames)}", flush=True)

    native_path = review_root / "native_frames" / run / "manifest.json"
    correction_contract = dict(profile="uniform24", environment=environment,
        source="own registered FF-AC rig with shared per-image ColorChecker normalization",
        rig_source=str(rig_source), rig_snapshot=str(rig_root), mass_fit_loaded=False,
        runtime_provenance=rig._ffac_colorchecker_refinement_provenance)
    write(native_path, dict(schema=1, dataset="ffac", run=run, kind="native_resolution_corrected_review_frames",
        frame_count=len(frames), canvas_wh=[width,height], analysis_image_shape_hw=list(baseline.shape[:2]),
        content_box_xywh=list(box), full_analysis_resolution=True, additional_spatial_resampling=False,
        jpeg_quality=98, frame_sources=frames, reference_registration=registrations,
        injection_points=project_injection_points(stages, x_m=x_m, y_m=y_m,
            canvas_wh=[width, height], content_box=box),
        timeline_revision=dict(kind="prepend-evaluated-baseline-v1",
            old_to_new_frame_index={str(i):i+1 for i in range(len(frames)-1)},
            source_manifest=str(source_native),source_manifest_sha256=digest(source_native),
            existing_predictions_recomputed=False,previous_predictions_used=False),
        reference=dict(kind="radiometrically_corrected_pre_injection_baseline", source_path=str(baseline_path),
            source_name=baseline_path.name, source_shape_hw=list(baseline.shape[:2]), content_box_xywh=list(box),
            included_in_timeline=True, timeline_frame_index=0, timeline_selection_status="pre_injection_negative_control",
            corrected_array_sha256=frames[0]["corrected_array_sha256"], model_evaluated_with_same_phase_pipeline=False),
        domain=dict(path=str(context_path),sha256=digest(context_path)),
        phase_topology=dict(injection_stages=list(stages)), correction_contract=correction_contract,
        phase_evaluation_domain=dict(name="registered_porous_medium_2d", continuous_image_porosity_used_as_hard_mask=False),
        created_utc=datetime.now(timezone.utc).isoformat()))
    manifest_path = output / "postflight_manifest.json"
    provenance = dict(target_binding=dict(dataset="ffac",run=run,native_manifest_sha256=digest(native_path),
        source_manifest=str(manifest_path)), rig_source=str(rig_source), rig_snapshot=str(rig_root),
        geometry=dict(indexing="ij", dimensions=list(map(float,metadata["dimensions"])),
            origin=list(map(float,metadata["origin"])), source="trusted target rig facies metadata",
            excluded_facies_ids=[0], categorical_identity_verified=True,
            categorical_mapping_audit=mapping_audit,
            legacy_boundary_values_retained=True,
            facies_sha256=digest(rig_root / "facies.npz"),
            permeability_sha256=digest(rig_root / "permeability.npz")),
        input_snapshot=records, correction_contract=correction_contract,
        source_timeline=str(source_native), source_timeline_sha256=digest(source_native),
        training_labels_used=False, mass_calibration_used=False, phase_model_loaded=False,
        historical_exposure="Target appeared in an older Objective V2 mass-calibration review; not claimed globally unseen")
    descriptor = dict(format="phase-transfer-target-context-v1",dataset="ffac",run=run,
        native_manifest_sha256=digest(native_path),path=str(context_path),sha256=digest(context_path),
        injection_stages=list(stages),facies_identity="same-series-categorical-labels",provenance=provenance)
    candidate = output / "prepared_inputs" / run
    candidate.mkdir(parents=True)
    manifest = dict(schema=1,dataset="ffac",selection_id=output.name,
        kind="label-free-prepared-phase-transfer-target",include_in_training=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        jobs=[dict(job_id=f"ffac_{run}_prepared_target",dataset="ffac",run=run,
            candidate_folder=str(candidate),review_output_dir=str(review_root),
            analysis_source_review_root=str(review_root),independent_reference_frames=[0],phase_transfer_context=descriptor)])
    for name, record in records.items():
        if digest(name) != record["sha256"] or digest(record["snapshot"]) != record["sha256"]:
            raise RuntimeError(f"Source or frozen snapshot changed: {name}")
    artifacts = {str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}
    report = dict(status="prepared_target_inputs_ready",manifest=str(manifest_path),dataset="ffac",run=run,
        frame_count=len(frames),native_shape=list(baseline.shape[:2]),native_manifest=str(native_path),
        native_manifest_sha256=digest(native_path),context=str(context_path),
        source_inputs_unchanged=True,phase_inference_run=False,training_run=False,
        baseline_jpeg_identical_to_frame_zero=digest(reference)==frames[0]["native_jpeg_sha256"],
        correction_flagged_frames=[row["frame_index"] for row in frames if any(d["flagged"] for d in row["correction_diagnostics"])],
        artifacts_sha256=artifacts)
    write(output / "preparation_report.json",report)
    # The discoverable campaign manifest is the completion marker. Never expose
    # a partial target while corrections or immutable-input checks are pending.
    write(manifest_path, manifest)
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts_sha256'},indent=2),flush=True)
    return report


def isolate_prepared_baseline(source, output):
    """Create a separate timeline revision, reusing verified label-free assets.

    This supports the initial prepared export that displayed its own baseline
    at t=0. No corrected image, physical array or historical source is changed.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(source) or source.is_relative_to(output)):
        raise ValueError("Baseline revision needs a fresh separate directory")
    source_manifest = source / "postflight_manifest.json"
    if not source_manifest.is_file():
        source_manifest = source / "manifest.json"
    report, manifest = read(source / "preparation_report.json"), read(source_manifest)
    if (manifest.get("kind") != "label-free-prepared-phase-transfer-target"
            or report.get("phase_inference_run") is not False or report.get("training_run") is not False
            or len(manifest.get("jobs", [])) != 1):
        raise ValueError("Only a completed label-free target can be revised")
    for relative, expected in report["artifacts_sha256"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError("Prepared source asset changed")
    original_hashes = {str(p.relative_to(source)):digest(p) for p in source.rglob("*") if p.is_file()}
    job = manifest["jobs"][0]
    native_relative = Path("manual_pairwise_review/native_frames") / job["run"] / "manifest.json"
    original_native = read(source / native_relative)
    first = original_native["frame_sources"][0]
    actual = float(first["acquired_elapsed_h"])
    if (not np.isfinite(actual) or actual >= 0 or first["elapsed_h"] != 0
            or first["selection_status"] != "pre_injection_negative_control"
            or not report["baseline_jpeg_identical_to_frame_zero"]):
        raise ValueError("Initial baseline must have an explicit verified negative acquisition time")
    def relocate(value):
        if isinstance(value, dict): return {k:relocate(v) for k,v in value.items()}
        if isinstance(value, list): return [relocate(v) for v in value]
        if isinstance(value, str):
            for old,new in ((str(source),str(output)),(source.as_posix(),output.as_posix())):
                if value==old or value.startswith(old+"/") or value.startswith(old+"\\"):
                    return new+value[len(old):]
        return value
    output.mkdir(parents=True)
    for relative in original_hashes:
        if relative in {source_manifest.name, "preparation_report.json", str(native_relative)}:
            continue
        target = output / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source / relative,target)
    # Preserve the original preparation receipt and native manifest verbatim.
    previous = output / "source_snapshot/initial_preparation"
    previous.mkdir(parents=True)
    for name,path in (("report.json",source / "preparation_report.json"),
                      ("manifest.json",source_manifest),
                      ("native_manifest.json",source / native_relative)):
        shutil.copy2(path,previous / name)
    native = relocate(original_native)
    native["frame_sources"][0]["elapsed_h"] = actual
    native["timeline_revision"] = dict(kind="prepend-evaluated-baseline-v1",
        source_manifest=str(source / native_relative),source_manifest_sha256=digest(source / native_relative),
        old_to_new_frame_index={str(i):i for i in range(native["frame_count"])},
        existing_predictions_recomputed=False,previous_predictions_used=False,
        baseline_time_source="verified own-target imaging protocol; baseline header retains reference t=0")
    write(output / native_relative,native)
    manifest = relocate(manifest)
    job = manifest["jobs"][0]
    job["independent_reference_frames"] = [0]
    descriptor = job["phase_transfer_context"]
    descriptor["native_manifest_sha256"] = digest(output / native_relative)
    descriptor["provenance"]["target_binding"]["native_manifest_sha256"] = descriptor["native_manifest_sha256"]
    descriptor["provenance"]["target_binding"]["source_manifest"] = str(output / "postflight_manifest.json")
    descriptor["provenance"]["baseline_isolation"] = dict(source_preparation=str(source),
        native_images_and_context_unchanged=True, elapsed_h=actual)
    # Candidate metadata has no fitted output and remains an empty placeholder.
    Path(job["candidate_folder"]).mkdir(parents=True,exist_ok=True)
    manifest["selection_id"] = output.name
    report = relocate(report)
    report.update(manifest=str(output / "postflight_manifest.json"),native_manifest_sha256=descriptor["native_manifest_sha256"],
        independent_reference_frames=[0],baseline_elapsed_h=actual,
        source_preparation=str(source),source_preparation_sha256=original_hashes)
    code = output / "source_snapshot/baseline_revision_code.py"
    shutil.copy2(Path(__file__),code)
    for relative,expected in original_hashes.items():
        if digest(source / relative) != expected:
            raise ValueError("Original prepared target changed during isolation")
    for relative,expected in original_hashes.items():
        if Path(relative).suffix != ".json" and digest(output / relative) != expected:
            raise ValueError("Revision altered a corrected image or physical archive")
    report["artifacts_sha256"] = {str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}
    write(output / "preparation_report.json",report)
    write(output / "postflight_manifest.json",manifest)
    print(json.dumps(dict(status=report["status"],manifest=str(output / "postflight_manifest.json"),
        baseline_elapsed_h=actual,frames=native["frame_count"],images_and_context_unchanged=True)),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--config", type=Path,
        help="Explicit frozen own-run configuration (original inputs remain unchanged)")
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-native",type=Path)
    modes.add_argument("--isolate-prepared",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.isolate_prepared:
        isolate_prepared_baseline(args.isolate_prepared,args.output)
    else:
        if not args.run:
            parser.error("--run is required with --source-native")
        prepare(args.run,args.source_native,args.output,config_path=args.config)


if __name__ == "__main__":
    main()
