"""Import unchanged, scale-independent calibration summaries into a new fit.

Only completed calibration-run receipts are admitted. Both original producer
receipts and numeric summary values are preserved; an explicit equivalence
record explains the new campaign binding. No RAW or held-out data are read.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

CONTRACT = "ffac-common-native-fit-summary-import-v1"
CAMPAIGN_CONTRACT = "ffac-common-native-mass-campaign-v1"
ENGINE_MUTABLE_FUNCTIONS = frozenset({"fit_shared_mass_scales", "freeze_calibration"})
DRIVER_MUTABLE_FUNCTIONS = frozenset({"require_workflow_plan", "freeze_stage"})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def asset(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha_file(path), size_bytes=path.stat().st_size)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def bound_json(descriptor):
    path = Path(descriptor["path"])
    if sha_file(path) != descriptor["sha256"]:
        raise ValueError(f"Import source SHA changed: {path}")
    return read_json(path)


def protected_ast(path, ignored_functions):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not ignored_functions <= functions:
        raise ValueError("Declared fit/freeze functions are missing from the producing source")
    protected = [ast.dump(node, include_attributes=False) for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in ignored_functions]
    return stamp(protected)


def code_equivalence(source, target, ignored_functions, *, source_expected, target_expected):
    if sha_file(source) != source_expected or sha_file(target) != target_expected:
        raise ValueError("Code snapshot does not match its producer/target campaign SHA")
    old = protected_ast(source, ignored_functions)
    new = protected_ast(target, ignored_functions)
    if old != new:
        raise ValueError("Scale-independent summary-producing code changed outside the declared fit/freeze functions")
    return dict(source=asset(source), target=asset(target), protected_ast_sha256=old,
        mutable_functions=sorted(ignored_functions), protected_summary_computation_unchanged=True)


def code_digest(binding, filename):
    candidates = [digest for path, digest in binding["code_sha256"].items() if Path(path).name == filename]
    if len(candidates) != 1:
        raise ValueError("Campaign lacks a unique bound implementation: " + filename)
    return candidates[0]


def optical_inventory(job):
    receipt_path = Path(job) / "calibration" / "path_package_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("complete") is not True:
        raise ValueError("Native pooled optical package is incomplete")
    folder = Path(receipt["path_folder"]).resolve()
    inventory = {}
    for raw, expected in receipt["files"].items():
        path = Path(raw).resolve()
        try:
            relative = path.relative_to(folder)
        except ValueError as error:
            raise ValueError("Optical receipt contains an asset outside its package") from error
        if sha_file(path) != expected:
            raise ValueError("Native pooled optical component changed")
        inventory[str(relative)] = expected
    if not inventory:
        raise ValueError("Native pooled optical package has no verified assets")
    if "metadata.json" not in inventory:
        raise ValueError("Native pooled package lacks bound scientific metadata")
    metadata_path = folder / "metadata.json"
    metadata = read_json(metadata_path)
    source_bindings = metadata.get("source_bindings")
    calibration_runs = metadata.get("calibration_runs")
    if not isinstance(source_bindings, dict) or not calibration_runs or set(source_bindings) != set(calibration_runs):
        raise ValueError("Native optical metadata lacks exact calibration-run producer bindings")
    collection_fields = ("contract", "stage", "run", "frame_count", "camera_time_selection", "sampling_stride_px",
        "sampling_only_not_image_downsampling", "production_images_full_resolution", "frames", "heldout_pixels_read", "complete")
    collection_science = {}
    for run, value in source_bindings.items():
        if (not isinstance(value, dict) or not set(collection_fields) <= set(value) or
                value["complete"] is not True or value["heldout_pixels_read"] is not False or
                value["run"] != run or value["stage"] != "collect"):
            raise ValueError("Native optical producer binding is not a completed calibration-only collection")
        arrays = value.get("assets", {}).get("arrays")
        if not isinstance(arrays, dict) or not arrays.get("sha256"):
            raise ValueError("Native optical producer binding lacks a numerical sample-array SHA")
        collection_science[run] = {**{key: value[key] for key in collection_fields},
            "sample_arrays_sha256": arrays["sha256"]}
    numerical = {name: digest for name, digest in inventory.items() if name != "metadata.json"}
    return dict(folder=str(folder), receipt=asset(receipt_path), inventory=numerical,
        inventory_sha256=stamp(numerical), metadata=asset(metadata_path),
        protected_metadata={key: value for key, value in metadata.items() if key != "source_bindings"},
        collection_scientific_sha256=stamp(collection_science), producer_bindings_sha256=stamp(source_bindings),
        metadata_source_bindings_may_differ=True)


def validate_context(source_job, target_job, *, source_engine, target_engine, source_driver, target_driver):
    source_job, target_job = Path(source_job).resolve(), Path(target_job).resolve()
    if source_job == target_job:
        raise ValueError("Import must create a different campaign; never rewrite producer artifacts")
    if (target_job / "calibration" / "freeze" / "frozen_calibration.json").exists():
        raise ValueError("Calibration summaries cannot be inserted after the target campaign freeze")
    source_path, target_path = source_job / "campaign_binding.json", target_job / "campaign_binding.json"
    old, new = read_json(source_path), read_json(target_path)
    if old.get("contract") != CAMPAIGN_CONTRACT or new.get("contract") != CAMPAIGN_CONTRACT:
        raise ValueError("Unsupported campaign binding contract")
    for key in ("spec_sha256", "input_plan", "sampling_stride_px", "no_outflow_policy",
                "excluded_strip_fit_policy", "mass_fit_blocked_positive_policy", "no_map_mass_rescaling", "no_group_or_run_parameter"):
        if old.get(key) != new.get(key):
            raise ValueError("Scale-independent summary inputs/policy changed: " + key)
    if old.get("excluded_strip_fit_policy") != "physical-dose-interval-v1":
        raise ValueError("Only the preserved physical-dose-interval summary contract is transferable")
    for filename in ("phase_ffac_common_mass_inputs.py", "phase_ffac_common_mass_production.py"):
        if code_digest(old, filename) != code_digest(new, filename):
            raise ValueError("Scientific input/output adapter changed; summaries cannot be imported")
        for binding in (old, new):
            paths = [path for path in binding["code_sha256"] if Path(path).name == filename]
            if sha_file(paths[0]) != binding["code_sha256"][paths[0]]:
                raise ValueError("Bound scientific input/output adapter bytes changed")
    plan = bound_json(old["input_plan"])
    split = plan["split"]
    calibration = tuple(split["calibration_runs"])
    forbidden = set(split["held_out_runs"]) | set(split["excluded_runs"])
    if not calibration or set(calibration) & forbidden:
        raise ValueError("Source-bound calibration cohort is empty or overlaps held-out/excluded runs")
    engine = code_equivalence(source_engine, target_engine, ENGINE_MUTABLE_FUNCTIONS,
        source_expected=code_digest(old, "phase_ffac_common_mass.py"),
        target_expected=code_digest(new, "phase_ffac_common_mass.py"))
    driver = code_equivalence(source_driver, target_driver, DRIVER_MUTABLE_FUNCTIONS,
        source_expected=old["driver_sha256"], target_expected=new["driver_sha256"])
    original_optical, target_optical = optical_inventory(source_job), optical_inventory(target_job)
    if original_optical["inventory"] != target_optical["inventory"]:
        raise ValueError("Pooled native optical components are not byte-identical")
    if original_optical["protected_metadata"] != target_optical["protected_metadata"]:
        raise ValueError("Protected native optical scientific metadata changed")
    if original_optical["collection_scientific_sha256"] != target_optical["collection_scientific_sha256"]:
        raise ValueError("Source-bound calibration collection frames/statistics changed")
    return dict(source_job=source_job, target_job=target_job, source_binding=old, target_binding=new,
        source_binding_sha256=stamp(old), target_binding_sha256=stamp(new), calibration_runs=calibration,
        forbidden_runs=forbidden, source_campaign=asset(source_path), target_campaign=asset(target_path),
        engine_equivalence=engine, driver_equivalence=driver,
        source_optical=original_optical, target_optical=target_optical)


def safe_relative_asset(raw):
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Summary receipt asset must stay inside its committed run folder")
    return path


def import_run(context, run):
    if run not in context["calibration_runs"] or run in context["forbidden_runs"]:
        raise ValueError("Held-out or excluded summary imports are forbidden")
    source = context["source_job"] / "work" / "fit_summaries" / run
    receipt_path = source / "receipt.json"
    payload = read_json(receipt_path)
    if (payload.get("complete") is not True or payload.get("contract") != CAMPAIGN_CONTRACT or
            payload.get("stage") != "summaries" or payload.get("run") != run or
            payload.get("campaign_binding_sha256") != context["source_binding_sha256"] or
            payload.get("heldout_pixels_read") is not False or
            payload.get("excluded_strip_fit_policy") != "physical-dose-interval-v1"):
        raise ValueError("Source summary receipt is incomplete, unbound, or not calibration-only")
    if Path(payload["path_folder"]).resolve() != Path(context["source_optical"]["folder"]):
        raise ValueError("Source summaries used another native pooled optical package")
    for summary in payload.get("summaries", ()):
        if summary.get("run") != run:
            raise ValueError("Source summary rows contain another run")
    source_asset = asset(receipt_path)
    summaries_sha = stamp(payload.get("summaries", []))
    provenance = dict(contract=CONTRACT, source_receipt=source_asset,
        source_campaign_binding=context["source_campaign"], source_campaign_binding_sha256=context["source_binding_sha256"],
        target_campaign_binding=context["target_campaign"], target_campaign_binding_sha256=context["target_binding_sha256"],
        summaries_sha256=summaries_sha, rejected_frames_sha256=stamp(payload.get("rejected_frames", [])),
        source_engine_sha256=context["engine_equivalence"]["source"]["sha256"],
        source_driver_sha256=context["driver_equivalence"]["source"]["sha256"],
        source_optical_inventory_sha256=context["source_optical"]["inventory_sha256"],
        imported_values_byte_equivalent=True, new_scientific_evaluation_performed=False,
        numeric_summary_representation="Source JSON objects preserved; exact canonical JSON SHA; original receipt bytes preserved separately",
        heldout_pixels_read=False, original_producer_receipt_preserved=True)
    destination = context["target_job"] / "work" / "fit_summaries" / run
    if destination.exists():
        saved = read_json(destination / "receipt.json")
        if (saved.get("import_provenance") != provenance or
                saved.get("campaign_binding_sha256") != context["target_binding_sha256"] or
                stamp(saved.get("summaries", [])) != summaries_sha):
            raise ValueError("Existing imported receipt differs; never overwrite it")
        for descriptor in saved.get("assets", {}).values():
            path = destination / safe_relative_asset(descriptor["path"])
            if sha_file(path) != descriptor["sha256"]:
                raise ValueError("Imported receipt asset changed")
        return dict(run=run, source=source_asset, imported=asset(destination / "receipt.json"), summaries_sha256=summaries_sha, cached=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".partial")
    temporary.mkdir(exist_ok=False)
    assets = {}
    for key, descriptor in payload.get("assets", {}).items():
        if key == "producer_receipt":
            raise ValueError("Producer asset key collides with immutable receipt preservation")
        relative = safe_relative_asset(descriptor["path"])
        original = source / relative
        if sha_file(original) != descriptor["sha256"]:
            raise ValueError("Original summary asset changed")
        copied = temporary / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
        if sha_file(copied) != descriptor["sha256"]:
            raise ValueError("Summary import did not preserve asset bytes")
        assets[key] = dict(descriptor)
    original_copy = temporary / "producer_receipt.json"
    shutil.copyfile(receipt_path, original_copy)
    if sha_file(original_copy) != source_asset["sha256"]:
        raise ValueError("Original producer receipt bytes changed during import")
    assets["producer_receipt"] = dict(path=original_copy.name, sha256=source_asset["sha256"])
    imported = dict(payload, campaign_binding_sha256=context["target_binding_sha256"],
        path_folder=context["target_optical"]["folder"], assets=assets, import_provenance=provenance)
    if stamp(imported.get("summaries", [])) != summaries_sha:
        raise ValueError("Import changed scale-independent numerical summary values")
    write_new(temporary / "receipt.json", imported)
    temporary.rename(destination)
    return dict(run=run, source=source_asset, imported=asset(destination / "receipt.json"), summaries_sha256=summaries_sha,
        numeric_summary_count=len(payload.get("summaries", [])), omitted_calibration_run=payload.get("omitted_calibration_run", False), cached=False)


def import_summaries(source_job, target_job, *, source_engine, target_engine, source_driver, target_driver, runs=None):
    context = validate_context(source_job, target_job, source_engine=source_engine, target_engine=target_engine,
        source_driver=source_driver, target_driver=target_driver)
    selected = tuple(context["calibration_runs"] if runs is None else runs)
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(context["calibration_runs"]):
        raise ValueError("Import selection must contain distinct calibration runs only")
    records = [import_run(context, run) for run in selected]
    receipt = context["target_job"] / ("summary_import_receipt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8] + ".json")
    evidence = {key: context[key] for key in ("source_campaign", "target_campaign", "source_binding_sha256", "target_binding_sha256", "engine_equivalence", "driver_equivalence", "source_optical", "target_optical")}
    write_new(receipt, dict(contract=CONTRACT, complete=True, run_count=len(records), all_calibration_runs_imported=set(selected)==set(context["calibration_runs"]),
        imports=records, equivalence=evidence, no_new_scientific_evaluation=True, no_raw_images_read=True,
        heldout_pixels_read=False, source_artifacts_unchanged=True, imported_producer_statistics_not_new_inference=True))
    return dict(receipt=str(receipt), receipt_sha256=sha_file(receipt), run_count=len(records), cached_count=sum(r["cached"] for r in records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-job", "target-job", "source-engine", "target-engine", "source-driver", "target-driver"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runs", nargs="+")
    args = parser.parse_args(argv)
    result = import_summaries(args.source_job, args.target_job, source_engine=args.source_engine,
        target_engine=args.target_engine, source_driver=args.source_driver, target_driver=args.target_driver, runs=args.runs)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
