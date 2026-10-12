"""Bind existing selected RAW sources to a new correction campaign.

This command copies receipt JSON only. It does not decode RAW, calculate phase
labels, alter source arrays, or overwrite a previous selection or candidate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path):
    return dict(path=str(path), sha256=sha(path))


def bound_json(descriptor):
    if sha(descriptor["path"]) != descriptor["sha256"]:
        raise ValueError("Changed source-bound metadata: " + descriptor["path"])
    return json.loads(Path(descriptor["path"]).read_text(encoding="utf-8-sig"))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    return asset(path)


def verify_import(receipt, source, parent_seal):
    if (receipt.get("complete") is not True
            or receipt.get("binding", {}).get("source") != source
            or receipt["binding"].get("frozen_parent") != parent_seal):
        raise ValueError("Imported source is not this exact acquisition/frozen parent")


def native_array(record, prefix):
    path, expected = record.get(prefix + "_path"), record.get(prefix + "_sha256")
    if not path or not expected:
        raise ValueError("Original native source omitted required descriptor: " + prefix)
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("Original native source asset is unavailable: " + path)
    stat = target.stat()
    return dict(path=path, sha256=expected, size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, source_SHA_verified_at_model_load=True)


def native_record(record):
    result = dict(log_linear_rgb=native_array(record, "v5_log_linear_rgb"),
                  valid_mask=native_array(record, "v5_log_linear_valid_mask"),
                  photographed_mask=native_array(record, "v5_photo_sand_mask"))
    if record.get("v5_extended_ring_path"):
        if "v5_extended_pad_px" not in record or "v5_extended_origin_xy" not in record:
            raise ValueError("A historical temporal ring lacks its declared geometry")
        result.update(temporal_ring=native_array(record, "v5_extended_ring"),
            origin_xy=list(record["v5_extended_origin_xy"]),
            pad_px=int(record["v5_extended_pad_px"]), origin_basis="explicit frozen source declaration")
    else:
        result.update(temporal_ring=None, origin_xy=[0, 0], pad_px=0,
                      origin_basis="original declared native crop; no synthetic ring")
    return result


def prepare(parent_job, new_job, candidate_name):
    parent_job, new_job = Path(parent_job), Path(new_job)
    if parent_job.resolve() == new_job.resolve():
        raise ValueError("Correction campaign must use a new output folder")
    old_plan_asset = asset(parent_job / "source_preparation/plan_with_pilot_imports.json")
    old_plan = bound_json(old_plan_asset)
    old_selection = asset(parent_job / "selection/selection.json")
    selection = bound_json(old_selection)
    if len(old_plan["runs"]) != 39 or selection["runs"] != list(old_plan["runs"]):
        raise ValueError("Corrected campaign must preserve the exact 39-run selection")
    inputs = new_job / "source_preparation"
    if (inputs / "plan.json").exists():
        raise FileExistsError("Source correction plan already exists")
    imports, import_rows, native_rows, inventories = {}, [], {}, {}
    for run, state in old_plan["runs"].items():
        inventory = bound_json(state["source_inventory"])
        native = bound_json(state["parent_native_manifest"])
        if native["run"] != run:
            raise ValueError("Original native manifest belongs to another run")
        baseline = native_record(native["frame_sources"][0])
        anchors = []
        for source in inventory["sources"]:
            old_receipt_path = parent_job / "selected_sources" / run / source["source_sha256"] / "receipt.json"
            if old_receipt_path.is_file():
                old_receipt = asset(old_receipt_path)
                saved = bound_json(old_receipt)
                verify_import(saved, source, old_plan["parent_chain"]["parent_seal"])
                target = new_job / "selected_sources" / run / source["source_sha256"] / "receipt.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if sha(target) != old_receipt["sha256"]:
                        raise ValueError("Existing staged receipt must not be overwritten")
                else:
                    with target.open("xb") as stream:
                        stream.write(old_receipt_path.read_bytes())
                imported = asset(target)
                if imported["sha256"] != old_receipt["sha256"]:
                    raise ValueError("Receipt byte equivalence failed")
                imports.setdefault(run, {})[source["source_sha256"]] = imported
                import_rows.append(dict(run=run, source_sha256=source["source_sha256"],
                    producer_receipt=old_receipt, staged_receipt=imported,
                    scientific_arrays_copied=False, producer_job_unchanged=True))
            if source.get("studio_anchor"):
                index = source["parent_source"]["parent_source_frame_index"]
                if index is None:
                    if source["parent_source"]["source_changed"] is not True:
                        raise ValueError("Unchanged anchor omitted its original native frame")
                    record = None
                else:
                    record = native["frame_sources"][index]
                cache = source["parent_source"].get("retained_source_cache")
                probability = None
                if cache is not None:
                    probability_receipt = cache["parent_probability_receipt"]
                    saved_probability = bound_json(probability_receipt)
                    probability = dict(receipt=probability_receipt,
                        extra_masks=saved_probability["files"]["extra_masks.packbits.npz"],
                        frozen_mask_order=saved_probability.get("extra_mask_names"),
                        original_inference_shape=saved_probability.get("shape_hw"))
                anchors.append(dict(source_sha256=source["source_sha256"],
                    frame_index=source["studio_frame_index"],
                    source_changed=source["parent_source"]["source_changed"],
                    native_input=native_record(record) if record is not None else None,
                    native_input_absent_reason=("Camera-clock revision has no historical native centre; preserve original full-rig masked branch" if record is None else None),
                    probability_input=probability))
        native_rows[run] = dict(parent_native_manifest=state["parent_native_manifest"],
            temporal_reference=native.get("v5_temporal_reference"),
            baseline=baseline, anchors=anchors,
            new_source_native_mask_policy="independently transported photographic geometry; transient masking only",
            extended_scientific_RAW_changed=False)
        inventories[run] = state["source_inventory"]
    import_proof = write_new(inputs / "source_imports/receipt.json", dict(
        contract="ffac-selected-corrected-source-imports-v1", complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), source_parent_plan=old_plan_asset,
        imported_count=len(import_rows), imports=import_rows,
        byte_exact_receipt_copies=True, original_assets_changed=False,
        science_copied_or_recomputed=False, original_producer_jobs_preserved=True))
    native_closure = write_new(inputs / "original_native_closure.json", dict(
        contract="ffac-selected-corrected-original-native-closure-v1", complete=True,
        source_parent_plan=old_plan_asset, runs=native_rows,
        run_count=len(native_rows), anchor_count=sum(len(r["anchors"]) for r in native_rows.values()),
        arrays_recomputed=False, giant_RAW_hashes_repeated=False,
        asset_existence_checked_on_primary_host=True,
        asset_SHA_verification_required_in_each_worker_session=True,
        current_baseline_and_selected_predecessor_masking_required=True,
        undeclared_temporal_ring_forbidden=True))
    ac59 = parent_job / "source_preparation/ac59_own_shape_recovery_v1/own_rig_binding.json"
    ac59_binding = asset(ac59)
    plan = deepcopy(old_plan)
    plan["source_imports"] = imports
    plan["binding"].update(corrected_campaign_parent_source_plan=old_plan_asset,
        corrected_campaign_root=str(new_job), consumer_candidate=candidate_name,
        exact_existing_source_imports=import_proof,
        original_native_mask_closure=native_closure,
        own_ac59_shape_binding=ac59_binding,
        source_imports=imports, source_arrays_changed=False,
        selection_contents_changed=False, model_or_labels_changed=False)
    plan["job_sha256"] = canonical_sha(plan["binding"])
    plan.update(source_import_proof=import_proof, original_native_closure=native_closure,
                source_geometry_overrides={"ac59": dict(own_persisted_shape_binding=ac59_binding)})
    plan_asset = write_new(inputs / "plan.json", plan)
    proof = write_new(inputs / "readiness.json", dict(
        contract="ffac-selected-corrected-source-readiness-v1", complete=True,
        source_plan=plan_asset, selection=old_selection,
        inventories=inventories, native_closure=native_closure, source_imports=import_proof,
        source_count=old_plan["totals"]["sources"], selected_count=old_plan["totals"]["selected"],
        imported_count=len(import_rows), candidate_name=candidate_name,
        source_preparation_executed=False, phase_inference_executed=False,
        new_worker_dispatch_requires_corrected_AC18_probe_pass=True,
        old_selection_TOML_source_maps_unchanged=True))
    return dict(plan=plan_asset, readiness=proof, imported_count=len(import_rows),
                native_closure=native_closure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job", type=Path, required=True)
    parser.add_argument("--new-job", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_job, args.new_job, args.candidate_name), indent=2))


if __name__ == "__main__":
    main()
