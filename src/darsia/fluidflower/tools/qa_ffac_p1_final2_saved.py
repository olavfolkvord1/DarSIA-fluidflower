"""Independent read-only verification of committed P1-final2 output identities.

This does not run the numerical rules, any distance transform, inference or
scoring. It reconstructs the four recorded changes from saved SHA-bound files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .refit_ffac_g3_downstream import atomic_json, canonical_sha256, utc_now

CONTRACT = "ffac-p1-final2-independent-saved-identities-v1"
PRODUCTION_CONTRACT = "ffac-g3-t4c-p1-final2-candidate-v1"
SPEC_SHA256 = "e843e61751b795984550cfe853548ea7ca4e50364a5cb44681621d485abd2c0b"
MASK_KEYS = dict(A="below_sand_floor_masks", B="floor_connectivity_masks",
                 C="floor_contact_fill_v2", D="gas_envelope_masks")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def asset_bytes(asset):
    require(isinstance(asset, dict) and isinstance(asset.get("path"), str), "Missing bound asset")
    path = Path(asset["path"])
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Asset changed while read")
    digest = hashlib.sha256(data).hexdigest()
    require(digest == asset.get("sha256"), f"Asset SHA mismatch: {path}")
    return data


def array(asset):
    archive = np.load(io.BytesIO(asset_bytes(asset)), allow_pickle=False)
    if isinstance(archive, np.lib.npyio.NpzFile):
        try:
            return archive[asset.get("array_key", "pixel_area_mm2")].copy()
        finally:
            archive.close()
    return archive


def classes(asset, shape):
    with np.load(io.BytesIO(asset_bytes(asset)), allow_pickle=False) as archive:
        result = {key:archive[key].copy() for key in archive.files}
    require(all(value.dtype == bool and value.shape == shape for value in result.values()), "Physical class grid changed")
    return result


def labels(asset, shape=None):
    with Image.open(io.BytesIO(asset_bytes(asset))) as image:
        result = np.array(image)
    require(result.dtype == np.uint8 and result.ndim == 2, "Labels are not a uint8 category grid")
    require(shape is None or result.shape == shape, "Label grid differs")
    require(set(np.unique(result)).issubset({0,1,2,255}), "Unsupported saved category")
    return result


def packed(asset, source, shape):
    require(asset.get("storage") == "packbits-npy" and asset.get("shape_hw") == list(shape)
        and asset.get("source_sha256") == source["source_sha256"]
        and asset.get("frame_index") == source["frame_index"]
        and asset.get("contract") == PRODUCTION_CONTRACT, "Packed mask lost source/grid binding")
    value = array(asset)
    size = int(np.prod(shape))
    require(value.dtype == np.uint8 and value.ndim == 1 and len(value) == (size+7)//8, "Invalid packbits storage")
    if size % 8:
        require(not np.unpackbits(value)[size:].any(), "Packed mask has nonzero padding")
    return np.unpackbits(value, count=size).reshape(shape).astype(bool)


def previous(asset, source, shape):
    require(asset.get("storage") == "npz-array" and asset.get("array_key") == "previous_labels"
        and asset.get("shape_hw") == list(shape) and asset.get("source_sha256") == source["source_sha256"]
        and asset.get("frame_index") == source["frame_index"] and asset.get("contract") == PRODUCTION_CONTRACT,
        "Previous labels lost source/grid binding")
    result = array(asset)
    require(result.dtype == np.uint8 and result.shape == shape, "Previous-label array changed")
    return result


def audit_frame(receipt_path, expected_source, pipeline_sha256, delta):
    wall,cpu = time.perf_counter(),time.process_time()
    receipt_path = Path(receipt_path)
    receipt_data = receipt_path.read_bytes()
    receipt = json.loads(receipt_data.decode("utf-8-sig"))
    source = receipt["binding"]["source"]
    require(receipt.get("contract") == PRODUCTION_CONTRACT and receipt.get("complete") is True,
            "Uncommitted or wrong-contract output")
    require(receipt["binding"]["pipeline_sha256"] == pipeline_sha256, "Output belongs to another pipeline")
    require(all(source[key] == expected_source[key] == receipt[key] for key in
        ("run","frame_index","source_sha256","elapsed_h")), "Source/time identity changed")
    parent_descriptor = receipt["binding"]["parent_receipt"]
    require(parent_descriptor == expected_source["parent_receipt"], "Parent receipt descriptor changed")
    parent = json.loads(asset_bytes(parent_descriptor).decode("utf-8-sig"))
    require(parent["binding"]["source"] == source and parent["labels_map"] == receipt["before_labels"],
            "Parent source/label binding changed")
    require(parent["delta"] == receipt["delta"] == delta, "Delta changed")
    require(receipt["full_extended_scientific"] == parent["full_extended_scientific"]
        and receipt["baseline_extended_scientific"] == parent["baseline_extended_scientific"], "Scientific images changed")
    require(all(receipt.get(key) is False for key in ("annotations_read","training","training_performed",
        "inference_performed","co2_inference_performed","new_gas_inference_performed","mass_eligible","bottom_mass_allowed")),
        "Scope or mass restriction changed")

    before = labels(receipt["before_labels"])
    shape = before.shape
    after = labels(receipt["labels_map"], shape)
    masks = {name:packed(receipt[key], source, shape) for name,key in MASK_KEYS.items()}
    reason_a = packed(receipt["fill_reason_masks"]["a"], source, shape)
    reason_b = packed(receipt["fill_reason_masks"]["b"], source, shape)
    count = np.zeros(shape, np.uint8)
    for mask in masks.values():
        count += mask
    changed = before != after
    require(np.all(count[changed] == 1), "Final changes do not lie in exactly one step mask")
    require(np.array_equal(before[count == 0], after[count == 0]), "Unflagged label changed")
    require(not ((((before == 2) != (after == 2)) & ~(masks["A"] | masks["D"])).any()), "Gas changed outside A/D")
    old_co2,new_co2 = (before == 1)|(before == 2),(after == 1)|(after == 2)
    require(not ((old_co2 != new_co2) & ~(masks["A"] | masks["B"] | masks["C"])).any(), "CO2 union changed outside A/B/C")
    require(not (reason_a & reason_b).any() and np.array_equal(reason_a | reason_b, masks["C"]), "C reason masks are not its disjoint partition")

    old_rev,new_rev = parent["rig_domain_revision"],receipt["rig_domain_revision"]
    require(old_rev["area_map"] == new_rev["area_map"], "Area descriptor changed")
    require(old_rev["phase_shape_hw"] == new_rev["phase_shape_hw"] == list(shape)
        and old_rev["native_origin_xy"] == new_rev["native_origin_xy"], "Coordinate grid changed")
    old_classes,new_classes = classes(old_rev["classes"],shape),classes(new_rev["classes"],shape)
    removed = new_classes["below_sand_floor"]
    require(not (removed & ~old_classes["rig_domain"]).any(), "Floor removal leaves old domain")
    require(np.array_equal(new_classes["rig_domain"],old_classes["rig_domain"] & ~removed), "Physical domain did not clip precisely at A")
    require(np.array_equal(masks["A"],removed & (before != 255)), "A flags do not match physical clipping")
    require(np.array_equal(after == 255,(before == 255)|removed), "Excluded category changed outside A")
    for key in ("modelled","filled_observed","filled_unobserved"):
        require(np.array_equal(new_classes[key],old_classes[key] & ~removed), f"Class {key} changed beyond A")
    for key in old_classes.keys()-{"rig_domain","modelled","filled_observed","filled_unobserved"}:
        require(np.array_equal(new_classes[key],old_classes[key]), f"Other physical class {key} changed")
    support = (new_classes["modelled"]|new_classes["filled_observed"]|new_classes["filled_unobserved"]) & new_classes["rig_domain"]
    require(not (masks["C"] & ~support).any(), "C fills unsupported data")
    require(source["elapsed_h"] > 0 or not masks["C"].any(), "C fills before injection")
    require(not ((after != 255) & ~new_classes["rig_domain"]).any(), "Final categories leave physical domain")

    area = array(new_rev["area_map"])
    require(area.shape == shape and np.issubdtype(area.dtype,np.floating)
        and np.isfinite(area).all() and not (area < 0).any(), "Area values/grid are invalid")
    def measure(mask):
        return float(area[mask].sum(dtype=np.float64))/100
    step_pixels,step_area,step_previous_gas,step_previous_aqueous = {},{},{},{}
    reconstructed = before.copy()
    targets = dict(A=255,B=0,C=1,D=1)
    for name in "ABCD":
        prior = previous(receipt["previous_labels"][name], source, shape)
        require(np.array_equal(prior[masks[name]], reconstructed[masks[name]])
            and not prior[~masks[name]].any(), f"Previous labels {name} do not reconstruct the recorded step")
        if name == "B":
            require(not (masks[name] & (prior == 2)).any(), "B removes gas")
        if name == "C":
            require(np.all(prior[masks[name]] == 0), "C was not no-CO2 before fill")
        if name == "D":
            require(np.all(prior[masks[name]] == 2), "D was not gas before conversion")
        reconstructed[masks[name]] = targets[name]
        step_pixels[name],step_area[name] = int(masks[name].sum()),measure(masks[name])
        step_previous_gas[name] = measure(masks[name] & (prior == 2))
        step_previous_aqueous[name] = measure(masks[name] & (prior == 1))
    require(np.array_equal(reconstructed,after), "Recorded steps do not reproduce saved labels exactly")
    params = new_rev["parameters"]
    require(abs(params["domain_area_cm2"]-measure(new_classes["rig_domain"])) <= 1e-9, "Domain area accounting differs")
    require(abs(params["excluded_below_sand_floor_area_cm2"]-measure(removed)) <= 1e-9, "Floor area accounting differs")
    for key in ("modelled","filled_observed","filled_unobserved"):
        require(abs(params["area_cm2_by_class"][key]-measure(new_classes[key])) <= 1e-9, "Class area accounting differs")
    for key,actual in (("step_pixels",step_pixels),("step_area_cm2",step_area),
        ("step_previous_gas_cm2",step_previous_gas),("step_previous_aqueous_cm2",step_previous_aqueous)):
        require(set(receipt["qa"][key]) == set(actual)
            and all(abs(receipt["qa"][key][name]-value) <= 1e-9 for name,value in actual.items()), "Recorded count/area differs: "+key)
    require(abs(receipt["qa"]["fill_reason_a_cm2"]-measure(reason_a)) <= 1e-9
        and abs(receipt["qa"]["fill_reason_b_cm2"]-measure(reason_b)) <= 1e-9, "Recorded C reason areas differ")
    return dict(contract=CONTRACT,all_pass=True,run=source["run"],frame_index=source["frame_index"],
        source_sha256=source["source_sha256"],elapsed_h=source["elapsed_h"],receipt_path=str(receipt_path),
        receipt_sha256=hashlib.sha256(receipt_data).hexdigest(),pipeline_sha256=pipeline_sha256,
        changed_pixels=int(changed.sum()),step_pixels=step_pixels,step_area_cm2=step_area,
        unchanged_stage_restoration_pixels=int(((count > 0) & ~changed).sum()),
        overlapping_unchanged_stage_pixels=int(((count > 1) & ~changed).sum()),
        overlapping_final_changed_pixels=int(((count > 1) & changed).sum()),
        step_previous_gas_cm2=step_previous_gas,step_previous_aqueous_cm2=step_previous_aqueous,
        fill_reason_a_cm2=measure(reason_a),fill_reason_b_cm2=measure(reason_b),
        domain_removed_cm2=measure(removed),area_asset_sha256=new_rev["area_map"]["sha256"],
        previous_labels_exact=True,unflagged_labels_bit_identical=True,gas_only_a_d=True,co2_union_only_a_b_c=True,
        physical_classes_clip_only_a=True,area_values_unchanged=True,c_reasons_partition_exact=True,
        scoring=False,inference=False,distance_transforms=False,backend="cpu-read-only",
        timing=dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu))


def audit_candidate(candidate,output,*,workers=4,require_complete=False):
    candidate,output = Path(candidate).resolve(),Path(output).resolve()
    require(not output.is_relative_to(candidate) and not candidate.is_relative_to(output), "Use a separate QA output folder")
    require(1 <= workers <= 4, "Use at most four audit readers")
    plan_path = candidate/"p1_final2_plan.json"
    plan = read(plan_path)
    require(plan["contract"] == PRODUCTION_CONTRACT and plan["binding"]["inputs"]["spec"]["sha256"] == SPEC_SHA256,
        "Audit candidate is not the locked final round")
    require(plan["job_sha256"] == canonical_sha256(plan["binding"])
        and plan["pipeline_sha256"] == canonical_sha256(dict(binding=plan["binding"],model_identity=plan["model_identity"])),
        "Immutable plan binding changed")
    parent_seal = json.loads(asset_bytes(plan["binding"]["inputs"]["parent_seal"]).decode("utf-8-sig"))
    require(parent_seal["complete"] is True and (parent_seal["run_count"],parent_seal["frame_count"]) == (43,864)
        and parent_seal["heads"] == plan["binding"]["heads"] and parent_seal["delta"] == plan["delta"],
        "Parent seal or frozen heads/delta changed")
    sources = plan["source_inventory"]
    require(len(sources) == 864 and len({(s["run"],s["frame_index"]) for s in sources}) == 864
        and len({s["run"] for s in sources}) == 43, "Expected inventory is not43/864")
    parent_sources = {(row["run"],row["frame_index"]):row["receipt"] for row in parent_seal["frames"]}
    require(len(parent_sources) == 864 and all(parent_sources.get((s["run"],s["frame_index"])) == s["parent_receipt"] for s in sources),
        "Expected source inventory differs from the exact parent seal")
    binding = dict(candidate=str(candidate),pipeline_sha256=plan["pipeline_sha256"],
        source_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),spec_sha256=SPEC_SHA256)
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"audit_binding.json",dict(contract=CONTRACT,binding=binding,auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    pending,completed,missing = [],[],[]
    for source in sources:
        path = candidate/"rig_camera_frames"/source["run"]/f"frame_{source['frame_index']:03d}.json"
        saved_path = output/"frames"/source["run"]/f"frame_{source['frame_index']:03d}.json"
        if not path.exists():
            missing.append(dict(run=source["run"],frame_index=source["frame_index"]))
        elif saved_path.exists():
            saved = read(saved_path)
            require(saved["pipeline_sha256"] == binding["pipeline_sha256"] and saved["all_pass"] is True
                and saved["receipt_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), "Committed audit receipt changed")
            completed.append(saved)
        else:
            pending.append((path,saved_path,source))
    def work(item):
        path,saved_path,source = item
        result = audit_frame(path,source,plan["pipeline_sha256"],plan["delta"])
        atomic_json(saved_path,result)
        return result
    wall,cpu=time.perf_counter(),time.process_time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(work,pending):
            completed.append(result)
    completed.sort(key=lambda row:(row["run"],row["frame_index"]))
    complete = len(completed) == 864 and not missing
    seal_path = candidate/"rig_camera_candidate_receipt.json"
    if require_complete:
        require(complete and seal_path.exists(), "Need sealed43/864 for the final identity report")
        seal = read(seal_path)
        require(seal["complete"] is True and seal["pipeline_sha256"] == binding["pipeline_sha256"]
            and (seal["run_count"],seal["frame_count"]) == (43,864), "Complete seal differs")
        descriptors = {(row["run"],row["frame_index"]):row["receipt"] for row in seal["frames"]}
        require(len(descriptors) == 864 and all(descriptors[(r["run"],r["frame_index"])]["sha256"] == r["receipt_sha256"] for r in completed),
            "Independent audit differs from sealed receipts")
    summary = dict(contract=CONTRACT,complete=complete,all_pass=complete,created_utc=utc_now(),binding=binding,
        audited_frames=len(completed),expected_frames=864,audited_runs=len({r['run'] for r in completed}),expected_runs=43,
        newly_audited_frames=len(pending),missing_frames=missing,workers=workers,
        step_pixels={name:sum(row["step_pixels"][name] for row in completed) for name in "ABCD"},
        changed_pixels=sum(row["changed_pixels"] for row in completed),
        audit_wall_seconds=time.perf_counter()-wall,audit_cpu_seconds=time.process_time()-cpu,
        all_audit_wall_seconds=sum(row["timing"]["wall_seconds"] for row in completed),
        numerical_modules_executed=False,inference=False,scoring=False)
    atomic_json(output/"snapshots"/f"snapshot_{time.time_ns()}.json",summary)
    if require_complete:
        atomic_json(output/"final_identity_summary.json",summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--require-complete",action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_candidate(args.candidate,args.output,workers=args.workers,require_complete=args.require_complete)),flush=True)


if __name__ == "__main__":
    main()
