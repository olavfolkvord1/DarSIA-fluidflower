"""Independent saved-file identities for P1 final 3, without inference.

Only the new audit directory is written by default. ``--candidate-summary``
explicitly publishes the final technical audit into the new candidate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .qa_ffac_p1_final2_saved import asset_bytes, array, labels, read, require
from .refit_ffac_g3_downstream import atomic_json, canonical_sha256, utc_now

CONTRACT = "ffac-p1-final3-independent-saved-identities-v1"
PRODUCTION_CONTRACT = "ffac-g3-t4c-p1-final3-candidate-v1"
PARENT_CONTRACT = "ffac-g3-t4c-p1-final2-candidate-v1"
SPEC_SHA256 = "f70dc1bcc6de5b3118a1983547bcca0aa41f057ab829b217c2068ea2edce0a77"
NOMINAL_AREA_MM2 = .027053
MASK_KEYS = dict(A="below_sand_floor_masks", B="floor_connectivity_masks",
                 C="floor_contact_fill_v2", D="gas_envelope_masks")


def verify_frozen_heads(heads, candidate):
    """Verify original bound heads and their copied candidate bytes once."""
    count = 0
    for value in heads.values():
        if "path" not in value:
            count += verify_frozen_heads(value, candidate)
            continue
        asset_bytes(value)
        source = Path(value["path"])
        parts = source.parts
        anchors = [n for n, part in enumerate(parts) if part in {"models", "regional_models"}]
        relative = Path(*parts[anchors[-1]:]) if anchors else Path(source.name)
        path = Path(candidate)/relative
        require(hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"],
                "Copied frozen head bytes changed: " + str(path))
        count += 1
    return count


def packed(asset, source, shape, *, contract):
    require(asset.get("storage") == "packbits-npy"
        and asset.get("shape_hw") == list(shape)
        and asset.get("source_sha256") == source["source_sha256"]
        and asset.get("frame_index") == source["frame_index"]
        and asset.get("contract") == contract, "Packed mask lost source/grid/contract binding")
    data = array(asset)
    size = int(np.prod(shape))
    require(data.dtype == np.uint8 and data.ndim == 1 and len(data) == (size + 7) // 8,
            "Invalid packbits storage")
    bits = np.unpackbits(data)
    require(not bits[size:].any(), "Packed mask has nonzero padding")
    return bits[:size].reshape(shape).astype(bool)


def previous(asset, source, shape):
    require(asset.get("storage") == "npz-array" and asset.get("array_key") == "previous_labels"
        and asset.get("shape_hw") == list(shape)
        and asset.get("source_sha256") == source["source_sha256"]
        and asset.get("frame_index") == source["frame_index"]
        and asset.get("contract") == PRODUCTION_CONTRACT,
        "Previous labels lost source/grid/contract binding")
    result = array(asset)
    require(result.dtype == np.uint8 and result.shape == shape, "Invalid previous-label array")
    return result


def check_label_identities(before, after, reason_b, grandparent, step_masks):
    """Array-level checks, independently of producer QA or numerical rule code."""
    require(before.dtype == after.dtype == grandparent.dtype == np.uint8
        and before.shape == after.shape == grandparent.shape == reason_b.shape,
        "Category grids differ")
    changed = before != after
    require(np.array_equal(changed, reason_b), "Changed pixels are not exactly reason (b)")
    require(np.all(before[changed] == 1) and np.all(after[changed] == 0),
            "Reason (b) was not exclusively aqueous 1 -> no-CO2 0")
    require(np.array_equal(before == 2, after == 2), "Gas changed")
    require(np.array_equal(before == 255, after == 255), "Excluded-domain category changed")
    require(np.array_equal(before[~reason_b], after[~reason_b]), "Another category changed")
    count = np.zeros(before.shape, np.uint8)
    for mask in step_masks.values():
        require(mask.dtype == bool and mask.shape == before.shape, "Step-mask grid differs")
        count += mask
    changed_from_grandparent = grandparent != after
    require(np.all(count[changed_from_grandparent] == 1),
            "P1-final changed pixels are not in exactly one A/B/C(a)/D mask")
    require(np.array_equal(grandparent[count == 0], after[count == 0]),
            "An unflagged P1-final category changed")
    return dict(reverted_pixels=int(changed.sum()),
        net_changed_from_p1_final_pixels=int(changed_from_grandparent.sum()),
        unchanged_stage_restoration_pixels=int(((count > 0) & ~changed_from_grandparent).sum()),
        overlapping_unchanged_stage_pixels=int(((count > 1) & ~changed_from_grandparent).sum()),
        overlapping_final_changed_pixels=int(((count > 1) & changed_from_grandparent).sum()),
        gas_bit_identical=True, domain_category_bit_identical=True,
        all_other_pixels_bit_identical=True, changed_pixels_exactly_reason_b=True,
        all_changes_aqueous_to_zero=True, net_changed_pixels_exactly_one_step=True)


def audit_frame(receipt_path, expected_source, pipeline_sha256, delta, candidate=None):
    wall, cpu = time.perf_counter(), time.process_time()
    receipt_path = Path(receipt_path)
    receipt_data = receipt_path.read_bytes()
    receipt = json.loads(receipt_data.decode("utf-8-sig"))
    source = receipt["binding"]["source"]
    require(receipt.get("contract") == PRODUCTION_CONTRACT and receipt.get("complete") is True,
            "Uncommitted or wrong-contract final3 output")
    require(receipt["binding"]["pipeline_sha256"] == pipeline_sha256, "Wrong pipeline output")
    keys = ("run", "frame_index", "source_sha256", "elapsed_h")
    require(all(source[key] == expected_source[key] == receipt[key] for key in keys),
            "Source/time identity changed")
    parent_asset = receipt["binding"]["parent_receipt"]
    require(parent_asset == expected_source["parent_receipt"], "Parent receipt binding changed")
    parent = json.loads(asset_bytes(parent_asset).decode("utf-8-sig"))
    require(parent.get("contract") == PARENT_CONTRACT and parent.get("complete") is True,
            "Parent is not committed final2")
    require(parent["binding"]["source"] == source
        and all(parent[key] == receipt[key] for key in keys + ("light_epoch_id",)),
        "Full source/time/epoch binding changed")
    require(parent["labels_map"] == receipt["before_labels"], "Parent labels are not exact final2")
    require(parent["delta"] == receipt["delta"] == delta, "Delta changed")
    for key in ("rig_domain_revision", "full_extended_scientific", "baseline_extended_scientific"):
        require(receipt[key] == parent[key], "Unchanged asset descriptor differs: " + key)
    require(all(receipt.get(key) is False for key in ("annotations_read", "training",
        "training_performed", "inference_performed", "co2_inference_performed",
        "new_gas_inference_performed", "mass_eligible", "bottom_mass_allowed")),
        "Scope or bottom mass restriction changed")
    if candidate is not None:
        expected = Path(candidate)/"production_labels"/source["run"]/f"frame_{source['frame_index']:03d}.png"
        require(Path(receipt["labels_map"]["path"]).resolve() == expected.resolve(),
                "Regression export is not production_labels/<run>/frame_NNN.png")
    before = labels(receipt["before_labels"])
    shape = before.shape
    after = labels(receipt["labels_map"], shape)
    grand_asset = parent["binding"]["parent_receipt"]
    grand = json.loads(asset_bytes(grand_asset).decode("utf-8-sig"))
    require(grand["labels_map"] == parent["before_labels"]
        and grand["binding"]["source"] == source
        and all(grand[key] == source[key] for key in keys), "P1-final source binding changed")
    original = labels(parent["before_labels"], shape)
    historical = parent["fill_reason_masks"]
    require(receipt["fill_reason_masks"] == historical, "Historical fill reasons changed")
    a = packed(historical["a"], source, shape, contract=PARENT_CONTRACT)
    b = packed(historical["b"], source, shape, contract=PARENT_CONTRACT)
    require(not (a & b).any(), "Historical reasons a/b overlap")
    masks = {}
    for step in "ABD":
        key = MASK_KEYS[step]
        require(receipt[key] == parent[key], "Unchanged step descriptor differs: " + step)
        masks[step] = packed(receipt[key], source, shape, contract=PARENT_CONTRACT)
        require(receipt["previous_labels"][step] == parent["previous_labels"][step],
                "Unchanged previous-label descriptor differs: " + step)
    masks["C"] = packed(receipt[MASK_KEYS["C"]], source, shape, contract=PRODUCTION_CONTRACT)
    require(np.array_equal(masks["C"], a), "New C is not exactly historical reason (a)")
    reverted = packed(receipt["contact_fill_b_reverted_masks"], source, shape,
                      contract=PRODUCTION_CONTRACT)
    require(np.array_equal(reverted, b), "Reverted flag differs from historical reason (b)")
    prior = previous(receipt["reverted_previous_labels"], source, shape)
    require(np.all(prior[b] == 1) and not prior[~b].any(),
            "Reverted flag previous labels do not record exactly category 1")
    del prior
    prior_c = previous(receipt["previous_labels"]["C"], source, shape)
    require(not prior_c.any(), "Retained C previous labels were not exactly zero")
    del prior_c
    checked = check_label_identities(before, after, b, original, masks)
    area = array(receipt["rig_domain_revision"]["area_map"])
    require(area.shape == shape and np.issubdtype(area.dtype, np.floating)
        and np.isfinite(area).all() and not (area < 0).any(), "Invalid unchanged measured area map")
    nominal = checked["reverted_pixels"] * NOMINAL_AREA_MM2 / 100.
    measured = float(np.sum(area, where=b, dtype=np.float64)) / 100.
    a_pixels = int(a.sum())
    return dict(contract=CONTRACT, all_pass=True, run=source["run"],
        frame_index=source["frame_index"], elapsed_h=source["elapsed_h"],
        source_sha256=source["source_sha256"], light_epoch_id=receipt["light_epoch_id"],
        receipt_path=str(receipt_path), receipt_sha256=hashlib.sha256(receipt_data).hexdigest(),
        parent_receipt=parent_asset, p1_final_receipt=grand_asset, pipeline_sha256=pipeline_sha256,
        area_asset_sha256=receipt["rig_domain_revision"]["area_map"]["sha256"],
        nominal_pixel_area_mm2=NOMINAL_AREA_MM2, reverted_nominal_cm2=nominal,
        reverted_measured_cm2=measured, retained_reason_a_pixels=a_pixels,
        retained_reason_a_nominal_cm2=a_pixels*NOMINAL_AREA_MM2/100.,
        retained_reason_a_measured_cm2=float(np.sum(area, where=a, dtype=np.float64))/100.,
        step_pixels={step:int(mask.sum()) for step,mask in masks.items()},
        area_domain_scientific_descriptors_bit_identical=True,
        source_time_epoch_binding_bit_identical=True, regression_layout_verified=True,
        scoring=False, inference=False, training=False, annotations_read=False,
        backend="cpu-read-only", **checked,
        timing=dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu))


def audit_candidate(candidate, output, *, workers=8, require_complete=False, candidate_summary=False):
    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    require(not output.is_relative_to(candidate) and not candidate.is_relative_to(output),
            "Use a separate QA output folder")
    require(1 <= workers <= 24, "Use 1-24 audit readers")
    plan_path = candidate/"p1_final3_plan.json"
    plan = read(plan_path)
    require(plan["contract"] == PRODUCTION_CONTRACT
        and plan["binding"]["inputs"]["spec"]["sha256"] == SPEC_SHA256, "Locked spec differs")
    require(plan["job_sha256"] == canonical_sha256(plan["binding"])
        and plan["pipeline_sha256"] == canonical_sha256(dict(binding=plan["binding"], model_identity=plan["model_identity"])),
        "Immutable plan binding changed")
    parent_seal = json.loads(asset_bytes(plan["binding"]["inputs"]["parent_seal"]).decode("utf-8-sig"))
    require(parent_seal["complete"] is True
        and (parent_seal["run_count"], parent_seal["frame_count"]) == (43, 864)
        and parent_seal["heads"] == plan["binding"]["heads"]
        and parent_seal["delta"] == plan["delta"], "Parent heads/delta or completion changed")
    frozen_head_files_verified = verify_frozen_heads(parent_seal["heads"], candidate)
    sources = plan["source_inventory"]
    require(len(sources) == 864 and len({(s["run"], s["frame_index"]) for s in sources}) == 864
        and len({s["run"] for s in sources}) == 43, "Expected inventory is not 43/864")
    parent_rows = {(row["run"], row["frame_index"]):row["receipt"] for row in parent_seal["frames"]}
    require(len(parent_rows) == 864
        and all(parent_rows.get((s["run"], s["frame_index"])) == s["parent_receipt"] for s in sources),
        "Expected source inventory differs from exact parent seal")
    binding = dict(candidate=str(candidate), pipeline_sha256=plan["pipeline_sha256"],
        source_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), spec_sha256=SPEC_SHA256)
    auditor_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    audit_binding = dict(contract=CONTRACT, binding=binding, auditor_sha256=auditor_sha)
    bind_path = output/"audit_binding.json"
    if bind_path.exists():
        require(read(bind_path) == audit_binding, "Audit code or immutable binding changed; use new output")
    else:
        atomic_json(bind_path, audit_binding)
    pending, completed, missing = [], [], []
    for source in sources:
        path = candidate/"rig_camera_frames"/source["run"]/f"frame_{source['frame_index']:03d}.json"
        saved_path = output/"frames"/source["run"]/f"frame_{source['frame_index']:03d}.json"
        if not path.exists():
            missing.append(dict(run=source["run"], frame_index=source["frame_index"]))
        elif saved_path.exists():
            result = read(saved_path)
            require(result["pipeline_sha256"] == binding["pipeline_sha256"] and result["all_pass"] is True
                and result["receipt_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(),
                "Committed audit receipt changed")
            completed.append(result)
        else:
            pending.append((path, saved_path, source))
    def work(item):
        path, saved_path, source = item
        result = audit_frame(path, source, plan["pipeline_sha256"], plan["delta"], candidate)
        atomic_json(saved_path, result)
        return result
    wall, cpu = time.perf_counter(), time.process_time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(work, item) for item in pending]
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed) % 32 == 0:
                print(json.dumps(dict(audited_frames=len(completed), expected_frames=864,
                    wall_seconds=time.perf_counter()-wall)), flush=True)
    completed.sort(key=lambda row:(row["run"], row["frame_index"]))
    complete = len(completed) == 864 and not missing
    seal_path = candidate/"rig_camera_candidate_receipt.json"
    if require_complete or candidate_summary:
        require(complete and seal_path.exists(), "Need sealed 43/864 for final identity report")
        seal = read(seal_path)
        require(seal["complete"] is True and seal["pipeline_sha256"] == binding["pipeline_sha256"]
            and (seal["run_count"], seal["frame_count"]) == (43, 864)
            and seal["heads"] == parent_seal["heads"] and seal["delta"] == parent_seal["delta"],
            "Complete seal or frozen heads/delta differs")
        descriptors = {(row["run"], row["frame_index"]):row["receipt"] for row in seal["frames"]}
        require(len(descriptors) == 864
            and all(descriptors[(r["run"], r["frame_index"])]["sha256"] == r["receipt_sha256"] for r in completed),
            "Independent audit differs from sealed receipts")
    summary = dict(contract=CONTRACT, complete=complete, all_pass=complete, created_utc=utc_now(),
        binding=binding, auditor_sha256=auditor_sha, audited_frames=len(completed),
        expected_frames=864, audited_runs=len({r["run"] for r in completed}), expected_runs=43,
        frozen_head_files_verified=frozen_head_files_verified, frozen_head_bytes_bit_identical=True,
        newly_audited_frames=len(pending), missing_frames=missing, workers=workers,
        reverted_pixels=sum(r["reverted_pixels"] for r in completed),
        reverted_nominal_cm2=sum(r["reverted_nominal_cm2"] for r in completed),
        reverted_measured_cm2=sum(r["reverted_measured_cm2"] for r in completed),
        retained_reason_a_pixels=sum(r["retained_reason_a_pixels"] for r in completed),
        retained_reason_a_nominal_cm2=sum(r["retained_reason_a_nominal_cm2"] for r in completed),
        retained_reason_a_measured_cm2=sum(r["retained_reason_a_measured_cm2"] for r in completed),
        gas_bit_identical_frames=sum(r["gas_bit_identical"] for r in completed),
        domain_category_bit_identical_frames=sum(r["domain_category_bit_identical"] for r in completed),
        net_changed_pixels_exactly_one_step_frames=sum(r["net_changed_pixels_exactly_one_step"] for r in completed),
        overlapping_final_changed_pixels=sum(r["overlapping_final_changed_pixels"] for r in completed),
        independent_output=str(output), audit_wall_seconds=time.perf_counter()-wall,
        audit_cpu_seconds=time.process_time()-cpu,
        all_frame_audit_wall_seconds=sum(r["timing"]["wall_seconds"] for r in completed),
        backend="cpu-read-only", numerical_modules_executed=False, inference=False, training=False,
        scoring=False, annotations_read=False, bottom_mass_allowed=False, mass_eligible=False)
    atomic_json(output/"snapshots"/f"snapshot_{time.time_ns()}.json", summary)
    if require_complete or candidate_summary:
        atomic_json(output/"final_identity_summary.json", summary)
    if candidate_summary:
        path = candidate/"production_qa"/"saved_qa.json"
        if path.exists():
            require(read(path) == summary, "Published saved QA already exists; do not overwrite")
        else:
            atomic_json(path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--candidate-summary", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_candidate(args.candidate, args.output, workers=args.workers,
        require_complete=args.require_complete, candidate_summary=args.candidate_summary)), flush=True)


if __name__ == "__main__":
    main()
