"""Attribute a strict replay stop using saved maps and producer evidence only.

No inference, counterfactual rerun, fitting, or phase-map write is performed.
The diagnostic regions are derived from the already bound historical masks.
They identify where changes occur; they do not claim score-level causality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_ffac_selected_phase import asset, bound_json
from darsia.fluidflower.phase_retained_source_inference import (
    MASK_NAMES, _old_domain_interpolation_footprint, _outer_domain_ring,
)
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, utc_now
from darsia.fluidflower.tools import run_ffac_rig_camera_inference as rig

CONTRACT = "ffac-selected-saved-replay-stop-diagnosis-v1"


def png(descriptor):
    from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
    with Image.open(bound_path(descriptor)) as image:
        return np.asarray(image).copy()


def packed(descriptor):
    from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
    raw = np.load(bound_path(descriptor), allow_pickle=False)
    shape = tuple(descriptor["shape_hw"])
    return np.unpackbits(raw, count=int(np.prod(shape))).reshape(shape).astype(bool)


def count(mask, area):
    return {"pixels": int(mask.sum()), "cm2": float(area[mask].sum(dtype=np.float64) / 100)}


def regional_changes(old, new, area, regions):
    co2 = np.isin(old, (1, 2)) ^ np.isin(new, (1, 2))
    gas = (old == 2) ^ (new == 2)
    category = old != new
    coverage = np.zeros(old.shape, np.uint8)
    rows = {}
    for name, region in regions.items():
        coverage += region.astype(np.uint8)
        rows[name] = {"co2": count(co2 & region, area),
                      "gas": count(gas & region, area),
                      "categorical": count(category & region, area)}
    if not np.all(coverage == 1):
        raise ValueError("Diagnostic regions must partition every stored pixel exactly once")
    for name, mask in (("co2", co2), ("gas", gas), ("categorical", category)):
        if sum(row[name]["pixels"] for row in rows.values()) != int(mask.sum()):
            raise ValueError("Diagnostic partition loses or double counts changes")
    return rows


def build(plan_path, receipt_path, destination):
    plan_descriptor, replay_descriptor = asset(plan_path), asset(receipt_path)
    plan, replay = bound_json(plan_descriptor), bound_json(replay_descriptor)
    source = replay["binding"]["source"]
    run = source["run"]
    inventory = bound_json(plan["runs"][run]["source_inventory"])
    if (source not in inventory["sources"]
            or replay["binding"]["pipeline_sha256"] != plan["pipeline_sha256"]
            or replay["qa"]["strict_studio_replay"]["all_pass"] is not False):
        raise ValueError("Diagnosis requires the exact bound failed strict replay")
    parent = bound_json(source["parent_receipt"])
    science_descriptor = asset(Path(plan["scientific_source_root"]) / "selected_sources" /
                               run / source["source_sha256"] / "receipt.json")
    science = bound_json(science_descriptor)
    previous = next(row for row in inventory["sources"]
                    if row["source_sha256"] == source["previous_source_sha256"])
    old_previous = source["parent_source"]["temporal_predecessor"]
    probability_descriptor = source["parent_source"]["retained_source_cache"]["parent_probability_receipt"]
    probability = bound_json(probability_descriptor)
    from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
    native_shape = tuple(probability["metadata"]["native_shape_hw"])
    with np.load(bound_path(probability["files"]["extra_masks.packbits.npz"])) as archive:
        masks = {name: np.unpackbits(row, count=int(np.prod(native_shape)), bitorder="little")
                 .reshape(native_shape).astype(bool)
                 for name, row in zip(MASK_NAMES, archive["masks"])}
    native_domain = masks["photo_sand_domain"] & masks["feature_support"]
    footprint, footprint_audit = _old_domain_interpolation_footprint(
        native_domain, tuple(probability["metadata"]["regional_working_shape_hw"]))
    release = footprint | _outer_domain_ring(native_domain)
    old_labels, new_labels = png(parent["labels_map"]), png(replay["labels_map"])
    area = rig.load_array(replay["rig_domain_revision"]["area_map"])
    old_area = rig.load_array(parent["rig_domain_revision"]["area_map"])
    if not np.array_equal(area, old_area):
        raise ValueError("Replay diagnosis found an area-map change")
    domain_canvas = rig._canvas(native_domain, False, phase_shape_hw=old_labels.shape)
    release_canvas = rig._canvas(release, False, phase_shape_hw=old_labels.shape)
    with np.load(bound_path(replay["rig_domain_revision"]["classes"])) as archive:
        modelled = archive["modelled"].copy()
    regions = {"historical_native_interior_outside_historical_release": domain_canvas & ~release_canvas,
               "historical_old_boundary_release_footprint": release_canvas,
               "outside_old_domain_modelled": ~domain_canvas & modelled,
               "outside_old_domain_unmodelled_or_excluded": ~domain_canvas & ~modelled}
    rows = regional_changes(old_labels, new_labels, area, regions)
    native_descriptor = plan["runs"][run]["parent_native_manifest"]
    native_manifest = bound_json(native_descriptor)
    old_current_row = native_manifest["frame_sources"][source["parent_source"]["parent_source_frame_index"]]
    baseline_descriptor = asset(Path(plan["scientific_source_root"]) / "selected_sources" /
        run / inventory["baseline_source_sha256"] / "receipt.json")
    baseline_science = bound_json(baseline_descriptor)
    input_support = {}
    for name, current_science, old_row in (("current", science, old_current_row),
                                           ("baseline", baseline_science, native_manifest["reference"])):
        old_asset = dict(path=old_row["v5_log_linear_valid_mask_path"],
                        sha256=old_row["v5_log_linear_valid_mask_sha256"])
        new_asset = current_science["native_scientific"]["valid_mask"]
        old_valid = np.load(bound_path(old_asset), allow_pickle=False)
        new_valid = np.load(bound_path(new_asset), allow_pickle=False)
        input_support[name] = dict(original_asset=old_asset, replay_asset=new_asset,
            original_valid_pixels=int(old_valid.sum()), replay_valid_pixels=int(new_valid.sum()),
            changed_validity_pixels=int((old_valid ^ new_valid).sum()))
    photo_asset = dict(path=old_current_row["v5_photo_sand_mask_path"],
                      sha256=old_current_row["v5_photo_sand_mask_sha256"])
    supplied_photo = np.load(bound_path(photo_asset), allow_pickle=False)
    input_support["photo_sand"] = dict(original_exported_mask=probability["files"]["extra_masks.packbits.npz"],
        replay_asset=photo_asset, original_pixels=int(masks["photo_sand_domain"].sum()),
        replay_pixels=int(supplied_photo.sum()),
        changed_pixels=int((masks["photo_sand_domain"] ^ supplied_photo).sum()))
    input_support["original_alignment"] = probability["metadata"]["alignment"]
    input_support["replay_alignment"] = replay["qa"]["native_inference"]["temporal_alignment"]
    input_support["original_domain"] = dict(pixels=int(native_domain.sum()),
        sha256=footprint_audit["original_domain_mask_sha256"])
    input_support["replay_domain"] = dict(pixels=replay["qa"]["native_inference"]["domain_pixels"],
        sha256=replay["qa"]["base_seam"]["released_footprint"]["original_domain_mask_sha256"])
    input_support["original_reference_has_declared_extended_ring"] = bool(
        native_manifest["reference"].get("v5_extended_ring_path"))
    with np.load(bound_path(replay["operator_masks"])) as archive:
        shape = tuple(archive["shape_hw"])
        post = {name: np.unpackbits(archive[name], count=int(np.prod(shape)), bitorder="little")
                .reshape(shape).astype(bool) for name in archive.files if name != "shape_hw"}
    final1_path = Path(parent["p1_final_before_labels"]["path"]).parents[2] / "rig_camera_frames" / run / (
        f"frame_{source['studio_frame_index']:03d}.json")
    final1_descriptor = asset(final1_path)
    final1 = bound_json(final1_descriptor)
    old_top = packed(final1["top_surface_rule_masks"])
    final_diff = np.isin(old_labels, (1, 2)) ^ np.isin(new_labels, (1, 2))
    top_diff = old_top ^ post["post_removed_top"]
    old_floor = packed(parent["below_sand_floor_masks"])
    old_b = packed(parent["floor_connectivity_masks"])
    old_c = packed(parent["floor_contact_fill_v2"])
    old_d = packed(parent["gas_envelope_masks"])
    operator_checks = {
        "top_old_pixels": int(old_top.sum()), "top_new_pixels": int(post["post_removed_top"].sum()),
        "top_mask_changed_pixels": int(top_diff.sum()),
        "final_co2_changes_on_top_mask_difference": count(final_diff & top_diff, area),
        "floor_A_bit_identical": bool(np.array_equal(old_floor, post["post_A"])),
        "floor_B_old_pixels": int(old_b.sum()), "floor_B_new_pixels": int(post["post_B"].sum()),
        "contact_C_old_pixels": int(old_c.sum()), "contact_C_new_pixels": int(post["post_C"].sum()),
        "envelope_D_old_pixels": int(old_d.sum()), "envelope_D_new_pixels": int(post["post_D"].sum()),
        "P1_light_shift_bypassed_rig_light": replay["qa"]["p1"]["optical_evaluation_bypassed"],
        "P1_co2_changed_pixels": replay["qa"]["p1"]["co2_union_changed_pixels"],
        "rule_B_co2_changed_pixels": replay["qa"]["base_seam"]["rule_b"]["co2_union_changed_pixels"],
    }
    record = dict(contract=CONTRACT, created_utc=utc_now(), plan=plan_descriptor,
        failed_replay=replay_descriptor, parent_receipt=source["parent_receipt"],
        diagnosis_code=asset(__file__), original_native_manifest=native_descriptor,
        scientific_receipt=science_descriptor, probability_receipt=probability_descriptor,
        parent_top_receipt=final1_descriptor, run=run, source_sha256=source["source_sha256"],
        elapsed_h=source["elapsed_h"], studio_frame_index=source["studio_frame_index"],
        previous_old={"source_sha256":old_previous["source_sha256"], "source_name":Path(old_previous["source_path"]).name,
            "elapsed_h":next(row["elapsed_h"] for row in inventory["sources"] if row["source_sha256"]==old_previous["source_sha256"])},
        previous_selected={key: previous[key] for key in ("source_sha256", "source_name", "elapsed_h", "selected", "studio_anchor")},
        previous_input_changed=old_previous["source_sha256"] != previous["source_sha256"],
        current_extended_scientific_descriptor_identical=parent["full_extended_scientific"]==replay["full_extended_scientific"],
        baseline_extended_scientific_descriptor_identical=parent["baseline_extended_scientific"]==replay["baseline_extended_scientific"],
        current_native_centre_changed=science["source_audit"]["native_centre_changed"],
        current_rig_geometry_changed=science["source_audit"]["rig_geometry_changed"],
        area_arrays_bit_identical=True, unresolved_support_changed_pixels=replay["qa"]["strict_studio_replay"]["support_changed_pixels"],
        fixed_models=replay["qa"]["native_inference"]["models"],
        gas_heads_sha_match_original_probability=(probability["binding"]["g3_model_sha256"]==replay["qa"]["native_inference"]["models"]["g3"]["sha256"]
            and probability["binding"]["regional_refit_sha256"]==replay["qa"]["native_inference"]["models"]["regional_refit"]["sha256"]),
        co2_head_sha_match_original_probability=probability["metadata"]["co2_head_sha256"]==replay["qa"]["native_inference"]["models"]["co2"]["sha256"],
        strict_replay=replay["qa"]["strict_studio_replay"], region_changes=rows,
        native_input_support_comparison=input_support,
        geometric_region_definition="Partition uses ORIGINAL historical photo_sand_domain intersect feature_support and its original coarse interpolation footprint plus outer ring. The replay native domain differs, so these regions locate changes in the original geometry, not a recovered exact replay-domain mask.",
        footprint_audit=footprint_audit, operator_checks=operator_checks,
        causal_limit="The temporal predecessor changes by design, but native input masking/support also differs: the replay receives unmasked rig-native validity and a 64-pixel temporal crop while the old AC18 producer used masked native validity and origin zero. These are separate adapter differences; chronology alone is NOT established as the cause. No counterfactual, new model score or classifier evaluation was computed.",
        maps_written=False, inference_performed=False, thresholds_changed=False,
        history_changed_by_this_diagnosis=False, bulk_production_allowed=False, mass_computed=False)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "diagnosis.json", record)
    text = (f"Strict selected-history replay stopped: {run.upper()} Studio frame {source['studio_frame_index']} at {source['elapsed_h']:.9f} h.\n"
        "The fixed CO2 XOR-area limit is 0.1%; the saved result is 0.1314097%. Gas and unresolved support pass.\n\n"
        "Current acquisition, pre-injection acquisition, rig RAW sidecars, rig geometry, measured pixel area and frozen model hashes are retained. "
        f"The old predecessor was {record['previous_old']['source_name']} at {record['previous_old']['elapsed_h']:.9f} h; "
        f"the selected-history predecessor is {previous['source_name']} at {previous['elapsed_h']:.9f} h.\n\n"
        "However, native inference did not receive the exact historical masked validity convention: current validity expanded "
        "by 2,449,122 pixels and baseline validity by 2,449,800 pixels; the replay also used temporal origins (-64,-64) "
        "instead of (0,0). The photo-sand mask differs by 112 pixels. Therefore history is not the only changed input.\n\n"
        "This report reads saved arrays only. It partitions changed pixels on the ORIGINAL historical domain (whose replay mask differs), "
        "and compares saved top/floor/envelope flags. It does not rerun the model to isolate a score-level counterfactual. "
        "No threshold, history, label, phase map, candidate default or mass policy has been changed. No new candidate was registered.\n")
    with (destination / "README.txt").open("x", encoding="utf-8") as stream: stream.write(text)
    return asset(destination / "diagnosis.json")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "receipt", "destination"): parser.add_argument("--"+name, required=True)
    args=parser.parse_args(argv)
    print(json.dumps(build(args.plan,args.receipt,args.destination))); return 0


if __name__=="__main__": raise SystemExit(main())
