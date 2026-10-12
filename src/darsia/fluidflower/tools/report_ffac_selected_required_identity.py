"""Read-only report of committed required-identity receipts; no numerical work."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path


def descriptor(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_bound(item):
    path = Path(item["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Saved receipt SHA differs: " + str(path))
    return json.loads(raw)


def build(root, destination):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError("A new report destination is required")
    plan_descriptor = descriptor(root / "identity_plan.json")
    plan = read_bound(plan_descriptor)
    binding = plan["phase_campaign_binding"]
    inventory = read_bound(binding["identity_inventory"])
    expected = {(run, row["studio_frame_index"]): row
                for run, entry in inventory["runs"].items() for row in entry["frames"]}
    inventory_cache = {}
    source_receipts, rows = [], []
    for path in sorted((root / "identity_frames").glob("*/frame_*.json")):
        item = descriptor(path)
        record = read_bound(item)
        key = (record["run"], record["original_studio_frame_index"])
        if key not in expected or record["identity_plan"] != plan_descriptor:
            raise ValueError("Unbound or unexpected saved identity row")
        if record["cpu_only"] is not True or record["internal_threads"] != 1:
            raise ValueError("Identity row is not from the required CPU execution")
        source_record = expected[key]["source_record"]
        source_inventory = source_record["inventory"]
        inventory_key = (source_inventory["path"], source_inventory["sha256"])
        if inventory_key not in inventory_cache:
            inventory_cache[inventory_key] = read_bound(source_inventory)
        source = next(row for row in inventory_cache[inventory_key]["sources"]
                      if row["work_index"] == source_record["work_index"])
        if source["source_sha256"] != record["source_sha256"] or source["elapsed_h"] != record["elapsed_h"]:
            raise ValueError("Camera source/time differs from the saved identity row")
        mask = record["frozen_gas"]
        inherited = record["stored_probabilities"]["inherited_map"]
        rows.append({"run": key[0], "studio_frame_index": key[1],
            "source_work_index": record["source_work_index"], "elapsed_h": record["elapsed_h"],
            "camera_datetime": source.get("camera_datetime"),
            "camera_datetime_status": "source_bound" if source.get("camera_datetime") else "not_recorded_in_source_inventory",
            "source_sha256": record["source_sha256"],
            "previous_source_sha256": record["previous_source_sha256"],
            "all_required_pass": record["all_pass"], "final_labels_changed": record["labels"]["changed_elements"],
            "inherited_union_exact": inherited["comparisons"]["co2_union"]["bit_identical"],
            "inherited_water_exact": inherited["comparisons"]["water"]["bit_identical"],
            "inherited_unresolved_exact": inherited["comparisons"]["unresolved"]["bit_identical"],
            "frozen_gas_changed_pixels": mask["changed_pixels"],
            "frozen_gas_added_pixels": mask["added_pixels"],
            "frozen_gas_removed_pixels": mask["removed_pixels"],
            "frozen_gas_first_yx": json.dumps(mask["first_yx"]),
            "first_required_difference": record["first_differing_required_product"],
            "seconds": record["seconds"], "receipt_sha256": item["sha256"]})
        source_receipts.append(item)
    stages = {}
    for stage in ("ac19", "full-runs", "all39"):
        path = root / (stage + "_stage.json")
        if path.exists():
            item = descriptor(path)
            record = read_bound(item)
            if record["identity_plan"] != plan_descriptor:
                raise ValueError("Stage receipt differs from the bound plan")
            stages[stage] = {"receipt": item, "complete": record["complete"]}
    stop_path = root / "identity_stop.json"
    stop = descriptor(stop_path) if stop_path.exists() else None
    failed = [row for row in rows if not row["all_required_pass"]]
    complete = len(rows) == len(expected) and not failed and stop is None and stages.get("all39", {}).get("complete") is True
    result = {"contract": "ffac-required-cpu-identity-readonly-report-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "report_tool": descriptor(__file__), "identity_plan": plan_descriptor,
        "frozen_gas_audit": binding["frozen_gas_audit"], "expected_distinct_frames": len(expected),
        "saved_frames": len(rows), "passed_frames": len(rows)-len(failed), "failed_frames": len(failed),
        "complete": complete, "stop": stop, "stage_receipts": stages, "row_receipts": source_receipts,
        "camera_source_inventories": [dict(path=key[0], sha256=key[1]) for key in inventory_cache],
        "frozen_gas_definition": "Diagnostic inherited gas/aqueous split; all other masks and fresh inherited union/water/255 are required.",
        "no_numerical_arrays_read": True, "no_model_calls": True, "no_maps_changed": True,
        "policy_a_started": False, "training": False, "mass_computed": False}
    destination.mkdir(parents=True, exist_ok=False)
    table = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (destination / "identity_frames.csv").write_text(table.getvalue(), encoding="utf-8", newline="")
    (destination / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    text = (f"Required CPU identity: {len(rows)}/{len(expected)} distinct frames saved; "
            f"{len(rows)-len(failed)} pass, {len(failed)} fail.\n"
            f"All required identity stages complete: {complete}.\n"
            "Each frame compares final labels, three stored probability layers, CO2, seven required extra masks, "
            "and the fresh inherited-map CO2 union/water/unresolved. The frozen_gas split is reported separately "
            "under the bound production reachability audit.\n"
            "This report reads only saved JSON receipts; no inference, mask modification, training, mass or Policy A dispatch.\n")
    if failed:
        first = failed[0]
        text += f"First saved required failure: {first['run']} Studio frame {first['studio_frame_index']}: {first['first_required_difference']}.\n"
    (destination / "SUMMARY.txt").write_text(text, encoding="utf-8")
    manifest = {"files": {path.name: descriptor(path) for path in destination.iterdir() if path.is_file()}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return descriptor(destination / "report.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.root, args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
