"""Seal fixed-metric ring QA rows as separate internal and blind-safe reports.

This helper does not create or modify any model, label, mask, or acceptance
threshold. Frame measurements are produced by phase_ffac_old_domain_ring_qa.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from darsia.fluidflower.phase_ffac_old_domain_ring_qa import CONTRACT, RADIUS_PX

BLIND_RUNS = frozenset(("ac30", "ac49", "ac54"))


def report_document(rows, *, public):
    """Omit individual blind rows and their provenance from the public report."""
    identities = [(row["run"], row["frame_index"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("QA report contains duplicate frame identities")
    for row in rows:
        for side in ("before", "after"):
            if row[side].get("contract") != CONTRACT or row[side].get("radius_px") != RADIUS_PX:
                raise ValueError("QA rows use a different fixed ring metric")
            if row[side].get("acceptance_threshold") is not None or row[side].get("acceptance_claimed") is not False:
                raise ValueError("This QA report cannot introduce an acceptance threshold")
    selected = [row for row in rows if not public or row["run"].casefold() not in BLIND_RUNS]
    selected.sort(key=lambda row: (row["run"], row["frame_index"]))
    return dict(contract=CONTRACT, radius_px=RADIUS_PX,
        report_scope="public nonblind frames" if public else "internal all selected frames",
        frame_count=len(selected), run_count=len(set(row["run"] for row in selected)),
        blind_individual_rows_included=not public,
        acceptance_threshold=None, acceptance_claimed=False, mass_eligible=False,
        changed_pixels_total=sum(row["change"]["changed_pixels"] for row in selected),
        changed_frames=sum(row["change"]["changed_pixels"] > 0 for row in selected),
        frames=selected)


def _csv_rows(document):
    for row in document["frames"]:
        result = {key: row[key] for key in ("run", "frame_index", "elapsed_h")}
        result["changed_pixels"] = row["change"]["changed_pixels"]
        for side in ("before", "after"):
            metric = row[side]
            result[f"{side}_ring_pixels"] = metric["ring_pixels"]
            for population in ("ring", "interior"):
                for key in ("eligible_pixels", "no_co2_pixels", "no_co2_share"):
                    result[f"{side}_{population}_{key}"] = metric[population][key]
            result[f"{side}_ring_minus_interior_no_co2_share"] = metric["no_co2_share_difference"]
        for phase in ("co2", "gas", "aqueous"):
            for key in ("before_area_cm2", "after_area_cm2", "signed_change_area_cm2"):
                result[f"{phase}_{key}"] = row["change"]["phases"][phase][key]
        cutoff = row["change"]["four_connected_minimum"]
        for key in ("minimum_scope", "component_minimum_area_weights", "pre_minimum_gas_pixels",
                    "withdrawn_components", "withdrawn_pixels", "withdrawn_area_cm2",
                    "withdrawn_original_inference_area_cm2", "cutoff_only_post_gas_pixels"):
            result[f"four_connected_minimum_{key}"] = cutoff[key] if cutoff else None
        yield result


def write_reports(rows, output_dir):
    """Write to a fresh local report directory, retaining internal evidence."""
    output = Path(output_dir)
    if str(output).startswith("\\\\"):
        raise ValueError("This read-only QA run writes new reports locally")
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for visibility, public in (("internal", False), ("public", True)):
        document = report_document(rows, public=public)
        path = output / f"ring_qa_{visibility}.json"
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        csv_rows = list(_csv_rows(document))
        csv_path = output / f"ring_qa_{visibility}.csv"
        if csv_rows:
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
        paths[visibility] = dict(json_path=str(path), csv_path=str(csv_path),
                                 frame_count=document["frame_count"])
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-rows-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    raw = json.loads(args.frame_rows_json.read_text(encoding="utf-8-sig"))
    rows = raw if isinstance(raw, list) else raw["frames"]
    print(json.dumps(write_reports(rows, args.output_dir)))


if __name__ == "__main__":
    main()
