"""Bind arm-specific FF_AC baselines and explicitly defer other references.

Injection-end and clear-water reference JPEGs must later be corrected by the
same photometry arm. Until then, Studio may show the pre-injection baseline but
must not silently use a prior model's reference catalog.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


FORMAT = "ffac-two-arm-reference-availability-v1"


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def write_new_or_match(path: Path, value: dict[str, Any]) -> None:
    """Resume a partial sidecar write only when the existing bytes agree."""
    if path.is_file():
        if read(path) != value:
            raise FileExistsError(f"Existing reference sidecar differs: {path}")
        return
    write_new(path, value)


def baseline_bindings(materialization: dict[str, Any], inventory_root: Path,
                      inventory_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for revision in materialization["revisions"]:
        run = str(revision["run"])
        original_manifest_path = Path(revision["source_manifest_path"])
        if digest(original_manifest_path) != revision["source_manifest_sha256"]:
            raise ValueError(f"Original native manifest changed: {original_manifest_path}")
        original_manifest = read(original_manifest_path)
        inventory_path = inventory_root / "runs" / f"{run}.json"
        frozen_inventory_row = inventory_rows[run]
        if Path(frozen_inventory_row["path"]).resolve() != inventory_path.resolve():
            raise ValueError(f"Chart inventory row path changed for {run}")
        inventory_sha = digest(inventory_path)
        if inventory_sha != frozen_inventory_row["sha256"]:
            raise ValueError(f"Frozen chart inventory row changed for {run}")
        inventory = read(inventory_path)
        raw_baseline = Path(inventory["raw_baseline_path"])
        raw_baseline_sha = digest(raw_baseline)
        if raw_baseline_sha != inventory["raw_baseline_sha256"]:
            raise ValueError(f"Raw baseline changed after chart target inventory: {raw_baseline}")
        prior_raw = (original_manifest.get("reference") or {}).get("source_path")
        if prior_raw and Path(prior_raw).resolve() != raw_baseline.resolve():
            raise ValueError(f"Historical {run} baseline path differs from current raw rig")
        prior_sha = (original_manifest.get("reference") or {}).get("source_sha256")
        if prior_sha and prior_sha != raw_baseline_sha:
            raise ValueError(f"Historical {run} baseline raw SHA differs")
        receipt_raw_sha = revision.get("original_baseline_sha256")
        if receipt_raw_sha and receipt_raw_sha != raw_baseline_sha:
            raise ValueError(f"Materialization receipt {run} baseline raw SHA differs")
        original_reference = Path(revision["source_review_root"]) / "native_references" / run / "pre_injection.jpg"
        materialized_reference = Path(revision["materialized_review_root"]) / "native_references" / run / "pre_injection.jpg"
        original_reference_sha = digest(original_reference)
        materialized_reference_sha = digest(materialized_reference)
        if materialized_reference_sha != revision["materialized_baseline_sha256"]:
            raise ValueError(f"Materialized baseline changed: {materialized_reference}")
        per_revision_receipt = (Path(revision["materialized_review_root"]).parents[0] /
                                "materialization_receipt.json")
        if not per_revision_receipt.is_file():
            raise FileNotFoundError(per_revision_receipt)
        per_revision_sha = digest(per_revision_receipt)
        detail = read(per_revision_receipt)
        for key in ("arm", "run", "source_manifest_sha256", "materialized_manifest_sha256"):
            if detail[key] != revision[key]:
                raise ValueError(f"Per-revision receipt {run} disagrees on {key}")
        manifest_path = Path(revision["materialized_manifest_path"])
        if digest(manifest_path) != revision["materialized_manifest_sha256"]:
            raise ValueError(f"Materialized native manifest changed: {manifest_path}")
        row = dict(format=FORMAT, arm=materialization["arm"], run=run,
            source_manifest_path=revision["source_manifest_path"],
            source_manifest_sha256=revision["source_manifest_sha256"],
            chart_inventory_path=str(inventory_path),
            chart_inventory_sha256=inventory_sha,
            materialization_receipt_path=str(per_revision_receipt),
            materialization_receipt_sha256=per_revision_sha,
            materialized_manifest_path=str(manifest_path),
            materialized_manifest_sha256=revision["materialized_manifest_sha256"],
            baseline=dict(available=True, raw_path=str(raw_baseline),
                          raw_sha256=raw_baseline_sha,
                          original_native_path=str(original_reference),
                          original_native_sha256=original_reference_sha,
                          materialized_native_path=str(materialized_reference),
                          materialized_native_sha256=materialized_reference_sha),
            injection_ends=dict(available=False,
                reason="Arm-specific corrected injection-end reference assets have not been materialized"),
            clearwater=dict(available=False,
                reason="Arm-specific corrected clear-water reference assets have not been materialized"),
            old_reference_assets_reused=False)
        rows.append(row)
    return rows


def audit(materialization_receipt_path: Path, inventory_root: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    materialization = read(materialization_receipt_path)
    if materialization.get("complete") is not True and not preflight_only:
        raise ValueError("Reference audit requires a complete two-arm source materialization")
    revisions = materialization["revisions"]
    if len(revisions) != materialization["source_revisions"]:
        raise ValueError("Materialization revision count differs from its receipt")
    if len({revision["source_manifest_sha256"] for revision in revisions}) != len(revisions):
        raise ValueError("Duplicate native source manifest in materialization receipt")
    if materialization.get("complete") is True:
        selected = (materialization.get("selection") or {}).get(
            "selected_source_manifest_sha256")
        if selected is not None and sorted(revision["source_manifest_sha256"]
                                           for revision in revisions) != sorted(selected):
            raise ValueError("Complete materialization omits a selected frozen source revision")
    inventory_receipt_path = inventory_root / "inventory.json"
    inventory_receipt = read(inventory_receipt_path)
    if (inventory_receipt.get("failures") or [] or
            inventory_receipt.get("sampled_runs") != inventory_receipt.get("expected_runs")):
        raise ValueError("Baseline chart inventory is incomplete")
    inventory_rows = {str(row["run"]): row for row in inventory_receipt["run_receipts"]}
    if len(inventory_rows) != len(inventory_receipt["run_receipts"]):
        raise ValueError("Duplicate run in baseline chart inventory")
    rows = baseline_bindings(materialization, inventory_root, inventory_rows)
    result = dict(format=FORMAT, arm=materialization["arm"],
        materialization_receipt_path=str(materialization_receipt_path),
        materialization_receipt_sha256=digest(materialization_receipt_path),
        chart_inventory_root=str(inventory_root),
        chart_inventory_receipt_path=str(inventory_receipt_path),
        chart_inventory_receipt_sha256=digest(inventory_receipt_path),
        source_revisions=len(rows),
        source_manifest_sha256=sorted(row["source_manifest_sha256"] for row in rows),
        references=rows, pre_injection_available=True,
        injection_ends_available=False, clearwater_available=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    if preflight_only:
        return result
    for row in rows:
        target = Path(row["baseline"]["materialized_native_path"]).parent / "reference_availability.json"
        write_new_or_match(target, row)
        row["reference_availability_path"] = str(target)
        row["reference_availability_sha256"] = digest(target)
    output = materialization_receipt_path.parent / "reference_availability_receipt.json"
    write_new(output, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--chart-inventory-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    result = audit(args.materialization_receipt, args.chart_inventory_root,
                   preflight_only=args.preflight_only)
    print(json.dumps({key:result[key] for key in ("format", "arm", "source_revisions",
                                                    "pre_injection_available", "injection_ends_available")}),
          flush=True)


if __name__ == "__main__":
    main()
