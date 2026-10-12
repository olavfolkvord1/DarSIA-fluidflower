"""Read-only input-bit comparison for failed baseline constructor assertions.

No darsia/model imports, no inference, no reference convention changes. Asset
SHA values below are the source-bound producer descriptors, not a new full
array SHA pass; small metadata files are verified before loading arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def descriptor(path):
    path = Path(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(item):
    raw = Path(item["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Metadata SHA changed: " + item["path"])
    return json.loads(raw)


def ordered(bits):
    return np.where(bits & np.uint32(0x80000000), ~bits,
                    bits | np.uint32(0x80000000)).astype(np.int64)


def compare(current, original, valid, *, chunk_rows=64):
    shapes = dict(current_shape=list(current.shape),original_shape=list(original.shape),
        current_dtype=str(current.dtype),original_dtype=str(original.dtype),valid_dtype=str(valid.dtype),
        valid_shape=list(valid.shape))
    if (valid.dtype != np.bool_ or original.ndim != 3 or original.shape[0] != 3
            or valid.shape != original.shape[1:]):
        return dict(shapes, exact_assertion_pass=False, original_support_shape_or_dtype_mismatch=True)
    if current.dtype != np.float32 or original.dtype != np.float32 or current.shape != original.shape:
        return dict(shapes, exact_assertion_pass=False, shape_or_dtype_mismatch=True)
    counts = np.zeros(3,dtype=np.int64)
    pixels = nan_components = 0
    first = None
    max_delta = 0.
    max_ulp = 0
    support_agrees = True
    for start in range(0,valid.shape[0],chunk_rows):
        mask = np.asarray(valid[start:start+chunk_rows])
        old = np.asarray(original[:,start:start+chunk_rows])
        new = np.asarray(current[:,start:start+chunk_rows])
        support_agrees &= bool(np.array_equal(mask,np.isfinite(old).all(axis=0)))
        diff = (old.view(np.uint32) != new.view(np.uint32)) & mask[None]
        counts += diff.sum(axis=(1,2))
        pixels += int(diff.any(axis=0).sum())
        finite = np.isfinite(old) & np.isfinite(new) & diff
        nan_components += int((diff & ~np.isfinite(new)).sum())
        if finite.any():
            max_delta = max(max_delta,float(np.max(np.abs(old[finite].astype(np.float64)-new[finite].astype(np.float64)))))
            max_ulp = max(max_ulp,int(np.max(np.abs(ordered(old[finite].view(np.uint32))-ordered(new[finite].view(np.uint32))))))
        if first is None and diff.any():
            c,y,x = (int(v) for v in np.argwhere(diff)[0])
            before, after = old[c,y,x], new[c,y,x]
            first = dict(channel=c,y=start+y,x=x,original_bits=int(before.view(np.uint32)),
                current_bits=int(after.view(np.uint32)),original_value=float(before),
                current_value=float(after) if np.isfinite(after) else str(after))
    return dict(shapes, exact_assertion_pass=int(counts.sum()) == 0,
        original_valid_pixels=int(np.count_nonzero(valid)),original_support_consistent=support_agrees,
        changed_components=int(counts.sum()),changed_pixels=pixels,per_channel_changed=counts.tolist(),
        current_nonfinite_changed_components=nan_components,first_difference=first,
        max_absolute_log_delta=max_delta,max_finite_ulp_delta=max_ulp,
        comparison="literal float32 uint32 bits on original valid support; invalid exterior ignored")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-plan",required=True);parser.add_argument("--root",required=True)
    parser.add_argument("--runs",nargs="+",required=True);parser.add_argument("--destination",required=True)
    args=parser.parse_args(argv)
    destination=Path(args.destination)
    if destination.exists():raise FileExistsError("New diagnostic destination required")
    plan_asset=descriptor(args.identity_plan);plan=read(plan_asset)
    rows=[]
    for run in args.runs:
        state=plan["runs"][run]
        inv=read(state["source_inventory"])
        native=read(state["parent_native_manifest"])["reference"]
        digest=inv["baseline_source_sha256"]
        receipt_asset=descriptor(Path(plan["scientific_source_root"])/"selected_sources"/run/digest/"receipt.json")
        receipt=read(receipt_asset)
        current_asset=receipt["native_scientific"]["log_linear_rgb"]
        old_asset=dict(path=native["v5_log_linear_rgb_path"],sha256=native["v5_log_linear_rgb_sha256"])
        valid_asset=dict(path=native["v5_log_linear_valid_mask_path"],sha256=native["v5_log_linear_valid_mask_sha256"])
        current=np.load(current_asset["path"],allow_pickle=False,mmap_mode="r")
        original=np.load(old_asset["path"],allow_pickle=False,mmap_mode="r")
        valid=np.load(valid_asset["path"],allow_pickle=False,mmap_mode="r")
        result=dict(run=run,baseline_source_sha256=digest,native_manifest=state["parent_native_manifest"],
            source_receipt=receipt_asset,current_log=current_asset,original_log=old_asset,
            original_valid=valid_asset,comparison=compare(current,original,valid))
        rows.append(result)
        print(json.dumps(dict(event="read_only_baseline_comparison",run=run,comparison=result["comparison"])),flush=True)
        del current,original,valid
    destination.mkdir(parents=True,exist_ok=False)
    report=dict(contract="ffac-required-identity-baseline-readonly-comparison-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),identity_plan=plan_asset,
        code=descriptor(__file__),rows=rows,model_executed=False,arrays_modified=False,
        full_array_sha_recomputed=False,asset_sha_provenance="exact bound producer descriptors",
        actual_first_failed_scheduler_run="not persisted by immutable producer; not inferred from comparison ordering")
    (destination/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(descriptor(destination/"comparison.json")))
    return 0


if __name__=="__main__":raise SystemExit(main())
