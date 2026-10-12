"""Independently verify a top-domain clip and report nominal before/after areas."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path


def verify_frame(row: dict) -> dict:
    before_path = resolve_review_path(row["parent_labels_map_path"])
    after_path = resolve_review_path(row["map_path"])
    with Image.open(before_path) as image:
        before = np.asarray(image, dtype=np.uint8).copy()
    with Image.open(after_path) as image:
        after = np.asarray(image, dtype=np.uint8).copy()
    with np.load(resolve_review_path(row["mask_path"]), allow_pickle=False) as masks:
        keep = masks["top_keep"].copy()
        old_photo = masks["photo_sand_original"].copy()
        new_photo = masks["photo_sand_revised"].copy()
    if (keep.dtype != np.bool_ or keep.shape != before.shape or after.shape != before.shape or
            old_photo.dtype!=np.bool_ or new_photo.dtype!=np.bool_ or
            old_photo.shape!=keep.shape or new_photo.shape!=keep.shape):
        raise ValueError("Top-mask or categorical grid differs")
    if not np.array_equal(new_photo, old_photo & keep):
        raise ValueError("Photo domain expanded or changed away from top clipping")
    if not keep[-1].all() or np.any(keep[:-1] & ~keep[1:]):
        raise ValueError("Top keep-mask is not column-monotone")
    if not np.array_equal(before[keep], after[keep]) or np.any(after[~keep] != 255):
        raise ValueError("Categorical labels changed inside sand or survived above it")
    with Image.open(resolve_review_path(row["parent_map_path"])) as image:
        old_rgba = np.asarray(image).copy()
    with Image.open(resolve_review_path(row["studio_map_path"])) as image:
        new_rgba = np.asarray(image).copy()
    x, y, w, h = row["content_box_xywh"]
    old_content, new_content = old_rgba[y:y+h, x:x+w], new_rgba[y:y+h, x:x+w]
    cyan_cleanup = keep & ~new_photo & (before == 255)
    retained_display = keep & ~cyan_cleanup
    if not np.array_equal(old_content[retained_display], new_content[retained_display]):
        raise ValueError("Studio phase overlay changed inside real sand")
    if np.any(new_content[~keep, 3]):
        raise ValueError("Studio phase overlay survives above the boundary")
    if np.any(new_content[cyan_cleanup, 3]):
        raise ValueError("Unresolved cyan survives outside photographically supported sand")
    result = {key: row[key] for key in ("run", "frame_index", "elapsed_h", "pixel_area_mm2")}
    for phase, old_mask, new_mask in (
        ("co2", (before == 1) | (before == 2), (after == 1) | (after == 2)),
        ("gas", before == 2, after == 2),
    ):
        old_count, new_count = int(old_mask.sum()), int(new_mask.sum())
        if (old_count, new_count) != (row[f"before_{phase}_pixels"], row[f"after_{phase}_pixels"]):
            raise ValueError("QA area counts differ from actual categorical files")
        result.update({
            f"before_{phase}_pixels": old_count, f"after_{phase}_pixels": new_count,
            f"before_{phase}_cm2": old_count * row["pixel_area_mm2"] / 100,
            f"after_{phase}_cm2": new_count * row["pixel_area_mm2"] / 100,
            f"removed_{phase}_cm2": (old_count-new_count) * row["pixel_area_mm2"] / 100,
            f"removed_{phase}_percent": 100 * (old_count-new_count) / old_count if old_count else 0.,
        })
    result.update(removed_photo_pixels=int((old_photo & ~new_photo).sum()),
                  retained_co2_changed_pixels=0, retained_gas_changed_pixels=0,
                  retained_labels_changed_pixels=0, retained_phase_overlay_changed_pixels=0,
                  cyan_removed_outside_photo_pixels=int(np.count_nonzero(cyan_cleanup & (old_content[...,3]>0))))
    return result


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _complete_inventory(declared: list[dict]) -> set[tuple[str, int]]:
    keys = {(row["run"], row["frame_index"]) for row in declared}
    if len(keys) != 864 or len(declared) != 864 or len({run for run, _ in keys}) != 43:
        raise ValueError("Report requires all 43 runs / 864 distinct frames")
    return keys


def _write_report(model: Path, output: Path, qa: dict, rows: list[dict], *,
                  streaming: dict | None = None) -> dict:
    """The same aggregation is used by completed and concurrent verification."""
    _complete_inventory(rows)
    rows = sorted(rows, key=lambda row:(row["run"],row["frame_index"]))
    summaries, snapshots = [], []
    for run in sorted({row["run"] for row in rows}):
        group = [row for row in rows if row["run"] == run]
        summary = dict(run=run, frame_count=len(group), first_h=min(r["elapsed_h"] for r in group),
                       last_h=max(r["elapsed_h"] for r in group))
        for phase in ("co2", "gas"):
            for kind in ("before", "after", "removed"):
                values = [r[f"{kind}_{phase}_cm2"] for r in group]
                summary[f"mean_{kind}_{phase}_cm2"] = sum(values)/len(values)
                summary[f"max_{kind}_{phase}_cm2"] = max(values)
        summary["retained_co2_changed_pixels"] = 0
        summary["retained_gas_changed_pixels"] = 0
        summaries.append(summary)
        for target in (0., 5., 12., 24.):
            nearest = min(group, key=lambda r: (abs(r["elapsed_h"]-target),r["frame_index"]))
            snapshots.append(dict(target_h=target, actual_h=nearest["elapsed_h"],
                                  time_offset_h=nearest["elapsed_h"]-target, **nearest))
    if streaming is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("Streaming verification output disappeared")
    _csv(output / "all_864_frames_before_after.csv", rows)
    _csv(output / "all_43_runs_before_after.csv", summaries)
    _csv(output / "all_43_runs_nearest_0_5_12_24h.csv", snapshots)
    cohort=[]
    for target in (0.,5.,12.,24.):
        group=[row for row in snapshots if row['target_h']==target]
        aggregate=dict(target_h=target,runs=len(group),min_actual_h=min(r['actual_h'] for r in group),
                       max_actual_h=max(r['actual_h'] for r in group),
                       max_abs_time_offset_h=max(abs(r['time_offset_h']) for r in group))
        for phase in ('co2','gas'):
            for kind in ('before','after','removed'):
                aggregate[f'sum_{kind}_{phase}_cm2']=sum(r[f'{kind}_{phase}_cm2'] for r in group)
            old=aggregate[f'sum_before_{phase}_cm2']
            aggregate[f'removed_{phase}_percent']=100*aggregate[f'sum_removed_{phase}_cm2']/old if old else 0.
        cohort.append(aggregate)
    _csv(output / 'cohort_nearest_0_5_12_24h.csv',cohort)
    result = dict(contract="ffac-top-surface-independent-area-report-v1", created_utc=utc_now(),
        model_root=str(model), model_identity=qa["model_identity"],
        qa_sha256=sha256_file(model / "production_qa/top_surface_qa.json"),
        runs=43, frames=864, all_pass=True,
        retained_co2_changed_pixels=0, retained_gas_changed_pixels=0,
        nominal_scale_only=True, mass_analysis=False,
        statistics_note="Run summaries are arithmetic means/maxima over saved frames, not time-integrated areas. "
                        "Snapshots use the nearest saved frame and report its actual time and offset; no interpolation.",
        files={p.name:sha256_file(p) for p in output.glob("*.csv")})
    if streaming is not None:
        result["streaming_verification"] = streaming
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False))
    return result


def report(model: Path, output: Path, *, workers: int = 8) -> dict:
    qa_path = model / "production_qa/top_surface_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    declared = qa.get("rows") or []
    _complete_inventory(declared)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(verify_frame, declared, chunksize=1))
    return _write_report(model, output, qa, rows)


FOLLOW_CONTRACT = "ffac-top-surface-concurrent-independent-verification-v1"
FRAME_CONTRACT = "ffac-g3-top-surface-candidate-v1"
QA_CONTRACT = "ffac-g3-top-surface-qa-v1"


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _atomic_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Independent verification will not overwrite a completed cache entry")
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temporary, path)


def _below_normal() -> None:
    import psutil
    process = psutil.Process()
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        process.nice(5)


def _file_bindings(receipt: dict) -> list[dict]:
    qa, binding = receipt["qa"], receipt["binding"]
    fields = {
        "map":receipt["labels_map"], "studio_map":receipt["maps"]["combined"],
        "parent_map":binding["parent_phase_map"],
        "parent_labels_map":binding["parent_labels_map"], "mask":binding["mask"],
    }
    for prefix, asset in fields.items():
        if (qa[prefix+"_sha256"] != asset["sha256"] or
                resolve_review_path(qa[prefix+"_path"]).resolve() !=
                resolve_review_path(asset["path"]).resolve()):
            raise ValueError("Frame QA differs from its immutable receipt assets")
    assets = [*receipt["maps"].values(), receipt["labels_map"], receipt["labels_archive"],
              *fields.values(),
              {"path":qa["native_manifest_path"],"sha256":qa["native_manifest_sha256"]},
              {"path":qa["photo_sand_mask_path"],"sha256":qa["photo_sand_mask_sha256"]}]
    files = {}
    for asset in assets:
        path = str(resolve_review_path(asset["path"]).resolve())
        digest = asset["sha256"]
        if (not isinstance(digest,str) or len(digest)!=64 or
                (path in files and files[path] != digest)):
            raise ValueError("Frame receipt file binding is ambiguous")
        files[path] = digest
    return [dict(path=path,sha256=digest) for path,digest in sorted(files.items())]


def _verify_hashes(files: list[dict]) -> None:
    for asset in files:
        if sha256_file(resolve_review_path(asset["path"])) != asset["sha256"]:
            raise ValueError("A verified frame file changed after its production receipt")


def _expected_binding(source: dict, pipeline: str) -> dict:
    return dict(run=source["run"],frame_index=int(source["frame_index"]),
                pipeline_sha256=pipeline,mask=source["mask"],
                parent_phase_map=source["parent_phase_map"],
                parent_labels_map=source["parent_labels_map"],
                shadow_native_manifest_sha256=source["shadow_native_manifest_sha256"])


def verify_completed_receipt(path: str, source: dict, pipeline: str, parent: str) -> dict:
    """Decode each independently bound categorical/overlay image exactly once."""
    receipt_path = resolve_review_path(path)
    receipt_sha = sha256_file(receipt_path)
    receipt = _read_json(receipt_path)
    if (receipt.get("contract")!=FRAME_CONTRACT or receipt.get("complete") is not True or
            receipt.get("binding") != _expected_binding(source,pipeline) or
            any(receipt.get(key) is not False for key in
                ("model_inference","training","component_filtering","photo_pixels_changed"))):
        raise ValueError("Only atomically completed, exact-plan clipping receipts may be verified")
    qa=receipt["qa"]
    if ((qa["run"],qa["frame_index"]) != (source["run"],source["frame_index"]) or
            qa["elapsed_h"]!=source["elapsed_h"] or
            qa["native_manifest_sha256"]!=source["shadow_native_manifest_sha256"] or
            qa["content_box_xywh"]!=source["content_box_xywh"]):
        raise ValueError("Frame QA identifies another source, time or geometry")
    parent_receipt_path=Path(parent)/"production_frames"/source["run"]/f"frame_{source['frame_index']:03d}.json"
    parent_receipt_sha=sha256_file(parent_receipt_path)
    parent_receipt=_read_json(parent_receipt_path)
    if qa["pixel_area_mm2"]!=parent_receipt["production_audit"]["pixel_area_mm2"]:
        raise ValueError("Frame QA changed the parent's nominal pixel area")
    files=_file_bindings(receipt)
    _verify_hashes(files)
    result=verify_frame(qa)
    _verify_hashes(files)
    if sha256_file(receipt_path)!=receipt_sha:
        raise ValueError("Atomic production receipt changed during independent verification")
    return dict(contract=FOLLOW_CONTRACT,run=source["run"],frame_index=source["frame_index"],
                receipt_path=str(receipt_path),receipt_sha256=receipt_sha,
                pipeline_sha256=pipeline,qa=qa,qa_canonical_sha256=_canonical_sha(qa),
                parent_receipt=dict(path=str(parent_receipt_path),sha256=parent_receipt_sha),
                files=files,result=result)


def validate_cached_frame(entry: dict) -> dict:
    """Check immutable bytes at sealing, without decoding those pixels again."""
    path=resolve_review_path(entry["receipt_path"])
    receipt=_read_json(path)
    if (entry.get("contract")!=FOLLOW_CONTRACT or sha256_file(path)!=entry["receipt_sha256"] or
            receipt.get("complete") is not True or receipt["qa"]!=entry["qa"] or
            _canonical_sha(receipt["qa"])!=entry["qa_canonical_sha256"] or
            _file_bindings(receipt)!=entry["files"]):
        raise ValueError("Concurrent verification cache no longer matches its production receipt")
    _verify_hashes(entry["files"])
    _verify_hashes([entry["parent_receipt"]])
    return entry["result"]


def _producer_state(exit_path: Path | None, pid: int | None, *, sealed: bool) -> bool:
    """Return readiness, failing on a producer failure before publishing CSVs."""
    if exit_path is not None:
        if not exit_path.is_file():
            if pid is not None:
                import psutil
                if not psutil.pid_exists(pid):
                    raise RuntimeError("Producer exited without its success receipt; report remains unpublished")
            return False
        result=_read_json(exit_path)
        if result.get("succeeded") is not True or result.get("returncode")!=0:
            raise RuntimeError("Producer failed; independent area reports remain unpublished")
        if not sealed:
            raise RuntimeError("Successful producer has no complete candidate seal")
        return True
    return sealed


def validate_sealed_coverage(qa: dict, sealed: dict, cached: dict,
                             expected: set[tuple[str,int]]) -> None:
    if (qa.get("contract")!=QA_CONTRACT or qa.get("all_pass") is not True or
            sealed.get("contract")!=FRAME_CONTRACT or sealed.get("complete") is not True or
            qa.get("run_count")!=43 or qa.get("frame_count")!=864 or
            sealed.get("model_identity")!=qa.get("model_identity")):
        raise ValueError("Streaming report requires the successful 43/864 production seal")
    declared=qa.get("rows") or []
    if (_complete_inventory(declared)!=expected or set(cached)!=expected or
            len(sealed.get("frames") or [])!=864):
        raise ValueError("Streaming report cannot publish partial or mismatched coverage")
    sealed_frames={(r["run"],r["frame_index"]):r for r in sealed["frames"]}
    if set(sealed_frames)!=expected:
        raise ValueError("Production seal has duplicate or missing frame receipts")
    for row in declared:
        key=row["run"],row["frame_index"]
        item=cached[key];proof=sealed_frames[key]
        if (row!=item["qa"] or _canonical_sha(row)!=item["qa_canonical_sha256"] or
                proof["receipt_sha256"]!=item["receipt_sha256"] or
                resolve_review_path(proof["receipt_path"]).resolve()!=
                resolve_review_path(item["receipt_path"]).resolve()):
            raise ValueError("Sealed QA or frame receipt differs from concurrent verification")


def follow_candidate(model: Path, output: Path, *, workers: int = 3,
                     poll_seconds: float = 15., max_wait_seconds: float = 21600.,
                     producer_exit_receipt: Path | None = None,
                     producer_pid: int | None = None) -> dict:
    """Verify completed atomic frame receipts concurrently with the producer."""
    if not 1<=workers<=8 or not 0<poll_seconds<=60 or max_wait_seconds<=0:
        raise ValueError("Invalid bounded streaming verifier resource/wait settings")
    _below_normal()
    model=model.resolve();output=output.resolve()
    plan_path=model/"top_surface_candidate_plan.json"
    plan=_read_json(plan_path);plan_sha=sha256_file(plan_path)
    expected=_complete_inventory(plan["source_inventory"])
    inventory={(r["run"],r["frame_index"]):r for r in plan["source_inventory"]}
    output.mkdir(parents=True,exist_ok=False)
    _atomic_new_json(output/"stream_binding.json",dict(contract=FOLLOW_CONTRACT,
        candidate=str(model),plan_sha256=plan_sha,pipeline_sha256=plan["pipeline_sha256"],
        verifier_code_sha256=sha256_file(Path(__file__)),workers=workers,
        producer_exit_receipt=str(producer_exit_receipt) if producer_exit_receipt else None,
        producer_pid=producer_pid,started_utc=utc_now()))
    cached,pending={},{}
    start=time.monotonic();next_scan=0.;last_log=0.;available={}
    qa_path=model/"production_qa/top_surface_qa.json"
    seal_path=model/"top_surface_candidate_receipt.json"
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                             initializer=_below_normal) as pool:
        while True:
            now=time.monotonic()
            if now-start>max_wait_seconds:
                raise TimeoutError("Bounded streaming verification expired without complete successful coverage")
            sealed=seal_path.is_file()
            producer_ready=_producer_state(producer_exit_receipt,producer_pid,sealed=sealed)
            if now>=next_scan:
                for path in sorted((model/"top_surface_frames").glob("*/frame_*.json")):
                    run=path.parent.name
                    try:index=int(path.stem.removeprefix("frame_"))
                    except ValueError:raise ValueError("Ambiguous production frame receipt name")
                    key=run,index
                    if key not in expected:
                        raise ValueError("Production receipt belongs to a different frame inventory")
                    if key not in cached and key not in pending.values():available[key]=path
                next_scan=now+poll_seconds
            while available and len(pending)<workers*2:
                key=min(available);path=available.pop(key)
                future=pool.submit(verify_completed_receipt,str(path),inventory[key],
                    plan["pipeline_sha256"],plan["binding"]["parent"])
                pending[future]=key
            if pending:
                done,_=wait(pending,timeout=min(1.,poll_seconds),return_when=FIRST_COMPLETED)
                for future in done:
                    key=pending.pop(future);entry=future.result()
                    if (entry["run"],entry["frame_index"])!=key:
                        raise ValueError("Independent verifier returned another assigned frame")
                    _atomic_new_json(output/"frame_verification_cache"/key[0]/f"frame_{key[1]:03d}.json",entry)
                    cached[key]=entry
            elif not producer_ready:
                time.sleep(min(poll_seconds, max(.1,next_scan-time.monotonic())))
            if time.monotonic()-last_log>=30. or len(cached)==864:
                print(json.dumps(dict(stage="independent_stream_verification",verified_frames=len(cached),
                    pending_frames=len(pending),ready_receipts=len(available),expected_frames=864,
                    producer_succeeded=producer_ready,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
                last_log=time.monotonic()
            if len(cached)==864 and not pending and producer_ready:
                if sha256_file(plan_path)!=plan_sha:
                    raise ValueError("Candidate source plan changed while verification followed it")
                qa=_read_json(qa_path);seal=_read_json(seal_path)
                if seal.get("source_plan_sha256")!=plan_sha or seal.get("qa")!={
                        "path":str(qa_path),"sha256":sha256_file(qa_path)}:
                    raise ValueError("Production seal differs from its exact plan/QA file")
                validate_sealed_coverage(qa,seal,cached,expected)
                rows=list(pool.map(validate_cached_frame,cached.values(),chunksize=1))
                result=_write_report(model,output,qa,rows,streaming=dict(
                    contract=FOLLOW_CONTRACT,pixel_frames_verified_once=864,
                    cache_rows_matched_exact_sealed_qa=864,receipt_and_file_sha_revalidated=864,
                    plan_sha256=plan_sha,producer_success_required=producer_exit_receipt is not None,
                    producer_exit_sha256=sha256_file(producer_exit_receipt) if producer_exit_receipt else None,
                    candidate_seal_sha256=sha256_file(seal_path),workers=workers,
                    verifier_code_sha256=sha256_file(Path(__file__))))
                return result


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--follow-candidate",action="store_true")
    parser.add_argument("--poll-seconds",type=float,default=15.)
    parser.add_argument("--max-wait-seconds",type=float,default=21600.)
    parser.add_argument("--producer-exit-receipt",type=Path)
    parser.add_argument("--producer-pid",type=int)
    args=parser.parse_args(argv)
    result=(follow_candidate(args.model,args.output,workers=args.workers,
        poll_seconds=args.poll_seconds,max_wait_seconds=args.max_wait_seconds,
        producer_exit_receipt=args.producer_exit_receipt,producer_pid=args.producer_pid)
        if args.follow_candidate else report(args.model,args.output,workers=args.workers))
    print(json.dumps(result),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
