"""Bounded canonical-C dependency closure for assigned frozen source tasks.

No inference, labels or maps are read. The frozen photo cohort's actual frame
validator is executed with a recording check hook: JSON SHA checks remain live,
while acquisition/array assets are listed for byte-exact staging verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class Dependencies:
    def __init__(self):
        self.assets, self.json_verified = {}, set()
        self.task, self.index = None, None

    def add(self, path, sha, role):
        if not isinstance(path, str) or not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha) is None:
            raise ValueError("Runtime dependency lacks its exact declared path/SHA: " + role)
        key = str(Path(path).resolve())
        value = self.assets.get(key)
        if value is not None and value["sha256"] != sha.lower():
            raise ValueError("One canonical dependency has conflicting SHAs: " + key)
        if value is None:
            local = bool(re.match(r"^[Cc]:", key))
            exists = Path(key).is_file() if local or Path(key).suffix.casefold() == ".json" else None
            value = dict(path=key, sha256=sha.lower(), canonical_C=local, exists=exists,
                bytes=Path(key).stat().st_size if exists else None, roles=[], uses=[])
            self.assets[key] = value
        if role not in value["roles"]:
            value["roles"].append(role)
        use = dict(task_id=self.task, frame_index=self.index)
        if use not in value["uses"]:
            value["uses"].append(use)
        return Path(key)

    def check(self, path, sha, role):
        result = self.add(path, sha, role)
        if result.suffix.casefold() == ".json" and str(result) not in self.json_verified:
            if file_sha(result) != sha.lower():
                raise ValueError("Metadata dependency changed: " + str(result))
            self.json_verified.add(str(result))
        return result

    def pairs(self, value, prefix="declared"):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                self.add(value["path"], value["sha256"], prefix)
            for key, path in value.items():
                if key.endswith("_path") and isinstance(path, str):
                    sha = value.get(key[:-5] + "_sha256")
                    if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha):
                        self.add(path, sha, prefix + ":" + key)
                if isinstance(path, (dict, list)):
                    self.pairs(path, prefix + ":" + key)
        elif isinstance(value, list):
            for row in value:
                self.pairs(row, prefix)


def bound_metadata_asset(deps, asset, *, base, role):
    """Record the declared path before any runtime storage alias is applied."""
    path = Path(asset["path"])
    if not path.is_absolute():
        path = Path(base) / path
    return read(deps.check(str(path), asset["sha256"], role))


def rig_loader_metadata(deps, native, native_path):
    """Close the three small certificates read by the rig reader constructor.

    ``load_native_source(..., verify_geometry=False)`` still reads both its
    parent native manifest and camera clock run. Their paths need staging even
    though geometry and scientific pixels are evaluated through producer data.
    """
    asset = native["ffac_rig_camera_sources"]["input_manifest"]
    inputs = bound_metadata_asset(deps, asset, base=Path(native_path).parent,
                                  role="rig loader input manifest")
    path = Path(asset["path"])
    if not path.is_absolute():
        path = Path(native_path).parent / path
    for key in ("parent_native_manifest", "camera_timeline_run"):
        bound_metadata_asset(deps, inputs[key], base=path.parent,
                             role="rig loader " + key)


def run(plan_path, host, output):
    plan = read(plan_path)
    sys.path.insert(0, str(Path(plan["frozen_code_root"]) / "src"))
    from darsia.fluidflower.tools import ffac_v5_track1_photo_context as photo
    from darsia.fluidflower.phase_gas_inference_chunks import previous_frame_indices
    from darsia.fluidflower.phase_training.v5_linear_input import RunV5LinearInput
    from darsia.fluidflower.phase_training import v5_linear_input as linear
    deps = Dependencies()
    original_check = photo._check
    photo._check = deps.check
    tasks = [t for t in plan["tasks"] if t["host"] == host]
    summary = []
    try:
        for task in tasks:
            deps.task, deps.index = task["task_id"], None
            native_path = deps.check(task["source"]["asset_manifest_path"],
                                     task["binding"]["native_manifest_sha256"], "exact source manifest")
            native = read(native_path)
            completed = {int(item["name"][6:9]) for item in task["imported_files"]
                         if re.fullmatch(r"frame_[0-9]+\.json", item["name"])}
            selected = sorted({r["frame_index"] for r in task["regions"]} - completed)
            predecessors = previous_frame_indices(len(native["frame_sources"]),
                tuple(task["source"]["independent_reference_frames"]))
            required = sorted(set(selected) | {predecessors[i] for i in selected if predecessors[i] is not None})
            if native.get("v5_track1_photo_geometry"):
                cohort = photo.Track1FramePhotoCohort(native_path)
                reader = RunV5LinearInput(native_manifest_path=native_path, run_id=task["source"]["run"],
                                           content_shape_hw=tuple(native["analysis_image_shape_hw"]))
                domain = (native.get("v5_photometry") or {}).get("original_domain")
                if domain:
                    deps.pairs(domain, "nominal source spacing")
                else:
                    parent = Path(task["binding"]["legacy_context_candidate"])
                    profile = read(parent / "phase_optical_regions.json")
                    entry = profile["runs"]["ffac/" + native["run"].casefold()]
                    context = (parent / entry["path"]).resolve()
                    if not context.is_relative_to(parent.resolve()):
                        raise ValueError("Legacy nominal spacing escaped its frozen candidate")
                    deps.add(str(context), entry["sha256"], "frozen parent nominal spacing")
                records = [(None, reader.reference)] + [(i, native["frame_sources"][i]) for i in required]
                for index, record in records:
                    deps.index = index
                    deps.pairs(record, "native scientific/source row")
                    if index is None:
                        jpeg = Path(task["source"]["review_root"]) / "native_references" / task["source"]["run"] / "pre_injection.jpg"
                        deps.add(str(jpeg), record["materialized_native_sha256"], "linear reader baseline JPEG")
                    else:
                        jpeg = Path(task["source"]["review_root"]) / "native_frames" / task["source"]["run"] / f"frame_{index:03d}.jpg"
                        deps.add(str(jpeg), record["native_jpeg_sha256"], "linear reader native JPEG")
                        cohort.frame(index)  # Actual frozen receipt/source consistency validation.
                # The decoder's bound-source method additionally enforces exact
                # revision paths. Record its checks without decoding float arrays.
                old_sha = linear._sha256
                def recorded_sha(path):
                    key = str(Path(path).resolve())
                    if key in deps.assets:
                        return deps.assets[key]["sha256"]
                    return old_sha(Path(path))
                linear._sha256 = recorded_sha
                try:
                    for index, record in records:
                        deps.index = index
                        frame_id = "pre_injection" if index is None else f"frame_{index:03d}"
                        jpeg = reader.review_root / ("native_references" if index is None else "native_frames") / reader.run_id / ("pre_injection.jpg" if index is None else frame_id+".jpg")
                        reader._bound_source(jpeg, frame_id)
                        if (native.get("v5_temporal_reference") or {}).get("contract") == "ffac-v5-source-backed-extended-ring-v1" and index not in selected:
                            reader._bound_extended_ring(record, jpeg)
                finally:
                    linear._sha256 = old_sha
                kind = "legacy exact photo cohort"
            else:
                # Rig T2 inputs use source-bound full-scientific descriptors.
                # Their nested SHA assets and prepared H receipts are closed
                # below over selected source frames only, without float reads.
                candidate = Path(plan["original_binding"]["frozen_candidate"])
                seam = read(candidate / "old_domain_seam_plan.json")
                rig_loader_metadata(deps, native, native_path)
                for index in selected:
                    deps.index = index
                    row = read(candidate / "rig_camera_frames" / task["source"]["run"].upper() / f"frame_{index:03d}.json")
                    source = row["binding"]["source"]
                    key = f"{source['run']}/frame_{index:03d}"
                    producer_asset = seam["source_producer_frames"][key]
                    original = read(deps.check(producer_asset["path"], producer_asset["sha256"], "rig source producer receipt"))
                    for name in ("full_extended_scientific", "baseline_extended_scientific"):
                        deps.pairs(original[name], name)
                        asset = original[name]["binding"]["prepared_source"]
                        prepared = read(deps.check(asset["path"], asset["sha256"], name+" prepared H receipt"))
                        deps.pairs(prepared["full_native_scientific"], name+" native scientific")
                    task_id = source["previous_scientific_task_id"]
                    if task_id is not None:
                        previous_asset = seam.get("previous_extended_scientific_sources", {}).get(key,
                            seam["extended_scientific_sources"].get(task_id))
                        if previous_asset:
                            previous = read(deps.check(previous_asset["path"], previous_asset["sha256"], "exact previous extended receipt"))
                        else:
                            matches = [s for s in seam["source_inventory"] if s["scientific_task_id"] == task_id]
                            if len(matches) != 1:
                                raise ValueError("Rig predecessor cannot be uniquely inventoried")
                            src = matches[0]
                            asset = seam["source_producer_frames"][f"{src['run']}/frame_{src['frame_index']:03d}"]
                            previous = read(deps.check(asset["path"], asset["sha256"], "exact previous producer receipt"))["full_extended_scientific"]
                        deps.pairs(previous, "exact rig predecessor")
                        asset = previous["binding"]["prepared_source"]
                        deps.pairs(read(deps.check(asset["path"], asset["sha256"], "previous prepared H receipt"))["full_native_scientific"], "previous native scientific")
                    deps.pairs(row["rig_domain_revision"]["classes"], "rig modelled/fill support")
                kind = "rig exact extended scientific"
            summary.append(dict(task_id=task["task_id"], extractor=kind,
                                remaining_frames=len(selected), current_or_previous_frames=len(required), baseline_included=True))
    finally:
        photo._check = original_check
    local = [v for v in deps.assets.values() if v["canonical_C"]]
    report = dict(contract="ffac-t2-assigned-source-runtime-dependency-closure-v1", host=host,
        plan_sha256=file_sha(plan_path), frozen_photo_context_sha256=file_sha(photo.__file__),
        source_count=len(tasks), remaining_frames=sum(x["remaining_frames"] for x in summary),
        source_summaries=summary, canonical_C_assets=sorted(local,key=lambda x:x["path"]),
        canonical_C_asset_count=len(local), canonical_C_bytes=sum(x["bytes"] or 0 for x in local),
        missing_local_C_assets=[x["path"] for x in local if not x["exists"]],
        roles=dict(Counter(role for x in local for role in x["roles"])),
        metadata_json_live_SHA_verified=len(deps.json_verified),
        actual_frozen_photo_frame_metadata_validation=True, pixels_decoded=False,
        numerical_worker_or_snapshot_modified=False, canonical_C_staging_required=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_summaries","canonical_C_assets","missing_local_C_assets","roles")}),flush=True)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--host",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.plan,args.host,args.output)
