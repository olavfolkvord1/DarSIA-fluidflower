"""Camera-clock FF_AC selection and immutable source-bound label bookkeeping.

This module reads capture metadata and sources, never corrects images, runs a
phase model, or edits an earlier timeline or annotation. The nominal grid and
nearest-image tie rule match the existing sparse FF_AC preparation.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import tomllib

from PIL import Image

from .storage_paths import resolve_storage_path

FORMAT = "ffac-camera-clock-timeline-v1"
LABEL_FORMAT = "ffac-camera-clock-source-bound-label-audit-v1"
NOMINAL_TIMES_H = (.25, .5, .75, 1., 1.25, 1.75, 2., 2.25, 2.75, 3., 4.,
                   5., 6., 8., 10., 12., 24., 48., 72., 96.)
RUNS = tuple(f"ac{i}" for i in (*range(17, 36), *range(38, 62)))


def canonical_path(value):
    """Resolve configured storage aliases without reading the obsolete source."""
    return resolve_storage_path(value)


def file_sha256(path):
    h = hashlib.sha256()
    with canonical_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(canonical_path(path).read_text(encoding="utf-8-sig"))


def _new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_capture(value, *, subseconds=None, offset=None):
    if isinstance(value, bytes):
        value = value.decode("ascii")
    text = str(value).strip().strip("\0")
    try:
        stamp = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        stamp = datetime.fromisoformat(text)
    if subseconds not in (None, ""):
        digits = str(subseconds).strip().strip("\0")
        if not digits.isdigit():
            raise ValueError("Invalid camera SubSecTimeOriginal")
        stamp = stamp.replace(microsecond=int((digits + "000000")[:6]))
    if offset not in (None, ""):
        stamp = datetime.fromisoformat(stamp.isoformat() + str(offset).strip())
    return stamp


def read_capture_time(path):
    """Read actual EXIF DateTimeOriginal, or capture metadata in the paired RAW.

    File/folder names and filesystem timestamps never supply camera time.
    DateTime on a processed JPEG is deliberately not accepted as capture time.
    """
    path = canonical_path(path)
    if path.suffix.casefold() in {".jpg", ".jpeg"}:
        with Image.open(path) as image:
            exif = image.getexif()
            capture = exif.get_ifd(0x8769)
            original = capture.get(0x9003) or exif.get(0x9003)
            if original:
                stamp = _parse_capture(original, subseconds=capture.get(0x9291),
                                       offset=capture.get(0x9011))
                return dict(camera_datetime=stamp.isoformat(),
                    timestamp_basis="EXIF.DateTimeOriginal", timestamp_source_path=str(path))
        raw = paired_raw(path)
        if raw is None:
            raise ValueError(f"No original camera capture time: {path}")
        return read_capture_time(raw)
    if path.suffix.casefold() in {".arw", ".tif", ".tiff"}:
        import tifffile
        with tifffile.TiffFile(path) as tiff:
            tags = tiff.pages[0].tags
            capture = tags[34665].value if 34665 in tags else {}
            original = capture.get("DateTimeOriginal")
            basis = "RAW.DateTimeOriginal"
            if not original and 306 in tags:
                original, basis = tags[306].value, "RAW.DateTime"
            if original:
                stamp = _parse_capture(original, subseconds=capture.get("SubsecTimeOriginal"),
                    offset=capture.get("OffsetTimeOriginal"))
                return dict(camera_datetime=stamp.isoformat(), timestamp_basis=basis,
                            timestamp_source_path=str(path))
    raise ValueError(f"No original camera capture time: {path}")


def paired_raw(path):
    path = canonical_path(path)
    for suffix in (".ARW", ".arw"):
        raw = path.with_suffix(suffix)
        if raw.is_file():
            return raw
    return None


def camera_anchor(reference, capture):
    elapsed = float(reference["elapsed_h"])
    if not math.isfinite(elapsed) or elapsed >= 0:
        raise ValueError("Camera anchor must be the own pre-injection reference")
    stamp = datetime.fromisoformat(capture["camera_datetime"])
    return dict(reference_source_path=reference["source_path"],
        reference_source_sha256=reference["source_sha256"], reference_elapsed_h=elapsed,
        reference_camera_datetime=stamp.isoformat(), timestamp_basis=capture["timestamp_basis"],
        timestamp_source_path=capture["timestamp_source_path"],
        origin_camera_datetime=(stamp - timedelta(hours=elapsed)).isoformat(),
        equation="t0 = camera(reference) - elapsed(reference)")


def camera_elapsed_hours(camera_datetime, anchor):
    return _camera_elapsed_seconds(camera_datetime, anchor) / 3600


def _camera_elapsed_seconds(camera_datetime, anchor):
    stamp = datetime.fromisoformat(camera_datetime)
    origin = datetime.fromisoformat(anchor["origin_camera_datetime"])
    if (stamp.tzinfo is None) != (origin.tzinfo is None):
        raise ValueError("Mixed offset-aware and local camera timestamps")
    return (stamp - origin).total_seconds()


def nearest_camera_selection(captures, anchor, target_hours=NOMINAL_TIMES_H):
    """Deduplicated nearest acquisition with the existing source-order tie rule."""
    rows = []
    seen = set()
    for index, capture in enumerate(captures):
        path = str(canonical_path(capture["source_path"]))
        if path.casefold() in seen:
            raise ValueError("Duplicate acquisition path in camera inventory")
        seen.add(path.casefold())
        seconds = _camera_elapsed_seconds(capture["camera_datetime"], anchor)
        if seconds > 0:
            rows.append((seconds, int(capture.get("source_index", index)), dict(capture)))
    rows.sort(key=lambda row: (row[0], row[1]))
    if not rows:
        raise ValueError("No positive-time camera acquisitions")
    seconds = [row[0] for row in rows]
    selected = {}
    for value in target_hours:
        target = float(value)
        if not math.isfinite(target) or target < 0:
            raise ValueError("Nominal camera target must be finite and nonnegative")
        position = bisect_left(seconds, target * 3600)
        if position <= 0:
            chosen = 0
        elif position >= len(rows):
            chosen = len(rows) - 1
        else:
            left, right = position - 1, position
            dl, dr = abs(seconds[left] - target * 3600), abs(seconds[right] - target * 3600)
            chosen = left if dl < dr or (dl == dr and rows[left][1] <= rows[right][1]) else right
        capture = rows[chosen][2]
        key = str(canonical_path(capture["source_path"])).casefold()
        if key not in selected:
            selected[key] = dict(capture, elapsed_h=rows[chosen][0] / 3600,
                selection_status="nearest_available_acquisition", nominal_target_hours=[])
        selected[key]["nominal_target_hours"].append(target)
    return sorted(selected.values(), key=lambda row: (row["elapsed_h"], row["source_index"]))


def source_temporal_predecessors(frames, independent_reference_frames=()):
    """Bind old temporal inputs by source, rather than a later timeline index."""
    independent = set(independent_reference_frames)
    result, previous = {}, None
    for index, frame in enumerate(frames):
        if frame["frame_index"] != index:
            raise ValueError("Non-contiguous source timeline")
        result[frame["source_sha256"]] = deepcopy(previous)
        if index not in independent:
            previous = dict(frame_index=index, source_path=frame["source_path"],
                source_sha256=frame["source_sha256"],
                v5_raw_path=frame.get("v5_raw_path"), v5_raw_sha256=frame.get("v5_raw_sha256"),
                v5_log_linear_rgb_path=frame.get("v5_log_linear_rgb_path"),
                v5_log_linear_rgb_sha256=frame.get("v5_log_linear_rgb_sha256"))
    return result


def selection_diff(old_frames, new_frames):
    rows = []
    for index in range(max(len(old_frames), len(new_frames))):
        old = old_frames[index] if index < len(old_frames) else None
        new = new_frames[index] if index < len(new_frames) else None
        rows.append(dict(frame_index=index,
            old_source_path=old["source_path"] if old else None,
            old_source_sha256=old["source_sha256"] if old else None,
            old_elapsed_h=float(old["elapsed_h"]) if old else None,
            new_source_path=new["source_path"] if new else None,
            new_source_sha256=new["source_sha256"] if new else None,
            new_elapsed_h=float(new["elapsed_h"]) if new else None,
            source_changed=old is None or new is None or old["source_sha256"] != new["source_sha256"]))
    return rows


def audit_label_sources(labels, old_frames, new_frames):
    """Keep every label's source SHA even when that source leaves the timeline."""
    new_by_sha = {}
    for row in new_frames:
        new_by_sha.setdefault(row["source_sha256"], []).append(row["frame_index"])
    rows, counts, seen = [], {}, set()
    for label in labels:
        label_id = label["annotation_id"]
        if label_id in seen:
            raise ValueError("Duplicate source-bound annotation ID")
        seen.add(label_id)
        index = int(label["frame_index"])
        if index < 0 or index >= len(old_frames):
            raise ValueError("Label has no original source frame")
        source = old_frames[index]
        if (abs(float(label["elapsed_h"]) - float(source["elapsed_h"])) > 1e-5
            or label.get("source_sha256", source["source_sha256"]) != source["source_sha256"]):
            raise ValueError("Label lost its original image/time binding")
        sha = source["source_sha256"]
        new_indices = new_by_sha.get(sha, [])
        row = dict(annotation_id=label_id, old_frame_index=index,
            source_path=source["source_path"], source_sha256=sha,
            before_source_sha256=sha, after_source_sha256=sha,
            old_elapsed_h=float(source["elapsed_h"]), new_timeline_frame_indices=new_indices,
            source_left_timeline=not new_indices, label_payload_sha256=_digest(label),
            label_moved_to_another_image=False)
        rows.append(row)
        counts.setdefault(sha, dict(source_path=source["source_path"], source_sha256=sha,
            old_frame_index=index, new_timeline_frame_indices=new_indices,
            label_count_before=0, label_count_after=0, source_left_timeline=not new_indices))
        counts[sha]["label_count_before"] += 1
        counts[sha]["label_count_after"] += 1
    return dict(contract=LABEL_FORMAT, labels_unchanged=True,
        labels=len(rows), rows=rows, counts_per_source_image=list(counts.values()),
        labels_whose_source_left_timeline=sum(row["source_left_timeline"] for row in rows),
        labels_reindexed_or_rewritten=False)


def configured_camera_paths(config_path):
    """Use the current acquisition pool and blacklist, but never protocol times."""
    from natsort import natsorted, ns
    import darsia
    config = tomllib.loads(canonical_path(config_path).read_text(encoding="utf-8-sig"))
    data, protocols = config["data"], config["protocols"]
    folders = [canonical_path(path) for path in data["folders"]]
    suffix = Path(data["baseline"]).suffix
    pool = natsorted(set(path for folder in folders for path in folder.iterdir()
        if path.is_file() and path.suffix == suffix), alg=ns.IGNORECASE)
    imaging = protocols["imaging"]
    blacklists = protocols.get("blacklist") or {}
    by_folder = {canonical_path(folder): darsia.ImagingProtocol(canonical_path(protocol),
        canonical_path(blacklists[folder]) if isinstance(blacklists, dict) and folder in blacklists else None)
        for folder, protocol in imaging.items()}
    available, excluded = [], []
    for index, path in enumerate(pool):
        protocol = by_folder[path.parent]
        try:
            invalid = protocol.is_blacklisted(path) or protocol.get_datetime(path) is None
        except ValueError:
            invalid = True
        if invalid:
            excluded.append(str(path))
        else:
            available.append((index, path))
    return available, dict(config_path=str(config_path), config_sha256=file_sha256(config_path),
        folders=[str(folder) for folder in folders], blacklisted_or_unavailable_paths=excluded,
        folder_clock_used_for_selection=False)


def _capture_row(item):
    index, path = item
    before = path.stat()
    row = dict(source_path=str(path), source_name=path.name, source_index=index,
               **read_capture_time(path))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Acquisition changed during camera metadata read")
    row.update(source_bytes=after.st_size, source_mtime_ns=after.st_mtime_ns)
    return row


def plan_camera_run(run, native_path, config_path, *, independent_reference_frames=(),
                    metadata_workers=2):
    native_path = canonical_path(native_path)
    native_sha, native = file_sha256(native_path), _read(native_path)
    if native.get("dataset") != "ffac" or native.get("run") != run:
        raise ValueError("Camera source belongs to another run")
    frames, reference = native["frame_sources"], native["reference"]
    if len(frames) != native["frame_count"]:
        raise ValueError("Incomplete old source timeline")
    anchor = camera_anchor(reference, read_capture_time(reference["source_path"]))
    if file_sha256(reference["source_path"]) != reference["source_sha256"]:
        raise ValueError("Camera reference source SHA changed")
    pool, pool_binding = configured_camera_paths(config_path)
    with ThreadPoolExecutor(max_workers=metadata_workers) as executor:
        captures = list(executor.map(_capture_row, pool))
    camera_by_path = {str(canonical_path(row["source_path"])).casefold(): row for row in captures}
    old_by_sha = {row["source_sha256"]: row for row in frames}
    predecessors = source_temporal_predecessors(frames, independent_reference_frames)
    baseline = deepcopy(frames[0])
    baseline_capture = read_capture_time(baseline["source_path"])
    baseline.update(baseline_capture,
        elapsed_h=camera_elapsed_hours(baseline_capture["camera_datetime"], anchor),
        nominal_target_hours=[], retained_old_frame_index=0,
        frozen_temporal_predecessor=predecessors[baseline["source_sha256"]])
    if baseline["selection_status"] != "pre_injection_negative_control" or baseline["elapsed_h"] >= 0:
        raise ValueError("The negative-control source must remain exact frame0")
    selected = nearest_camera_selection(captures, anchor)
    new_frames = [baseline]
    for selected_row in selected:
        selected_row = dict(selected_row)
        sha = file_sha256(selected_row["source_path"])
        if read_capture_time(selected_row["source_path"])["camera_datetime"] != selected_row["camera_datetime"]:
            raise ValueError("Selected camera timestamp changed during SHA binding")
        old = old_by_sha.get(sha)
        if old:
            row = deepcopy(old)
            row.update(selected_row, retained_old_frame_index=old["frame_index"],
                frozen_temporal_predecessor=predecessors[sha],
                temporal_input_policy="retain-parent-predecessor-by-source-sha")
        else:
            raw = paired_raw(selected_row["source_path"])
            if raw is None:
                raise ValueError("New acquisition needs its own paired RAW")
            row = dict(selected_row, retained_old_frame_index=None,
                v5_raw_path=str(raw), v5_raw_sha256=file_sha256(raw),
                temporal_input_policy="new-source-explicit-predecessor-binding-required")
        row.update(source_sha256=sha, frame_index=len(new_frames))
        new_frames.append(row)
    old_camera_rows = []
    for row in frames:
        if file_sha256(row["source_path"]) != row["source_sha256"]:
            raise ValueError("Old selected camera image SHA changed")
        capture = camera_by_path.get(str(canonical_path(row["source_path"])).casefold())
        capture = capture or read_capture_time(row["source_path"])
        elapsed = camera_elapsed_hours(capture["camera_datetime"], anchor)
        old_camera_rows.append(dict(frame_index=row["frame_index"], source_name=row["source_name"],
            source_sha256=row["source_sha256"], timeline_h=float(row["elapsed_h"]),
            camera_h=elapsed, timeline_minus_camera_h=float(row["elapsed_h"])-elapsed))
    # No mutation or inference is hidden in these new metadata records.
    return dict(contract=FORMAT, run=run, native_manifest_path=str(native_path),
        native_manifest_sha256=native_sha, acquisition_pool=pool_binding,
        anchor=anchor, nominal_times_h=list(NOMINAL_TIMES_H),
        acquisition_count=len(captures), camera_inventory_sha256=_digest(captures),
        camera_inventory=captures, old_frame_count=len(frames), frame_count=len(new_frames),
        old_timeline_camera_audit=old_camera_rows, frames=new_frames,
        selection_diff=selection_diff(frames, new_frames),
        max_new_timeline_minus_camera_h=max(abs(row["elapsed_h"]-
            camera_elapsed_hours(row["camera_datetime"], anchor)) for row in new_frames),
        parent_processor=deepcopy(native["correction_contract"]["processor"]),
        labels_read=False, model_inference=False, raw_materialization=False)


def plan_camera_timeline(parent_candidate, output, spec_path, label_snapshot, *, workers=8,
                         resume=False):
    """Create read-only selection/label receipts for the exact current43 cohort."""
    from darsia.fluidflower.series_paths import series_path
    parent, output = canonical_path(parent_candidate), Path(output)
    post_path = parent / "validation/postflight_manifest.json"
    post = _read(post_path)
    jobs = {row["run"]: row for row in post["jobs"]}
    if set(jobs) != set(RUNS) or len(post["jobs"]) != 43:
        raise ValueError("Camera selection needs the exact43 parent cohort")
    binding = dict(contract=FORMAT, parent_candidate=str(parent),
        parent_postflight_sha256=file_sha256(post_path), spec_path=str(spec_path),
        spec_sha256=file_sha256(spec_path), code_sha256=file_sha256(__file__),
        nominal_times_h=list(NOMINAL_TIMES_H), camera_capture_time_only=True,
        labels_unchanged=True, cpu_only=True, training=False)
    plan_path = output / "camera_timeline_plan.json"
    if plan_path.exists():
        previous = _read(plan_path)
        if not resume or previous["binding"] != binding:
            raise ValueError("Use a new camera audit namespace")
        for row in previous["runs"]:
            if file_sha256(row["receipt_path"]) != row["receipt_sha256"]:
                raise ValueError("Camera run receipt changed")
        return previous
    output.mkdir(parents=True, exist_ok=True)
    results = []
    def run_plan(run):
        job = jobs[run]
        native = canonical_path(job["review_output_dir"]) / "native_frames" / run / "manifest.json"
        config = series_path("ffac", "run_configs") / f"{run}.toml"
        receipt = output / "runs" / f"{run}.json"
        if receipt.exists():
            result = _read(receipt)
            if (not resume or result.get("binding") != binding
                or result["native_manifest_sha256"] != file_sha256(native)
                or result["acquisition_pool"]["config_sha256"] != file_sha256(config)):
                raise ValueError("Camera run resume has different source/code/spec")
            for row in result["frames"]:
                if file_sha256(row["source_path"]) != row["source_sha256"]:
                    raise ValueError("Resumed selected source changed")
        else:
            result = plan_camera_run(run, native, config,
                independent_reference_frames=job.get("independent_reference_frames") or ())
            result["binding"] = binding
            _new_json(receipt, result)
        return result, receipt
    # The three documented clock defects are made available first for the new
    # materializer; selection still verifies every acquisition in every run.
    ordered = sorted(RUNS, key=lambda run: (run not in {"ac31", "ac23", "ac44"}, int(run[2:])))
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 12))) as executor:
        futures = {executor.submit(run_plan, run): run for run in ordered}
        for future in as_completed(futures):
            result, receipt = future.result()
            results.append((result, receipt))
            print(f"Camera selection complete: {result['run']} ({len(results)}/43)", flush=True)
    results.sort(key=lambda pair: int(pair[0]["run"][2:]))
    if sum(result["frame_count"] for result, _ in results) != 864:
        raise ValueError("Camera reselection changes total frame count; inspect per-run receipts")
    rows = [dict(run=result["run"], **row) for result, _ in results for row in result["selection_diff"]]
    if len(rows) != 864:
        raise ValueError("Camera frame index coverage differs from exact864")
    label_audit = sealed_round1_ac31_audit(label_snapshot, results)
    label_path = output / "ac31_round1_source_label_audit.json"
    _new_json(label_path, label_audit)
    csv_path = output / "selection_diff.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plan = dict(binding=binding, run_count=43, frame_count=864,
        runs=[dict(run=result["run"], receipt_path=str(receipt), receipt_sha256=file_sha256(receipt),
            frames=result["frame_count"], changed_sources=sum(row["source_changed"] for row in result["selection_diff"]))
            for result, receipt in results], changed_frames=[row for row in rows if row["source_changed"]],
        changed_source_count=sum(row["source_changed"] for row in rows),
        unchanged_source_shas_confirmed=all(row["old_source_sha256"] == row["new_source_sha256"]
            for row in rows if not row["source_changed"]),
        max_timeline_minus_camera_h=max(result["max_new_timeline_minus_camera_h"] for result, _ in results),
        label_audit=dict(path=str(label_path), sha256=file_sha256(label_path)),
        selection_diff_csv=dict(path=str(csv_path), sha256=file_sha256(csv_path)))
    _new_json(plan_path, plan)
    return plan


def sealed_round1_ac31_audit(snapshot, results):
    """Open only the permitted sealed round-1 CSV and original source metadata."""
    snapshot = Path(snapshot)
    seal = _read(snapshot / "SNAPSHOT_MANIFEST.json")
    for filename in ("pairwise_review_annotations.csv", "review_session.json"):
        path = snapshot / filename
        entry = seal["files"][filename]
        if file_sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Sealed round-1 source/label file changed")
    session = _read(snapshot / "review_session.json")
    post_path = canonical_path(session["combined_root"]) / "postflight_manifest.json"
    post = _read(post_path)
    job = next(row for row in post["jobs"] if row["run"] == "ac31")
    native_path = canonical_path(job["analysis_source_review_root"]) / "native_frames/ac31/manifest.json"
    native = _read(native_path)
    if file_sha256(native_path) != job["frame_geometry_manifest_sha256"]:
        raise ValueError("Round-1 label's original native source changed")
    with (snapshot / "pairwise_review_annotations.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = [row for row in csv.DictReader(stream) if row["run"].casefold() == "ac31"]
    for row in labels:
        if (row["candidate_id"] != session["selected_candidates"]["ac31"]
            or row["candidate_id"] != job["candidate_id"] or row["params_sha256"] != job["params_sha256"]):
            raise ValueError("Round-1 AC31 label candidate/source binding changed")
    result = next(result for result, _ in results if result["run"] == "ac31")
    old = _read(result["native_manifest_path"])["frame_sources"]
    if [row["source_sha256"] for row in old] != [row["source_sha256"] for row in native["frame_sources"]]:
        raise ValueError("Camera parent differs from round-1 label image sources")
    audit = audit_label_sources(labels, native["frame_sources"], result["frames"])
    audit.update(run="ac31", snapshot_path=str(snapshot),
        annotation_csv_sha256=seal["files"]["pairwise_review_annotations.csv"]["sha256"],
        original_native_manifest=dict(path=str(native_path), sha256=file_sha256(native_path)),
        source_image_sha_not_frame_index_is_label_identity=True)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--round1-snapshot", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_camera_timeline(args.parent, args.output, args.spec, args.round1_snapshot,
                                workers=args.workers, resume=args.resume)
    print(f"Camera timeline audit ready: {plan['run_count']} runs, {plan['frame_count']} frames", flush=True)


if __name__ == "__main__":
    main()
