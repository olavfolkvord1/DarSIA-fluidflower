"""Materialize a new current43 photographic Track1 inference source batch.

This never trains, changes annotations, or overwrites an earlier source batch.
The frozen calibration selects the exact source version for each of 43 runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import traceback

from darsia.fluidflower.project import sha256_file, utc_now

FORMAT = "ffac-track1-display-mask-inference-source-batch-v1"


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def current_rows(photo_aggregate: Path, calibration: Path) -> list[dict]:
    rows = []
    for row in read(photo_aggregate)["rows"]:
        side = calibration.parent / f"{row['run']}_baseline.json"
        if side.is_file() and read(side)["source"].get(
                "source_manifest_sha256") == row["source_manifest_sha256"]:
            rows.append(row)
    if (len(rows) != 43 or len({row["run"] for row in rows}) != 43 or
            sum(len(row["frames"]) for row in rows) != 864):
        raise ValueError("Frozen calibration does not select exact current43/864")
    return sorted(rows, key=lambda row: int(row["run"][2:]))


def _source_cache_policy(settings: dict) -> dict:
    from darsia.fluidflower import phase_source_hash_cache
    from darsia.fluidflower.tools import materialize_ffac_v5_sources
    return dict(phase_source_hash_cache.policy_identity(),
        enabled=settings.get("cache_immutable_source_sha") is True,
        cache_helper_sha256=sha256_file(Path(phase_source_hash_cache.__file__)),
        source_materializer_sha256=sha256_file(Path(materialize_ffac_v5_sources.__file__)))


def _source_cache_allowlist(row: dict) -> dict[Path, str | None]:
    """Pin only this exact frozen row's inputs, never a directory-wide cache."""
    allowed: dict[Path, str | None] = {}
    def add(path_value, digest=None):
        if not path_value:
            return
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in allowed and digest is not None and allowed[path] not in (None, digest):
            raise ValueError(f"Contradictory exact-row source SHA: {path}")
        if path not in allowed or digest is not None:
            allowed[path] = digest
    source = Path(row["source_manifest_path"])
    if sha256_file(source) != row["source_manifest_sha256"]:
        raise ValueError("Frozen row source manifest changed before SHA cache scope")
    add(source, row["source_manifest_sha256"])
    original = read(source)
    for frame in row["frames"]:
        receipt = Path(frame["receipt_path"])
        if sha256_file(receipt) != frame["receipt_sha256"]:
            raise ValueError("Frozen transport receipt changed before SHA cache scope")
        add(receipt, frame["receipt_sha256"])
        value = read(receipt)
        # These receipt inputs already have their exact frozen SHA. The
        # processor still validates every receipt's source/H relationships.
        for key, path in value.items():
            if key.endswith("_path"):
                digest = value.get(key[:-5] + "_sha256")
                if isinstance(path, str) and isinstance(digest, str) and len(digest) == 64:
                    add(path, digest)
        camera = value.get("camera_jpeg_path")
        if camera:
            paired_raw = Path(camera).with_suffix(".ARW")
            if paired_raw.is_file():
                # This original same-stem acquisition has no transport SHA;
                # first actual read pins it, existing processor checks bind it
                # in the output certificate, and the exit seal reads it again.
                add(paired_raw)
    # AC61's prepared frame0 has three explicit acquisition members.
    for frame in original.get("frame_sources") or []:
        for member in frame.get("configured_baseline_members") or []:
            add(member["path"], member.get("sha256"))
            paired_raw = Path(member["path"]).with_suffix(".ARW")
            if paired_raw.is_file():
                add(paired_raw)
    return allowed


def worker(row: dict, settings: dict, indices: set[int] | None = None) -> dict:
    if settings.get("cache_immutable_source_sha") is not True:
        return _worker(row, settings, indices)
    from darsia.fluidflower.phase_source_hash_cache import immutable_source_hash_cache
    allowed = _source_cache_allowlist(row)
    with immutable_source_hash_cache(allowed,
            excluded_roots=[Path(settings["output_root"])]) as cache:
        result = _worker(row, settings, indices)
    # A changed input fails the context exit before this revision is accepted
    # by the parent batch. No aggregate or worker completion can precede seal.
    result["source_verification_cache"] = cache.receipt()
    return result


def _worker(row: dict, settings: dict, indices: set[int] | None = None) -> dict:
    import cv2
    cv2.setNumThreads(1)
    # AC44's transport receipt binds an archived implementation byte-for-byte.
    # Import that exact real file, rather than changing its geometry witness.
    if row["run"] == "ac44" and settings.get("ac44_geometry_module_path"):
        import importlib.util
        import sys
        path = Path(settings["ac44_geometry_module_path"])
        expected = settings["ac44_geometry_module_sha256"]
        if sha256_file(path) != expected:
            raise ValueError("Archived AC44 geometry module changed")
        name = "darsia.fluidflower.tools.ffac_raw_geometry"
        previous = sys.modules.get(name)
        if previous is not None and sha256_file(Path(previous.__file__)) != expected:
            raise ValueError("Another AC44 geometry implementation was already imported")
        if previous is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    from darsia.fluidflower.tools.ffac_track1_v4r5_source_processor import (
        Track1V4R5SourceProcessor,
    )
    from darsia.fluidflower.tools.ffac_track1_v4r5_ac44_alt_processor import (
        Track1V4R5AC44AltProcessor,
    )
    from darsia.fluidflower.tools.materialize_ffac_v5_sources import materialize_revision
    source = Path(row["source_manifest_path"])
    cls = Track1V4R5AC44AltProcessor if row["run"] == "ac44" else Track1V4R5SourceProcessor
    processor = cls(
        row, calibration_manifest_path=Path(settings["calibration"]),
        photo_aggregate_path=Path(settings["photo_aggregate"]),
        enclosed_hole_policy="fill_enclosed_v1",
        display_mode=settings.get("display_mode", "photographic_v1"),
        **({"isolated_invalid_policy": settings["isolated_invalid_policy"]}
           if settings.get("isolated_invalid_policy") else {}))
    revision = dict(run=row["run"], manifest_path=str(source),
                    manifest_sha256=row["source_manifest_sha256"],
                    review_root=str(source.parents[2]))
    receipt = materialize_revision(
        revision, output_root=Path(settings["output_root"]), processor=processor,
        reference_state_path=Path(settings["reference_state"]),
        reference_binding_path=Path(settings["reference_binding"]),
        frame_geometry_row=row, frame_indices=indices,
        source_only_preparation=indices is None)
    if receipt.get("frame_count") != (len(indices) if indices else len(row["frames"])):
        raise ValueError("Materialized frame count differs from frozen source")
    root = Path(settings["output_root"]) / "revisions" / row["source_manifest_sha256"]
    if indices is not None:
        return dict(run=row["run"], pilot_only=True, frame_count=len(indices))
    return dict(
        run=row["run"], source_manifest_sha256=row["source_manifest_sha256"],
        source_manifest_path=str(source), frame_count=receipt["frame_count"],
        receipt_path=str(root / "materialization_receipt.json"),
        receipt_sha256=sha256_file(root / "materialization_receipt.json"),
        materialized_review_root=receipt["materialized_review_root"],
        materialized_manifest_path=receipt["materialized_manifest_path"],
        materialized_manifest_sha256=receipt["materialized_manifest_sha256"],
    )


def materialize(settings: dict, *, workers: int, smoke_run: str | None = None,
                smoke_indices: set[int] | None = None) -> dict:
    from darsia.fluidflower.tools.ffac_track1_v4r5_source_processor import Track1V4R5SourceProcessor
    from darsia.fluidflower.tools.ffac_track1_photographic_display import display_contract
    from darsia.fluidflower.tools.materialize_ffac_v5_sources import _write_new
    rows = current_rows(Path(settings["photo_aggregate"]), Path(settings["calibration"]))
    if smoke_run:
        rows = [row for row in rows if row["run"] == smoke_run]
        if len(rows) != 1:
            raise ValueError("Smoke run is not a unique frozen source")
    elif settings.get("run_priority"):
        priority = settings["run_priority"]
        if (len(priority) != len(set(priority)) or
                not set(priority) <= {row["run"] for row in rows}):
            raise ValueError("Source priority is not a distinct frozen-cohort subset")
        rank = {run: index for index, run in enumerate(priority)}
        rows.sort(key=lambda row: (rank.get(row["run"], len(rank)), int(row["run"][2:])))
    root = Path(settings["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    identity = Track1V4R5SourceProcessor(
        rows[0], calibration_manifest_path=Path(settings["calibration"]),
        photo_aggregate_path=Path(settings["photo_aggregate"]),
        enclosed_hole_policy="fill_enclosed_v1",
        display_mode=settings.get("display_mode", "photographic_v1"),
        **({"isolated_invalid_policy": settings["isolated_invalid_policy"]}
           if settings.get("isolated_invalid_policy") else {})).identity()
    job = dict(format=FORMAT, settings=settings, processor=identity,
               source_versions=[dict(run=row["run"], sha256=row["source_manifest_sha256"])
                                for row in current_rows(Path(settings["photo_aggregate"]),
                                                        Path(settings["calibration"]))],
               inference_only=True, retraining=False, labels_changed=False,
               pooled_color_path_sampling_policy="keep",
               source_verification_cache_policy=_source_cache_policy(settings))
    job_path = root / "inference_source_job.json"
    if job_path.exists():
        if read(job_path) != job:
            raise ValueError("Destination already belongs to a different immutable job")
    else:
        _write_new(job_path, job)
    print(json.dumps(dict(event="launch", utc=utc_now(), workers=workers,
                          runs=len(rows), frames=sum(len(row["frames"]) for row in rows),
                          output_root=str(root))), flush=True)
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(worker, row, settings,
                               smoke_indices if smoke_run else None): row for row in rows}
        for future in as_completed(pending):
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(dict(event="run_complete", utc=utc_now(), **result)), flush=True)
            except Exception as error:
                result = dict(event="failed", run=pending[future]["run"],
                              error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc())
                errors.append(result)
                print(json.dumps(result), flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} exact source revisions failed; no maps started")
    if smoke_run:
        return dict(pilot_only=True, revisions=results)
    results.sort(key=lambda row: int(row["run"][2:]))
    receipt = dict(
        format=FORMAT, created_utc=utc_now(), output_root=str(root),
        source_revisions=43, timeline_frames=864, revisions=results,
        canonical_mapping=results,
        processor=identity, photographic_display=identity.get("display_transfer", display_contract()),
        source_job_sha256=sha256_file(job_path),
        photometry_manifest_sha256=sha256_file(Path(settings["calibration"])),
        photo_aggregate_sha256=sha256_file(Path(settings["photo_aggregate"])),
        inference_only=True, retraining=False, labels_changed=False,
        scientific_acceptance=False, mass_eligible=False,
        source_verification_cache_policy=_source_cache_policy(settings),
        pooled_color_path_sampling_policy="keep")
    target = root / "materialization_receipt.json"
    if target.exists():
        previous = read(target)
        if {key: val for key, val in previous.items() if key != "created_utc"} != {
                key: val for key, val in receipt.items() if key != "created_utc"}:
            raise ValueError("Completed batch receipt differs from exact inference sources")
        receipt = previous
    else:
        _write_new(target, receipt)
    print(json.dumps(dict(event="complete", utc=utc_now(), receipt_path=str(target),
                          receipt_sha256=sha256_file(target), frames=864)), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("photo-aggregate", "calibration", "reference-state", "reference-binding", "output-root"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke-run")
    parser.add_argument("--smoke-indices", default="0,14")
    parser.add_argument("--cache-immutable-source-sha", action="store_true",
        help="Reuse exact input SHAs inside each worker; actual SHA seal before acceptance")
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise ValueError("CPU source worker count must be between1 and24")
    settings = {key: str(getattr(args, key)) for key in (
        "photo_aggregate", "calibration", "reference_state", "reference_binding", "output_root")}
    settings["cache_immutable_source_sha"] = args.cache_immutable_source_sha
    materialize(settings, workers=args.workers, smoke_run=args.smoke_run,
                smoke_indices={int(value) for value in args.smoke_indices.split(",")})


if __name__ == "__main__":
    main()
