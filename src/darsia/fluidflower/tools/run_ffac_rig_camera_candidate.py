"""Immutable CPU producer for the combined camera-clock/rig-domain revision.

Planning and source radiometry are separate from accepted-geometry inference.
The preparation stage writes source-specific float assets, never phase labels,
and may run before the rigid-wall certificate exists. No command activates or
registers a model. Each completed source is resumed only after SHA verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import ctypes
import io
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from darsia.fluidflower import phase_ffac_rig_camera_metadata as metadata
from darsia.fluidflower.phase_ffac_rig_camera_runtime import SPEC_SHA256, bound_path
from darsia.fluidflower.phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
from .refit_ffac_g3_downstream import (
    JobLease, ProgressLog, atomic_json, below_normal_priority, canonical_sha256,
    preflight_space, read_json, sha256_file, utc_now,
)
from .build_ffac_g3_approved_candidate import _write_bytes

CONTRACT = "ffac-rig-camera-candidate-producer-v1"
SOURCE_RECEIPT_CONTRACT = "ffac-rig-camera-prepared-source-v1"
PARENT_GEOMETRY_CONTRACT = "ffac-rig-camera-preserved-source-geometry-v1"
PLAN_FILE = "rig_camera_candidate_plan.json"
GIB = 1024 ** 3
HEAD_FILES = {
    "models/generic/co2_presence_signal.json": CO2_SHA256,
    "models/g3/gas_head.json": G3_SHA256,
    "regional_model.json": REFIT_SHA256,
    "models/generic/gas_within_co2_signal.json":
        "74e7d6e166b04d158eb04f09cdc310ec38dbbd2b148bcd00a25aedb8962020e3",
}


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path))


def available_memory_bytes():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_physical", "available_physical", "total_page", "available_page",
                    "total_virtual", "available_virtual", "extended_virtual")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return int(status.available_physical)
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def admitted_workers(requested=24, *, cpu_count=None, available_bytes=None,
                     reserve_bytes=32 * GIB, bytes_per_worker=6 * GIB):
    """Admit up to 24 single-thread CPU workers, leaving UI/OS capacity."""
    if type(requested) is not int or not 1 <= requested <= 24:
        raise ValueError("The rig producer accepts 1–24 CPU workers")
    cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    free = available_memory_bytes() if available_bytes is None else int(available_bytes)
    # A laptop cannot reserve 32 GiB; retain at least one quarter of its free
    # memory, while the workstation keeps the full requested OS/UI reserve.
    reserve = min(int(reserve_bytes), max(2 * GIB, free // 4))
    memory_workers = (free - reserve) // int(bytes_per_worker)
    if memory_workers < 1:
        raise MemoryError("Insufficient free memory for one bounded native RAW worker")
    return min(requested, max(1, cpus - 2), int(memory_workers))


def worker_initializer():
    global _THREAD_LIMITER
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    below_normal_priority()
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    _THREAD_LIMITER = threadpool_limits(limits=1)


def _code_binding():
    from darsia.fluidflower import phase_ffac_rig_raw_source as raw
    return dict(producer_sha256=sha256_file(__file__),
                raw_processor_sha256=sha256_file(raw.__file__),
                metadata_sha256=sha256_file(metadata.__file__))


def _source_key(run, sha, reference=False):
    return run + "/" + ("reference-" if reference else "source-") + sha


def inventory_sources(camera_runs, changed_geometry):
    """Inventory selected images, real B and departed temporal dependencies."""
    geometry = {(row["run"], row["frame_index"]): row for row in changed_geometry}
    expected_changed = set()
    tasks, frames, references = {}, [], []
    for run_asset in camera_runs:
        camera = read_json(bound_path(run_asset["camera_run"]))
        native_asset = run_asset["parent_native_manifest"]
        native = read_json(bound_path(native_asset))
        run = camera["run"]
        if native.get("run") != run or camera["native_manifest_sha256"] != native_asset["sha256"]:
            raise ValueError("Camera timeline differs from its exact parent source")
        if camera["frame_count"] != len(camera["frames"]) or camera["max_new_timeline_minus_camera_h"] > .05:
            raise ValueError("Camera timeline is incomplete or exceeds its 0.05 h tolerance")
        old_by_sha = {row["source_sha256"]: row for row in native["frame_sources"]}

        def add_source(row, *, old_index=None, reference=False, geometry_row=None):
            key = _source_key(run, row["source_sha256"], reference)
            raw_path = row.get("v5_raw_path") if not reference else row["v5_reference_state"]["camera_arw"]
            raw_sha = row.get("v5_raw_sha256") if not reference else row["v5_reference_state"]["camera_arw_sha256"]
            task = dict(task_id=key, run=run, source=dict(path=row["source_path"], sha256=row["source_sha256"]),
                raw=dict(path=raw_path, sha256=raw_sha),
                parent_native_manifest=deepcopy(native_asset), parent_frame_index=old_index,
                parent_reference=reference, own_geometry=deepcopy(geometry_row))
            if key in tasks and tasks[key] != task:
                raise ValueError("One source SHA has inconsistent parent/RAW geometry bindings")
            tasks[key] = task
            return key

        ref = native["reference"]
        references.append(dict(run=run, scientific_task_id=add_source(ref, reference=True)))
        for index, selected in enumerate(camera["frames"]):
            if selected["frame_index"] != index:
                raise ValueError("Selected camera positions are not contiguous")
            old_index = selected.get("retained_old_frame_index")
            own = None
            if old_index is None:
                expected_changed.add((run, index))
                own = geometry.get((run, index))
                if (own is None or own["camera_jpeg_sha256"] != selected["source_sha256"]
                        or own["camera_arw_sha256"] != selected["v5_raw_sha256"]
                        or own.get("new_geometry_measured_from_own_source") is not True):
                    raise ValueError("New selected camera source lacks its own geometry certificate")
                bound_path(dict(path=own["geometry_certificate_path"], sha256=own["geometry_certificate_sha256"]))
                row = selected
            else:
                row = native["frame_sources"][old_index]
                if row["source_sha256"] != selected["source_sha256"]:
                    raise ValueError("Retained camera source was assigned another parent's index")
            current = add_source(row, old_index=old_index, geometry_row=own)
            predecessor = selected.get("frozen_temporal_predecessor") if old_index is not None else (
                None if index <= 1 else camera["frames"][index - 1])
            previous = None
            if predecessor is not None:
                sha = predecessor["source_sha256"]
                if old_index is not None:
                    old_previous = old_by_sha.get(sha)
                    if old_previous is None:
                        raise ValueError("Retained source's frozen temporal input is missing")
                    previous = add_source(old_previous, old_index=old_previous["frame_index"])
                else:
                    previous = _source_key(run, sha)
                    if previous not in tasks:
                        raise ValueError("New source camera-clock predecessor was not selected earlier")
            frames.append(dict(run=run, frame_index=index, elapsed_h=selected["elapsed_h"],
                source_sha256=selected["source_sha256"], source_path=selected["source_path"],
                source_changed=old_index is None, parent_source_frame_index=old_index,
                scientific_task_id=current, previous_scientific_task_id=previous,
                temporal_input_policy="camera-clock-prior-source" if old_index is None else "retain-parent-predecessor-by-source-sha",
                temporal_predecessor=deepcopy(predecessor), camera_run=deepcopy(run_asset["camera_run"])))
    if set(geometry) != expected_changed:
        raise ValueError("Changed-source geometry inventory differs from the selected source set")
    keys = {(row["run"], row["frame_index"]) for row in frames}
    if len(camera_runs) != 43 or len(frames) != 864 or len(keys) != 864 or len(references) != 43:
        raise ValueError("The producer requires the complete43/864 camera inventory and43 real references")
    return dict(frames=sorted(frames, key=lambda row: (row["run"], row["frame_index"])),
                references=references, scientific_tasks=sorted(tasks.values(), key=lambda row: row["task_id"]),
                changed_source_count=len(expected_changed), source_task_count=len(tasks))


def create_plan(parent, output, camera_plan, source_geometry, spec, *, resume=False):
    """Validate small immutable inputs and create a detached preparation plan."""
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.tools.register_track1_studio_candidate import registration_path, registration_warning
    from darsia.fluidflower.phase_rig_domain_inference import require_frozen_heads
    parent, root = Path(parent).resolve(), Path(output).resolve()
    if root == parent or root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("Use a new detached rig/camera revision folder")
    spec_asset, camera_asset, geometry_asset = artifact(spec), artifact(camera_plan), artifact(source_geometry)
    if spec_asset["sha256"] != SPEC_SHA256:
        raise ValueError("The producer binds only the combined rig/domain/camera specification")
    require_frozen_heads(parent)
    if not registration_warning(parent).startswith("Studio review candidate: approved G3"):
        raise ValueError("The rig revision needs an intact, registered Rule B parent")
    identity = list(_model_identity(parent))
    registration = artifact(registration_path(identity[0]))
    if read_json(registration["path"]).get("registration_kind") != "approved_g3_barrier_rule_b":
        raise ValueError("The combined revision requires the exact approved Rule B parent")
    camera, changed = read_json(camera_asset["path"]), read_json(geometry_asset["path"])
    if (camera.get("run_count"), camera.get("frame_count")) != (43, 864):
        raise ValueError("Camera plan must select the exact43/864 cohort")
    if camera["binding"]["spec_sha256"] != SPEC_SHA256:
        raise ValueError("Camera source plan binds another specification")
    if (Path(camera["binding"]["parent_candidate"]).resolve() != parent
            or camera["binding"]["parent_postflight_sha256"] != sha256_file(parent / "validation/postflight_manifest.json")
            or changed.get("binding", {}).get("camera_timeline_plan") != camera_asset):
        raise ValueError("Camera/source geometry binds another exact frozen parent or timeline plan")
    if changed.get("all_source_geometry_checks_pass") is not True:
        raise ValueError("Changed sources have not passed independent photographic geometry")
    runs = []
    for row in camera["runs"]:
        run_asset = dict(path=row["receipt_path"], sha256=row["receipt_sha256"])
        run = read_json(bound_path(run_asset))
        runs.append(dict(run=run["run"], camera_run=run_asset,
            parent_native_manifest=dict(path=run["native_manifest_path"], sha256=run["native_manifest_sha256"])))
    inventory = inventory_sources(runs, changed["sources"])
    if inventory["changed_source_count"] != camera["changed_source_count"]:
        raise ValueError("Camera plan changed-source count differs from source geometry")
    binding = dict(contract=CONTRACT, parent=str(parent), candidate=str(root),
        parent_model_identity=identity, parent_registration=registration,
        specification=spec_asset, spec_sha256=SPEC_SHA256, camera_plan=camera_asset,
        changed_source_geometry_manifest=geometry_asset, code=_code_binding(),
        configuration=metadata.CONFIGURATION, frozen_heads=HEAD_FILES,
        cpu_only=True, training=False, labels_changed=False, automatic_activation=False,
        publication_scope="studio_validation_only", mass_eligible=False, scientific_acceptance=False)
    plan_path = root / PLAN_FILE
    if plan_path.exists():
        plan = read_json(plan_path)
        if not resume or plan.get("binding") != binding or plan.get("source_inventory") != inventory:
            raise ValueError("Preparation resume differs from its immutable code/source/spec plan")
        return plan
    if root.exists():
        raise ValueError("Refusing an existing folder without this exact preparation plan")
    if resume:
        raise ValueError("Cannot resume an absent preparation plan")
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(binding=binding, job_sha256=canonical_sha256(binding), created_utc=utc_now(),
        run_count=43, frame_count=864, source_runs=runs, source_inventory=inventory)
    atomic_json(plan_path, plan)
    return plan


def _npy(path, values):
    encoded = io.BytesIO()
    np.save(encoded, values, allow_pickle=False)
    return _write_bytes(Path(path), encoded.getvalue())


def _source_folder(root, task):
    return Path(root) / "scientific_sources" / task["task_id"]


def verify_parent_float_bits(corrected, parent_cache):
    """Check the exact persisted parent bits, with bounded native row tiles."""
    old = np.load(parent_cache["log_linear_rgb"]["path"], mmap_mode="r", allow_pickle=False)
    valid = np.load(parent_cache["valid_mask"]["path"], mmap_mode="r", allow_pickle=False)
    if old.dtype != np.float32 or old.shape != corrected.shape or valid.dtype != bool or valid.shape != corrected.shape[1:]:
        raise ValueError("Parent scientific cache no longer shares the exact native grid")
    for first in range(0, corrected.shape[1], 128):
        last = first + 128
        supported = valid[first:last]
        if not np.array_equal(corrected[:, first:last].view(np.uint32)[:, supported],
                              old[:, first:last].view(np.uint32)[:, supported]):
            raise ValueError("Expanded scientific source altered retained parent float bits")


def verify_prepared_source(root, task, job_sha256):
    path = _source_folder(root, task) / "prepared_source.json"
    receipt = read_json(path)
    if (receipt.get("contract") != SOURCE_RECEIPT_CONTRACT or receipt.get("complete") is not True
            or receipt.get("job_sha256") != job_sha256 or receipt.get("task") != task):
        raise ValueError("Completed scientific source differs from this exact preparation")
    full = receipt["full_native_scientific"]
    for asset in (full["log_linear_rgb"], full["valid_mask"], full["receipt"], receipt["geometry_receipt"]):
        bound_path(asset)
    certificate = read_json(full["receipt"]["path"])
    if (certificate["source_sha256"] != task["source"]["sha256"]
            or certificate["log_linear_rgb_sha256"] != full["log_linear_rgb"]["sha256"]
            or certificate["valid_mask_sha256"] != full["valid_mask"]["sha256"]
            or certificate["parent_cache"] != full["parent_cache"]
            or certificate.get("frozen_photometry") is not True
            or (full["parent_cache"] is not None and certificate.get("parent_valid_bits_preserved") is not True)):
        raise ValueError("Scientific output no longer agrees with its source correction certificate")
    for name, dtype, shape in (("log_linear_rgb", np.dtype("float32"), (3, *metadata.SOURCE_SHAPE)),
                               ("valid_mask", np.dtype("bool"), tuple(metadata.SOURCE_SHAPE))):
        values = np.load(full[name]["path"], mmap_mode="r", allow_pickle=False)
        if values.dtype != dtype or values.shape != shape:
            raise ValueError("Scientific prepared source changed its full-native grid/dtype")
    return receipt


def prepare_source(root, task, job_sha256, *, processor=None):
    """Materialize one acquisition with exact frozen radiometry and parent bits."""
    from darsia.fluidflower.phase_ffac_rig_raw_source import RigRawSourceProcessor
    folder = _source_folder(root, task)
    if (folder / "prepared_source.json").exists():
        return verify_prepared_source(root, task, job_sha256)
    bound_path(task["parent_native_manifest"])
    bound_path(task["source"])
    processor = processor or RigRawSourceProcessor(task["parent_native_manifest"]["path"])
    if processor.identity["parent_native_manifest_sha256"] != task["parent_native_manifest"]["sha256"]:
        raise ValueError("RAW processor belongs to another exact parent run source")
    own = task["own_geometry"]
    geometry = own if own is not None else processor.parent_geometry(task["parent_frame_index"])
    if geometry["camera_arw_sha256"] != task["raw"]["sha256"]:
        raise ValueError("Scientific task and source geometry identify different RAW acquisitions")
    if own is not None:
        geometry_asset = dict(path=own["geometry_certificate_path"], sha256=own["geometry_certificate_sha256"])
        bound_path(geometry_asset)
    else:
        # This is an explicitly preserved-source certificate, not a borrowed
        # geometry for a changed acquisition. Its transport is hash-bound by
        # the sealed processor's native/cohort/calibration identity.
        certificate = dict(contract=PARENT_GEOMETRY_CONTRACT, run=task["run"],
            source_sha256=task["source"]["sha256"], parent_native_manifest=task["parent_native_manifest"],
            parent_frame_index=task["parent_frame_index"], parent_reference=task["parent_reference"],
            H_ac28_to_frame_native=metadata.homography(geometry["H_ac28_to_frame_native"]),
            H_v5_B_to_frame_native=metadata.homography(geometry["H_v5_B_to_frame_native"]),
            frozen_transport_sha256=geometry.get("receipt_sha256"),
            frozen_processor_identity=processor.identity, own_source_geometry=True,
            input_camera_jpeg=dict(path=geometry["camera_jpeg_path"],sha256=geometry["camera_jpeg_sha256"]),
            input_camera_arw=dict(path=geometry["camera_arw_path"],sha256=geometry["camera_arw_sha256"]))
        geometry_path = folder / "geometry.json"
        atomic_json(geometry_path, certificate)
        geometry_asset = artifact(geometry_path)
    corrected, audit = processor.correct(geometry, parent_frame_index=task["parent_frame_index"],
                                          parent_reference=task["parent_reference"])
    if corrected.dtype != np.float32 or corrected.shape != (3, *metadata.SOURCE_SHAPE):
        raise ValueError("Frozen RAW processor returned another native float grid")
    valid = np.isfinite(corrected).all(0)
    # Partially finite RGB channels are unavailable; the scientific decoder's
    # exact validity policy requires all three channels to be NaN there.
    corrected[:, ~valid] = np.nan
    parent = processor.native["reference"] if task["parent_reference"] else (
        None if task["parent_frame_index"] is None else processor.native["frame_sources"][task["parent_frame_index"]])
    parent_cache = None if parent is None else dict(
        log_linear_rgb=dict(path=parent["v5_log_linear_rgb_path"],sha256=parent["v5_log_linear_rgb_sha256"]),
        valid_mask=dict(path=parent["v5_log_linear_valid_mask_path"],sha256=parent["v5_log_linear_valid_mask_sha256"]))
    if parent_cache is not None and (audit.get("parent_cache") or {}).get("parent_float_bytes_preserved") is not True:
        raise ValueError("Expanded RAW output lacks exact preserved-parent-float certification")
    if parent_cache is not None:
        verify_parent_float_bits(corrected, parent_cache)
    log_asset = _npy(folder / "log_linear_rgb.npy", corrected)
    valid_asset = _npy(folder / "valid_mask.npy", valid)
    science_path = folder / "scientific_receipt.json"
    atomic_json(science_path, dict(contract=metadata.SCIENTIFIC_CONTRACT,
        source_sha256=task["source"]["sha256"], log_linear_rgb_sha256=log_asset["sha256"],
        valid_mask_sha256=valid_asset["sha256"], parent_cache=parent_cache,
        frozen_photometry=True, parent_valid_bits_preserved=parent_cache is not None,
        raw_source=task["raw"], geometry_receipt=geometry_asset,
        native_shape_hw=metadata.SOURCE_SHAPE, unit_denominator=65535,
        producer_audit=audit, cpu_only=True, training=False, labels_changed=False))
    full = dict(contract=metadata.SCIENTIFIC_CONTRACT, source_sha256=task["source"]["sha256"],
        native_shape_hw=metadata.SOURCE_SHAPE, unit_denominator=65535,
        log_linear_rgb=log_asset, valid_mask=valid_asset, receipt=artifact(science_path), parent_cache=parent_cache)
    receipt = dict(contract=SOURCE_RECEIPT_CONTRACT, complete=True, job_sha256=job_sha256,
        task=task, geometry_receipt=geometry_asset, geometry_frame=geometry,
        full_native_scientific=full, cpu_only=True, training=False, labels_changed=False)
    atomic_json(folder / "prepared_source.json", receipt)
    return receipt


def _prepare_run(root, tasks, job_sha256):
    from darsia.fluidflower.phase_ffac_rig_raw_source import RigRawSourceProcessor
    processor = None
    receipts = []
    progress_root = Path(root) / "source_progress" / tasks[0]["run"]
    progress_root.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(progress_root)
    for task in tasks:
        path = _source_folder(root, task) / "prepared_source.json"
        if path.exists():
            receipt = verify_prepared_source(root, task, job_sha256)
        else:
            if processor is None:
                processor = RigRawSourceProcessor(task["parent_native_manifest"]["path"])
            receipt = prepare_source(root, task, job_sha256, processor=processor)
        receipts.append(dict(task_id=task["task_id"], receipt=artifact(path)))
        progress.event("source_prepared", task_id=task["task_id"], completed_sources=len(receipts),
                       total_sources=len(tasks), receipt_path=str(path))
    return receipts


def _pilot_tasks(plan):
    inventory = plan["source_inventory"]
    selected = [row for row in inventory["frames"] if row["run"] == "ac28" and row["frame_index"] in (8, 16)]
    if len(selected) != 2:
        raise ValueError("Preparation pilot requires AC28 frames8 and16")
    task_ids = {row["scientific_task_id"] for row in selected}
    task_ids.update(row["previous_scientific_task_id"] for row in selected if row["previous_scientific_task_id"])
    task_ids.add(next(row["scientific_task_id"] for row in inventory["references"] if row["run"] == "ac28"))
    return [row for row in inventory["scientific_tasks"] if row["task_id"] in task_ids]


def prepare_scientific_sources(root, plan, *, workers=24, pilot=False, source_tasks=None):
    """Run source-only CPU preparation; final rig geometry is not required."""
    root = Path(root)
    if plan["binding"]["code"] != _code_binding():
        raise ValueError("Scientific preparation code changed after its immutable plan")
    workers = admitted_workers(workers)
    if pilot and source_tasks:
        raise ValueError("Choose either the fixed AC28 pilot or explicit source tasks")
    tasks = _pilot_tasks(plan) if pilot else plan["source_inventory"]["scientific_tasks"]
    subset = sorted(set(source_tasks or []))
    if subset:
        tasks = [task for task in tasks if task["task_id"] in subset]
        if {task["task_id"] for task in tasks} != set(subset):
            raise ValueError("Requested source task is outside the exact immutable inventory")
    count = len(tasks)
    bytes_per_source = int(np.prod(metadata.SOURCE_SHAPE)) * (3 * 4 + 1)
    missing = sum(not (_source_folder(root, task) / "prepared_source.json").exists() for task in tasks)
    preflight_space(root, missing * bytes_per_source + 10 * GIB)
    grouped = {}
    for task in tasks:
        grouped.setdefault(task["run"], []).append(task)
    below_normal_priority()
    log, started, completed = ProgressLog(root), time.monotonic(), []
    with JobLease(root, plan["job_sha256"]):
        log.event("scientific_preparation_started", total_sources=count, workers=workers,
                  threads=1, available_memory_bytes=available_memory_bytes(), pilot=pilot)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=worker_initializer) as pool:
            pending = [pool.submit(_prepare_run, str(root), rows, plan["job_sha256"])
                       for run, rows in sorted(grouped.items())]
            for future in as_completed(pending):
                completed.extend(future.result())
                elapsed = time.monotonic() - started
                log.event("scientific_preparation", completed_sources=len(completed), total_sources=count,
                          elapsed_seconds=elapsed, eta_seconds=elapsed * (count - len(completed)) / len(completed))
        completed.sort(key=lambda row: row["task_id"])
        receipt = dict(contract=CONTRACT, job_sha256=plan["job_sha256"], stage="scientific-preparation",
            complete=not pilot and not subset, pilot=pilot, source_subset=subset,
            source_count=len(completed), sources=completed,
            run_count=43, frame_count=864, phase_labels_produced=False,
            source_metadata_finalized=False, geometry_acceptance_required_for_inference=True,
            cpu_only=True, training=False, labels_changed=False, automatic_activation=False)
        name = ("scientific_preparation_subset_" + canonical_sha256(subset)[:16] + ".json") if subset else (
            "scientific_preparation_pilot.json" if pilot else "scientific_preparation_receipt.json")
        destination = root / name
        atomic_json(destination, receipt)
        log.event("scientific_preparation_completed", total_sources=count, pilot=pilot)
        return receipt


def require_inference_geometry(geometry_manifest, geometry_acceptance, plan):
    """The planning/science stages cannot silently approve rigid-wall geometry."""
    if geometry_manifest is None or geometry_acceptance is None:
        raise ValueError("Inference requires a complete <=3px certified rig geometry and acceptance receipt")
    geometry_asset, acceptance_asset = artifact(geometry_manifest), artifact(geometry_acceptance)
    acceptance = read_json(acceptance_asset["path"])
    metadata.validate_geometry_acceptance(acceptance,
        geometry_sha256=geometry_asset["sha256"], spec_sha256=SPEC_SHA256)
    for name in ("rig_check_measurement", "relative_geometry_certificate"):
        bound_path(acceptance[name])
    geometry = read_json(geometry_asset["path"])
    indexed = {(row["run"], row["frame_index"]): row for row in geometry.get("frames", [])}
    expected = {(row["run"], row["frame_index"]): row for row in plan["source_inventory"]["frames"]}
    if len(indexed) != 864 or len(geometry.get("frames", [])) != 864 or set(indexed) != set(expected):
        raise ValueError("Accepted rig geometry must cover every exact selected camera frame")
    for key, row in indexed.items():
        if row.get("source_sha256") != expected[key]["source_sha256"]:
            raise ValueError("Accepted rig geometry belongs to a different selected acquisition")
    return dict(geometry_manifest=geometry_asset, geometry_acceptance=acceptance_asset, geometry=geometry)


def run_candidate(root, plan, *, geometry_manifest=None, geometry_acceptance=None, workers=24):
    """Fail before model work until accepted geometry/policy are available."""
    gate = require_inference_geometry(geometry_manifest, geometry_acceptance, plan)
    preparation = Path(root) / "scientific_preparation_receipt.json"
    if not preparation.is_file() or read_json(preparation).get("complete") is not True:
        raise ValueError("Complete source scientific preparation is required before inference")
    # The final producer is deliberately a separate stage: source preparation
    # can be executed while real geometry is being resolved. The accepted
    # context/cached-parent inference and publication are wired below as soon
    # as their source-bound contracts are available, rather than using a fake
    # old candidate alias or fabricating class/area descriptors.
    raise ValueError("Accepted geometry is present, but the dedicated inference/package stage is not yet implemented")


def load_prepared_plan(path):
    """Import a sealed source stage without pretending to resume its new code."""
    path = Path(path).resolve()
    plan = read_json(path)
    if (plan.get("binding", {}).get("contract") != CONTRACT
            or plan["binding"].get("spec_sha256") != SPEC_SHA256
            or plan.get("job_sha256") != canonical_sha256(plan["binding"])
            or (plan.get("run_count"), plan.get("frame_count")) != (43, 864)):
        raise ValueError("Invalid immutable scientific preparation plan")
    root = Path(plan["binding"]["candidate"])
    if path != (root / PLAN_FILE).resolve():
        raise ValueError("Preparation plan does not belong to its declared detached revision")
    inventory = plan["source_inventory"]
    frames = inventory.get("frames", [])
    runs = {row["run"] for row in frames}
    if (len(frames) != 864 or len({(row["run"], row["frame_index"]) for row in frames}) != 864
            or len(runs) != 43 or len(inventory.get("references", [])) != 43
            or {row["run"] for row in inventory["references"]} != runs):
        raise ValueError("Preparation plan lost its complete selected/reference source coverage")
    preparation = read_json(root / "scientific_preparation_receipt.json")
    expected = {task["task_id"] for task in plan["source_inventory"]["scientific_tasks"]}
    if (preparation.get("complete") is not True or preparation.get("job_sha256") != plan["job_sha256"]
            or preparation.get("source_count") != len(expected)
            or len(preparation.get("sources", [])) != len(expected)
            or {row["task_id"] for row in preparation["sources"]} != expected):
        raise ValueError("Scientific preparation is partial or differs from the exact source inventory")
    tasks = {row["task_id"]: row for row in inventory["scientific_tasks"]}
    for row in preparation["sources"]:
        source_path = bound_path(row["receipt"])
        expected_path = (_source_folder(root, tasks[row["task_id"]]) / "prepared_source.json").resolve()
        completed = read_json(source_path)
        if (source_path != expected_path or completed.get("complete") is not True
                or completed.get("contract") != SOURCE_RECEIPT_CONTRACT
                or completed.get("job_sha256") != plan["job_sha256"]
                or completed.get("task") != tasks[row["task_id"]]):
            raise ValueError("Imported preparation source receipt differs from its exact task")
    return dict(plan=plan, plan_asset=artifact(path), preparation_asset=artifact(root / "scientific_preparation_receipt.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-plan")
    parser.add_argument("--source-geometry")
    parser.add_argument("--spec")
    parser.add_argument("--prepared-plan", help="Import the completed immutable upstream science stage for --run")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Prepare AC28 frames8/16, their frozen temporal inputs and real B")
    parser.add_argument("--source-task", action="append", help="Prepare only this exact source task (repeatable; never seals full preparation)")
    parser.add_argument("--geometry-manifest")
    parser.add_argument("--geometry-acceptance")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.prepared_plan:
        if not args.run:
            parser.error("--prepared-plan is an upstream import for --run only")
        state = load_prepared_plan(args.prepared_plan)
        plan = state["plan"]
        if Path(args.output).resolve() != Path(plan["binding"]["candidate"]).resolve():
            parser.error("--output must equal the imported source stage's detached revision")
    else:
        if any(value is None for value in (args.parent, args.camera_plan, args.source_geometry, args.spec)):
            parser.error("Planning/preparation requires --parent, --camera-plan, --source-geometry and --spec")
        plan = create_plan(args.parent, args.output, args.camera_plan, args.source_geometry, args.spec, resume=args.resume)
    if args.prepare:
        prepare_scientific_sources(args.output, plan, workers=args.workers, pilot=args.pilot, source_tasks=args.source_task)
    elif args.run:
        run_candidate(args.output, plan, geometry_manifest=args.geometry_manifest,
                      geometry_acceptance=args.geometry_acceptance, workers=args.workers)
    else:
        print("Validated camera/source plan:43 runs/864 frames; phase inference has not started.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
