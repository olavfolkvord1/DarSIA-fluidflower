"""Detached CPU orchestration for the approved G3 downstream overnight export.

Only probability layers and a regional gas head are produced. Final maps,
display candidates, Studio registration and round-2 labels are outside this
driver. Launch this module in a hidden detached process after its smoke test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import AbstractContextManager
import ctypes
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence
import uuid


CONTRACT = "ffac-g3-downstream-overnight-v1"
EXPORT_CONTRACT = "ffac-g3-downstream-probability-layers-v1"
ROUND1_ANNOTATION_SHA256 = "56c0a0477e244f4ad9b15f5370b8d97ec9ddecf058a3053084426f4d589c7dbc"
LABELLED_FRAMES = {
    "ac39": (1, 2, 3, 4, 6, 7, 9, 10, 12, 14, 15),
    "ac50": (1, 2, 3, 7, 12, 13, 16),
    "ac31": (1, 2, 3, 9, 10),
    "ac48": (3, 7, 9),
    "ac53": (3, 7, 9),
    "ac35": (3, 4, 5, 7, 9),
}
QA_FRAME = ("ac28", 8)
SMOKE_FRAMES = (("ac28", 0), QA_FRAME)
GIB = 1024 ** 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Commit an immutable receipt; an existing different receipt fails closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to overwrite an existing receipt: {path}")
        return
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # All receipt paths have one owner under the process lease. Windows rename
    # fails if a competing destination appeared; POSIX uses a no-clobber link.
    try:
        if os.name == "nt":
            temporary.rename(path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_text(path: Path, value: str) -> None:
    """Commit an immutable human-readable report beside the layer receipts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise ValueError(f"Refusing to overwrite an existing report: {path}")
        return
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            temporary.rename(path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _process_birth(pid: int) -> str | None:
    """Return a live process token, distinguishing a reused Windows PID."""
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 4
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            if ctypes.get_last_error() == 5:
                return "access-denied-live"
            return None
        try:
            exit_code = ctypes.c_uint32()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return "unknown-live"
            if exit_code.value != 259:
                return None
            creation, exit_time, kernel_time, user_time = [ctypes.c_uint64() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return "unknown-live"
            return str(creation.value)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return "access-denied-live"
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        return stat.read_text().rsplit(")", 1)[1].split()[19]
    return "live"


class JobLease(AbstractContextManager):
    """Exclusive PID lease, recovering only a verifiably dead former owner."""
    def __init__(self, output: Path, job_sha256: str):
        self.path = Path(output) / "job.pidlock"
        self.owner = dict(pid=os.getpid(), process_birth=_process_birth(os.getpid()),
                          job_sha256=job_sha256, token=uuid.uuid4().hex, acquired_utc=utc_now())

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(3):
            try:
                with self.path.open("x", encoding="utf-8") as stream:
                    json.dump(self.owner, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                return self
            except FileExistsError:
                previous = read_json(self.path)
                birth = _process_birth(int(previous["pid"]))
                if birth is not None and (birth == previous["process_birth"]
                                          or birth in {"unknown-live", "access-denied-live", "live"}):
                    raise RuntimeError(f"This local job is already owned by PID {previous['pid']}")
                if read_json(self.path) != previous:
                    continue
                stale = self.path.with_name("job.pidlock.stale-" + uuid.uuid4().hex)
                try:
                    self.path.rename(stale)
                except FileNotFoundError:
                    continue
        raise RuntimeError("Could not acquire the exclusive local job lease")

    def __exit__(self, exc_type, exc, traceback):
        if self.path.exists() and read_json(self.path).get("token") == self.owner["token"]:
            self.path.unlink()
        return False


def below_normal_priority() -> bool:
    if os.name != "nt":
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())
    return True


def worker_initializer(threads: int) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(threads)
    below_normal_priority()


def validate_allocation(workers: int, threads: int, cpu_count: int | None = None) -> None:
    cores = cpu_count or os.cpu_count() or 1
    if not 1 <= workers <= 12 or threads not in (1, 2):
        raise ValueError("CPU export is bounded to 1–12 workers and 1–2 threads per worker")
    if workers * threads > max(1, cores - 2):
        raise ValueError("The export allocation must leave at least two logical CPU cores free")


class ProgressLog:
    def __init__(self, output: Path):
        self.path = Path(output) / "progress.jsonl"

    def event(self, stage: str, **fields) -> None:
        value = dict(timestamp_utc=utc_now(), stage=stage, **fields)
        line = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)


def validate_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict]:
    indexed = {}
    runs = {}
    for source in rows:
        row = dict(source)
        run, index = str(row["run"]).casefold(), int(row["frame_index"])
        if not run.startswith("ac") or index < 0 or (run, index) in indexed:
            raise ValueError("Source inventory contains a duplicate or invalid native frame")
        if list(row["native_shape_hw"]) != [2979, 5472]:
            raise ValueError("Source inventory departed from the specified native grid")
        if not row.get("native_manifest_path") or len(row.get("native_manifest_sha256", "")) != 64:
            raise ValueError("Native frame inventory lacks its immutable source binding")
        row.update(run=run, frame_index=index)
        indexed[(run, index)] = row
        runs.setdefault(run, []).append(index)
    if len(runs) != 43 or len(indexed) != 864:
        raise ValueError("Overnight exports require the exact 43-run / 864-frame source inventory")
    if any(sorted(indices) != list(range(len(indices))) for indices in runs.values()):
        raise ValueError("A native source run has missing frame indices")
    required = {(run, index) for run, indices in LABELLED_FRAMES.items() for index in indices} | set(SMOKE_FRAMES)
    if not required.issubset(indexed):
        raise ValueError("The source inventory omits a prescribed labelled or QA frame")
    return indexed


def ordered_stages(indexed: Mapping[tuple[str, int], dict]) -> tuple[list[dict], list[dict]]:
    first = [dict(indexed[(run, index)], labelled=True)
             for run, indices in LABELLED_FRAMES.items() for index in indices]
    first.append(dict(indexed[QA_FRAME], labelled=True))
    keys = {(row["run"], row["frame_index"]) for row in first}
    rest = [dict(indexed[key], labelled=False) for key in sorted(indexed) if key not in keys]
    return first, rest


def estimate_output_bytes(rows: Sequence[Mapping[str, Any]], *, labelled_count: int) -> int:
    """Conservative native + compact regional storage, including labelled overhead."""
    # The ordinary native pixel area is about .027 mm2. Bounding regional
    # working grids at 5% of native pixels exceeds their 1 mm-grid density.
    pixels = sum(int(row["native_shape_hw"][0]) * int(row["native_shape_hw"][1]) for row in rows)
    compact = pixels * (1 + 5 / 8 + .05 * 4)
    maximum_pixels = max((int(row["native_shape_hw"][0]) * int(row["native_shape_hw"][1]) for row in rows), default=0)
    labelled_overhead = labelled_count * maximum_pixels * 5
    return int(compact + labelled_overhead + 2 * GIB)


def preflight_space(output: Path, required_bytes: int) -> dict:
    existing = Path(output)
    while not existing.exists():
        if existing == existing.parent:
            raise ValueError("Cannot resolve an existing output filesystem")
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    if usage.free < required_bytes:
        raise OSError(f"Insufficient free space: need {required_bytes} bytes, have {usage.free}")
    return dict(required_bytes=required_bytes, free_bytes=usage.free, filesystem_path=str(existing))


def build_job_binding(candidate: Path, experiment: Path, snapshot: Path,
                      spec: Path, indexed: Mapping[tuple[str, int], dict]) -> dict:
    from darsia.fluidflower import phase_g3_probability_layers, phase_g3_regional_refit
    package = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__), Path(phase_g3_probability_layers.__file__),
                  Path(phase_g3_regional_refit.__file__), package / "phase_gas_flatfield.py",
                  package / "phase_gas_regional_context.py", package / "phase_regional_gas.py"]
    code_paths.extend(package / name for name in (
        "phase_optical_regions.py", "phase_gas_inference_chunks.py", "phase_gas_variant_sources.py",
        "phase_features.py", "phase_learning.py", "phase_free_gas_output_guard.py",
        "phase_training/logistic.py", "phase_training/v5_linear_input.py",
        "tools/train_ffac_gas_flatfield_variants.py", "tools/train_phase_optical_regions.py",
        "tools/train_phase_model_further.py", "tools/ffac_v5_track1_photo_context.py",
        "tools/ffac_v5_temporal_alignment.py", "tools/phase_inference_benchmark_backends.py"))
    annotations = snapshot / "pairwise_review_annotations.csv"
    if sha256_file(annotations) != ROUND1_ANNOTATION_SHA256:
        raise ValueError("Regional refit must use the specified sealed round-1 snapshot")
    frozen = {
        name: sha256_file(candidate / name) for name in (
            "models/generic/co2_presence_signal.json", "regional_model.json", "phase_optical_regions.json")
    }
    fit = read_json(experiment / "fit_receipt.json")
    if fit["frozen_candidate"] != str(candidate):
        raise ValueError("G3 experiment is bound to another frozen candidate")
    for name, digest in frozen.items():
        if fit["frozen_files_sha256"][name] != digest:
            raise ValueError("Frozen model changed after G3 training")
    g3 = experiment / "variants/G3/gas_head.json"
    if sha256_file(g3) != fit["variants"]["G3"]["model_sha256"]:
        raise ValueError("The G3 head changed after the original fit")
    if (sha256_file(g3) != phase_g3_regional_refit.G3_SHA256
            or frozen["models/generic/co2_presence_signal.json"] != phase_g3_regional_refit.CO2_SHA256):
        raise ValueError("Pixel heads differ from the exact G3/CO2 heads specified for downstream refit")
    source_rows = [indexed[key] for key in sorted(indexed)]
    return dict(contract=CONTRACT, frozen_candidate=str(candidate), experiment=str(experiment),
                snapshot=str(snapshot), round1_annotations_sha256=ROUND1_ANNOTATION_SHA256,
                snapshot_manifest_sha256=sha256_file(snapshot / "SNAPSHOT_MANIFEST.json"),
                spec_path=str(spec), spec_sha256=sha256_file(spec),
                frozen_files_sha256=frozen, g3_model_path=str(g3), g3_model_sha256=sha256_file(g3),
                source_inventory_sha256=canonical_sha256(source_rows), source_inventory=source_rows,
                code_sha256={str(path): sha256_file(path) for path in code_paths},
                runtime_versions={name: importlib.metadata.version(name)
                                  for name in ("numpy", "scipy", "threadpoolctl")},
                resource_policy=dict(cpu_only=True, maximum_workers=12, maximum_native_threads=2,
                                     maximum_fit_threads=12,
                                     logical_cores_reserved=2, windows_priority="BELOW_NORMAL_PRIORITY_CLASS"),
                scope=dict(probability_exports_all_43_runs=True, total_frames=864,
                           regional_gas_refit_only=True, g3_head_frozen=True, co2_union_frozen=True,
                           round2_labels_used=False, final_maps=False, studio_candidate=False))


def bind_job(output: Path, binding: dict, *, resume: bool) -> dict:
    path = output / "job_request.json"
    if path.exists():
        request = read_json(path)
        if request["binding"] != binding or request["job_sha256"] != canonical_sha256(binding):
            raise ValueError("Resume source, models, spec or code differ from the immutable job request")
        return request
    if resume:
        raise ValueError("--resume requires an existing immutable job request")
    request = dict(created_utc=utc_now(), binding=binding, job_sha256=canonical_sha256(binding))
    atomic_json(path, request)
    return request


def verified_frame_receipt(path: Path, root: Path, *, run: str, frame_index: int,
                           labelled: bool, model_path: Path, provisional: bool) -> dict | None:
    if not path.exists():
        return None
    receipt = read_json(path)
    binding = receipt["binding"]
    metadata = receipt["metadata"]
    if (str(binding["run"]).casefold() != run or int(binding["frame_index"]) != frame_index
            or str(metadata["run"]).casefold() != run or int(metadata["frame_index"]) != frame_index
            or bool(binding["labelled"]) != labelled or bool(binding["provisional"]) != provisional
            or binding["regional_refit_sha256"] != sha256_file(model_path)):
        raise ValueError("Completed frame receipt belongs to another native frame")
    for item in receipt["files"].values():
        file_path = Path(item["path"])
        if not file_path.is_absolute():
            file_path = path.parent / file_path
        file_path = file_path.resolve()
        if not file_path.is_relative_to(root.resolve()) or sha256_file(file_path) != item["sha256"]:
            raise ValueError("Completed probability layer changed or escaped the local namespace")
    # The exporter additionally checks its full source/model/precision binding
    # on every call, including a verified completed receipt. That is cheap and
    # avoids duplicating its scientific contract in this orchestration module.
    return dict(receipt, receipt_path=str(path), receipt_sha256=sha256_file(path))


def export_chunk(candidate: str, experiment: str, model_path: str, output: str,
                 rows: Sequence[dict], threads: int, provisional: bool) -> list[dict]:
    from darsia.fluidflower.phase_g3_probability_layers import export_frame
    results = []
    for row in rows:
        started = time.perf_counter()
        receipt_path = Path(output) / "layers" / row["run"] / f"frame_{row['frame_index']:03d}" / "receipt.json"
        prior = verified_frame_receipt(receipt_path, Path(output), run=row["run"],
            frame_index=row["frame_index"], labelled=row["labelled"],
            model_path=Path(model_path), provisional=provisional)
        receipt = export_frame(Path(candidate), Path(experiment), Path(model_path), Path(output),
                               row["run"], row["frame_index"], labelled=row["labelled"],
                               threads=threads, provisional=provisional)
        verified_frame_receipt(receipt_path, Path(output), run=row["run"],
            frame_index=row["frame_index"], labelled=row["labelled"],
            model_path=Path(model_path), provisional=provisional)
        results.append(dict(run=row["run"], frame_index=row["frame_index"],
                            receipt_path=str(receipt_path), receipt_sha256=sha256_file(receipt_path),
                            seconds=time.perf_counter() - started, resumed=prior is not None))
        print(json.dumps(dict(timestamp_utc=utc_now(), stage="frame_receipt_verified",
                              run=row["run"], frame_index=row["frame_index"],
                              resumed=prior is not None)), flush=True)
    return results


def chunks_by_run(rows: Sequence[dict], chunk_frames: int) -> list[list[dict]]:
    if not 1 <= chunk_frames <= 8:
        raise ValueError("Frame chunks must contain 1–8 frames")
    chunks, current = [], []
    for row in rows:
        if current and (current[0]["run"] != row["run"] or len(current) == chunk_frames):
            chunks.append(current)
            current = []
        current.append(row)
    if current:
        chunks.append(current)
    return chunks


def run_export_stage(stage: str, rows: Sequence[dict], *, candidate: Path, experiment: Path,
                     model_path: Path, output: Path, workers: int, threads: int,
                     chunk_frames: int, log: ProgressLog, provisional: bool = False) -> list[dict]:
    chunks = chunks_by_run(rows, chunk_frames)
    started = time.perf_counter()
    complete = []
    log.event(stage + "_start", completed_frames=0, total_frames=len(rows), eta_seconds=None)
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(chunks))),
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initializer, initargs=(threads,)) as pool:
        futures = {pool.submit(export_chunk, str(candidate), str(experiment), str(model_path),
                               str(output), chunk, threads, provisional): chunk for chunk in chunks}
        for future in as_completed(futures):
            results = future.result()
            expected = {(row["run"], row["frame_index"]) for row in futures[future]}
            observed = {(row["run"], row["frame_index"]) for row in results}
            if expected != observed or len(results) != len(expected):
                raise ValueError("A worker receipt omitted or duplicated an assigned native frame")
            complete.extend(results)
            elapsed = time.perf_counter() - started
            eta = elapsed / len(complete) * (len(rows) - len(complete))
            log.event(stage, completed_frames=len(complete), total_frames=len(rows),
                      elapsed_seconds=elapsed, eta_seconds=eta)
    if len(complete) != len(rows):
        raise ValueError("Export stage did not complete its exact assigned frame inventory")
    return sorted(complete, key=lambda row: (row["run"], row["frame_index"]))


def _report_frame_receipts(output: Path, frames: Sequence[dict], *, model_sha256: str) -> list[dict]:
    """Read immutable receipt metadata, without opening any scientific array."""
    result, seen = [], set()
    for row in sorted(frames, key=lambda item: (item["run"], item["frame_index"])):
        key = (str(row["run"]).casefold(), int(row["frame_index"]))
        path = Path(row["receipt_path"]).resolve()
        if key in seen or not path.is_relative_to(output.resolve()):
            raise ValueError("Export report contains a duplicate or escaped frame receipt")
        if sha256_file(path) != row["receipt_sha256"]:
            raise ValueError("Export receipt changed before reporting")
        receipt = read_json(path)
        binding, metadata = receipt["binding"], receipt["metadata"]
        labelled = bool(binding["labelled"])
        expected_layout = ("native-float16" if labelled else
                           "compact-g3-u8-regional-own-grid-float16")
        if (receipt.get("contract") != EXPORT_CONTRACT or receipt.get("complete") is not True
                or binding["provisional"] is not False
                or (str(binding["run"]).casefold(), int(binding["frame_index"])) != key
                or (str(metadata["run"]).casefold(), int(metadata["frame_index"])) != key
                or binding["regional_refit_sha256"] != model_sha256
                or metadata["storage_layout"] != expected_layout
                or list(metadata["native_shape_hw"]) != [2979, 5472]):
            raise ValueError("Export report requires complete, source-bound, non-provisional layers")
        result.append(dict(run=key[0], frame_index=key[1], receipt_path=str(path),
                           receipt_sha256=row["receipt_sha256"], labelled=labelled,
                           metadata=metadata))
        seen.add(key)
    return result


def write_labelled_layers_ready(output: Path, frames: Sequence[dict], *,
                               job_sha256: str, model_sha256: str) -> dict:
    """Write the notification marker as soon as the prescribed B layers exist."""
    reviewed = _report_frame_receipts(output, frames, model_sha256=model_sha256)
    required = {(run, index) for run, indices in LABELLED_FRAMES.items() for index in indices} | {QA_FRAME}
    if {(row["run"], row["frame_index"]) for row in reviewed} != required or not all(row["labelled"] for row in reviewed):
        raise ValueError("Labelled ready marker requires all 35 prescribed native-float16 frames")
    payload = dict(contract=CONTRACT, export_contract=EXPORT_CONTRACT,
                   stage="labelled_layers_ready", job_sha256=job_sha256,
                   regional_model_sha256=model_sha256, frame_count=len(reviewed),
                   native_shape_hw=[2979, 5472], storage_layout="native-float16", provisional=False,
                   frames=[{key: row[key] for key in ("run", "frame_index", "receipt_path", "receipt_sha256")}
                           for row in reviewed],
                   qa_status_at_marker_creation="pending_ac28_frame8", notification_ready=True,
                   round2_labels_read=False, test_scoring_performed=False,
                   final_maps_written=False, studio_candidate_written=False)
    path = output / "labelled_layers_ready.json"
    if path.exists():
        saved = read_json(path)
        if {key: value for key, value in saved.items() if key != "created_utc"} != payload:
            raise ValueError("Existing labelled ready marker binds different exports")
    else:
        atomic_json(path, dict(created_utc=utc_now(), **payload))
    return dict(path=str(path), sha256=sha256_file(path))


def write_export_inventory(output: Path, frames: Sequence[dict], *, job_sha256: str,
                           model_sha256: str, qa_path: Path) -> dict:
    """Report the export inventory and method, without label or phase statistics."""
    reviewed = _report_frame_receipts(output, frames, model_sha256=model_sha256)
    by_run = {}
    for frame in reviewed:
        by_run.setdefault(frame["run"], []).append(frame)
    if len(reviewed) != 864 or len(by_run) != 43:
        raise ValueError("Final export inventory requires all 43 runs and 864 frames")
    qa_sha = sha256_file(qa_path)
    rows = []
    for run, entries in sorted(by_run.items()):
        if [row["frame_index"] for row in entries] != list(range(len(entries))):
            raise ValueError("Export inventory omitted a native source frame")
        grids = sorted({tuple(row["metadata"]["regional_working_shape_hw"]) for row in entries})
        pixel_areas = sorted({float(row["metadata"]["pixel_area_mm2"]) for row in entries})
        if not pixel_areas or any(area <= 0 for area in pixel_areas):
            raise ValueError("Export inventory lacks positive native pixel area metadata")
        rows.append(dict(run=run, frame_count=len(entries),
                         native_float16_frames=sum(row["labelled"] for row in entries),
                         compact_uint8_frames=sum(not row["labelled"] for row in entries),
                         native_height=2979, native_width=5472,
                         regional_working_grids_hw=json.dumps(grids, separators=(",", ":")),
                         storage_modes=";".join(sorted({row["metadata"]["storage_layout"] for row in entries})),
                         pixel_area_mm2_values=json.dumps(pixel_areas, separators=(",", ":")),
                         qa_receipt_path=str(qa_path) if run == "ac28" else "",
                         qa_receipt_sha256=qa_sha if run == "ac28" else ""))
    summary = dict(contract=CONTRACT, report="export_inventory", job_sha256=job_sha256,
                   regional_model_sha256=model_sha256, run_count=len(rows), frame_count=len(reviewed),
                   per_run=rows, frames=[{key: row[key] for key in
                       ("run", "frame_index", "receipt_path", "receipt_sha256", "labelled")}
                       for row in reviewed],
                   qa_receipt_path=str(qa_path), qa_receipt_sha256=qa_sha,
                   round2_labels_read=False, test_scoring_performed=False,
                   categorical_outputs_created=False)
    inventory_path = output / "export_inventory.json"
    atomic_json(inventory_path, summary)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_path = output / "export_inventory.csv"
    atomic_text(csv_path, stream.getvalue())
    methods_path = output / "export_inventory_methods.md"
    atomic_text(methods_path, "\n".join([
        "This inventory describes the exported probability files and source-grid metadata. It contains no label scores or phase-area statistics.",
        "",
        "The CPU job first refits only the regional gas head using the original G3 training inventory and sealed round-1 gas contours. G3, the pixel CO2 head, the regional CO2 head and the other regional settings remain frozen. The refit receipt records completeness, sampling and invalid-feature policies.",
        "",
        "Task B writes native-grid float16 probabilities for the 34 prescribed frames of the six labelled runs and AC28 frame 8. All other frames use uint8 G3 codes round(254*p), with 255 reserved for invalid features, and float16 regional probabilities on their own working grids. Every frame has the same layers/run/frame_NNN file layout and a non-provisional SHA-bound receipt.",
        "",
        "The native grid is 2979 by 5472. CO2 is the frozen final categorical union stored as little-endian packbits. Source-photo support and reconstruction masks are separate lossless packed fields. Pixel area in the CSV comes from each frame's nominal source-grid spacing; it does not confer physical-scale or mass eligibility.",
        "",
        "Regional probabilities are resized with cv2.INTER_LINEAR from float32 working-grid values to the native grid, with the threshold applied after resizing, as documented in each frame receipt. Native float16 arrays preserve the prescribed labelled export format. AC28 frame 8 QA is recorded separately in qa_ac28_frame8.json; the inventory links that receipt without reproducing its statistics.",
        "",
        "The table counts source frames and storage modes per run, and lists the native/regional grids and pixel-area metadata. Exact frame/file SHA256 values remain in the linked receipts. After the NAS copy, nas_transfer_receipt.json supplies the local-to-NAS path-prefix mapping for these immutable receipts.",
        "",
        "No round-2 labels are read, no 27-configuration scoring is performed, and no final categorical maps or Studio candidate are produced. This report is named export_inventory because the available specification contains no section 4c table definition.",
        "",
    ]))
    return {name: dict(path=str(path), sha256=sha256_file(path)) for name, path in (
        ("json", inventory_path), ("csv", csv_path), ("methods", methods_path))}


def copy_file_verified(source: Path, destination: Path, *, attempts: int = 4,
                       sleep=time.sleep) -> dict:
    expected = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise ValueError("Existing NAS output differs; never overwrite it")
        return dict(path=str(destination), sha256=expected, reused=True)
    for attempt in range(attempts):
        temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != expected:
                raise OSError("NAS copy failed its SHA256 verification")
            if destination.exists():
                if sha256_file(destination) != expected:
                    raise ValueError("Concurrent NAS output differs; never overwrite it")
                temporary.unlink()
            else:
                temporary.rename(destination)
            return dict(path=str(destination), sha256=expected, reused=False)
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            sleep(min(2 ** attempt, 8))
    raise RuntimeError("NAS copy retry loop exhausted")


def copy_tree_verified(local: Path, nas: Path, *, log: ProgressLog) -> dict:
    files = [path for path in sorted(local.rglob("*")) if path.is_file()
             and path.name not in {"job.pidlock", "progress.jsonl"}
             and ".partial-" not in path.name and not path.name.startswith("job.pidlock.stale-")]
    preflight_space(nas, sum(path.stat().st_size for path in files) + GIB)
    copied = []
    for index, source in enumerate(files, 1):
        copied.append(copy_file_verified(source, nas / source.relative_to(local)))
        if index % 100 == 0 or index == len(files):
            log.event("nas_copy", completed_files=index, total_files=len(files))
    log.event("nas_copy_complete", completed_files=len(files), total_files=len(files))
    transfer = dict(contract=CONTRACT, local_source_root=str(local.resolve()),
                    nas_output_root=str(nas.resolve()),
                    path_resolution_rule="Replace the absolute local_source_root prefix in original immutable receipts with nas_output_root when reading the copied layers.",
                    files=[dict(relative_path=str(path.relative_to(local)), sha256=sha256_file(path))
                           for path in files if path.name != "nas_transfer_receipt.json"])
    transfer_path = local / "nas_transfer_receipt.json"
    atomic_json(transfer_path, transfer)
    copied.append(copy_file_verified(transfer_path, nas / transfer_path.name))
    if log.path.exists():
        # The complete log snapshot is copied last, after its final copy event.
        log_destination = nas / "progress.jsonl"
        if log_destination.exists() and sha256_file(log_destination) != sha256_file(log.path):
            log_destination = nas / ("progress-resume-" + uuid.uuid4().hex + ".jsonl")
        copied.append(copy_file_verified(log.path, log_destination))
    return dict(nas_output=str(nas), files=copied)


def _qa_receipt(receipt: dict, experiment: Path) -> dict:
    from darsia.fluidflower.phase_g3_probability_layers import (
        qa_ac28_frame8, resolve_current_g3_map)
    path = resolve_current_g3_map(experiment, "ac28", 8)
    qa = qa_ac28_frame8(receipt, path)
    if "qa_pass" in qa:
        passed = bool(qa["qa_pass"])
    else:
        agreement = qa.get("faithful_pixel_agreement", qa.get("pixel_agreement"))
        if agreement is None:
            raise ValueError("AC28 QA did not return an explicit reconstruction agreement")
        passed = float(agreement) >= .999
    if not passed:
        raise ValueError("AC28 QA reconstruction is below the required 99.9 percent agreement")
    return qa


def orchestrate(*, frozen_candidate: Path, experiment: Path, snapshot: Path,
                spec: Path, output: Path, mode: str, nas_output: Path | None = None,
                workers: int = 12, threads: int = 2, chunk_frames: int = 4) -> dict:
    from darsia.fluidflower.phase_g3_probability_layers import inventory_frames
    from darsia.fluidflower.phase_g3_regional_refit import fit_regional_g3
    if mode not in {"smoke", "run", "resume"}:
        raise ValueError("Mode must be smoke, run or resume")
    candidate, experiment, snapshot, spec, output = [Path(path).resolve() for path in
        (frozen_candidate, experiment, snapshot, spec, output)]
    for frozen in (candidate, experiment, snapshot):
        if output.is_relative_to(frozen) or frozen.is_relative_to(output):
            raise ValueError("Local output must be detached from frozen source and model inputs")
    if output.drive.startswith("\\"):
        raise ValueError("Overnight work must be written to local disk before the NAS copy")
    if nas_output is not None:
        nas_output = Path(nas_output).resolve()
        if any(nas_output.is_relative_to(path) or path.is_relative_to(nas_output)
               for path in (candidate, experiment, snapshot, output)):
            raise ValueError("NAS output must be a new detached destination")
    validate_allocation(workers, threads)
    indexed = validate_inventory(inventory_frames(candidate, experiment))
    first, rest = ordered_stages(indexed)
    full_estimate = estimate_output_bytes(list(indexed.values()), labelled_count=len(first))
    smoke_rows = [dict(indexed[key], labelled=True) for key in SMOKE_FRAMES]
    required = (estimate_output_bytes(smoke_rows, labelled_count=len(smoke_rows))
                if mode == "smoke" else max(20 * GIB, full_estimate))
    space = preflight_space(output, required)
    output.mkdir(parents=True, exist_ok=True)
    binding = build_job_binding(candidate, experiment, snapshot, spec, indexed)
    request = bind_job(output, binding, resume=mode == "resume")
    with JobLease(output, request["job_sha256"]):
        priority_set = below_normal_priority()
        log = ProgressLog(output)
        log.event("job_start", mode=mode, workers=workers, native_threads=threads,
                  priority_below_normal=priority_set, disk_preflight=space,
                  expected_full_output_bytes=full_estimate, job_sha256=request["job_sha256"])
        if mode == "smoke":
            if (output / "smoke_receipt.json").is_file():
                smoke = read_json(output / "smoke_receipt.json")
                if smoke["job_sha256"] != request["job_sha256"]:
                    raise ValueError("Existing smoke test binds another job")
                for frame in smoke["frames"]:
                    path = Path(frame["receipt_path"])
                    if sha256_file(path) != frame["receipt_sha256"]:
                        raise ValueError("Existing smoke frame receipt changed")
                    verified_frame_receipt(path, output / "smoke", run=frame["run"],
                        frame_index=frame["frame_index"], labelled=True,
                        model_path=candidate / "regional_model.json", provisional=True)
                log.event("smoke_already_complete")
                return smoke
            results = run_export_stage("smoke", smoke_rows, candidate=candidate, experiment=experiment,
                model_path=candidate / "regional_model.json", output=output / "smoke",
                workers=1, threads=threads, chunk_frames=2, log=log, provisional=True)
            mean_seconds = sum(row["seconds"] for row in results) / len(results)
            smoke = dict(contract=CONTRACT, stage="smoke_complete", completed_utc=utc_now(),
                         job_sha256=request["job_sha256"], frames=results, provisional_regional_refit=True,
                         seconds_per_frame=mean_seconds,
                         expected_full_export_seconds=mean_seconds * len(indexed) / workers,
                         regional_refit_time_included=False, cpu_only=True,
                         final_maps_written=False, studio_candidate_written=False)
            qa_frame = next(row for row in results if (row["run"], row["frame_index"]) == QA_FRAME)
            smoke["qa_ac28_frame8"] = _qa_receipt(
                dict(read_json(qa_frame["receipt_path"]), receipt_path=qa_frame["receipt_path"]), experiment)
            atomic_json(output / "smoke_receipt.json", smoke)
            log.event("smoke_complete", seconds_per_frame=mean_seconds,
                      expected_full_export_seconds=smoke["expected_full_export_seconds"])
            return smoke
        if not (output / "smoke_receipt.json").is_file():
            raise ValueError("A verified two-frame smoke test is required before --run or --resume")
        smoke = read_json(output / "smoke_receipt.json")
        if smoke["job_sha256"] != request["job_sha256"]:
            raise ValueError("Smoke test is bound to another job request")
        for frame in smoke["frames"]:
            path = Path(frame["receipt_path"])
            if sha256_file(path) != frame["receipt_sha256"]:
                raise ValueError("Smoke receipt changed before the overnight job")
            verified_frame_receipt(path, output / "smoke", run=frame["run"], frame_index=frame["frame_index"],
                                   labelled=True, model_path=candidate / "regional_model.json", provisional=True)
        started = time.perf_counter()
        fit_threads = min(12, workers * threads)
        log.event("regional_refit_start", completed_frames=0, total_frames=len(indexed),
                  fit_threads=fit_threads, eta_seconds=None)
        refit = fit_regional_g3(candidate, experiment, snapshot, output / "regional_refit",
                               workers=workers, threads=threads, fit_threads=fit_threads)
        model_path = Path(refit.get("regional_model_path") or refit.get("model_path")
                          or output / "regional_refit/regional_model.json").resolve()
        if not model_path.is_relative_to(output.resolve()) or not model_path.is_file():
            raise ValueError("Regional refit model is missing or escaped the local work directory")
        model_sha = sha256_file(model_path)
        log.event("regional_refit_complete", model_sha256=model_sha)
        labelled = run_export_stage("labelled_export", first, candidate=candidate, experiment=experiment,
            model_path=model_path, output=output, workers=workers, threads=threads,
            chunk_frames=chunk_frames, log=log)
        ready_marker = write_labelled_layers_ready(output, labelled,
            job_sha256=request["job_sha256"], model_sha256=model_sha)
        log.event("labelled_layers_ready", completed_frames=len(labelled), total_frames=len(first),
                  marker_path=ready_marker["path"], marker_sha256=ready_marker["sha256"])
        qa_frame = next(row for row in labelled if (row["run"], row["frame_index"]) == QA_FRAME)
        qa = _qa_receipt(dict(read_json(qa_frame["receipt_path"]), receipt_path=qa_frame["receipt_path"]), experiment)
        qa_path = output / "qa_ac28_frame8.json"
        if not qa_path.exists():
            atomic_json(qa_path, {"contract": CONTRACT, "job_sha256": request["job_sha256"], **qa})
        log.event("ac28_qa_pass", frame_index=8)
        remaining = run_export_stage("remaining_export", rest, candidate=candidate, experiment=experiment,
            model_path=model_path, output=output, workers=workers, threads=threads,
            chunk_frames=chunk_frames, log=log)
        export_inventory = write_export_inventory(output, labelled + remaining,
            job_sha256=request["job_sha256"], model_sha256=model_sha, qa_path=qa_path)
        complete = dict(contract=CONTRACT, stage="probability_exports_complete", job_sha256=request["job_sha256"],
                        regional_model_path=str(model_path), regional_model_sha256=model_sha,
                        regional_refit_receipt_sha256=sha256_file(output / "regional_refit/regional_refit_receipt.json"),
                        frames=sorted(({key: row[key] for key in
                                       ("run", "frame_index", "receipt_path", "receipt_sha256")}
                                      for row in labelled + remaining),
                                      key=lambda row: (row["run"], row["frame_index"])),
                        qa_receipt_path=str(qa_path), qa_receipt_sha256=sha256_file(qa_path),
                        labelled_layers_ready=ready_marker, export_inventory=export_inventory,
                        cpu_only=True, final_maps_written=False, studio_candidate_written=False)
        atomic_json(output / "completion_receipt.json", complete)
        log.event("probability_exports_complete", completed_frames=len(complete["frames"]),
                  total_frames=len(indexed), elapsed_seconds=time.perf_counter() - started, eta_seconds=0)
        if nas_output is not None:
            copy_tree_verified(output, nas_output, log=log)
        return complete


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-candidate", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nas-output", type=Path)
    available_cores = max(1, (os.cpu_count() or 1) - 2)
    default_threads = min(2, available_cores)
    parser.add_argument("--workers", type=int, default=min(12, max(1, available_cores // default_threads)))
    parser.add_argument("--threads", type=int, default=default_threads)
    parser.add_argument("--chunk-frames", type=int, default=4)
    args = parser.parse_args(argv)
    orchestrate(frozen_candidate=args.frozen_candidate, experiment=args.experiment,
        snapshot=args.snapshot, spec=args.spec, output=args.output, nas_output=args.nas_output,
        mode="smoke" if args.smoke else "resume" if args.resume else "run",
        workers=args.workers, threads=args.threads, chunk_frames=args.chunk_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
