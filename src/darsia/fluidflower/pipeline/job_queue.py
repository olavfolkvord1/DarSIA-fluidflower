"""Whole-run queue with atomic claims and attempt-specific publication leases.

The shared filesystem must provide atomic rename within one share. Progress is
written only for actual I/O or frame advances, never as a synthetic heartbeat.
An expired attempt keeps its output, but loses the right to publish a run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import threading
import traceback
import uuid
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


class LeaseLost(RuntimeError):
    """This attempt has been reclaimed or already published."""


def utc() -> str:
    """Return an aware UTC timestamp for queue lifecycle evidence.
    """
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    """Read UTF-8 JSON metadata, rejecting duplicate keys and nonfinite numbers.
    """
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(
        Path(path).read_bytes().decode("utf-8-sig"),
        object_pairs_hook=unique,
        parse_constant=invalid,
    )


def _windows_rename(source: Path, destination: Path) -> None:
    """Rename by one exclusive DELETE handle, without replacing a destination.

    Excluding FILE_SHARE_DELETE prevents concurrent source handles from moving
    the same file through several destinations. Ordinary readers may finish;
    the caller retains the existing bounded sharing-error retry policy.
    """
    import ctypes
    from ctypes import wintypes

    class RenameInfo(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", ctypes.c_ubyte),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    # DELETE only; share reads/writes, but never a second delete/rename handle.
    handle = kernel.CreateFileW(
        os.path.abspath(source), 0x00010000, 0x00000003, None,
        3, 0x02000000, None,  # OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        target = os.path.abspath(destination).encode("utf-16-le")
        buffer = ctypes.create_string_buffer(ctypes.sizeof(RenameInfo) + len(target))
        info = RenameInfo.from_buffer(buffer)
        info.ReplaceIfExists = False
        info.RootDirectory = None
        info.FileNameLength = len(target)
        ctypes.memmove(ctypes.addressof(buffer) + RenameInfo.FileName.offset,
                       target, len(target))
        if not kernel.SetFileInformationByHandle(handle, 3, buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def rename(source: Path, destination: Path) -> None:
    """Atomic move; briefly let concurrent Windows metadata readers close."""
    deadline = time.monotonic() + 1.0
    while True:
        try:
            if os.name == "nt":
                _windows_rename(source, destination)
            else:
                os.rename(source, destination)
            return
        except PermissionError as exc:
            if (os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32}
                    or time.monotonic() >= deadline):
                raise
            time.sleep(0.01)


def write_immutable(path: Path, document: dict, *, create_parents=True) -> None:
    """Publish complete UTF-8 bytes without ever replacing an existing file."""
    path = Path(path)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    data = (json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "nt":
        rename(temporary, path)  # Windows refuses an existing destination.
    else:
        os.link(temporary, path)  # POSIX rename would replace a file.
        temporary.unlink()  # Only the successful extra hardlink is removed.


def descriptor(path: Path) -> dict:
    """Bind the exact bytes of a file to its path, SHA-256, and byte count.
    """
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def require_unc(value: str) -> None:
    """Require a shared UNC input path and reject local drives and device paths.
    """
    if (not isinstance(value, str)
            or not PureWindowsPath(value).drive.startswith("\\\\")
            or value.replace("/", "\\").startswith(("\\\\?\\", "\\\\.\\"))):
        raise ValueError(f"UNC path required: {value!r}")


def name(value: str) -> str:
    """Validate and normalize a filesystem-safe run or machine identifier.
    """
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Unsafe run or machine name: {value!r}")
    key = value.casefold()
    if key.endswith(".") or key.split(".")[0] in {
        "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }:
        raise ValueError(f"Reserved filesystem name: {value!r}")
    return key


def validate_task(task: dict) -> None:
    """Validate whole-run metadata without reading scientific inputs.

    The task declares full code, calibration, and configuration identities, a
    UNC configuration path, and supported backend and working precision.
    """
    name(task["run"])
    if task.get("pipeline") == "native-phase-statistics":
        from .phase1_protocol import validate_phase1_task
        validate_phase1_task(task)
    else:
        for key, length in (("commit", 40), ("calibration_sha256", 64)):
            if not re.fullmatch(f"[0-9a-f]{{{length}}}", task.get(key, "")):
                raise ValueError(f"Task {key} must be a full lowercase SHA")
    config = task["configuration"]
    if task.get("format") == "darsia-alias-task-v1":
        from ..store_map import validate_input_reference
        from .job_inputs import validate_catalog

        if task.get("format_version") != 1 or task.get("legacy_paths") is not False:
            raise ValueError("Explicit alias-task version/legacy flag required")
        validate_input_reference(config)
        validate_input_reference(task["series"])
        validate_input_reference(task["series_manifest"])
        catalog = validate_catalog(task["input_catalog"])
        if catalog["run"] != task["run"] or catalog["configuration"] != config:
            raise ValueError("Alias task/catalogue source identity differs")
        cases = task.get("ordered_case_ids")
        if (not isinstance(cases,list) or not cases or len({name(case) for case in cases}) != len(cases)):
            raise ValueError("Exact distinct selected case IDs required")
    elif task.get("format") is not None and "path" not in config:
        raise ValueError("Unsupported explicit task format")
    else:
        require_unc(config["path"])
    if not re.fullmatch(r"[0-9a-f]{64}", config.get("sha256", "")):
        raise ValueError("Scientific configuration SHA required")
    if task.get("backend", "any") not in {"any", "cpu", "cuda"}:
        raise ValueError("Task backend must be any, cpu or cuda")
    if task.get("precision", "float32") not in {"float32", "float64"}:
        raise ValueError("Task precision must be float32 or float64")


@dataclass(frozen=True)
class Claim:
    task: dict
    machine: str
    token: str
    claimed: Path
    lease: Path
    output: Path
    input_rows: tuple = ()
    publication_factory: object = None
    local_output: object = None


@dataclass(frozen=True)
class LocalOutputBinding:
    """In-process positive configured-results admission, never serialized as a task."""

    machine_id: str
    results_root: Path
    path: Path
    admission: object


def _own_asset(root, reference):
    """Resolve only this shared bundle's explicit relative metadata descriptor."""
    if (not isinstance(reference, dict)
            or set(reference) != {"relative_path", "sha256", "bytes"}
            or not isinstance(reference["relative_path"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", ""))
            or type(reference["bytes"]) is not int or not 0 <= reference["bytes"] <= 128 * 1024 * 1024):
        raise ValueError("Exact bounded shared metadata reference required")
    relative = reference["relative_path"]
    if (not relative or "\\" in relative or PureWindowsPath(relative).drive
            or relative.startswith("/") or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("Shared metadata reference must stay inside its bundle")
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative.split("/"))
    for entry in (path, *path.parents):
        if entry == root.parent:
            break
        info = entry.lstat()
        if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Shared publication cannot follow symlink/reparse entries")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("Shared metadata escapes its original bundle")
    if path.stat().st_size != reference["bytes"]:
        raise ValueError("Shared metadata byte count differs before bounded hashing")
    actual = descriptor(path)
    if actual["sha256"] != reference["sha256"] or actual["size_bytes"] != reference["bytes"]:
        raise ValueError("Immutable shared publication metadata changed")
    return path


def _own_reference(root, path):
    path, root = Path(path), Path(root)
    relative = path.relative_to(root).as_posix()
    actual = descriptor(path)
    result = dict(relative_path=relative, sha256=actual["sha256"], bytes=actual["size_bytes"])
    _own_asset(root, result)
    return result


def _write_immutable_raw(path, raw):
    """Copy original completion bytes exactly, with the original exclusive fence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "nt":
        rename(temporary, path)
    else:
        os.link(temporary, path)
        temporary.unlink()  # Only the successful extra link is removed.


class JobQueue:
    """Filesystem protocol; workers and controllers share this implementation."""

    def __init__(self, root: Path, *, stale_seconds=600, clock=time.time, on_error=None,
                 server_clock=None, machine_id=None):
        """Open the shared queue; stale_seconds measures time since actual attempt progress."""
        self.root = Path(root)
        self.stale_seconds = stale_seconds
        self.clock = clock
        # clock remains a source-compatible argument, never a lease-time source.
        if type(stale_seconds) is not int or stale_seconds <= 0:
            raise ValueError("A positive integer actual-progress lease is required")
        self._server_clock = server_clock
        self._machine_id = name(machine_id) if machine_id is not None else None
        self._progress_lock = threading.Lock()
        self.on_error = on_error
        for directory in ("queue", "claimed", "leases", "publishing", "reclaimed", "results", "done", "failed", "logs"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def _server(self, machine=None):
        if self._server_clock is None:
            identity = self._machine_id or (name(machine) if machine is not None else None)
            if identity is None:
                raise RuntimeError("An actual machine_id or injected server clock is required for lease timing")
            from .server_clock import ServerFileClock
            self._server_clock = ServerFileClock(self.root, identity)
        return self._server_clock

    def _old_progress(self, owner, lease):
        """Read only shared progress; never follow an alias owner's local output."""
        candidates = [lease / "owner.json", *lease.glob("progress_*.json")]
        if owner["task"].get("format") != "darsia-alias-task-v1":
            # Legacy output is proven canonical/shared before it is inspected.
            expected = self.root / "results" / name(owner["task"]["run"]) / ".attempts" / owner["token"]
            if PureWindowsPath(owner["output_path"]) != PureWindowsPath(str(expected)):
                raise ValueError("Legacy progress path differs from its canonical shared attempt")
            candidates.extend(expected.glob("progress_*.json"))
        from .server_clock import file_mtime_ns
        return max(candidates, key=file_mtime_ns)

    def _prior_resume_lineage(self, run, token):
        """Bind only the unique actual predecessor of this queue generation."""
        matches = []
        for path in (self.root / "reclaimed" / run).glob("*/reclaimed.json"):
            value = read_json(path)
            if value.get("next_attempt") == token:
                if value.get("run") != run:
                    raise ValueError("Reclaimed predecessor belongs to another run")
                matches.append(path)
        if len(matches) > 1:
            raise ValueError("Queue generation has more than one actual predecessor")
        return _own_reference(self.root, matches[0]) if matches else None

    def enqueue(self, task: dict) -> Path:
        """Reserve one whole-run output identity and publish its immutable pending task."""
        validate_task(task)
        run = name(task["run"])
        result_root = self.root / "results" / run
        output = ("results/" + run if task.get("format") == "darsia-alias-task-v1"
                  else task.get("output_directory", str(result_root)))
        value = dict(task, output_directory=output, attempt_id=uuid.uuid4().hex)
        self._validate_output(value)
        result_root.mkdir()  # Also reserves the case-insensitive run identity.
        write_immutable(result_root / "task.json", value)
        destination = self.root / "queue" / (run + ".json")
        write_immutable(destination, value)
        return destination

    def _validate_output(self, task):
        actual = PureWindowsPath(task["output_directory"])
        expected = PureWindowsPath(str(self.root / "results" / name(task["run"])))
        if task.get("format") == "darsia-alias-task-v1":
            if task["output_directory"] != "results/" + name(task["run"]):
                raise ValueError("Alias task output must be its relative canonical run identity")
        else:
            if actual != expected or ".." in actual.parts:
                raise ValueError("Task output_directory must be its canonical job/results/run root")
            if PureWindowsPath(str(self.root)).drive.startswith("\\\\"):
                require_unc(task["output_directory"])
        job = self.root / "job.json"
        if job.exists():
            value = read_json(job)
            if "expected_runs" in value and task["run"] not in value["expected_runs"]:
                raise ValueError("Task is outside job.json expected_runs")

    def claim(self, machine: str, backend: str, owner: dict, *, availability=None,
              output_factory=None, publication_factory=None, exclude_runs=()) -> Claim | None:
        """Atomically claim one backend-compatible run and bind a fresh attempt lease to its owner."""
        machine = name(machine)
        if isinstance(exclude_runs, (str, bytes)):
            raise ValueError("exclude_runs must be an explicit worker-incarnation collection")
        excluded = {name(run) for run in exclude_runs}
        for pending in sorted((self.root / "queue").glob("*.json")):
            try:
                task = read_json(pending)
                validate_task(task)
                self._validate_output(task)
                if pending.name != name(task["run"]) + ".json":
                    raise ValueError("Queue filename differs from task run")
                if not re.fullmatch(r"[0-9a-f]{32}", task.get("attempt_id", "")):
                    raise ValueError("Queue attempt identity is missing or malformed")
            except FileNotFoundError:
                continue
            except (ValueError, TypeError, KeyError):
                self._quarantine(pending, machine, owner, traceback.format_exc())
                continue
            if task.get("backend", "any") not in {"any", backend}:
                continue
            if name(task["run"]) in excluded:
                continue
            alias = task.get("format") == "darsia-alias-task-v1"
            rows = ()
            before_task = deepcopy(task)
            if alias:
                job = read_json(self.root / "job.json")
                if (job.get("format") != "darsia-alias-job-v1"
                        or job.get("execution_ready") is not True
                        or job.get("runtime_support") != "darsia-role-io-runtime-v1"
                        or task.get("execution_ready") is not True
                        or task["input_catalog"].get("runtime_support") != "darsia-role-io-runtime-v1"):
                    raise RuntimeError("Prepared J2 metadata is not explicitly ready for J3 execution")
                if self._machine_id is None:
                    raise RuntimeError("Alias claim requires its explicit physical store machine_id")
                if not all(callable(callback) for callback in (availability, output_factory, publication_factory)):
                    raise RuntimeError("Alias claim requires availability, admitted local output and shared publication bindings")
                if not callable(getattr(publication_factory, "publish_progress", None)):
                    raise RuntimeError("Alias claim requires shared original frame-commit publication")
                # Availability includes original prefix/retirement admission.
                # None explicitly skips this machine; True is never admission.
                from .store_runtime import ResolvedInput, RunUnavailable
                try:
                    rows = availability(deepcopy(task))
                except RunUnavailable:
                    continue
                if rows is None:
                    continue
                if (not isinstance(rows, tuple) or not rows
                        or any(not isinstance(row, ResolvedInput) for row in rows)
                        or [row.reference for row in rows] !=
                           [item["reference"] for item in task["input_catalog"]["inputs"]]
                        or [row.uses for row in rows] !=
                           [tuple(item["uses"]) for item in task["input_catalog"]["inputs"]]):
                    raise ValueError("Availability must bind the exact ordered role catalogue; boolean admission is forbidden")
            run, token = name(task["run"]), task["attempt_id"]
            if pending.name != run + ".json" or not re.fullmatch(r"[0-9a-f]{32}", token):
                raise ValueError("Queue filename or attempt identity differs")
            claimed = self.root / "claimed" / machine / pending.name
            claimed.parent.mkdir(parents=True, exist_ok=True)
            try:
                if claimed.exists():
                    continue
                rename(pending, claimed)
            except FileNotFoundError:
                continue
            # Bind the generation actually renamed, not a pre-rename read.
            task = read_json(claimed)
            validate_task(task)
            self._validate_output(task)
            token = task["attempt_id"]
            if alias and task != before_task:
                # A different generation won while availability was inspected.
                # Never pair stale rows with the generation actually renamed.
                rename(claimed, pending)
                continue
            if task.get("backend", "any") not in {"any", backend}:
                rename(claimed, pending)
                continue
            lease = self.root / "leases" / (run + "." + token)
            lease.mkdir()
            output_reference = None
            if alias:
                binding = output_factory(deepcopy(task), token)
                output, output_reference = self._admit_output(binding, task, machine, token)
            else:
                output = self.root / "results" / run / ".attempts" / token
            output.mkdir(parents=True)
            marker = lease / "claimed.json"
            write_immutable(marker, dict(run=run, attempt_id=token, machine_id=machine,
                                        event="actual_claim_bound"), create_parents=False)
            from .server_clock import file_mtime_ns
            metadata = dict(task=task, machine=machine, token=token, actual_owner=owner,
                            claimed_unix_ns=file_mtime_ns(marker), claimed_utc=utc(),
                            lease_clock_basis="shared-filesystem-mtime")
            if alias:
                metadata.update(format="darsia-alias-lease-owner-v1",
                                claimed=_own_reference(self.root, claimed), output=output_reference)
            else:
                metadata.update(claimed_path=str(claimed), output_path=str(output))
            write_immutable(lease / "owner.json", metadata, create_parents=False)
            claim = Claim(task, machine, token, claimed, lease, output, rows,
                          publication_factory if alias else None, binding if alias else None)
            if read_json(claimed) != task:
                raise LeaseLost("Queue generation changed during claim")
            self.assert_owner(claim)
            return claim
        return None

    def _admit_output(self, binding, task, machine, token):
        from ..job_filesystem import valid_results_admission
        if (not isinstance(binding, LocalOutputBinding)
                or not valid_results_admission(binding.admission, binding.results_root)
                or self._machine_id is None
                or name(binding.machine_id) != self._machine_id):
            raise ValueError("A positive configured result-root admission is required")
        root, output = Path(binding.results_root), Path(binding.path)
        relative = name(task["job_id"]) + "/" + name(task["run"]) + "/.attempts/" + token
        expected = root.joinpath(*relative.split("/"))
        if (not root.is_absolute() or not output.is_absolute()
                or output != expected or ".." in output.parts):
            raise ValueError("Output differs from its declared results root/job/run/token")
        checked = Path(binding.admission.checked_container).resolve(strict=True)
        actual_root = root.resolve()
        if (not actual_root.is_relative_to(checked)
                or not output.resolve().is_relative_to(actual_root)):
            raise ValueError("Output escaped its positively admitted container")
        return output, dict(machine_id=binding.machine_id, root_alias="results", relative_path=relative)

    def _quarantine(self, pending, machine, owner, text):
        try:
            run = name(pending.stem)
        except ValueError:
            run = "invalid-" + hashlib.sha256(pending.name.encode()).hexdigest()[:24]
        token = uuid.uuid4().hex
        archived = self.root / "claimed" / machine / f"{run}.invalid.{token}.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        try:
            rename(pending, archived)
        except FileNotFoundError:
            return
        value = dict(complete=False, run=run, attempt_id=token,
                     state="invalid_task", traceback=text,
                     original_queue_bytes=descriptor(archived), actual_worker=owner,
                     failed_utc=utc(), scientific_computation_started=False)
        attempt = self.root / "failed" / ".attempts" / run / (token + ".json")
        write_immutable(attempt, value)
        try:
            write_immutable(self.root / "failed" / (run + ".json"), value)
        except FileExistsError:
            pass  # Keep the first terminal failure and this distinct attempt.
        if self.on_error is not None:
            self.on_error(dict(kind="invalid_task", **value))

    def assert_owner(self, claim: Claim) -> None:
        """Require the exact current claim generation before writing or publishing."""
        try:
            owner = read_json(claim.lease / "owner.json")
            task = read_json(claim.claimed)
        except FileNotFoundError as exc:
            raise LeaseLost(f"Attempt {claim.token} no longer owns {claim.task['run']}") from exc
        if owner["token"] != claim.token or task != claim.task:
            raise LeaseLost("Claim identity changed")

    def progress(self, claim: Claim, event: dict) -> None:
        """Record an actual I/O or frame-advance event with a nanosecond timestamp under ownership."""
        with self._progress_lock:
            self.assert_owner(claim)
            sequence = len(list(claim.output.glob("progress_*.json")))
            value = dict(
                run=claim.task["run"], attempt_id=claim.token, sequence=sequence,
                event=event, written_unix_ns=self._server(claim.machine).now_ns(), written_utc=utc(),
            )
            # Keep the original local operational record; it is never read by
            # another machine as lease evidence or an implicit payload source.
            write_immutable(claim.output / f"progress_{sequence:08d}.json", value)
            shared = dict(value)
            if claim.task.get("format") == "darsia-alias-task-v1":
                # Preserve the original event as passive producer provenance.
                # No consumer needs to open its producer-local paths: actual
                # frame evidence is the separately verified shared byte copy.
                self._portable_event(event)  # Validate actual progress fields.
                if event["state"] == "frame_completed":
                    shared["published_frame"] = self._frame_progress_reference(claim, event)
                if event["state"] == "phase1_frame_completed":
                    from .phase1_protocol import shared_phase1_progress
                    shared["published_phase1_frame"] = shared_phase1_progress(self, claim, event)
                    shared["detail_scope"] = "Original phase-one event provenance; exact phase1_commit bytes are published_phase1_frame within this shared bundle"
                else:
                    shared["detail_scope"] = "Original event provenance; frame evidence is published_frame within this shared bundle"
            write_immutable(claim.lease / f"progress_{sequence:08d}.json", shared,
                            create_parents=False)
            self.assert_owner(claim)

    def _frame_progress_reference(self, claim, event):
        """Bind the original committed-frame metadata without foreign-local reads."""
        publisher = getattr(claim.publication_factory, "publish_progress", None)
        if not callable(publisher):
            raise RuntimeError("Alias frame progress requires an original commit byte-copy publisher")
        record = event.get("publication")
        case = event.get("case_id")
        if (event.get("complete") is not True or event.get("run") != claim.task["run"]
                or case not in claim.task["ordered_case_ids"] or not isinstance(record, dict)
                or record.get("frame_id") != case):
            raise ValueError("Frame progress differs from this claimed task and original publication")
        original = record["files"]["commit.json"]
        size = original.get("bytes", original.get("size_bytes"))
        if (not re.fullmatch(r"[0-9a-f]{64}", original.get("sha256", ""))
                or type(size) is not int or not 0 <= size <= 128 * 1024 * 1024):
            raise ValueError("Original committed-frame SHA and size are required")
        self.assert_owner(claim)
        reference = publisher(claim, deepcopy(event))
        self.assert_owner(claim)
        copied = _own_asset(claim.lease, reference)
        if reference["sha256"] != original["sha256"] or reference["bytes"] != size:
            raise ValueError("Published frame is not the original commit.json bytes")
        commit = read_json(copied)
        expected_files = {key: value for key, value in record["files"].items() if key != "commit.json"}
        if (commit.get("complete") is not True or commit.get("frame_id") != case
                or commit.get("manifest_sha256") != record.get("manifest_sha256")
                or commit.get("files") != expected_files):
            raise ValueError("Published original frame commit differs from its publication record")
        return deepcopy(reference)

    @staticmethod
    def _portable_event(event):
        if not isinstance(event, dict) or not isinstance(event.get("state"), str):
            raise ValueError("An actual named I/O/frame progress event is required")
        name(event["state"])
        result = {"state": event["state"]}
        for key in ("run", "case_id"):
            if key in event:
                name(event[key])
                result[key] = event[key]
        for key in ("frame_index", "bytes"):
            if key in event:
                if type(event[key]) is not int or event[key] < 0:
                    raise ValueError("Actual progress counters must be nonnegative integers")
                result[key] = event[key]
        if "complete" in event:
            if type(event["complete"]) is not bool:
                raise ValueError("Actual completion progress must be a boolean")
            result["complete"] = event["complete"]
        if "compute_seconds" in event:
            if type(event["compute_seconds"]) not in {int, float} or event["compute_seconds"] < 0:
                raise ValueError("Actual compute time must be a nonnegative number")
            result["compute_seconds"] = event["compute_seconds"]
        return result

    def reclaim_stale(self) -> list[dict]:
        """Archive expired claim generations and enqueue fresh attempts without allowing stale writers."""
        reclaimed = []
        for lease in sorted((self.root / "leases").iterdir()):
            try:
                owner = read_json(lease / "owner.json")
            except FileNotFoundError:
                continue
            task, token = owner["task"], owner["token"]
            run = name(task["run"])
            try:
                progress = self._old_progress(owner, lease)
                from .server_clock import file_mtime_ns
                latest_ns = file_mtime_ns(progress)
                if not self._server().stale(progress, self.stale_seconds):
                    continue
            except FileNotFoundError:
                # Publication/reclaim may have atomically moved this lease.
                continue
            claimed_path = (self.root.joinpath(*owner["claimed"]["relative_path"].split("/"))
                            if task.get("format") == "darsia-alias-task-v1"
                            else Path(owner["claimed_path"]))
            if task.get("format") == "darsia-alias-task-v1":
                expected = self.root / "claimed" / name(owner["machine"]) / (run + ".json")
                if claimed_path != expected:
                    raise ValueError("Portable claim reference differs from its canonical shared location")
            archive = self.root / "reclaimed" / run / token
            archive.mkdir(parents=True, exist_ok=True)
            try:
                rename(lease, archive / "lease")
            except FileNotFoundError:
                continue  # Publication or another reclaimer won the rename.
            except FileExistsError:
                continue
            for old in (claimed_path, self.root / "queue" / (run + ".json")):
                if old.exists() and read_json(old).get("attempt_id") == token:
                    rename(old, archive / "task.json")
            if (self.root / "publishing" / run).exists():
                raise RuntimeError("Reclaimed run already has a publication owner")
            retry = dict(task, attempt_id=uuid.uuid4().hex)
            write_immutable(self.root / "queue" / (run + ".json"), retry)
            record = dict(run=task["run"], previous_attempt=token,
                          next_attempt=retry["attempt_id"], reclaimed_utc=utc(),
                          last_actual_progress_unix_ns=latest_ns)
            write_immutable(archive / "reclaimed.json", record)
            reclaimed.append(record)
        # A process can die after the task rename but before owner.json exists.
        # No science starts before that owner document and the ownership check.
        for claimed in (self.root / "claimed").glob("*/*.json"):
            if ".invalid." in claimed.name:
                continue
            try:
                task = read_json(claimed)
                run, token = name(task["run"]), task["attempt_id"]
                lease = self.root / "leases" / (run + "." + token)
                if (lease / "owner.json").exists() or (self.root / "publishing" / run).exists():
                    continue
                # Rename can preserve an arbitrarily old queue mtime. Start
                # this ownerless window at the first shared observation instead.
                marker = self.root / "claim_observations" / (run + "." + token + ".json")
                try:
                    write_immutable(marker, dict(run=run, attempt_id=token,
                        claimed_relative_path=claimed.relative_to(self.root).as_posix(),
                        state="ownerless_claim_first_observed"))
                except FileExistsError:
                    pass
                observation = read_json(marker)
                if (observation.get("run") != run or observation.get("attempt_id") != token
                        or observation.get("claimed_relative_path") != claimed.relative_to(self.root).as_posix()):
                    raise ValueError("Ownerless observation belongs to another claim")
                from .server_clock import file_mtime_ns
                last_ns = file_mtime_ns(marker)
                if not self._server().stale(marker, self.stale_seconds):
                    continue
                prior_lineage = self._prior_resume_lineage(run, token)
                archive = self.root / "reclaimed" / run / token
                archive.mkdir(parents=True, exist_ok=True)
                if lease.exists():
                    rename(lease, archive / "lease")
                rename(claimed, archive / "task.json")
            except FileNotFoundError:
                continue
            except FileExistsError:
                continue
            if (self.root / "publishing" / run).exists():
                continue
            retry = dict(task, attempt_id=uuid.uuid4().hex)
            write_immutable(self.root / "queue" / (run + ".json"), retry)
            record = dict(run=task["run"], previous_attempt=token,
                          next_attempt=retry["attempt_id"], reclaimed_utc=utc(),
                          last_actual_progress_unix_ns=last_ns,
                          # An owner may have appeared between the initial
                          # observation and the atomic retirement. Preserve all
                          # ordinary prefix/OS-retirement guards in that case.
                          interrupted_claim_before_science=not (archive / "lease" / "owner.json").exists(),
                          prior_resume_lineage=prior_lineage,
                          preserve_prior_resume_lineage=True)
            write_immutable(archive / "reclaimed.json", record)
            reclaimed.append(record)
        return reclaimed

    def reject_before_science(self, claim: Claim, reason: str, owner: dict) -> dict:
        """Retire only an empty alias admission, then let another machine try it.

        Worker code must add the returned run to its incarnation's exclude_runs.
        No terminal failed/done record is produced. This is not OS retirement
        and cannot be used to waive an existing scientific prefix/retirement.
        """
        if claim.task.get("format") != "darsia-alias-task-v1":
            raise ValueError("Pre-science input rejection is only an alias admission path")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("The literal input-admission failure is required")
        self.assert_owner(claim)
        if any(claim.output.iterdir()) or list(claim.lease.glob("progress_*.json")):
            raise ValueError("Pre-science rejection cannot discard an attempt with output/progress")
        run = name(claim.task["run"])
        prior_lineage = self._prior_resume_lineage(run, claim.token)
        archive = self.root / "reclaimed" / run / claim.token
        archive.mkdir(parents=True, exist_ok=True)
        try:
            rename(claim.lease, archive / "lease")
        except FileNotFoundError as exc:
            raise LeaseLost("Reclaim/publication already retired this input admission") from exc
        if read_json(claim.claimed) != claim.task:
            raise LeaseLost("Input rejection lost its exact claim generation")
        rename(claim.claimed, archive / "task.json")
        if (self.root / "publishing" / run).exists():
            raise RuntimeError("Input-rejected run already has a publication owner")
        retry = dict(claim.task, attempt_id=uuid.uuid4().hex)
        record = dict(run=claim.task["run"], previous_attempt=claim.token,
                      next_attempt=retry["attempt_id"], reclaimed_utc=utc(),
                      interrupted_claim_before_science=True, input_admission_rejected=True,
                      prior_resume_lineage=prior_lineage,
                      preserve_prior_resume_lineage=True,
                      reason=reason, actual_worker=owner,
                      exclude_run_for_this_worker_incarnation=True)
        write_immutable(archive / "reclaimed.json", record)
        write_immutable(self.root / "queue" / (run + ".json"), retry)
        return record

    def _publish(self, claim: Claim, document: dict, success: bool) -> dict:
        alias = claim.task.get("format") == "darsia-alias-task-v1"
        if alias and not callable(claim.publication_factory):
            raise RuntimeError("Alias publication requires its original immutable shared-bundle binding")
        candidate = claim.output / ("candidate_completed.json" if success else "failed.json")
        write_immutable(candidate, document)
        self.assert_owner(claim)
        destination = self.root / "publishing" / name(claim.task["run"])
        # Prepare recovery metadata before the ownership transition. A crash
        # after rename can then finish publication without scientific replay.
        prepared = None
        if alias:
            original = descriptor(candidate)
            prepared = claim.publication_factory(claim, dict(original), success)
            self.assert_owner(claim)  # Copies never acquire publication rights.
            required = {"candidate", "products_index"} if success else {"candidate"}
            optional = {"payload_publication"} if success else set()
            if (not isinstance(prepared, dict) or not required <= set(prepared)
                    or not set(prepared) <= required | optional):
                raise ValueError("Explicit shared candidate/index publication descriptors required")
            _own_asset(claim.lease, prepared["candidate"])
            if (prepared["candidate"]["sha256"] != original["sha256"]
                    or prepared["candidate"]["bytes"] != original["size_bytes"]):
                raise ValueError("Shared candidate is not the original Main completion/failure bytes")
            if success:
                index_path = _own_asset(claim.lease, prepared["products_index"])
                self._validate_alias_index(index_path, prepared["candidate"], claim.task["run"], claim.token)
                self._validate_alias_payload(claim.lease, prepared.get("payload_publication"),
                    prepared["products_index"], claim.task["run"], claim.token)
        try:
            publication = dict(attempt_id=claim.token, success=success,
                               candidate=descriptor(candidate))
            if alias:
                publication = dict(format="darsia-alias-publication-v1", run=claim.task["run"],
                                   attempt_id=claim.token, success=success, **prepared)
            write_immutable(claim.lease / "publication.json", publication, create_parents=False)
        except FileNotFoundError as exc:
            raise LeaseLost("Reclaim closed the publication lease") from exc
        self.assert_owner(claim)
        try:
            rename(claim.lease, destination)
        except FileNotFoundError as exc:
            raise LeaseLost("Reclaim won before publication") from exc
        self._finish_publication(destination)
        return document

    def _finish_publication(self, directory: Path) -> None:
        publication = read_json(directory / "publication.json")
        if publication.get("format") == "darsia-alias-publication-v1":
            self._finish_alias_publication(directory, publication)
            return
        candidate = publication["candidate"]
        actual = descriptor(Path(candidate["path"]))
        if actual != candidate:
            raise ValueError("Immutable candidate changed")
        value = read_json(Path(candidate["path"]))
        run = name(value["run"])
        target = (self.root / "results" / run / "completed.json"
                  if publication["success"] else self.root / "failed" / (run + ".json"))
        if target.exists():
            if read_json(target) != value:
                raise ValueError("A different terminal result already exists")
        else:
            try:
                write_immutable(target, value)
            except FileExistsError:
                if read_json(target) != value:
                    raise ValueError("A different terminal result already exists")
        if publication["success"]:
            done = dict(run=value["run"], attempt_id=publication["attempt_id"],
                        completed=descriptor(target))
            done_path = self.root / "done" / (run + ".json")
            if done_path.exists():
                if read_json(done_path) != done:
                    raise ValueError("Done identity changed")
            else:
                try:
                    write_immutable(done_path, done)
                except FileExistsError:
                    if read_json(done_path) != done:
                        raise ValueError("Done identity changed")

    def _finish_alias_publication(self, directory, publication):
        """Finish only the immutable moved shared bundle, never producer C: paths."""
        owner = read_json(directory / "owner.json")
        task, token = owner["task"], owner["token"]
        run = name(task["run"])
        if (task.get("format") != "darsia-alias-task-v1"
                or publication.get("run") != task["run"]
                or publication.get("attempt_id") != token
                or type(publication.get("success")) is not bool):
            raise ValueError("Shared publication owner/run/attempt differs")
        source = _own_asset(directory, publication["candidate"])
        raw = source.read_bytes()
        if (len(raw) != publication["candidate"]["bytes"]
                or hashlib.sha256(raw).hexdigest() != publication["candidate"]["sha256"]):
            raise ValueError("Immutable shared candidate changed during recovery")
        value = read_json(source)
        if value.get("run") != task["run"] or value.get("complete") is not publication["success"]:
            raise ValueError("Shared candidate run/completion differs")
        if publication["success"]:
            if value.get("worker_references_closed") is not True:
                raise ValueError("Shared completion still has open scientific references")
            index_path = _own_asset(directory, publication["products_index"])
            self._validate_alias_index(index_path, publication["candidate"], task["run"], token)
            self._validate_alias_payload(directory, publication.get("payload_publication"),
                publication["products_index"], task["run"], token)
        target = (self.root / "results" / run / "completed.json"
                  if publication["success"] else self.root / "failed" / (run + ".json"))
        self._shared_target(target)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("A different terminal result already exists")
        else:
            try:
                _write_immutable_raw(target, raw)
            except FileExistsError:
                if target.read_bytes() != raw:
                    raise ValueError("A different terminal result already exists")
        if publication["success"]:
            done = dict(format="darsia-alias-done-v1", run=task["run"], attempt_id=token,
                        completed=_own_reference(self.root, target),
                        products_index=_own_reference(self.root, index_path))
            if "payload_publication" in publication:
                payload = _own_asset(directory, publication["payload_publication"])
                done["payload_publication"] = _own_reference(self.root, payload)
            done_path = self.root / "done" / (run + ".json")
            self._shared_target(done_path)
            try:
                write_immutable(done_path, done)
            except FileExistsError:
                if read_json(done_path) != done:
                    raise ValueError("Done identity changed")

    def _validate_alias_payload(self, directory, reference, index_reference, run, token):
        """Validate default-product receipt identity; never read large payloads here."""
        index_path = _own_asset(directory, index_reference)
        index = read_json(index_path)
        job = read_json(self.root / "job.json")
        kinds = job.get("publish", ["film", "lossless_export"])
        if (not isinstance(kinds, (list, tuple)) or len(kinds) != len(set(kinds))
                or any(not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", kind)
                       for kind in kinds)):
            raise ValueError("The immutable job publish kinds are malformed")
        selected = {row["product_id"]: row for row in index["products"] if row["kind"] in kinds}
        if not selected:
            if reference is not None:
                raise ValueError("A payload publication cannot claim absent matching products")
            return
        if reference is None:
            raise ValueError("The job's indexed publish kinds require their explicit payload publication")
        path = _own_asset(directory, reference)
        receipt = read_json(path)
        bound = receipt.get("products_index")
        size = bound.get("bytes", bound.get("size_bytes")) if isinstance(bound, dict) else None
        if (receipt.get("format") != "darsia-product-publication-v1"
                or not isinstance(bound, dict) or bound.get("sha256") != index_reference["sha256"]
                or size != index_reference["bytes"]
                or receipt.get("run") != run or receipt.get("original_attempt_id") != token
                or receipt.get("file_writers_closed") is not True
                or receipt.get("verification") != "Every explicitly copied product SHA-256 and byte count verified"
                or not isinstance(receipt.get("copies"), list)):
            raise ValueError("Payload publication is not bound to this original index/run/attempt")
        copies = receipt["copies"]
        if (len(copies) != len(selected)
                or {copy.get("product_id") for copy in copies} != set(selected)):
            raise ValueError("Payload publication differs from the exact job publish kinds")
        for copy in copies:
            row = selected[copy["product_id"]]
            if (copy.get("relative_path") != "products/" + copy["product_id"]
                    or any(copy.get(key) != row[key] for key in ("machine_id", "local_path", "sha256", "bytes"))):
                raise ValueError("Published payload receipt differs from its original indexed product")

    @staticmethod
    def _validate_alias_index(index_path, candidate, run, token):
        index = read_json(index_path)
        if (index.get("format") != "darsia-products-index-v1"
                or index.get("run") != run or index.get("attempt_id") != token):
            raise ValueError("Shared products index differs from the publication owner")
        from .products_publication import read_published_metadata
        raw = read_published_metadata(descriptor(index_path))
        if (len(raw) != candidate["bytes"]
                or hashlib.sha256(raw).hexdigest() != candidate["sha256"]):
            raise ValueError("Products index does not bind the original shared completion bytes")

    def _shared_target(self, path):
        root = self.root.resolve(strict=True)
        if not path.parent.resolve().is_relative_to(root):
            raise ValueError("Shared terminal target escaped the current job")
        for entry in (path, *path.parents):
            if entry == self.root.parent:
                break
            try:
                info = entry.lstat()
            except FileNotFoundError:
                continue
            if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Shared terminal target contains symlink/reparse entries")

    def recover_publications(self) -> None:
        """Finish pending publications while leaving matching completed/done documents untouched."""
        for directory in (self.root / "publishing").iterdir():
            if (directory / "publication.json").exists():
                self._finish_publication(directory)

    def complete(self, claim: Claim, result: dict) -> dict:
        """Publish a genuine whole-run result only after scientific references are closed."""
        if result.get("complete") is not True or result.get("run") != claim.task["run"]:
            raise ValueError("A genuine complete per-run scientific result is required")
        if result.get("worker_references_closed") is not True:
            raise ValueError("Scientific references are still open")
        return self._publish(claim, result, True)

    def fail(self, claim: Claim, traceback_text: str, owner: dict) -> dict:
        """Publish the current attempt failure with its verbatim traceback and actual owner."""
        value = dict(complete=False, run=claim.task["run"], attempt_id=claim.token,
                     task=claim.task, actual_worker=owner, traceback=traceback_text,
                     failed_utc=utc())
        return self._publish(claim, value, False)

    def aggregate(self) -> dict:
        """Verify immutable completion bindings and report the expected whole-run results and failures."""
        results, failures = {}, {}
        for path in sorted((self.root / "done").glob("*.json")):
            done = read_json(path)
            if done.get("format") == "darsia-alias-done-v1":
                completed = _own_asset(self.root, done["completed"])
                _own_asset(self.root, done["products_index"])
                results[done["run"]] = read_json(completed)
            else:
                actual = descriptor(Path(done["completed"]["path"]))
                if actual != done["completed"]:
                    raise ValueError("Published result changed")
                results[done["run"]] = read_json(Path(actual["path"]))
        for path in sorted((self.root / "failed").glob("*.json")):
            failures[read_json(path)["run"]] = read_json(path)
        tasks = [read_json(p) for p in (self.root / "results").glob("*/task.json")]
        expected = {t["run"] for t in tasks}
        job = self.root / "job.json"
        if job.exists() and "expected_runs" in read_json(job):
            expected = set(read_json(job)["expected_runs"])
        return dict(complete=bool(expected) and set(results) == expected and not failures,
                    task_count=len(expected), completed_count=len(results),
                    failed_count=len(failures), results=results, failures=failures)
