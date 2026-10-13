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
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


class LeaseLost(RuntimeError):
    """This attempt has been reclaimed or already published."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
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


def rename(source: Path, destination: Path) -> None:
    """Atomic move; briefly let concurrent Windows metadata readers close."""
    deadline = time.monotonic() + 1.0
    while True:
        try:
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
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def require_unc(value: str) -> None:
    if (not isinstance(value, str)
            or not PureWindowsPath(value).drive.startswith("\\\\")
            or value.replace("/", "\\").startswith(("\\\\?\\", "\\\\.\\"))):
        raise ValueError(f"UNC path required: {value!r}")


def name(value: str) -> str:
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
    name(task["run"])
    for key, length in (("commit", 40), ("calibration_sha256", 64)):
        if not re.fullmatch(f"[0-9a-f]{{{length}}}", task.get(key, "")):
            raise ValueError(f"Task {key} must be a full lowercase SHA")
    config = task["configuration"]
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


class JobQueue:
    """Filesystem protocol; workers and controllers share this implementation."""

    def __init__(self, root: Path, *, stale_seconds=600, clock=time.time, on_error=None):
        self.root = Path(root)
        self.stale_seconds = stale_seconds
        self.clock = clock
        self._progress_lock = threading.Lock()
        self.on_error = on_error
        for directory in ("queue", "claimed", "leases", "publishing", "reclaimed", "results", "done", "failed", "logs"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def enqueue(self, task: dict) -> Path:
        validate_task(task)
        run = name(task["run"])
        result_root = self.root / "results" / run
        value = dict(task, output_directory=task.get("output_directory", str(result_root)),
                     attempt_id=uuid.uuid4().hex)
        self._validate_output(value)
        result_root.mkdir()  # Also reserves the case-insensitive run identity.
        write_immutable(result_root / "task.json", value)
        destination = self.root / "queue" / (run + ".json")
        write_immutable(destination, value)
        return destination

    def _validate_output(self, task):
        actual = PureWindowsPath(task["output_directory"])
        expected = PureWindowsPath(str(self.root / "results" / name(task["run"])))
        if actual != expected or ".." in actual.parts:
            raise ValueError("Task output_directory must be its canonical job/results/run root")
        if PureWindowsPath(str(self.root)).drive.startswith("\\\\"):
            require_unc(task["output_directory"])
        job = self.root / "job.json"
        if job.exists():
            value = read_json(job)
            if "expected_runs" in value and task["run"] not in value["expected_runs"]:
                raise ValueError("Task is outside job.json expected_runs")

    def claim(self, machine: str, backend: str, owner: dict) -> Claim | None:
        machine = name(machine)
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
            if task.get("backend", "any") not in {"any", backend}:
                rename(claimed, pending)
                continue
            os.utime(claimed, (self.clock(), self.clock()))
            lease = self.root / "leases" / (run + "." + token)
            lease.mkdir()
            output = self.root / "results" / run / ".attempts" / token
            output.mkdir(parents=True)
            write_immutable(lease / "owner.json", dict(
                task=task, machine=machine, token=token, claimed_path=str(claimed),
                output_path=str(output), actual_owner=owner,
                claimed_unix_ns=int(self.clock() * 1e9), claimed_utc=utc(),
            ), create_parents=False)
            claim = Claim(task, machine, token, claimed, lease, output)
            if read_json(claimed) != task:
                raise LeaseLost("Queue generation changed during claim")
            self.assert_owner(claim)
            return claim
        return None

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
        try:
            owner = read_json(claim.lease / "owner.json")
            task = read_json(claim.claimed)
        except FileNotFoundError as exc:
            raise LeaseLost(f"Attempt {claim.token} no longer owns {claim.task['run']}") from exc
        if owner["token"] != claim.token or task != claim.task:
            raise LeaseLost("Claim identity changed")

    def progress(self, claim: Claim, event: dict) -> None:
        with self._progress_lock:
            self.assert_owner(claim)
            sequence = len(list(claim.output.glob("progress_*.json")))
            write_immutable(claim.output / f"progress_{sequence:08d}.json", dict(
                run=claim.task["run"], attempt_id=claim.token, sequence=sequence,
                event=event, written_unix_ns=int(self.clock() * 1e9), written_utc=utc(),
            ))
            self.assert_owner(claim)

    def reclaim_stale(self) -> list[dict]:
        reclaimed = []
        for lease in sorted((self.root / "leases").iterdir()):
            try:
                owner = read_json(lease / "owner.json")
            except FileNotFoundError:
                continue
            task, token = owner["task"], owner["token"]
            run = name(task["run"])
            progress = sorted(Path(owner["output_path"]).glob("progress_*.json"))
            latest_ns = owner["claimed_unix_ns"]
            if progress:
                latest_ns = read_json(progress[-1])["written_unix_ns"]
            if self.clock() - latest_ns / 1e9 < self.stale_seconds:
                continue
            archive = self.root / "reclaimed" / run / token
            archive.mkdir(parents=True, exist_ok=True)
            try:
                rename(lease, archive / "lease")
            except FileNotFoundError:
                continue  # Publication or another reclaimer won the rename.
            except FileExistsError:
                continue
            for old in (Path(owner["claimed_path"]), self.root / "queue" / (run + ".json")):
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
                last_ns = claimed.stat().st_mtime_ns
                if self.clock() - last_ns / 1e9 < self.stale_seconds:
                    continue
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
                          interrupted_claim_before_science=True)
            write_immutable(archive / "reclaimed.json", record)
            reclaimed.append(record)
        return reclaimed

    def _publish(self, claim: Claim, document: dict, success: bool) -> dict:
        candidate = claim.output / ("candidate_completed.json" if success else "failed.json")
        write_immutable(candidate, document)
        self.assert_owner(claim)
        destination = self.root / "publishing" / name(claim.task["run"])
        # Prepare recovery metadata before the ownership transition. A crash
        # after rename can then finish publication without scientific replay.
        try:
            write_immutable(claim.lease / "publication.json", dict(
                attempt_id=claim.token, success=success, candidate=descriptor(candidate),
            ), create_parents=False)
        except FileNotFoundError as exc:
            raise LeaseLost("Reclaim closed the publication lease") from exc
        try:
            rename(claim.lease, destination)
        except FileNotFoundError as exc:
            raise LeaseLost("Reclaim won before publication") from exc
        self._finish_publication(destination)
        return document

    def _finish_publication(self, directory: Path) -> None:
        publication = read_json(directory / "publication.json")
        candidate = publication["candidate"]
        actual = descriptor(Path(candidate["path"]))
        if actual != candidate:
            raise ValueError("Immutable candidate changed")
        value = read_json(Path(candidate["path"]))
        run = name(value["run"])
        target = (self.root / "results" / run / "completed.json"
                  if publication["success"] else self.root / "failed" / (run + ".json"))
        try:
            write_immutable(target, value)
        except FileExistsError:
            if read_json(target) != value:
                raise ValueError("A different terminal result already exists")
        if publication["success"]:
            done = dict(run=value["run"], attempt_id=publication["attempt_id"],
                        completed=descriptor(target))
            try:
                write_immutable(self.root / "done" / (run + ".json"), done)
            except FileExistsError:
                if read_json(self.root / "done" / (run + ".json")) != done:
                    raise ValueError("Done identity changed")

    def recover_publications(self) -> None:
        for directory in (self.root / "publishing").iterdir():
            if (directory / "publication.json").exists():
                self._finish_publication(directory)

    def complete(self, claim: Claim, result: dict) -> dict:
        if result.get("complete") is not True or result.get("run") != claim.task["run"]:
            raise ValueError("A genuine complete per-run scientific result is required")
        if result.get("worker_references_closed") is not True:
            raise ValueError("Scientific references are still open")
        return self._publish(claim, result, True)

    def fail(self, claim: Claim, traceback_text: str, owner: dict) -> dict:
        value = dict(complete=False, run=claim.task["run"], attempt_id=claim.token,
                     task=claim.task, actual_worker=owner, traceback=traceback_text,
                     failed_utc=utc())
        return self._publish(claim, value, False)

    def aggregate(self) -> dict:
        results, failures = {}, {}
        for path in sorted((self.root / "done").glob("*.json")):
            done = read_json(path)
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
