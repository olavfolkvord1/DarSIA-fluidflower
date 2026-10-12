"""Durable local cache for single-owner calibration jobs.

The distributed queue guarantees that one job has one active owner.  Optuna's
standard file journal nevertheless flushes and fsyncs every individual
parameter and attribute operation.  This module preserves JournalStorage's
operation stream while committing one local batch when a trial reaches a
terminal state.  Campaign artifacts are mirrored to shared storage only at
bounded checkpoints and are published atomically when the job completes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

from optuna.storages.journal import BaseJournalBackend
from optuna.trial import TrialState


JOURNAL_SCHEMA = 1
CACHE_SCHEMA = 2
DEFAULT_CHECKPOINT_TRIALS = 100
DEFAULT_CHECKPOINT_RETRY_SECONDS = 30.0
DEFAULT_ATOMIC_IO_RETRIES = 8
DEFAULT_ATOMIC_IO_RETRY_BASE_SECONDS = 0.25


_T = TypeVar("_T")


def _atomic_io_attempts() -> int:
    return max(
        1,
        int(
            os.environ.get(
                "FFAC_ATOMIC_IO_RETRIES",
                DEFAULT_ATOMIC_IO_RETRIES,
            )
        ),
    )


def _retry_atomic_write(destination: Path, operation: Callable[[], _T]) -> _T:
    """Retry an atomic write when a transient SMB outage drops its parent."""

    attempts = _atomic_io_attempts()
    for attempt in range(attempts):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            return operation()
        except OSError:
            if attempt + 1 >= attempts:
                raise
            delay = min(
                3.0,
                DEFAULT_ATOMIC_IO_RETRY_BASE_SECONDS * (2**attempt),
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _discard_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # The shared parent may still be unavailable. The dot-prefixed file is
        # never treated as a published artifact and can be overwritten later.
        pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one closed local file atomically and verify size and digest."""

    temporary = destination.with_name(
        f".{destination.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )

    def _copy_once() -> dict[str, Any]:
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as reader, temporary.open("wb") as writer:
                for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    writer.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)
        finally:
            _discard_temporary(temporary)
        if destination.stat().st_size != size:
            raise OSError(
                f"Shared-cache copy size mismatch for {destination}: "
                f"{destination.stat().st_size} != {size}"
            )
        return {
            "path": str(destination),
            "bytes": size,
            "sha256": digest.hexdigest(),
        }

    return _retry_atomic_write(destination, _copy_once)


def _atomic_copy_prefix(
    source: Path,
    destination: Path,
    byte_count: int,
) -> dict[str, Any]:
    """Copy an immutable prefix while the append-only source may keep growing."""

    expected = max(0, int(byte_count))
    temporary = destination.with_name(
        f".{destination.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )

    def _copy_once() -> dict[str, Any]:
        digest = hashlib.sha256()
        remaining = expected
        try:
            with source.open("rb") as reader, temporary.open("wb") as writer:
                while remaining:
                    chunk = reader.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise OSError(
                            f"Checkpoint source shrank while copying {source}: "
                            f"{expected - remaining} < {expected}"
                        )
                    writer.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)
        finally:
            _discard_temporary(temporary)
        if destination.stat().st_size != expected:
            raise OSError(
                f"Shared-cache copy size mismatch for {destination}: "
                f"{destination.stat().st_size} != {expected}"
            )
        return {
            "path": str(destination),
            "bytes": expected,
            "sha256": digest.hexdigest(),
        }

    return _retry_atomic_write(destination, _copy_once)


def _atomic_bytes(data: bytes, destination: Path) -> dict[str, Any]:
    temporary = destination.with_name(
        f".{destination.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )

    def _write_once() -> dict[str, Any]:
        try:
            with temporary.open("wb") as writer:
                writer.write(data)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)
        finally:
            _discard_temporary(temporary)
        return {
            "path": str(destination),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    return _retry_atomic_write(destination, _write_once)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )

    def _write_once() -> None:
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, indent=2, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            _discard_temporary(temporary)

    _retry_atomic_write(path, _write_once)


def _finished_state(log: Mapping[str, Any]) -> bool:
    if int(log.get("op_code", -1)) != 6:  # SET_TRIAL_STATE_VALUES
        return False
    try:
        return TrialState(int(log["state"])).is_finished()
    except (KeyError, TypeError, ValueError):
        return False


class BatchedJournalBackend(BaseJournalBackend):
    """Process-local Optuna journal with one fsync per completed trial."""

    def __init__(self, file_path: str | Path) -> None:
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._logs: list[dict[str, Any]] = []
        self._persisted_count = 0
        self._load()

    @property
    def log_count(self) -> int:
        with self._lock:
            return len(self._logs)

    @property
    def persisted_count(self) -> int:
        with self._lock:
            return self._persisted_count

    def _load(self) -> None:
        if not self.path.is_file():
            return
        data = self.path.read_bytes()
        last_newline = data.rfind(b"\n")
        complete = data[: last_newline + 1] if last_newline >= 0 else b""
        if len(complete) != len(data):
            # A killed process may leave one incomplete final batch.  No trial
            # in that batch was durably committed, so truncating it is safe.
            with self.path.open("r+b") as stream:
                stream.truncate(len(complete))
                stream.flush()
                os.fsync(stream.fileno())
        for line_number, raw_line in enumerate(complete.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid local Optuna journal batch at {self.path}:{line_number}"
                ) from exc
            if int(payload.get("schema", -1)) != JOURNAL_SCHEMA:
                raise ValueError(f"Unsupported journal schema in {self.path}")
            logs = payload.get("logs")
            if not isinstance(logs, list):
                raise ValueError(f"Journal batch contains no log list in {self.path}")
            expected_first = len(self._logs)
            if int(payload.get("first_log", -1)) != expected_first:
                raise ValueError(
                    f"Non-contiguous journal batch in {self.path}: "
                    f"{payload.get('first_log')} != {expected_first}"
                )
            canonical = _canonical_json(logs)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if digest != str(payload.get("sha256", "")):
                raise ValueError(f"Journal checksum mismatch in {self.path}")
            self._logs.extend(logs)
        self._persisted_count = len(self._logs)

    def read_logs(self, log_number_from: int) -> Iterable[dict[str, Any]]:
        with self._lock:
            return list(self._logs[max(0, int(log_number_from)) :])

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        if not logs:
            return
        with self._lock:
            self._logs.extend(dict(log) for log in logs)
            if any(_finished_state(log) for log in logs):
                self.checkpoint()

    def checkpoint(self) -> None:
        """Durably append every operation not included in an earlier batch."""

        with self._lock:
            if self._persisted_count >= len(self._logs):
                return
            first = self._persisted_count
            logs = self._logs[first:]
            canonical = _canonical_json(logs)
            payload = {
                "schema": JOURNAL_SCHEMA,
                "first_log": first,
                "log_count": len(logs),
                "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "logs": logs,
            }
            encoded = (
                json.dumps(payload, separators=(",", ":"), default=str) + "\n"
            ).encode("utf-8")
            with self.path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._persisted_count = len(self._logs)


class LocalJobCache:
    """Own local trial artifacts and publish bounded snapshots to a NAS root."""

    def __init__(
        self,
        *,
        campaign_id: str,
        output_root: Path,
        job_id: str,
        canonical_folder: Path,
        checkpoint_trials: int | None = None,
    ) -> None:
        configured_root = os.environ.get("FFAC_LOCAL_CAMPAIGN_CACHE_ROOT", "").strip()
        base = Path(
            configured_root
            or Path(
                os.environ.get("LOCALAPPDATA", str(Path.cwd() / "logs"))
            )
            / "ff_ac"
            / "campaign_cache"
        )
        namespace = hashlib.sha256(
            str(output_root).casefold().encode("utf-8")
        ).hexdigest()[:12]
        self.local_dir = base / campaign_id / namespace / job_id
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.local_dir / "trials.jsonl"
        self.best_path = self.local_dir / "best_live.json"
        self.journal_path = self.local_dir / "study_journal.batched.jsonl"
        self.state_path = self.local_dir / "cache_state.json"
        self.output_root = Path(output_root)
        self.campaign_id = str(campaign_id)
        self.job_id = str(job_id)
        self.canonical_folder = Path(canonical_folder)
        self.remote_dir = self.output_root / "cache_checkpoints" / job_id
        self.remote_manifest = self.remote_dir / "checkpoint.json"
        self.checkpoint_trials = max(
            1,
            int(
                checkpoint_trials
                if checkpoint_trials is not None
                else os.environ.get(
                    "FFAC_CACHE_CHECKPOINT_TRIALS", DEFAULT_CHECKPOINT_TRIALS
                )
            ),
        )
        self.last_checkpoint_completed = 0
        self._next_checkpoint_attempt_monotonic = 0.0
        self._checkpoint_executor: ThreadPoolExecutor | None = None
        self._checkpoint_future: Future[dict[str, Any]] | None = None
        self._checkpoint_lock = threading.RLock()
        self._async_checkpoints = os.environ.get(
            "FFAC_ASYNC_CHECKPOINTS", "1"
        ).strip().casefold() not in {"0", "false", "no", "off"}
        self._restore_best_available()
        self._last_checkpoint_scheduled = self.last_checkpoint_completed

    def _remote_artifact_path(
        self,
        artifact_name: str,
        metadata: Mapping[str, Any] | None,
        fallback_name: str,
    ) -> Path:
        if isinstance(metadata, Mapping):
            relative = str(metadata.get("relative_path", "")).strip()
            if relative:
                candidate = (self.remote_dir / relative).resolve()
                remote_root = self.remote_dir.resolve()
                if remote_root == candidate or remote_root in candidate.parents:
                    return candidate
                raise OSError(
                    f"Checkpoint artifact escapes remote root: {artifact_name}"
                )
        return self.remote_dir / fallback_name

    def _restore_best_available(self) -> None:
        remote_state: dict[str, Any] = {}
        if self.remote_manifest.is_file():
            try:
                remote_state = json.loads(
                    self.remote_manifest.read_text(encoding="utf-8-sig")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                remote_state = {}
        local_state: dict[str, Any] = {}
        if self.state_path.is_file():
            try:
                local_state = json.loads(
                    self.state_path.read_text(encoding="utf-8-sig")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                local_state = {}
        remote_completed = int(remote_state.get("completed_trials", 0) or 0)
        local_completed = int(local_state.get("completed_trials", 0) or 0)
        self.last_checkpoint_completed = max(remote_completed, local_completed)
        restore_remote_snapshot = remote_completed > local_completed
        remote_artifacts = remote_state.get("artifacts", {})
        candidates = (
            (
                "journal",
                "study_journal.batched.jsonl",
                self.journal_path,
            ),
            ("rows", "trials.jsonl", self.rows_path),
            ("best", "best_live.json", self.best_path),
        )
        for artifact_name, fallback_name, local in candidates:
            metadata = (
                remote_artifacts.get(artifact_name, {})
                if isinstance(remote_artifacts, dict)
                else {}
            )
            remote = self._remote_artifact_path(
                artifact_name,
                metadata,
                fallback_name,
            )
            if not remote.is_file():
                continue
            if local.is_file() and not restore_remote_snapshot:
                continue
            if isinstance(metadata, dict) and metadata:
                expected_size = int(metadata.get("bytes", -1))
                expected_digest = str(metadata.get("sha256", ""))
                if expected_size >= 0 and remote.stat().st_size != expected_size:
                    raise OSError(
                        f"Shared checkpoint size mismatch for {remote}: "
                        f"{remote.stat().st_size} != {expected_size}"
                    )
                if expected_digest and _sha256_file(remote) != expected_digest:
                    raise OSError(f"Shared checkpoint checksum mismatch for {remote}")
            local.parent.mkdir(parents=True, exist_ok=True)
            temporary = local.with_suffix(local.suffix + ".restore.tmp")
            shutil.copyfile(remote, temporary)
            os.replace(temporary, local)
        if restore_remote_snapshot:
            _atomic_json(self.state_path, remote_state)
        canonical_rows = self.canonical_folder / "trials.jsonl"
        if not self.rows_path.is_file() and canonical_rows.is_file():
            shutil.copyfile(canonical_rows, self.rows_path)
        canonical_best = self.canonical_folder / "best_live.json"
        if not self.best_path.is_file() and canonical_best.is_file():
            shutil.copyfile(canonical_best, self.best_path)

    def append_row(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(dict(payload), separators=(",", ":"), default=str) + "\n"
        with self.rows_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)

    def load_rows(self) -> list[dict[str, Any]]:
        if not self.rows_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.rows_path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    def load_best(self) -> dict[str, Any] | None:
        if not self.best_path.is_file():
            return None
        payload = json.loads(self.best_path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else None

    def write_best(self, payload: Mapping[str, Any]) -> None:
        _atomic_json(self.best_path, dict(payload))

    def _checkpoint_snapshot(
        self,
        *,
        completed_trials: int,
        journal: BatchedJournalBackend,
    ) -> dict[str, Any]:
        journal.checkpoint()
        artifacts: dict[str, dict[str, Any]] = {}
        for name, source in (
            ("journal", self.journal_path),
            ("rows", self.rows_path),
        ):
            if source.is_file():
                artifacts[name] = {
                    "source": source,
                    "bytes": source.stat().st_size,
                }
        if self.best_path.is_file():
            # best_live.json is atomically replaced, not append-only. Capturing
            # its small payload here keeps the queued checkpoint self-consistent.
            artifacts["best"] = {
                "source": self.best_path,
                "data": self.best_path.read_bytes(),
            }
        completed = int(completed_trials)
        generation_name = (
            f"{completed:08d}_{socket.gethostname()}_{os.getpid()}_"
            f"{time.time_ns()}"
        )
        return {
            "completed_trials": completed,
            "journal_log_count": journal.log_count,
            "generation_name": generation_name,
            "artifacts": artifacts,
        }

    def _publish_checkpoint_snapshot(
        self,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        completed = int(snapshot["completed_trials"])
        generation_name = str(snapshot["generation_name"])
        generation = self.remote_dir / "generations" / generation_name
        previous_generation: Path | None = None
        if self.remote_manifest.is_file():
            try:
                previous = json.loads(
                    self.remote_manifest.read_text(encoding="utf-8-sig")
                )
                previous_relative = str(
                    previous.get("generation_relative_path", "")
                ).strip()
                if previous_relative:
                    previous_generation = self.remote_dir / previous_relative
            except (OSError, ValueError, json.JSONDecodeError):
                previous_generation = None
        generation.mkdir(parents=True, exist_ok=False)
        artifacts: dict[str, Any] = {}
        raw_artifacts = snapshot.get("artifacts", {})
        if isinstance(raw_artifacts, Mapping):
            for name, raw in raw_artifacts.items():
                if not isinstance(raw, Mapping):
                    continue
                source = Path(raw["source"])
                destination = generation / source.name
                data = raw.get("data")
                if isinstance(data, bytes):
                    copied = _atomic_bytes(data, destination)
                else:
                    copied = _atomic_copy_prefix(
                        source,
                        destination,
                        int(raw["bytes"]),
                    )
                copied["relative_path"] = str(
                    Path("generations") / generation_name / source.name
                )
                artifacts[str(name)] = copied
        payload = {
            "schema": CACHE_SCHEMA,
            "campaign_id": self.campaign_id,
            "job_id": self.job_id,
            "completed_trials": completed,
            "checkpoint_trials": self.checkpoint_trials,
            "journal_log_count": int(snapshot["journal_log_count"]),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation_relative_path": str(
                Path("generations") / generation_name
            ),
            "artifacts": artifacts,
            "status": "complete",
            "periodic_mode": (
                "async-prefix-snapshot" if self._async_checkpoints else "synchronous"
            ),
        }
        _atomic_json(generation / "checkpoint.json", payload)
        # This pointer changes only after every immutable artifact is durable.
        _atomic_json(self.remote_manifest, payload)
        _atomic_json(self.state_path, payload)
        if (
            previous_generation is not None
            and previous_generation != generation
            and previous_generation.is_dir()
        ):
            try:
                remote_root = (self.remote_dir / "generations").resolve()
                resolved_previous = previous_generation.resolve()
                if remote_root in resolved_previous.parents:
                    shutil.rmtree(resolved_previous)
            except OSError:
                pass
        return payload

    def _checkpoint_failure(
        self,
        completed: int,
        exc: BaseException,
    ) -> dict[str, Any]:
        self._next_checkpoint_attempt_monotonic = (
            time.monotonic() + DEFAULT_CHECKPOINT_RETRY_SECONDS
        )
        self._last_checkpoint_scheduled = self.last_checkpoint_completed
        return {
            "schema": CACHE_SCHEMA,
            "status": "deferred",
            "completed_trials": int(completed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "retry_after_seconds": DEFAULT_CHECKPOINT_RETRY_SECONDS,
        }

    def wait_for_checkpoint(
        self,
        *,
        required: bool = False,
    ) -> dict[str, Any] | None:
        with self._checkpoint_lock:
            future = self._checkpoint_future
        if future is None:
            return None
        try:
            payload = future.result()
        except OSError as exc:
            with self._checkpoint_lock:
                self._checkpoint_future = None
                failed = self._checkpoint_failure(
                    self._last_checkpoint_scheduled,
                    exc,
                )
            if required:
                raise
            return failed
        except BaseException:
            with self._checkpoint_lock:
                self._checkpoint_future = None
                self._last_checkpoint_scheduled = self.last_checkpoint_completed
            raise
        with self._checkpoint_lock:
            self._checkpoint_future = None
            completed = int(payload["completed_trials"])
            self.last_checkpoint_completed = max(
                self.last_checkpoint_completed,
                completed,
            )
            self._last_checkpoint_scheduled = self.last_checkpoint_completed
            self._next_checkpoint_attempt_monotonic = 0.0
        return payload

    def _collect_finished_checkpoint(self) -> dict[str, Any] | None:
        with self._checkpoint_lock:
            future = self._checkpoint_future
        if future is None or not future.done():
            return None
        return self.wait_for_checkpoint(required=False)

    def checkpoint(
        self,
        *,
        completed_trials: int,
        journal: BatchedJournalBackend,
        force: bool = False,
        required: bool = False,
    ) -> dict[str, Any] | None:
        completed = int(completed_trials)
        self._collect_finished_checkpoint()
        now = time.monotonic()
        checkpoint_base = max(
            self.last_checkpoint_completed,
            self._last_checkpoint_scheduled,
        )
        if (
            not force
            and (
                completed - checkpoint_base < self.checkpoint_trials
                or now < self._next_checkpoint_attempt_monotonic
            )
        ):
            return None

        if required or not self._async_checkpoints:
            # A failed best-effort periodic upload must not suppress a fresh,
            # mandatory final publication attempt.
            self.wait_for_checkpoint(required=False)
            snapshot = self._checkpoint_snapshot(
                completed_trials=completed,
                journal=journal,
            )
            try:
                payload = self._publish_checkpoint_snapshot(snapshot)
            except OSError as exc:
                failed = self._checkpoint_failure(completed, exc)
                if required:
                    raise
                return failed
            self.last_checkpoint_completed = completed
            self._last_checkpoint_scheduled = completed
            self._next_checkpoint_attempt_monotonic = 0.0
            return payload

        with self._checkpoint_lock:
            if self._checkpoint_future is not None:
                return {
                    "schema": CACHE_SCHEMA,
                    "status": "busy",
                    "completed_trials": completed,
                    "scheduled_completed_trials": self._last_checkpoint_scheduled,
                }
            snapshot = self._checkpoint_snapshot(
                completed_trials=completed,
                journal=journal,
            )
            if self._checkpoint_executor is None:
                self._checkpoint_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="ffac-checkpoint",
                )
            self._checkpoint_future = self._checkpoint_executor.submit(
                self._publish_checkpoint_snapshot,
                snapshot,
            )
            self._last_checkpoint_scheduled = completed
        return {
            "schema": CACHE_SCHEMA,
            "status": "queued",
            "completed_trials": completed,
            "periodic_mode": "async-prefix-snapshot",
        }

    def publish_final(
        self,
        *,
        completed_trials: int,
        journal: BatchedJournalBackend,
    ) -> dict[str, Any]:
        checkpoint = self.checkpoint(
            completed_trials=completed_trials,
            journal=journal,
            force=True,
            required=True,
        ) or {}
        published: dict[str, Any] = {}
        for name, source, destination in (
            (
                "trials",
                self.rows_path,
                self.canonical_folder / "trials.jsonl",
            ),
            (
                "journal",
                self.journal_path,
                self.canonical_folder / "study_journal.batched.jsonl",
            ),
            (
                "best",
                self.best_path,
                self.canonical_folder / "best_live.json",
            ),
        ):
            if source.is_file():
                published[name] = _atomic_copy(source, destination)
        provenance = {
            "schema": CACHE_SCHEMA,
            "storage_contract": "local-batched-journal-generation-v3-async",
            "completed_trials": int(completed_trials),
            "checkpoint": checkpoint,
            "published": published,
            "periodic_checkpoint_mode": (
                "async-prefix-snapshot" if self._async_checkpoints else "synchronous"
            ),
            "local_cache_deleted_after_publish": True,
            "host": socket.gethostname(),
            "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(self.canonical_folder / "storage_provenance.json", provenance)
        self.close(wait=True)
        shutil.rmtree(self.local_dir)
        return provenance

    def close(self, *, wait: bool = True) -> dict[str, Any] | None:
        payload = self.wait_for_checkpoint(required=False) if wait else None
        with self._checkpoint_lock:
            executor = self._checkpoint_executor
            self._checkpoint_executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=not wait)
        return payload

    def local_provenance(self) -> dict[str, Any]:
        return {
            "storage_contract": "local-batched-journal-generation-v3-async",
            "local_dir": str(self.local_dir),
            "checkpoint_trials": self.checkpoint_trials,
            "last_checkpoint_completed": self.last_checkpoint_completed,
            "last_checkpoint_scheduled": self._last_checkpoint_scheduled,
            "periodic_checkpoint_mode": (
                "async-prefix-snapshot" if self._async_checkpoints else "synchronous"
            ),
            "journal_sha256": (
                _sha256_file(self.journal_path) if self.journal_path.is_file() else None
            ),
        }
