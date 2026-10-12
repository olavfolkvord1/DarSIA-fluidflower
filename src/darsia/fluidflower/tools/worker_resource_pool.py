"""Crash-recoverable host-local slots for calibration worker resources.

The distributed campaign queue lives on shared storage, while CPU context
building and accelerator ownership are host-local concerns.  Atomic directory
creation gives independent worker processes a small, portable semaphore without
introducing another service or sharing GPU objects between processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import time
import uuid
from typing import Any

import psutil


LEASE_SCHEMA = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_is_alive(pid: int, expected_created: float | None = None) -> bool:
    if pid <= 0 or not psutil.pid_exists(pid):
        return False
    try:
        process = psutil.Process(pid)
        if expected_created is not None and abs(
            float(process.create_time()) - float(expected_created)
        ) > 1e-3:
            return False
        return process.is_running()
    except (psutil.Error, OSError):
        return False


@dataclass
class LocalSlotLease:
    """One acquired slot which is released only by its creation token."""

    pool: "LocalSlotPool"
    index: int
    path: Path
    token: str
    wait_seconds: float
    acquired_monotonic: float
    released: bool = False

    @property
    def hold_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.acquired_monotonic)

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        lease_path = self.path / "lease.json"
        try:
            payload = json.loads(lease_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if str(payload.get("token", "")) != self.token:
            return
        try:
            lease_path.unlink(missing_ok=True)
            for temporary in self.path.glob(".*.tmp"):
                temporary.unlink(missing_ok=True)
            self.path.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # A later acquisition pass can reclaim an ownerless slot.  Never
            # remove a directory recursively here because ownership may have
            # changed after an interrupted release.
            pass

    def __enter__(self) -> "LocalSlotLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class LocalSlotPool:
    """A bounded host-local resource pool shared by independent processes."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str,
        slots: int,
        poll_seconds: float = 0.20,
        incomplete_lease_grace_seconds: float = 30.0,
    ) -> None:
        self.root = Path(root)
        self.name = str(name)
        self.slots = max(1, int(slots))
        self.poll_seconds = max(0.02, float(poll_seconds))
        self.incomplete_lease_grace_seconds = max(
            1.0, float(incomplete_lease_grace_seconds)
        )
        self.pool_dir = self.root / self.name
        self.stale_dir = self.root / "stale" / self.name
        self.pool_dir.mkdir(parents=True, exist_ok=True)

    def _slot_path(self, index: int) -> Path:
        return self.pool_dir / f"slot_{int(index):03d}.lock"

    def _archive_stale(self, path: Path, *, reason: str) -> bool:
        self.stale_dir.mkdir(parents=True, exist_ok=True)
        destination = self.stale_dir / (
            f"{path.name}.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
            f".{reason}.{uuid.uuid4().hex[:8]}"
        )
        try:
            os.replace(path, destination)
        except (FileNotFoundError, PermissionError, OSError):
            return False
        return True

    def _reclaim_if_stale(self, path: Path) -> bool:
        lease_path = path / "lease.json"
        try:
            payload = json.loads(lease_path.read_text(encoding="utf-8-sig"))
            pid = int(payload.get("pid", -1))
            raw_created = payload.get("process_created")
            process_created = (
                float(raw_created) if raw_created is not None else None
            )
        except FileNotFoundError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                return False
            if age < self.incomplete_lease_grace_seconds:
                return False
            return self._archive_stale(path, reason="incomplete")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return self._archive_stale(path, reason="unreadable")
        if _process_is_alive(pid, process_created):
            return False
        return self._archive_stale(path, reason="dead-owner")

    def _try_acquire(self, index: int, *, started: float) -> LocalSlotLease | None:
        path = self._slot_path(index)
        token = uuid.uuid4().hex
        try:
            path.mkdir()
        except FileExistsError:
            self._reclaim_if_stale(path)
            return None
        payload = {
            "schema": LEASE_SCHEMA,
            "pool": self.name,
            "slot": int(index),
            "token": token,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "process_created": psutil.Process(os.getpid()).create_time(),
            "acquired_utc": _utc_now(),
        }
        try:
            _atomic_json(path / "lease.json", payload)
        except Exception:
            try:
                path.rmdir()
            except OSError:
                pass
            raise
        acquired = time.perf_counter()
        return LocalSlotLease(
            pool=self,
            index=int(index),
            path=path,
            token=token,
            wait_seconds=max(0.0, acquired - started),
            acquired_monotonic=acquired,
        )

    def acquire(self, *, timeout_seconds: float | None = None) -> LocalSlotLease:
        started = time.perf_counter()
        start_index = os.getpid() % self.slots
        while True:
            for offset in range(self.slots):
                index = (start_index + offset) % self.slots
                lease = self._try_acquire(index, started=started)
                if lease is not None:
                    return lease
            if timeout_seconds is not None:
                elapsed = time.perf_counter() - started
                if elapsed >= float(timeout_seconds):
                    raise TimeoutError(
                        f"Timed out waiting for {self.name} after {elapsed:.2f}s"
                    )
            time.sleep(self.poll_seconds)

    def snapshot(self) -> dict[str, Any]:
        occupied: list[dict[str, Any]] = []
        for index in range(self.slots):
            path = self._slot_path(index)
            if not path.is_dir():
                continue
            try:
                lease = json.loads(
                    (path / "lease.json").read_text(encoding="utf-8-sig")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                lease = {"slot": index, "status": "unreadable"}
            occupied.append(lease)
        return {
            "name": self.name,
            "slots": self.slots,
            "occupied": occupied,
            "available": max(0, self.slots - len(occupied)),
            "root": str(self.root),
        }
