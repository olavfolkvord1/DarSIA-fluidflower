"""Transport-only queue for a frozen Studio mass campaign.

This module does not run mass calculations or publish Studio results.  A task is
one immutable file; a generation-specific directory is its exclusive SMB claim.
Only a coordinator may advance generations or accept verified attempt outputs.
All worker outputs stay below ``attempts/`` until a separate scientific
finalization gate is implemented.

The higher-level coordinator must verify all model, photo, phase-map, TOML and
code hashes on each participating host before creating the campaign. This
transport layer verifies the supplied model eligibility receipt but cannot
establish scientific correctness or backend parity. Only one coordinator may
call ``recover``, ``accept`` or ``reject_offer`` at a time. Host clocks must be
synchronized for expiry decisions; the grace period should exceed expected
clock skew and SMB delays.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA = 1
_RETRY_DELAYS = (0.1, 0.3, 0.7)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_BINDINGS = frozenset({
    "model_id", "model_sha256", "native_manifest_sha256",
    "phase_map_manifest_sha256", "concentration_toml_sha256", "method",
    "gas_policy", "code_sha256",
})


class QueueError(RuntimeError):
    """An invalid contract or shared-storage failure stops queue progress."""


class LostLease(QueueError):
    """An attempt has been superseded and cannot offer output."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exists(path: Path) -> bool:
    """Unlike Path.exists, propagate inaccessible SMB paths."""
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def _atomic_json(path: Path, value: Any) -> None:
    """Publish complete JSON via a same-directory rename; propagate I/O errors."""
    payload = _json_bytes(value)
    for retry in range(len(_RETRY_DELAYS) + 1):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return
        except OSError as exc:
            if retry == len(_RETRY_DELAYS):
                raise QueueError(f"Shared-storage write failed: {path}") from exc
            time.sleep(_RETRY_DELAYS[retry])
        finally:
            try:
                temporary.unlink()
            except (FileNotFoundError, OSError):
                pass


def _read_json(path: Path) -> Any:
    for retry in range(len(_RETRY_DELAYS) + 1):
        try:
            return json.loads(path.read_bytes())
        except OSError as exc:
            if retry == len(_RETRY_DELAYS):
                raise QueueError(f"Shared-storage read failed: {path}") from exc
            time.sleep(_RETRY_DELAYS[retry])
        except ValueError as exc:
            raise QueueError(f"Cannot read valid queue JSON: {path}") from exc
    raise AssertionError("Unreachable")


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must match {_SAFE_ID.pattern}")
    return value


def _sha_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_seconds(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise QueueError(f"Invalid {label}") from exc
    if not math.isfinite(number):
        raise QueueError(f"Invalid {label}")
    return number


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema") != SCHEMA:
        raise ValueError("Unsupported campaign schema")
    try:
        uuid.UUID(str(request["campaign_id"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("campaign_id must be a UUID") from exc
    if request.get("purpose") not in {"diagnostic", "production"}:
        raise ValueError("Campaign purpose must explicitly be diagnostic or production")
    bindings = request.get("bindings")
    if not isinstance(bindings, dict) or not _REQUIRED_BINDINGS <= bindings.keys():
        raise ValueError("Campaign is missing frozen model, input, method or code bindings")
    for key in _REQUIRED_BINDINGS | {"physical_profile_receipt_sha256",
                                      "image_selection_receipt_sha256"}:
        if key not in bindings or bindings[key] is None:
            continue
        if key.endswith("sha256"):
            _sha_value(bindings[key], key)
        elif not isinstance(bindings[key], str) or not bindings[key]:
            raise ValueError(f"Empty campaign binding: {key}")
    eligibility = request.get("model_eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError("A frozen model eligibility receipt is required")
    _sha_value(eligibility.get("receipt_sha256"), "eligibility receipt SHA")
    if not isinstance(eligibility.get("receipt_path"), str) or not eligibility["receipt_path"]:
        raise ValueError("Eligibility receipt path is required")
    for key in ("qa_status", "publication_scope"):
        if not isinstance(eligibility.get(key), str) or not eligibility[key]:
            raise ValueError(f"Eligibility {key} is required")
    if type(eligibility.get("deployment_allowed")) is not bool:
        raise ValueError("Eligibility deployment_allowed must be explicit")
    if (request["purpose"] == "production" and
            (eligibility["qa_status"] != "pass" or
             not eligibility["deployment_allowed"] or
             eligibility["publication_scope"] == "studio_validation_only")):
        raise ValueError("A QA-fail or validation-only phase model cannot enter production mass")
    attempts = request.get("max_attempts")
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError("max_attempts must be an integer in [1, 20]")
    tasks = request.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Campaign needs an ordered nonempty task list")
    seen_ids: set[str] = set()
    seen_work: set[tuple[Any, ...]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Task must be a JSON object")
        task_id = _safe_id(task.get("task_id"), "task_id")
        stage = task.get("stage")
        run = task.get("run")
        variant = task.get("variant")
        if stage not in {"prepare", "fit", "render"}:
            raise ValueError(f"Invalid task stage: {task_id}")
        if stage == "fit":
            if run is not None:
                raise ValueError("A shared fit task cannot own a run")
        else:
            _safe_id(run, "run")
        if variant is not None:
            _safe_id(variant, "variant")
        _sha_value(task.get("input_sha256"), "task input_sha256")
        indices = task.get("frame_indices")
        times = task.get("frame_times_h")
        if (not isinstance(indices, list) or not isinstance(times, list)
                or len(indices) != len(times) or (stage != "fit" and not indices)):
            raise ValueError(f"Missing frame coverage: {task_id}")
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError(f"Invalid frame index: {task_id}")
        if indices != sorted(set(indices)):
            raise ValueError(f"Frame order/uniqueness failed: {task_id}")
        if any(type(t) not in (int, float) or not float("-inf") < t < float("inf")
               for t in times):
            raise ValueError(f"Invalid frame time: {task_id}")
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError(f"Frame time order failed: {task_id}")
        key = (stage, run, variant)
        if task_id in seen_ids or key in seen_work:
            raise ValueError(f"Duplicate task or stage/run/variant: {task_id}")
        seen_ids.add(task_id)
        seen_work.add(key)


def _verify_eligibility(request: Mapping[str, Any]) -> None:
    """Re-read the external QA receipt; metadata alone is not authorization."""
    spec = request["model_eligibility"]
    path = Path(spec["receipt_path"])
    if _sha_file(path) != spec["receipt_sha256"]:
        raise QueueError("Model eligibility receipt SHA changed")
    receipt = _read_json(path)
    for key in ("qa_status", "publication_scope", "deployment_allowed"):
        if receipt.get(key) != spec[key]:
            raise QueueError(f"Model eligibility receipt differs in {key}")
    if (receipt.get("model_id") != request["bindings"]["model_id"] or
            receipt.get("model_sha256") != request["bindings"]["model_sha256"]):
        raise QueueError("Eligibility receipt belongs to another model")
    if request["purpose"] == "production" and receipt.get("scientific_acceptance") is not True:
        raise QueueError("Production mass requires scientific model acceptance")
    if (request["purpose"] == "production" and os.name == "nt"
            and not spec["receipt_path"].startswith("\\\\")):
        raise QueueError("Production eligibility receipt must be on a shared UNC path")


@dataclass(frozen=True)
class Lease:
    queue: "CampaignQueue"
    task: Mapping[str, Any]
    generation: int
    token: str
    worker_id: str
    host: str
    claim_dir: Path
    output_dir: Path
    heartbeat_error: list[BaseException] = field(default_factory=list, compare=False)

    @property
    def task_id(self) -> str:
        return str(self.task["task_id"])

    def assert_current(self) -> None:
        if self.heartbeat_error:
            raise LostLease("Shared-storage heartbeat failed") from self.heartbeat_error[0]
        self.queue._assert_campaign()
        state = self.queue._generation(self.task_id)
        if (state["generation"] != self.generation or state["status"] != "open"
                or _exists(self.queue.root / "accepted" / f"{self.task_id}.json")):
            raise LostLease(f"Superseded attempt: {self.task_id} generation {self.generation}")
        claim = _read_json(self.claim_dir / "claim.json")
        if claim["token"] != self.token:
            raise LostLease("Claim token changed")

    def heartbeat(self, *, lease_seconds: float = 120.0) -> None:
        self.assert_current()
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = time.time()
        entry = {"campaign_sha256": self.queue.request_sha256,
                 "task_id": self.task_id, "generation": self.generation,
                 "token": self.token, "worker_id": self.worker_id, "host": self.host,
                 "seen_at": now, "expires_at": now + lease_seconds}
        _atomic_json(self.claim_dir / "lease.json", entry)
        _atomic_json(self.queue.root / "workers" / f"{self.worker_id}.json", entry)
        self.assert_current()

    def submit(self, *, metadata: Mapping[str, Any] | None = None) -> Path:
        """Offer all attempt files; only coordinator acceptance is authoritative."""
        self.assert_current()
        root = self.output_dir.resolve(strict=True)
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_symlink():
                raise QueueError("Attempt output cannot contain symlinks")
            if not path.is_file():
                continue
            if not path.resolve(strict=True).is_relative_to(root):
                raise QueueError("Attempt output escapes its directory")
            files.append({"path": path.relative_to(self.output_dir).as_posix(),
                          "sha256": _sha_file(path), "size": path.stat().st_size})
        if not files:
            raise QueueError("Cannot offer an empty attempt output")
        receipt = {"schema": SCHEMA, "campaign_sha256": self.queue.request_sha256,
                   "task_id": self.task_id, "task_sha256": self.queue.task_sha256(self.task_id),
                   "generation": self.generation, "token": self.token,
                   "worker_id": self.worker_id, "host": self.host,
                   "output_dir": str(self.output_dir), "files": files,
                   "metadata": dict(metadata or {})}
        offer = self.queue.root / "offers" / self.task_id / f"g{self.generation:06d}.json"
        offer.parent.mkdir(parents=True, exist_ok=True)
        if _exists(offer):
            if _read_json(offer) != receipt:
                raise QueueError("A different offer already exists for this generation")
            return offer
        _atomic_json(offer, receipt)
        self.assert_current()
        return offer

    def fail(self, reason: str) -> Path:
        self.assert_current()
        if not reason:
            raise ValueError("Failure reason is required")
        path = self.queue.root / "failures" / self.task_id / f"g{self.generation:06d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"task_id": self.task_id, "generation": self.generation,
                   "token": self.token, "reason": reason}
        if _exists(path) and _read_json(path) != payload:
            raise QueueError("Failure record changed")
        if not _exists(path):
            _atomic_json(path, payload)
        return path


class CampaignQueue:
    def __init__(self, root: Path, request: dict[str, Any], request_sha256: str):
        self.root = Path(root)
        self.request = request
        self.request_sha256 = request_sha256
        self._tasks = {item["task_id"]: item for item in request["tasks"]}

    def _assert_campaign(self) -> None:
        if _sha_file(self.root / "campaign_request.json") != self.request_sha256:
            raise QueueError("Immutable campaign request SHA changed")
        ready = _read_json(self.root / "ready.json")
        if ready.get("campaign_sha256") != self.request_sha256:
            raise QueueError("Campaign ready binding changed")
        _verify_eligibility(self.request)

    @classmethod
    def create(cls, root: str | Path, request: Mapping[str, Any]) -> "CampaignQueue":
        _validate_request(request)
        root = Path(root)
        if (request["purpose"] == "production" and os.name == "nt"
                and not str(root).startswith("\\\\")):
            raise ValueError("Production queue root must be a shared UNC path")
        _verify_eligibility(request)
        root.mkdir(exist_ok=False)
        for name in ("tasks", "generations", "claims", "attempts", "offers",
                     "failures", "accepted", "workers", "control"):
            (root / name).mkdir()
        frozen = json.loads(_json_bytes(request))
        _atomic_json(root / "campaign_request.json", frozen)
        digest = _sha_file(root / "campaign_request.json")
        task_hashes = {}
        for task in frozen["tasks"]:
            task_id = task["task_id"]
            task_path = root / "tasks" / f"{task_id}.json"
            _atomic_json(task_path,
                         {"campaign_sha256": digest, "task": task})
            task_hashes[task_id] = _sha_file(task_path)
            _atomic_json(root / "generations" / f"{task_id}.json",
                         {"generation": 1, "status": "open", "history": []})
        _atomic_json(root / "ready.json", {"campaign_sha256": digest,
                                            "task_count": len(frozen["tasks"]),
                                            "task_sha256": task_hashes})
        return cls.open(root)

    @classmethod
    def open(cls, root: str | Path) -> "CampaignQueue":
        root = Path(root)
        request_path = root / "campaign_request.json"
        request = _read_json(request_path)
        if request_path.read_bytes() != _json_bytes(request):
            raise QueueError("Immutable campaign request bytes changed")
        _validate_request(request)
        _verify_eligibility(request)
        digest = _sha_file(request_path)
        ready = _read_json(root / "ready.json")
        if (ready.get("campaign_sha256") != digest or
                ready.get("task_count") != len(request["tasks"]) or
                set(ready.get("task_sha256", {})) !=
                {task["task_id"] for task in request["tasks"]}):
            raise QueueError("Campaign is incomplete or its request changed")
        queue = cls(root, request, digest)
        expected = {f"{task_id}.json" for task_id in queue._tasks}
        if {path.name for path in (root / "tasks").glob("*.json")} != expected:
            raise QueueError("Task file inventory differs from frozen request")
        for task_id in queue._tasks:
            if queue.task_sha256(task_id) != ready["task_sha256"][task_id]:
                raise QueueError(f"Immutable task file changed: {task_id}")
            queue._generation(task_id)
        return queue

    def task_sha256(self, task_id: str) -> str:
        task = self._tasks[task_id]
        path = self.root / "tasks" / f"{task_id}.json"
        actual = path.read_bytes()
        expected = _json_bytes({"campaign_sha256": self.request_sha256, "task": task})
        if actual != expected:
            raise QueueError(f"Immutable task changed: {task_id}")
        return _sha_bytes(actual)

    def _generation(self, task_id: str) -> dict[str, Any]:
        state = _read_json(self.root / "generations" / f"{task_id}.json")
        if (type(state.get("generation")) is not int or state["generation"] < 1
                or state.get("status") not in {"open", "failed"}
                or not isinstance(state.get("history"), list)):
            raise QueueError(f"Invalid generation state: {task_id}")
        return state

    def claim(self, *, worker_id: str, host: str | None = None,
              lease_seconds: float = 120.0) -> Lease | None:
        self._assert_campaign()
        worker_id = _safe_id(worker_id, "worker_id")
        host = _safe_id(host or socket.gethostname(), "host")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        for task_id, task in self._tasks.items():
            self.task_sha256(task_id)
            if _exists(self.root / "accepted" / f"{task_id}.json"):
                continue
            state = self._generation(task_id)
            if state["status"] != "open":
                continue
            generation = state["generation"]
            claim_dir = self.root / "claims" / task_id / f"g{generation:06d}"
            claim_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                claim_dir.mkdir()  # atomic exclusive claim on the shared volume
            except FileExistsError:
                continue
            token = uuid.uuid4().hex
            _atomic_json(claim_dir / "claim.json",
                         {"task_id": task_id, "generation": generation,
                          "token": token, "worker_id": worker_id, "host": host,
                          "claimed_at": time.time(), "lease_seconds": lease_seconds})
            output = self.root / "attempts" / task_id / f"g{generation:06d}-{token}"
            output.mkdir(parents=True, exist_ok=False)
            lease = Lease(self, task, generation, token, worker_id, host,
                          claim_dir, output)
            lease.heartbeat(lease_seconds=lease_seconds)
            return lease
        return None

    def set_worker_limit(self, host: str, count: int) -> None:
        host = _safe_id(host, "host")
        if type(count) is not int or count < 0:
            raise ValueError("Worker limit must be a nonnegative integer")
        _atomic_json(self.root / "control" / f"{host}.json",
                     {"campaign_sha256": self.request_sha256,
                      "host": host, "limit": count})

    def worker_limit(self, host: str) -> int:
        host = _safe_id(host, "host")
        value = _read_json(self.root / "control" / f"{host}.json")
        if (value.get("campaign_sha256") != self.request_sha256
                or value.get("host") != host or type(value.get("limit")) is not int
                or value["limit"] < 0):
            raise QueueError("Invalid host worker control")
        return value["limit"]

    def _advance(self, task_id: str, generation: int, reason: str) -> str:
        """Coordinator only. Supersede an attempt without touching its output."""
        state = self._generation(task_id)
        if state["generation"] != generation or state["status"] != "open":
            raise QueueError("Generation changed during recovery")
        history = [*state["history"], {"generation": generation,
                                       "reason": reason, "time": time.time()}]
        if generation >= self.request["max_attempts"]:
            updated = {"generation": generation, "status": "failed", "history": history}
        else:
            updated = {"generation": generation + 1, "status": "open", "history": history}
        _atomic_json(self.root / "generations" / f"{task_id}.json", updated)
        return updated["status"]

    def recover(self, *, grace_seconds: float = 30.0,
                now: float | None = None) -> list[dict[str, Any]]:
        """Coordinator only: retry reported failures or expired, absent workers."""
        self._assert_campaign()
        if grace_seconds < 0:
            raise ValueError("grace_seconds cannot be negative")
        now = time.time() if now is None else now
        changed = []
        for task_id in self._tasks:
            if _exists(self.root / "accepted" / f"{task_id}.json"):
                continue
            state = self._generation(task_id)
            if state["status"] != "open":
                continue
            generation = state["generation"]
            claim_dir = self.root / "claims" / task_id / f"g{generation:06d}"
            if not _exists(claim_dir):
                continue
            failure_path = self.root / "failures" / task_id / f"g{generation:06d}.json"
            if _exists(failure_path):
                failure = _read_json(failure_path)
                claim = _read_json(claim_dir / "claim.json")
                if (failure.get("token") != claim["token"]
                        or failure.get("generation") != generation):
                    raise QueueError("Failure record does not match current claim")
                reason = f"worker failure: {failure['reason']}"
            else:
                claim_path = claim_dir / "claim.json"
                if _exists(claim_path):
                    claim = _read_json(claim_path)
                    if (claim.get("task_id") != task_id or
                            claim.get("generation") != generation):
                        raise QueueError("Claim identity differs from current task")
                    lease_path = claim_dir / "lease.json"
                    if _exists(lease_path):
                        heartbeat = _read_json(lease_path)
                        if (heartbeat.get("task_id") != task_id or
                                heartbeat.get("generation") != generation or
                                heartbeat.get("token") != claim["token"]):
                            raise QueueError("Lease heartbeat identity changed")
                        expiry = _finite_seconds(heartbeat["expires_at"], "lease expiry")
                    else:
                        expiry = (_finite_seconds(claim["claimed_at"], "claim time") +
                                  _finite_seconds(claim["lease_seconds"], "lease length"))
                    worker_path = self.root / "workers" / f"{claim['worker_id']}.json"
                    if _exists(worker_path):
                        worker_heartbeat = _read_json(worker_path)
                        worker_expiry = (_finite_seconds(worker_heartbeat["expires_at"],
                                                         "worker heartbeat expiry")
                                         if worker_heartbeat.get("token") == claim["token"]
                                         else expiry)
                    else:
                        worker_expiry = expiry
                else:
                    # A crash between atomic mkdir and claim metadata is recoverable.
                    expiry = claim_dir.stat().st_mtime + 120.0
                    worker_expiry = expiry
                if now <= max(expiry, worker_expiry) + grace_seconds:
                    continue
                offer = self.root / "offers" / task_id / f"g{generation:06d}.json"
                if _exists(offer):
                    continue  # coordinator must accept or explicitly reject it
                reason = "lease and worker heartbeat expired"
            result = self._advance(task_id, generation, reason)
            changed.append({"task_id": task_id, "generation": generation,
                            "status": result, "reason": reason})
        return changed

    def accept(self, task_id: str) -> dict[str, Any]:
        """Coordinator only: verify one current offer and commit its receipt."""
        self._assert_campaign()
        task_id = _safe_id(task_id, "task_id")
        if task_id not in self._tasks:
            raise QueueError("Unknown task")
        accepted = self.root / "accepted" / f"{task_id}.json"
        if _exists(accepted):
            receipt = _read_json(accepted)
            if receipt.get("task_sha256") != self.task_sha256(task_id):
                raise QueueError("Accepted receipt lost its task binding")
            self._verify_files(receipt)
            return receipt
        state = self._generation(task_id)
        if state["status"] != "open":
            raise QueueError("Task has exhausted attempts")
        generation = state["generation"]
        claim_dir = self.root / "claims" / task_id / f"g{generation:06d}"
        claim = _read_json(claim_dir / "claim.json")
        offer = self.root / "offers" / task_id / f"g{generation:06d}.json"
        receipt = _read_json(offer)
        if (receipt.get("schema") != SCHEMA or receipt.get("campaign_sha256") != self.request_sha256
                or receipt.get("task_id") != task_id
                or receipt.get("task_sha256") != self.task_sha256(task_id)
                or receipt.get("generation") != generation
                or receipt.get("token") != claim["token"]
                or receipt.get("worker_id") != claim["worker_id"]
                or receipt.get("host") != claim["host"]):
            raise QueueError("Offer has stale or invalid campaign/task/lease identity")
        self._verify_files(receipt)
        if self._generation(task_id)["generation"] != generation:
            raise LostLease("Generation changed during acceptance")
        _atomic_json(accepted, receipt)
        return receipt

    def reject_offer(self, task_id: str, reason: str) -> str:
        """Coordinator only: reject current offer and fence it before retry."""
        self._assert_campaign()
        task_id = _safe_id(task_id, "task_id")
        if task_id not in self._tasks or not reason:
            raise ValueError("Known task and rejection reason are required")
        if _exists(self.root / "accepted" / f"{task_id}.json"):
            raise QueueError("Cannot reject an accepted task")
        state = self._generation(task_id)
        offer = self.root / "offers" / task_id / f"g{state['generation']:06d}.json"
        if not _exists(offer):
            raise QueueError("No current offer to reject")
        return self._advance(task_id, state["generation"], f"coordinator rejected: {reason}")

    def _verify_files(self, receipt: Mapping[str, Any]) -> None:
        task_id = receipt["task_id"]
        generation = receipt["generation"]
        token = receipt["token"]
        expected = self.root / "attempts" / task_id / f"g{generation:06d}-{token}"
        if Path(receipt["output_dir"]).resolve() != expected.resolve():
            raise QueueError("Receipt points outside its attempt directory")
        listed = receipt.get("files")
        if not isinstance(listed, list) or not listed:
            raise QueueError("Receipt has no output inventory")
        paths = [item["path"] for item in listed]
        if len(paths) != len(set(paths)):
            raise QueueError("Duplicate output path in receipt")
        entries = list(expected.rglob("*"))
        if any(path.is_symlink() for path in entries):
            raise QueueError("Attempt output contains symlinks")
        actual = {path.relative_to(expected).as_posix() for path in entries if path.is_file()}
        if actual != set(paths):
            raise QueueError("Attempt output inventory changed")
        for item in listed:
            path = expected / item["path"]
            if (path.is_symlink() or not path.resolve(strict=True).is_relative_to(expected.resolve())
                    or path.stat().st_size != item["size"]
                    or _sha_file(path) != _sha_value(item["sha256"], "output sha256")):
                raise QueueError(f"Attempt output changed: {path}")

    def status(self) -> dict[str, Any]:
        counts = {"pending": 0, "in_progress": 0, "accepted": 0, "failed": 0}
        for task_id in self._tasks:
            if _exists(self.root / "accepted" / f"{task_id}.json"):
                counts["accepted"] += 1
            else:
                state = self._generation(task_id)
                if state["status"] == "failed":
                    counts["failed"] += 1
                elif _exists(self.root / "claims" / task_id /
                             f"g{state['generation']:06d}"):
                    counts["in_progress"] += 1
                else:
                    counts["pending"] += 1
        return {"campaign_sha256": self.request_sha256, "counts": counts}


def run_worker(queue: CampaignQueue, execute: Callable[[Mapping[str, Any], Path], Mapping[str, Any]],
               *, host: str, slot: int, worker_id: str, lease_seconds: float = 120.0,
               pulse_seconds: float = 30.0, poll_seconds: float = 1.0,
               stop_when_idle: bool = False, max_tasks: int | None = None) -> int:
    """A host slot checks its live limit before each claim and drains current work."""
    if slot < 0 or pulse_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("Invalid worker slot or timing")
    completed = 0
    while max_tasks is None or completed < max_tasks:
        if slot >= queue.worker_limit(host):
            return completed
        lease = queue.claim(worker_id=worker_id, host=host, lease_seconds=lease_seconds)
        if lease is None:
            if stop_when_idle:
                return completed
            time.sleep(poll_seconds)
            continue
        if slot >= queue.worker_limit(host):
            lease.fail("host worker limit decreased before execution")
            return completed
        stop = threading.Event()

        def pulse() -> None:
            while not stop.wait(pulse_seconds):
                try:
                    lease.heartbeat(lease_seconds=lease_seconds)
                except BaseException as exc:  # propagated to main worker after execute
                    lease.heartbeat_error.append(exc)
                    return

        thread = threading.Thread(target=pulse, daemon=True)
        thread.start()
        try:
            metadata = execute(lease.task, lease.output_dir)
            if not isinstance(metadata, Mapping):
                raise QueueError("Worker execute must return receipt metadata")
            lease.submit(metadata=metadata)
            completed += 1
        except LostLease:
            raise
        except Exception as exc:
            if lease.heartbeat_error:
                raise QueueError("Heartbeat failed; attempt output is untrusted") from exc
            lease.fail(repr(exc))
        finally:
            stop.set()
            thread.join(timeout=max(1.0, pulse_seconds))
    return completed


def reconcile_host_slots(queue: CampaignQueue, host: str, running: dict[int, Any],
                         launch: Callable[[int], Any]) -> dict[int, Any]:
    """Spawn missing host slots; excess slots drain through ``run_worker``.

    ``launch`` returns a process-like object with ``poll()``. The caller owns
    its long-lived supervision loop and logs; no child is forcibly terminated.
    """
    limit = queue.worker_limit(host)
    for slot, child in list(running.items()):
        if child.poll() is not None:
            del running[slot]
    for slot in range(limit):
        if len(running) >= limit:
            break
        if slot not in running:
            running[slot] = launch(slot)
    return running
