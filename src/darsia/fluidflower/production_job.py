"""Immutable manifests and local-first storage for native production workflows.

This module coordinates files and recovery. It never evaluates an image, changes
a calibration, promotes an incomplete frame, or grants production permission.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as distribution_metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping
import uuid

from .production_io import NasIO, normalize_locator

MANIFEST_FORMAT = "darsia-production-job-v1"
RESTART_FORMAT = "darsia-production-restart-v1"
PURPOSES = frozenset({"file_handling_acceptance", "profile", "equivalence", "production"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_identity(worktree: str | Path) -> dict:
    """Identify a clean Git checkout and initialized, unchanged submodules."""
    root = Path(worktree).resolve()
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Code root must be the worktree root")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Production and pilots require a clean worktree")
    # Preserve the status prefix: leading '-' means an uninitialized submodule.
    submodules = subprocess.run(["git", "-C", str(root), "submodule", "status", "--recursive"],
                                check=True, capture_output=True, text=True).stdout.rstrip("\r\n")
    if any(line and line[0] in "-+U" for line in submodules.splitlines()):
        raise ValueError("Submodules must be initialized at their recorded commits")
    return {"commit": git("rev-parse", "HEAD"), "clean_tree": True,
            "submodules": submodules.splitlines()}


def environment_lock(*, freeze_text: str | None = None) -> dict:
    """Record the actual interpreter and package inventory, without changing it."""
    supplied_inventory = freeze_text is not None
    packages = []
    if freeze_text is None:
        for distribution in distribution_metadata.distributions():
            packages.append({"name": distribution.metadata["Name"], "version": distribution.version,
                             "direct_url": distribution.read_text("direct_url.json")})
        packages.sort(key=lambda item: (item["name"].casefold(), item["version"]))
        freeze_text = "".join(f"{item['name']}=={item['version']}\n" for item in packages)
    freeze_text = freeze_text.replace("\r\n", "\n")
    return {"python_version": sys.version, "python_executable": sys.executable,
            "python_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "python_size_bytes": Path(sys.executable).stat().st_size,
            "inventory_provider": "test_supplied_inventory" if supplied_inventory else "importlib.metadata",
            "distribution_metadata": packages,
            "pip_freeze": freeze_text,
            "pip_freeze_sha256": hashlib.sha256(freeze_text.encode("utf-8")).hexdigest()}


def validate_manifest(document: Mapping) -> None:
    """Validate the single, direct manifest before any worker or array read."""
    if document.get("format") != MANIFEST_FORMAT or document.get("purpose") not in PURPOSES:
        raise ValueError("Unsupported production manifest or purpose")
    code = document.get("code") or {}
    if not re.fullmatch(r"[0-9a-f]{40}", code.get("commit", "")) or code.get("clean_tree") is not True:
        raise ValueError("Manifest needs the clean code commit")
    env = document.get("environment") or {}
    if (not env.get("python_version") or not env.get("python_executable") or
        not re.fullmatch(r"[0-9a-f]{64}", env.get("python_sha256", "")) or
        type(env.get("python_size_bytes")) is not int or env["python_size_bytes"] <= 0 or
        hashlib.sha256(env.get("pip_freeze", "").encode("utf-8")).hexdigest() != env.get("pip_freeze_sha256")):
        raise ValueError("Invalid environment lock")
    settings = document.get("settings") or {}
    if settings.get("precision") not in {"float32", "float64"}:
        raise ValueError("Each job has exactly one explicit precision")
    if settings.get("backend") not in {"cpu", "cuda", "opencl"}:
        raise ValueError("Unsupported backend")
    threads, workers, logical = (settings.get(k) for k in ("threads", "workers", "logical_cpus"))
    if any(type(x) is not int or x <= 0 for x in (threads, workers, logical)) or threads * workers > logical:
        raise ValueError("Workers times threads exceed logical CPUs")
    if not settings.get("host") or settings.get("backend") != "cpu" and not settings.get("device"):
        raise ValueError("GPU settings need an explicit device identity")
    for name in ("calibration", "specs", "inputs"):
        if not document.get(name):
            raise ValueError(f"Manifest lacks {name}")
    # Direct file descriptors only: never a descriptor that names another descriptor.
    assets = [document["calibration"], *document["specs"].values(), *document["inputs"].values()]
    for asset in assets:
        if (not isinstance(asset, Mapping) or set(asset) != {"path", "sha256", "size_bytes"} or
            not isinstance(asset.get("path"), str) or not asset["path"] or
            not re.fullmatch(r"[0-9a-f]{64}", asset.get("sha256", "")) or
            type(asset.get("size_bytes")) is not int or asset["size_bytes"] < 0):
            raise ValueError("Manifest file descriptors need path, SHA-256 and size")
    scientific = document.get("scientific_input_roles", list(document["inputs"]))
    assurance = document.get("assurance_input_roles", [])
    if (not isinstance(scientific, list) or not isinstance(assurance, list) or
        any(not isinstance(role, str) for role in [*scientific, *assurance]) or
        len(set(scientific)) != len(scientific) or len(set(assurance)) != len(assurance) or
        set(scientific) & set(assurance) or set(scientific) | set(assurance) != set(document["inputs"])):
        raise ValueError("Scientific and assurance roles must partition the flat input table")
    if not isinstance(document.get("created_utc"), str):
        raise ValueError("Exact UTC timestamps are strings")
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames or any(not isinstance(f, str) for f in frames) or len(set(frames)) != len(frames):
        raise ValueError("A job needs a unique, fixed frame order")
    for frame in frames:
        path = Path(frame)
        if path.is_absolute() or path.drive or not path.parts or any(part.startswith(".") for part in path.parts):
            raise ValueError("Manifest frame IDs must stay within the frame root")


def create_manifest(path: str | Path, *, worktree, settings, calibration, specs,
                    inputs, purpose, frames, environment=None, io=None,
                    scientific_input_roles=None, assurance_input_roles=()) -> dict:
    """Write the job manifest once. No worker is started by this operation."""
    io = io or NasIO()
    document = {"format": MANIFEST_FORMAT, "job_id": uuid.uuid4().hex,
                "created_utc": utc_now(), "purpose": purpose,
                "code": code_identity(worktree),
                "environment": environment or environment_lock(), "settings": dict(settings),
                "calibration": io.verified_asset(calibration),
                "specs": {k: io.verified_asset(v) for k, v in specs.items()},
                "inputs": {k: io.verified_asset(v) for k, v in inputs.items()},
                "scientific_input_roles": list(inputs) if scientific_input_roles is None else list(scientific_input_roles),
                "assurance_input_roles": list(assurance_input_roles),
                "frames": list(frames), "production_permission_granted": False}
    validate_manifest(document)
    io.write_json(path, document)
    return io.asset(path)


def plan_batches(*, available_bytes: int, input_bytes: int,
                 output_bytes_per_frame: int, frame_count: int) -> list[int]:
    """Keep a 20 percent output-space margin; return whole-frame batch sizes."""
    if any(type(x) is not int or x < 0 for x in (available_bytes, input_bytes, frame_count)):
        raise ValueError("Invalid disk-space estimate")
    if type(output_bytes_per_frame) is not int or output_bytes_per_frame <= 0:
        raise ValueError("Positive per-frame output estimate required")
    capacity = (available_bytes - input_bytes) // math.ceil(output_bytes_per_frame * 1.2)
    if frame_count and capacity < 1:
        raise OSError("Local disk cannot hold one frame with the 20 percent margin")
    batches = []
    while frame_count:
        count = min(capacity, frame_count)
        batches.append(count)
        frame_count -= count
    return batches


def worker_identity(pid: int | None = None) -> dict:
    """Store the process epoch as a string, never through a JSON float round trip."""
    import psutil
    process = psutil.Process(pid)
    return {"pid": process.pid, "process_create_time": str(process.create_time())}


def require_workers_retired(records) -> None:
    import psutil
    for record in records:
        if type(record.get("pid")) is not int or not isinstance(record.get("process_create_time"), str):
            raise ValueError("Recovery needs actual PID/start-time pairs")
        try:
            process = psutil.Process(record["pid"])
            if str(process.create_time()) == record["process_create_time"]:
                raise RuntimeError(f"Job worker has not exited: {record['pid']}")
        except psutil.NoSuchProcess:
            pass


class LocalProductionJob:
    """One immutable job, one frame commit format, and one recovery procedure."""
    def __init__(self, manifest_descriptor, local_root, nas_root, *, io=None):
        self.io = io or NasIO()
        self.manifest_asset = self.io.verified_asset(manifest_descriptor)
        manifest_bytes = self.io.read_bytes(self.manifest_asset["path"])
        if hashlib.sha256(manifest_bytes).hexdigest() != self.manifest_asset["sha256"]:
            raise ValueError("Manifest changed between verification and read")
        self.manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        validate_manifest(self.manifest)
        self.local_root, self.nas_root = Path(local_root), Path(nas_root)
        if normalize_locator(local_root).startswith("\\\\"):
            raise ValueError("Computation and controls require a local disk")
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.frames_root = self.local_root / "frames"
        self.controls_root = self.local_root / "ops"
        self.frames_root.mkdir(exist_ok=True)
        self.controls_root.mkdir(exist_ok=True)

    def assert_runtime(self, worktree, *, production_go=False, restart_descriptor=None) -> None:
        if self.manifest["purpose"] == "production" and production_go is not True:
            raise PermissionError("Olav's separate production GO is required")
        expected_commit = self.manifest["code"]["commit"]
        if restart_descriptor is not None:
            restart_asset = self.io.verified_asset(restart_descriptor)
            restart_bytes = self.io.read_bytes(restart_asset["path"])
            if hashlib.sha256(restart_bytes).hexdigest() != restart_asset["sha256"]:
                raise ValueError("Restart record changed during verification")
            restart = json.loads(restart_bytes)
            if restart.get("format") != RESTART_FORMAT or restart.get("manifest_sha256") != self.manifest_asset["sha256"] or restart.get("initial_commit") != expected_commit:
                raise ValueError("Restart record belongs to a different job")
            expected_commit = restart.get("fix_commit") or expected_commit
        if code_identity(worktree)["commit"] != expected_commit:
            raise ValueError("Runtime differs from the manifest commit")
        if environment_lock() != self.manifest["environment"]:
            raise ValueError("Runtime differs from the manifest environment lock")

    def stage_inputs(self) -> dict:
        cache = self.local_root / "inputs"
        cache.mkdir(exist_ok=True)
        result = {}
        for role, descriptor in self.manifest["inputs"].items():
            # Content names decouple identity from either machine's path spelling.
            target = cache / descriptor["sha256"] / Path(descriptor["path"]).name
            if target.exists():
                result[role] = self.io.asset(self.io.verify({**descriptor, "path": str(target)}))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                self.io.copy_file(descriptor, target)
                result[role] = self.io.asset(target)
        return result

    def check_local_space(self, *, output_bytes_per_frame: int) -> list[int]:
        return plan_batches(available_bytes=shutil.disk_usage(self.local_root).free,
            input_bytes=sum(d["size_bytes"] for d in self.manifest["inputs"].values()),
            output_bytes_per_frame=output_bytes_per_frame, frame_count=len(self.manifest["frames"]))

    def register_worker(self, identity, *, frame_id: str) -> dict:
        """The coordinator registers each real child before releasing its work."""
        self._frame_path(frame_id)
        actual = worker_identity(identity["pid"])
        if actual != identity:
            raise ValueError("Worker PID/start time changed before registration")
        record = {"worker": actual, "frame_id": frame_id,
                  "manifest_sha256": self.manifest_asset["sha256"], "registered_utc": utc_now()}
        key = hashlib.sha256((str(actual["pid"]) + ":" + actual["process_create_time"]).encode()).hexdigest()
        folder = self.controls_root / "workers"
        folder.mkdir(exist_ok=True)
        self.io.write_json(folder / f"{key}.json", record)
        return record

    def _frame_path(self, frame_id: str) -> Path:
        parts = Path(frame_id).parts
        if not parts or Path(frame_id).is_absolute() or Path(frame_id).drive or any(p in {"..", "."} or p.startswith(".") for p in parts):
            raise ValueError("Frame ID must be a relative, committed frame locator")
        if frame_id not in self.manifest["frames"]:
            raise ValueError("Frame is not selected by the job manifest")
        return self.frames_root / frame_id

    def begin_frame(self, frame_id: str) -> Path:
        final = self._frame_path(frame_id)
        if final.exists():
            raise FileExistsError("Verify/reuse committed frames; never recompute or overwrite")
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(f".{final.name}.{uuid.uuid4().hex}.partial")
        temporary.mkdir(exist_ok=False)
        return temporary

    def commit_frame(self, frame_id: str, temporary: Path, *, product_metadata: Mapping) -> dict:
        final = self._frame_path(frame_id)
        temporary = Path(temporary)
        if temporary.parent.resolve() != final.parent.resolve() or not temporary.name.startswith(f".{final.name}.") or not temporary.name.endswith(".partial"):
            raise ValueError("Frame temporary must be its owned local staging folder")
        if product_metadata.get("precision") != self.manifest["settings"]["precision"]:
            raise ValueError("A result keeps the precision it was computed in")
        files = {}
        for p in sorted(temporary.rglob("*")):
            if p.is_symlink():
                raise ValueError("Frame products cannot be symbolic links")
            if p.is_file():
                if p.name == "commit.json":
                    raise ValueError("A worker cannot supply a commit receipt")
                asset = self.io.asset(p)
                files[p.relative_to(temporary).as_posix()] = {k: asset[k] for k in ("sha256", "size_bytes")}
        if not files:
            raise ValueError("Cannot commit an empty frame")
        commit_id = uuid.uuid4().hex
        receipt = {"complete": True, "commit_id": commit_id, "frame_id": frame_id, "job_id": self.manifest["job_id"],
                   "manifest_sha256": self.manifest_asset["sha256"],
                   "settings": self.manifest["settings"], "products": dict(product_metadata),
                   "files": files, "committed_utc": utc_now()}
        self.io.write_json(temporary / "commit.json", receipt)
        inventory = {**files, "commit.json": self.io.asset(temporary / "commit.json")}
        # Register the exact receipt before publication. A crash here leaves only
        # an uncommitted partial; recovery never reads or promotes its contents.
        self.io.write_json(self.controls_root / f"frame_commit_{commit_id}.json", {
            "manifest_sha256": self.manifest_asset["sha256"], "frame_id": frame_id,
            "commit": inventory["commit.json"], "files": files})
        self.io.publish_directory(temporary, final, inventory)
        return receipt

    def verify_frame(self, frame_id: str) -> dict:
        folder = self._frame_path(frame_id)
        if folder.name.endswith(".partial"):
            raise ValueError("A partial is never a committed frame")
        receipt = self.io.read_json(folder / "commit.json")
        if (receipt.get("complete") is not True or receipt.get("frame_id") != frame_id or
            receipt.get("manifest_sha256") != self.manifest_asset["sha256"] or
            receipt.get("settings") != self.manifest["settings"]):
            raise ValueError("Committed frame has a different job/manifest/setting")
        commit_id = receipt.get("commit_id", "")
        if len(commit_id) != 32 or any(c not in "0123456789abcdef" for c in commit_id):
            raise ValueError("Committed frame lacks its registered commit ID")
        registration = self.io.read_json(self.controls_root / f"frame_commit_{commit_id}.json")
        if (registration.get("manifest_sha256") != self.manifest_asset["sha256"] or
            registration.get("frame_id") != frame_id or registration.get("files") != receipt["files"]):
            raise ValueError("Frame differs from its immutable commit registration")
        self.io.verify({**registration["commit"], "path": str(folder / "commit.json")})
        expected = set(receipt["files"]) | {"commit.json"}
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
        if actual != expected or any(p.is_symlink() for p in folder.rglob("*")):
            raise ValueError("Committed frame file inventory differs")
        for relative, descriptor in receipt["files"].items():
            p = folder / relative
            if p.resolve().is_relative_to(folder.resolve()) is not True:
                raise ValueError("Product escaped its frame folder")
            self.io.verify({**descriptor, "path": str(p)})
        return receipt

    def copy_frame_to_nas(self, frame_id: str) -> dict:
        receipt = self.verify_frame(frame_id)
        local = self._frame_path(frame_id)
        destination = self.nas_root / "frames" / frame_id
        inventory = {**receipt["files"], "commit.json": self.io.asset(local / "commit.json")}
        self.io.copy_file(self.manifest_asset, self.nas_root / "manifest.json")
        self.io.copy_directory(local, destination, inventory)
        # Retain local data. Deletion is a separate Olav-authorized operation.
        record = {"frame_id": frame_id, "local_retained": True,
                  "nas_directory": str(destination), "files": inventory,
                  "manifest_sha256": self.manifest_asset["sha256"], "verified_utc": utc_now()}
        path = self.controls_root / f"nas_copy_{receipt['commit_id']}.json"
        if path.exists():
            existing = self.io.read_json(path)
            if any(existing.get(key) != record[key] for key in ("frame_id", "nas_directory", "files", "manifest_sha256")):
                raise ValueError("NAS copy record differs from the existing verified record")
            return existing
        self.io.write_json(path, record)
        return record

    def restart(self, *, workers, stop, current_worktree, fix_commit=None,
                original_source_review=None, fix_source_review=None) -> dict:
        """One coordinator pass; reuse verified finals and ignore all partials."""
        registered = []
        for p in sorted((self.controls_root / "workers").glob("*.json")):
            value = self.io.read_json(p)
            if value.get("manifest_sha256") != self.manifest_asset["sha256"]:
                raise ValueError("Worker registration names another manifest")
            registered.append(value["worker"])
        identities = {(r["pid"], r["process_create_time"]): r for r in [*registered, *workers]}
        workers = list(identities.values())
        require_workers_retired(workers)
        current = code_identity(current_worktree)["commit"]
        original = self.manifest["code"]["commit"]
        if current != original and fix_commit != current:
            raise ValueError("Recovery code change must name its actual fix commit")
        reviews = {}
        if (original_source_review is None) != (fix_source_review is None):
            raise ValueError("Restart assurance requires both original and fix source reviews")
        if original_source_review is not None:
            if fix_commit != current:
                raise ValueError("Restart source review must name the actual fix commit")
            def identity(bound):
                return (normalize_locator(bound["path"]), bound["sha256"], bound["size_bytes"])
            original_asset = self.io.verified_asset(original_source_review)
            assurance = self.manifest.get("assurance_input_roles", [])
            if identity(original_asset) not in {
                    identity(self.manifest["inputs"][role]) for role in assurance}:
                raise ValueError("Original source review must be a manifest assurance input")
            for key, bound, commit in (("original_source_review", original_asset, original),
                                       ("fix_source_review", fix_source_review, current)):
                asset = self.io.verified_asset(bound)
                review = self.io.read_json(asset)
                if (review.get("contract") != "darsia-production-io-source-review-v1"
                        or review.get("complete") is not True
                        or review.get("io_only_diff_passed") is not True
                        or review.get("execution_commit") != commit):
                    raise ValueError("Restart needs the complete source review for its actual commit")
                reviews[key] = asset
        carried, unfinished = [], []
        for frame_id in self.manifest["frames"]:
            if self._frame_path(frame_id).exists():
                self.verify_frame(frame_id)
                carried.append(frame_id)
            else:
                unfinished.append(frame_id)
        record = {"format": RESTART_FORMAT, "created_utc": utc_now(), "stop": dict(stop),
                  "manifest_sha256": self.manifest_asset["sha256"], "workers_retired": list(workers),
                  "initial_commit": original, "fix_commit": fix_commit,
                  "carried_frames": carried, "recompute_frames": unfinished,
                  "partial_folders_read_or_modified": False}
        record.update(reviews)
        self.io.write_json(self.controls_root / f"restart_{uuid.uuid4().hex}.json", record)
        return record
