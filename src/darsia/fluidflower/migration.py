"""Create-only, hash-verified imports for FluidFlower projects.

The migration layer has deliberately narrow semantics: source files are read,
never renamed or modified, and a destination is published only after every
copied byte has been verified.  Existing destinations are always rejected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any
import uuid


COPY_PLAN_FORMAT = "darsia-fluidflower-copy-plan-v1"
COPY_MANIFEST_FORMAT = "darsia-fluidflower-copy-manifest-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_relative_path(value: str | Path) -> PurePosixPath:
    raw = str(value).replace("\\", "/").strip()
    relative = PurePosixPath(raw)
    if not raw or relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"Unsafe destination-relative path: {value!r}")
    if ":" in relative.parts[0]:
        raise ValueError(f"Destination path contains a drive prefix: {value!r}")
    return relative


def build_copy_plan(
    destination_root: Path,
    items: Sequence[Mapping[str, Any]],
    *,
    title: str,
    created_by: str = "",
) -> dict[str, Any]:
    """Freeze source identities for one new, never-overwritten destination."""

    destination = Path(destination_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    rows: list[dict[str, Any]] = []
    destinations: set[str] = set()
    for index, raw in enumerate(items, start=1):
        source = Path(str(raw.get("source") or "")).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Copy source {index} is unavailable: {source}")
        relative = _normalized_relative_path(
            str(raw.get("destination") or source.name)
        )
        destination_key = relative.as_posix().casefold()
        if destination_key in destinations:
            raise ValueError(f"Duplicate copy destination: {relative.as_posix()}")
        destinations.add(destination_key)
        stat = source.stat()
        rows.append(
            {
                "source": str(source),
                "destination": relative.as_posix(),
                "kind": str(raw.get("kind") or "file").strip(),
                "run_id": str(raw.get("run_id") or "").strip().casefold(),
                "size_bytes": int(stat.st_size),
                "source_modified_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(source),
            }
        )
    return {
        "schema": 1,
        "format": COPY_PLAN_FORMAT,
        "plan_id": uuid.uuid4().hex,
        "created_utc": _utc_now(),
        "created_by": str(created_by).strip(),
        "title": str(title).strip(),
        "destination_root": str(destination),
        "policy": {
            "source_access": "read-only",
            "destination": "must-not-exist",
            "overwrite": "forbidden",
            "publication": "atomic-after-sha256-verification",
        },
        "items": rows,
        "total_files": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
    }


def validate_copy_plan(
    plan: Mapping[str, Any],
    *,
    generated_files: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Validate a frozen plan without writing source or destination files."""

    errors: list[str] = []
    warnings: list[str] = []
    if plan.get("format") != COPY_PLAN_FORMAT or int(plan.get("schema") or 0) != 1:
        errors.append("Unsupported copy-plan format")
    destination = Path(str(plan.get("destination_root") or "")).expanduser().resolve()
    if destination.exists():
        errors.append(f"Destination already exists: {destination}")
    destinations: set[str] = set()
    for index, raw in enumerate(plan.get("items") or [], start=1):
        try:
            relative = _normalized_relative_path(str(raw.get("destination") or ""))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        key = relative.as_posix().casefold()
        if key in destinations:
            errors.append(f"Duplicate destination in row {index}: {relative}")
        destinations.add(key)
        source = Path(str(raw.get("source") or "")).expanduser().resolve()
        if not source.is_file():
            errors.append(f"Source is unavailable: {source}")
            continue
        stat = source.stat()
        if int(stat.st_size) != int(raw.get("size_bytes") or -1):
            errors.append(f"Source size changed after planning: {source}")
            continue
        if _sha256_file(source) != str(raw.get("sha256") or ""):
            errors.append(f"Source content changed after planning: {source}")
        elif int(stat.st_mtime_ns) != int(raw.get("source_modified_ns") or -1):
            warnings.append(f"Source timestamp changed but content did not: {source}")
    for raw_path, value in (generated_files or {}).items():
        try:
            relative = _normalized_relative_path(raw_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        key = relative.as_posix().casefold()
        if key in destinations:
            errors.append(f"Generated file collides with copied file: {relative}")
        destinations.add(key)
        if not isinstance(value, bytes):
            errors.append(f"Generated file must be bytes: {relative}")
    return {
        "valid": not errors,
        "destination_root": str(destination),
        "errors": errors,
        "warnings": warnings,
        "copy_files": len(plan.get("items") or []),
        "generated_files": len(generated_files or {}),
    }


def _safe_stage_path(stage: Path, relative: PurePosixPath) -> Path:
    candidate = stage.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(stage.resolve())
    except ValueError as exc:
        raise ValueError(f"Destination escapes staging root: {relative}") from exc
    return candidate


def execute_copy_plan(
    plan: Mapping[str, Any],
    *,
    generated_files: Mapping[str, bytes] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Materialize and atomically publish a new hash-verified copy package."""

    frozen = deepcopy(dict(plan))
    generated = dict(generated_files or {})
    validation = validate_copy_plan(frozen, generated_files=generated)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    if dry_run:
        return {"status": "dry-run", "plan": frozen, "validation": validation}

    destination = Path(frozen["destination_root"]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    stage = destination.with_name(
        f".{destination.name}.{frozen['plan_id'][:12]}.building"
    )
    if stage.exists():
        raise FileExistsError(f"Staging folder already exists: {stage}")

    copied_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    try:
        stage.mkdir(parents=False, exist_ok=False)
        for row in frozen.get("items") or []:
            source = Path(row["source"])
            if _sha256_file(source) != row["sha256"]:
                raise RuntimeError(f"Source changed during copy: {source}")
            relative = _normalized_relative_path(row["destination"])
            target = _safe_stage_path(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as read_stream, target.open("xb") as write_stream:
                shutil.copyfileobj(read_stream, write_stream, length=4 * 1024 * 1024)
                write_stream.flush()
                os.fsync(write_stream.fileno())
            shutil.copystat(source, target)
            copied_sha256 = _sha256_file(target)
            if copied_sha256 != row["sha256"]:
                raise RuntimeError(f"Copied file failed SHA-256 verification: {target}")
            if _sha256_file(source) != row["sha256"]:
                raise RuntimeError(f"Source changed while it was copied: {source}")
            copied_rows.append(
                {
                    **deepcopy(dict(row)),
                    "copied_path": relative.as_posix(),
                    "copied_sha256": copied_sha256,
                    "verified": True,
                }
            )
        for raw_path, value in generated.items():
            relative = _normalized_relative_path(raw_path)
            target = _safe_stage_path(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            digest = _sha256_bytes(value)
            if _sha256_file(target) != digest:
                raise RuntimeError(f"Generated file failed SHA-256 verification: {target}")
            generated_rows.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": len(value),
                    "sha256": digest,
                    "verified": True,
                }
            )
        manifest = {
            "schema": 1,
            "format": COPY_MANIFEST_FORMAT,
            "plan_id": frozen["plan_id"],
            "created_utc": frozen["created_utc"],
            "published_utc": _utc_now(),
            "created_by": frozen.get("created_by") or "",
            "title": frozen.get("title") or "",
            "destination_root": str(destination),
            "policy": deepcopy(frozen.get("policy") or {}),
            "copied_files": copied_rows,
            "generated_files": generated_rows,
        }
        manifest_path = stage / "migration_manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(
                (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, destination)
    except Exception:
        if stage.exists() and stage.parent.resolve() == destination.parent.resolve():
            shutil.rmtree(stage)
        raise
    return verify_copy_manifest(destination)


def verify_copy_manifest(root: Path) -> dict[str, Any]:
    """Verify every materialized destination file without requiring source access."""

    destination = Path(root).expanduser().resolve()
    manifest_path = destination / "migration_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if payload.get("format") != COPY_MANIFEST_FORMAT:
        raise ValueError(f"Unsupported copy manifest: {manifest_path}")
    errors: list[str] = []
    verified = 0
    rows = list(payload.get("copied_files") or []) + list(
        payload.get("generated_files") or []
    )
    for row in rows:
        relative = _normalized_relative_path(
            str(row.get("copied_path") or row.get("path") or "")
        )
        path = _safe_stage_path(destination, relative)
        expected = str(row.get("copied_sha256") or row.get("sha256") or "")
        if not path.is_file():
            errors.append(f"Missing copied file: {relative}")
            continue
        if _sha256_file(path) != expected:
            errors.append(f"Copied file changed: {relative}")
            continue
        verified += 1
    return {
        "status": "verified" if not errors else "invalid",
        "root": str(destination),
        "manifest": str(manifest_path),
        "verified_files": verified,
        "errors": errors,
        "valid": not errors,
    }


__all__ = [
    "COPY_MANIFEST_FORMAT",
    "COPY_PLAN_FORMAT",
    "build_copy_plan",
    "execute_copy_plan",
    "validate_copy_plan",
    "verify_copy_manifest",
]
