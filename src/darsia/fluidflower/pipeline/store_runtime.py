"""Resolve a role-bound run through this worker's local store map.

This module admits locations and exact input bytes only. It does not select
frames, alter parameter values, invoke a numerical producer or own a claim.
The shared task remains unchanged; physical paths exist in this process only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import stat

from ..job_filesystem import inspect_results_root
from ..store_map import resolve_input_reference, require_real_containment
from .job_queue import name
from .job_inputs import validate_catalog, _json_bytes, MAX_METADATA_BYTES


class RunUnavailable(ValueError):
    """This machine must leave the pending whole run for another worker."""


class InputHashMismatch(ValueError):
    """An owned, not-yet-started claim must be retired with this reason."""


def _version(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


@dataclass(frozen=True)
class ResolvedInput:
    reference: dict
    path: Path
    uses: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedInputs:
    """Admission evidence scoped to one run; never a replacement source SHA."""

    run: str
    rows: tuple[ResolvedInput, ...]
    versions: dict
    digests: dict

    def for_use(self, use):
        rows = [row for row in self.rows if use in row.uses]
        if len(rows) != 1:
            raise InputHashMismatch("Exactly one input is required for consumer " + use)
        return rows[0]

    def metadata(self, use):
        row = self.for_use(use)
        before = row.path.stat()
        if _version(before) != self.versions[str(row.path)]:
            raise InputHashMismatch("Verified input changed before metadata read: " + use)
        with row.path.open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise InputHashMismatch("Input metadata exceeds its size limit: " + use)
        if _version(row.path.stat()) != _version(before) or len(raw) != row.reference["bytes"]:
            raise InputHashMismatch("Verified input changed during metadata read: " + use)
        return _json_bytes(raw)


def resolve_catalog(task, stores, series_id):
    """Availability before claim: metadata/stat only, no payload hashes or writes."""
    catalog = validate_catalog(task["input_catalog"])
    if catalog["run"] != task["run"] or catalog["configuration"] != task["configuration"]:
        raise ValueError("Task and role catalogue identities differ")
    roots = stores.roots(series_id)
    rows = []
    references = [item["reference"] for item in catalog["inputs"]]
    for field in ("series", "series_manifest"):
        if task[field] not in references:
            raise ValueError("Role catalogue omits task input " + field)
    for item in catalog["inputs"]:
        reference = item["reference"]
        try:
            located = resolve_input_reference(reference, stores, series_id)
            path = require_real_containment(roots[reference["root_alias"]], located)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise RunUnavailable("Input is not a regular file")
            if info.st_size != reference["bytes"]:
                raise RunUnavailable("Input byte count differs")
        except (OSError, ValueError) as error:
            raise RunUnavailable("Input unavailable for " + ", ".join(item["uses"]) +
                                 ": " + str(error)) from error
        rows.append(ResolvedInput(dict(reference), path, tuple(item["uses"])))
    return tuple(rows)


def verify_inputs(task, rows):
    """Hash each distinct physical input once after claim, before any frame starts."""
    versions, digests = {}, {}
    for row in rows:
        key = str(row.path)
        if key not in digests:
            before = row.path.stat()
            digest = hashlib.sha256()
            with row.path.open("rb") as stream:
                while block := stream.read(8 * 1024 * 1024):
                    digest.update(block)
            after = row.path.stat()
            if _version(before) != _version(after):
                raise InputHashMismatch("Input changed during admission: " + key)
            versions[key], digests[key] = _version(after), digest.hexdigest()
        if (digests[key] != row.reference["sha256"]
                or versions[key][2] != row.reference["bytes"]):
            raise InputHashMismatch("Input SHA/bytes differ for " + ", ".join(row.uses))
    return VerifiedInputs(task["run"], tuple(rows), versions, digests)


def local_attempt_output(stores, series_id, job_id, run, attempt_id):
    """Admit the explicit local or SMB/NFS result root without creating it."""
    roots = stores.roots(series_id)
    if "results" not in roots:
        raise RunUnavailable("No explicit results root is configured")
    root = Path(roots["results"])
    evidence = inspect_results_root(root)
    if not evidence.allowed:
        raise RunUnavailable("Results root is not admitted: " +
                             evidence.reason + ": " + evidence.evidence)
    from .capacity import remote_results_owner
    remote_results_owner(stores,series_id,root,admission=evidence)
    import re

    if re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
        raise ValueError("A real queue attempt identity is required")
    output = root / name(job_id) / name(run) / ".attempts" / attempt_id
    parent = output
    while not parent.exists():
        if parent == parent.parent:
            raise RunUnavailable("No existing configured result-root ancestor")
        parent = parent.parent
    actual_root = root.resolve()
    if not output.resolve().is_relative_to(actual_root):
        raise RunUnavailable("Output path escapes its configured results root")
    if not output.resolve().is_relative_to(parent.resolve()):
        raise RunUnavailable("Output ancestor changed during location admission")
    return output, evidence
