"""Atomic, byte-preserving local source cache for the concentration movies.

Image identity remains its original manifest path; configured storage aliases
select the authoritative location used for byte transport.
No image re-encoding, metadata rewriting, or missing-frame substitution occurs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from PIL import Image

from darsia.fluidflower.storage_paths import resolve_storage_path

INPUT_POLICY = "exact-jpeg-bytes-verified-unc-atomic-local-cache-v1"


def source_candidates(source_path):
    # A configured move is authoritative: unavailable destinations must not
    # silently fall back to an old copy on the former drive.
    return [resolve_storage_path(source_path)]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _copy_exact(source, temporary):
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as incoming, temporary.open("xb") as outgoing:
        before = os.fstat(incoming.fileno())
        while chunk := incoming.read(2 * 1024 * 1024):
            outgoing.write(chunk)
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(incoming.fileno())
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if total != before.st_size or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise OSError(f"Source changed or copy was incomplete: {source}")
    with Image.open(temporary) as image:
        # Camera .JPG files may be MPO JPEG containers with an embedded
        # auxiliary image. Preserve the entire container and its EXIF bytes.
        if image.format not in {"JPEG", "MPO"}:
            raise ValueError(f"Movie source is not a JPEG/MPO container: {source}")
        image.verify()
    return digest.hexdigest(), total, before.st_mtime_ns


class MovieInputCache:
    def __init__(self, root, run, retry_delays=(1., 3., 10., 30., 60.)):
        self.root = Path(root) / "inputs" / run
        self.retry_delays = tuple(retry_delays)

    def stage(self, source_path):
        candidates = source_candidates(source_path)
        identity = str(candidates[0]).replace("\\", "/").casefold()
        key = hashlib.sha256(identity.encode()).hexdigest()[:24]
        directory = self.root / key
        target = directory / candidates[0].name
        receipt_path = directory / "source.json"
        if target.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["source_identity"] != identity or receipt["input_policy"] != INPUT_POLICY:
                raise ValueError(f"Movie input cache identity changed: {receipt_path}")
            if target.stat().st_size != receipt["source_bytes"] or file_sha256(target) != receipt["source_sha256"]:
                raise ValueError(f"Movie input cache checksum mismatch: {target}")
            return target, {**receipt, "input_cache_hit": True}

        directory.mkdir(parents=True, exist_ok=True)
        failures = []
        for attempt in range(len(self.retry_delays) + 1):
            if attempt:
                time.sleep(self.retry_delays[attempt - 1])
            for source in candidates:
                suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
                temporary = target.with_name(target.name + suffix)
                receipt_temporary = receipt_path.with_name(receipt_path.name + suffix)
                try:
                    digest, total, modified = _copy_exact(source, temporary)
                    if receipt_path.exists():
                        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                        if existing["source_identity"] != identity or existing["source_sha256"] != digest:
                            raise ValueError(f"Conflicting immutable source bytes: {target}")
                    receipt = {"input_policy": INPUT_POLICY, "source_identity": identity,
                        "original_source_path": str(source_path), "source_transport_path": str(source),
                        "source_sha256": digest, "source_bytes": total, "source_mtime_ns": modified,
                        "local_source_path": str(target), "input_cache_receipt": str(receipt_path),
                        "input_copy_attempt": attempt + 1, "input_read_failures": failures}
                    receipt_temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                    temporary.replace(target)
                    receipt_temporary.replace(receipt_path)
                    return target, {**receipt, "input_cache_hit": False}
                except OSError as error:
                    failures.append({"attempt": attempt + 1, "path": str(source), "error": repr(error)})
                finally:
                    temporary.unlink(missing_ok=True)
                    receipt_temporary.unlink(missing_ok=True)
        raise OSError(f"Unable to stage movie source after {len(self.retry_delays)+1} attempts: "
                      f"{source_path}; last error: {failures[-1]['error']}")
