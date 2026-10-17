"""Pure prospective host cache policy; no probing, mkdir or legacy migration.

Cache settings do not make results, models, reviews or user state disposable.
Admission accounting is not an OS filesystem quota. A numerical quota must be
configured from a separately accepted measurement; there is no default size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

PURPOSES = frozenset({"jobs", "gates", "tests", "clones", "display", "logs"})
PHYSICAL_RESERVE_BYTES = 40 * 1024**3


class CachePolicyError(ValueError):
    pass


def nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise CachePolicyError(label + " must be an explicit nonnegative integer")
    return value


def _absolute(value, *, allow_volume_root=False):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CachePolicyError("An explicit cache root is required")
    win = PureWindowsPath(value)
    path = win if win.drive or "\\" in value else PurePosixPath(value)
    if (not path.is_absolute() or ".." in path.parts
            or str(path).startswith(("\\\\", "//"))
            or (not allow_volume_root and path == type(path)(path.anchor))):
        raise CachePolicyError("Cache root must be a normalized local non-volume-root path")
    return path


def _descriptor(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))
            or not isinstance(value["path"], str) or not value["path"]
            or not 0 < nonnegative(value["size_bytes"], "Profile bytes") <= 16 * 1024**2):
        raise CachePolicyError("Cache profile requires exact bounded metadata path/SHA/size")
    return dict(value)


@dataclass(frozen=True)
class CachePolicy:
    root: str
    quota_bytes: int
    failed_retention_hours: int
    profiles: dict

    def purpose(self, purpose):
        if purpose not in PURPOSES:
            raise CachePolicyError("Unknown named cache purpose: " + str(purpose))
        return Path(self.root) / purpose


def cache_settings(document, *, required=False):
    """Validate metadata only. Actual filesystem locality is admitted later."""
    value = document.get("cache")
    if value is None:
        if required:
            raise CachePolicyError("Configure one measured [cache] policy before managed admission")
        return None
    if (not isinstance(value, dict) or set(value) - {
            "root", "quota_bytes", "failed_retention_hours", "profiles"}
            or not {"root", "quota_bytes", "profiles"} <= set(value)):
        raise CachePolicyError("Cache needs root, quota_bytes and measured profiles")
    root = _absolute(value["root"])
    quota = nonnegative(value["quota_bytes"], "Cache quota bytes")
    if quota == 0:
        raise CachePolicyError("Measure and configure the cache quota; zero is not admission")
    hours = value.get("failed_retention_hours", 48)
    if type(hours) is not int or not 0 < hours <= 48:
        raise CachePolicyError("Failed retention must be at most 48 hours")
    profiles = value["profiles"]
    if (not isinstance(profiles, dict) or not profiles or set(profiles) - PURPOSES):
        raise CachePolicyError("Bound measured profiles by named cache purpose")
    for blocks in document.get("stores", {}).values():
        for location in blocks.values():
            try:
                other = _absolute(location, allow_volume_root=True)
            except CachePolicyError:
                continue  # UNC/invalid names need actual filesystem/alias admission later.
            if type(root) is type(other) and (root.is_relative_to(other) or other.is_relative_to(root)):
                raise CachePolicyError("Managed cache must be disjoint from durable/input store roots")
    return CachePolicy(str(root), quota, hours,
                       {key: _descriptor(row) for key, row in profiles.items()})


def budget_report(policy, *, used_bytes, reserved_unwritten_bytes, request_bytes,
                  physical_free_bytes, required_noncache_bytes=0,
                  eligible_reclaimable_bytes=0):
    """Pure conservative accounting; reclaimable is never subtracted as free."""
    used = nonnegative(used_bytes, "Observed cache logical bytes")
    reserved = nonnegative(reserved_unwritten_bytes, "Active unwritten cache bytes")
    request = nonnegative(request_bytes, "Requested cache maximum bytes")
    noncache = nonnegative(required_noncache_bytes, "Separate noncache allocation bytes")
    free = nonnegative(physical_free_bytes, "Fresh physical free bytes")
    reclaimable = nonnegative(eligible_reclaimable_bytes, "Eligible logical bytes")
    headroom = policy.quota_bytes - used - reserved
    # Keep the measured free-space observation separate from logical quota.
    # Declared pending lengths supply a conservative allocation estimate, not
    # a proved bound on physical allocation, metadata, slack or other writers.
    needed = request + reserved + noncache + PHYSICAL_RESERVE_BYTES
    return dict(required_cache_bytes=request, current_cache_logical_bytes=used,
                active_cache_remaining_bytes=reserved, quota_bytes=policy.quota_bytes,
                quota_headroom_bytes=headroom, post_admission_quota_headroom_bytes=headroom-request,
                eligible_reclaimable_logical_bytes=reclaimable,
                required_noncache_bytes=noncache, reserve_bytes=PHYSICAL_RESERVE_BYTES,
                needed_physical_bytes=needed, free_physical_bytes=free,
                needed_physical_bytes_semantics="estimated pending allocation plus reserve; not a physical upper bound",
                expected_cache_allocation_bytes=request+reserved,
                physical_reserve_guard_satisfied=free >= needed,
                future_cache_physical_allocation_bound=None,
                sufficient=headroom >= request and free >= needed,
                size_semantics="logical file lengths; not physical reclaimed allocation",
                enforcement="admission reservation; not a filesystem hard cap",
                hard_filesystem_quota_verified=False)


def cache_status_line(machine, observation, previous=None):
    """Format an existing measurement; never invent a fresh total or delta."""
    total = nonnegative(observation["logical_bytes"], "Measured cache bytes")
    when = observation["measured_utc"]
    comparable = (previous is not None and previous.get("root") == observation.get("root")
                  and previous.get("semantics") == observation.get("semantics")
                  and previous.get("measured_utc") != when)
    delta = total - previous["logical_bytes"] if comparable else "UNMEASURED"
    largest = sorted(observation.get("purposes", []),
                     key=lambda row: nonnegative(row["logical_bytes"], "Purpose bytes"), reverse=True)[:3]
    return dict(machine=machine, root=observation.get("root"), logical_bytes=total,
                measured_utc=when, delta_bytes=delta, largest_three=largest,
                current_unmeasured_delta=not comparable, hard_filesystem_quota_verified=False)
