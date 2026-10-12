"""Measured worker admission for an immutable FF_AC native-image campaign.

This module does not start work.  A caller runs the *same source-bound pilot*
in separate output roots at each admitted concurrency, verifies the native
payload hashes, and records the observations below.  RAM headroom alone is
not a reason to increase concurrency when NAS or serial work limits rate.
"""

from __future__ import annotations

from dataclasses import dataclass


STEPS = (8, 12, 16, 24)


@dataclass(frozen=True)
class PilotObservation:
    workers: int
    workload_sha256: str
    payload_sha256: str
    frames: int
    wall_seconds: float
    peak_worker_rss_bytes: int
    peak_worker_temporary_bytes: int
    failures: int = 0

    @property
    def frames_per_second(self) -> float:
        return self.frames / self.wall_seconds


@dataclass(frozen=True)
class Admission:
    next_workers: int | None
    reason: str
    observed_gain: float | None
    projected_ram_gb: float | None
    projected_temporary_gb: float | None


def next_pilot(
    observations: list[PilotObservation], *, available_ram_gb: float,
    free_output_gb: float, logical_cpus: int, memory_reserve_gb: float = 24.0,
    disk_reserve_gb: float = 50.0, worker_headroom: float = 1.35,
    minimum_gain: float = 1.05,
) -> Admission:
    """Admit only the next 8→12→16→24 pilot after parity and rate checks.

    ``available_ram_gb`` and ``free_output_gb`` are measured with earlier
    pilots stopped. The per-worker peaks are measured values, inflated by
    ``worker_headroom``. Reusing this decision for a different source cohort
    or after a hardware change requires a fresh pilot.
    """
    if (available_ram_gb <= 0 or free_output_gb <= 0 or logical_cpus < 1
            or memory_reserve_gb < 0 or disk_reserve_gb < 0
            or worker_headroom < 1 or minimum_gain < 1):
        raise ValueError("Invalid resource-admission limits")
    if len(observations) > len(STEPS):
        raise ValueError("Too many pilot observations")
    if any(obs.workers != STEPS[index] for index, obs in enumerate(observations)):
        raise ValueError("Pilots must follow 8, 12, 16, 24 without skipping")
    for obs in observations:
        if (len(obs.workload_sha256) != 64 or len(obs.payload_sha256) != 64
                or obs.frames <= 0 or obs.wall_seconds <= 0
                or obs.peak_worker_rss_bytes <= 0
                or obs.peak_worker_temporary_bytes < 0
                or obs.failures < 0):
            raise ValueError("Invalid measured pilot observation")
    if observations:
        first = observations[0]
        if any((obs.workload_sha256 != first.workload_sha256
                or obs.payload_sha256 != first.payload_sha256
                or obs.frames != first.frames) for obs in observations):
            return Admission(None, "Native pilot source or payload parity changed", None, None, None)
        if any(obs.failures for obs in observations):
            return Admission(None, "Native pilot failed; no automatic retry", None, None, None)
    if len(observations) == len(STEPS):
        return Admission(None, "24-worker pilot complete", None, None, None)

    gain = None
    if len(observations) >= 2:
        gain = (observations[-1].frames_per_second /
                observations[-2].frames_per_second)
        if gain < minimum_gain:
            return Admission(None, "Measured throughput did not justify more workers",
                             gain, None, None)

    candidate = STEPS[len(observations)]
    if candidate > logical_cpus:
        return Admission(None, "Candidate exceeds logical CPU count", gain, None, None)
    if not observations:
        # A renderer-specific bound is still needed before the first pilot.
        return Admission(candidate, "Run the first exact-source 8-worker pilot",
                         None, None, None)
    peak_rss = max(obs.peak_worker_rss_bytes for obs in observations)
    peak_temp = max(obs.peak_worker_temporary_bytes for obs in observations)
    projected_ram = memory_reserve_gb + candidate * peak_rss * worker_headroom / 2**30
    projected_temp = disk_reserve_gb + candidate * peak_temp * worker_headroom / 2**30
    if available_ram_gb < projected_ram:
        return Admission(None, "Insufficient measured RAM headroom", gain,
                         projected_ram, projected_temp)
    if free_output_gb < projected_temp:
        return Admission(None, "Insufficient measured output-disk headroom", gain,
                         projected_ram, projected_temp)
    return Admission(candidate, "Admit next exact-source pilot; measure before scaling again",
                     gain, projected_ram, projected_temp)
