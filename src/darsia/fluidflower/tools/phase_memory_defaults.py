"""Small machines keep the old memory profile: the memory-hungry speed-ups default to off there.

Three exact (bit-identical) execution variants trade memory for time:

* the pipelined final decision (``PHASE_PIPELINED_DECISION``) keeps a second photograph in flight and, for
  its self-check, a copy of the sequence;
* the proposal helper processes (``PHASE_PROPOSAL_PROCESSES``) hold up to five photographs' fields plus one
  working set per helper;
* the concurrent preparation (``PHASE_PREPARATION_THREADS``) corrects four 37-megapixel photographs at once.

Together they add roughly 6-10 GiB to an ordered run, and the batch's own memory guard budgets 8 GiB per
run. On the workstation (256 GiB) that does not matter; on a laptop it is the difference between running
and swapping. So on a machine with less than ``FULL_GIB`` of installed memory the three switches default to
their old values for the duration of the call. A switch that is set in the environment is never touched,
``PHASE_MEMORY_DEFAULTS=0`` leaves every default alone, and no value changes either way: the maps are the
same files, only slower to arrive.
"""
from __future__ import annotations

from contextlib import contextmanager
import os

VERSION = "memory-aware-defaults-v1"
ENV = "PHASE_MEMORY_DEFAULTS"
FULL_GIB = 96.0
LOW_MEMORY = {"PHASE_PIPELINED_DECISION": "0", "PHASE_PROPOSAL_PROCESSES": "0", "PHASE_PREPARATION_THREADS": "1"}


def total_memory_gib() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except Exception:                                   # noqa: BLE001 - unknown memory changes nothing
        return None


def decide(environ=None, total_gib: float | None = None) -> dict:
    """What would be set for a call on this machine, and why; changes nothing itself."""
    environ = os.environ if environ is None else environ
    report = dict(version=VERSION, all_speedups_from_gib=FULL_GIB, installed_gib=None, profile="unchanged",
                  set_for_this_call={}, kept_from_environment={})
    if str(environ.get(ENV, "1")).strip().lower() in {"0", "false", "no", "off"}:
        return dict(report, reason=f"{ENV} is off")
    total = total_memory_gib() if total_gib is None else float(total_gib)
    if total is None:
        return dict(report, reason="installed memory unknown")
    report["installed_gib"] = round(total, 1)
    if total >= FULL_GIB:
        return dict(report, profile="all-speedups")
    report["profile"] = "low-memory"
    for name, value in LOW_MEMORY.items():
        if str(environ.get(name, "")).strip():
            report["kept_from_environment"][name] = environ[name]
        else:
            report["set_for_this_call"][name] = value
    return report


@contextmanager
def scoped(environ=None, total_gib: float | None = None):
    """The low-memory defaults while the block runs; the environment is as before afterwards."""
    environ = os.environ if environ is None else environ
    report = decide(environ, total_gib)
    for name, value in report["set_for_this_call"].items():
        environ[name] = value
    try:
        yield report
    finally:
        for name in report["set_for_this_call"]:
            environ.pop(name, None)
