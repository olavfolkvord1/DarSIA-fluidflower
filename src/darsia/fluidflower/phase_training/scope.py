"""Explicit series isolation for training and prepared model validation.

Old bundles without a scope retain their historical cross-series behavior.
Series-specific models share algorithms, never another series' training rows.
"""
from __future__ import annotations

import json
from pathlib import Path


def series_id(value):
    value = str(value).strip().casefold().replace("_", "")
    if not value or not value.isalnum():
        raise ValueError("A nonempty experimental-series ID is required")
    return value


def requested_series(settings):
    mode = settings.get("model_scope", "shared")
    if mode not in {"shared", "single-series"}:
        raise ValueError(f"Unknown model scope: {mode}")
    if mode == "shared":
        return None
    selected = {series_id(value) for value in settings.get("series") or []}
    if len(selected) != 1:
        raise ValueError("A series-specific model requires exactly one experimental series")
    return next(iter(selected))


def partition_bundle(decisions, manifest, selected):
    """Return a derived subset; source dictionaries and annotations stay intact."""
    selected = series_id(selected)
    assignments = {}
    for job in manifest.get("jobs") or []:
        run = str(job.get("run") or "").casefold()
        dataset = series_id(job.get("dataset") or "")
        if run in assignments and assignments[run] != dataset:
            raise ValueError(f"Ambiguous series assignment for {run}")
        assignments[run] = dataset
    kept = {}
    for run, decision in (decisions.get("decisions") or {}).items():
        if str(run).casefold() not in assignments:
            raise ValueError(f"No source job identifies series for {run}")
        if assignments[str(run).casefold()] == selected:
            kept[run] = decision
    jobs = [job for job in manifest.get("jobs") or []
            if series_id(job["dataset"]) == selected]
    return dict(decisions, decisions=kept), dict(manifest, jobs=jobs)


def validate_training_series(sources, regions, selected):
    if selected is None:
        return
    if not sources or not regions:
        raise ValueError(f"No reviewed evidence for {selected}")
    actual = {series_id(row.dataset) for row in (*sources, *regions)}
    if actual != {series_id(selected)}:
        raise ValueError(f"Series isolation failed: expected {selected}, found {sorted(actual)}")


def validation_jobs(model_root, jobs):
    """Prepared campaigns for a scoped model contain only its declared series."""
    root = Path(model_root)
    scopes = []
    for head in ("co2_presence", "gas_within_co2"):
        path = root / "models/generic" / f"{head}_signal.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("training_provenance") or {}
        scopes.append(provenance.get("series_scope"))
    if not any(scopes):
        return jobs
    if scopes[0] != scopes[1] or not scopes[0]:
        raise ValueError("Phase heads have inconsistent experimental-series scope")
    selected = series_id(scopes[0])
    allowed = [job for job in jobs if series_id(job.get("dataset") or "") == selected]
    if not allowed:
        raise ValueError(f"This model is restricted to {selected}; no matching experiments were supplied")
    return allowed
