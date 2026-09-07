"""Read-only discovery and provenance summaries for optical phase models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from darsia.fluidflower.phase_learning import inspect_phase_model_bundle


PHASE_MODEL_HISTORY_FORMAT = "darsia-fluidflower-phase-model-history-v1"


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected one JSON object in {path}")
    return dict(payload)


def _model_directories(root: Path, *, max_depth: int) -> list[Path]:
    """Find model-version folders without descending into their large assets."""

    discovered: list[Path] = []
    pending: list[tuple[Path, int]] = [(root, 0)]
    ignored = {
        "input_snapshot",
        "models",
        "result_staging",
        "cache",
        ".git",
        "postflight",
        "native_phase_maps",
        "manual_pairwise_review",
        "frames",
        "movies",
    }
    while pending:
        folder, depth = pending.pop()
        try:
            names = {entry.name for entry in os.scandir(folder)}
        except OSError:
            continue
        if "summary.json" in names or "training_request.json" in names:
            discovered.append(folder)
            continue
        if depth >= max_depth:
            continue
        if depth > 0 and not any(
            "model" in part.casefold() for part in (folder.name, folder.parent.name)
        ):
            continue
        try:
            children = [
                Path(entry.path)
                for entry in os.scandir(folder)
                if entry.is_dir(follow_symlinks=False) and entry.name not in ignored
            ]
        except OSError:
            continue
        pending.extend((child, depth + 1) for child in children)
    return discovered


def _head_history(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = summary.get("models") or {}
    if not isinstance(models, Mapping):
        return rows
    for adapter, raw_model in models.items():
        if not isinstance(raw_model, Mapping):
            continue
        heads = raw_model.get("heads") or {}
        if not isinstance(heads, Mapping):
            continue
        for head, raw_head in heads.items():
            if not isinstance(raw_head, Mapping):
                continue
            final_fit = raw_head.get("final_fit") or {}
            if not isinstance(final_fit, Mapping):
                final_fit = {}
            rows.append(
                {
                    "adapter": str(adapter),
                    "head": str(head),
                    "balanced_accuracy": raw_head.get(
                        "oof_threshold_balanced_accuracy"
                    ),
                    "threshold": raw_head.get("oof_threshold"),
                    "regularization": raw_head.get("selected_regularization"),
                    "deployment_allowed": bool(
                        raw_head.get("deployment_allowed", False)
                    ),
                    "iterations": int(final_fit.get("iterations") or 0),
                    "converged": bool(final_fit.get("converged", False)),
                    "backend": str(final_fit.get("backend") or ""),
                    "precision": str(final_fit.get("precision") or ""),
                    "objective": final_fit.get("objective"),
                    "model_path": str(raw_head.get("model_path") or ""),
                    "model_sha256": str(raw_head.get("model_sha256") or ""),
                }
            )
    return rows


def _primary_metric(heads: Sequence[Mapping[str, Any]], head: str) -> float | None:
    matches = [
        row
        for row in heads
        if row.get("adapter") == "generic" and row.get("head") == head
    ]
    if not matches:
        matches = [row for row in heads if row.get("head") == head]
    value = matches[0].get("balanced_accuracy") if matches else None
    return float(value) if isinstance(value, (int, float)) else None


def _created_timestamp(value: object, fallback: float) -> float:
    text = str(value or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return fallback


def inspect_phase_model_history_entry(root: Path) -> dict[str, Any]:
    """Summarize one immutable training attempt without changing any files."""

    model_root = Path(root).expanduser().resolve()
    request_path = model_root / "training_request.json"
    summary_path = model_root / "summary.json"
    request: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    errors: list[str] = []
    if request_path.is_file():
        try:
            request = _read_mapping(request_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Training request: {type(exc).__name__}: {exc}")
    if summary_path.is_file():
        try:
            summary = _read_mapping(summary_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Summary: {type(exc).__name__}: {exc}")

    bundle: dict[str, Any] = {}
    if summary:
        try:
            bundle = inspect_phase_model_bundle(model_root)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Model bundle: {type(exc).__name__}: {exc}")

    inventory = summary.get("inventory") or {}
    if not isinstance(inventory, Mapping):
        inventory = {}
    heads = _head_history(summary)
    production_heads = [
        row for row in heads if "no-darsia-change" not in str(row.get("adapter"))
    ]
    internal_holdout_passed = bool(production_heads) and all(
        bool(row.get("deployment_allowed")) for row in production_heads
    )
    deployment = summary.get("deployment")
    human_validation = summary.get("human_validation")
    legacy = not bool(request)
    if errors:
        state = "invalid"
        state_label = "Invalid or incomplete output"
    elif not summary:
        state = "training_incomplete"
        state_label = "Training incomplete"
    elif legacy:
        state = "historical"
        state_label = "Historical model (read-only)"
    elif not internal_holdout_passed:
        state = "internal_validation_failed"
        state_label = "Internal validation failed"
    elif not human_validation:
        state = "human_validation_required"
        state_label = "Human validation required"
    elif not deployment:
        state = "validated_not_active"
        state_label = "Validated, not active"
    else:
        state = "active"
        state_label = "Active production model"

    ablation: dict[str, Any] = {}
    ablation_path = model_root / "change_evidence_ablation.json"
    if ablation_path.is_file():
        try:
            ablation = _read_mapping(ablation_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Change ablation: {type(exc).__name__}: {exc}")

    fallback_timestamp = max(
        (path.stat().st_mtime for path in (summary_path, request_path) if path.is_file()),
        default=model_root.stat().st_mtime,
    )
    created_utc = str(
        summary.get("created_utc") or request.get("created_utc") or ""
    )
    bundles = request.get("bundles") or summary.get("source_snapshot") or []
    training_bundles = (
        list(bundles)
        if isinstance(bundles, Sequence) and not isinstance(bundles, (str, bytes))
        else []
    )
    return {
        "schema": 1,
        "format": PHASE_MODEL_HISTORY_FORMAT,
        "model_id": str(
            summary.get("campaign_id") or request.get("model_id") or model_root.name
        ),
        "root": str(model_root),
        "created_utc": created_utc,
        "created_timestamp": _created_timestamp(created_utc, fallback_timestamp),
        "created_by": str(request.get("created_by") or "Not recorded"),
        "historical_format": legacy,
        "state": state,
        "state_label": state_label,
        "internal_holdout_passed": internal_holdout_passed,
        "human_validation": human_validation,
        "deployment": deployment,
        "datasets": list(bundle.get("datasets") or inventory.get("datasets") or []),
        "runs": list(bundle.get("runs") or inventory.get("runs") or []),
        "region_count": int(inventory.get("regions") or 0),
        "radiometry_profile": summary.get("radiometry_profile"),
        "radiometry_label": (
            "Uniform24 / " + str(summary["radiometry_profile"]["sha256"])[:16]
            if summary.get("radiometry_profile") else "Legacy / untracked prepared images"
        ),
        "pixel_rows": int(inventory.get("pixel_rows") or 0),
        "phase_classes": dict(inventory.get("phase_classes") or {}),
        "adapters": list(bundle.get("adapters") or []),
        "change_detector": str(bundle.get("change_detector") or "Not available"),
        "requested_backend": str(summary.get("backend_requested") or ""),
        "heads": heads,
        "co2_balanced_accuracy": _primary_metric(heads, "co2_presence"),
        "gas_balanced_accuracy": _primary_metric(heads, "gas_within_co2"),
        "change_ablation_delta": ablation.get("mean_balanced_accuracy_delta"),
        "training_bundles": training_bundles,
        "errors": errors,
    }


def discover_phase_model_history(
    library_root: Path,
    *,
    max_depth: int = 4,
) -> list[dict[str, Any]]:
    """Return newest-first model history from a user-selected model library."""

    root = Path(library_root).expanduser().resolve()
    if not root.is_dir():
        return []
    directories = _model_directories(root, max_depth=max(0, int(max_depth)))
    if root.name.casefold() in {"phase_models", "shared phase models"}:
        try:
            siblings = [
                Path(entry.path)
                for entry in os.scandir(root.parent)
                if entry.is_dir(follow_symlinks=False)
                and Path(entry.path, "summary.json").is_file()
                and Path(entry.path, "models").is_dir()
            ]
        except OSError:
            siblings = []
        directories.extend(siblings)
    unique = {str(path.resolve()).casefold(): path.resolve() for path in directories}
    entries = [inspect_phase_model_history_entry(path) for path in unique.values()]
    return sorted(
        entries,
        key=lambda row: (
            float(row.get("created_timestamp") or 0.0),
            str(row.get("model_id") or ""),
        ),
        reverse=True,
    )


__all__ = [
    "PHASE_MODEL_HISTORY_FORMAT",
    "discover_phase_model_history",
    "inspect_phase_model_history_entry",
]
