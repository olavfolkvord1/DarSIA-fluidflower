"""Load and apply an isolated calibration candidate to a mass-analysis context."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_SIGNAL_LABEL_PATTERN = re.compile(r"^signal\.label(?P<label>\d+)\.")
_COLORCHECKER_ENV_PREFIX = "FFAC_COLORCHECKER_"
_PREPROCESSING_ENV_NAMES = frozenset(("FFAC_COLOR_PATH_CONFIG_OVERLAY",))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _params_sha256(params: dict[str, Any]) -> str:
    encoded = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_path(candidate_dir: Path | str, run: str) -> Path:
    return Path(candidate_dir) / f"final_full_scale_{run.lower()}.json"


def load_candidate(candidate_dir: Path | str, run: str) -> tuple[Path, dict[str, Any]]:
    path = candidate_path(candidate_dir, run)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration candidate not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    payload_run = str(payload.get("run", run)).strip().lower()
    params = payload.get("params")
    if payload_run != run.lower():
        raise ValueError(f"Candidate {path} belongs to {payload_run}, not {run}.")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Candidate has no parameter mapping: {path}")
    return path, payload


def candidate_environment(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return preprocessing environment recorded with a calibration candidate."""

    selection = payload.get("selection")
    environment = (
        selection.get("environment_overrides")
        if isinstance(selection, Mapping)
        else None
    )
    if not isinstance(environment, Mapping):
        environment = payload.get("environment_overrides")
    if not isinstance(environment, Mapping):
        return {}
    return {
        str(name): str(value)
        for name, value in environment.items()
        if str(name).startswith(_COLORCHECKER_ENV_PREFIX)
        or str(name) in _PREPROCESSING_ENV_NAMES
    }


def candidate_directory_environment(
    candidate_dir: Path | str,
) -> dict[str, str]:
    """Resolve one consistent optical preprocessing contract for candidates."""

    paths = sorted(Path(candidate_dir).glob("final_full_scale_ac*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No final_full_scale_ac*.json candidates in {candidate_dir}"
        )
    contracts: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        contract = tuple(sorted(candidate_environment(payload).items()))
        contracts.setdefault(contract, []).append(path.name)
    if len(contracts) > 1:
        details = "; ".join(
            f"{dict(contract)}: {', '.join(names[:3])}"
            for contract, names in contracts.items()
        )
        raise RuntimeError(
            "Calibration candidate directory mixes incompatible optical "
            f"preprocessing contracts: {details}"
        )
    return dict(next(iter(contracts)))


def _signal_labels(calibration: Any) -> list[int]:
    signal_model = getattr(getattr(calibration, "signal_model", None), "model", None)
    if isinstance(signal_model, dict):
        per_phase = signal_model.get(1, signal_model)
    else:
        try:
            per_phase = signal_model[1]
        except (IndexError, KeyError, TypeError):
            per_phase = None
    if hasattr(per_phase, "keys"):
        labels = sorted(int(label) for label in per_phase.keys())
        if labels:
            return labels
    raise RuntimeError("Could not determine active signal labels for candidate.")


def _candidate_signal_labels(params: dict[str, Any]) -> list[int]:
    labels = {
        int(match.group("label"))
        for name in params
        if (match := _SIGNAL_LABEL_PATTERN.match(str(name))) is not None
    }
    return sorted(labels)


def _prepare_hierarchical_mapping(
    calibration: Any,
    params: dict[str, Any],
    labels: list[int],
    fluidflower: Any | None,
) -> dict[int, int] | None:
    if not any(str(name).startswith("signal.facies") for name in params):
        return None
    existing = getattr(calibration, "_ffac_signal_label_to_facies", None)
    if isinstance(existing, dict) and all(label in existing for label in labels):
        return {int(label): int(existing[label]) for label in labels}
    if fluidflower is None:
        raise RuntimeError(
            "Hierarchical calibration candidate requires the FluidFlower label-to-"
            "facies mapping; pass fluidflower when applying the candidate."
        )
    try:
        from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
            _infer_signal_label_to_facies,
        )
    except ImportError:
        from darsia.fluidflower.tools.auto_calibrate_color_to_mass import _infer_signal_label_to_facies

    mapping = _infer_signal_label_to_facies(
        fluidflower,
        labels,
        use_facies=False,
    )
    setattr(calibration, "_ffac_signal_label_to_facies", dict(mapping))
    return {int(label): int(mapping[label]) for label in labels}


def apply_candidate(
    calibration: Any,
    candidate_dir: Path | str,
    run: str,
    *,
    fluidflower: Any | None = None,
) -> dict[str, Any]:
    """Apply candidate signal/flash parameters without modifying saved calibration."""

    try:
        from darsia.fluidflower.tools.auto_calibrate_color_to_mass import apply_params
    except ImportError:
        from darsia.fluidflower.tools.auto_calibrate_color_to_mass import apply_params

    path, payload = load_candidate(candidate_dir, run)
    params = dict(payload["params"])
    model_labels = _signal_labels(calibration)
    candidate_labels = _candidate_signal_labels(params)
    labels = candidate_labels or model_labels
    unknown = sorted(set(labels).difference(model_labels))
    if unknown:
        raise RuntimeError(
            f"Candidate {path} references signal labels absent from the loaded "
            f"calibration: {unknown}"
        )
    label_to_facies = _prepare_hierarchical_mapping(
        calibration,
        params,
        labels,
        fluidflower,
    )
    updated = apply_params(
        calibration,
        params,
        labels=labels,
        np_module=np,
    )
    if updated <= 0:
        raise RuntimeError(f"Candidate {path} did not update the loaded calibration.")
    noise_gate_payload = payload.get("aqueous_noise_gate")
    if isinstance(noise_gate_payload, dict) and noise_gate_payload.get(
        "thresholds"
    ):
        try:
            from darsia.fluidflower.tools.aqueous_noise_gate import AqueousNoiseGate
        except ImportError:
            from darsia.fluidflower.tools.aqueous_noise_gate import AqueousNoiseGate

        calibration.aqueous_noise_gate = AqueousNoiseGate.from_dict(
            noise_gate_payload
        )
    else:
        calibration.aqueous_noise_gate = None
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "params_sha256": _params_sha256(params),
        "objective_full_scale": payload.get("objective_full_scale"),
        "source": payload.get("source"),
        "labels": labels,
        "label_to_facies": label_to_facies,
        "updated_parameters": updated,
        "aqueous_noise_gate": (
            calibration.aqueous_noise_gate.to_dict()
            if calibration.aqueous_noise_gate is not None
            else None
        ),
    }
