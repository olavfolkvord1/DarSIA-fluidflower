"""Explicit numerical contracts for FF-AC calibration evaluators.

The historical accelerator names are intentionally retained.  New production
studies should use the precision-qualified names so a study cannot silently
switch numerical implementation after a worker restart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping


CONTRACT_SCHEMA = 2
FLOAT32_SEMANTICS = "phase-mass-float32-v1-hard-gas-mask"
CPU_REFERENCE_SEMANTICS = "phase-mass-cpu-float64-reference-v1"
LEGACY_SEMANTICS = "legacy-mixed-precision-v1"


@dataclass(frozen=True)
class CalibrationBackendSpec:
    name: str
    implementation: str
    device_family: str
    precision: str
    semantic_contract: str
    study_family: str
    legacy: bool = False
    authoritative_reference: bool = False

    @property
    def study_compatibility_key(self) -> str:
        """Key preventing precision/semantic mixing inside one study."""

        return f"{self.study_family}|{self.precision}|{self.semantic_contract}"

    @property
    def execution_key(self) -> str:
        """Exact implementation key recorded for every worker/evaluation."""

        return f"{self.name}|{self.implementation}|{self.precision}"


_SPECS = {
    "legacy": CalibrationBackendSpec(
        "legacy",
        "legacy",
        "cpu",
        "legacy",
        LEGACY_SEMANTICS,
        "legacy:legacy",
        legacy=True,
    ),
    "prepared": CalibrationBackendSpec(
        "prepared",
        "prepared",
        "cpu",
        "float64",
        LEGACY_SEMANTICS,
        "legacy:prepared",
        legacy=True,
    ),
    "cpu-float64": CalibrationBackendSpec(
        "cpu-float64",
        "prepared",
        "cpu",
        "float64",
        CPU_REFERENCE_SEMANTICS,
        "cpu-float64-reference",
        authoritative_reference=True,
    ),
    "cuda": CalibrationBackendSpec(
        "cuda",
        "cuda",
        "cuda",
        "mixed",
        LEGACY_SEMANTICS,
        "legacy:cuda",
        legacy=True,
    ),
    "opencl": CalibrationBackendSpec(
        "opencl",
        "opencl",
        "opencl",
        "mixed",
        LEGACY_SEMANTICS,
        "legacy:opencl",
        legacy=True,
    ),
    "cuda-float32": CalibrationBackendSpec(
        "cuda-float32",
        "cuda-float32",
        "cuda",
        "float32",
        FLOAT32_SEMANTICS,
        "gpu-float32",
    ),
    "opencl-float32": CalibrationBackendSpec(
        "opencl-float32",
        "opencl-float32-packed",
        "opencl",
        "float32",
        FLOAT32_SEMANTICS,
        "gpu-float32",
    ),
    "mps-metal": CalibrationBackendSpec(
        "mps-metal",
        "mps-metal",
        "metal",
        "float32",
        FLOAT32_SEMANTICS,
        "legacy:mps-metal",
        legacy=True,
    ),
    "metal-float32": CalibrationBackendSpec(
        "metal-float32",
        "mps-metal",
        "metal",
        "float32",
        FLOAT32_SEMANTICS,
        "gpu-float32",
    ),
}

_ALIASES = {
    "cpu64": "cpu-float64",
    "cpu-fp64": "cpu-float64",
    "cuda32": "cuda-float32",
    "cuda-fp32": "cuda-float32",
    "ocl32": "opencl-float32",
    "opencl32": "opencl-float32",
    "opencl-fp32": "opencl-float32",
    "metal32": "metal-float32",
    "metal-fp32": "metal-float32",
}


def normalize_backend_name(value: str | None) -> str:
    name = str(value or "prepared").strip().lower().replace("_", "-")
    name = _ALIASES.get(name, name)
    if name not in _SPECS:
        expected = ", ".join(sorted(_SPECS))
        raise ValueError(f"Unknown calibration backend {value!r}; expected {expected}")
    return name


def backend_spec(value: str | None) -> CalibrationBackendSpec:
    return _SPECS[normalize_backend_name(value)]


def backend_contract_payload(
    backend: str,
    *,
    scope: str,
    identity: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = backend_spec(backend)
    payload: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "scope": str(scope),
        "identity": str(identity),
        "backend": asdict(spec),
        "execution_key": spec.execution_key,
        "study_compatibility_key": spec.study_compatibility_key,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def _read_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable backend contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid backend contract {path}: expected a JSON object")
    return payload


def lock_backend_contract(
    path: Path,
    backend: str,
    *,
    scope: str,
    identity: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically create or verify a persistent backend contract."""

    expected = backend_contract_payload(
        backend,
        scope=scope,
        identity=identity,
        extra=extra,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        existing = _read_contract(path)
    else:
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(expected, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        existing = expected

    actual_key = str(existing.get("study_compatibility_key") or "")
    expected_key = str(expected["study_compatibility_key"])
    if actual_key != expected_key:
        raise RuntimeError(
            "Calibration backend contract mismatch for "
            f"{identity}: existing={actual_key!r}, requested={expected_key!r}. "
            "Start a new study instead of mixing numerical precision or "
            "semantic contracts."
        )
    if str(existing.get("scope") or "") != str(scope) or str(
        existing.get("identity") or ""
    ) != str(identity):
        raise RuntimeError(
            f"Calibration backend contract identity mismatch in {path}."
        )
    return existing


def validate_quality_dtype(backend: str, quality_dtype: str) -> None:
    """Fail closed when an explicit backend receives the wrong input dtype."""

    spec = backend_spec(backend)
    dtype = str(quality_dtype).strip().lower()
    if spec.legacy:
        return
    if spec.precision == "float32" and dtype != "float32":
        raise ValueError(f"{spec.name} requires quality_dtype=float32, got {dtype!r}")
    if spec.name == "cpu-float64" and dtype != "float64":
        raise ValueError("cpu-float64 requires quality_dtype=float64")


def production_backend_names() -> tuple[str, ...]:
    return ("cpu-float64", "cuda-float32", "opencl-float32", "metal-float32")


def all_backend_names() -> tuple[str, ...]:
    return tuple(_SPECS)
