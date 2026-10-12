"""Prepare isolated run-native colour paths for detailed FF-AC layers.

The established facies calibration is never modified. This script activates
the ``relative_colorpath_labels`` config overlay and writes all generated
artifacts to that embedding's dedicated folder before a hierarchical or fully
layer-specific Optuna screen is allowed to start.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import hashlib
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.darsia_workflow_compat import (  # noqa: E402
    calibration_color_paths,
)
from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (  # noqa: E402
    calibration_color_to_mass_analysis,
)
from darsia.presets.workflows.config.fluidflower_config import (  # noqa: E402
    FluidFlowerConfig,
)
from darsia.presets.workflows.rig import Rig  # noqa: E402


DEFAULT_RUNS = ("ac20", "ac24", "ac26", "ac27", "ac32", "ac60")
DEFAULT_OUTPUT = configured_tool_path('ffac', 'calibration_logs') / 'final_campaign_preflight_20260806/label_basis'
EXPECTED_ACTIVE_LABELS = (0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def config_paths(run: str, config_dir: Path) -> list[Path]:
    paths = [
        config_dir.parent / "common.toml",
        config_dir / f"{run}.toml",
        config_dir.parent / "labels_basis.toml",
    ]
    stamp = config_dir / ".color_state" / f"{run}.txt"
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip().lower() == "on":
        paths.append(config_dir.parent / "coloron.toml")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing labels-basis config input(s): "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def inspect_run(run: str, config_dir: Path) -> dict[str, Any]:
    paths = config_paths(run, config_dir)
    config = FluidFlowerConfig(paths, require_data=True, require_results=True)
    embedding = config.calibration.mass.color
    if embedding is None:
        raise RuntimeError(f"[{run}] mass colour embedding did not resolve")
    basis = str(getattr(embedding.basis, "value", embedding.basis)).lower()
    if embedding.embedding_id != "relative_colorpath_labels" or basis != "labels":
        raise RuntimeError(
            f"[{run}] unexpected embedding {embedding.embedding_id!r}/{basis!r}"
        )

    rig_path = Path(config.rig.path)
    labels_path = rig_path / "labels.npz"
    facies_path = rig_path / "facies.npz"
    if not labels_path.is_file() or not facies_path.is_file():
        raise FileNotFoundError(f"[{run}] detailed label/facies rig maps are missing")
    detailed = np.load(labels_path, allow_pickle=True)["array"]
    facies = np.load(facies_path, allow_pickle=True)["array"]
    if detailed.shape != facies.shape:
        raise RuntimeError(f"[{run}] label/facies shape mismatch")
    active = sorted(
        int(label)
        for label in np.unique(detailed)
        if int(label) not in set(embedding.ignore_labels)
    )
    if active != list(EXPECTED_ACTIVE_LABELS):
        raise RuntimeError(
            f"[{run}] unexpected active detailed labels: {active}; "
            f"expected {list(EXPECTED_ACTIVE_LABELS)}"
        )

    color_folder = Path(embedding.color_paths_folder)
    mass_folder = Path(embedding.color_to_mass_folder)
    color_metadata = color_folder / "metadata.json"
    mass_metadata = mass_folder / "metadata.json"
    return {
        "run": run,
        "config_paths": [str(path) for path in paths],
        "embedding_id": embedding.embedding_id,
        "basis": basis,
        "active_labels": active,
        "ignore_labels": list(embedding.ignore_labels),
        "reference_label": int(embedding.reference_label),
        "color_paths_folder": str(color_folder),
        "color_to_mass_folder": str(mass_folder),
        "baseline_spectrum_folder": str(embedding.baseline_color_spectrum_folder),
        "color_paths_ready": color_metadata.is_file(),
        "color_to_mass_ready": mass_metadata.is_file(),
        "isolated_from_facies": (
            "relative_colorpath_labels" in str(color_folder).lower()
            and "from_labels" in str(color_folder).lower()
        ),
        "rig_maps": {
            "labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
            "facies": {"path": str(facies_path), "sha256": _sha256(facies_path)},
        },
    }


def prepare_run(run: str, config_dir: Path) -> dict[str, Any]:
    before = inspect_run(run, config_dir)
    if before["color_paths_ready"] and before["color_to_mass_ready"]:
        return {**before, "action": "reused-complete"}
    if before["color_paths_ready"] or before["color_to_mass_ready"]:
        raise RuntimeError(
            f"[{run}] labels-basis output is only partially present. Refusing to "
            "overwrite it automatically; inspect the dedicated embedding folder first."
        )

    paths = [Path(path) for path in before["config_paths"]]
    calibration_color_paths(Rig, paths, show=False)
    calibration_color_to_mass_analysis(
        Rig,
        paths,
        reset=True,
        show=False,
        default=True,
    )
    after = inspect_run(run, config_dir)
    if not after["color_paths_ready"] or not after["color_to_mass_ready"]:
        raise RuntimeError(f"[{run}] labels-basis calibration did not complete")
    return {**after, "action": "created"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare"))
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--config-dir", default="config_seg6/run_ac")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    output = Path(args.output)
    records: list[dict[str, Any]] = []
    for index, raw_run in enumerate(args.runs, start=1):
        run = str(raw_run).lower()
        print(f"[{index}/{len(args.runs)}] {run}: {args.action}", flush=True)
        record = (
            prepare_run(run, config_dir)
            if args.action == "prepare"
            else inspect_run(run, config_dir)
        )
        records.append(record)
        _atomic_json(
            output / f"{run}.json",
            {"checked_at_utc": _utc_now(), **record},
        )

    payload = {
        "schema": 1,
        "created_at_utc": _utc_now(),
        "host": socket.gethostname().upper(),
        "action": args.action,
        "runs": records,
        "all_isolated": all(row["isolated_from_facies"] for row in records),
        "all_ready": all(
            row["color_paths_ready"] and row["color_to_mass_ready"]
            for row in records
        ),
    }
    _atomic_json(output / "manifest.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
