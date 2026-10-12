"""AC52 pilot with one optical contract for images and colour paths.

The first ColorChecker pilot deliberately retained the established colour-path
embedding. This follow-up regenerates that embedding from images processed by
the same per-frame ColorChecker normalization used during Optuna evaluation.
All generated artefacts live below the campaign root; the production
calibration in ``Z:/Albus/Results`` is never overwritten.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.darsia_workflow_compat import (  # noqa: E402
    calibration_color_paths,
    calibration_color_paths_and_mass,
    calibration_color_to_mass_analysis,
)
from darsia.presets.workflows.rig import Rig  # noqa: E402
from darsia.fluidflower.tools import ac_colorchecker_refinement_pilot as first_pilot  # noqa: E402
from darsia.fluidflower.tools import ac_objective_v2_campaign as objective_v2  # noqa: E402
from darsia.fluidflower.tools import ac_spatial_calibration_campaign as engine  # noqa: E402
from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (  # noqa: E402
    _configure_runtime_colorchecker_refinement,
)


CAMPAIGN_ID = "colorchecker_refinement_rebuilt_paths_20260826_5000"
DEFAULT_OUTPUT_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
RUNS = ("ac52",)
MAX_ITERS = 5000
_base_spec = objective_v2.SEARCH_SPECS[0]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def color_path_root(output_root: Path) -> Path:
    return output_root / "run_native_color_embedding" / "ac52"


def overlay_path(output_root: Path) -> Path:
    return output_root / "config" / "ac52_color_path_overlay.toml"


def write_overlay(output_root: Path) -> Path:
    """Write a minimal deep-merge overlay for an isolated embedding root."""

    destination = overlay_path(output_root).resolve()
    calibration_folder = color_path_root(output_root).resolve().as_posix()
    _atomic_text(
        destination,
        "[color.path.relative_colorpath_labels]\n"
        f'calibration_folder = "{calibration_folder}"\n',
    )
    return destination


def _config_paths(output_root: Path) -> list[Path]:
    config_dir = project_path('config_seg6/run_ac')
    paths = [
        config_dir.parent / "common.toml",
        config_dir / "ac52.toml",
        config_dir.parent / "labels_basis.toml",
    ]
    stamp = config_dir / ".color_state" / "ac52.txt"
    if (
        stamp.is_file()
        and stamp.read_text(encoding="utf-8-sig").strip().lower() == "on"
    ):
        paths.append(config_dir.parent / "coloron.toml")
    paths.append(write_overlay(output_root))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing AC52 colour-path input(s): "
            + ", ".join(str(path) for path in missing)
        )
    return paths


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    missing = object()
    previous: dict[str, object | str] = {
        name: os.environ.get(name, missing) for name in values
    }
    os.environ.update({str(name): str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)


class RefinedColorPathRig(Rig):
    """Install FF-AC normalization before calibration images are read."""

    @classmethod
    def load(cls, folder: Path, corrections_config: Any = None) -> Rig:
        rig = super().load(folder, corrections_config=corrections_config)
        provenance = _configure_runtime_colorchecker_refinement(rig)
        rig._ffac_colorchecker_refinement_provenance = provenance
        return rig


def _embedding_ready(root: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            root / "color_paths" / "from_labels" / "metadata.json",
            root / "color_to_mass" / "from_labels" / "metadata.json",
        )
    )


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    files = sorted(
        path
        for folder in (
            root / "color_paths" / "from_labels",
            root / "color_to_mass" / "from_labels",
            root / "baseline_color_spectrum",
        )
        if folder.is_dir()
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".npz"}
    )
    return [
        {
            "path": str(path.resolve()),
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def prepare_color_paths(output_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Generate the isolated AC52 embedding with matching preprocessing."""

    output_root = output_root.resolve()
    target = color_path_root(output_root)
    manifest_path = output_root / "color_path_preflight.json"
    if _embedding_ready(target) and manifest_path.is_file() and not force:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not payload.get("artifacts"):
            payload["artifacts"] = _artifact_inventory(target)
            _atomic_json(manifest_path, payload)
        return payload
    if target.exists() and any(target.iterdir()):
        if not force:
            raise RuntimeError(
                "Isolated colour-path folder is partial or unverified; use "
                f"--force after inspection: {target}"
            )
        archive = target.with_name(
            target.name
            + ".archived_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.move(str(target), str(archive))

    paths = _config_paths(output_root)
    environment = dict(first_pilot.COLORCHECKER_ENVIRONMENT)
    with _temporary_environment(environment):
        calibration_color_paths(RefinedColorPathRig, paths, show=False)
        calibration_color_to_mass_analysis(
            RefinedColorPathRig,
            paths,
            reset=True,
            show=False,
            default=True,
        )
    if not _embedding_ready(target):
        raise RuntimeError(
            f"Isolated AC52 colour-path calibration did not complete: {target}"
        )

    metadata_paths = (
        target / "color_paths" / "from_labels" / "metadata.json",
        target / "color_to_mass" / "from_labels" / "metadata.json",
    )
    payload = {
        "schema": 1,
        "status": "complete",
        "run": "ac52",
        "basis": "labels",
        "created_utc": _utc_now(),
        "host": socket.gethostname().upper(),
        "overlay": str(overlay_path(output_root).resolve()),
        "calibration_root": str(target.resolve()),
        "production_results_untouched": True,
        "environment": environment,
        "config_paths": [str(path.resolve()) for path in paths],
        "metadata": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in metadata_paths
        ],
        "artifacts": _artifact_inventory(target),
    }
    _atomic_json(manifest_path, payload)
    return payload


def _search_specs(output_root: Path) -> tuple[engine.SearchSpec, ...]:
    base = _base_spec
    environment = dict(first_pilot.COLORCHECKER_ENVIRONMENT)
    environment["FFAC_COLOR_PATH_CONFIG_OVERLAY"] = str(
        overlay_path(output_root).resolve()
    )
    return (
        replace(
            base,
            name="expanded25_physical_v2_colorchecker_rebuilt_paths",
            variant=replace(
                base.variant,
                name="expanded25_physical_v2_colorchecker_rebuilt_paths",
            ),
            environment_overrides=tuple(environment.items()),
            rationale=(
                "Objective V2 with a single optical contract: both the AC52 "
                "colour-path embedding and every evaluation image use the "
                "same run-native ColorChecker normalization."
            ),
        ),
    )


SEARCH_SPECS = _search_specs(DEFAULT_OUTPUT_ROOT)


def configure_engine(output_root: Path | None = None) -> None:
    global SEARCH_SPECS

    resolved_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
    objective_v2.configure_engine()
    engine.CAMPAIGN_ID = CAMPAIGN_ID
    engine.DEFAULT_OUTPUT_ROOT = resolved_root
    engine.PILOT_RUNS = RUNS
    engine.RUNS = RUNS
    engine.ALL_RUNS = RUNS
    engine.MAX_ITERS = MAX_ITERS
    engine.OPTIMIZE_JOB_ORDER = "run-major"
    engine.WORKER_SCRIPT = Path(__file__).resolve()
    SEARCH_SPECS = _search_specs(resolved_root)
    engine.SEARCH_SPECS = SEARCH_SPECS
    engine.SPECS_BY_NAME = {spec.name: spec for spec in engine.SEARCH_SPECS}


def _extract_output_root(argv: Sequence[str]) -> Path:
    values = list(argv)
    if "--output-root" not in values:
        return DEFAULT_OUTPUT_ROOT
    index = values.index("--output-root")
    if index + 1 >= len(values):
        raise SystemExit("--output-root requires a path")
    return Path(values[index + 1])


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    output_root = _extract_output_root(arguments).resolve()
    if arguments and arguments[0] == "prepare-color-paths":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("action", choices=("prepare-color-paths",))
        parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        parser.add_argument("--force", action="store_true")
        parsed = parser.parse_args(arguments)
        print(
            json.dumps(
                prepare_color_paths(parsed.output_root, force=parsed.force),
                indent=2,
            )
        )
        return

    action = arguments[0] if arguments else ""
    if action in {"prepare", "launch"}:
        prepare_color_paths(output_root, force=False)
    elif action == "worker" and not _embedding_ready(color_path_root(output_root)):
        raise SystemExit(
            "Run prepare-color-paths (or prepare) before starting a worker."
        )
    configure_engine(output_root)
    if "--tracks" not in arguments:
        arguments.extend(("--tracks", "optimize"))
    engine.main(arguments)


if __name__ == "__main__":
    main()
