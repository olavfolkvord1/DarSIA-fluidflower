"""Freeze a guarded Windows Scheduler task for one FF_UM/FF_BM candidate.

This module only inspects inputs and writes local launch scaffolding. It never
registers or starts a task. Run ``scheduled_training_job register`` and
``scheduled_training_job start`` as separate, deliberate operations after QA of
the frozen command. The candidate remains outside Studio until publication.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tomllib

from darsia.fluidflower.tools import scheduled_training_job as runner


FAST_PROFILE = "generic-co2-fast-v1"
FEATURE_VERSION = "baseline-relative-bright-grain-v5"
HEADS = ["co2_presence", "gas_within_co2"]
GRID = [1e-5, 1e-3, 1e-1]
COORDINATOR = "darsia.fluidflower.tools.train_studio_optical_then_region_series"
IMPORT_REQUIRED_SERIES = {"ffac", "ffum", "ffbm"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _local(path: Path, label: str) -> Path:
    path = path.resolve()
    _require(path.is_absolute() and bool(path.drive) and not str(path).startswith("\\\\"),
             f"{label} must be on a local drive")
    return path


def _series_root(parent: Path, series: str) -> Path:
    _require(parent.name.startswith("phase-model-") and parent.is_dir()
             and parent.parent.name == "models", "Select a complete phase-model parent")
    root = parent.parent.parent
    manifest = root / "series.toml"
    _require(manifest.is_file(), f"Series manifest missing: {manifest}")
    with manifest.open("rb") as stream:
        metadata = tomllib.load(stream)
    identity = re.sub(r"[_-]", "", str(metadata.get("series_id", "")).casefold())
    _require(metadata.get("schema_version") == 1 and identity == series,
             "Parent belongs to another experimental series")
    return root


def build_spec(*, series: str, parent: Path, cache: Path, request_path: Path,
               job: Path, repository: Path, log_dir: Path, task_name: str,
               actor: str | None, backend: str, cpus: int | None,
               memory_gb: float | None, registry: Path | None,
               lock_file: Path | None) -> dict:
    """Validate the immutable inputs and return a scheduler config read-only."""
    _require(series in {"ffum", "ffbm"}, "Select ffum or ffbm")
    _require(backend in {"auto", "cpu", "cuda"}, "Invalid backend")
    _require(cpus is None or cpus > 0, "CPU worker count must be positive")
    _require(memory_gb is None or memory_gb > 0, "Memory budget must be positive")
    parent = parent.resolve()
    project = _series_root(parent, series)
    library = project / "models"
    results = project / "results"
    _require(results.is_dir(), f"Series results folder missing: {results}")
    job = job.resolve()
    _require(job.is_relative_to(results) and job != results and not job.exists(),
             "Candidate job must be a fresh folder under this series' results")
    cache = _local(cache, "Series cache")
    _require(cache.is_dir(), f"Series cache missing: {cache}")
    request_path = _local(request_path, "Frozen optical request")
    _require(request_path.is_file() and request_path.name == "training_request.json"
             and request_path.is_relative_to(cache / "studio-continuations"),
             "Frozen request must be inside the selected series' local cache")
    request = runner.read_json(request_path)
    settings = request.get("settings") or {}
    reviewer = request.get("created_by")
    _require(reviewer and (actor is None or actor == reviewer),
             "Reviewer must match the immutable optical request")
    _require(Path(request.get("output_root", "")).resolve() == request_path.parent
             and request.get("model_id") == request_path.parent.name,
             "Frozen request identity or output root differs from its path")
    _require(settings.get("optical_training_profile") == FAST_PROFILE
             and settings.get("adapters") == ["generic"]
             and settings.get("fit_heads") == HEADS
             and settings.get("regularization_grid") == GRID
             and settings.get("feature_version") == FEATURE_VERSION
             and settings.get("preserve_native_revisions") is True
             and not request.get("evaluation_only"),
             "Request is not the two-head v5 generic fast profile")
    retry = settings.get("bm_mil_retry")
    retry_sources = []
    if retry:
        from darsia.fluidflower.tools.prepare_bm_mil_retry import (
            ITERATIONS, verify_retry_request)
        _require(series == "ffbm" and settings.get("mil_max_iters") == ITERATIONS
                 and settings.get("disable_resume_completed_heads") is True
                 and job != Path(retry.get("failed_job", "")).resolve(),
                 "BM retry must use its exact fresh optimizer budget and job identity")
        verify_retry_request(request)
        retry_sources = [Path(retry["receipt_path"]), Path(retry["source_request_path"])]
    repository = _local(repository, "DarSIA repository")
    _require((repository / "src" / "darsia" / "fluidflower").is_dir(),
             "DarSIA source tree is missing")
    executable = repository / ".venv-fluidflower" / "Scripts" / "python.exe"
    pythonw = executable.with_name("pythonw.exe")
    _require(executable.is_file() and pythonw.is_file(),
             "DarSIA fluidflower Python runtime is missing")
    # APPDATA is virtualized inside Codex's MSIX process: the same spelled
    # roaming file returned different bytes in a detached Scheduler probe.
    # Require a series-owned local snapshot visible identically in both. The
    # inherited extraction module still resolves FF_AC at import time even
    # during FF_UM/BM training, so all three registered roots are required.
    _require(registry is not None, "Provide a frozen registry snapshot under the series cache")
    registry = _local(registry, "Frozen series registry")
    _require(registry.is_file() and registry.is_relative_to(cache),
             "Frozen series registry must be a file under the selected series cache")
    with registry.open("rb") as stream:
        registry_data = tomllib.load(stream)
    rows = registry_data.get("series") or []
    indexed = {re.sub(r"[_-]", "", str(row.get("id", "")).casefold()): row
               for row in rows}
    _require(registry_data.get("schema_version") == 1
             and len(rows) == len(indexed) == len(IMPORT_REQUIRED_SERIES)
             and set(indexed) == IMPORT_REQUIRED_SERIES
             and Path(indexed[series].get("root", "")).resolve() == project
             and Path(registry_data.get("storage", {}).get("cache_root", "")).resolve() == cache,
             "Frozen registry must bind FF_AC, FF_UM, FF_BM and this series' cache")
    for key, row in indexed.items():
        root = Path(row.get("root", "")).resolve()
        manifest = root / "series.toml"
        _require(manifest.is_file(), f"Frozen registry root is unavailable: {root}")
        with manifest.open("rb") as stream:
            root_data = tomllib.load(stream)
        _require(re.sub(r"[_-]", "", str(root_data.get("series_id", "")).casefold()) == key,
                 f"Frozen registry points to the wrong project: {key}")
    lock_file = _local(lock_file or Path(os.environ.get("LOCALAPPDATA", str(
        Path.home() / "AppData" / "Local"))) / "DarSIA" /
        "phase_model_training.lock", "Cross-series lock")
    log_dir = _local(log_dir, "Task log folder")
    _require(log_dir.is_relative_to(cache) and not log_dir.exists(),
             "Task logs must be a fresh folder in the selected series' cache")
    runner._task_name(task_name)
    _require(task_name.split("\\")[1].casefold() == series,
             "Scheduler task folder must match the selected series")
    args = ["-u", "-m", COORDINATOR,
            "--series", series, "--parent", str(parent), "--library", str(library),
            "--cache", str(cache), "--job", str(job), "--actor", reviewer,
            "--backend", backend, "--prepared-optical-request", str(request_path)]
    if cpus is not None:
        args += ["--cpus", str(cpus)]
    if memory_gb is not None:
        args += ["--memory-gb", str(memory_gb)]
    args += ["--execute"]
    guarded_sources = [request_path, parent / "summary.json",
                       parent / "training_request.json",
                       parent / "review_source" / "postflight_manifest.json",
                       project / "series.toml", registry,
                       repository / "src" / "darsia" / "fluidflower" / "tools" /
                           "train_studio_optical_then_region_series.py",
                       repository / "src" / "darsia" / "fluidflower" / "tools" /
                           "train_phase_model_further.py",
                       repository / "src" / "darsia" / "fluidflower" /
                           "phase_training" / "trainer.py",
                       repository / "src" / "darsia" / "fluidflower" /
                           "phase_training" / "logistic.py",
                       repository / "src" / "darsia" / "fluidflower" /
                           "phase_workflow.py", *retry_sources]
    if retry:
        guarded_sources.append(repository / "src" / "darsia" / "fluidflower" /
                               "tools" / "prepare_bm_mil_retry.py")
    _require(all(path.is_file() for path in guarded_sources),
             "One or more frozen parent, source or code files are missing")
    return dict(schema=runner.SCHEMA, task_name=task_name,
                executable=str(executable), argv=args, cwd=str(repository),
                runner_pythonw=str(pythonw), log_dir=str(log_dir),
                environment={"PYTHONPATH": str(repository / "src"),
                             "DARSIA_SERIES_REGISTRY": str(registry),
                             f"DARSIA_{series.upper()}_PROJECT_ROOT": str(project),
                             "DARSIA_FLUIDFLOWER_CACHE": str(cache)},
                required_paths=[str(path) for path in guarded_sources],
                required_sha256={str(path): runner.sha256(path) for path in guarded_sources},
                absent_paths=[str(job)], lock_file=str(lock_file))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", choices=("ffum", "ffbm"), required=True)
    for name in ("parent", "cache", "request", "job", "task-name"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repository", type=Path,
                        default=Path(__file__).resolve().parents[4])
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--actor")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--registry", type=Path, required=True,
                        help="immutable selected-series TOML under --cache")
    parser.add_argument("--lock-file", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="read-only launch plan")
    mode.add_argument("--prepare", action="store_true", help="freeze local Scheduler files; do not start")
    args = parser.parse_args(argv)
    cache = Path(args.cache)
    job = Path(args.job)
    log_dir = args.log_dir or cache / "scheduled-candidates" / job.name
    spec = build_spec(series=args.series, parent=Path(args.parent), cache=cache,
        request_path=Path(args.request), job=job, repository=args.repository,
        log_dir=log_dir, task_name=args.task_name, actor=args.actor,
        backend=args.backend, cpus=args.cpus, memory_gb=args.memory_gb,
        registry=args.registry, lock_file=args.lock_file)
    if args.prepare:
        config = log_dir.with_suffix(".launch.json")
        config.parent.mkdir(parents=True, exist_ok=True)
        with config.open("x", encoding="utf-8") as stream:
            json.dump(spec, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        runner.prepare(config)
    print(json.dumps(dict(prepared=args.prepare, started=False,
                          config_sha256=runner.sha256(log_dir.with_suffix(".launch.json"))
                          if args.prepare else None, spec=spec),
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
