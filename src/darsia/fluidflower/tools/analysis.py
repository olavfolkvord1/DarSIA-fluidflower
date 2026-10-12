"""CLI for analysis with optional ff_ac-owned GPU mass backends."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import timezone
from pathlib import Path
from typing import Any, Callable

import darsia
from darsia.presets.workflows import user_interface_analysis as analysis_ui
from darsia.presets.workflows.rig import Rig

try:
    from darsia.fluidflower.tools.calibration_candidate import (
        apply_candidate,
        candidate_directory_environment,
    )
    from darsia.fluidflower.tools.optimized_mass_analysis import optimized_analysis_mass_from_context
    from darsia.fluidflower.tools.spatial_mass_evaluator import (
        AcceleratedColorToMassAnalysis,
        create_spatial_mass_evaluator,
        ensure_aqueous_noise_gate,
        ensure_titration_flash,
        write_mass_backend_manifest,
    )
except ImportError:
    from darsia.fluidflower.tools.calibration_candidate import (
        apply_candidate,
        candidate_directory_environment,
    )
    from darsia.fluidflower.tools.optimized_mass_analysis import optimized_analysis_mass_from_context
    from darsia.fluidflower.tools.spatial_mass_evaluator import (
        AcceleratedColorToMassAnalysis,
        create_spatial_mass_evaluator,
        ensure_aqueous_noise_gate,
        ensure_titration_flash,
        write_mass_backend_manifest,
    )


logger = logging.getLogger(__name__)

_CALIBRATION_PROVENANCE = Path(
    "calibration/color/relative_colorpath/color_to_mass/"
    "from_facies/ffac_final_calibration.json"
)


def _install_candidate_colorchecker_preprocessing(
    candidate_dir: Path | None,
) -> dict[str, Any]:
    """Install candidate-recorded preprocessing before DarSIA builds its context."""

    if candidate_dir is None:
        return {"active": False, "source": "no-candidate-directory"}
    environment = candidate_directory_environment(candidate_dir)
    active = environment.get("FFAC_COLORCHECKER_REFINEMENT", "off").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not active:
        os.environ["FFAC_COLORCHECKER_REFINEMENT"] = "off"
        return {
            "active": False,
            "source": str(candidate_dir),
            "environment": environment,
        }
    os.environ.update(environment)
    if getattr(Rig, "_ffac_colorchecker_load_hook", False):
        return {
            "active": True,
            "source": str(candidate_dir),
            "environment": environment,
            "hook": "already-installed",
        }

    try:
        from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
            _configure_runtime_colorchecker_refinement,
        )
    except ImportError:
        from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
            _configure_runtime_colorchecker_refinement,
        )

    original_load = Rig.load.__func__

    def load_with_candidate_preprocessing(
        cls: type[Rig],
        folder: Path,
        corrections_config: Any = None,
    ) -> Rig:
        rig = original_load(cls, folder, corrections_config)
        provenance = _configure_runtime_colorchecker_refinement(rig)
        rig._ffac_colorchecker_refinement_provenance = provenance
        return rig

    Rig.load = classmethod(load_with_candidate_preprocessing)
    Rig._ffac_colorchecker_load_hook = True
    return {
        "active": True,
        "source": str(candidate_dir),
        "environment": environment,
        "hook": "installed-before-context-build",
    }


def _append_candidate_config_overlays(
    argv: list[str], preprocessing: dict[str, Any]
) -> list[str]:
    """Append candidate-owned config overlays to DarSIA's --config values."""

    environment = preprocessing.get("environment")
    if not isinstance(environment, dict):
        return list(argv)
    raw_overlay = str(
        environment.get("FFAC_COLOR_PATH_CONFIG_OVERLAY", "")
    ).strip()
    if not raw_overlay:
        return list(argv)
    overlay = Path(raw_overlay).expanduser()
    if not overlay.is_absolute():
        raise ValueError(
            "Candidate colour-path config overlay must be absolute: "
            f"{raw_overlay!r}"
        )
    if not overlay.is_file():
        raise FileNotFoundError(
            f"Candidate colour-path config overlay not found: {overlay}"
        )

    resolved = list(argv)
    try:
        config_index = resolved.index("--config")
    except ValueError as exc:
        raise ValueError(
            "A candidate colour-path overlay requires DarSIA --config arguments"
        ) from exc
    insert_at = config_index + 1
    while insert_at < len(resolved) and not resolved[insert_at].startswith("--"):
        insert_at += 1
    overlay_text = str(overlay)
    if overlay_text not in resolved[config_index + 1 : insert_at]:
        resolved.insert(insert_at, overlay_text)
    return resolved


def _extract_backend_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--mass-backend",
        choices=["auto", "cpu", "cuda", "opencl"],
        default=os.environ.get("FFAC_MASS_BACKEND", "cpu"),
    )
    parser.add_argument(
        "--mass-output-dtype",
        choices=["float32", "float64"],
        default=os.environ.get("FFAC_MASS_OUTPUT_DTYPE", "float32"),
    )
    parser.add_argument(
        "--mass-cuda-color-interpretation",
        choices=["auto", "on", "off"],
        default=os.environ.get(
            "FFAC_MASS_CUDA_COLOR_INTERPRETATION",
            "off",
        ),
    )
    parser.add_argument(
        "--titration-flash",
        choices=["auto", "on", "off"],
        default=os.environ.get("FFAC_TITRATION_FLASH_MODE", "auto"),
    )
    parser.add_argument(
        "--mass-numerical-modes",
        default=os.environ.get("FFAC_MASS_NUMERICAL_MODES", "mass"),
        help="Comma-separated mass products that retain numerical exports.",
    )
    parser.add_argument(
        "--wasserstein-cache-factor",
        type=float,
        default=float(os.environ.get("FFAC_WASSERSTEIN_CACHE_FACTOR", "0.1")),
        help="Precompute mass-preserving, geometry-weighted Wasserstein input.",
    )
    parser.add_argument(
        "--wasserstein-cache-dtype",
        choices=["float32", "float64"],
        default=os.environ.get("FFAC_WASSERSTEIN_CACHE_DTYPE", "float64"),
    )
    parser.add_argument(
        "--mass-csv-checkpoint-images",
        type=int,
        default=int(os.environ.get("FFAC_MASS_CSV_CHECKPOINT_IMAGES", "10")),
        help="Atomically checkpoint integrated_mass.csv after this many images.",
    )
    parser.add_argument(
        "--mass-prefetch-images",
        type=int,
        choices=[0, 1],
        default=int(os.environ.get("FFAC_MASS_PREFETCH_IMAGES", "0")),
        help="Read/correct one next image in a bounded background thread.",
    )
    parser.add_argument(
        "--strict-analysis-images",
        action="store_true",
        help="Fail a run when any representative time has no image in tolerance.",
    )
    parser.add_argument(
        "--mapped-drive-compat",
        action="store_true",
        help="Match imaging protocols against mapped-drive paths before UNC resolution.",
    )
    parser.add_argument(
        "--mass-results-root",
        default=os.environ.get("FFAC_MASS_RESULTS_ROOT"),
        help=(
            "Write mass products below ROOT/<run> while loading setup and "
            "calibration from the configured production results folder."
        ),
    )
    parser.add_argument(
        "--mass-calibration-candidate-dir",
        default=os.environ.get("FFAC_MASS_CALIBRATION_CANDIDATE_DIR"),
        help=(
            "Apply final_full_scale_<run>.json from this directory in memory "
            "without replacing the saved production calibration."
        ),
    )
    parser.add_argument("--mass-shard-index", type=int, default=None)
    parser.add_argument("--mass-shard-count", type=int, default=None)
    parser.add_argument("--mass-shard-id", default=None)
    return parser.parse_known_args(argv)


def _enable_mapped_drive_protocol_compatibility() -> None:
    """Allow protocol lookup before Path.resolve() rewrites mapped drives."""

    original = darsia.ProtocolledExperiment._protocol_for_path

    def mapped_drive_aware(self, path: Path):
        if self.imaging_protocols is not None:
            for folder, protocol in self.imaging_protocols.items():
                try:
                    path.relative_to(folder)
                    return protocol
                except ValueError:
                    continue
        return original(self, path)

    darsia.ProtocolledExperiment._protocol_for_path = mapped_drive_aware


def _configure_template_registration(
    context: Any,
    production_results: Path,
    run: str,
) -> tuple[Callable[[Any], Any] | None, dict[str, Any]]:
    """Apply the image registration recorded with the final calibration."""

    existing_transform = getattr(
        context,
        "_ffac_template_image_transform",
        None,
    )
    existing_metadata = getattr(
        context,
        "_ffac_template_registration_metadata",
        None,
    )
    if existing_metadata is not None:
        return existing_transform, existing_metadata

    provenance_path = production_results / _CALIBRATION_PROVENANCE
    if not provenance_path.is_file():
        metadata = {
            "enabled": False,
            "mode": "off",
            "source": "no-final-calibration-provenance",
        }
        context._ffac_template_image_transform = None
        context._ffac_template_registration_metadata = metadata
        return None, metadata

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Invalid final calibration provenance: {provenance_path}"
        ) from exc

    provenance_run = str(provenance.get("run", "")).strip().lower()
    if provenance_run and provenance_run != run:
        raise RuntimeError(
            "Final calibration provenance belongs to "
            f"{provenance_run}, not {run}: {provenance_path}"
        )
    target = str(provenance.get("template_registration", "off")).strip().lower()
    if target in {"", "0", "false", "no", "none", "off"}:
        metadata = {
            "enabled": False,
            "mode": "off",
            "source": str(provenance_path),
        }
        context._ffac_template_image_transform = None
        context._ffac_template_registration_metadata = metadata
        return None, metadata

    mode = str(
        provenance.get("template_registration_mode", "partial_affine")
    ).strip()
    calibration = context.color_to_mass_analysis
    if calibration is None:
        raise RuntimeError(
            f"{run} requires template registration but has no mass calibration."
        )

    try:
        import numpy as np

        try:
            from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
                _setup_template_registration,
                _warp_registration_image,
            )
        except ImportError:
            from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
                _setup_template_registration,
                _warp_registration_image,
            )

        requested_environment = {
            "FFAC_TEMPLATE_REGISTRATION": target,
            "FFAC_TEMPLATE_REGISTRATION_MODE": mode,
            "FFAC_TEMPLATE_REGISTRATION_STRICT": "on",
        }
        previous = {
            name: os.environ.get(name) for name in requested_environment
        }
        try:
            os.environ.update(requested_environment)
            transform, applied_mode = _setup_template_registration(
                run=run,
                config=context.config,
                fluidflower=context.fluidflower,
                calibration=calibration,
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply required {target!r} template registration for {run}."
        ) from exc

    if transform is None or applied_mode == "off":
        raise RuntimeError(
            f"Final calibration for {run} requires {target!r} registration, "
            "but no transform was produced."
        )
    matrix = np.asarray(transform, dtype=np.float32).copy()

    def transform_image(image: Any) -> Any:
        return _warp_registration_image(
            image,
            matrix,
            mode=applied_mode,
        )

    metadata = {
        "enabled": True,
        "target": target,
        "mode": applied_mode,
        "matrix": matrix.tolist(),
        "source": str(provenance_path),
    }
    context._ffac_template_image_transform = transform_image
    context._ffac_template_registration_metadata = metadata
    return transform_image, metadata


def _write_selection_manifest(
    context: Any,
    *,
    all_images: bool,
    allow_missing: bool,
    results_folder: Path | None = None,
    destination: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Apply per-time tolerances omitted by the current DarSIA selector."""

    destination = destination or (
        (results_folder or Path(context.config.data.results))
        / "provenance"
        / "mass_image_selection.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    selection: dict[str, Any] = {
        "schema": 1,
        "mode": "all" if all_images else "configured",
        "allow_missing": allow_missing,
        "requested_count": None,
        "selected_count": len(context.image_paths),
        "missing_count": 0,
        "max_abs_delta_seconds": None,
        "rows": [],
    }

    time_data = getattr(context.config.analysis, "data", None)
    requests = (
        time_data.get_times_with_uncertainty()
        if not all_images and hasattr(time_data, "get_times_with_uncertainty")
        else []
    )
    if requests:
        numerical_times = [round(float(time_h) * 3600.0, 9) for time_h, _ in requests]
        if len(numerical_times) != len(set(numerical_times)):
            raise RuntimeError("Analysis schedule contains duplicate numerical times.")

        selected: list[Path] = []
        missing: list[float] = []
        rows: list[dict[str, Any]] = []
        for requested_h, tolerance_h in requests:
            path = context.experiment.find_images_for_times(
                float(requested_h),
                tol=float(tolerance_h) * 3600.0,
            )
            if path is None:
                missing.append(float(requested_h))
                rows.append(
                    {
                        "requested_hours": float(requested_h),
                        "tolerance_seconds": float(tolerance_h) * 3600.0,
                        "path": None,
                        "actual_hours": None,
                        "delta_seconds": None,
                    }
                )
                continue
            path = Path(path)
            actual_date = context.experiment.get_datetime(path)
            actual_h = float(context.experiment.time_since_start(actual_date))
            rows.append(
                {
                    "requested_hours": float(requested_h),
                    "tolerance_seconds": float(tolerance_h) * 3600.0,
                    "path": str(path),
                    "actual_datetime": actual_date.astimezone(timezone.utc).isoformat()
                    if actual_date.tzinfo is not None
                    else actual_date.isoformat(),
                    "actual_hours": actual_h,
                    "delta_seconds": (actual_h - float(requested_h)) * 3600.0,
                }
            )
            selected.append(path)

        duplicates = sorted(
            str(path)
            for path in set(selected)
            if selected.count(path) > 1
        )
        if duplicates:
            raise RuntimeError(
                "Representative time selection reused an image for multiple targets: "
                + ", ".join(duplicates[:5])
            )
        if missing and not allow_missing:
            rendered = ", ".join(f"{value:.4f} h" for value in missing[:10])
            raise RuntimeError(
                f"Missing {len(missing)} representative image(s) within tolerance: "
                f"{rendered}"
            )
        if not selected:
            raise RuntimeError("Representative image selection returned no images.")
        context.image_paths = selected
        finite_deltas = [
            abs(float(row["delta_seconds"]))
            for row in rows
            if row["delta_seconds"] is not None
        ]
        selection.update(
            {
                "requested_count": len(requests),
                "selected_count": len(selected),
                "missing_count": len(missing),
                "max_abs_delta_seconds": max(finite_deltas, default=None),
                "rows": rows,
            }
        )

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination, selection


def _install_mass_backend(
    backend: str,
    output_dtype: str,
    *,
    titration_mode: str,
    numerical_modes: set[str],
    all_images: bool,
    allow_missing: bool,
    wasserstein_cache_factor: float,
    wasserstein_cache_dtype: str,
    csv_checkpoint_images: int,
    cuda_color_interpretation: str,
    prefetch_images: int,
    results_root: Path | None,
    calibration_candidate_dir: Path | None,
    shard_index: int | None,
    shard_count: int | None,
    shard_id: str | None,
) -> None:
    def accelerated_mass_from_context(
        context: Any,
        show: bool = False,
        stream_callback: Any = None,
        progress_callback: Any = None,
    ) -> None:
        calibration = context.color_to_mass_analysis
        if calibration is None:
            raise RuntimeError("Mass analysis context has no color-to-mass pipeline.")
        production_results = Path(context.config.data.results)
        run = production_results.name.lower()
        results_folder = (
            results_root / run if results_root is not None else production_results
        )
        shard_root = (
            results_folder / "provenance" / "mass_shards" / str(shard_id)
            if shard_id is not None
            else None
        )
        selection_path, selection = _write_selection_manifest(
            context,
            all_images=all_images,
            allow_missing=allow_missing,
            results_folder=results_folder,
            destination=(
                shard_root / "selection.json"
                if shard_root is not None
                else None
            ),
        )
        if shard_index is not None and shard_count is not None:
            all_selected = list(context.image_paths)
            context.image_paths = all_selected[shard_index::shard_count]
            if not context.image_paths:
                raise RuntimeError(
                    f"Mass shard {shard_id} selected no images from "
                    f"{len(all_selected)} available images."
                )
            selection["shard"] = {
                "id": shard_id,
                "index": shard_index,
                "count": shard_count,
                "full_selected_count": len(all_selected),
                "selected_count": len(context.image_paths),
                "image_paths": [str(path) for path in context.image_paths],
            }
            temporary = selection_path.with_name(
                f".{selection_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps(selection, indent=2), encoding="utf-8")
            temporary.replace(selection_path)
        documented_missing_seconds = sorted(
            {
                round(float(row["requested_hours"]) * 3600.0, 6)
                for row in selection.get("rows", [])
                if row.get("path") is None
            }
        )
        titration = ensure_titration_flash(context, titration_mode)
        noise_gate = ensure_aqueous_noise_gate(context, "auto")
        calibration = context.color_to_mass_analysis
        candidate = (
            apply_candidate(
                calibration,
                calibration_candidate_dir,
                run,
                fluidflower=context.fluidflower,
            )
            if calibration_candidate_dir is not None
            else None
        )
        if candidate is not None:
            noise_gate = {
                "mode": "candidate",
                "enabled": candidate.get("aqueous_noise_gate") is not None,
                "source": "candidate",
                "path": candidate.get("path"),
                "model": candidate.get("aqueous_noise_gate"),
            }
        image_transform, template_registration = (
            _configure_template_registration(
                context,
                production_results,
                run,
            )
        )
        evaluator = create_spatial_mass_evaluator(
            calibration,
            backend=backend,
            output_dtype=output_dtype,
            cuda_color_interpretation=cuda_color_interpretation,
        )
        manifest_path = write_mass_backend_manifest(
            context,
            evaluator,
            path=(
                shard_root / "backend.json"
                if shard_root is not None
                else results_folder / "provenance" / "mass_backend.json"
            ),
            runtime={
                "titration": titration,
                "aqueous_noise_gate": noise_gate,
                "production_results": str(production_results),
                "output_results": str(results_folder),
                "calibration_candidate": candidate,
                "colorchecker_refinement": getattr(
                    context.fluidflower,
                    "_ffac_colorchecker_refinement_provenance",
                    {"active": False},
                ),
                "template_registration": template_registration,
                "selection_manifest": str(selection_path),
                "selection_requested": selection["requested_count"],
                "selection_selected": selection["selected_count"],
                "numerical_modes": sorted(numerical_modes),
                "wasserstein_cache_factor": wasserstein_cache_factor,
                "wasserstein_cache_dtype": wasserstein_cache_dtype,
                "csv_checkpoint_images": csv_checkpoint_images,
                "cuda_color_interpretation": cuda_color_interpretation,
                "prefetch_images": prefetch_images,
                "shard": selection.get("shard"),
            },
        )
        logger.info(
            "Mass backend=%s dtype=%s provenance=%s",
            evaluator.backend_name,
            output_dtype,
            manifest_path,
        )
        context.color_to_mass_analysis = AcceleratedColorToMassAnalysis(
            calibration, evaluator
        )
        try:
            optimized_analysis_mass_from_context(
                context,
                results_folder=(
                    results_folder if results_root is not None else None
                ),
                numerical_modes=numerical_modes,
                wasserstein_cache_factor=wasserstein_cache_factor,
                wasserstein_cache_dtype=wasserstein_cache_dtype,
                csv_checkpoint_images=csv_checkpoint_images,
                prefetch_images=prefetch_images,
                image_transform=image_transform,
                shard_id=shard_id,
                documented_missing_seconds=documented_missing_seconds,
                show=show,
                stream_callback=stream_callback,
                progress_callback=progress_callback,
            )
        finally:
            context.color_to_mass_analysis = calibration
            evaluator.close()

    analysis_ui.analysis_mass_from_context = accelerated_mass_from_context


def main() -> None:
    backend_args, remaining = _extract_backend_args(sys.argv[1:])
    shard_values = (
        backend_args.mass_shard_index,
        backend_args.mass_shard_count,
        backend_args.mass_shard_id,
    )
    if any(value is not None for value in shard_values):
        if any(value is None for value in shard_values):
            raise ValueError(
                "Mass sharding requires --mass-shard-index, "
                "--mass-shard-count, and --mass-shard-id together."
            )
        if backend_args.mass_shard_count <= 0 or not (
            0 <= backend_args.mass_shard_index < backend_args.mass_shard_count
        ):
            raise ValueError("Invalid mass shard index/count.")
    if backend_args.mapped_drive_compat:
        _enable_mapped_drive_protocol_compatibility()
    all_images = "--all" in remaining
    numerical_modes = {
        value.strip()
        for value in backend_args.mass_numerical_modes.split(",")
        if value.strip()
    }
    if not numerical_modes:
        raise ValueError("--mass-numerical-modes must contain at least one mode.")
    candidate_dir = (
        Path(backend_args.mass_calibration_candidate_dir)
        if backend_args.mass_calibration_candidate_dir
        else None
    )
    preprocessing = _install_candidate_colorchecker_preprocessing(candidate_dir)
    remaining = _append_candidate_config_overlays(remaining, preprocessing)
    sys.argv = [sys.argv[0], *remaining]
    logger.info("Candidate preprocessing contract: %s", preprocessing)
    _install_mass_backend(
        backend=backend_args.mass_backend,
        output_dtype=backend_args.mass_output_dtype,
        titration_mode=backend_args.titration_flash,
        numerical_modes=numerical_modes,
        all_images=all_images,
        allow_missing=not backend_args.strict_analysis_images,
        wasserstein_cache_factor=backend_args.wasserstein_cache_factor,
        wasserstein_cache_dtype=backend_args.wasserstein_cache_dtype,
        csv_checkpoint_images=backend_args.mass_csv_checkpoint_images,
        cuda_color_interpretation=(
            backend_args.mass_cuda_color_interpretation
        ),
        prefetch_images=backend_args.mass_prefetch_images,
        results_root=(
            Path(backend_args.mass_results_root)
            if backend_args.mass_results_root
            else None
        ),
        calibration_candidate_dir=candidate_dir,
        shard_index=backend_args.mass_shard_index,
        shard_count=backend_args.mass_shard_count,
        shard_id=backend_args.mass_shard_id,
    )
    analysis_ui.preset_analysis(Rig)


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    main()
