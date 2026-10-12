"""ff_ac-owned mass loop with bounded I/O and Wasserstein precomputation."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import darsia
import numpy as np
import pandas as pd
from darsia.presets.workflows.analysis.image_export_formats import ImageExportFormats
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    publish_image_progress,
)
from darsia.presets.workflows.analysis.scalar_products import analysis_scalar_products
from darsia.presets.workflows.analysis.streaming import publish_stream_images
from darsia.presets.workflows.config.analysis import (
    SUPPORTED_ANALYSIS_MASS_EXPORT_MODES,
)

try:
    from darsia.fluidflower.tools.mass_wasserstein_cache import MassWassersteinCacheWriter
except ImportError:
    from darsia.fluidflower.tools.mass_wasserstein_cache import MassWassersteinCacheWriter


logger = logging.getLogger(__name__)

_DEFAULT_MASS_EXPORT_MODES = ["mass"]


def _safe_finite_max(name: str, values: Any) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(
            f"Cannot determine scalar export vmax: {name!r} has no finite values."
        )
    return float(np.max(finite))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_dataframe_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        pd.DataFrame.from_records(rows, columns=columns).to_csv(
            temporary,
            index=False,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _documented_missing_selection_seconds(results_folder: Path) -> list[float]:
    """Read requested acquisition gaps from the atomic selection manifest."""

    path = results_folder / "provenance" / "mass_image_selection.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid mass image selection manifest: {path}.") from exc
    missing: list[float] = []
    for row in payload.get("rows", []):
        if row.get("path") is not None:
            continue
        try:
            seconds = float(row["requested_hours"]) * 3600.0
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid missing-time row in mass image selection: {path}."
            ) from exc
        if not np.isfinite(seconds):
            raise RuntimeError(
                f"Non-finite missing-time row in mass image selection: {path}."
            )
        missing.append(seconds)
    return sorted(set(missing))


def _integral(values: np.ndarray, voxel_volume: Any) -> float:
    array = np.asarray(values)
    weight = np.broadcast_to(np.asarray(voxel_volume), array.shape[:2])
    valid = np.isfinite(array) & np.isfinite(weight)
    if not np.any(valid):
        return 0.0
    return float(
        np.einsum(
            "i,i->",
            np.asarray(weight[valid], dtype=np.float64),
            np.asarray(array[valid], dtype=np.float64),
            optimize=False,
        )
    )


@dataclass(frozen=True)
class _Region:
    name: str
    roi: Any
    voxel_volume: np.ndarray
    label_mask: np.ndarray | None = None

    def integrate(self, image: darsia.Image) -> float:
        values = np.asarray(image.subregion(self.roi).img)
        if self.label_mask is None:
            return _integral(values, self.voxel_volume)
        weight = np.broadcast_to(self.voxel_volume, values.shape[:2])
        return float(
            np.einsum(
                "ij,ij,ij->",
                weight,
                values,
                self.label_mask,
                optimize=False,
            )
        )


def _regions(context: Any) -> tuple[list[_Region], list[_Region]]:
    mass_config = context.config.analysis.mass
    geometry = context.fluidflower.geometry
    plain: list[_Region] = []
    labelled: list[_Region] = []
    for roi_config in mass_config.roi.values():
        sub_geometry = geometry.subregion(roi_config.roi)
        plain.append(
            _Region(
                name=roi_config.name,
                roi=roi_config.roi,
                voxel_volume=np.asarray(sub_geometry.voxel_volume),
            )
        )
    for config in mass_config.roi_and_label.values():
        if context.analysis_labels is None:
            raise RuntimeError(
                "Label-restricted mass domains require analysis labels."
            )
        sub_geometry = geometry.subregion(config.roi)
        label_mask = (
            np.asarray(context.analysis_labels.subregion(config.roi).img)
            == config.label
        )
        labelled.append(
            _Region(
                name=config.name,
                roi=config.roi,
                voxel_volume=np.asarray(sub_geometry.voxel_volume),
                label_mask=label_mask,
            )
        )
    return plain, labelled


def _select_formats(
    exporter: ImageExportFormats,
    *,
    numerical: bool,
) -> list[Any]:
    if numerical:
        return list(exporter.formats)
    return [
        spec
        for spec in exporter.formats
        if spec.type not in {"npz", "npy", "csv"}
    ]


def _prepare_export_image_without_eager_copy(
    image: darsia.Image,
    spec: Any,
) -> darsia.Image:
    """Match DarSIA export preparation without copying before a resize."""

    prepared = image
    if spec.resolution is not None:
        rows, cols = spec.resolution
        if spec.keep_ratio:
            base_rows, base_cols = image.img.shape[:2]
            cols = max(
                1,
                int(round(base_cols * rows / max(base_rows, 1))),
            )
        prepared = darsia.resize(image, shape=(rows, cols))

    dtype_name = spec.dtype
    if dtype_name is not None and dtype_name.startswith("np."):
        dtype_name = dtype_name[3:]
    if dtype_name is not None:
        dtype = np.dtype(dtype_name)
        if prepared.img.dtype != dtype:
            prepared = prepared.astype(dtype.type)
    return prepared


def _export_image(
    image: darsia.Image,
    folder: Path,
    stem: str,
    *,
    exporter: ImageExportFormats,
    retain_numerical: bool,
    scalar_write_kwargs: dict[str, float] | None,
) -> None:
    original_formats = exporter.formats
    selected_formats = _select_formats(
        exporter,
        numerical=retain_numerical,
    )
    try:
        for spec in selected_formats:
            if spec.type == "npz" and spec.resolution is None:
                out_dir = folder / spec.folder_name
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = exporter._filename_stem(image, stem, spec)
                destination = out_dir / f"{filename}.npz"
                temporary = destination.with_name(
                    f".{destination.stem}.{os.getpid()}."
                    f"{time.time_ns()}.tmp.npz"
                )
                dtype_name = spec.dtype
                if dtype_name is not None and dtype_name.startswith("np."):
                    dtype_name = dtype_name[3:]
                dtype = None if dtype_name is None else np.dtype(dtype_name)
                values = np.asarray(image.img)
                if dtype is not None and values.dtype != dtype:
                    values = values.astype(dtype, copy=False)
                try:
                    np.savez(
                        temporary,
                        array=values,
                        metadata=image.metadata(),
                    )
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                continue
            if spec.type in {"jpg", "png"}:
                prepared = _prepare_export_image_without_eager_copy(
                    image,
                    spec,
                )
                out_dir = folder / spec.folder_name
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = exporter._filename_stem(image, stem, spec)
                destination = out_dir / f"{filename}.{spec.type}"
                write_kwargs: dict[str, Any] = {}
                if spec.type == "jpg":
                    write_kwargs["quality"] = (
                        spec.quality if spec.quality is not None else 50
                    )
                else:
                    write_kwargs["compression"] = (
                        spec.compression if spec.compression is not None else 6
                    )
                if prepared.scalar:
                    cmap = exporter._resolve_cmap(spec.cmap)
                    if cmap is not None:
                        write_kwargs["cmap"] = cmap
                    if scalar_write_kwargs is not None:
                        write_kwargs.update(scalar_write_kwargs)
                prepared.write(destination, **write_kwargs)
                continue

            exporter.formats = [spec]
            exporter.export_image(
                image,
                folder,
                stem,
                supported_types={"jpg", "png", "npz", "npy", "csv"},
                jpg_quality=50,
                scalar_write_kwargs=scalar_write_kwargs,
            )
    finally:
        exporter.formats = original_formats


def _iter_corrected_images(
    fluidflower: Any,
    image_paths: list[Path],
    *,
    prefetch_images: int,
    image_transform: Callable[[Any], Any] | None = None,
):
    if prefetch_images not in {0, 1}:
        raise ValueError("Mass image prefetch must be 0 or 1.")
    if not image_paths:
        return
    if prefetch_images == 0:
        for path in image_paths:
            started = time.perf_counter()
            image = fluidflower.read_image(path)
            if image_transform is not None:
                image = image_transform(image)
            duration = time.perf_counter() - started
            yield path, image, duration, duration
        return

    def read(path: Path):
        started = time.perf_counter()
        image = fluidflower.read_image(path)
        if image_transform is not None:
            image = image_transform(image)
        return image, time.perf_counter() - started

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="ffac-image-prefetch",
    ) as executor:
        future = executor.submit(read, image_paths[0])
        for index, path in enumerate(image_paths):
            wait_started = time.perf_counter()
            image, read_duration = future.result()
            wait_duration = time.perf_counter() - wait_started
            if index + 1 < len(image_paths):
                future = executor.submit(read, image_paths[index + 1])
            yield path, image, read_duration, wait_duration


def optimized_analysis_mass_from_context(
    context: Any,
    results_folder: Path | None = None,
    *,
    numerical_modes: set[str],
    wasserstein_cache_factor: float = 0.1,
    wasserstein_cache_dtype: str = "float64",
    csv_checkpoint_images: int = 10,
    prefetch_images: int = 0,
    image_transform: Callable[[Any], Any] | None = None,
    shard_id: str | None = None,
    documented_missing_seconds: list[float] | None = None,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
) -> None:
    """Run mass analysis while preserving DarSIA's products and CSV contract."""

    config = context.config
    experiment = context.experiment
    fluidflower = context.fluidflower
    image_paths = context.image_paths
    color_to_mass_analysis = context.color_to_mass_analysis
    if config.data is None or config.analysis is None:
        raise ValueError("Mass analysis requires data and analysis configuration.")
    if config.analysis.mass is None or color_to_mass_analysis is None:
        raise ValueError("Mass analysis context is incomplete.")
    if not hasattr(color_to_mass_analysis, "co2_mass_analysis"):
        raise AttributeError(
            "Mass rescaling requires co2_mass_analysis on the color-to-mass model."
        )
    co2_mass_analysis = color_to_mass_analysis.co2_mass_analysis
    configured_results = Path(config.data.results)
    output_results = (
        Path(results_folder) if results_folder is not None else configured_results
    )

    raw_export_modes = getattr(config.analysis.mass, "export", None)
    export_modes = (
        _DEFAULT_MASS_EXPORT_MODES
        if raw_export_modes is None
        else [
            mode
            for mode in raw_export_modes
            if mode in SUPPORTED_ANALYSIS_MASS_EXPORT_MODES
        ]
    )
    output_folders = {
        "mass": (
            output_results / "mass"
            if results_folder is not None
            else Path(config.analysis.mass.folder)
        ),
        "rescaled_mass": output_results / "rescaled_mass",
        "extensive_mass": output_results / "extensive_mass",
        "extensive_rescaled_mass": output_results / "extensive_rescaled_mass",
        "saturation_g": output_results / "saturation_g",
        "rescaled_saturation_g": output_results / "rescaled_saturation_g",
        "concentration_aq": output_results / "concentration_aq",
        "rescaled_concentration_aq": output_results
        / "rescaled_concentration_aq",
    }
    for mode in export_modes:
        output_folders[mode].mkdir(parents=True, exist_ok=True)
    exporter = ImageExportFormats.from_analysis_config(
        config,
        fallback_formats=["npz", "jpg"],
    )

    vmax_mass = max(
        _safe_finite_max("solubility_co2", co2_mass_analysis.solubility_co2),
        _safe_finite_max(
            "density_gaseous_co2",
            co2_mass_analysis.density_gaseous_co2,
        ),
    )
    geometry_voxel_size = float(
        np.prod(np.asarray(fluidflower.geometry.voxel_size))
    )
    depth = getattr(fluidflower.geometry, "depth", 1.0)
    max_depth = _safe_finite_max(
        "geometry.depth",
        depth.img if hasattr(depth, "img") else depth,
    )
    vmax_extensive_mass = geometry_voxel_size * max_depth * vmax_mass
    scalar_kwargs = {
        "mass": {"vmin": 0.0, "vmax": vmax_mass},
        "rescaled_mass": {"vmin": 0.0, "vmax": vmax_mass},
        "extensive_mass": {"vmin": 0.0, "vmax": vmax_extensive_mass},
        "extensive_rescaled_mass": {
            "vmin": 0.0,
            "vmax": vmax_extensive_mass,
        },
        "saturation_g": {"vmin": 0.0, "vmax": 1.0},
        "rescaled_saturation_g": {"vmin": 0.0, "vmax": 1.0},
        "concentration_aq": {"vmin": 0.0, "vmax": 1.0},
        "rescaled_concentration_aq": {"vmin": 0.0, "vmax": 1.0},
    }

    plain_regions, labelled_regions = _regions(context)
    detected = [f"{region.name}_detected_mass" for region in plain_regions]
    detected += [f"{region.name}_detected_mass" for region in labelled_regions]
    detected_g = [
        f"{region.name}_detected_mass_g"
        for region in [*plain_regions, *labelled_regions]
    ]
    detected_aq = [
        f"{region.name}_detected_mass_aq"
        for region in [*plain_regions, *labelled_regions]
    ]
    exact = [f"{region.name}_exact_mass" for region in plain_regions]
    columns = [
        "time",
        "datetime",
        "image_stem",
        "detected_mass_total",
        "exact_mass_total",
        "detected_mass_total_rescaled",
        "mass_scaling_factor",
        *exact,
        *detected,
        *detected_g,
        *detected_aq,
    ]
    if shard_id is not None:
        if not shard_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in shard_id
        ):
            raise ValueError(f"Invalid mass-analysis shard id {shard_id!r}.")
        csv_path = (
            output_results / "sparse_data" / "mass_shards" / f"{shard_id}.csv"
        )
        timing_path = (
            output_results
            / "provenance"
            / "mass_shards"
            / shard_id
            / "timing.json"
        )
        cache_manifest_name = f"manifest.shard-{shard_id}.json"
    else:
        csv_path = output_results / "sparse_data" / "integrated_mass.csv"
        timing_path = output_results / "provenance" / "mass_timing.json"
        cache_manifest_name = "manifest.json"
    checkpoint = max(1, int(csv_checkpoint_images))
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    if documented_missing_seconds is None:
        documented_missing_seconds = _documented_missing_selection_seconds(
            output_results
        )
    else:
        documented_missing_seconds = sorted(
            {round(float(value), 6) for value in documented_missing_seconds}
        )
    cache_writer = (
        MassWassersteinCacheWriter(
            output_folders["mass"],
            fluidflower.geometry,
            wasserstein_cache_factor,
            output_dtype=wasserstein_cache_dtype,
            documented_missing_seconds=documented_missing_seconds,
            manifest_name=cache_manifest_name,
        )
        if wasserstein_cache_factor > 0.0
        else None
    )

    full_voxel_volume = np.asarray(fluidflower.geometry.voxel_volume)
    step_started = time.perf_counter()
    image_total = len(image_paths)
    completed = False
    try:
        image_stream = _iter_corrected_images(
            fluidflower,
            image_paths,
            prefetch_images=prefetch_images,
            image_transform=image_transform,
        )
        for image_index, (
            path,
            image,
            read_duration,
            read_wait_duration,
        ) in enumerate(image_stream, start=1):
            image_started = time.perf_counter()
            stage: dict[str, float] = {
                "read_correct_s": read_duration,
                "read_wait_s": read_wait_duration,
            }

            started = time.perf_counter()
            result = color_to_mass_analysis(image)
            stage["mass_evaluate_s"] = time.perf_counter() - started
            backend_timing = getattr(
                color_to_mass_analysis,
                "last_timing",
                {},
            )
            stage.update(
                {
                    f"evaluator_{key}": float(value)
                    for key, value in backend_timing.items()
                }
            )

            started = time.perf_counter()
            products, rescaled = analysis_scalar_products(
                mass_analysis_result=result,
                requested_modes={
                    "rescaled_mass",
                    "rescaled_saturation_g",
                    "rescaled_concentration_aq",
                },
                geometry=fluidflower.geometry,
                injection_protocol=experiment.injection_protocol,
                co2_mass_analysis=co2_mass_analysis,
                date=image.date,
                expert_knowledge_adapter=context.expert_knowledge_adapter,
            )
            if rescaled is None:
                raise RuntimeError("Mass rescaling unexpectedly returned no products.")
            mass = products["mass_total"]
            mass_g = products["mass_g"]
            mass_aq = products["mass_aq"]
            rescaled_mass = products["rescaled_mass"]
            stage["scalar_products_s"] = time.perf_counter() - started

            started = time.perf_counter()
            if cache_writer is not None:
                cache_writer.write(
                    mass,
                    source_stem=path.stem,
                    source_image=path,
                )
            stage["wasserstein_cache_s"] = time.perf_counter() - started

            export_images: dict[str, darsia.Image] = {
                "mass": mass,
                "rescaled_mass": rescaled_mass,
                "saturation_g": products["saturation_g"],
                "rescaled_saturation_g": products["rescaled_saturation_g"],
                "concentration_aq": products["concentration_aq"],
                "rescaled_concentration_aq": products[
                    "rescaled_concentration_aq"
                ],
            }
            if "extensive_mass" in export_modes:
                export_images["extensive_mass"] = (
                    fluidflower.geometry.make_extensive(mass)
                )
            if "extensive_rescaled_mass" in export_modes:
                export_images["extensive_rescaled_mass"] = (
                    fluidflower.geometry.make_extensive(rescaled_mass)
                )
            started = time.perf_counter()
            for mode in export_modes:
                _export_image(
                    export_images[mode],
                    output_folders[mode],
                    path.stem,
                    exporter=exporter,
                    retain_numerical=mode in numerical_modes,
                    scalar_write_kwargs=scalar_kwargs.get(mode),
                )
            stage["export_s"] = time.perf_counter() - started

            started = time.perf_counter()
            row: dict[str, Any] = {
                "time": result.time,
                "datetime": image.date,
                "image_stem": path.stem,
                "detected_mass_total": rescaled.detected_mass_total,
                "exact_mass_total": rescaled.exact_mass_total,
                "detected_mass_total_rescaled": _integral(
                    rescaled_mass.img,
                    full_voxel_volume,
                ),
                "mass_scaling_factor": rescaled.mass_scaling_factor,
            }
            for region in plain_regions:
                row[f"{region.name}_exact_mass"] = (
                    experiment.injection_protocol.injected_mass(
                        date=image.date,
                        roi=region.roi,
                    )
                )
                row[f"{region.name}_detected_mass"] = region.integrate(mass)
                row[f"{region.name}_detected_mass_g"] = region.integrate(mass_g)
                row[f"{region.name}_detected_mass_aq"] = region.integrate(mass_aq)
            for region in labelled_regions:
                row[f"{region.name}_detected_mass"] = region.integrate(mass)
                row[f"{region.name}_detected_mass_g"] = region.integrate(mass_g)
                row[f"{region.name}_detected_mass_aq"] = region.integrate(mass_aq)
            rows.append(row)
            stage["integrate_s"] = time.perf_counter() - started

            started = time.perf_counter()
            if len(rows) % checkpoint == 0 or image_index == image_total:
                _write_dataframe_atomic(csv_path, rows, columns)
            stage["csv_checkpoint_s"] = time.perf_counter() - started

            if show:
                import matplotlib.pyplot as plt

                image.show(title=f"Image at {path.stem}", delay=True)
                mass.show(title=f"Mass at {path.stem}", delay=True)
                mass_g.show(title=f"Mass G at {path.stem}", delay=True)
                mass_aq.show(title=f"Mass AQ at {path.stem}", delay=True)
                plt.show()

            started = time.perf_counter()
            publish_stream_images(
                stream_callback=stream_callback,
                image_payload={
                    "mass_source_image": image,
                    "mass_total": mass,
                    "mass_g": mass_g,
                    "mass_aq": mass_aq,
                    "rescaled_mass": rescaled_mass,
                    "rescaled_saturation_g": products["rescaled_saturation_g"],
                    "rescaled_concentration_aq": products[
                        "rescaled_concentration_aq"
                    ],
                },
                logger=logger,
                error_message=f"Failed to stream mass previews for {path!s}.",
            )
            stage["stream_s"] = time.perf_counter() - started
            stage["total_s"] = time.perf_counter() - image_started
            timing_rows.append(
                {
                    "image_index": image_index,
                    "image_stem": path.stem,
                    **stage,
                }
            )

            logger.info(
                "Processed %s at time %s in %.2f s "
                "(read %.2f, evaluate %.2f, products %.2f, cache %.2f, "
                "export %.2f, integrate %.2f)",
                path.stem,
                result.time,
                stage["total_s"],
                stage["read_correct_s"],
                stage["mass_evaluate_s"],
                stage["scalar_products_s"],
                stage["wasserstein_cache_s"],
                stage["export_s"],
                stage["integrate_s"],
            )
            for region in plain_regions:
                exact_mass = float(row[f"{region.name}_exact_mass"])
                detected_mass = float(row[f"{region.name}_detected_mass"])
                error = (
                    (detected_mass - exact_mass) / max(exact_mass, 1e-8)
                    if exact_mass
                    else 0.0
                )
                logger.info(
                    "  %s: detected=%.6f exact=%.6f error=%.2f%%",
                    region.name,
                    detected_mass,
                    exact_mass,
                    100.0 * error,
                )

            publish_image_progress(
                progress_callback,
                step="mass",
                image_path=str(path.resolve()),
                image_index=image_index,
                image_total=image_total,
                image_duration_s=stage["total_s"],
                step_elapsed_s=time.perf_counter() - step_started,
            )
            if len(timing_rows) % checkpoint == 0 or image_index == image_total:
                _write_json_atomic(
                    timing_path,
                    {
                        "schema": 1,
                        "complete": False,
                        "shard_id": shard_id,
                        "image_count": len(timing_rows),
                        "step_elapsed_s": time.perf_counter() - step_started,
                        "wasserstein_cache_factor": (
                            None
                            if cache_writer is None
                            else cache_writer.factor
                        ),
                        "documented_missing_requested_seconds": (
                            documented_missing_seconds
                        ),
                        "images": timing_rows,
                    },
                )
        completed = True
    finally:
        if rows:
            _write_dataframe_atomic(csv_path, rows, columns)
        if cache_writer is not None:
            cache_writer.close(complete=completed)
        if timing_rows:
            _write_json_atomic(
                timing_path,
                {
                    "schema": 1,
                    "complete": completed,
                    "shard_id": shard_id,
                    "image_count": len(timing_rows),
                    "step_elapsed_s": time.perf_counter() - step_started,
                    "wasserstein_cache_factor": (
                        None if cache_writer is None else cache_writer.factor
                    ),
                    "documented_missing_requested_seconds": (
                        documented_missing_seconds
                    ),
                    "images": timing_rows,
                },
            )
