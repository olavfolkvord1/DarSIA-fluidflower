#!/usr/bin/env python3
"""Materialize full-resolution phase-to-mass inputs for FF-AC and FF-UM.

This is the reference bridge between the frozen cross-series phase model and
the generic FluidFlower mass-analysis contract. Rendered phase maps are never
used as numerical inputs.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.mass_input_materialization import (
    MassInputRunWriter,
    PreparedMassInputFrame,
    write_mass_input_manifest,
)
from darsia.fluidflower.mass_inputs import inspect_mass_input_manifest
from darsia.fluidflower.project import sha256_file

from darsia.fluidflower.tools import phase_to_mass_pilot as pilot
from darsia.fluidflower.tools import (
    schema5_crossseries_reviewed_physics_blind_campaign as schema5,
)
from darsia.fluidflower.tools.observation_support import btb_acidification_coordinate
from darsia.fluidflower.tools.phase_geometry_contract import (
    frame_geometry_metadata, mass_revision_identifiers, registration_revision,
)

BRIDGE_ID = "crossseries-v5-phase-to-mass-inputs-20260901"
DEFAULT_OUTPUT_ROOT = local_cache_root() / BRIDGE_ID
DEFAULT_TARGET_TIMES_H = (0.25, 1.0, 4.0, 12.0, 24.0, 48.0)
DEFAULT_RUNS = {"ffac": "ac60", "ffum": "run18"}
CAMPAIGN_MANIFESTS = {
    "ffac": configured_tool_path('ffac', 'calibration_logs') / 'schema5_crossseries_reviewed_physics_blind_20260831/campaign_manifest.json',
    "ffum": configured_tool_path('ffum', 'analysis_v2') / 'schema5_crossseries_reviewed_physics_blind_20260831/campaign_manifest.json',
}


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _spatial_array(
    value: Any,
    shape: tuple[int, int],
    label: str,
    *,
    active_mask: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(getattr(value, "img", value), dtype=np.float32)
    raw = np.squeeze(raw)
    if raw.ndim == 0:
        raw = np.full(shape, float(raw), dtype=np.float32)
    if raw.shape != shape:
        raise RuntimeError(f"{label} shape {raw.shape} differs from {shape}")
    invalid_inside = np.asarray(active_mask, dtype=bool) & ~np.isfinite(raw)
    if np.any(invalid_inside):
        raise RuntimeError(
            f"{label} contains {np.count_nonzero(invalid_inside)} non-finite "
            "values inside the active analysis domain"
        )
    return np.where(active_mask, raw, np.float32(0.0)).astype(
        np.float32, copy=False
    )


def _rig_static_fields(
    source: Any, active_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = tuple(int(value) for value in source.baseline.shape[:2])
    geometry = getattr(source.rig, "geometry", None)
    if geometry is None or not hasattr(geometry, "depth"):
        raise RuntimeError(f"{source.run} has no explicit DarSIA depth field")
    depth = _spatial_array(
        geometry.depth, shape, "depth", active_mask=active_mask
    )
    porosity_source = getattr(source.rig, "porosity", None)
    if porosity_source is None:
        porosity_source = getattr(geometry, "porosity", None)
    if porosity_source is None:
        raise RuntimeError(f"{source.run} has no explicit DarSIA porosity field")
    permeability_source = getattr(source.rig, "permeability", None)
    if permeability_source is None:
        raise RuntimeError(f"{source.run} has no explicit DarSIA permeability field")
    porosity = _spatial_array(
        porosity_source, shape, "porosity", active_mask=active_mask
    )
    permeability = _spatial_array(
        permeability_source,
        shape,
        "permeability",
        active_mask=active_mask,
    )
    return depth, porosity, permeability


def _registered_grid(transfer: Any, source: Any) -> dict[str, Any]:
    x_m, y_m = transfer._coordinate_axes(source.rig.baseline)
    dx = float(np.median(np.abs(np.diff(x_m))))
    dy = float(np.median(np.abs(np.diff(y_m))))
    width_m = float(np.max(x_m) - np.min(x_m) + dx)
    height_m = float(np.max(y_m) - np.min(y_m) + dy)
    if not np.isfinite(width_m) or not np.isfinite(height_m):
        raise RuntimeError("Registered physical dimensions are non-finite")
    return {
        "coordinate_system": "corrected_registered_rig_grid",
        "shape_rc": [int(value) for value in source.baseline.shape[:2]],
        "width_m": width_m,
        "height_m": height_m,
        "cell_width_m": dx,
        "cell_height_m": dy,
    }


def _candidate_id(dataset: str, run: str, model_digest: str) -> str:
    manifest_path = CAMPAIGN_MANIFESTS[dataset]
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for row in payload.get("jobs") or []:
            if (
                str(row.get("run") or "").casefold() == run.casefold()
                and str(row.get("role") or "") == pilot.PHASE_CANDIDATE
            ):
                return str(row["candidate_id"])
    return f"{dataset}__{run}__{pilot.PHASE_CANDIDATE}__{model_digest[:16]}"


def _phase_provenance(
    dataset: str,
    transfer: Any,
    gas_models: Mapping[str, Any],
    co2_models: Mapping[str, Any],
) -> dict[str, Any]:
    gas_path = Path(schema5.GAS_MODEL_PATHS[pilot.PHASE_CANDIDATE])
    co2_path = Path(schema5.CO2_MODEL_PATHS[pilot.PHASE_CANDIDATE])
    campaign_path = CAMPAIGN_MANIFESTS[dataset]
    return {
        "bridge_id": BRIDGE_ID,
        "phase_candidate_role": pilot.PHASE_CANDIDATE,
        "gas_model": {
            "path": str(gas_path),
            "sha256": sha256_file(gas_path),
            "model_sha256": transfer._model_digest(
                gas_models[pilot.PHASE_CANDIDATE]
            ),
        },
        "co2_model": {
            "path": str(co2_path),
            "sha256": sha256_file(co2_path),
            "model_sha256": transfer._model_digest(
                co2_models[pilot.PHASE_CANDIDATE]
            ),
        },
        "campaign_manifest": (
            {
                "path": str(campaign_path),
                "sha256": sha256_file(campaign_path),
            }
            if campaign_path.is_file()
            else None
        ),
        "feature_contract": gas_models[
            pilot.PHASE_CANDIDATE
        ].feature_contract,
        "hierarchical_phase": True,
        "causal_physics_settings": dict(schema5.PHYSICS_SETTINGS.__dict__),
        "rendered_phase_maps_used_as_input": False,
        "numerical_resolution": "native_corrected_registered_grid",
    }


def _source_reference(
    transfer: Any,
    frame_info: Any,
    image: np.ndarray,
    correction_rows: Sequence[Mapping[str, Any]],
    *,
    registration_geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(frame_info.path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "corrected_array_sha256": transfer._array_sha256(image),
        "correction_diagnostics": [dict(row) for row in correction_rows],
        **({"registration_geometry":dict(registration_geometry)} if registration_geometry is not None else {}),
    }


def materialize_run(
    dataset: str,
    run: str,
    *,
    output_root: Path,
    target_times_h: Sequence[float],
) -> Path:
    started = time.perf_counter()
    transfer, gas_models, co2_models = pilot._configure_phase_runtime()
    transfer.TARGET_TIMES_H[dataset] = tuple(float(value) for value in target_times_h)
    phase_provenance = _phase_provenance(
        dataset, transfer, gas_models, co2_models
    )
    model_digest = str(phase_provenance["gas_model"]["model_sha256"])
    run_root = Path(output_root).expanduser().resolve() / dataset / run.casefold()

    with transfer._open_source(dataset, run, require_clearwater=True) as source:
        preflight = transfer._negative_control_preflight(
            gas_models,
            source,
            co2_models,
            hierarchical_phase=True,
        )
        if not bool(preflight.get("passed")):
            raise RuntimeError(f"{dataset}/{run} failed phase negative controls")
        aqueous_models = transfer._aqueous_models_by_candidate(
            gas_models, co2_models
        )
        aqueous_digests = {
            name: transfer._model_digest(model)
            for name, model in aqueous_models.items()
            if model is not None
        }
        aqueous_calibration = dict(
            preflight["aqueous_negative_calibration_by_candidate"]
        )
        gas_calibration = dict(
            preflight["conditional_gas_negative_calibration_by_candidate"]
        )
        evaluation_mask = transfer._phase_evaluation_mask(
            source, hierarchical_phase=True
        )
        causal_context = transfer._causal_topology_context(
            source, (pilot.PHASE_CANDIDATE,)
        )
        depth, porosity, permeability = _rig_static_fields(
            source, evaluation_mask
        )
        grid = _registered_grid(transfer, source)
        correction_identity = {
            "contract": dict(source.correction_contract),
            "baseline_sha256": transfer._array_sha256(source.baseline),
            "clearwater_contract": dict(source.clearwater_contract or {}),
        }
        geometry_identity = {
            "grid": grid,
            "depth_sha256": _array_digest(depth),
            "porosity_sha256": _array_digest(porosity),
            "permeability_sha256": _array_digest(permeability),
            **({"rig_registration": registration_revision(source.correction_contract)}
               if registration_revision(source.correction_contract) is not None else {}),
        }
        revision_identifiers = mass_revision_identifiers(source.correction_contract)
        candidate_id = _candidate_id(dataset, run, model_digest)
        writer = MassInputRunWriter(
            run_root,
            run_id=run,
            phase_candidate_id=candidate_id,
            grid=grid,
            depth=depth,
            porosity=porosity,
            permeability=permeability,
            permeability_unit="source_unit",
            provenance={
                **phase_provenance,
                "correction": correction_identity,
                "geometry": geometry_identity,
                "phase_evaluation_domain": source.phase_evaluation_domain,
            },
        )
        baseline_acid = btb_acidification_coordinate(source.baseline)
        previous: np.ndarray | None = None
        for index, frame_info in enumerate(source.frames):
            frame_started = time.perf_counter()
            image, correction_rows = pilot._transient_read(source, frame_info.path)
            registration_geometry = frame_geometry_metadata(
                source.correction_contract, correction_rows, image.shape[:2], active_mask=evaluation_mask)
            prepared = transfer._PreparedReviewFrame(
                index=index + 1,
                frame_info=frame_info,
                image=image,
                correction_rows=correction_rows,
                read_seconds=0.0,
            )
            inferred = transfer._infer_review_frame(
                prepared,
                previous,
                source.baseline,
                source.clearwater,
                gas_models,
                aqueous_models,
                aqueous_digests,
                aqueous_calibration,
                evaluation_mask,
                dataset=dataset,
                hierarchical_phase=True,
            )
            states, _ = transfer._resolve_review_phase_states(
                inferred,
                gas_calibration,
                evaluation_mask,
                causal_context,
                hierarchical_phase=True,
            )
            phase_state = states[pilot.PHASE_CANDIDATE]
            gas_probability = np.clip(
                np.asarray(phase_state.probability, dtype=np.float32), 0.0, 1.0
            )
            aqueous_probability = np.clip(
                np.asarray(
                    phase_state.aqueous_probability, dtype=np.float32
                ),
                0.0,
                1.0,
            )
            co2_probability = np.clip(
                gas_probability + aqueous_probability, 0.0, 1.0
            )
            acidification = np.asarray(
                btb_acidification_coordinate(image) - baseline_acid,
                dtype=np.float32,
            )
            acquired, injected_mass_kg = pilot._injected_mass(
                source.experiment, frame_info.path
            )
            state = source.experiment.pressure_temperature_protocol.get_state(
                acquired
            )
            gradient = (
                source.experiment.pressure_temperature_protocol.get_gradient(
                    acquired
                )
            )
            writer.add_frame(
                PreparedMassInputFrame(
                    frame_id=f"frame-{index:04d}",
                    time_hours=float(frame_info.elapsed_h),
                    injected_mass_g=1000.0 * injected_mass_kg,
                    pressure_bar=float(state.pressure),
                    temperature_c=float(state.temperature),
                    pressure_gradient_bar_per_m=float(gradient.pressure),
                    temperature_gradient_c_per_m=float(gradient.temperature),
                    optical_acidification=acidification,
                    phase_co2_probability=co2_probability,
                    phase_free_gas_probability=gas_probability,
                    corrected_rgb=image,
                    source_reference=_source_reference(
                        transfer, frame_info, image, correction_rows,
                        registration_geometry=registration_geometry,
                    ),
                )
            )
            previous = image
            print(
                f"[{dataset}/{run}] {index + 1}/{len(source.frames)} "
                f"t={frame_info.elapsed_h:.3f} h, "
                f"{time.perf_counter() - frame_started:.1f} s",
                flush=True,
            )
        run_input = writer.finalize()
        from darsia.fluidflower.phase_pixel_area import carry_mass_area_bindings
        run_input = carry_mass_area_bindings(run_input)

    upstream = {
        "phase_revision_id": f"{schema5.CAMPAIGN_ID}:{candidate_id}",
        "phase_revision_sha256": (
            sha256_file(CAMPAIGN_MANIFESTS[dataset])
            if CAMPAIGN_MANIFESTS[dataset].is_file()
            else model_digest
        ),
        "image_correction_revision_id": revision_identifiers["image_correction_revision_id"],
        "image_correction_revision_sha256": _json_digest(correction_identity),
        "geometry_revision_id": revision_identifiers["geometry_revision_id"],
        "geometry_revision_sha256": _json_digest(geometry_identity),
    }
    manifest_path = write_mass_input_manifest(
        run_root / "mass_input_manifest.json",
        run_inputs=[run_input],
        upstream=upstream,
        provenance={
            "bridge_id": BRIDGE_ID,
            "dataset": dataset,
            "run": run.casefold(),
            "elapsed_seconds": time.perf_counter() - started,
            "scientific_contract": {
                "phase_model_scope": "visible_registered_camera_plane_2d",
                "mass_or_concentration_supplied_by_phase_model": False,
                "depth_or_3d_supplied_by_phase_model": False,
                "injected_mass_use": "diagnostic_and_later_calibration_only",
                "native_numerical_fields": True,
            },
        },
    )
    report = inspect_mass_input_manifest(
        str(manifest_path),
        run_bindings=[
            {"run_id": run.casefold(), "phase_candidate_id": candidate_id}
        ],
        maximum_images_per_run=len(run_input["target_times_hours"]),
        require_preview_images=True,
    )
    if not report["valid"]:
        raise RuntimeError(f"Prepared manifest failed validation: {report['issues']}")
    print(f"VALID {manifest_path}", flush=True)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("ffac", "ffum", "all"), default="all"
    )
    parser.add_argument("--ffac-run", default=DEFAULT_RUNS["ffac"])
    parser.add_argument("--ffum-run", default=DEFAULT_RUNS["ffum"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_TIMES_H),
        help="Requested experiment times in hours.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    datasets = ("ffac", "ffum") if args.dataset == "all" else (args.dataset,)
    outputs: dict[str, str] = {}
    for dataset in datasets:
        run = args.ffac_run if dataset == "ffac" else args.ffum_run
        outputs[dataset] = str(
            materialize_run(
                dataset,
                run,
                output_root=args.output_root,
                target_times_h=args.times,
            )
        )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
