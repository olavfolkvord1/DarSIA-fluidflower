"""Compute movie frames with frozen September 11 phases and September 12 optics.

The movie adds acquired images, not calibration observations. Only the original
reviewed frames advance temporal model history; inserted frames are evaluated
on a private copy of the immediately preceding reviewed history. Optical and
physical concentration parameters, reference images, and dilution anchor stay
fixed. Run one series per process, and use new output directories only.
"""
from __future__ import annotations
from darsia.fluidflower.tools._paths import load_snapshot_module

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import copy
from datetime import datetime
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "4")

REPO = series_root("ffac")

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower.tools.phase_mass_overnight_data import read, write, sha, small, crop_rgb, decode_labels
from darsia.fluidflower.tools.phase_mass_joint_optics import build_joint_reference, joint_optical_signals
from darsia.fluidflower.tools.phase_mass_dark_channel_checks import fixed_concentration
from darsia.fluidflower.tools.concentration_movie_input_cache import MovieInputCache

ROOT = local_cache_root() / 'phase_diagnostics/concentration_movies_20260912'
INVENTORY = local_cache_root() / 'phase_diagnostics/full_movie_inventory_20260912'
PARENT = local_cache_root() / 'phase_diagnostics/phase_mass_overnight_20260912'
PHYSICS = local_cache_root() / 'phase_diagnostics/phase_mass_physics_20260912'
RUNS = {"run18": "ffum", "ac61": "ffac"}
POLICY = "frozen-reviewed-anchor-history-v1"


def save_npz(path, **arrays):
    """Atomic level-one compression, preserving all array dtypes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for key, array in arrays.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    temporary.replace(path)


def load_rgb(path, box):
    x, y, w, h = box
    with Image.open(path) as image:
        if image.size == (w, h):
            return np.asarray(image.convert("RGB"))
        return np.asarray(image.crop((x, y, x+w, y+h)).convert("RGB"))


def encode_review_rgb(rgb, manifest, run, elapsed_h=0.):
    """Use the original native review JPEG encoding, including canvas alignment.

    JPEG's 8-pixel block alignment changes if the ROI is encoded in isolation.
    Encode the unchanged original canvas geometry, then decode/crop with PIL,
    just as the frozen phase materializer did.
    """
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer
    width, height = manifest["canvas_wh"]
    canvas = transfer._plain_frame(np.clip(np.asarray(rgb, np.float32), 0, 1),
        run=run, index=0, count=1, elapsed_h=elapsed_h,
        width=width, height=height, native=True)
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    if not success:
        raise RuntimeError("Native corrected JPEG encoding failed")
    x, y, w, h = manifest["content_box_xywh"]
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        result = np.asarray(image.crop((x, y, x+w, y+h)).convert("RGB"))
    return result, encoded


class FrozenConcentration:
    """Cheap optical/mass stage; accepts native categorical labels or volumes."""

    def __init__(self, run, root=ROOT):
        self.run, self.dataset, self.root = run, RUNS[run], Path(root)
        self.source = PARENT / self.dataset / run
        self.path = PHYSICS / self.dataset / run
        self.meta = read(self.source / "prepared.json")
        self.fit_path = self.path / "fits/joint_chromatic_od__mass_only/result.json"
        self.fit = read(self.fit_path)
        self.parameters = self.fit["parameters"]
        self.frames = self.meta["frames"]
        self.width = int(self.meta["analysis_width"])
        self.arrays = {key: np.load(self.source / (key + ".npy"), mmap_mode="r")
                       for key in ("rgb", "aq_volume", "gas_volume")}
        self.shape = self.arrays["aq_volume"].shape[1:]
        with np.load(self.source / "references.npz") as z:
            self.references = {key: z[key] for key in z.files}
        self.reference = build_joint_reference(self.references["baseline"],
            self.references["clearwater"], self.references["facies"],
            domain_mask=self.references["pore_volume"] > 0)
        if not self.reference.usable:
            raise ValueError("Frozen joint optical reference is unusable")
        maximum = max(row["injected_mass_g"] for row in self.frames)
        self.anchor = next(row for row in self.frames if row["injected_mass_g"] >= .9999*maximum)
        self.native_manifest = read(self.meta["manifest"])
        if any(row.get("rig_domain_revision") for row in self.native_manifest.get("frame_sources", ())):
            raise ValueError("Frozen concentration movie has fixed-grid physical volumes and cannot use the new rig-domain revision without source-bound area and physical-coverage recalibration.")
        self.model_root = Path(self.meta["model_root"])
        self.grid = self.meta["grid"]
        self.pixel_area = self.grid["width_m"]*self.grid["height_m"] / np.prod(self.shape)
        self.provenance = {
            "format": "frozen-concentration-movie-frame-v1", "run": run, "dataset": self.dataset,
            "model_id": self.meta["model_id"], "model_root": str(self.model_root),
            "concentration_curve": str(self.fit_path), "curve_sha256": sha(self.fit_path),
            "parameters": self.parameters, "optical_reference_sha256": sha(self.source / "references.npz"),
            "original_prepared_sha256": sha(self.source / "prepared.json"),
            "temporal_policy": POLICY, "dilution_anchor": self.anchor,
            "analysis_width": self.width, "shape_hw": list(self.shape),
            "grid": self.grid, "additional_smoothing": False, "refitted": False,
            "concentration_units": "g/L", "phase_fractions": "fraction of pixel pore volume, not model probability",
            "reference_policy": "baseline plus clearwater optical reference; cumulative injected mass at each acquisition",
            "concentration_status": "inverse mass-calibrated estimate, no independent local chemical standard",
        }
        method_binding = {"temporal_policy": POLICY, "curve_sha256": self.provenance["curve_sha256"],
            "references_sha256": self.provenance["optical_reference_sha256"],
            "model_validation_sha256": sha(self.model_root / "validation/validation_binding.json"),
            "optics_code_sha256": sha(project_path('scripts/phase_mass_joint_optics.py')),
            "concentration_code_sha256": sha(project_path('scripts/phase_mass_dark_channel_checks.py')),
            "inference_backend_sha256": sha(project_path('scripts/phase_inference_benchmark_backends.py')),
            "analysis_width": self.width, "dilution_anchor": self.anchor,
            "numeric_contract": "unchanged-curve-times-solubility-with-frozen-pore-volume-dilution-anchor-v1",
            "phase_contract": "original-reviewed-labels-plus-inserted-frame-disposable-reviewed-anchor-history"}
        self.provenance["method_binding"] = method_binding
        self.provenance["method_sha256"] = hashlib.sha256(json.dumps(method_binding, sort_keys=True).encode()).hexdigest()

    def dilution(self, elapsed_h, affected_volume):
        if elapsed_h <= self.anchor["elapsed_h"]:
            return 1.
        return min(1., max(affected_volume/self.anchor["affected_pore_volume_m3"], 1.) **
                   (-self.parameters["beta"]))

    def fields(self, rgb, av, gv, solubility, density, record, *, frozen_index=None):
        if frozen_index is not None:
            reduced = np.asarray(self.arrays["rgb"][frozen_index])
        else:
            if np.asarray(rgb).dtype != np.uint8:
                raise ValueError("Current RGB must be the native corrected review uint8 representation")
            reduced = small(np.asarray(rgb, np.float32)/255, self.width)
        av, gv = np.asarray(av, np.float32), np.asarray(gv, np.float32)
        if reduced.shape[:2] != self.shape or av.shape != self.shape or gv.shape != self.shape:
            raise ValueError("Optical and physical analysis grids differ")
        if not all(np.isfinite(array).all() for array in (reduced, av, gv, solubility, density)):
            raise ValueError("Nonfinite physical or optical field")
        if (av < 0).any() or (gv < 0).any():
            raise ValueError("Negative phase pore volume")
        affected = float(av.sum(dtype=np.float64) + gv.sum(dtype=np.float64))
        d = self.dilution(float(record["elapsed_h"]), affected)
        signal = joint_optical_signals(reduced, self.reference)["joint_chromatic_od"]
        c = fixed_concentration(signal, self.parameters, d, solubility)
        sg = self.parameters["gas_saturation"]
        gas = float(np.sum(gv.astype(float)*density)*1000*sg)
        gas_liquid = float(np.sum(gv.astype(float)*solubility)*1000*(1-sg))
        aqueous = float(np.sum(c.astype(float)*av)*1000) + gas_liquid
        pv = self.references["pore_volume"]
        aq_fraction = np.divide(av, pv, out=np.zeros_like(av), where=pv > 0)
        gas_fraction = np.divide(gv, pv, out=np.zeros_like(gv), where=pv > 0)
        if np.max(aq_fraction+gas_fraction) > 1.0001:
            raise ValueError("Phase fractions exceed available pore volume")
        metadata = {**record, "run": self.run, "dataset": self.dataset,
            "dilution_factor": d, "affected_pore_volume_m3": affected,
            "aqueous_g": aqueous, "gas_g": gas, "total_g": aqueous+gas,
            "aqueous_mass_g": aqueous, "gas_mass_g": gas, "total_mass_g": aqueous+gas,
            "residual_liquid_in_gas_g": gas_liquid, "frozen_frame_index": frozen_index,
            "max_visible_concentration_g_l": float(c[av > 0].max()) if (av > 0).any() else 0.,
            "concentration_curve_sha256": self.provenance["curve_sha256"],
            "method_sha256": self.provenance["method_sha256"],
            "model_id": self.meta["model_id"], "temporal_policy": POLICY,
            "estimated_concentration": True, "registered_corrected_photograph": True}
        return {"photograph_rgb": np.rint(np.clip(reduced, 0, 1)*255).astype(np.uint8),
                "concentration_g_l": c, "aqueous_fraction": aq_fraction,
                "gas_fraction": gas_fraction}, metadata

    def frozen_fields(self, index, extra=None):
        row = {**self.frames[index], "source_path":
               self.native_manifest["frame_sources"][index].get("source_path"),
               "source_rgb_sha256": self.meta["source_assets"][index]["rgb_sha256"],
               "source_phase_sha256": self.meta["source_assets"][index]["map_sha256"],
               "source_kind": "original_reviewed_frame", **(extra or {})}
        with np.load(self.source / f"properties_{index:03d}.npz") as z:
            solubility, density = z["solubility"], z["density"]
        arrays, metadata = self.fields(None, self.arrays["aq_volume"][index],
            self.arrays["gas_volume"][index], solubility, density, row, frozen_index=index)
        return arrays, metadata

    def write_frame(self, index, arrays, metadata, subdir="fields"):
        target = self.root / self.run / subdir / f"frame_{index:05d}.npz"
        save_npz(target, **arrays, record_json=np.asarray(json.dumps(metadata, allow_nan=False)))
        write(target.with_suffix(".json"), {**metadata, "field_path": str(target), "field_bytes": target.stat().st_size})
        return target

    def movie_manifest(self, *, include_blacklisted=False):
        inventory = read(INVENTORY / (self.run + ".json"))
        frames = [{"source_name": "Pre-injection baseline", "source_path": None,
                   "elapsed_h": float(self.frames[0]["elapsed_h"]), "source_kind": "original_baseline",
                   "frozen_frame_index": 0}]
        frames.extend(row for row in inventory["frames"] if row["post_injection_through_72h"]
                      and (include_blacklisted or not row["blacklisted"]))
        by_name = {row["source_name"].casefold(): index for index, row in enumerate(self.frames)}
        for index, row in enumerate(frames):
            row.update(index=index, frozen_frame_index=(0 if index == 0 else by_name.get(row["source_name"].casefold())),
                       fields_path=str(self.root / self.run / "fields" / f"frame_{index:05d}.npz"),
                       record_path=str(self.root / self.run / "fields" / f"frame_{index:05d}.json"))
        manifest = {**self.provenance, "expected_total": len(frames), "frames": frames,
                    "complete": False, "include_blacklisted": include_blacklisted,
                    "source_inventory": str(INVENTORY / (self.run + ".json")),
                    "source_inventory_sha256": sha(INVENTORY / (self.run + ".json")),
                    "blacklist_excluded": [row for row in inventory["frames"]
                        if row["post_injection_through_72h"] and row["blacklisted"] and not include_blacklisted]}
        write(self.root / self.run / "manifest.json", manifest)
        write(self.root / self.run / "method.json", self.provenance)
        return manifest


class NativeCorrection:
    """The unchanged DarSIA correction chain and native pore-depth grid."""

    def __init__(self, frozen, *, root=None):
        self.frozen = frozen
        self.root = Path(root or frozen.root)
        self.stack = ExitStack()
        self.input_cache = MovieInputCache(self.root, self.frozen.run)

    def __enter__(self):
        from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer
        from darsia.fluidflower.tools.phase_mass_overnight_data import open_reference_source
        from darsia.fluidflower.tools.phase_geometry_contract import require_matching_geometry
        self.transfer = transfer
        started = time.perf_counter()
        private_root = self.root / self.frozen.run / "worker_sources" / str(os.getpid())
        manager = (transfer._open_source("ffac", self.frozen.run, require_clearwater=False)
                   if self.frozen.dataset == "ffac" else
                   open_reference_source(transfer, private_root, self.frozen.dataset, self.frozen.run))
        # The legacy source opener regenerates an identical shared TOML overlay.
        # Read/verify that existing overlay instead: parallel movie workers must
        # neither rewrite scientific inputs nor race over its fixed temp name.
        fieldwise = transfer.optical_control.fieldwise
        original_overlay = fieldwise.write_overlay
        def existing_overlay(output_root, profile, run):
            path = fieldwise.overlay_path(output_root, profile, run).resolve()
            folder = fieldwise._toml_filesystem_path(fieldwise.embedding_root(output_root, profile, run).resolve())
            expected = '[color.path.relative_colorpath_labels]\n' + f'calibration_folder = "{folder}"\n'
            if path.read_text(encoding="utf-8-sig") != expected:
                raise ValueError(f"Existing original correction overlay changed: {path}")
            return path
        try:
            fieldwise.write_overlay = existing_overlay
            self.source = self.stack.enter_context(manager)
        finally:
            fieldwise.write_overlay = original_overlay
        self.geometry_binding = require_matching_geometry(self.frozen.native_manifest,
            self.source.correction_contract, context="full movie frozen optical model")
        native_shape = tuple(self.frozen.grid["shape_rc"])
        if self.source.baseline.shape[:2] != native_shape:
            raise ValueError("New source geometry differs from frozen model")
        facies = np.asarray(self.source.rig.facies.img)
        domain = ~np.isin(facies, [0] if self.frozen.dataset == "ffac" else [1])
        depth = np.asarray(getattr(self.source.rig.geometry.depth, "img", self.source.rig.geometry.depth), np.float32)
        porosity = np.asarray(getattr(self.source.rig.porosity, "img", self.source.rig.porosity), np.float32)
        self.pore_depth = np.where(domain, depth*porosity, 0).astype(np.float32)
        self.pore_depth = np.broadcast_to(self.pore_depth, native_shape)
        reduced = small(self.pore_depth, self.frozen.width)*self.frozen.pixel_area
        if not np.allclose(reduced, self.frozen.references["pore_volume"], rtol=2e-5, atol=1e-12):
            raise ValueError("Reconstructed pore geometry differs from original concentration geometry")
        self.open_seconds = time.perf_counter()-started
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def state(self, row):
        from darsia.fluidflower.mass_materialization import _mass_analysis
        # Inventory timestamps were read from each acquisition. Z: and UNC
        # aliases name identical files but fail the experiment's path matcher.
        acquired = datetime.fromisoformat(row["acquired"])
        protocol = self.source.experiment.injection_protocol
        try:
            injected = protocol.injected_mass(date=acquired)
        except TypeError:
            injected = protocol.injected_mass(acquired)
        mass = float(np.sum(np.asarray(injected, np.float64)))
        p = self.source.experiment.pressure_temperature_protocol
        state, gradient = p.get_state(acquired), p.get_gradient(acquired)
        record = {**row, "injected_mass_g": mass*1000, "acquired": str(acquired),
            "pressure_bar": float(state.pressure), "temperature_c": float(state.temperature),
            "pressure_gradient_bar_per_m": float(gradient.pressure),
            "temperature_gradient_c_per_m": float(gradient.temperature)}
        g = self.frozen.grid
        analyzer = _mass_analysis(self.frozen.shape, g["width_m"], g["height_m"],
            record["pressure_bar"], record["temperature_c"],
            record["pressure_gradient_bar_per_m"], record["temperature_gradient_c_per_m"])
        solubility = np.broadcast_to(np.asarray(analyzer.solubility_co2), self.frozen.shape).astype(np.float32)
        density = np.broadcast_to(np.asarray(analyzer.density_gaseous_co2), self.frozen.shape).astype(np.float32)
        return record, solubility, density

    def correct(self, row):
        import darsia
        started = time.perf_counter()
        path, input_receipt = self.input_cache.stage(row["source_path"])
        digest = input_receipt["source_sha256"]
        image = darsia.imread(path, transformations=self.source.rig.corrections,
            reference_date=self.source.rig.reference_date, name=path.name, dtype=np.float32)
        float_rgb = np.asarray(image.img, np.float32)[..., :3]
        corrected_seconds = time.perf_counter()-started
        rgb, encoded = encode_review_rgb(float_rgb, self.frozen.native_manifest,
                                         self.frozen.run, float(row["elapsed_h"]))
        record = {**row, **input_receipt, "source_sha256": digest, "source_kind": "full_native_correction",
            "correction_seconds": corrected_seconds,
            "encoding_seconds": time.perf_counter()-started-corrected_seconds,
            "native_corrected_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "native_shape_hw": list(rgb.shape[:2]), "correction_contract": self.source.correction_contract,
            "geometry_binding": self.geometry_binding,
            "correction_diagnostics": self.transfer._correction_diagnostics(self.source.rig)}
        return rgb, record, encoded

    def volumes(self, masks):
        result = [small(self.pore_depth*np.asarray(masks[key], np.float32), self.frozen.width)*self.frozen.pixel_area
                  for key in ("aqueous", "gas")]
        native = float(np.sum(self.pore_depth*(masks["aqueous"] | masks["gas"]), dtype=np.float64))
        native *= self.frozen.grid["width_m"]*self.frozen.grid["height_m"]/self.pore_depth.size
        reduced = sum(float(a.sum(dtype=np.float64)) for a in result)
        if not np.isclose(native, reduced, rtol=2e-5, atol=1e-12):
            raise ValueError("Native to analysis-grid pore volume is not conserved")
        return result


def _sequence_nodes(sequence):
    """Walk concrete storage only; boundary wrapper has forwarding __getattr__."""
    seen, stack = set(), [sequence]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for name in ("parent", "inherited", "base", "provider", "exposure_history"):
            child = vars(node).get(name)
            if child is not None and hasattr(child, "__dict__"):
                stack.append(child)


def fork_sequence(sequence):
    """Copy all history; share immutable geometry/reference fields only."""
    memo = {}
    static_names = {"context", "settings", "boundary_settings", "calibration", "calibration_baseline",
                    "native_reference_chroma", "native_reference_brightness", "reference",
                    "_facies_fields", "valid", "facies", "material_cost", "_edges", "paths"}
    for node in _sequence_nodes(sequence):
        for name, value in vars(node).items():
            if name in static_names:
                memo[id(value)] = value
            elif name == "last_diagnostics":
                memo[id(value)] = {}
            elif name == "last_result" and value is not None:
                # A displayed immutable result is not causal history and need
                # not be copied into a disposable reference calculation.
                memo[id(value)] = None
    return copy.deepcopy(sequence, memo)


def history_digest(sequence):
    digest = hashlib.sha256()
    names = {"owner", "history", "gas_anchors", "last_time", "last_observed_h", "absent_since_h",
             "last_strong_h", "first_arrival_h", "previous_supported", "directions",
             "optical_aqueous_memory", "front_memory", "reference_time", "arrival", "frame_number", "ever_co2",
             "previous_time", "first_observed_h"}
    for node in _sequence_nodes(sequence):
        for name, value in sorted(vars(node).items()):
            if name not in names:
                continue
            digest.update(name.encode())
            values = value if isinstance(value, list) else [value]
            for item in values:
                digest.update(np.asarray(item).tobytes() if isinstance(item, np.ndarray) else repr(item).encode())
    return digest.hexdigest()


class FrozenPhaseSequence:
    """Original optical predictor plus causal history from reviewed anchors."""

    def __init__(self, frozen, *, backend="cuda"):
        from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
        from darsia.fluidflower.phase_runtime import load_physics_profile, load_run_context, physics_runtime_identity, phase_physics_sequence
        from darsia.fluidflower.tools.phase_inference_benchmark_backends import CudaFeaturePredictor, CpuFeaturePredictor, partition
        self.frozen, self.backend, self.partition = frozen, backend, partition
        self.model = HierarchicalPhaseModel.load(frozen.model_root, adapter="generic")
        self.profile = load_physics_profile(frozen.model_root)
        self.binding = read(frozen.model_root / "validation/validation_binding.json")
        actual = physics_runtime_identity(self.profile)
        if actual != self.binding["physics_runtime"]:
            raise ValueError("Installed physical sequence changed from original model runtime")
        context, _ = load_run_context(frozen.model_root, self.profile, frozen.dataset, frozen.run,
                                      native_manifest_sha256=frozen.meta["manifest_sha256"])
        self.sequence = phase_physics_sequence(context, self.profile)
        self.box = frozen.native_manifest["content_box_xywh"]
        first_asset = Path(frozen.meta["source_assets"][0]["rgb"])
        baseline_path = first_asset.parent / "pre_injection.jpg"
        if not baseline_path.exists():
            baseline_path = Path(frozen.meta["manifest"]).parents[2] / "native_references" / frozen.run / "pre_injection.jpg"
        self.baseline = load_rgb(baseline_path, self.box)
        self.sequence.bind_reference(self.baseline)
        if backend == "cuda":
            wrapper = frozen.model_root / "validation/resource_wrapper_snapshot/materialize_shared_gpu.py"
            module = load_snapshot_module("movie_original_gpu_wrapper", wrapper)
            self.predictor = module.TransientCudaFeaturePredictor(self.model, self.baseline)
        elif backend == "cpu":
            self.predictor = CpuFeaturePredictor(self.model, self.baseline)
        else:
            raise ValueError("Unknown optical backend")
        self.previous = None
        self.previous_index = None
        self.completed_anchor = -1
        self.replay_reports = []
        self.by_name = {r["source_name"].casefold(): i for i, r in enumerate(frozen.frames)}

    def _predict(self, rgb, key):
        started = time.perf_counter()
        raw = self.predictor.predict(rgb, self.previous, current_key=key,
                                     previous_key=("anchor", self.previous_index))
        raw = self.partition(np.stack([raw["co2_presence_probability"],
                                      raw["gas_probability_conditional_on_co2"]], axis=-1))
        return raw, time.perf_counter()-started

    @staticmethod
    def _apply(sequence, raw, elapsed_h, rgb):
        if hasattr(sequence, "apply_with_reference"):
            return sequence.apply_with_reference(raw, elapsed_h, rgb)
        return sequence.apply(raw, elapsed_h)

    def advance_through(self, elapsed_h, *, inclusive=False):
        from darsia.fluidflower.phase_rendering import two_phase_masks
        for index in range(self.completed_anchor+1, len(self.frozen.frames)):
            frame = self.frozen.frames[index]
            if frame["elapsed_h"] > elapsed_h or (not inclusive and frame["elapsed_h"] == elapsed_h):
                break
            started = time.perf_counter()
            asset = self.frozen.meta["source_assets"][index]
            rgb = load_rgb(asset["rgb"], self.box)
            raw, seconds = self._predict(rgb, ("anchor", index))
            output, report = self._apply(self.sequence, raw, frame["elapsed_h"], rgb)
            masks = two_phase_masks(output)
            labels = decode_labels(asset["map"], self.box)
            mismatch = {key: int(np.count_nonzero(masks[key] != (labels == value)))
                        for key, value in (("aqueous", 1), ("gas", 2))}
            if any(mismatch.values()):
                raise ValueError(f"Original anchor phase replay mismatch {self.frozen.run}/{index}: {mismatch}")
            self.previous, self.previous_index = rgb, index
            self.completed_anchor = index
            item = {"frame": index, "elapsed_h": frame["elapsed_h"], "source_name": frame["source_name"],
                    "raw_inference_seconds": seconds, "total_seconds": time.perf_counter()-started,
                    "exact_labels": True, "mismatched_pixels": mismatch}
            self.replay_reports.append(item)
            print("ANCHOR " + json.dumps(item), flush=True)
            del raw, output, masks, labels
            gc.collect()

    def evaluate(self, rgb, row):
        from darsia.fluidflower.phase_rendering import two_phase_masks
        elapsed = float(row["elapsed_h"])
        self.advance_through(elapsed)
        before = history_digest(self.sequence)
        started = time.perf_counter()
        trial = fork_sequence(self.sequence)
        raw, inference_seconds = self._predict(rgb, ("inserted", row["source_name"]))
        physics_start = time.perf_counter()
        output, report = self._apply(trial, raw, elapsed, rgb)
        masks = two_phase_masks(output)
        if history_digest(self.sequence) != before:
            raise ValueError("Inserted frame changed canonical reviewed history")
        result = {"raw_inference_seconds": inference_seconds,
                  "physics_seconds": time.perf_counter()-physics_start,
                  "phase_seconds": time.perf_counter()-started,
                  "previous_reviewed_anchor": self.previous_index,
                  "reviewed_history_sha256": before, "canonical_history_unchanged": True,
                  "physics_report": report}
        del raw, output, trial
        gc.collect()
        return masks, result

    def close(self):
        if hasattr(self.predictor, "close"):
            self.predictor.close()


def benchmark(run, root=ROOT, backend="cuda"):
    """Two early raw acquisitions plus an original anchor parity check."""
    frozen = FrozenConcentration(run, root)
    inventory = read(INVENTORY / (run + ".json"))
    first = [r for r in inventory["frames"] if r["post_injection_through_72h"]][:2]
    result = {"run": run, "temporal_policy": POLICY, "rows": [], "complete": False}
    dest = Path(root) / run / "benchmark"
    dest.mkdir(parents=True, exist_ok=True)
    with NativeCorrection(frozen) as correction:
        result["source_open_seconds"] = correction.open_seconds
        print("SOURCE_OPEN " + str(correction.open_seconds), flush=True)
        phase = FrozenPhaseSequence(frozen, backend=backend)
        try:
            for index, row in enumerate(first, 1):
                started = time.perf_counter()
                rgb, record, encoded = correction.correct(row)
                print("CORRECTION " + json.dumps({k: record[k] for k in ("source_name", "correction_seconds", "encoding_seconds")}), flush=True)
                masks, timings = phase.evaluate(rgb, row)
                av, gv = correction.volumes(masks)
                state, solubility, density = correction.state(record)
                fields, metadata = frozen.fields(rgb, av, gv, solubility, density, {**state, **timings})
                path = frozen.write_frame(index, fields, metadata, "benchmark")
                result["rows"].append({**metadata, "field_path": str(path), "total_seconds": time.perf_counter()-started})
                write(dest / "benchmark.json", result)
                print("FRAME " + json.dumps({"source": row["source_name"], "total_seconds": result["rows"][-1]["total_seconds"], **timings}, default=str), flush=True)
            # Replay one additional original positive anchor, which validates
            # features, baseline/previous semantics, physical state and labels.
            phase.advance_through(frozen.frames[1]["elapsed_h"], inclusive=True)
            result["anchor_replay"] = phase.replay_reports
            # Newly corrected anchor should reproduce the historical JPEG ROI.
            anchor_row = next(r for r in inventory["frames"] if r["source_name"] == frozen.frames[1]["source_name"])
            rgb, record, _ = correction.correct(anchor_row)
            old = load_rgb(frozen.meta["source_assets"][1]["rgb"], phase.box)
            delta = np.abs(rgb.astype(np.int16)-old.astype(np.int16))
            result["correction_anchor_parity"] = {"source_name": anchor_row["source_name"],
                "mean_absolute_dn": float(delta.mean()), "maximum_absolute_dn": int(delta.max()),
                "fraction_pixels_changed": float(np.mean(np.any(delta != 0, axis=-1))), **record}
            result["complete"] = True
            write(dest / "benchmark.json", result)
            write(Path(root) / run / "method.json", frozen.provenance)
        finally:
            phase.close()
    return result


_WORKER = None


class MovieWorker:
    """One long-lived correction and phase context for increasing movie rows."""

    def __init__(self, run, root=ROOT, backend="cuda", threads=2):
        from threadpoolctl import threadpool_limits
        cv2.setNumThreads(threads)
        self.thread_limit = threadpool_limits(limits=threads)
        self.frozen = FrozenConcentration(run, root)
        self.correction = NativeCorrection(self.frozen).__enter__()
        self.phase = FrozenPhaseSequence(self.frozen, backend=backend)
        self.last_index = -1

    def close(self):
        self.phase.close()
        self.correction.__exit__(None, None, None)
        self.thread_limit.restore_original_limits()

    def compute(self, row, *, subdir="fields"):
        index = int(row["index"])
        if index <= self.last_index:
            raise ValueError("A worker's movie rows must increase strictly")
        self.last_index = index
        started = time.perf_counter()
        target = self.frozen.root / self.frozen.run / subdir / f"frame_{index:05d}.npz"
        metadata_path = target.with_suffix(".json")
        if target.exists() and metadata_path.exists():
            cached = read(metadata_path)
            if (cached.get("model_id") == self.frozen.meta["model_id"]
                    and cached.get("concentration_curve_sha256") == self.frozen.provenance["curve_sha256"]
                    and cached.get("temporal_policy") == POLICY
                    and cached.get("source_name") == row["source_name"]
                    and abs(float(cached["elapsed_h"])-float(row["elapsed_h"])) < 1e-9):
                return {"index": index, "source_name": row["source_name"], "cached": True,
                        "fields_path": str(target), "elapsed_h": row["elapsed_h"]}
            raise ValueError(f"Existing movie field has a different frozen contract: {target}")
        frozen_index = row.get("frozen_frame_index")
        if frozen_index is not None:
            frozen_index = int(frozen_index)
            self.phase.advance_through(self.frozen.frames[frozen_index]["elapsed_h"], inclusive=True)
            record = dict(row)
            if record.get("source_path"):
                _, input_receipt = self.correction.input_cache.stage(record["source_path"])
                record.update(input_receipt)
            else:
                record["source_sha256"] = self.frozen.meta["source_assets"][0]["rgb_sha256"]
                record["source_hash_kind"] = "composite baseline corrected JPEG"
            arrays, metadata = self.frozen.frozen_fields(frozen_index, record)
            metadata["original_anchor_labels_verified"] = True
        else:
            rgb, record, _ = self.correction.correct(row)
            masks, timings = self.phase.evaluate(rgb, row)
            av, gv = self.correction.volumes(masks)
            record, solubility, density = self.correction.state(record)
            arrays, metadata = self.frozen.fields(rgb, av, gv, solubility, density,
                                                 {**record, **timings})
        metadata.update(status="complete", index=index, compute_seconds=time.perf_counter()-started)
        self.frozen.write_frame(index, arrays, metadata, subdir)
        write(self.frozen.root / self.frozen.run / "worker_status" / f"worker_{os.getpid()}.json",
              {"index": index, "source_name": row["source_name"], "elapsed_h": row["elapsed_h"],
               "compute_seconds": metadata["compute_seconds"], "anchor_replay": self.phase.replay_reports})
        del arrays
        gc.collect()
        return {"index": index, "source_name": row["source_name"], "cached": False,
                "fields_path": str(target), "elapsed_h": row["elapsed_h"],
                "compute_seconds": metadata["compute_seconds"]}


def _initialize_worker(run, root, backend, threads):
    global _WORKER
    _WORKER = MovieWorker(run, root, backend, threads)
    atexit.register(_WORKER.close)


def _compute_chunk(rows):
    result = []
    for row in rows:
        item = _WORKER.compute(row)
        result.append(item)
        print("COMPUTED " + json.dumps(item), flush=True)
    return result


def compute_movie(run, root=ROOT, *, workers=4, threads=2, backend="cuda", include_blacklisted=False,
                  start_index=0, stop_index=None, chunk_size=8):
    """Explicit full-run entrypoint; no scientific fitting or publication.

    FIFO increasing chunks give each worker an increasing subsequence. Workers
    independently replay the immutable reviewed anchors; their inserted movie
    frames can never influence another frame's physical or optical context.
    """
    frozen = FrozenConcentration(run, root)
    manifest_path = Path(root) / run / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["include_blacklisted"] != include_blacklisted:
            raise ValueError("Existing movie manifest has a different blacklist policy")
        for key in ("dataset", "run", "model_id", "curve_sha256", "optical_reference_sha256",
                    "original_prepared_sha256", "temporal_policy", "method_sha256", "parameters"):
            if manifest.get(key) != frozen.provenance[key]:
                raise ValueError(f"Existing movie manifest differs from frozen {key}")
        if manifest["source_inventory_sha256"] != sha(INVENTORY / (run + ".json")):
            raise ValueError("Movie source inventory changed after manifest creation")
    else:
        manifest = frozen.movie_manifest(include_blacklisted=include_blacklisted)
    frames = manifest["frames"][start_index:stop_index]
    status_path = Path(root) / run / "compute_status.json"
    failure_path = Path(root) / run / "compute_failure.json"
    # A compute invocation is an explicit resume. Clear only its own stale
    # failure signal after validating the complete frozen source contract.
    failure_path.unlink(missing_ok=True)
    status = {"run": run, "stage": "running", "requested_frames": len(frames),
              "expected_total": manifest["expected_total"], "workers": workers, "threads_each": threads,
              "completed_requested": 0, "started_unix": time.time(), "errors": []}
    write(status_path, status)
    chunks = [frames[i:i+chunk_size] for i in range(0, len(frames), chunk_size)]
    try:
        if workers == 1:
            _initialize_worker(run, root, backend, threads)
            try:
                for chunk in chunks:
                    status["completed_requested"] += len(_compute_chunk(chunk))
                    write(status_path, status)
            finally:
                _WORKER.close()
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker,
                                     initargs=(run, root, backend, threads)) as executor:
                for completed in executor.map(_compute_chunk, chunks):
                    status["completed_requested"] += len(completed)
                    write(status_path, status)
        missing = [row["index"] for row in manifest["frames"]
                   if not Path(row["fields_path"]).exists() or not Path(row["record_path"]).exists()]
        manifest.update(complete=not missing, available_frame_count=manifest["expected_total"]-len(missing),
                        missing_frame_indices=missing)
        write(manifest_path, manifest)
        status.update(stage="complete" if not missing else "requested_subset_complete", missing_frame_indices=missing,
                      finished_unix=time.time())
        write(status_path, status)
    except BaseException as error:
        status.update(stage="failed", error=repr(error), finished_unix=time.time())
        write(status_path, status)
        write(failure_path, status)
        raise
    return status


def positive_probe(run, root=ROOT, backend="cuda"):
    """Validate disposable history after the first nonempty reviewed anchor."""
    frozen = FrozenConcentration(run, root)
    manifest = read(Path(root) / run / "manifest.json")
    positive = frozen.frames[1]
    row = next(r for r in manifest["frames"] if r["elapsed_h"] > positive["elapsed_h"]
               and r.get("frozen_frame_index") is None)
    worker = MovieWorker(run, root, backend, threads=4)
    try:
        result = worker.compute(row, subdir="positive-probe")
        write(Path(root) / run / "positive-probe" / "verification.json",
              {**result, "anchor_replay": worker.phase.replay_reports})
        return result
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("benchmark", "positive-probe", "compute", "manifest", "frozen-preview"), default="benchmark")
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--include-blacklisted", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    if args.stage == "benchmark":
        benchmark(args.run, args.root, args.backend)
    elif args.stage == "positive-probe":
        positive_probe(args.run, args.root, args.backend)
    elif args.stage == "compute":
        try:
            compute_movie(args.run, args.root, workers=args.workers, threads=args.threads, backend=args.backend,
                          include_blacklisted=args.include_blacklisted, start_index=args.start_index, stop_index=args.stop_index,
                          chunk_size=args.chunk_size)
        except BaseException as error:
            write(args.root / args.run / "compute_failure.json",
                  {"run": args.run, "stage": "failed", "error": repr(error), "finished_unix": time.time()})
            raise
    elif args.stage == "manifest":
        FrozenConcentration(args.run, args.root).movie_manifest(include_blacklisted=args.include_blacklisted)
    else:
        frozen = FrozenConcentration(args.run, args.root)
        for index in (0, 1, len(frozen.frames)-2):
            arrays, metadata = frozen.frozen_fields(index)
            frozen.write_frame(index, arrays, metadata, "frozen-preview")
        write(args.root / args.run / "method.json", frozen.provenance)


if __name__ == "__main__":
    main()
