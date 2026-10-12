"""One frozen native DarSIA concentration response for the FF_AC series.

This module does not train a phase model, change a phase label, or force a mass
map to a dose.  Its two fitted numbers are common response scales.  Every
optical response, flash, density and spatial integration uses DarSIA classes.
The measured per-frame geometry and run recipe are inputs, not fitted effects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

import darsia
from .concentration_native import NativeConcentrationAnalysis, PreparedNativeFrame
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import HeterogeneousColorToMassAnalysis
from darsia.signals.color.color_embedding import ColorEmbeddingBasis


CONTRACT = "ffac-common-native-mass-calibration-v1"
EXCLUDED_RUNS = ("ac20", "ac24", "ac27", "ac51")
HELDOUT_RUNS = ("ac19", "ac25", "ac35", "ac41", "ac47", "ac52", "ac56", "ac60", "ac21", "ac22", "ac32", "ac44")
ALL_RUNS = tuple(f"ac{i}" for i in range(17, 62) if i not in (36, 37))
CALIBRATION_RUNS = tuple(run for run in ALL_RUNS if run not in (*EXCLUDED_RUNS, *HELDOUT_RUNS))
SUPPORTED_FACIES = (1, 2, 5, 7, 8)
UNQUANTIFIED_FACIES = (0, 3, 4, 6, 255)
DEFAULT_FLASH_BOUNDS = dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.)
MOLAR_MASS_CO2_G_MOL = 44.0095
SUMMARY_INTERVALS = 4096


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalize_run(run: str) -> str:
    value = str(run).lower()
    if value not in ALL_RUNS:
        raise ValueError(f"Unknown FF_AC run: {run}")
    return value


def validate_cohort(calibration_runs=CALIBRATION_RUNS, heldout_runs=HELDOUT_RUNS) -> None:
    calibration = tuple(map(normalize_run, calibration_runs))
    heldout = tuple(map(normalize_run, heldout_runs))
    if len(calibration) != 27 or len(set(calibration)) != 27 or set(calibration) != set(CALIBRATION_RUNS):
        raise ValueError("The common calibration must contain exactly the 27 pre-declared calibration runs")
    if len(heldout) != 12 or len(set(heldout)) != 12 or set(heldout) != set(HELDOUT_RUNS):
        raise ValueError("The twelve held-out runs are fixed before fitting")


@dataclass(frozen=True)
class CommonFrameArrays:
    run_id: str
    frame_index: int
    current_rgb: np.ndarray
    baseline_rgb: np.ndarray
    clear_water_rgb: np.ndarray
    clear_water_supported_mask: np.ndarray
    phase_labels: np.ndarray
    facies: np.ndarray
    domain: np.ndarray
    supported_mask: np.ndarray
    mass_eligible_mask: np.ndarray
    depth_m: np.ndarray | float
    porosity: np.ndarray | float
    pixel_area_m2: np.ndarray | float
    dimensions_m: tuple[float, float]
    pressure_bar: float
    temperature_c: float
    elapsed_h: float
    nominal_injected_mass_g: float | None
    recipe: Mapping[str, Any]
    provenance: Mapping[str, Any]
    pressure_gradient_bar_per_m: float = 0.
    temperature_gradient_c_per_m: float = 0.
    excluded_masks: Mapping[str, np.ndarray] | None = None
    l1_neighbourhood_mask: np.ndarray | None = None
    unknown_physical_mask: np.ndarray | None = None

    def __post_init__(self):
        run = normalize_run(self.run_id)
        if run in EXCLUDED_RUNS:
            raise ValueError("Excluded chemical outliers have no role in this calibration or production")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("An exact native frame index is required")
        shape = self.phase_labels.shape
        if self.phase_labels.ndim != 2 or not np.isin(self.phase_labels, (0, 1, 2, 255)).all():
            raise ValueError("Locked P1 final 3 phase labels must be categorical 0/1/2/255")
        for name in ("current_rgb", "baseline_rgb", "clear_water_rgb"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (*shape, 3):
                raise TypeError("Exact float32 HWC scientific RAW is required; a display image cannot enter the model")
        for name in ("domain", "supported_mask", "clear_water_supported_mask", "mass_eligible_mask"):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.bool_:
                raise TypeError(f"{name} must be a registered native boolean mask")
        if self.facies.shape != shape or not np.issubdtype(self.facies.dtype, np.integer) or np.any(self.facies < 0):
            raise ValueError("Facies must be independently registered, nonnegative categorical labels")
        if np.any(self.mass_eligible_mask & ~self.domain):
            raise ValueError("Mass-eligible support extends outside the measured sand domain")
        if self.unknown_physical_mask is not None and (self.unknown_physical_mask.shape != shape or self.unknown_physical_mask.dtype != np.bool_):
            raise ValueError("Unknown physical support must be explicit on the locked native grid")
        physical = self.known_physical_domain
        if not physical.any():
            raise ValueError("No measured depth/porosity domain is available")
        for name in ("depth_m", "porosity", "pixel_area_m2"):
            values = np.broadcast_to(np.asarray(getattr(self, name)), shape)
            selected = self.domain if name == "pixel_area_m2" else physical
            if not np.isfinite(values[selected]).all() or np.any(values[selected] <= 0):
                raise ValueError(f"Measured finite positive {name} is required on its independently supported domain")
            if name == "porosity" and np.any(values > 1):
                raise ValueError("Porosity exceeds one")
        if (len(self.dimensions_m) != 2 or not np.isfinite(self.dimensions_m).all()
                or min(self.dimensions_m) <= 0 or not np.isfinite([self.pressure_bar, self.temperature_c, self.elapsed_h]).all()
                or self.pressure_bar <= 0):
            raise ValueError("Explicit physical dimensions, pressure, temperature and camera time are required")
        observed = self.domain & self.supported_mask
        for value in (self.current_rgb, self.baseline_rgb):
            if not np.isfinite(value[observed]).all() or np.any(value[observed] < 0):
                raise ValueError("Scientific supported RAW has invalid values")
        if self.nominal_injected_mass_g is not None and (not np.isfinite(self.nominal_injected_mass_g) or self.nominal_injected_mass_g < 0):
            raise ValueError("Injected mass must be an independently measured nonnegative input")
        covered = np.zeros(shape, bool)
        for name, mask in (self.excluded_masks or {}).items():
            if mask.shape != shape or mask.dtype != np.bool_ or np.any(mask & ~self.domain):
                raise ValueError(f"Excluded {name} support is not on the native physical grid")
            if np.any(covered & mask):
                raise ValueError("Separately reported exclusion masks overlap")
            covered |= mask
        if np.any(covered & self.mass_eligible_mask):
            raise ValueError("Blocked bottom/fill support entered the main mass")
        if self.l1_neighbourhood_mask is not None and (self.l1_neighbourhood_mask.shape != shape or self.l1_neighbourhood_mask.dtype != np.bool_):
            raise ValueError("L1 sensitivity requires its independently bound registered 20-mm neighbourhood")

    @property
    def time_hours(self):
        return self.elapsed_h

    @property
    def known_physical_domain(self):
        shape = self.domain.shape
        depth = np.broadcast_to(np.asarray(self.depth_m), shape)
        pore = np.broadcast_to(np.asarray(self.porosity), shape)
        known = self.domain & np.isfinite(depth) & np.isfinite(pore) & (depth > 0) & (pore > 0)
        if self.unknown_physical_mask is not None:
            known &= ~self.unknown_physical_mask
        return known


def reference_normalized_rgb(frame: CommonFrameArrays):
    """Measured dye-free reference removes local optical response, not mass.

    This identical formula is applied before path fitting and evaluation.  It
    contains no optimized gain or chemistry-specific transformation.  Unknown
    clear-water support is excluded explicitly, never extended by a fit.
    """
    clear = frame.clear_water_rgb
    support = frame.known_physical_domain & frame.supported_mask & frame.clear_water_supported_mask
    support &= np.isfinite(clear).all(axis=2) & (clear > 0).all(axis=2)
    current = np.zeros(frame.current_rgb.shape, np.float32)
    baseline = np.zeros(frame.baseline_rgb.shape, np.float32)
    np.divide(frame.current_rgb, clear, out=current, where=support[..., None])
    np.divide(frame.baseline_rgb, clear, out=baseline, where=support[..., None])
    if not np.isfinite(current[support]).all() or not np.isfinite(baseline[support]).all():
        raise ValueError("Clear-water normalization produced nonfinite scientific signal")
    return current, baseline, support


class UnclippedNativeConcentrationAnalysis(NativeConcentrationAnalysis):
    """Remove only an adapter's display-range restriction, never clip RAW.

    DarSIA OpticalImage and RGB ColorPath explicitly support arbitrary finite
    float RGB.  The inherited adapter's [0,1] gate is not part of these tools.
    """
    def __init__(self, calibration_folder, *, baseline_rgb, labels, domain, depth_m, porosity, dimensions_m, basis="facies", calibration_status="one common corrected RAW calibration"):
        self.folder = Path(calibration_folder).resolve()
        self.labels = np.asarray(labels)
        self.domain = np.asarray(domain, bool)
        self.shape = self.domain.shape
        self.dimensions = tuple(map(float, dimensions_m))
        if self.labels.shape != self.shape or self.domain.ndim != 2 or not self.domain.any() or not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("Native registered categorical facies and domain are required")
        depth = np.broadcast_to(np.asarray(depth_m, np.float32), self.shape).copy()
        pore = np.broadcast_to(np.asarray(porosity, np.float32), self.shape).copy()
        if not np.isfinite(depth).all() or not np.isfinite(pore).all() or np.any(depth[self.domain] <= 0) or np.any(pore[self.domain] <= 0) or np.any(pore > 1):
            raise ValueError("Explicit measured depth and porosity are required")
        self.geometry = darsia.ExtrudedPorousGeometry(porosity=np.where(self.domain, pore, 0), depth=depth,
            space_dim=2, num_voxels=self.shape, dimensions=list(self.dimensions))
        self._nominal_voxel_volume = self.geometry.voxel_volume.copy()
        self._nominal_pixel_area_m2 = float(np.prod(self.dimensions))/np.prod(self.shape)
        self.pixel_area_m2 = self._nominal_pixel_area_m2
        self.baseline = self._rgb(baseline_rgb)
        self.label_image = darsia.ScalarImage(self.labels, dimensions=list(self.dimensions))
        metadata_path = self.folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.basis = ColorEmbeddingBasis(basis)
        if metadata.get("contract") != CONTRACT or metadata.get("basis") != self.basis.value or not set(map(int, np.unique(self.labels))).issubset(metadata["label_ids"]):
            raise ValueError("Per-frame facies are outside the source-bound common native calibration basis")
        self.ignored_labels = tuple(map(int, metadata["ignore_labels"]))
        self._unquantified_domain = self.domain & np.isin(self.labels, self.ignored_labels)
        color_paths, functions = {}, {}
        files = [metadata_path, self.folder / "flash/flash.json"]
        for label in np.unique(self.labels):
            color_base = self.folder / "color_path_interpretation" / f"color_path_interpretation_{int(label)}"
            signal_base = self.folder / "signal_model" / f"signal_model_{int(label)}"
            color_paths[int(label)] = darsia.ColorPathInterpolation.load(color_base)
            functions[int(label)] = darsia.PWTransformation.load(signal_base)
            files.extend((color_base.with_suffix(".json"), signal_base.with_suffix(".csv")))
        self.source_hashes = {str(path): sha256(path) for path in files}
        # A common package is a superset: a transported frame can legitimately
        # contain fewer facies.  Validate the subset, then construct the exact
        # native pipeline rather than changing saved metadata or image labels.
        self.pipeline = HeterogeneousColorToMassAnalysis(baseline=self.baseline, labels=self.label_image,
            color_mode=darsia.ColorMode.RELATIVE, color_path_interpretation=color_paths, signal_functions=functions,
            flash=darsia.SimpleFlash.load(self.folder / "flash/flash"), co2_mass_analysis=None, geometry=self.geometry,
            restoration=None, ignore_labels=list(self.ignored_labels), basis=self.basis)
        self.calibration_status = calibration_status
        self.legacy_basis_confirmed = False
        self._physics = None
        self._physics_key = None

    def _rgb(self, rgb):
        values = np.asarray(rgb)
        if values.dtype != np.float32 or values.shape != (*self.shape, 3) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Unclipped finite float32 scientific RAW is required")
        return darsia.OpticalImage(values.copy(), color_space="RGB", dimensions=list(self.dimensions))


def fit_pooled_color_paths(samples_by_facies: Mapping[int, Mapping[str, np.ndarray]], *, min_samples=16, min_runs=2):
    """Native ColorPaths from balanced run-level robust RAW-difference anchors.

    Samples are exact native pixel centres, never resized images.  Every run
    has equal influence on the knots.  Hold-out samples are rejected before
    their values are read.  No signal is borrowed for an unsupported facies.
    """
    paths, audit = {}, {}
    for facies in sorted(samples_by_facies):
        if int(facies) not in SUPPORTED_FACIES:
            audit[str(facies)] = dict(status="unquantified", reason="pre-declared unsupported facies")
            continue
        runs = samples_by_facies[facies]
        if any(normalize_run(run) not in CALIBRATION_RUNS for run in runs):
            raise ValueError("Held-out or excluded samples cannot fit common ColorPaths")
        anchors, counts = [], {}
        for run, raw in sorted(runs.items()):
            values = np.asarray(raw)
            if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                raise TypeError("Native relative RAW path samples must be finite float32 Nx3")
            counts[run] = len(values)
            if len(values) < min_samples:
                continue
            norms = np.linalg.norm(values, axis=1)
            if np.quantile(norms, .98) <= 1e-8:
                continue
            rows = []
            for fraction in (.25, .55, .85, .98):
                lower, upper = np.quantile(norms, (max(0., fraction-.075), min(1., fraction+.075)))
                rows.append(np.median(values[(norms >= lower) & (norms <= upper)], axis=0))
            anchors.append(np.asarray(rows, np.float32))
        if len(anchors) < min_runs:
            audit[str(facies)] = dict(status="unquantified", reason="insufficient independent calibration-run optical span", sample_counts=counts)
            continue
        knots = np.vstack((np.zeros(3, np.float32), np.median(np.stack(anchors), axis=0))).astype(np.float32)
        keep = np.r_[True, np.linalg.norm(np.diff(knots, axis=0), axis=1) > 1e-9]
        knots = knots[keep]
        if len(knots) < 2:
            audit[str(facies)] = dict(status="unquantified", reason="zero common color-path span", sample_counts=counts)
            continue
        path = darsia.ColorPath(colors=[row for row in knots], mode="rgb", name=f"FF_AC common RAW facies {facies}")
        paths[int(facies)] = darsia.ColorPathInterpolation(color_path=path, color_mode=darsia.ColorMode.RELATIVE, values=np.asarray(path.relative_distances))
        audit[str(facies)] = dict(status="supported", run_count=len(anchors), sample_counts=counts, knots=knots.tolist(), equal_run_weight=True)
    if not paths:
        raise ValueError("No physically supported facies has a common optical response")
    return paths, audit


def sample_frame_for_common_paths(frame: CommonFrameArrays, *, stride_px=8, interior_radius_px=None):
    """Sample native supported centres after identical measured-reference IO."""
    import cv2
    if normalize_run(frame.run_id) not in CALIBRATION_RUNS:
        raise ValueError("Held-out and excluded frames cannot enter path sampling")
    if frame.provenance.get("physical_context_verified") is not True:
        raise ValueError("Native facies, domain, depth and porosity must be independently source-verified")
    radius = frame.provenance.get("facies_interior_radius_px") if interior_radius_px is None else interior_radius_px
    if type(radius) is not int or radius < 0 or type(stride_px) is not int or stride_px < 1:
        raise ValueError("A fixed verified facies boundary-error margin and native sampling stride are required")
    current, baseline, support = reference_normalized_rgb(frame)
    support &= frame.mass_eligible_mask & np.isin(frame.phase_labels, (1, 2))
    yy, xx = np.mgrid[0:frame.domain.shape[0]:stride_px, 0:frame.domain.shape[1]:stride_px]
    samples, counts = {}, {}
    for label in SUPPORTED_FACIES:
        mask = np.asarray(frame.facies == label, np.uint8)
        depth = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        selected = support[yy, xx] & (frame.facies[yy, xx] == label) & (depth[yy, xx] >= radius)
        values = (current[yy[selected], xx[selected]]-baseline[yy[selected], xx[selected]]).astype(np.float32)
        samples[label] = values
        counts[str(label)] = len(values)
    return samples, dict(run=normalize_run(frame.run_id), frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        sampling_stride_px=stride_px, facies_interior_radius_px=radius, sample_counts=counts,
        representation="corrected linear current/clearwater minus registered baseline/clearwater",
        source_binding=dict(frame.provenance), pooled_path_used_locked_phase=True)


def save_common_path_package(folder: str | Path, paths, *, path_audit=None, source_bindings=None, flash_bounds=None):
    """Write a native-compatible calibration folder only at a new destination."""
    root = Path(folder)
    if root.exists():
        raise FileExistsError("Common calibration artifacts are immutable; use a new folder")
    bounds = dict(DEFAULT_FLASH_BOUNDS if flash_bounds is None else flash_bounds)
    if set(bounds) != set(DEFAULT_FLASH_BOUNDS) or not np.isfinite(list(bounds.values())).all() or bounds["max_value_aq"] <= bounds["min_value_aq"] or bounds["max_value_g"] <= bounds["min_value_g"]:
        raise ValueError("Explicit common finite positive flash windows are required")
    if not set(paths).issubset(SUPPORTED_FACIES) or not paths:
        raise ValueError("Only independently supported pooled facies paths are allowed")
    root.mkdir(parents=True)
    ignored = tuple(sorted(set((*UNQUANTIFIED_FACIES, *SUPPORTED_FACIES)) - set(paths)))
    for label in sorted(set(paths) | set(ignored)):
        interpretation = paths.get(label)
        if interpretation is None:
            # Load-only ignored function: it is explicitly excluded from every
            # mass and shown as an unknown bound, not borrowed as a sand path.
            dummy = darsia.ColorPath(colors=[np.zeros(3), np.full(3, 1e-8)], mode="rgb")
            interpretation = darsia.ColorPathInterpolation(color_path=dummy, color_mode=darsia.ColorMode.RELATIVE, values=[0., 1.])
        interpretation.save(root / "color_path_interpretation" / f"color_path_interpretation_{label}")
        darsia.PWTransformation(supports=[0., 1.], values=[0., 1.]).save(root / "signal_model" / f"signal_model_{label}")
    darsia.SimpleFlash(**bounds).save(root / "flash" / "flash")
    metadata = dict(contract=CONTRACT, basis="facies", label_ids=sorted(set(paths) | set(ignored)), ignore_labels=list(ignored), color_mode=darsia.ColorMode.RELATIVE.value,
        representation="corrected float32 camera-linear RAW divided by measured registered clear-water RAW; native current-minus-registered-baseline",
        shared_signal="identity native ColorPath coordinate for every supported facies", flash_bounds=bounds,
        calibration_runs=list(CALIBRATION_RUNS), heldout_runs=list(HELDOUT_RUNS), source_bindings=source_bindings or {}, path_audit=path_audit or {},
        fitted_per_run_parameters=False, map_mass_rescaling=False, dilution="none")
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def _recipe(recipe: Mapping[str, Any]):
    """Measured recipe concentrations; no undocumented run-specific fitting."""
    values = {name: recipe.get(name) for name in ("alkalinity_M", "btb_M")}
    if any(value is None or not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Every run needs explicit documented recipe or explicitly reported common fallback")
    return {key: float(value) for key, value in values.items()}


def prepare_native_response(frame: CommonFrameArrays, path_folder, *, flash_kind="titration", frozen_calibration_path=None):
    """Apply exact native optics and flash once; no mass fit is applied here."""
    run = normalize_run(frame.run_id)
    if run in HELDOUT_RUNS:
        if frozen_calibration_path is None:
            raise ValueError("Freeze the common calibration before computing a held-out response")
        freeze, _ = verify_frozen_calibration(frozen_calibration_path)
        if Path(freeze["path_folder"]).resolve() != Path(path_folder).resolve():
            raise ValueError("Held-out optics differ from the frozen common calibration")
    folder = Path(path_folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("contract") != CONTRACT:
        raise ValueError("Historical JPEG or another optical package cannot enter this common RAW calibration")
    known_facies = ~np.isin(frame.facies, metadata["ignore_labels"])
    current, baseline, observed = reference_normalized_rgb(frame)
    safe_baseline = np.where(observed[..., None], baseline, 0.).astype(np.float32)
    safe_current = np.where(observed[..., None], current, safe_baseline).astype(np.float32)
    analysis = UnclippedNativeConcentrationAnalysis(folder, baseline_rgb=safe_baseline, labels=frame.facies,
        domain=frame.known_physical_domain, depth_m=np.where(frame.known_physical_domain, frame.depth_m, 0.), porosity=np.where(frame.known_physical_domain, frame.porosity, 0.), dimensions_m=frame.dimensions_m,
        basis="facies", calibration_status="one common corrected RAW calibration")
    analysis.set_pixel_area(frame.pixel_area_m2)
    phase = np.where(frame.known_physical_domain & (frame.phase_labels != 255), frame.phase_labels, 0).astype(np.uint8)
    # Native thermodynamics supplies the measured state and hydrostatic field.
    placeholder = PreparedNativeFrame(run, frame.elapsed_h, np.zeros(phase.shape, np.float32), np.zeros(phase.shape, np.float32), phase,
        frame.pressure_bar, frame.temperature_c, frame.nominal_injected_mass_g, frame.pressure_gradient_bar_per_m,
        frame.temperature_gradient_c_per_m, frame.pixel_area_m2)
    physics = analysis._mass_physics(placeholder)
    color = analysis.pipeline.call_color_interpretation(analysis._rgb(safe_current))
    signal = analysis.pipeline.call_pH_analysis(color)
    bounds = metadata["flash_bounds"]
    if flash_kind == "titration":
        solubility_M = float(np.median(physics.solubility_co2[frame.known_physical_domain])) / MOLAR_MASS_CO2_G_MOL
        flash = darsia.TitrationFlash(**bounds, **_recipe(frame.recipe), co2_sat_M=solubility_M)
    elif flash_kind == "linear":
        flash = darsia.SimpleFlash(**bounds)
    else:
        raise ValueError("Unknown common flash sensitivity")
    aqueous, gas = flash(signal)
    prepared = replace(placeholder, aqueous_relative=np.asarray(aqueous.img, np.float32), gas_saturation=np.asarray(gas.img, np.float32))
    quantifiable = observed & known_facies
    return dict(analysis=analysis, prepared=prepared, native_signal=np.asarray(signal.img, np.float32),
        physics=physics, quantifiable=quantifiable, flash=flash, metadata=metadata, frame=frame)


def _scaled_response(response, parameters, *, phase_constraint=True, gas_equilibrium=True, port_as_aqueous=False, allow_provisional_l1_sensitivity=False):
    frame, prepared = response["frame"], response["prepared"]
    scale_aq, scale_g = float(parameters["aqueous_scale"]), float(parameters["gas_scale"])
    if not np.isfinite([scale_aq, scale_g]).all() or min(scale_aq, scale_g) <= 0:
        raise ValueError("One positive finite shared aqueous/gas pair is required")
    known = response["quantifiable"]
    aqueous = np.clip(prepared.aqueous_relative * scale_aq, 0, 1)
    gas = np.clip(prepared.gas_saturation * scale_g, 0, 1)
    gas_support = known & (frame.phase_labels == 2) if phase_constraint else known & (gas > 0)
    aqueous_support = known & np.isin(frame.phase_labels, (1, 2)) if phase_constraint else known
    if port_as_aqueous:
        verified = frame.provenance.get("l1_position_verified") is True
        provisional = (allow_provisional_l1_sensitivity and frame.provenance.get("l1_position_source_bound") is True
            and frame.provenance.get("l1_position_status") == "protocol_proxy_unverified"
            and bool(frame.provenance.get("l1_position_source_sha256")))
        if frame.l1_neighbourhood_mask is None or not (verified or provisional):
            raise ValueError("L1 sensitivity is unknown without a source-verified injector position")
        if frame.elapsed_h >= 4:
            gas_support &= ~frame.l1_neighbourhood_mask
            # Reclassified gas is aqueous equilibrium; CO2 support is unchanged.
            aqueous[known & frame.l1_neighbourhood_mask & (frame.phase_labels == 2)] = 1.
    aqueous = np.where(aqueous_support, aqueous, 0).astype(np.float32)
    gas = np.where(gas_support, gas, 0).astype(np.float32)
    if gas_equilibrium:
        aqueous[gas_support] = 1.
    return aqueous, gas


def evaluate_native_response(response, parameters, *, fields=True, phase_constraint=True, gas_equilibrium=True, port_as_aqueous=False, allow_provisional_l1_sensitivity=False):
    """Native mass integration on main support, blocked support and unknowns."""
    frame, analysis, physics = response["frame"], response["analysis"], response["physics"]
    aqueous, gas = _scaled_response(response, parameters, phase_constraint=phase_constraint, gas_equilibrium=gas_equilibrium, port_as_aqueous=port_as_aqueous, allow_provisional_l1_sensitivity=allow_provisional_l1_sensitivity)
    result = physics.mass_analysis(analysis._scalar(aqueous, frame.elapsed_h), analysis._scalar(gas, frame.elapsed_h))
    eligible = frame.known_physical_domain & frame.mass_eligible_mask
    def integral(value, mask):
        return float(analysis.geometry.integrate(np.where(mask, value, 0.))) * 1000.
    def masses(mask):
        values = dict(aqueous_mass_g=integral(result.mass_aq.img, mask), gas_mass_g=integral(result.mass_g.img, mask), total_mass_g=integral(result.mass.img, mask))
        if not np.isclose(values["total_mass_g"], values["aqueous_mass_g"]+values["gas_mass_g"], rtol=1e-7, atol=1e-8):
            raise ValueError("Native component integration failed mass additivity")
        return values
    phase_support = frame.domain & np.isin(frame.phase_labels, (1, 2)) if phase_constraint else frame.domain
    unknown_physical = phase_support & ~frame.known_physical_domain
    unknown = phase_support & ~response["quantifiable"]
    maximum = np.maximum(physics.solubility_co2, physics.density_gaseous_co2)
    if phase_constraint:
        maximum = np.where(frame.phase_labels == 2, maximum, physics.solubility_co2)
    def capacity(mask):
        selected = phase_support & mask
        unknown_count = int(np.count_nonzero(unknown_physical & mask))
        known_upper = integral(maximum, selected & frame.known_physical_domain)
        return dict(co2_phase_pixels=int(np.count_nonzero(selected)),
            unknown_physical_phase_pixels=unknown_count,
            known_physical_capacity_upper_g=known_upper,
            capacity_upper_g=None if unknown_count else known_upper,
            capacity_rule="aqueous: pore volume times native solubility; gas: pore volume times max(native solubility, native gas density)" if phase_constraint else "phase-free: pore volume times max(native solubility, native gas density)",
            unknown_geometry_rule="unbounded; no imputed depth or porosity")
    main = masses(eligible)
    pore_volume = integral(np.ones(frame.domain.shape, np.float32), eligible)
    density_average = None if pore_volume <= 0 else integral(physics.density_gaseous_co2, eligible)/pore_volume
    excluded = {}
    covered = np.zeros(frame.domain.shape, bool)
    for name, mask in sorted((frame.excluded_masks or {}).items()):
        covered |= mask
        excluded[name] = dict(**masses(mask), unquantified_phase_pixels=int(np.count_nonzero(unknown & mask)), unknown_mass_upper_g=None if np.any(unknown_physical & mask) else integral(maximum, unknown & mask), known_geometry_unknown_mass_upper_g=integral(maximum, unknown & mask), unknown_physical_phase_pixels=int(np.count_nonzero(unknown_physical & mask)), main_total_includes_this=False,
            phase_mass_capacity=capacity(mask))
    other = frame.domain & ~frame.mass_eligible_mask & ~covered
    if other.any():
        excluded["other_not_mass_eligible"] = dict(**masses(other), unquantified_phase_pixels=int(np.count_nonzero(unknown & other)), unknown_mass_upper_g=None if np.any(unknown_physical & other) else integral(maximum, unknown & other), known_geometry_unknown_mass_upper_g=integral(maximum, unknown & other), unknown_physical_phase_pixels=int(np.count_nonzero(unknown_physical & other)), main_total_includes_this=False,
            phase_mass_capacity=capacity(other))
    labels, counts = np.unique(frame.facies[unknown & eligible], return_counts=True)
    output = dict(contract=CONTRACT, run=normalize_run(frame.run_id), frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        nominal_injected_mass_g=frame.nominal_injected_mass_g, **main, excluded_mass=excluded,
        mass_not_eligible_g=sum(row["total_mass_g"] for row in excluded.values()),
        unquantified_phase_pixels=int(np.count_nonzero(unknown & eligible)),
        unquantified_phase_pixels_by_facies={str(int(label)): int(count) for label, count in zip(labels, counts)},
        unknown_mass_upper_g=integral(maximum, unknown & eligible), total_mass_lower_g=main["total_mass_g"],
        total_mass_upper_g=None if np.any(unknown_physical & frame.mass_eligible_mask) else main["total_mass_g"]+integral(maximum, unknown & eligible),
        unknown_physical_phase_pixels=int(np.count_nonzero(unknown_physical)),
        unknown_physical_main_phase_pixels=int(np.count_nonzero(unknown_physical & frame.mass_eligible_mask)),
        unknown_physical_phase_area_cm2=float(np.sum(np.broadcast_to(np.asarray(frame.pixel_area_m2), frame.domain.shape)[unknown_physical]))*1e4,
        unknown_physical_mass_bound="unbounded: measured depth/porosity missing" if unknown_physical.any() else "none",
        solubility_max_g_l=float(np.max(physics.solubility_co2[frame.domain])),
        gas_density_pore_volume_mean_g_l=density_average,
        gas_density_min_g_l=float(np.min(physics.density_gaseous_co2[frame.domain])),
        gas_density_max_g_l=float(np.max(physics.density_gaseous_co2[frame.domain])),
        overflow_gas_density_g_l=float(np.median(physics.density_gaseous_co2[0])),
        overflow_density_method="DarSIA native NIST density at the atmospheric/top-row measured pressure and temperature",
        gas_pixels_with_s_g_zero=int(np.count_nonzero(eligible & (frame.phase_labels == 2) & response["quantifiable"] & (gas == 0))),
        native_gas_pixels_with_s_g_zero=int(np.count_nonzero(eligible & (frame.phase_labels == 2) & response["quantifiable"] & (response["prepared"].gas_saturation == 0))),
        phase_constraint=bool(phase_constraint), gas_equilibrium=bool(gas_equilibrium),
        provisional_l1_location=bool(port_as_aqueous and frame.provenance.get("l1_position_verified") is not True),
        phase_labels_modified=False, map_mass_rescaling=False, dilution="none", recipe=dict(frame.recipe),
        caveats=list(frame.recipe.get("caveats", ())), source_binding=dict(frame.provenance), parameters=dict(parameters))
    if fields:
        concentration = np.asarray(aqueous*physics.solubility_co2, np.float32)
        saturation = gas.copy()
        # Unknown support is explicit in both the maps and the reported bound.
        concentration[unknown] = np.nan
        saturation[unknown & (frame.phase_labels == 2) if phase_constraint else unknown] = np.nan
        output["fields"] = dict(c_aq_g_l=concentration, concentration_g_l=concentration, c_aq_relative=aqueous, s_g=saturation,
            gas_saturation=saturation, phase_labels=frame.phase_labels.copy(), unquantified_phase_mask=unknown,
            main_mass_eligible_mask=eligible, mass_density_g_l=np.asarray(result.mass.img, np.float32),
            aqueous_mass_density_g_l=np.asarray(result.mass_aq.img, np.float32), gas_mass_density_g_l=np.asarray(result.mass_g.img, np.float32))
    return output


def summarize_native_response(response, *, no_outflow_evidence: Mapping[str, Any]):
    """Small native-geometry-weighted response histogram for a common fit."""
    frame, analysis, physics = response["frame"], response["analysis"], response["physics"]
    if normalize_run(frame.run_id) not in CALIBRATION_RUNS:
        raise ValueError("A held-out run cannot enter even the mass-fitting summaries")
    if no_outflow_evidence.get("no_co2_left_rig") is not True or not no_outflow_evidence.get("source"):
        raise ValueError("A fitting frame needs explicit sourced evidence of no CO2 outflow")
    if frame.nominal_injected_mass_g is None or frame.nominal_injected_mass_g <= 0:
        raise ValueError("A fitting frame needs positive measured cumulative injected mass")
    support = frame.known_physical_domain & frame.mass_eligible_mask & response["quantifiable"] & np.isin(frame.phase_labels, (1, 2))
    prepared = response["prepared"]
    aq = np.rint(np.clip(prepared.aqueous_relative[support], 0, 1)*SUMMARY_INTERVALS).astype(np.int32)
    gas = np.rint(np.clip(prepared.gas_saturation[support], 0, 1)*SUMMARY_INTERVALS).astype(np.int32)
    codes, inverse = np.unique(aq*(SUMMARY_INTERVALS+1)+gas, return_inverse=True)
    is_gas = frame.phase_labels[support] == 2
    weight_aq = analysis.geometry.make_extensive(analysis._scalar(physics.solubility_co2)).img[support]*1000.
    weight_gas = analysis.geometry.make_extensive(analysis._scalar(physics.density_gaseous_co2)).img[support]*1000.
    aggregate = lambda values: np.bincount(inverse, weights=values, minlength=len(codes)).tolist()
    diagnostic = evaluate_native_response(response, dict(aqueous_scale=1., gas_scale=1.), fields=False)
    unknown_main_upper = (None if diagnostic["unknown_physical_main_phase_pixels"]
        else diagnostic["unknown_mass_upper_g"])
    excluded_capacities = {name: row["phase_mass_capacity"]
        for name, row in diagnostic["excluded_mass"].items()}
    excluded_upper = (None if any(row["capacity_upper_g"] is None for row in excluded_capacities.values())
        else sum(row["capacity_upper_g"] for row in excluded_capacities.values()))
    missing_upper = (None if unknown_main_upper is None or excluded_upper is None
        else unknown_main_upper + excluded_upper)
    return dict(contract=CONTRACT, run=normalize_run(frame.run_id), frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        nominal_injected_mass_g=frame.nominal_injected_mass_g, no_outflow_evidence=dict(no_outflow_evidence), intervals=SUMMARY_INTERVALS,
        aqueous_response=(codes//(SUMMARY_INTERVALS+1)/SUMMARY_INTERVALS).tolist(), gas_response=(codes%(SUMMARY_INTERVALS+1)/SUMMARY_INTERVALS).tolist(),
        aqueous_capacity_g=aggregate(weight_aq*~is_gas), gas_aqueous_capacity_g=aggregate(weight_aq*is_gas), gas_capacity_g=aggregate(weight_gas*is_gas),
        unknown_mass_upper_g=diagnostic["unknown_mass_upper_g"], unquantified_phase_pixels=diagnostic["unquantified_phase_pixels"],
        unknown_main_capacity_upper_g=unknown_main_upper,
        excluded_strip_capacity_upper_g=excluded_upper,
        excluded_strip_capacities=excluded_capacities,
        mass_balance_missing_capacity_upper_g=missing_upper,
        mass_balance_constraint=dict(
            prediction="known quantifiable main mass only; excluded-strip mass never added to the main prediction",
            admitted_dose_interval="[known main mass, known main mass + unknown-main capacity + excluded-strip capacity]",
            interval_upper="unbounded" if missing_upper is None else "finite physical capacity",
            unbounded_reasons=(
                (["unknown_physical_main_CO2"] if unknown_main_upper is None else [])
                + ["unknown_physical_excluded_CO2:" + name for name, row in excluded_capacities.items() if row["capacity_upper_g"] is None]),
            excluded_and_main_masks_disjoint=True,
            computed_excluded_mass_used_in_fit=False, missing_mass_fitted_per_frame=False),
        unknown_physical_phase_pixels=diagnostic["unknown_physical_phase_pixels"], unknown_physical_phase_area_cm2=diagnostic["unknown_physical_phase_area_cm2"], source_binding=dict(frame.provenance))


def evaluate_common_summary(summary, parameters):
    aq = np.clip(np.asarray(summary["aqueous_response"])*parameters["aqueous_scale"], 0, 1)
    gas = np.clip(np.asarray(summary["gas_response"])*parameters["gas_scale"], 0, 1)
    aw, gaw, gw = (np.asarray(summary[key]) for key in ("aqueous_capacity_g", "gas_aqueous_capacity_g", "gas_capacity_g"))
    # The gas-water branch is equilibrium even when the scaled aqueous response
    # would be below saturation.  This matches exact native mass_analysis.
    mass_aq = float(np.sum(aw*aq+gaw*(1-gas)))
    mass_g = float(np.sum(gw*gas))
    error = float((parameters["aqueous_scale"]*np.sum(aw)+parameters["gas_scale"]*np.sum(gaw+gw))/(2*summary["intervals"]))
    return dict(aqueous_mass_g=mass_aq, gas_mass_g=mass_g, total_mass_g=mass_aq+mass_g, summary_mass_error_bound_g=error)


def fit_shared_mass_scales(summaries: Sequence[Mapping[str, Any]], *, unknown_policy="bounded_interval"):
    """One positive aqueous/gas pair, using no held-out values or overflow fit."""
    if unknown_policy not in ("bounded_interval", "known_only"):
        raise ValueError("Unknown unquantified-facies fitting policy")
    by_run = {run: [] for run in CALIBRATION_RUNS}
    for row in summaries:
        run = normalize_run(row["run"])
        if run not in by_run:
            raise ValueError("Held-out or excluded rows entered the shared mass fit")
        if row.get("contract") != CONTRACT or row.get("no_outflow_evidence", {}).get("no_co2_left_rig") is not True or not row["no_outflow_evidence"].get("source"):
            raise ValueError("Every fitting row must carry the predeclared no-outflow evidence")
        target = row.get("nominal_injected_mass_g")
        if target is None or not np.isfinite(target) or target <= 0:
            raise ValueError("Every fitting row must have a positive measured target")
        names = ("aqueous_response", "gas_response", "aqueous_capacity_g", "gas_aqueous_capacity_g", "gas_capacity_g")
        values = [np.asarray(row[name]) for name in names]
        if row.get("intervals") != SUMMARY_INTERVALS or any(value.ndim != 1 or value.shape != values[0].shape or not np.isfinite(value).all() or np.any(value < 0) for value in values) or any(np.any(value > 1) for value in values[:2]):
            raise ValueError("Invalid native response histogram")
        if not np.isfinite(row.get("unknown_mass_upper_g", 0)) or row.get("unknown_mass_upper_g", 0) < 0:
            raise ValueError("Unknown support has no physical upper bound")
        missing_upper = row.get("mass_balance_missing_capacity_upper_g", row.get("unknown_mass_upper_g", 0.))
        if missing_upper is None:
            if (row.get("mass_balance_constraint", {}).get("interval_upper") != "unbounded"
                    or not row["mass_balance_constraint"].get("unbounded_reasons")):
                raise ValueError("An unbounded dose interval requires explicit missing-geometry evidence")
        elif not np.isfinite(missing_upper) or missing_upper < 0:
            raise ValueError("Invalid physical missing-mass capacity")
        by_run[run].append(row)
    eligible_runs = [run for run, rows in by_run.items() if rows]
    omitted_runs = [run for run, rows in by_run.items() if not rows]
    if not eligible_runs:
        raise ValueError("The shared fit has no eligible source-bound calibration observations")
    if len({(row["run"], row["frame_index"]) for row in summaries}) != len(summaries):
        raise ValueError("A fitting frame was counted twice")
    # The native output clips concentration and saturation to their physical
    # ranges.  A bound on their common positive gains would be an additional,
    # unjustified constraint (in particular, the old 0.05..4 adapter bounds).
    cached = []
    for row in summaries:
        rows = by_run[normalize_run(row["run"])]
        normalizer = max(item["nominal_injected_mass_g"] for item in rows)*np.sqrt(len(rows))
        upper = row.get("mass_balance_missing_capacity_upper_g", row.get("unknown_mass_upper_g", 0.)) if unknown_policy == "bounded_interval" else 0.
        cached.append((*[np.asarray(row[key], np.float64) for key in names],
            row["nominal_injected_mass_g"], upper, normalizer))
    def positive_pair(theta):
        # Only guard floating-point overflow/underflow; this does not bound
        # scientific gains.  Beyond saturation the native response is flat.
        values = np.maximum(np.exp(np.minimum(theta, np.log(np.finfo(np.float64).max))), np.finfo(np.float64).tiny)
        return dict(aqueous_scale=float(values[0]), gas_scale=float(values[1]))
    def residual_and_jacobian(theta):
        pair = positive_pair(theta)
        residuals, jacobian = [], []
        for a, g, aw, gaw, gw, target, upper, normalizer in cached:
            qa, qg = a*pair["aqueous_scale"], g*pair["gas_scale"]
            aq, gas = np.minimum(qa, 1.), np.minimum(qg, 1.)
            predicted = float(np.sum(aw*aq+gaw*(1-gas)+gw*gas))
            delta = max(predicted-target, 0.)
            if upper is not None:
                delta += min(predicted+upper-target, 0.)
            residuals.append(delta/normalizer)
            active = upper == 0 or predicted > target or (upper is not None and predicted+upper < target)
            jacobian.append([np.sum(aw*np.where(qa < 1., qa, 0.))/normalizer,
                np.sum((gw-gaw)*np.where(qg < 1., qg, 0.))/normalizer] if active else [0., 0.])
        return np.asarray(residuals), np.asarray(jacobian)
    fits = [least_squares(lambda theta: residual_and_jacobian(theta)[0], np.log(initial),
        jac=lambda theta: residual_and_jacobian(theta)[1], x_scale=1., max_nfev=500,
        ftol=1e-10, xtol=1e-10, gtol=1e-10) for initial in ((1., 1.), (20., 1.))]
    successful = [index for index, item in enumerate(fits) if item.success]
    candidates = successful or list(range(len(fits)))
    selected = min(candidates, key=lambda index: (float(fits[index].cost), index))
    objective_equivalent = (len(successful) == len(fits)
        and abs(float(fits[0].cost)-float(fits[1].cost)) <= 1e-12+1e-8*max(float(item.cost) for item in fits))
    if objective_equivalent:
        # Declare the tie rule before any held-out data: keep the first
        # successful common start, rather than tune an unidentified gain.
        selected = successful[0]
    fit = fits[selected]
    pair = positive_pair(fit.x)
    original_pair = dict(pair)
    multistart = [dict(initial_scales=list(initial), parameters=positive_pair(item.x),
        optimizer_cost=float(item.cost), fit_success=bool(item.success), evaluations=int(item.nfev))
        for initial, item in zip(((1., 1.), (20., 1.)), fits)]
    before = np.asarray([evaluate_common_summary(row, pair)["total_mass_g"] for row in summaries])
    _, jacobian = residual_and_jacobian(np.log(list(pair.values())))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.count_nonzero(singular > max(float(singular.max(initial=0))*1e-6, 1e-10)))
    gas_values = [g[(g > 0) & ((gaw+gw) > 0)] for a, g, aw, gaw, gw, *_ in cached]
    positive_gas = np.concatenate(gas_values)
    gas_plateau = dict(applied=False, point_parameter_identified=rank == 2,
        calibration_predictions_identified=False)
    if positive_gas.size and rank == 1 and np.all(positive_gas*pair["gas_scale"] >= 1.) and np.all(jacobian[:, 1] == 0.) and np.any(jacobian[:, 0] != 0.):
        minimum = float(positive_gas.min())
        lower = minimum-.5/SUMMARY_INTERVALS
        if lower > 0.:
            threshold = 1./lower
            representative = np.float32(threshold)
            if float(representative) < threshold:
                representative = np.nextafter(representative, np.float32(np.inf))
            candidate = dict(pair, gas_scale=float(representative))
            after = np.asarray([evaluate_common_summary(row, candidate)["total_mass_g"] for row in summaries])
            if objective_equivalent and np.array_equal(before, after):
                pair = candidate
                gas_plateau = dict(applied=True, point_parameter_identified=False,
                    calibration_predictions_identified=True, calibration_predictions_bit_identical=True,
                    aqueous_log_derivative_identified=True, multistart_same_objective=True,
                    selection="smallest float32 common gain above the lower-endpoint saturation threshold of weighted calibration gas responses",
                    minimum_weighted_positive_histogram_response=minimum,
                    histogram_half_bin=.5/SUMMARY_INTERVALS,
                    conservative_response_lower_bound=lower,
                    histogram_gas_gain_lower_bound=1./minimum,
                    conservative_gas_gain_lower_bound=threshold,
                    representative_gas_gain=float(representative),
                    zero_response_gas_capacity_g=float(sum(np.sum((gaw+gw)[g == 0.]) for a, g, aw, gaw, gw, *_ in cached)),
                    native_pixel_verification_required=True,
                    scientific_interpretation="calibration gas predictions are on a saturation plateau; the gas coefficient has only a lower bound, not a point estimate")
    final_residual, final_jacobian = residual_and_jacobian(np.log(list(pair.values())))
    singular = np.linalg.svd(final_jacobian, compute_uv=False)
    rank = int(np.count_nonzero(singular > max(float(singular.max(initial=0))*1e-6, 1e-10)))
    constraints = []
    for row in summaries:
        predicted = evaluate_common_summary(row, pair)["total_mass_g"]
        upper = row.get("mass_balance_missing_capacity_upper_g", row.get("unknown_mass_upper_g", 0.)) if unknown_policy == "bounded_interval" else 0.
        target = row["nominal_injected_mass_g"]
        constraints.append(dict(run=row["run"], frame_index=row["frame_index"],
            known_main_mass_g=predicted, injected_mass_g=target,
            missing_capacity_upper_g=upper,
            interval_upper_g=None if upper is None else predicted+upper,
            constraint_type="one_sided_lower_mass" if upper is None else "bounded_dose_interval",
            active_direction="known_main_exceeds_dose" if predicted > target
                else "main_plus_capacity_below_dose" if upper is not None and predicted+upper < target
                else "dose_inside_admissible_interval",
            constraint_evidence=row.get("mass_balance_constraint", {})))
    return dict(contract=CONTRACT, parameters=pair, calibration_runs=list(CALIBRATION_RUNS), heldout_runs=list(HELDOUT_RUNS),
        eligible_calibration_runs=eligible_runs, omitted_calibration_runs=omitted_runs,
        omitted_calibration_run_reason="no eligible source-bound observation was supplied; no run-specific parameter or reference substituted",
        frames_used=[dict(run=row["run"], frame_index=row["frame_index"], elapsed_h=row["elapsed_h"], nominal_injected_mass_g=row["nominal_injected_mass_g"], no_outflow_evidence=row["no_outflow_evidence"], source_binding=row.get("source_binding", {})) for row in summaries],
        fit_success=bool(fit.success), fit_message=str(fit.message), jacobian_rank=rank,
        jacobian_singular_values=singular.tolist(), jacobian_coordinates="log common scales",
        optimizer_cost=float(.5*np.dot(final_residual, final_residual)), residual_norm=float(np.linalg.norm(final_residual)),
        shared_scale_bounds=[0., None], shared_scale_bounds_source="strictly positive unbounded common gains; native concentration and saturation retain physical bounds 0..1",
        parameter_domain="positive_log_unbounded", optimizer_multistart=multistart,
        multistart_objective_equivalent=objective_equivalent,
        multistart_selection_policy="first successful (1,1) start when objectives agree within 1e-12 + 1e-8 times cost; otherwise lowest-cost successful start",
        selected_optimizer_start=selected, optimizer_pair_before_plateau_selection=original_pair,
        gas_plateau=gas_plateau,
        review_only=True, calibration_predictions_identified=rank == 2,
        nonunique_parameter_warning=None if rank == 2 else "The common pair is not point-identified; the selected empirical review pair follows the pre-heldout multistart tie rule, and other equivalent pairs may change interval-inactive predictions",
        parameter_identifiability_warning=None if rank == 2 else "Gas gain is lower-bounded on the calibration saturation plateau, not point-identified" if gas_plateau["applied"] else "Two shared scales are not separately identifiable from these observations",
        identifiable_shared_pair=rank == 2, per_frame_constraints=constraints,
        parameters_at_bounds=[],
        unknown_policy=unknown_policy, overflow_in_objective=False, map_mass_rescaling=False, dilution="none", equal_run_weight=True,
        predictions=[dict(run=row["run"], frame_index=row["frame_index"], **evaluate_common_summary(row, pair)) for row in summaries],
        exact_final_native_evaluation_required=True)


def freeze_calibration(folder, *, path_folder, fit_result, spec_sha256, input_plan_sha256, extra_files=None):
    """Publish one immutable freeze before the first held-out computation."""
    validate_cohort(fit_result["calibration_runs"], fit_result["heldout_runs"])
    if fit_result.get("fit_success") is not True:
        raise ValueError("Cannot freeze an unsuccessful common fit")
    pair = fit_result.get("parameters", {})
    if set(pair) != {"aqueous_scale", "gas_scale"} or any(not np.isfinite(value) or value <= 0. for value in pair.values()):
        raise ValueError("A frozen common pair must be finite and strictly positive")
    if fit_result.get("jacobian_rank") not in (1, 2):
        raise ValueError("Cannot freeze an unidentified shared aqueous/gas pair; report the fit and unknown-facies interval")
    if fit_result.get("jacobian_rank") == 1 and (fit_result.get("review_only") is not True or not fit_result.get("nonunique_parameter_warning")):
        raise ValueError("A nonunique common pair needs explicit review-only and multistart uncertainty reporting")
    plateau = fit_result.get("gas_plateau", {})
    root = Path(folder)
    if root.exists():
        raise FileExistsError("Freeze records are immutable; use a new calibration folder")
    root.mkdir(parents=True)
    (root / "shared_mass_scales.json").write_text(json.dumps(fit_result, indent=2, allow_nan=False), encoding="utf-8")
    package = Path(path_folder).resolve()
    inventory = {str(path.resolve()): sha256(path) for path in sorted(package.rglob("*")) if path.is_file()}
    inventory[str((root / "shared_mass_scales.json").resolve())] = sha256(root / "shared_mass_scales.json")
    for key, value in (extra_files or {}).items():
        path = Path(key).resolve()
        actual = sha256(path)
        if value is not None and value != actual:
            raise ValueError("A source binding changed before the calibration freeze")
        inventory[str(path)] = actual
    implementations = {obj.__name__: sha256(inspect.getfile(obj)) for obj in (darsia.ColorPath, darsia.ColorPathInterpolation, darsia.ConcentrationAnalysis,
        darsia.PWTransformation, darsia.TitrationFlash, darsia.SimpleFlash, darsia.CO2MassAnalysis, darsia.ExtrudedPorousGeometry, NativeConcentrationAnalysis)}
    payload = dict(contract=CONTRACT, frozen=True, frozen_utc=datetime.now(timezone.utc).isoformat(),
        calibration_runs=list(CALIBRATION_RUNS), heldout_runs=list(HELDOUT_RUNS), excluded_runs=list(EXCLUDED_RUNS), calibration_files=inventory,
        path_folder=str(package), parameters=dict(fit_result["parameters"]), shared_scales=dict(fit_result["parameters"]), spec_sha256=str(spec_sha256), input_plan_sha256=str(input_plan_sha256),
        identifiable_shared_pair=bool(fit_result.get("identifiable_shared_pair")),
        parameter_identifiability_warning=fit_result.get("parameter_identifiability_warning"), gas_plateau=plateau,
        review_only=True, nonunique_parameter_warning=fit_result.get("nonunique_parameter_warning"),
        calibration_predictions_identified=bool(fit_result.get("calibration_predictions_identified")),
        optimizer_multistart=fit_result.get("optimizer_multistart", []),
        multistart_selection_policy=fit_result.get("multistart_selection_policy"),
        common_engine_sha256=sha256(__file__), native_implementations=implementations,
        native_implementation_files={str(Path(inspect.getfile(obj)).resolve()): sha256(inspect.getfile(obj)) for obj in (darsia.ColorPath, darsia.ColorPathInterpolation, darsia.ConcentrationAnalysis,
            darsia.PWTransformation, darsia.TitrationFlash, darsia.SimpleFlash, darsia.CO2MassAnalysis, darsia.ExtrudedPorousGeometry, NativeConcentrationAnalysis)}, map_mass_rescaling=False,
        gas_c_aq="native relative concentration 1 = local CO2 solubility; output g/L", dilution="none", overflow_in_objective=False)
    destination = root / "frozen_calibration.json"
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (root / "freeze_receipt.json").write_text(json.dumps(dict(path=str(destination.resolve()), sha256=sha256(destination), frozen_utc=payload["frozen_utc"]), indent=2), encoding="utf-8")
    return destination


_VERIFIED_FREEZES = {}


def verify_frozen_calibration(freeze_path, *, expected_sha256=None):
    path = Path(freeze_path).resolve()
    actual = sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Frozen common calibration SHA-256 mismatch")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("contract") != CONTRACT or freeze.get("frozen") is not True:
        raise ValueError("Held-out and production evaluation require a published common freeze")
    validate_cohort(freeze["calibration_runs"], freeze["heldout_runs"])
    if freeze.get("map_mass_rescaling") is not False or freeze.get("overflow_in_objective") is not False:
        raise ValueError("The frozen common model violates the no-rescaling/overflow-compare-only contract")
    for name, expected in {**freeze["calibration_files"], **freeze.get("native_implementation_files", {})}.items():
        source = Path(name)
        stat = source.stat()
        key = (str(source.resolve()), stat.st_size, stat.st_mtime_ns, expected)
        if key not in _VERIFIED_FREEZES:
            if sha256(source) != expected:
                raise ValueError(f"Frozen calibration file changed: {source}")
            _VERIFIED_FREEZES[key] = True
    if sha256(__file__) != freeze["common_engine_sha256"]:
        raise ValueError("Common mass implementation changed after freeze")
    return freeze, actual


def evaluate_common_frame(frame: CommonFrameArrays, freeze_path, *, expected_freeze_sha256=None, fields=True, flash_kind="titration", phase_constraint=True, gas_equilibrium=True, port_as_aqueous=False, allow_provisional_l1_sensitivity=False):
    freeze, digest = verify_frozen_calibration(freeze_path, expected_sha256=expected_freeze_sha256)
    response = prepare_native_response(frame, freeze["path_folder"], flash_kind=flash_kind, frozen_calibration_path=freeze_path)
    output = evaluate_native_response(response, freeze["parameters"], fields=fields, phase_constraint=phase_constraint, gas_equilibrium=gas_equilibrium, port_as_aqueous=port_as_aqueous, allow_provisional_l1_sensitivity=allow_provisional_l1_sensitivity)
    output.update(frozen_calibration_sha256=digest, frozen_utc=freeze["frozen_utc"], flash_kind=flash_kind,
        evaluation_role="held_out" if normalize_run(frame.run_id) in HELDOUT_RUNS else "calibration_run_production")
    return output


def evaluate_common_prepared_response(response, freeze_path, *, expected_freeze_sha256=None, fields=True, flash_kind="titration", phase_constraint=True, gas_equilibrium=True, port_as_aqueous=False, allow_provisional_l1_sensitivity=False):
    """Reuse one native optical pass for every sensitivity of this exact frame."""
    freeze, digest = verify_frozen_calibration(freeze_path, expected_sha256=expected_freeze_sha256)
    if Path(response["analysis"].folder).resolve() != Path(freeze["path_folder"]).resolve():
        raise ValueError("Prepared optical response uses another calibration package")
    source = response
    if flash_kind == "linear":
        flash = darsia.SimpleFlash(**response["metadata"]["flash_bounds"])
        aqueous, gas = flash(response["analysis"]._scalar(response["native_signal"], response["frame"].elapsed_h))
        source = dict(response, flash=flash, prepared=replace(response["prepared"], aqueous_relative=np.asarray(aqueous.img, np.float32), gas_saturation=np.asarray(gas.img, np.float32)))
    elif flash_kind != "titration":
        raise ValueError("Unknown common flash sensitivity")
    elif not isinstance(response["flash"], darsia.TitrationFlash):
        raise ValueError("The reusable main optical response must have the native titration flash")
    output = evaluate_native_response(source, freeze["parameters"], fields=fields, phase_constraint=phase_constraint, gas_equilibrium=gas_equilibrium, port_as_aqueous=port_as_aqueous, allow_provisional_l1_sensitivity=allow_provisional_l1_sensitivity)
    output.update(frozen_calibration_sha256=digest, frozen_utc=freeze["frozen_utc"], flash_kind=flash_kind,
        evaluation_role="held_out" if normalize_run(response["frame"].run_id) in HELDOUT_RUNS else "calibration_run_production")
    return output
