"""Native DarSIA concentration and mass on immutable, prepared phase support.

The optical calibration is an explicit saved DarSIA artifact. Applying it does
not fit anything. Optional mass fitting creates a separate parameter record;
it never replaces the color paths, saved calibration, or selected phase labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

import darsia
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.signals.color.color_embedding import ColorEmbeddingBasis


VERSION = "native-darsia-fixed-phase-mass-v1"
SUMMARY_INTERVALS = 4096


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value, shape, name, *, positive=False):
    result = np.broadcast_to(np.asarray(value, dtype=np.float32), shape).copy()
    if not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f"Invalid {name}")
    return result


@dataclass(frozen=True)
class CalibrationParameters:
    aqueous_scale: float = 1.0
    gas_scale: float = 1.0

    def __post_init__(self):
        if not np.isfinite([self.aqueous_scale, self.gas_scale]).all() or min(
            self.aqueous_scale, self.gas_scale
        ) <= 0:
            raise ValueError("Calibration scales must be finite and positive")


@dataclass(frozen=True)
class PreparedNativeFrame:
    run_id: str
    time_hours: float
    aqueous_relative: np.ndarray
    gas_saturation: np.ndarray
    phase_labels: np.ndarray
    pressure_bar: float
    temperature_c: float
    nominal_injected_mass_g: float | None = None
    pressure_gradient_bar_per_m: float = 0.0
    temperature_gradient_c_per_m: float = 0.0
    pixel_area_m2: np.ndarray | float | None = None


@dataclass(frozen=True)
class PreparedSignalFrame:
    """Native facies signal before flash clipping; no mass fit is applied."""

    run_id: str
    time_hours: float
    native_signal: np.ndarray
    phase_labels: np.ndarray
    pressure_bar: float
    temperature_c: float
    nominal_injected_mass_g: float | None = None
    pressure_gradient_bar_per_m: float = 0.0
    temperature_gradient_c_per_m: float = 0.0
    pixel_area_m2: np.ndarray | float | None = None


def _store_area_metadata(folder, metadata):
    area = metadata.get("pixel_area_m2")
    if isinstance(area, np.ndarray):
        np.save(folder / "pixel_area_m2.npy", area, allow_pickle=False)
        metadata["pixel_area_m2"] = None
        metadata["pixel_area_m2_array"] = True


def _load_area_metadata(folder, metadata, mmap_mode):
    if metadata.pop("pixel_area_m2_array", False):
        metadata["pixel_area_m2"] = np.load(folder / "pixel_area_m2.npy", mmap_mode=mmap_mode, allow_pickle=False)


def save_signal_frame(folder, frame):
    """Store five bytes per pixel, publishing frame metadata after its arrays."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = ("native_signal", "phase_labels")
    metadata = {key: getattr(frame, key) for key in frame.__dataclass_fields__ if key not in arrays}
    metadata["format"] = "native-darsia-preflash-frame-v1"
    _store_area_metadata(folder, metadata)
    for key in arrays:
        np.save(folder / (key + ".npy"), getattr(frame, key), allow_pickle=False)
    temporary = folder / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
    temporary.replace(folder / "metadata.json")


def load_signal_frame(folder, *, mmap_mode="r"):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.pop("format") != "native-darsia-preflash-frame-v1":
        raise ValueError("Unknown preflash frame cache format")
    for key in ("native_signal", "phase_labels"):
        metadata[key] = np.load(folder / (key + ".npy"), mmap_mode=mmap_mode, allow_pickle=False)
    _load_area_metadata(folder, metadata, mmap_mode)
    return PreparedSignalFrame(**metadata)


def save_prepared_frame(folder, frame):
    """Persist one frame; publish metadata last, permitting bounded mmap reads."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = ("aqueous_relative", "gas_saturation", "phase_labels")
    metadata = {name: getattr(frame, name) for name in frame.__dataclass_fields__ if name not in arrays}
    metadata["format"] = VERSION
    _store_area_metadata(folder, metadata)
    for name in arrays:
        np.save(folder / (name + ".npy"), getattr(frame, name), allow_pickle=False)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_prepared_frame(folder, *, mmap_mode="r"):
    """Load one frame without materializing its three arrays in memory."""
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.pop("format") != VERSION:
        raise ValueError("Unknown prepared native concentration format")
    for name in ("aqueous_relative", "gas_saturation", "phase_labels"):
        metadata[name] = np.load(folder / (name + ".npy"), mmap_mode=mmap_mode, allow_pickle=False)
    _load_area_metadata(folder, metadata, mmap_mode)
    return PreparedNativeFrame(**metadata)


class NativeConcentrationAnalysis:
    """Apply a saved native calibration on one explicitly registered grid.

    ``labels`` is the exact calibration basis (facies or segmentation labels),
    not the water/aqueous/gas classification. Depth, porosity, dimensions and
    atmospheric state are supplied measured inputs; none is inferred here.
    """

    def __init__(self, calibration_folder, *, baseline_rgb, labels, domain,
                 depth_m, porosity, dimensions_m, basis="facies",
                 allow_legacy_basis=False, restoration=None,
                 calibration_status="provisional imported native calibration"):
        self.folder = Path(calibration_folder).resolve()
        self.labels = np.asarray(labels)
        self.domain = np.asarray(domain, dtype=bool)
        if (self.labels.shape != self.domain.shape or self.domain.ndim != 2
                or not self.domain.any() or not np.issubdtype(self.labels.dtype, np.integer)
                or np.any(self.labels < 0)):
            raise ValueError("Finite, nonnegative calibration labels and registered domain required")
        self.shape = self.domain.shape
        self.dimensions = tuple(map(float, dimensions_m))
        if len(self.dimensions) != 2 or not np.isfinite(self.dimensions).all() or min(self.dimensions) <= 0:
            raise ValueError("Measured physical height and width are required")
        depth = _array(depth_m, self.shape, "depth")
        if np.any(depth < 0) or np.any(depth[self.domain] <= 0):
            raise ValueError("Depth must be positive inside the measured porous domain")
        pore = _array(porosity, self.shape, "porosity")
        if np.any(pore < 0) or np.any(pore > 1) or np.any(pore[self.domain] <= 0):
            raise ValueError("Porosity must be in (0,1] in the measured porous domain")
        self.geometry = darsia.ExtrudedPorousGeometry(
            porosity=np.where(self.domain, pore, 0), depth=depth,
            space_dim=2, num_voxels=self.shape, dimensions=list(self.dimensions))
        self._nominal_voxel_volume = self.geometry.voxel_volume.copy()
        self._nominal_pixel_area_m2 = float(np.prod(self.dimensions)) / np.prod(self.shape)
        self.pixel_area_m2 = self._nominal_pixel_area_m2
        self.baseline = self._rgb(baseline_rgb)
        self.label_image = darsia.ScalarImage(self.labels, dimensions=list(self.dimensions))
        metadata_path = self.folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        self.ignored_labels = tuple(map(int, metadata.get("ignore_labels", [])))
        self._unquantified_domain = self.domain & np.isin(self.labels, self.ignored_labels)
        self.basis = ColorEmbeddingBasis(basis)
        legacy = not metadata.get("basis") or metadata.get("label_ids") is None
        if legacy and not allow_legacy_basis:
            raise ValueError("Legacy calibration needs an explicitly confirmed label basis")
        files = [metadata_path, self.folder / "flash/flash.json"]
        for label in np.unique(self.labels):
            files.extend((
                self.folder / "color_path_interpretation" / f"color_path_interpretation_{label}.json",
                self.folder / "signal_model" / f"signal_model_{label}.csv"))
        if any(not path.is_file() for path in files):
            raise FileNotFoundError("Incomplete native calibration: " + ", ".join(str(p) for p in files if not p.is_file()))
        self.source_hashes = {str(path): _sha(path) for path in files}
        # Loading the optical pipeline does not evaluate any thermodynamic state.
        self.pipeline = HeterogeneousColorToMassAnalysis.load(
            self.folder, baseline=self.baseline, labels=self.label_image,
            co2_mass_analysis=None, geometry=self.geometry,
            restoration=restoration, basis=self.basis)
        self.calibration_status = calibration_status
        self.legacy_basis_confirmed = bool(legacy)
        self._physics = None
        self._physics_key = None

    def set_pixel_area(self, pixel_area_m2=None):
        """Keep DarSIA integration while supplying recorded per-frame areas."""
        from .phase_pixel_area import validate_pixel_area
        area = (self._nominal_pixel_area_m2 if pixel_area_m2 is None else
                validate_pixel_area(pixel_area_m2, self.shape))
        self.pixel_area_m2 = area
        # ExtrudedPorousGeometry already includes depth and porosity. Replace
        # only its planar factor, then invalidate its native integration cache.
        self.geometry.voxel_volume = self._nominal_voxel_volume * (area / self._nominal_pixel_area_m2)
        self.geometry.cached_voxel_volume = self.geometry.voxel_volume.copy()

    def _rgb(self, rgb):
        values = np.asarray(rgb, dtype=np.float32)
        if values.shape != (*self.shape, 3) or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("Corrected float32 RGB on the registered grid in [0,1] required")
        return darsia.OpticalImage(values.copy(), color_space="RGB", dimensions=list(self.dimensions))

    def _scalar(self, values, time_hours=None):
        image = darsia.ScalarImage(np.asarray(values, dtype=np.float32), dimensions=list(self.dimensions))
        image.time = None if time_hours is None else float(time_hours) * 3600
        return image

    def verify_sources(self):
        if any(_sha(path) != expected for path, expected in self.source_hashes.items()):
            raise ValueError("Saved native calibration changed during concentration analysis")

    def provenance(self):
        self.verify_sources()
        native = (darsia.ConcentrationAnalysis, darsia.ColorPath, darsia.ColorPathInterpolation,
                  darsia.HeterogeneousModel, darsia.PWTransformation, type(self.pipeline.flash),
                  HeterogeneousColorToMassAnalysis, darsia.CO2MassAnalysis, darsia.ExtrudedPorousGeometry)
        return dict(format=VERSION, calibration_status=self.calibration_status,
            calibration_files=dict(self.source_hashes), basis=self.basis.value,
            ignored_calibration_labels=list(self.ignored_labels),
            legacy_basis_explicitly_confirmed=self.legacy_basis_confirmed,
            native_implementations={obj.__name__: _sha(inspect.getfile(obj)) for obj in native},
            flash_model=type(self.pipeline.flash).__name__,
            flash_origin=('existing FluidFlower fork extension' if type(self.pipeline.flash).__name__ == 'TitrationFlash'
                          else 'upstream DarSIA'),
            phase_policy="Selected categorical support is fixed; native saturation is zero outside gas support",
            concentration_units="g/L (numerically equal to kg/m3)", mass_units="g",
            local_concentration_validation=False, host_sha256=_sha(__file__))

    def prepare(self, rgb, phase_labels, *, run_id, time_hours, pressure_bar,
                temperature_c, nominal_injected_mass_g=None,
                pressure_gradient_bar_per_m=0.0, temperature_gradient_c_per_m=0.0,
                pixel_area_m2=None):
        self.set_pixel_area(pixel_area_m2)
        phase = np.asarray(phase_labels)
        if phase.shape != self.shape or not np.isin(phase, [0, 1, 2]).all():
            raise ValueError("Fixed categorical phase support must use water=0, aqueous=1, gas=2")
        if np.any(phase[~self.domain] != 0):
            raise ValueError("Selected phase support extends outside the registered domain")
        if not np.isfinite([time_hours, pressure_bar, temperature_c,
                           pressure_gradient_bar_per_m, temperature_gradient_c_per_m]).all() or pressure_bar <= 0:
            raise ValueError("Explicit finite time, pressure and temperature are required")
        if nominal_injected_mass_g is not None and (not np.isfinite(nominal_injected_mass_g) or nominal_injected_mass_g < 0):
            raise ValueError("Invalid nominal injected mass")
        color = self.pipeline.call_color_interpretation(self._rgb(rgb))
        signal = self.pipeline.call_pH_analysis(color)
        aqueous, gas = self.pipeline.flash(signal)
        return PreparedNativeFrame(str(run_id), float(time_hours),
            np.asarray(aqueous.img, np.float32), np.asarray(gas.img, np.float32),
            phase.astype(np.uint8, copy=True), float(pressure_bar), float(temperature_c),
            None if nominal_injected_mass_g is None else float(nominal_injected_mass_g),
            float(pressure_gradient_bar_per_m), float(temperature_gradient_c_per_m), pixel_area_m2)

    def _validate_frame(self, frame):
        self.set_pixel_area(getattr(frame, "pixel_area_m2", None))
        if any(array.shape != self.shape for array in (frame.phase_labels, frame.aqueous_relative, frame.gas_saturation)):
            raise ValueError("Prepared frame belongs to another physical grid")
        if (not np.isin(frame.phase_labels, [0, 1, 2]).all()
                or np.any(frame.phase_labels[~self.domain] != 0)):
            raise ValueError("Prepared phase support is invalid or extends outside the domain")
        if any(not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1)
               for array in (frame.aqueous_relative, frame.gas_saturation)):
            raise ValueError("Native aqueous and gas responses must be finite in [0,1]")
        if (not np.isfinite([frame.time_hours, frame.pressure_bar, frame.temperature_c,
                            frame.pressure_gradient_bar_per_m, frame.temperature_gradient_c_per_m]).all()
                or frame.pressure_bar <= 0):
            raise ValueError("Prepared atmospheric state and time must be finite")

    def prepare_signal(self, rgb, phase_labels, *, run_id, time_hours, pressure_bar,
                       temperature_c, nominal_injected_mass_g=None,
                       pressure_gradient_bar_per_m=0.0, temperature_gradient_c_per_m=0.0,
                       pixel_area_m2=None):
        """Keep the observed signal before distinct colors are clipped by flash."""
        self.set_pixel_area(pixel_area_m2)
        phase = np.asarray(phase_labels)
        if phase.shape != self.shape or not np.isin(phase, [0, 1, 2]).all():
            raise ValueError("Fixed categorical phase support must use water=0, aqueous=1, gas=2")
        if np.any(phase[~self.domain] != 0):
            raise ValueError("Selected phase support extends outside the registered domain")
        if not np.isfinite([time_hours, pressure_bar, temperature_c,
                           pressure_gradient_bar_per_m, temperature_gradient_c_per_m]).all() or pressure_bar <= 0:
            raise ValueError("Explicit finite time, pressure and temperature are required")
        if nominal_injected_mass_g is not None and (not np.isfinite(nominal_injected_mass_g) or nominal_injected_mass_g < 0):
            raise ValueError("Invalid nominal injected mass")
        color = self.pipeline.call_color_interpretation(self._rgb(rgb))
        signal = self.pipeline.call_pH_analysis(color)
        return PreparedSignalFrame(str(run_id), float(time_hours), np.asarray(signal.img, np.float32),
            phase.astype(np.uint8, copy=True), float(pressure_bar), float(temperature_c),
            None if nominal_injected_mass_g is None else float(nominal_injected_mass_g),
            float(pressure_gradient_bar_per_m), float(temperature_gradient_c_per_m), pixel_area_m2)

    def _validate_signal_frame(self, frame):
        self.set_pixel_area(getattr(frame, "pixel_area_m2", None))
        if frame.native_signal.shape != self.shape or frame.phase_labels.shape != self.shape:
            raise ValueError("Prepared signal belongs to another physical grid")
        if (not np.isin(frame.phase_labels, [0, 1, 2]).all()
                or np.any(frame.phase_labels[~self.domain] != 0)):
            raise ValueError("Prepared phase support is invalid or extends outside the domain")
        if not np.isfinite(frame.native_signal).all() or np.any(frame.native_signal < 0):
            raise ValueError("Prepared native signal must be finite and nonnegative")
        if (not np.isfinite([frame.time_hours, frame.pressure_bar, frame.temperature_c,
                            frame.pressure_gradient_bar_per_m, frame.temperature_gradient_c_per_m]).all()
                or frame.pressure_bar <= 0):
            raise ValueError("Prepared atmospheric state and time must be finite")

    def signal_flash(self):
        if type(self.pipeline.flash) is not darsia.SimpleFlash:
            raise ValueError("Native preflash power calibration currently requires saved SimpleFlash chemistry")
        return {key: float(getattr(self.pipeline.flash, key)) for key in
                ("min_value_aq", "max_value_aq", "min_value_g", "max_value_g")}

    def signal_provenance(self):
        from . import concentration_signal_calibration as signal
        record = self.provenance()
        record.update(signal_calibration_format=signal.SIGNAL_VERSION,
                      signal_calibration_sha256=_sha(signal.__file__), flash=self.signal_flash())
        return record

    def empty_signal_summary(self, phase_labels, *, run_id, time_hours, nominal_injected_mass_g=None):
        from .concentration_signal_calibration import SIGNAL_VERSION
        self._validate_empty(phase_labels, time_hours, nominal_injected_mass_g)
        return dict(format=SIGNAL_VERSION, run=run_id, elapsed_h=float(time_hours),
            nominal_injected_mass_g=nominal_injected_mass_g, unquantified_phase_pixels=0,
            unknown_mass_upper_g=0., signal_values=[], aqueous_capacity_g=[],
            gas_aqueous_capacity_g=[], gas_capacity_g=[], signal_quantization_half_width=0.,
            flash=self.signal_flash(), state_not_required=True)

    def _unknown_mass_upper(self, phase, physics):
        unknown = self._unquantified_domain & (phase != 0)
        if not unknown.any():
            return 0.
        # Aqueous support holds at most solubility * pore volume. Gas support
        # has complementary water/gas saturations, so the maximum is the
        # larger of its aqueous and gas capacities, not their sum.
        density = np.where(phase == 2,
                           np.maximum(physics.solubility_co2, physics.density_gaseous_co2),
                           physics.solubility_co2)
        return float(self.geometry.integrate(np.where(unknown, density, 0))) * 1000

    def summarize_signal(self, frame):
        """Reduce native preflash observations, with bounds for uncalibrated sand."""
        from .concentration_signal_calibration import SIGNAL_VERSION
        self._validate_signal_frame(frame)
        physics = self._mass_physics(frame)
        known = self.domain & ~self._unquantified_domain & (frame.phase_labels != 0)
        values = frame.native_signal[known]
        step = max(float(values.max(initial=0.)), 1.) / 8192
        codes, inverse = np.unique(np.rint(values / step).astype(np.int32), return_inverse=True)
        is_gas = frame.phase_labels[known] == 2
        aq_weight = self.geometry.make_extensive(self._scalar(physics.solubility_co2)).img[known] * 1000
        gas_weight = self.geometry.make_extensive(self._scalar(physics.density_gaseous_co2)).img[known] * 1000
        def aggregate(weight):
            return np.bincount(inverse, weights=weight, minlength=len(codes)).tolist()
        return dict(format=SIGNAL_VERSION, run=frame.run_id, elapsed_h=frame.time_hours,
            nominal_injected_mass_g=frame.nominal_injected_mass_g,
            signal_values=(codes.astype(np.float64) * step).tolist(),
            aqueous_capacity_g=aggregate(aq_weight * ~is_gas),
            gas_aqueous_capacity_g=aggregate(aq_weight * is_gas), gas_capacity_g=aggregate(gas_weight * is_gas),
            unquantified_phase_pixels=int(np.count_nonzero(self._unquantified_domain & (frame.phase_labels != 0))),
            unknown_mass_upper_g=self._unknown_mass_upper(frame.phase_labels, physics),
            signal_quantization_half_width=step / 2, flash=self.signal_flash())

    def evaluate_signal(self, frame, parameters=None, *, fields=True):
        from .concentration_signal_calibration import SignalCalibrationParameters, signal_response
        self._validate_signal_frame(frame)
        parameters = parameters or SignalCalibrationParameters()
        aqueous, gas = signal_response(frame.native_signal, parameters, self.signal_flash())
        prepared = PreparedNativeFrame(frame.run_id, frame.time_hours, aqueous, gas, frame.phase_labels,
            frame.pressure_bar, frame.temperature_c, frame.nominal_injected_mass_g,
            frame.pressure_gradient_bar_per_m, frame.temperature_gradient_c_per_m,
            getattr(frame, "pixel_area_m2", None))
        result = self.evaluate(prepared, fields=fields)
        physics = self._mass_physics(frame)
        unknown = self._unknown_mass_upper(frame.phase_labels, physics)
        known_aqueous = self.domain & ~self._unquantified_domain & (frame.phase_labels == 1)
        water_volume_l = float(self.geometry.integrate(known_aqueous.astype(np.float32))) * 1000
        # The volume mean follows the observed optical response; it is never
        # assigned by distributing the nominal dose over the growing plume.
        mass_in_aqueous_support_g = float(self.geometry.integrate(
            np.where(known_aqueous, aqueous * physics.solubility_co2, 0))) * 1000
        result.update(total_mass_lower_g=result['total_mass_g'],
            total_mass_upper_g=result['total_mass_g'] + unknown, unknown_mass_upper_g=unknown,
            aqueous_support_area_m2=float(np.sum(np.broadcast_to(self.pixel_area_m2, self.shape),
                                                  where=frame.phase_labels == 1, dtype=np.float64)),
            quantified_aqueous_pore_volume_l=water_volume_l,
            mean_aqueous_concentration_g_l=(mass_in_aqueous_support_g / water_volume_l if water_volume_l else None),
            concentration_mapping='time-independent monotone native preflash response',
            unknown_bound_interpretation='physical capacity bound, not a statistical confidence interval')
        return result

    def _mass_physics(self, frame):
        key = (frame.pressure_bar, frame.temperature_c,
               frame.pressure_gradient_bar_per_m, frame.temperature_gradient_c_per_m)
        if key != self._physics_key:
            self._physics = darsia.CO2MassAnalysis(
                self._scalar(np.zeros(self.shape, dtype=np.float32)),
                atmospheric_pressure=key[0], atmospheric_temperature=key[1],
                atmospheric_pressure_gradient=key[2], atmospheric_temperature_gradient=key[3])
            self._physics_key = key
        return self._physics

    def _validate_empty(self, phase_labels, time_hours, nominal_injected_mass_g):
        phase = np.asarray(phase_labels)
        if phase.shape != self.shape or np.any(phase != 0):
            raise ValueError("Empty evaluation requires all-zero fixed phase labels on the registered grid")
        if not np.isfinite(time_hours):
            raise ValueError("Empty evaluation requires a finite frame time")
        if nominal_injected_mass_g is not None and (
                not np.isfinite(nominal_injected_mass_g) or nominal_injected_mass_g < 0):
            raise ValueError("Invalid nominal injected mass")
        return phase

    def empty_evaluation(self, phase_labels, *, run_id, time_hours,
                         nominal_injected_mass_g=None, fields=True):
        """Exact zero mass for an empty fixed support, without requiring PT.

        This is the native mass formula's zero-support invariant, not a fitted
        concentration or an assumed thermodynamic state. Solubility is unknown
        and deliberately omitted rather than extrapolated for pre-injection.
        """
        phase = self._validate_empty(phase_labels, time_hours, nominal_injected_mass_g)
        output = dict(run=str(run_id), elapsed_h=float(time_hours),
            nominal_injected_mass_g=nominal_injected_mass_g,
            total_mass_g=0., aqueous_mass_g=0., gas_mass_g=0.,
            solubility_max_g_l=None, phase_support_unchanged=True,
            gas_support_with_zero_native_saturation_pixels=0,
            unquantified_phase_pixels=0, unquantified_phase_pixels_by_calibration_label={},
            quantification_complete=True, mass_estimate_status="all_selected_support_evaluated",
            state_not_required=True, thermodynamic_evaluation="skipped_empty_fixed_phase_support")
        if fields:
            output["fields"] = {name: np.zeros(self.shape, dtype=np.float32) for name in (
                "concentration_g_l", "aqueous_relative", "gas_saturation", "mass_density_g_l",
                "gas_mass_density_g_l", "aqueous_mass_density_g_l")}
            output["fields"].update(phase_labels=phase.astype(np.uint8, copy=True),
                                    unquantified_phase_mask=np.zeros(self.shape, dtype=bool))
        return output

    def empty_summary(self, phase_labels, *, run_id, time_hours, nominal_injected_mass_g=None):
        """Represent an empty frame in a fitting timeline without synthetic PT."""
        self._validate_empty(phase_labels, time_hours, nominal_injected_mass_g)
        return dict(format=VERSION, run=str(run_id), elapsed_h=float(time_hours),
            nominal_injected_mass_g=nominal_injected_mass_g, unquantified_phase_pixels=0,
            phase_support_pixels=0, unknown_mass_upper_g=0., unknown_gas_mass_upper_g=0.,
            state_not_required=True, thermodynamic_evaluation="skipped_empty_fixed_phase_support",
            intervals=SUMMARY_INTERVALS, aqueous_response=[], gas_response=[],
            aqueous_capacity_g=[], gas_aqueous_capacity_g=[], gas_capacity_g=[])

    def summarize_frame(self, frame):
        """Bounded response histogram for fitting; final maps use exact native IO.

        Each bin stores native geometry-integrated capacity, so differing depth,
        porosity, pressure and pixel area retain their physical weights. Only
        native response values are rounded; exact zero/one endpoints remain.
        """
        self._validate_frame(frame)
        physics = self._mass_physics(frame)
        phase_support = self.domain & (frame.phase_labels != 0)
        unknown = self._unquantified_domain & phase_support
        support = phase_support & ~unknown
        aq = np.rint(frame.aqueous_relative[support] * SUMMARY_INTERVALS).astype(np.int32)
        gas = np.rint(frame.gas_saturation[support] * SUMMARY_INTERVALS).astype(np.int32)
        codes, inverse = np.unique(aq * (SUMMARY_INTERVALS + 1) + gas, return_inverse=True)
        is_gas = frame.phase_labels[support] == 2
        aq_weight = self.geometry.make_extensive(self._scalar(physics.solubility_co2)).img[support] * 1000
        gas_weight = self.geometry.make_extensive(self._scalar(physics.density_gaseous_co2)).img[support] * 1000
        def aggregate(weight):
            return np.bincount(inverse, weights=weight, minlength=len(codes)).tolist()
        return dict(format=VERSION, run=frame.run_id, elapsed_h=frame.time_hours,
            nominal_injected_mass_g=frame.nominal_injected_mass_g,
            unquantified_phase_pixels=int(np.count_nonzero(unknown)),
            phase_support_pixels=int(np.count_nonzero(phase_support)),
            unknown_mass_upper_g=self._unknown_mass_upper(frame.phase_labels, physics),
            unknown_gas_mass_upper_g=float(self.geometry.integrate(np.where(
                unknown & (frame.phase_labels == 2), physics.density_gaseous_co2, 0))) * 1000,
            intervals=SUMMARY_INTERVALS,
            aqueous_response=(codes // (SUMMARY_INTERVALS + 1) / SUMMARY_INTERVALS).tolist(),
            gas_response=(codes % (SUMMARY_INTERVALS + 1) / SUMMARY_INTERVALS).tolist(),
            aqueous_capacity_g=aggregate(aq_weight * ~is_gas),
            gas_aqueous_capacity_g=aggregate(aq_weight * is_gas),
            gas_capacity_g=aggregate(gas_weight * is_gas))

    def evaluate(self, frame, parameters=CalibrationParameters(), *, fields=True):
        self._validate_frame(frame)
        aqueous = np.clip(frame.aqueous_relative * parameters.aqueous_scale, 0, 1)
        gas = np.clip(frame.gas_saturation * parameters.gas_scale, 0, 1)
        known = self.domain & ~self._unquantified_domain
        aqueous = np.where(known & (frame.phase_labels != 0), aqueous, 0)
        gas = np.where(known & (frame.phase_labels == 2), gas, 0)
        physics = self._mass_physics(frame)
        result = physics.mass_analysis(self._scalar(aqueous, frame.time_hours), self._scalar(gas, frame.time_hours))
        masses = {name: float(self.geometry.integrate(value)) * 1000 for name, value in (
            ("total_mass_g", result.mass), ("aqueous_mass_g", result.mass_aq), ("gas_mass_g", result.mass_g))}
        if not np.isclose(masses["total_mass_g"], masses["aqueous_mass_g"] + masses["gas_mass_g"], rtol=1e-6, atol=1e-7):
            raise ValueError("Native mass integration does not conserve component totals")
        output = dict(run=frame.run_id, elapsed_h=frame.time_hours,
            nominal_injected_mass_g=frame.nominal_injected_mass_g, **masses,
            solubility_max_g_l=float(np.max(physics.solubility_co2[self.domain])),
            phase_support_unchanged=True,
            gas_support_with_zero_native_saturation_pixels=int(np.count_nonzero((frame.phase_labels == 2) & (gas <= 0))))
        unknown = self._unquantified_domain & (frame.phase_labels != 0)
        unknown_labels, counts = np.unique(self.labels[unknown], return_counts=True)
        unknown_mass_upper = self._unknown_mass_upper(frame.phase_labels, physics)
        unknown_gas_mass_upper = float(self.geometry.integrate(np.where(
            unknown & (frame.phase_labels == 2), physics.density_gaseous_co2, 0))) * 1000
        output.update(unquantified_phase_pixels=int(np.count_nonzero(unknown)),
            phase_support_pixels=int(np.count_nonzero(self.domain & (frame.phase_labels != 0))),
            unquantified_phase_pixels_by_calibration_label={str(int(label)): int(count)
                for label, count in zip(unknown_labels, counts)},
            quantification_complete=not bool(unknown.any()),
            mass_estimate_status="partial_known_support_only" if unknown.any() else "all_selected_support_evaluated",
            total_mass_lower_g=masses["total_mass_g"],
            total_mass_upper_g=masses["total_mass_g"] + unknown_mass_upper,
            unknown_mass_upper_g=unknown_mass_upper,
            gas_mass_lower_g=masses["gas_mass_g"],
            gas_mass_upper_g=masses["gas_mass_g"] + unknown_gas_mass_upper,
            unknown_gas_mass_upper_g=unknown_gas_mass_upper)
        if fields:
            output["fields"] = dict(
                concentration_g_l=np.asarray(aqueous * physics.solubility_co2, np.float32),
                aqueous_relative=np.asarray(aqueous, np.float32), gas_saturation=np.asarray(gas, np.float32),
                mass_density_g_l=np.asarray(result.mass.img, np.float32),
                gas_mass_density_g_l=np.asarray(result.mass_g.img, np.float32),
                aqueous_mass_density_g_l=np.asarray(result.mass_aq.img, np.float32),
                phase_labels=frame.phase_labels.copy(), unquantified_phase_mask=unknown)
        return output


def evaluate_summary(summary, parameters=CalibrationParameters()):
    """Evaluate a small fitting histogram, with a conservative rounding bound."""
    aq = np.clip(np.asarray(summary["aqueous_response"]) * parameters.aqueous_scale, 0, 1)
    gas = np.clip(np.asarray(summary["gas_response"]) * parameters.gas_scale, 0, 1)
    aw = np.asarray(summary["aqueous_capacity_g"])
    gaw = np.asarray(summary["gas_aqueous_capacity_g"])
    gw = np.asarray(summary["gas_capacity_g"])
    aqueous_mass = float(np.sum(aw * aq + gaw * aq * (1 - gas)))
    gas_mass = float(np.sum(gw * gas))
    error = (parameters.aqueous_scale * np.sum(aw + gaw)
             + parameters.gas_scale * np.sum(gaw + gw)) / (2 * summary["intervals"])
    return dict(run=summary["run"], elapsed_h=summary["elapsed_h"],
        nominal_injected_mass_g=summary["nominal_injected_mass_g"],
        aqueous_mass_g=aqueous_mass, gas_mass_g=gas_mass,
        total_mass_g=aqueous_mass + gas_mass, summary_mass_error_bound_g=float(error),
        unquantified_phase_pixels=int(summary.get("unquantified_phase_pixels", 0)),
        phase_support_pixels=int(summary.get("phase_support_pixels", 0)),
        total_mass_lower_g=aqueous_mass + gas_mass,
        total_mass_upper_g=aqueous_mass + gas_mass + float(summary.get("unknown_mass_upper_g", 0.)),
        gas_mass_lower_g=gas_mass,
        gas_mass_upper_g=gas_mass + float(summary.get("unknown_gas_mass_upper_g", 0.)))


def fit_summaries(summaries_by_run, native_provenance_by_run, *, scope="individual",
                  overflow_mode="off", references=(), unknown_policy="complete_frames"):
    """Explicit two-parameter mass calibration of the saved native response.

    References are already interpreted measurements, never raw displaced-water
    readings: {run, frame_index, quantity: gas_mass_g|total_mass_g,
    mass_g, sigma_g, source}. Comparison never affects the fitted parameters.
    """
    runs = summaries_by_run
    if scope not in {"individual", "shared"} or (scope == "individual" and len(runs) != 1) or not runs:
        raise ValueError("Individual fitting requires one run; shared fitting requires explicit runs")
    if scope == "shared" and len(runs) < 2:
        raise ValueError("Shared fitting requires at least two compatible runs")
    if overflow_mode not in {"off", "compare", "constrain"}:
        raise ValueError("Unknown overflow mode")
    if unknown_policy not in {"complete_frames", "bounded_interval"}:
        raise ValueError("Unknown missing-calibration policy")
    refs = list(references)
    if overflow_mode != "off" and not refs:
        raise ValueError("Overflow comparison/constraints require explicit measured references")
    for ref in refs:
        if (ref.get("run") not in runs or ref.get("quantity") not in {"gas_mass_g", "total_mass_g"}
                or not ref.get("source") or not np.isfinite([ref.get("mass_g", np.nan), ref.get("sigma_g", np.nan)]).all()
                or ref["mass_g"] < 0 or ref["sigma_g"] <= 0
                or not 0 <= int(ref["frame_index"]) < len(runs[ref["run"]])):
            raise ValueError("A typed, sourced overflow-derived mass and positive uncertainty are required")
    selected = {}
    for run, frames in runs.items():
        if run not in native_provenance_by_run:
            raise ValueError("Native calibration provenance is required for every run")
        for frame in frames:
            if unknown_policy == "complete_frames" and frame.get("unquantified_phase_pixels", 0):
                raise ValueError(f"Cannot fit {run}: selected CO2 phase is present in ignored native calibration labels; calibrate those labels first")
            if (frame.get("unquantified_phase_pixels", 0) and
                    (frame.get("phase_support_pixels", 0) < frame["unquantified_phase_pixels"]
                     or not np.isfinite([frame.get("unknown_mass_upper_g", np.nan),
                                         frame.get("unknown_gas_mass_upper_g", np.nan)]).all()
                     or min(frame["unknown_mass_upper_g"], frame["unknown_gas_mass_upper_g"]) < 0)):
                raise ValueError("Ignored calibration support requires finite physical mass bounds")
            names = ("aqueous_response", "gas_response", "aqueous_capacity_g",
                     "gas_aqueous_capacity_g", "gas_capacity_g")
            arrays = [np.asarray(frame[name]) for name in names]
            if (frame.get("format") != VERSION or frame.get("intervals") != SUMMARY_INTERVALS
                    or any(array.ndim != 1 or array.shape != arrays[0].shape
                           or not np.isfinite(array).all() or np.any(array < 0) for array in arrays)
                    or any(np.any(array > 1) for array in arrays[:2])):
                raise ValueError("Invalid native response summary")
            nominal = frame["nominal_injected_mass_g"]
            if nominal is not None and (not np.isfinite(nominal) or nominal < 0):
                raise ValueError("Invalid nominal injected mass")
        indices = [i for i, frame in enumerate(frames) if frame["nominal_injected_mass_g"] is not None and frame["nominal_injected_mass_g"] > 0]
        if len(indices) < 3 or any(frame["run"] != run for frame in frames):
            raise ValueError("Every run needs at least three positive injected-mass observations on its own timeline")
        selected[run] = indices
    def predictions(x):
        parameters = CalibrationParameters(*map(float, x))
        return {run: [evaluate_summary(frame, parameters) for frame in frames]
                for run, frames in runs.items()}
    def bounded_residual(predicted, upper, target):
        return np.maximum(predicted-target, 0.) + np.minimum(predicted+upper-target, 0.)
    def residuals(x):
        values = predictions(x)
        result = []
        for run, frames in runs.items():
            indices = selected[run]
            dose = max(float(frames[i]["nominal_injected_mass_g"]) for i in indices)
            result.extend((bounded_residual(values[run][i]["total_mass_g"],
                           frames[i].get("unknown_mass_upper_g", 0.) if unknown_policy == "bounded_interval" else 0.,
                           frames[i]["nominal_injected_mass_g"]))
                          / dose / np.sqrt(len(indices)) for i in indices)
        if overflow_mode == "constrain":
            for r in refs:
                frame = runs[r["run"]][int(r["frame_index"])]
                upper_key = "unknown_gas_mass_upper_g" if r["quantity"] == "gas_mass_g" else "unknown_mass_upper_g"
                upper = frame.get(upper_key, 0.) if unknown_policy == "bounded_interval" else 0.
                result.append(bounded_residual(values[r["run"]][int(r["frame_index"])][r["quantity"]],
                                               upper, r["mass_g"]) / r["sigma_g"])
        return np.asarray(result)
    fit = least_squares(residuals, [1., 1.], bounds=([.05, .05], [4., 4.]), max_nfev=100, diff_step=1e-3)
    parameters = CalibrationParameters(*map(float, fit.x))
    output = predictions(fit.x)
    singular_values = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.count_nonzero(singular_values > max(float(singular_values.max(initial=0)) * 1e-6, 1e-10)))
    comparisons = [] if overflow_mode == "off" else [dict(r,
        predicted_mass_g=output[r["run"]][int(r["frame_index"])][r["quantity"]]) for r in refs]
    unknown_coverage = {run: dict(frames_with_unknown=sum(bool(frame.get("unquantified_phase_pixels", 0)) for frame in frames),
        maximum_unknown_phase_fraction=max((frame.get("unquantified_phase_pixels", 0) /
            max(frame.get("phase_support_pixels", 0), 1) for frame in frames), default=0.),
        maximum_unknown_mass_upper_g=max((frame.get("unknown_mass_upper_g", 0.) for frame in frames), default=0.))
        for run, frames in runs.items()}
    return dict(format=VERSION, scope=scope, fit_performed=True, parameters=asdict(parameters),
        runs=list(runs), training_indices=selected, optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message), jacobian_rank=rank,
        parameter_identifiability_warning=(None if rank == 2 else
            "These mass observations do not independently constrain both scale parameters"),
        parameters_at_bounds=[name for name, value in asdict(parameters).items() if min(value-.05, 4.-value) < 1e-3],
        overflow_mode=overflow_mode, overflow_comparisons=comparisons,
        unknown_policy=unknown_policy, unknown_coverage=unknown_coverage,
        warnings=(["Ignored native facies remain unquantified; nominal-mass fitting uses physical mass intervals."
                   ] if any(value["frames_with_unknown"] for value in unknown_coverage.values()) else []),
        predictions=output, calibration_reference="nominal injected mass; no inferred loss subtraction",
        validation_scope="calibration residuals, not independent concentration validation",
        identifiable_local_concentration=False,
        parameter_meaning="global scales of native relative aqueous concentration and native gas saturation; clipped to [0,1]",
        fitting_reduction=dict(method="Native geometry-weighted joint response histogram",
            intervals=SUMMARY_INTERVALS, fields_are_quantized=False,
            final_exact_native_evaluation_required=True),
        native_calibrations=native_provenance_by_run)


def fit_calibration(runs: Mapping[str, tuple[NativeConcentrationAnalysis, Sequence[PreparedNativeFrame]]],
                    *, scope="individual", overflow_mode="off", references=(),
                    unknown_policy="complete_frames"):
    """Convenience wrapper, accepting lazy frame sequences; no optimizer rereads.

    Multi-run workers should call ``fit_summaries`` after preparing one run at a
    time, then evaluate native frames individually using its saved parameters.
    """
    summaries = {run: [analysis.summarize_frame(frame) for frame in frames]
                 for run, (analysis, frames) in runs.items()}
    provenance = {run: analysis.provenance() for run, (analysis, _) in runs.items()}
    output = fit_summaries(summaries, provenance, scope=scope,
                          overflow_mode=overflow_mode, references=references,
                          unknown_policy=unknown_policy)
    parameters = CalibrationParameters(**output["parameters"])
    exact = {run: [analysis.evaluate(frame, parameters, fields=False) for frame in frames]
             for run, (analysis, frames) in runs.items()}
    output["summary_predictions"] = output["predictions"]
    output["predictions"] = exact
    output["fitting_reduction"]["final_exact_native_evaluation_required"] = False
    output["fitting_reduction"]["max_observed_total_mass_error_g"] = max(
        abs(exact[run][i]["total_mass_g"] - output["summary_predictions"][run][i]["total_mass_g"])
        for run, frames in summaries.items() for i in range(len(frames)))
    for comparison in output["overflow_comparisons"]:
        comparison["predicted_mass_g"] = exact[comparison["run"]][int(comparison["frame_index"])][comparison["quantity"]]
    return output
