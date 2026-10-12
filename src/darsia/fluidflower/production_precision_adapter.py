"""Unqualified guarded production precision extension; scratch preparation.

Original source modules/classes are never mutated. Original selected loader,
runner, metadata/source/freeze/admission checks and frame validation code run
with call-local namespaces. Only the documented pixel operations/working dtype
are substituted. This is a NEW scientific implementation, not an IO-only source
substitution or an R3 implementation. Root must bind/review its actual source
and qualify actual products before profile/equivalence/production use.

The context must already come from the original guarded context preparation;
this module does not initialize jobs, acquire permission, fit, select inputs,
publish, restart, or infer a calibration/precision/backend.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import FunctionType


def _clone(function, **overrides):
    cloned = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                          function.__name__, function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__annotations__ = function.__annotations__
    cloned.__module__ = function.__module__
    cloned.__qualname__ = function.__qualname__
    return cloned


class _Proxy:
    def __init__(self, original, **overrides):
        self.original, self.overrides = original, overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


class _Reader:
    """Delegate every guarded byte read; isolate numerical baseline caches."""
    def __init__(self, original, asset_reader=None):
        if (asset_reader is not None and original is not asset_reader
                and getattr(original, "asset_reader", None) is not asset_reader):
            raise ValueError("Precision cache must delegate the same source-bound asset reader")
        self.original = original
        self.asset_reader = asset_reader
        self.raw_baselines = {}

    def __getattr__(self, name):
        return getattr(self.original, name)


def _working_frame_class(original, math):
    """All original dataclass rules execute; only expected working RGB dtype varies.

    No validation is skipped or performed on an artificial downcast copy.
    The inherited constructor calls this post-init on the actual working arrays.
    Original storage encoding validation happens separately before decoding.
    """
    import numpy as np
    original_validate = _clone(original.__post_init__, np=_Proxy(np, float32=math.dtype))
    class WorkingFrame(original):
        def __post_init__(self):
            for name in ("current_rgb", "baseline_rgb", "clear_water_rgb"):
                math.require_pixels(getattr(self, name))
            original_validate(self)
    return WorkingFrame


def _to_native_frame(frame, WorkingFrame, *, l1_neighbourhood_mask=None):
    """Original source.to_native_frame mapping, with the typed full-rule class."""
    import numpy as np
    known = frame.domain & ~frame.unknown_physical_mask
    bottom = frame.bottom_strip_mask & known
    below = frame.below_photo_filled_mask & known & ~bottom
    return WorkingFrame(run_id=frame.run, frame_index=frame.frame_index,
        current_rgb=frame.current_rgb, baseline_rgb=frame.baseline_rgb,
        clear_water_rgb=frame.clear_water_rgb,
        clear_water_supported_mask=frame.clear_water_supported_mask,
        phase_labels=frame.phase_labels, facies=frame.facies, domain=frame.domain,
        supported_mask=frame.supported_mask, mass_eligible_mask=frame.mass_eligible_mask,
        depth_m=frame.depth_m, porosity=frame.porosity,
        pixel_area_m2=frame.pixel_area_m2, dimensions_m=frame.dimensions_m,
        pressure_bar=frame.atmosphere["pressure_bar"], temperature_c=frame.atmosphere["temperature_c"],
        pressure_gradient_bar_per_m=frame.atmosphere.get("pressure_gradient_bar_per_m", 0.),
        temperature_gradient_c_per_m=frame.atmosphere.get("temperature_gradient_c_per_m", 0.),
        elapsed_h=frame.elapsed_h, nominal_injected_mass_g=frame.nominal_injected_mass_g,
        recipe=frame.recipe, provenance=dict(frame.provenance,
            unknown_physical_domain_cm2=float(np.sum(frame.pixel_area_m2[frame.unknown_physical_mask], dtype=np.float64)*1e4),
            unknown_physical_co2_cm2=float(np.sum(frame.pixel_area_m2[frame.unknown_physical_mask & np.isin(frame.phase_labels, (1, 2))], dtype=np.float64)*1e4),
            unknown_physical_mass_g=None),
        excluded_masks={"bottom_strip": bottom, "below_photo_filled": below},
        l1_neighbourhood_mask=l1_neighbourhood_mask,
        unknown_physical_mask=frame.unknown_physical_mask)


def _compose_scientific_raw(source, descriptor, reader, math):
    """Original scientific assets/shape/bounds/storage/support checks, typed exp."""
    import numpy as np
    source._scientific_assets(descriptor)
    shape = tuple(descriptor["phase_shape_hw"])
    centre = descriptor["native_centre"]
    hc, wc = tuple(centre["native_shape_hw"])
    ox, oy = descriptor["native_origin_xy"]
    x, y = -int(ox), -int(oy)
    if min(x, y) < 0 or x+wc > shape[1] or y+hc > shape[0]:
        raise ValueError("Scientific RAW centre lies outside the phase canvas")
    rgb = np.zeros((*shape, 3), math.dtype)
    valid = np.zeros(shape, bool)
    covered = np.zeros(shape, bool)
    log = reader.array(centre["log_linear_rgb"])
    mask = reader.array(centre["valid_mask"])
    if log.dtype != np.float32 or log.shape != (3, hc, wc) or mask.dtype != bool or mask.shape != (hc, wc):
        raise ValueError("Scientific centre encoding is not exact float32 log RGB")
    def place(values, box, support=None):
        a, b, c, d = map(int, box)
        if (not 0 <= a < c <= shape[1] or not 0 <= b < d <= shape[0]
                or covered[b:d, a:c].any() or values.dtype != np.float32
                or values.shape != (3, d-b, c-a)):
            raise ValueError("Scientific windows overlap or use another native grid")
        good = np.isfinite(values).all(0)
        if support is not None:
            good &= support
        for channel in range(3):
            target = rgb[b:d, a:c, channel]
            target[good] = math.decode_log_rgb(values[channel][good])
        valid[b:d, a:c] = good
        covered[b:d, a:c] = True
    place(log, (x, y, x+wc, y+hc), mask)
    for window in descriptor["ring_windows"]:
        place(reader.array(window["log_linear_rgb"]), window["canvas_xyxy"])
    if not np.isfinite(rgb[valid]).all() or np.any(rgb[valid] < 0):
        raise ValueError("Scientific RAW overflows; no clipping is permitted")
    return rgb, valid


def _transport_physical_reference(reference, matrix, shape_hw, math, cv2):
    """Original shape/categorical/physical support checks; typed continuous warp."""
    import numpy as np
    h, w = map(int, shape_hw)
    source_domain = np.asarray(reference["domain"])
    if source_domain.ndim != 2 or source_domain.dtype != bool:
        raise ValueError("Physical reference needs exact boolean domain")
    result = {}
    for name in ("facies", "domain", "porosity", "depth_m"):
        values = np.asarray(reference[name])
        if values.shape != source_domain.shape:
            raise ValueError("Physical reference fields use different geometries")
        categorical = name in ("facies", "domain")
        if name == "facies" and not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Facies is categorical")
        if categorical:
            transformed = cv2.warpPerspective(values.astype(np.uint8), np.asarray(matrix, np.float64),
                (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            transformed = math.warp_continuous(values, matrix, (h, w), cv2_module=cv2)
        result[name] = transformed.astype(bool) if name == "domain" else transformed
    coverage = math.warp_continuous(source_domain, matrix, (h, w), cv2_module=cv2)
    valid = result["domain"] & (coverage >= 1-1e-6)
    valid &= np.isfinite(result["depth_m"]) & (result["depth_m"] > 0)
    valid &= np.isfinite(result["porosity"]) & (result["porosity"] > 0) & (result["porosity"] <= 1)
    result["physical_supported_mask"] = valid
    result["unknown_physical_mask"] = ~valid
    return result


def _registered_clearwater(source, clear_water_reference, matrix_ac28_to_frame,
                          shape_hw, reader, math, cv2, *, expected_run=None,
                          expected_native_manifest_sha256=None,
                          expected_photometry_sha256=None):
    """Original frozen clearwater metadata/storage/registration/support guards."""
    import numpy as np
    receipt_descriptor = clear_water_reference["receipt"]
    receipt = reader.json(receipt_descriptor)
    if receipt.get("encoding") != "float32-camera-linear-unit-rgb":
        raise ValueError("Clearwater must use the frozen scientific RAW encoding")
    if expected_run is not None and receipt.get("run") != expected_run:
        raise ValueError("Clearwater scientific receipt belongs to another run")
    photometry_equal = (expected_photometry_sha256 is not None
                       and receipt.get("v5_photometry_sha256") == expected_photometry_sha256)
    if expected_photometry_sha256 is not None and not photometry_equal:
        raise ValueError("Clearwater and production RAW have different frozen photometry bindings")
    if expected_native_manifest_sha256 is not None and receipt.get("native_manifest_sha256") != expected_native_manifest_sha256 and not photometry_equal:
        raise ValueError("Clearwater and production RAW have different frozen photometry bindings")
    if "H_ac28_to_clearwater_grid" not in clear_water_reference:
        raise ValueError("Clearwater requires explicit source-bound native-B geometry")
    stored = reader.array(dict(path=receipt["rgb_path"], sha256=receipt["rgb_sha256"]))
    valid = reader.array(dict(path=receipt["valid_mask_path"], sha256=receipt["valid_mask_sha256"]))
    if stored.dtype != np.float32 or stored.shape != (*valid.shape, 3) or valid.dtype != bool:
        raise ValueError("Scientific clearwater shape/dtype differs from its native grid")
    source_valid = valid & np.isfinite(stored).all(-1) & (stored > 0).all(-1)
    matrix = np.asarray(clear_water_reference["H_ac28_to_clearwater_grid"], np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Clearwater native-B registration is invalid")
    warp = np.asarray(matrix_ac28_to_frame, np.float64) @ np.linalg.inv(matrix)
    h, w = map(int, shape_hw)
    values = math.warp_continuous(np.where(source_valid[..., None], stored, 0), warp,
                                  (h, w), cv2_module=cv2)
    coverage = math.warp_continuous(source_valid, warp, (h, w), cv2_module=cv2)
    supported = (coverage >= 1-1e-6) & np.isfinite(values).all(-1) & (values > 0).all(-1)
    return values, supported, dict(receipt=receipt_descriptor, native_B_to_current=warp.tolist(),
        unsupported_not_extrapolated=True, source_bound_geometry=clear_water_reference.get("geometry_binding"),
        production_native_manifest_sha256=expected_native_manifest_sha256,
        reference_native_manifest_sha256=receipt.get("native_manifest_sha256"),
        exact_photometry_sha256=expected_photometry_sha256, alias_metadata_equivalence=photometry_equal)


def _analysis_classes(r1, math):
    """New per-instance numerical classes; original ctor/model/source checks run."""
    import numpy as np
    darsia = r1.darsia
    class TypedColor:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "original"), name)

        def __call__(self, colors):
            math.require_pixels(colors)
            original = self.original
            path = original.color_path
            supports = path.colors if original.color_mode == darsia.ColorMode.ABSOLUTE else path.relative_colors
            if original.ignore_spectrum is None or colors.ndim == 1:
                parameter = math.fit_color_path(colors, supports, path.equidistant_distances)
            else:
                # Exact original branch/threshold; ignored pixels are NOT relabelled.
                mask = np.linalg.norm(colors, axis=1) > 1e-1
                parameter = np.zeros(colors.shape[0], dtype=math.dtype)
                parameter[mask] = math.fit_color_path(colors[mask], supports, path.equidistant_distances)
                sub_colors = colors[mask]
                sub_parameter = math.fit_color_path(sub_colors, supports, path.equidistant_distances)
                parameter[mask] = sub_parameter
            return math.interpret_color_path(parameter, path.equidistant_distances, original.values)

    class ColorLoader:
        @staticmethod
        def load(path):
            return TypedColor(darsia.ColorPathInterpolation.load(path))

    class TypedSignal:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "original"), name)

        def __call__(self, values):
            return math.piecewise_linear(values, self.original.supports, self.original.values,
                                         endpoint="error")

    class SignalLoader:
        @staticmethod
        def load(path):
            return TypedSignal(darsia.PWTransformation.load(path))

    class TypedHeterogeneous(darsia.HeterogeneousModel):
        def __init__(self, obj, labels, ignore_labels=None):
            super().__init__(obj, labels, ignore_labels=ignore_labels)
            self._working_labels = labels.img

        def __call__(self, signal):
            return math.per_facies(signal, self._working_labels, self.obj,
                                   ignore_labels=self.ignore_labels)

    class TypedClip(darsia.ClipModel):
        def __call__(self, image):
            lower = None if self._min_value is None else math.dtype(self._min_value)
            upper = None if self._max_value is None else math.dtype(self._max_value)
            if isinstance(image, np.ndarray):
                math.require_pixels(image)
                return np.clip(image, lower, upper)
            if isinstance(image, darsia.Image):
                math.require_pixels(image.img)
                result = image.copy()
                result.img = np.clip(result.img, lower, upper)
                return result
            raise ValueError("invalid input type")

    class TypedPipeline(r1.HeterogeneousColorToMassAnalysis):
        __init__ = _clone(r1.HeterogeneousColorToMassAnalysis.__init__,
                         darsia=_Proxy(darsia, HeterogeneousModel=TypedHeterogeneous,
                                       ClipModel=TypedClip))

    class TypedPhysics(darsia.CO2MassAnalysis):
        def atmospheric_temperature_map(self):
            height = math.pixels(self.height_map)
            return np.ones_like(height)*math.dtype(self.atmospheric_temperature) + (
                math.dtype(self.atmospheric_temperature_gradient)*height)

        def setup_density_gaseous_co2(self):
            # Original frozen small tables and interp1d objects remain float64.
            interpolate = darsia.CO2MassAnalysis.setup_density_gaseous_co2.__globals__["interpolate"]
            self.co2_density_interpolator_20 = interpolate.interp1d(*self.data_NIST_20, kind="linear")
            self.co2_density_interpolator_23 = interpolate.interp1d(*self.data_NIST_23, kind="linear")
            self.co2_solubility_interpolator_20 = interpolate.interp1d(*self.solubility_co2_20, kind="linear")
            self.co2_solubility_interpolator_23 = interpolate.interp1d(*self.solubility_co2_23, kind="linear")
            result = math.hydrostatic_fields(temperature=self.atmospheric_temperature_map(),
                height=self.height_map, top_pressure=self.top_atmospheric_pressure(),
                water_density_20=self.water_density_20, water_density_23=self.water_density_23,
                density_20=self.data_NIST_20, density_23=self.data_NIST_23,
                solubility_20=self.solubility_co2_20, solubility_23=self.solubility_co2_23)
            self.density_gaseous_co2, self.solubility_co2 = result["density"], result["solubility"]

    physics_namespace = _Proxy(darsia, CO2MassAnalysis=TypedPhysics)
    constructor_namespace = _Proxy(darsia, ColorPathInterpolation=ColorLoader, PWTransformation=SignalLoader)
    class TypedAnalysis(r1.UnclippedNativeConcentrationAnalysis):
        # Geometry weights/depth/porosity stay float64 in BOTH pixel modes.
        __init__ = _clone(r1.UnclippedNativeConcentrationAnalysis.__init__,
            np=_Proxy(np, float32=np.float64), darsia=constructor_namespace,
            HeterogeneousColorToMassAnalysis=TypedPipeline)

        def _rgb(self, rgb):
            values = math.require_pixels(rgb)
            if values.shape != (*self.shape, 3) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Unclipped finite working-precision scientific RAW is required")
            return darsia.OpticalImage(values.copy(), color_space="RGB", dimensions=list(self.dimensions))

        _scalar = _clone(r1.NativeConcentrationAnalysis._scalar, np=_Proxy(np, float32=math.dtype))
        _mass_physics = _clone(r1.NativeConcentrationAnalysis._mass_physics,
                              np=_Proxy(np, float32=math.dtype), darsia=physics_namespace)

    class TypedFlash(darsia.TitrationFlash):
        def __call__(self, signal):
            aqueous, gas = math.titration(signal.img, self)
            return (darsia.full_like(signal, aqueous, dtype=math.dtype),
                    darsia.full_like(signal, gas, dtype=math.dtype))
    return TypedAnalysis, TypedFlash


class PrecisionAdapter:
    """Whole guarded selected-load + retained-N1 route, explicit math provider.

    Public ABI matches retained.load_response/evaluate_main/run_case. Providers
    operate on host NumPy buffers with one declared dtype; a CUDA provider must
    explicitly report its backend and transfers (never fallback). Native object
    fields/frame/analysis/prepared/physics/flash retain their original ABI.
    """
    def __init__(self, context, precision, *, pixel_math=None, retained_module=None):
        import numpy as np
        if precision not in ("float32", "float64"):
            raise ValueError("Explicit native working precision required")
        if pixel_math is None:
            # Relative import when merged; direct sibling import for scratch.
            try:
                from .production_precision import PixelMath
            except ImportError:
                from production_precision import PixelMath
            pixel_math = PixelMath(precision)
        if pixel_math.precision != precision or pixel_math.dtype != getattr(np, precision):
            raise ValueError("Requested precision and mathematical implementation differ")
        required = ("pixels", "require_pixels", "decode_log_rgb", "warp_continuous",
            "reference_normalized", "relative_rgb", "per_facies", "fit_color_path",
            "interpret_color_path", "piecewise_linear", "titration", "hydrostatic_fields",
            "selected_n1_fields", "integrate", "facts")
        if any(not callable(getattr(pixel_math, name, None)) for name in required):
            raise TypeError("Complete declared precision mathematical provider required")
        backend_facts = pixel_math.facts()
        if (backend_facts.get("precision") != precision
                or backend_facts.get("backend") not in {"numpy_cpu", "cuda", "opencl"}):
            raise ValueError("Exact working precision/backend facts required; no implicit fallback")
        if retained_module is None:
            from darsia.fluidflower import production_retained as retained_module
        self.context, self.precision, self.math, self.retained = context, precision, pixel_math, retained_module
        self.backend_facts = backend_facts
        self.reader = _Reader(context.reader, context.asset_reader)
        self._WorkingFrame = _working_frame_class(context.modules["engine"].r1.CommonFrameArrays, pixel_math)
        self._TypedAnalysis, self._TypedFlash = _analysis_classes(context.modules["engine"].r1, pixel_math)

    def _facts(self):
        facts = self.math.facts()
        if (facts.get("precision") != self.precision
                or facts.get("backend") != self.backend_facts["backend"]):
            raise ValueError("Pixel implementation changed precision/backend; implicit fallback forbidden")
        return dict(facts)

    def _prepare(self, frame, path_folder, parts, swi, scope, frozen_calibration, *, asset_reader=None):
        import numpy as np
        context, math = self.context, self.math
        if scope != "frozen_evaluation" or frozen_calibration is None:
            raise ValueError("Production requires the actual chosen freeze before pixels")
        parameters, digest = self.retained._selected_n1(context.modules, self._selection, asset_reader)
        r1, r2 = context.modules["engine"].r1, context.modules["engine"].r2
        kwargs = {} if asset_reader is None else {"asset_reader": asset_reader}
        original, original_digest = r2.verify_original_round1_freeze(parts["r1_freeze"]["path"], **kwargs)
        if Path(original["path_folder"]).resolve() != Path(path_folder).resolve():
            raise ValueError("Production must reuse the unchanged round-1 pooled colour paths")
        def normalized(frame):
            clear = frame.clear_water_rgb
            support = frame.known_physical_domain & frame.supported_mask & frame.clear_water_supported_mask
            support &= np.isfinite(clear).all(axis=2) & (clear > 0).all(axis=2)
            current, baseline = math.reference_normalized(frame.current_rgb, frame.baseline_rgb, clear, support)
            if not np.isfinite(current[support]).all() or not np.isfinite(baseline[support]).all():
                raise ValueError("Clear-water normalization produced nonfinite scientific signal")
            return current, baseline, support
        prepare = _clone(r1.prepare_native_response,
            np=_Proxy(np, float32=math.dtype), reference_normalized_rgb=normalized,
            UnclippedNativeConcentrationAnalysis=self._TypedAnalysis,
            darsia=_Proxy(r1.darsia, TitrationFlash=self._TypedFlash),
            verify_frozen_calibration=r2.verify_original_round1_freeze)
        response = prepare(frame, path_folder, flash_kind="titration",
            frozen_calibration_path=parts["r1_freeze"]["path"] if r1.normalize_run(frame.run_id) in r2.HELDOUT_RUNS else None,
            **kwargs)
        measured_swi = np.full(frame.domain.shape, np.nan, math.dtype)
        for label, value in swi.items():
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Swi must be a measured sand property in [0,1]")
            measured_swi[frame.facies == int(label)] = math.dtype(value)
        response.update(swi=measured_swi, frozen_parameters=parameters,
            round1_freeze_sha256=original_digest, selected_calibration_sha256=digest,
            cross_host_source_equivalence=original["cross_host_source_equivalence"],
            precision=self.precision, pixel_backend=self.backend_facts["backend"],
            pixel_backend_facts=self._facts(),
            input_storage_dtype="float32-log-RGB/frozen-clearwater",
            working_pixel_dtype=self.precision, parameter_dtype="float64", reduction_dtype="float64",
            scientific_eligibility=False, production_permission_granted=False)
        if asset_reader is not None:
            response["production_io_source_provenance"] = original["production_io_source_provenance"]
        return response

    def load_response(self, case_id, *, selected_calibration):
        import numpy as np
        context, math = self.context, self.math
        if case_id not in context.cases:
            raise ValueError("Case is outside the locked native diagnostic")
        if context.boundary_guard is not None:
            context.boundary_guard()
        # Choice validation is retained; no B0 selection shortcut or inferred B1.
        self.retained._selected_n1(context.modules, selected_calibration, context.asset_reader)
        self._selection = selected_calibration
        adapter, source = context.modules["adapter"], context.modules["source"]
        cv2 = adapter.cv2
        def compose(descriptor, reader):
            return _compose_scientific_raw(source, descriptor, reader, math)
        def clearwater(reference, matrix, shape, reader, **kwargs):
            return _registered_clearwater(source, reference, matrix, shape, reader, math, cv2, **kwargs)
        def transport(reference, matrix, shape):
            return _transport_physical_reference(reference, matrix, shape, math, cv2)
        def to_frame(frame, **kwargs):
            return _to_native_frame(frame, self._WorkingFrame, **kwargs)
        def continuous_warp(values, matrix, dsize, *, flags, borderMode, borderValue):
            if flags != cv2.INTER_LINEAR or borderMode != cv2.BORDER_CONSTANT or borderValue != 0:
                raise ValueError("Selected loader continuous warp differs from its frozen method")
            return math.warp_continuous(values, matrix, (dsize[1], dsize[0]), cv2_module=cv2)
        source_view = _Proxy(source, compose_scientific_raw=compose, registered_clearwater=clearwater,
                             transport_physical_reference=transport, to_native_frame=to_frame)
        # The sole np.float32 use in original load_selected is coverage warp.
        # Its metadata/admission/categorical source code and all checks are exact.
        selected = _clone(adapter.load_selected, source=source_view, np=_Proxy(np, float32=math.dtype),
                          cv2=_Proxy(cv2, warpPerspective=continuous_warp),
                          _prepare_response=self._prepare)
        def loader(**kwargs):
            return selected(**kwargs, **({} if context.asset_reader is None else {"asset_reader": context.asset_reader}))
        session = dict(context.session, loader=loader, reader=self.reader)
        load = _clone(context.load_entry, SESSION=session)
        result = load(context.cases[case_id]["entry"], scope="frozen_evaluation",
                      frozen_calibration=context.configuration["actual_R3_freeze"]["path"])
        for name in ("current_rgb", "baseline_rgb", "clear_water_rgb"):
            math.require_pixels(getattr(result["response"]["frame"], name))
        if context.boundary_guard is not None:
            context.boundary_guard()
        return result

    def evaluate_main(self, response, anchor, *, selected_calibration, domain_masks, kept_fields=()):
        import numpy as np
        context, math, retained = self.context, self.math, self.retained
        if response.get("precision") != self.precision:
            raise ValueError("Response did not use this explicit working precision")
        if set(kept_fields)-retained.FIELDS or set(domain_masks) != {"full_rig", "box1", "box2"}:
            raise ValueError("Supported retained fields and all three frozen domain masks are required")
        parameters, digest = retained._selected_n1(context.modules, selected_calibration, context.asset_reader)
        if response.get("frozen_parameters") != parameters:
            raise ValueError("Response differs from the one frozen production calibration")
        engine, frame = context.modules["engine"], response["frame"]
        for name, mask in domain_masks.items():
            if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != frame.domain.shape:
                raise ValueError("Frozen domain mask differs from the native full-resolution grid")
            if name == "full_rig" and not np.array_equal(mask, frame.domain):
                raise ValueError("Full-rig domain must be the exact original frozen phase domain")
        analysis, physics = response["analysis"], response["physics"]
        partition = retained._main_partition(response)  # original finite/positive physics guards
        factor = engine.r2.dilution_factor(frame.elapsed_h, partition["affected_pore_volume_m3"],
                                          anchor, parameters.get("beta", 0.))
        maps = math.selected_n1_fields(response["prepared"].aqueous_relative,
            gain=parameters["gain"], factor=factor, swi=response["swi"],
            aqueous_known=partition["aq_known"], gas_known=partition["gas_known"],
            solubility=physics.solubility_co2, gas_density=physics.density_gaseous_co2)
        eligible = frame.mass_eligible_mask & frame.known_physical_domain
        def integrate(value, mask):
            # Original Geometry.integrate itself remains unchanged; its weights
            # are genuinely float64 before multiply and repeated axis-zero sums.
            analysis.geometry._prepare_cached_voxel_volume(list(value.shape[:analysis.geometry.space_dim]))
            if np.asarray(analysis.geometry.cached_voxel_volume).dtype != np.float64:
                raise TypeError("Geometry weights must be float64 before reduction")
            return float(analysis.geometry.integrate(np.where(mask, value, 0.)))*1000
        def masses(mask):
            return dict(total_mass_g=integrate(maps["mass_total"], mask),
                        aqueous_mass_g=integrate(maps["mass_aqueous"], mask),
                        gas_mass_g=integrate(maps["mass_gas"], mask))
        domains = {name: masses(eligible & mask) for name, mask in domain_masks.items()}
        excluded, covered = {}, np.zeros(frame.domain.shape, bool)
        for name, mask in sorted((frame.excluded_masks or {}).items()):
            covered |= mask
            excluded[name] = masses(mask)
        excluded["other_not_mass_eligible"] = masses(frame.domain & ~frame.mass_eligible_mask & ~covered)
        fields = {}
        if "c_aq_g_l" in kept_fields or "unquantified_phase_mask" in kept_fields or "s_g" in kept_fields:
            unknown = frame.domain & np.isin(frame.phase_labels, (1, 2)) & ~(
                partition["aq_known"] | partition["gas_known"])
            if "c_aq_g_l" in kept_fields:
                concentration = maps["concentration"]
                concentration[unknown] = np.nan
                fields["c_aq_g_l"] = concentration
            if "unquantified_phase_mask" in kept_fields:
                fields["unquantified_phase_mask"] = unknown
            if "s_g" in kept_fields:
                gas = maps["gas"]
                gas[unknown & (frame.phase_labels == 2)] = np.nan
                fields["s_g"] = gas
        if "c_aq_relative" in kept_fields: fields["c_aq_relative"] = maps["aqueous"]
        if "phase_labels" in kept_fields: fields["phase_labels"] = frame.phase_labels.copy()
        if "main_mass_eligible_mask" in kept_fields: fields["main_mass_eligible_mask"] = eligible
        for name, value in (("mass_density_g_l", "mass_total"),
                            ("aqueous_mass_density_g_l", "mass_aqueous"),
                            ("gas_mass_density_g_l", "mass_gas")):
            if name in kept_fields: fields[name] = maps[value]
        return dict(contract="darsia-production-retained-main-precision-draft-v1", parameters=parameters,
            calibration_sha256=digest, scalar_reference="native_final_calibration_fields", run=frame.run_id,
            frame_index=frame.frame_index, elapsed_h=frame.elapsed_h, dilution_factor=factor,
            **domains["full_rig"], domains=domains, excluded_mass=excluded,
            affected_pore_volume_m3=partition["affected_pore_volume_m3"], fields=fields,
            precision=self.precision, pixel_backend=response["pixel_backend"],
            pixel_backend_facts=self._facts(),
            input_storage_dtype=response["input_storage_dtype"], working_pixel_dtype=self.precision,
            parameter_dtype="float64", reduction_dtype="float64",
            scientific_eligibility=False, production_permission_granted=False)

    def run_case(self, case_id, *, selected_calibration, anchor, domain_masks, kept_fields):
        loaded = self.load_response(case_id, selected_calibration=selected_calibration)
        guard = self.context.boundary_guard
        if guard is not None:
            guard()
        result = self.evaluate_main(loaded["response"], anchor, selected_calibration=selected_calibration,
                                    domain_masks=domain_masks, kept_fields=kept_fields)
        if guard is not None:
            guard()
        return result

    def main_products(self, result):
        """Original retained export/mask/units, with truthful chosen mass dtype."""
        import numpy as np
        if result.get("precision") != self.precision:
            raise ValueError("Products differ from the declared working precision")
        fields = result["fields"]
        required = {"c_aq_g_l", "s_g", "phase_labels", "main_mass_eligible_mask",
                    "mass_density_g_l", "aqueous_mass_density_g_l", "gas_mass_density_g_l"}
        if not required.issubset(fields):
            raise ValueError("Every retained native §3 product and the main mask must be present")
        mask = fields["main_mass_eligible_mask"]
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise ValueError("Main product mask must retain the original boolean grid")
        products = dict(concentration_aqueous=fields["c_aq_g_l"], saturation_free_gas=fields["s_g"],
                        phase_labels=fields["phase_labels"])
        for name, native in (("mass_total", "mass_density_g_l"),
                             ("mass_aqueous", "aqueous_mass_density_g_l"),
                             ("mass_free_gas", "gas_mass_density_g_l")):
            value = fields[native]
            if value.dtype != self.math.dtype or value.shape != mask.shape:
                raise ValueError("Retained mass map must keep chosen precision and exact grid")
            products[name] = np.where(mask, value, 0.)
        return dict(products=products, units={"concentration_aqueous": "g/L", "saturation_free_gas": "1",
            "mass_total": "g/L", "mass_aqueous": "g/L", "mass_free_gas": "g/L", "phase_labels": "category"},
            required_transport_weight_units="L pore volume per pixel", native_mass_scalar_units="g",
            main_mask_applied=True, raw_native_fields_retained=True, precision=self.precision,
            pixel_backend=result["pixel_backend"], scientific_eligibility=False, production_permission_granted=False)
