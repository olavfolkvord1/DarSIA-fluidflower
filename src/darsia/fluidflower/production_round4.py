"""FF_AC v4: corrected physical support, N1 refit, fixed checks and products.

The original selected loader, pooled colour paths, phase maps, dose inputs,
two-stage fitter and check implementation are reused. This module does not
qualify a backend, select a precision, or start a worker on its own.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType, SimpleNamespace
import json

SPEC_SHA256 = "bf69e427d39e4b351dbdeb2f4e3067a6724f51f6b4cc84587ba41e959e3d9143"
U0_RULE_SHA256 = "40c503b7e311d23cb398283eb79e5102566a800532538879f0a44f0ef2cf0721"
AMENDMENT_SHA256 = "f1567b3f8724b41c6401e8d72d139c45910cd4a6aeba09dba75436da2a0882d0"
PRIORITY_RUNS = ("ac19", "ac28", "ac47", "ac52", "ac56")
LATE_TARGETS = (56., 64., 72., 96.)


def clone(function, **overrides):
    result = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class Proxy:
    def __init__(self, original, **overrides):
        self.original, self.overrides = original, overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


def selected_scope(validated, *, engine):
    """Exact indexed acquisitions; metadata only, no phase/RAW array decoding."""
    entries = validated["entries"]
    expected = set(engine.FIT_RUNS) | set(engine.HELDOUT_RUNS) | set(engine.DESCRIPTIVE_RUNS)
    if {row["run"] for row in entries} != expected or len(expected) != 39:
        raise ValueError("Round 4 requires the original selected 39-run inventory")
    by_run = {run: sorted((row for row in entries if row["run"] == run),
                         key=lambda row: (row["elapsed_h"], row["frame_index"]))
              for run in sorted(expected)}
    scopes = {}
    for run, rows in by_run.items():
        anchors = [row for row in rows if float(row["selected_target_h"]) == 2.25]
        if len(anchors) != 1:
            raise ValueError(f"Exactly one selected 2.25 h acquisition is required for {run}")
        scopes[run] = {f"{run}_f{row['frame_index']}": {
            "entry": row, "anchor_entry": anchors[0]} for row in rows}
    return scopes


def prepare_context(configuration, reader, cases, *, native_module, boundary_guard):
    """Original full admission/setup with an explicit whole-run selected scope."""
    if not callable(boundary_guard) or not cases:
        raise ValueError("Actual live guard and exact selected cases are required")
    if len({row["entry"]["run"] for row in cases.values()}) != 1:
        raise ValueError("A worker owns one whole run")
    def config(value):
        if value != configuration:
            raise ValueError("Round 4 configuration changed")
        return value
    def scope(value):
        if value != reader.json(configuration["profile_selection"]):
            raise ValueError("Selected source inventory changed")
        return cases
    def candidate(value, asset_reader):
        if (value != configuration or asset_reader.source_qualification_state != "round4_locked_sources"
                or native_module._execution_identity(asset_reader) != asset_reader.execution_identity):
            raise ValueError("Round 4 source/commit identity changed")
        return asset_reader
    prepare = clone(native_module.prepare_native_context, _configuration=config, _cases=scope,
                    _candidate_reader=candidate)
    context = prepare(configuration, reader, boundary_guard=boundary_guard)
    context.provenance.update(round4_spec_sha256=SPEC_SHA256, whole_run=True,
                              original_four_proof_claimed=False)
    return context


def make_reader_class(base, stream_module, lock_descriptor, *, io):
    """Existing manifest/staging reader with the actual v4 code-lock admission."""
    class LockedReader(base):
        def _qualify_sources(self, manifest, qualification):
            if qualification != {"round4_lock": lock_descriptor}:
                raise ValueError("Actual round-4 lock binding required")
            lock = io.read_json(lock_descriptor, max_bytes=lock_descriptor["size_bytes"])
            setting = manifest["settings"]
            backend = setting["backend"]
            key = setting["host"]+"/"+backend+"/"+setting["precision"]
            if (lock.get("round4_spec_sha256") != SPEC_SHA256
                    or lock.get("commit") != self.code_commit
                    or lock.get("precision") != setting["precision"]
                    or key not in lock.get("qualified_resources", {})
                    or lock["qualified_resources"][key].get("passed") is not True
                    or not self._same_source_bindings(lock["source_bindings_by_host"][setting["host"]], self.source_bindings)):
                raise ValueError("Unqualified precision/backend/host or changed original source mapping")
            for row in self.source_bindings:
                self.io.verify(row["current_source"])
                self.path(row["original_source"])
            for descriptor in lock["scientific_sources_by_host"][setting["host"]].values():
                self.io.verify(descriptor)
            self._assurance_locators = {self._canonical_key(manifest["inputs"][role]["path"])
                                       for role in manifest["assurance_input_roles"]}
            self.source_qualification_state = "round4_locked_sources"
            self.source_qualification = qualification
    staged_class = stream_module.stream_reader_class(LockedReader)
    class PrefetchedLockedReader(staged_class):
        def array(self, descriptor):
            cache = getattr(self, "_prefetched", {})
            logical = self.logical_descriptor(descriptor)
            bound = self._stage_logical(descriptor)
            key = self._canonical_key(bound["path"])
            if key not in cache:
                return super().array(descriptor)
            if ((logical["sha256"], logical["size_bytes"]) in self._opaque_contents
                    or self._canonical_key(logical["path"]) in self._assurance_locators
                    or self._frame_stack is None):
                raise ValueError("Prefetch requires guarded numerical frame inputs")
            import numpy as np
            value, contract = cache[key], self._array_contracts[key]
            self.path(bound)
            if (tuple(value.shape) != contract["shape"] or value.dtype != np.dtype(contract["dtype"])
                    or value.flags.writeable or value.dtype.hasobject):
                raise ValueError("Prefetched array differs from the immutable NPY contract")
            return value
    return PrefetchedLockedReader


def anchor_from_row(row):
    if (float(row["selected_target_h"]) != 2.25 or row["affected_pore_volume_m3"] <= 0
            or abs(row["elapsed_h"]-2.247) > .13):
        raise ValueError("Original injection-end anchor is unavailable")
    return {key: row[key] for key in ("elapsed_h", "affected_pore_volume_m3", "frame_index", "source_binding")}


def anchor_from_response(context, response, entry):
    if float(entry["selected_target_h"]) != 2.25:
        raise ValueError("Only the exact selected 2.25 h source supplies the dilution anchor")
    return context.modules["engine"].r2.anchor_from_response(response)


def measure_boundaries(sources, *, sandmap, output, io, boundary_guard, started_monotonic):
    """Measure the original sand/photo edges in AC28 and the other camera group.

    Each source supplies an already admitted context/adapter and a fit-run
    acquisition. Only this map step uses the baseline photograph; no N1 fit or
    heldout calculation occurs here. A failed single correction keeps the map.
    """
    import time, gc
    import numpy as np
    from scipy.ndimage import binary_erosion
    measurements, details = [], []
    if {value["run"] for value in sources} != {"ac28", "ac39"}:
        raise ValueError("AC28 and fit-run AC39 cover the two original camera groups")
    for value in sources:
        boundary_guard()
        context, adapter, case_id = value["context"], value["adapter"], value["case_id"]
        if value["run"] not in context.modules["engine"].FIT_RUNS:
            raise ValueError("No heldout acquisition may enter the prefreeze map measurement")
        with context.reader.frame_inputs(boundary_guard=boundary_guard):
            loaded = adapter.load_response(case_id, scope="calibration")
            frame = loaded["response"]["frame"]
            entry = context.cases[case_id]["entry"]
            receipt = context.reader.json(entry["receipt"])
            revision = receipt["rig_domain_revision"]
            geometry = receipt["full_extended_scientific"]["binding"]["geometry"]
            native_source = context.modules["source"]
            reference = native_source.load_physical_reference(context.session["parts"]["physical_reference"], context.reader)
            matrix = native_source.matrix_ac28_to_canvas(geometry, revision["native_origin_xy"])
            cv2 = context.modules["adapter"].cv2
            facies = cv2.warpPerspective(np.asarray(reference["facies"], np.uint8), matrix,
                frame.domain.shape[::-1], flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            photo = np.asarray(frame.baseline_rgb)
            region = binary_erosion(frame.domain & (facies > 0), iterations=20)
            samples = sandmap.layer_boundary_samples(facies, region, stride=32, max_samples=2000)
            measured = sandmap.measure_layer_edges(photo, samples, search_radius_px=32, min_contrast=.02)
            group = "AC17-38" if frame.run_id in {"ac28"} else "AC39-61"
            measurements.append(dict(run=frame.run_id, camera_group=group,
                map_xy=measured["map_xy"], photo_xy=measured["photo_xy"], normals_xy=measured["normals_xy"],
                H_ac28_to_frame=matrix))
            details.append(dict(run=frame.run_id, camera_group=group, phase_receipt=entry["receipt"],
                baseline_source=receipt["baseline_extended_scientific"]["binding"],
                baseline_source_sha256=receipt["baseline_extended_scientific"]["source_sha256"],
                physical_reference=context.session["parts"]["physical_reference"],
                samples=len(samples), accepted_samples=int(np.count_nonzero(measured["accepted"])),
                rejected_samples=measured["rejected_count"], contrasts=measured["contrast"].tolist(),
                method=measured["method"], minimum_linear_RGB_contrast=.02,
                sharp_region="positive original facies inside domain, eroded20px; only measured sharp photo edges accepted"))
            del loaded, frame, photo, facies, region, reference
            context.captured.clear();gc.collect()
    boundary_guard()
    correction, report = sandmap.choose_alignment(measurements[0], measurements[1:],
        elapsed_seconds=time.monotonic()-started_monotonic,
        expected_camera_groups=("AC17-38", "AC39-61"))
    report.update(complete=True, round4_spec_sha256=SPEC_SHA256, correction_matrix=correction.tolist(),
        elapsed_seconds=time.monotonic()-started_monotonic, source_measurements=details,
        current_map_retained=not report["transform_applied"])
    io.write_json(output, report)
    return correction, io.asset(output)


def prepare_scope_metadata(cpu_packet, output, *, native_module, selection_module, io):
    """Reuse the existing metadata collector for one immutable round-4 inventory."""
    from darsia.fluidflower.production_io import normalize_locator
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previous = cpu_packet["runs"]["ac18"]["float64"]
    manifest = io.read_json(previous["manifest"], max_bytes=previous["manifest"]["size_bytes"])
    reader = selection_module.MetadataReader(manifest, previous["staged_inputs"],
                                             cpu_packet["logical_aliases"], io=io)
    old = reader.json(previous["configuration"])
    modules = native_module._modules()
    campaign = reader.json(old["campaign"])
    validated = modules["runner"].validate_selected_plan(campaign["plan"], reader)
    scopes = selected_scope(validated, engine=modules["engine"])
    inventories, union = {}, {}
    for run, cases in scopes.items():
        selection_path = output/run/"selected_scope.json"
        io.write_json(selection_path, dict(round4_spec_sha256=SPEC_SHA256, run=run, cases=cases))
        selection = io.asset(selection_path)
        reader.register(selection)
        config = dict(old, profile_selection=selection, round4_spec_sha256=SPEC_SHA256)
        config_path = output/run/"native_configuration.json"
        io.write_json(config_path, config)
        configuration = io.asset(config_path)
        reader.register(configuration)
        def exact_config(value):
            if value != config:
                raise ValueError("Round 4 scope configuration changed")
            return value
        def exact_cases(value):
            if value != reader.json(selection):
                raise ValueError("Selected run scope changed")
            return cases
        collector = clone(native_module.enumerate_native_closure,
            _configuration=exact_config, _cases=exact_cases, LOCKED_CASES=cases)
        closure = collector(config, reader, modules=modules,
                            size_resolver=reader.size, include_anchors=True)
        if closure["scientific_arrays_read"] is not False:
            raise ValueError("Metadata preparation decoded scientific arrays")
        closure["assets"].append(dict(descriptor=configuration, roles=["round4_configuration"]))
        for row in closure["assets"]:
            bound = row["descriptor"]
            key = normalize_locator(bound["path"])
            if key in union and (union[key]["descriptor"]["sha256"], union[key]["descriptor"]["size_bytes"]) != (bound["sha256"], bound["size_bytes"]):
                raise ValueError("Conflicting selected source identity")
            item = union.setdefault(key, dict(descriptor=bound, roles=set(), runs=set()))
            item["roles"].update(row["roles"])
            item["runs"].add(run)
        inventories[run] = dict(selection=selection, configuration=configuration,
            case_ids=list(cases), frame_count=len(cases), calibration=run in modules["engine"].FIT_RUNS,
            descriptive=run in modules["engine"].DESCRIPTIVE_RUNS)
        print(json.dumps({"round4_metadata_run": run, "frames": len(cases),
                          "assets": len(closure["assets"])}), flush=True)
    assets = [dict(descriptor=value["descriptor"], roles=sorted(value["roles"]),
                   working_runs=sorted(value["runs"])) for _, value in sorted(union.items())]
    for index, item in enumerate(assets):
        for run in item["working_runs"]:
            inventories[run].setdefault("input_roles", []).append(index)
    document = dict(round4_spec_sha256=SPEC_SHA256, complete=True, campaign=old["campaign"],
        gate=validated["gate"], plan=campaign["plan"], runs=inventories, assets=assets,
        common_roles=[index for index, item in enumerate(assets) if len(item["working_runs"]) == 39],
        original_source_provenance=cpu_packet["source_provenance"],
        logical_aliases=cpu_packet["logical_aliases"], opaque_descriptors=cpu_packet["opaque_descriptors"],
        scientific_arrays_read=False, backend_assignment=None, precision=None,
        production_permission_granted=False)
    path = output/"selected_inventory.json"
    io.write_json(path, document)
    return io.asset(path)


def make_adapter_class(precision_module, sandmap_module):
    """Keep the qualified precision arithmetic; change only v4 physics/scope."""
    import numpy as np

    class Round4Adapter(precision_module.PrecisionAdapter):
        def __init__(self, *args, sandmap_provider, **kwargs):
            super().__init__(*args, **kwargs)
            self.sandmap_provider = sandmap_provider
            self.scope, self._selection = None, None

        def _prepare(self, frame, path_folder, parts, swi, scope, frozen_calibration,
                     *, asset_reader=None):
            context, math = self.context, self.math
            engine = context.modules["engine"]
            if scope == "calibration":
                if frame.run_id not in engine.FIT_RUNS or frozen_calibration is not None:
                    raise ValueError("Only fit runs may be decoded before the new freeze")
                parameters, digest = None, None
            elif scope == "frozen_evaluation" and frozen_calibration is not None:
                parameters, digest = self.retained._selected_n1(
                    context.modules, self._selection, asset_reader)
            else:
                raise ValueError("Round 4 evaluation requires its actual frozen calibration")
            r1, r2 = engine.r1, engine.r2
            kwargs = {} if asset_reader is None else {"asset_reader": asset_reader}
            original, original_digest = r2.verify_original_round1_freeze(
                parts["r1_freeze"]["path"], **kwargs)
            if Path(original["path_folder"]).resolve() != Path(path_folder).resolve():
                raise ValueError("Pooled round-1 colour paths changed")
            def normalized(frame):
                clear = frame.clear_water_rgb
                support = frame.known_physical_domain & frame.supported_mask & frame.clear_water_supported_mask
                support &= np.isfinite(clear).all(axis=2) & (clear > 0).all(axis=2)
                current, baseline = math.reference_normalized(
                    frame.current_rgb, frame.baseline_rgb, clear, support)
                if not np.isfinite(current[support]).all() or not np.isfinite(baseline[support]).all():
                    raise ValueError("Nonfinite clear-water-normalized scientific signal")
                return current, baseline, support
            prepare = clone(r1.prepare_native_response,
                np=Proxy(np, float32=math.dtype), reference_normalized_rgb=normalized,
                UnclippedNativeConcentrationAnalysis=self._TypedAnalysis,
                darsia=Proxy(r1.darsia, TitrationFlash=self._TypedFlash),
                verify_frozen_calibration=r2.verify_original_round1_freeze)
            response = prepare(frame, path_folder, flash_kind="titration",
                frozen_calibration_path=parts["r1_freeze"]["path"]
                if frame.run_id in engine.HELDOUT_RUNS else None, **kwargs)
            measured_swi = np.full(frame.domain.shape, np.nan, math.dtype)
            # v4 binds Swi to the actual facies.csv table read by native setup.
            for label, value in context.session["swi"].items():
                value = float(value)
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Measured Swi must be in [0,1]")
                measured_swi[frame.facies == int(label)] = math.dtype(value)
            response.update(swi=measured_swi, frozen_parameters=parameters,
                round1_freeze_sha256=original_digest, selected_calibration_sha256=digest,
                cross_host_source_equivalence=original["cross_host_source_equivalence"],
                precision=self.precision, pixel_backend=self.backend_facts["backend"],
                pixel_backend_facts=self._facts(),
                input_storage_dtype="float32-log-RGB/frozen-clearwater",
                working_pixel_dtype=self.precision, parameter_dtype="float64", reduction_dtype="float64",
                round4_spec_sha256=SPEC_SHA256)
            if asset_reader is not None:
                response["production_io_source_provenance"] = original["production_io_source_provenance"]
            return response

        def load_response(self, case_id, *, scope, selected_calibration=None):
            context, math = self.context, self.math
            if case_id not in context.cases:
                raise ValueError("Acquisition outside the worker's selected run")
            if scope == "calibration" and selected_calibration is not None:
                raise ValueError("No fitted calibration is applied during extraction")
            if scope == "frozen_evaluation":
                if selected_calibration is None:
                    raise ValueError("Freeze before heldout pixels")
                self.retained._selected_n1(context.modules, selected_calibration, context.asset_reader)
            self.scope, self._selection = scope, selected_calibration
            if context.boundary_guard is not None:
                context.boundary_guard()
            adapter, source = context.modules["adapter"], context.modules["source"]
            cv2 = adapter.cv2
            def compose(descriptor, reader):
                return precision_module._compose_scientific_raw(source, descriptor, reader, math)
            def clearwater(reference, matrix, shape, reader, **kwargs):
                return precision_module._registered_clearwater(source, reference, matrix, shape,
                    reader, math, cv2, **kwargs)
            def transport(reference, matrix, shape):
                return precision_module._transport_physical_reference(reference, matrix, shape, math, cv2)
            def typed_frame(prepared, **kwargs):
                return precision_module._to_native_frame(prepared, self._WorkingFrame, **kwargs)
            to_frame = sandmap_module.make_to_native_frame(typed_frame,
                lambda prepared: self.sandmap_provider(context, case_id, prepared))
            def continuous_warp(values, matrix, dsize, *, flags, borderMode, borderValue):
                if flags != cv2.INTER_LINEAR or borderMode != cv2.BORDER_CONSTANT or borderValue != 0:
                    raise ValueError("Original continuous warp changed")
                return math.warp_continuous(values, matrix, (dsize[1], dsize[0]), cv2_module=cv2)
            source_view = Proxy(source, compose_scientific_raw=compose, registered_clearwater=clearwater,
                transport_physical_reference=transport, to_native_frame=to_frame)
            selected = sandmap_module.make_selected_loader(adapter.load_selected,
                source_proxy=source_view, prepare_response=self._prepare,
                numpy_proxy=Proxy(np, float32=math.dtype),
                cv2_proxy=Proxy(cv2, warpPerspective=continuous_warp))
            def loader(**kwargs):
                return selected(**kwargs, **({} if context.asset_reader is None
                                             else {"asset_reader": context.asset_reader}))
            session = dict(context.session, loader=loader, reader=self.reader)
            load = clone(context.load_entry, SESSION=session)
            freeze = None if selected_calibration is None else selected_calibration["asset"]["path"]
            result = load(context.cases[case_id]["entry"], scope=scope, frozen_calibration=freeze)
            for name in ("current_rgb", "baseline_rgb", "clear_water_rgb"):
                math.require_pixels(getattr(result["response"]["frame"], name))
            if context.boundary_guard is not None:
                context.boundary_guard()
            return result

        def evaluate_loro(self, response, anchor, *, fold_descriptor, freeze_descriptor,
                          domain_masks, kept_fields=()):
            """Original qualified pixel evaluation with the genuinely omitted-run fit."""
            import copy
            context = self.context
            run = response["frame"].run_id
            engine = context.modules["engine"]
            if run not in engine.FIT_RUNS:
                raise ValueError("No heldout/descriptive image can be used as a fit-run fold")
            freeze, digest = engine.verify_round3_freeze(freeze_descriptor["path"],
                asset_reader=context.asset_reader)
            if digest != freeze_descriptor["sha256"] or freeze.get("round4_spec_sha256") != SPEC_SHA256:
                raise ValueError("LORO needs the actual published round-4 freeze")
            if not any(context.reader._same_descriptor(fold_descriptor, value)
                       for value in freeze["calibration_files"].values()):
                raise ValueError("The omitted-run fit is absent from the actual freeze")
            fitted = context.reader.json(fold_descriptor)
            if (fitted.get("heldout_fit_run") != run or fitted.get("stop_required") is not False
                    or fitted.get("final_heldout_used") is not False
                    or set(fitted.get("calibration_runs", ())) != set(engine.FIT_RUNS)-{run}):
                raise ValueError("Stopped or contaminated leave-one-run-out fit")
            parameters = fitted["parameters"]
            engine.r2._validate_parameters(parameters)
            def choice(modules, selected, asset_reader):
                if selected != {"verifier": "round4_loro", "asset": fold_descriptor}:
                    raise ValueError("Omitted-run fit selection changed")
                context.reader.path(fold_descriptor)
                return dict(parameters), fold_descriptor["sha256"]
            view = copy.copy(self)
            view.retained = Proxy(self.retained, _selected_n1=choice)
            adjusted = dict(response, frozen_parameters=dict(parameters))
            return precision_module.PrecisionAdapter.evaluate_main(view, adjusted, anchor,
                selected_calibration={"verifier": "round4_loro", "asset": fold_descriptor},
                domain_masks=domain_masks, kept_fields=kept_fields)

    return Round4Adapter


def summarize(context, loaded, entry, scope, *, adapter, backend="cpu"):
    """Original paired estimates and U_rest on the corrected physical map."""
    import numpy as np
    engine = context.modules["engine"]
    own = clone(engine.prepare_own_colour_response,
        r1=Proxy(engine.r1, UnclippedNativeConcentrationAnalysis=adapter._TypedAnalysis),
        np=Proxy(np, float32=adapter.math.dtype))
    def typed_own(*args, **kwargs):
        response, path_folder, geometry = args[:3]
        frame = response["frame"]
        observed = (frame.domain & np.asarray(kwargs["photo_support"], bool)
            & frame.clear_water_supported_mask & np.isfinite(frame.clear_water_rgb).all(axis=2)
            & (frame.clear_water_rgb > 0).all(axis=2) & geometry.valid)
        available = observed & ~np.isin(frame.facies, response["metadata"]["ignore_labels"])
        # Same registered RGB, same baseline, same divisor, same facies, same
        # analysis domain and same frozen flash: the already computed native
        # interpretation is exactly the required own-colour response.
        if (np.array_equal(geometry.valid & frame.domain, frame.known_physical_domain)
                and np.array_equal(available, response["quantifiable"])):
            value = dict(aqueous_relative=response["prepared"].aqueous_relative,
                available=available, audit=dict(native_optics_used=True, original_flash_object_used=True,
                    main_analysis_modified=False, colour_paths_modified=False,
                    same_geometry_domain=True, same_measured_optical_support=True,
                    reused_original_native_interpretation=True,
                    measured_colour_pixels=int(available.sum()), photo_support_required=True))
            context.captured["own_colour"] = value
            return value
        if context.asset_reader is not None:
            kwargs["asset_reader"] = context.asset_reader
        value = own(*args, **kwargs)
        context.captured["own_colour"] = value
        return value
    local_engine = context.summary_from_loaded.__globals__["engine"]
    def baseline_main(response, parameters, anchor, **unused):
        # The original R3 summarizer asks for N1 only. Its former generic R2
        # wrapper evaluated a discarded N2 array and forced maps to float32.
        # Use the already qualified chosen-precision N1 math and identical
        # Geometry.integrate operation, with the same fixed raw-summary gain.
        if parameters != dict(candidate="N1", gain=engine.GAIN, beta=0.):
            raise ValueError("Extraction baseline must retain the original fixed N1 parameters")
        frame, physics, analysis = response["frame"], response["physics"], response["analysis"]
        partition = adapter.retained._main_partition(response)
        factor = engine.r2.dilution_factor(frame.elapsed_h, partition["affected_pore_volume_m3"], anchor, 0.)
        maps = adapter.math.selected_n1_fields(response["prepared"].aqueous_relative,
            gain=parameters["gain"], factor=factor, swi=response["swi"],
            aqueous_known=partition["aq_known"], gas_known=partition["gas_known"],
            solubility=physics.solubility_co2, gas_density=physics.density_gaseous_co2)
        eligible = frame.mass_eligible_mask & frame.known_physical_domain
        def integrate(field, mask):
            analysis.geometry._prepare_cached_voxel_volume(list(field.shape[:analysis.geometry.space_dim]))
            if np.asarray(analysis.geometry.cached_voxel_volume).dtype != np.float64:
                raise TypeError("Original integration weights must remain float64")
            return float(analysis.geometry.integrate(np.where(mask, field, 0.)))*1000
        gas = eligible & partition["gas_known"]
        return dict(total_mass_g=integrate(maps["mass_total"], eligible),
                    gas_mass_g=integrate(maps["mass_gas"], gas),
                    gas_water_mass_g=integrate(maps["mass_aqueous"], gas))
    native_summary = clone(engine.summarize_round3_response,
        r2=Proxy(engine.r2, evaluate_round2_frame=baseline_main))
    summary = clone(context.summary_from_loaded,
                    engine=Proxy(local_engine, prepare_own_colour_response=typed_own,
                                 summarize_round3_response=native_summary))
    row = summary(loaded, entry, scope, backend)
    row.update(round4_spec_sha256=SPEC_SHA256, round4_amendment_sha256=AMENDMENT_SHA256,
               round4_U_rest_policy="original_capacity_on_corrected_map", corrected_physical_support=True,
               main_geometry_modified=True)
    return row


def fit_candidate(rows, gate, *, fitter, excluded_run=None):
    if any(row.get("round4_spec_sha256") != SPEC_SHA256
           or row.get("round4_amendment_sha256") != AMENDMENT_SHA256
           for row in rows):
        raise ValueError("Fit requires corrected round-4 rows under amendment 1")
    fitted = fitter.fit_round3_candidate(rows, selected_phase_gate=gate, excluded_run=excluded_run)
    from collections import Counter
    counts = Counter()
    for row in fitted["frame_decisions"]:
        time_bin = "injection" if row["stage"] == "stage1" else fitter.post_time_bin(row["selected_target_h"])
        reasons = row["omitted_reasons"] or ["included"]
        for reason in reasons:
            counts[(row["stage"], time_bin, reason)] += 1
    fitted["eligibility_counts_by_stage_time_bin_reason"] = [
        dict(stage=stage, time_bin=time_bin, reason=reason, frames=number)
        for (stage, time_bin, reason), number in sorted(counts.items())]
    fitted["round4_spec_sha256"] = SPEC_SHA256
    fitted["amendment_sha256"] = AMENDMENT_SHA256
    return fitted


def publish_freeze(job, fits, calibration_files, gate, plan, *, engine, io, precision, sandmap):
    """One immutable global freeze after every original global/fold audit passes."""
    expected = {None, *engine.FIT_RUNS}
    if set(fits) != expected:
        raise ValueError("Global and all 26 omitted-run fits must finish")
    for omitted, fitted in fits.items():
        if (fitted.get("stop_required") is not False or fitted.get("final_heldout_used") is not False
                or fitted.get("heldout_fit_run") != omitted
                or set(fitted["calibration_runs"]) != set(engine.FIT_RUNS)-({omitted} if omitted else set())
                or any(audit.get("stop_required") is not False for audit in fitted["grid_audits"].values())):
            raise ValueError("Stopped, contaminated or incompletely audited N1 fit")
    path = Path(job)/"calibration/freeze/frozen_calibration.json"
    value = dict(contract=engine.FREEZE_CONTRACT, frozen=True,
        frozen_utc=datetime.now(timezone.utc).isoformat(), parameters=fits[None]["parameters"],
        calibration_runs=list(engine.FIT_RUNS), heldout_runs=list(engine.HELDOUT_RUNS),
        descriptive_runs=list(engine.DESCRIPTIVE_RUNS), round2_freeze_sha256=engine.ROUND2_FREEZE_SHA256,
        calibration_files=calibration_files, selected_phase_gate=gate, selected_input_plan=plan,
        spec_sha256=SPEC_SHA256, round4_spec_sha256=SPEC_SHA256,
        amendment_sha256=AMENDMENT_SHA256,
        U_rest_policy="original round3 capacity on corrected sand map; no fit fallback",
        check_fallback_policy="per check only when primary assessable times are too few",
        fallback_rule_sha256=U0_RULE_SHA256, precision=precision, sandmap=sandmap,
        stage2_eligibility_policy=fits[None]["stage2_eligibility_policy"],
        dilution_anchor_policy="explicit selected 2.25 h target; original volume/time convention",
        gas_branch_fitted=False, colour_paths_refitted=False, map_mass_rescaling=False,
        overflow_in_objective=False, production_automatically_allowed=False)
    io.write_json(path, value)
    engine.verify_round3_freeze(path)
    return io.asset(path)


def fit_all(statistics, gate, output_root, *, fitter, engine, io, boundary_guard,
            precision, physical_correction, plan, workers=8):
    """Original 27 fits on one shared set of saved native statistics."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import gzip, csv
    import numpy as np
    import threadpoolctl
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Original fit-worker ceiling is16")
    rows = []
    for descriptor in statistics:
        boundary_guard()
        with gzip.open(io.verify(descriptor), "rt", encoding="utf-8") as handle:
            row = json.load(handle)
        if (row.get("round4_spec_sha256") != SPEC_SHA256
                or row.get("round4_amendment_sha256") != AMENDMENT_SHA256
                or row.get("precision") != precision
                or row.get("physical_correction") != physical_correction):
            raise ValueError("All fit rows must use the actual corrected map and common precision")
        # These are the exact original np.asarray conversions. Cache them once;
        # no fitting equation, weight, histogram bin or optimizer is changed.
        paired = row["paired_estimates"]
        for name in ("own_native_relative","column_native_relative","solubility_ratio","kind","aqueous_capacity_g"):
            paired[name] = np.asarray(paired[name], np.int64 if name == "kind" else np.float64)
            paired[name].flags.writeable = False
        for name in ("capacity_g", "native_relative"):
            row["main_aqueous_histogram"][name] = np.asarray(row["main_aqueous_histogram"][name], np.float64)
            row["main_aqueous_histogram"][name].flags.writeable = False
        rows.append(row)
    if ({row["run"] for row in rows} != set(engine.FIT_RUNS)
            or len({(row["run"], row["frame_index"]) for row in rows}) != len(rows)):
        raise ValueError("Exactly the26 fit runs with unique original acquisitions are required")
    output = Path(output_root)/"calibration/fits"
    output.mkdir(parents=True, exist_ok=False)
    fits, files = {}, {}
    def fit_one(omitted):
        boundary_guard()
        fitted = fit_candidate(rows, gate, fitter=fitter, excluded_run=omitted)
        boundary_guard()
        path = output/("global.json" if omitted is None else "loro_"+omitted+".json")
        io.write_json(path, fitted)
        return omitted, fitted, io.asset(path)
    with threadpoolctl.threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers, thread_name_prefix="original-N1-fit") as pool:
        futures = [pool.submit(fit_one, omitted) for omitted in (None, *engine.FIT_RUNS)]
        for future in as_completed(futures):
            omitted, fitted, asset = future.result()
            fits[omitted] = fitted
            files["global" if omitted is None else "loro_"+omitted] = asset
            print(json.dumps(dict(round4_fit=omitted or "global", complete=True,
                stop_required=fitted["stop_required"], parameters=fitted["parameters"])), flush=True)
    eligibility = Path(output_root)/"calibration/frame_eligibility_counts.csv"
    with eligibility.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("fit","stage","time_bin","reason","frames"))
        writer.writeheader()
        for omitted,fitted in fits.items():
            writer.writerows(dict(fit=omitted or "global", **row)
                for row in fitted["eligibility_counts_by_stage_time_bin_reason"])
    return dict(fits=fits, files=files, eligibility=io.asset(eligibility), statistics_count=len(rows))


def exact_prediction(row, result, parameters, anchor, *, engine, freeze=None, reader=None, omitted=None):
    if (row.get("round4_spec_sha256") != SPEC_SHA256
            or row.get("round4_amendment_sha256") != AMENDMENT_SHA256):
        raise ValueError("Round 4 predictions require corrected amendment-1 statistics")
    kwargs = {} if reader is None else {"asset_reader": reader}
    prediction = engine.evaluate_round3_summary(row, parameters, anchor,
        frozen_calibration=freeze, **kwargs)
    main = float(result["total_mass_g"])
    urest = row["urest_g"]
    lo = main+prediction["E_lo_g"]
    hi = None if urest is None else main+prediction["E_hi_g"]+urest
    dose = prediction["injected_mass_g"]
    width = None if urest is None else prediction["E_hi_g"]-prediction["E_lo_g"]+urest
    residual = None if width is None or dose is None or dose <= 0 else engine.r2.interval_residual(lo, dose, width)
    prediction.update(main_mass_g=main, total_mass_g=main,
        aqueous_mass_g=result["aqueous_mass_g"], gas_mass_g=result["gas_mass_g"],
        gas_share=0. if main == 0 else result["gas_mass_g"]/main,
        U_rest_g=urest, W_g=width, W_over_d=None if width is None or dose is None or dose <= 0 else width/dose,
        interval_lower_g=lo, interval_upper_g=hi, signed_interval_residual=residual,
        delta=None if residual is None else abs(residual), exact_native_main_replay=True,
        selected_target_h=row["selected_target_h"], scope=row["scope"],
        round4_spec_sha256=SPEC_SHA256, precision=result["precision"], heldout_fit_run=omitted,
        evaluation_is_leave_one_run_out=omitted is not None, main_geometry_modified=True,
        amendment_sha256=AMENDMENT_SHA256)
    def scalar_part(part):
        return {key: ({label:scalar_part(value) for label,value in item.items()}
                      if key == "by_facies" else item)
                for key,item in part.items() if key != "own_colour_histogram"}
    prediction["excluded_parts"] = {name:scalar_part(part)
                                    for name,part in prediction.get("excluded_parts", {}).items()}
    return prediction


def zero_capacity_prediction(row, *, engine):
    lower, upper = row["main_mass_g"]+row["E_lo_g"], row["main_mass_g"]+row["E_hi_g"]
    width, dose = row["E_hi_g"]-row["E_lo_g"], row["injected_mass_g"]
    residual = None if dose is None or dose <= 0 else engine.r2.interval_residual(lower, dose, width)
    return dict(row, U_rest_g=0., W_g=width,
        W_over_d=None if dose is None or dose <= 0 else width/dose,
        interval_lower_g=lower, interval_upper_g=upper, signed_interval_residual=residual,
        delta=None if residual is None else abs(residual), fallback_U_rest_rule=U0_RULE_SHA256)


def score(heldout, loro, *, original_runner):
    if any(row.get("round4_spec_sha256") != SPEC_SHA256 or row.get("amendment_sha256") != AMENDMENT_SHA256
           for row in [*heldout, *loro]):
        raise ValueError("Fixed round-4 amendment-1 predictions are required for all checks")
    primary = original_runner.score_literal_checks(heldout, loro)
    fallback = original_runner.score_literal_checks(
        [zero_capacity_prediction(row, engine=original_runner.engine) for row in heldout],
        [zero_capacity_prediction(row, engine=original_runner.engine) for row in loro])
    decided = dict(primary)
    rules = {}
    for key, minimum in (("late_check", 4), ("after48_check", 3)):
        use_fallback = primary[key]["assessable_time_count"] < minimum
        decided[key] = fallback[key] if use_fallback else primary[key]
        rules[key] = "U_rest_zero_fallback_insufficient_assessable_times" if use_fallback else "original_U_rest_capacity"
    decided.update(pass_gate=all(row["pass_gate"] for row in primary["check1"])
        and decided["late_check"]["pass_gate"] and decided["after48_check"]["pass_gate"],
        primary_rule=primary, zero_capacity_rule=fallback, deciding_rule=rules,
        round4_spec_sha256=SPEC_SHA256, amendment_sha256=AMENDMENT_SHA256)
    return decided


def check_tables(heldout, loro, *, original_runner):
    """Both fixed rules per run/time, including the rule that decides each check."""
    checks = score(heldout, loro, original_runner=original_runner)
    table = []
    for row in [*heldout, *loro]:
        target = float(row["selected_target_h"])
        check = "late_check" if target in (4., 8., 12., 24., 48.) else "after48_check" if target in LATE_TARGETS else None
        if check is None:
            continue
        for rule, value in (("original_U_rest_capacity", row),
                            ("U_rest_zero_fallback_insufficient_assessable_times",
                             zero_capacity_prediction(row, engine=original_runner.engine))):
            table.append(dict(run=row["run"], frame_index=row["frame_index"], target_h=target,
                elapsed_h=row["elapsed_h"], check=check, rule=rule,
                deciding_rule=checks["deciding_rule"][check],
                main_mass_g=value["main_mass_g"], injected_mass_g=value["injected_mass_g"],
                E_lo_g=value["E_lo_g"], E_hi_g=value["E_hi_g"], U_rest_g=value["U_rest_g"],
                W_g=value["W_g"], W_over_d=value["W_over_d"], delta=value["delta"],
                counted=value["W_over_d"] is not None and value["W_over_d"] <= .05 and value["delta"] is not None))
    return checks, table


def save_checks(heldout, loro, output_root, *, original_runner, io, freeze, physical_correction):
    """Actual check results and both predeclared U_rest tables for Claude."""
    import csv
    checks, table = check_tables(heldout, loro, original_runner=original_runner)
    root = Path(output_root)/"checks"
    root.mkdir(parents=True, exist_ok=False)
    checks.update(complete=True, all_checks_passed=checks["pass_gate"],
        urest_amendment_sha256=AMENDMENT_SHA256, frozen_calibration=freeze,
        physical_correction=physical_correction, working_precision="float64",
        heldout_count=len(heldout), loro_count=len(loro), claude_score_state="awaiting_once")
    io.write_json(root/"checks.json", checks)
    io.write_json(root/"heldout_predictions.json", heldout)
    io.write_json(root/"loro_predictions.json", loro)
    with (root/"urest_both_rules.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader();writer.writerows(table)
    text = ["# FF_AC round4 checks", "", "Frozen calibration: `"+freeze["sha256"]+"`.",
        "", "U_rest is computed on the corrected sand map. The fit uses the original rule with no fallback.",
        "", "Check1: "+("PASS" if all(row["pass_gate"] for row in checks["check1"]) else "FAIL")+"."]
    for key,label in (("late_check","Check2"),("after48_check","Check3")):
        value = checks[key]
        text += ["", label+": "+("PASS" if value["pass_gate"] else "FAIL")+". Deciding rule: `"+
                 checks["deciding_rule"][key]+"`; assessable times: "+str(value["assessable_time_count"])+"."]
    text += ["", "Both rules for every run/time are in urest_both_rules.csv. An assessable primary failure cannot use the fallback.",
             "", "All checks passed: "+str(checks["pass_gate"])+". Claude scores these saved tables once."]
    (root/"report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return dict(checks=io.asset(root/"checks.json"), table=io.asset(root/"urest_both_rules.csv"),
        heldout=io.asset(root/"heldout_predictions.json"), loro=io.asset(root/"loro_predictions.json"),
        report=io.asset(root/"report.md"), all_checks_passed=checks["pass_gate"])


def published_records(publications, *, io):
    """Read small exact committed records; never decode products for reporting."""
    records, roots, statistics = [], [], {}
    for publication in publications:
        folder = Path(publication["nas_directory"])
        descriptor = dict(publication["files"]["result.json"], path=str(folder/"result.json"))
        record = io.read_json(descriptor, max_bytes=descriptor["size_bytes"])
        if (record.get("complete") is not True or record.get("case_id") != publication["frame_id"]
                or record.get("round4_spec_sha256") != SPEC_SHA256):
            raise ValueError("A published record differs from the exact committed round4 frame")
        records.append(record);roots.append(str(folder))
        if "native_statistics.json.gz" in publication["files"]:
            statistics[record["case_id"]] = dict(publication["files"]["native_statistics.json.gz"],
                path=str(folder/"native_statistics.json.gz"))
    return records, roots, statistics


def render_completed_run(publications, local_output, nas_output, *, products, renderer,
                         io, boundary_guard, checks, failed_check=None, delegation=None):
    """Adopt the genuinely frozen validation products after the check decision.

    Heldout priority films use their already completed N1 products. No image,
    colour interpretation or calibration is computed again to make a film.
    """
    records, roots, _ = published_records(publications, io=io)
    if not records or len({row["run"] for row in records}) != 1:
        raise ValueError("One complete selected run is required")
    if failed_check is None and checks.get("all_checks_passed") is not True:
        raise PermissionError("Unmarked production media require actual checks1–3PASS")
    if failed_check is not None:
        if records[0]["run"] not in PRIORITY_RUNS or delegation is None:
            raise PermissionError("Only the five explicitly delegated provisional runs may be rendered")
        records = [dict(row, provisional=True, failed_check=failed_check,
                        provisional_delegation=delegation) for row in records]
    film = products.render_run(records, frame_roots=roots, output_dir=local_output,
        renderer=renderer, io=io, guard=boundary_guard, fps=3,
        failed_check=failed_check, delegation=delegation)
    table = products.mass_tables(records, Path(local_output)/"mass.csv", io=io, guard=boundary_guard)
    plot = products.plot_masses(records, Path(local_output)/"mass.png", io=io, guard=boundary_guard)
    publication = products.publish_media(local_output, nas_output, io=io, guard=boundary_guard)
    return dict(complete=True, run=records[0]["run"], source_frame_count=len(records), film=film,
        mass_table=table, mass_plot=plot, nas_output=str(nas_output), publication=publication,
        calibration_sha256=records[0]["bindings"]["calibration_sha256"],
        provisional=failed_check is not None, failed_check=failed_check,
        numerical_frames_recomputed=0, records=records)


def verified_copy_stage(manifest, manifest_descriptor, staged_inputs, *, cache_root,
                        copy_identities, stream_module):
    """Bind exact NasIO-verified copies without a second global payload hash pass."""
    from darsia.fluidflower.production_io import normalize_locator
    if set(staged_inputs) != set(manifest["inputs"]):
        raise ValueError("Every declared input must be copied")
    rows = {}
    for role, logical in manifest["inputs"].items():
        local = staged_inputs[role]
        if (logical["sha256"], logical["size_bytes"]) != (local["sha256"], local["size_bytes"]):
            raise ValueError("Copied bytes differ from the immutable logical source")
        path = Path(local["path"])
        if not path.resolve().is_relative_to(Path(cache_root).resolve()):
            raise ValueError("Copied input escaped the owned local cache")
        identity = stream_module.local_identity(path)
        if identity != copy_identities[normalize_locator(str(path))]:
            raise ValueError("NasIO-verified local copy changed before stage admission")
        rows[role] = dict(logical=logical, local=local, local_identity=identity)
    return dict(format="darsia-production-verified-stage-v1", complete=True,
        manifest=manifest_descriptor, cache_root=str(Path(cache_root).resolve()), inputs=rows,
        payload_sha_verified_by_coordinator=True, arrays_decoded=False,
        verification_source="original NasIO copy_file verified source, temporary and destination",
        subsequent_read_policy="pinned_local_identity_no_rehash", production_permission_granted=False)


def run_selected_worker(task, *, dependencies, reader, identity, boundary_guard):
    """One whole selected AC run, original math, one-frame prefetch and NAS copy."""
    from concurrent.futures import ThreadPoolExecutor
    import gc, gzip, time
    import numpy as np
    from darsia.fluidflower import production_job, production_native, production_retained
    from darsia.presets.workflows.config.roi_registry import RoiRegistry
    io = reader.io
    stage, run = task["stage"], task["run"]
    if stage not in {"extraction", "heldout", "loro", "production", "provisional"}:
        raise ValueError("Unknown round-4 scientific stage")
    configuration = reader.json(task["configuration"])
    cases = reader.json(configuration["profile_selection"])["cases"]
    with reader.frame_inputs(boundary_guard=boundary_guard):
        context = prepare_context(configuration, reader, cases, native_module=production_native,
                                  boundary_guard=boundary_guard)
    engine = context.modules["engine"]
    if stage == "extraction" and run not in engine.FIT_RUNS:
        raise ValueError("Heldout/descriptive images cannot enter calibration extraction")
    if stage == "heldout" and run not in engine.HELDOUT_RUNS:
        raise ValueError("Heldout stage needs an original heldout run")
    if stage == "loro" and run not in engine.FIT_RUNS:
        raise ValueError("LORO needs an original fit run")
    factory = make_adapter_class(dependencies["precision_adapter"], dependencies["sandmap"])
    provider = dependencies["provider_factory"](task["settings"])
    adapter = factory(context, task["settings"]["precision"], pixel_math=provider,
        retained_module=production_retained,
        sandmap_provider=lambda current_context, case_id, prepared:
            dependencies["products"].geometry_provider(current_context, case_id, prepared,
                correction_matrix=task.get("correction_matrix")))
    job = production_job.LocalProductionJob(task["manifest"], local_root=task["local_results"],
        nas_root=task["nas_results"], io=io)
    selected = None if stage == "extraction" else {"verifier": "round3", "asset": task["freeze"]}
    anchor = None
    frame_ids = list(cases)
    if stage == "loro":
        frame_ids = [case for case in frame_ids
                     if float(cases[case]["entry"]["selected_target_h"]) in LATE_TARGETS]
    if stage != "extraction":
        anchor_id = next(case for case, value in cases.items()
                         if float(value["entry"]["selected_target_h"]) == 2.25)
        with reader.frame_inputs(boundary_guard=boundary_guard):
            loaded = adapter.load_response(anchor_id, scope="frozen_evaluation", selected_calibration=selected)
            anchor = anchor_from_response(context, loaded["response"], cases[anchor_id]["entry"])
            del loaded
            context.captured.clear()
            gc.collect()
    rois = RoiRegistry().load(reader.path(task["boxes"])).resolve_rois(["box1", "box2"])
    prefetch = dependencies["lifecycle"].OneFramePrefetch(reader, max_bytes=1024**3)
    publish = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verified-NAS-copy")
    publication_callback = dependencies.get("publication_callback")
    if publication_callback is not None and not callable(publication_callback):
        raise ValueError("A host publication callback must return actual verified NAS evidence")
    pending, records = None, []
    clock = time.perf_counter()
    try:
        if frame_ids:
            prefetch.submit(frame_ids[0], dependencies["prefetch_plan"](context, reader, frame_ids[0]))
        for index, case_id in enumerate(frame_ids):
            boundary_guard()
            entry = cases[case_id]["entry"]
            plan = dependencies["prefetch_plan"](context, reader, case_id)
            with prefetch.consume(case_id, boundary_guard=boundary_guard) as prefetched:
                reader._prefetched = {reader._canonical_key(reader._stage_logical(plan[name])["path"]): value
                                     for name, value in prefetched.items()}
                if index+1 < len(frame_ids):
                    next_id = frame_ids[index+1]
                    prefetch.submit(next_id, dependencies["prefetch_plan"](context, reader, next_id))
                with reader.frame_inputs(boundary_guard=boundary_guard):
                    started = time.perf_counter()
                    scope = "calibration" if stage == "extraction" else "frozen_evaluation"
                    loaded = adapter.load_response(case_id, scope=scope, selected_calibration=selected)
                    response, frame = loaded["response"], loaded["response"]["frame"]
                    row, result, prediction, export, product_record = None, None, None, None, None
                    temporary = job.begin_frame(case_id)
                    cached = task.get("statistics_by_case", {}).get(case_id)
                    if cached is not None:
                        with gzip.open(reader.path(cached), "rt", encoding="utf-8") as handle:
                            row = json.load(handle)
                        if (row.get("round4_spec_sha256") != SPEC_SHA256
                                or row.get("round4_amendment_sha256") != AMENDMENT_SHA256
                                or row.get("precision") != task["settings"]["precision"]
                                or row.get("source_binding") != entry["source_binding"]
                                or row.get("frame_index") != entry["frame_index"]
                                or row.get("physical_correction") != task.get("physical_correction")
                                or row.get("round4_code_sha256") != task["code_sha256"]):
                            raise ValueError("Cached statistics differ from this exact corrected frame/code/precision")
                    else:
                        row = summarize(context, loaded, entry, scope, adapter=adapter, backend="cpu")
                        row.update(precision=task["settings"]["precision"], backend=task["settings"]["backend"])
                        row.update(physical_correction=task.get("physical_correction"),
                                   round4_code_sha256=task["code_sha256"])
                        with gzip.open(temporary/"native_statistics.json.gz", "wt", encoding="utf-8", compresslevel=3) as handle:
                            json.dump(context.modules["runner"]._jsonable(row), handle,
                                      allow_nan=False, separators=(",", ":"))
                    if stage != "extraction":
                        darsia = engine.r1.darsia
                        image = response["analysis"]._scalar(frame.phase_labels, frame.elapsed_h)
                        masks = {"full_rig": frame.domain,
                            **{name: darsia.roi_to_mask(roi.roi, image, mode="voxels").img
                               for name, roi in rois.items()}}
                        fields = () if stage == "loro" else production_retained.FIELDS
                        if stage == "loro":
                            result = adapter.evaluate_loro(response, anchor, fold_descriptor=task["fold"],
                                freeze_descriptor=task["freeze"], domain_masks=masks, kept_fields=fields)
                            parameters = reader.json(task["fold"])["parameters"]
                        else:
                            result = adapter.evaluate_main(response, anchor, selected_calibration=selected,
                                domain_masks=masks, kept_fields=fields)
                            parameters = result["parameters"]
                        result.update(contract="ffac-round4-native-N1-v4", round4_spec_sha256=SPEC_SHA256,
                            amendment_sha256=AMENDMENT_SHA256, stage=stage,
                            scientific_eligibility=True,
                            round4_admission="actual PartA resource lock and original source/live guards; authorized v4 physical correction",
                            production_permission_granted=stage == "production",
                            provisional=stage == "provisional")
                        if row is not None:
                            prediction = exact_prediction(row, result, parameters, anchor, engine=engine,
                                freeze=task["freeze"]["path"], reader=context.asset_reader,
                                omitted=run if stage == "loro" else None)
                        if stage != "loro":
                            export = adapter.main_products(result)
                            phase = reader.json(entry["receipt"])
                            bindings = dict(source_sha256=entry["source_binding"]["source_sha256"],
                                phase_sha256=phase["labels_map"]["sha256"],
                                calibration_sha256=task["freeze"]["sha256"], code_sha256=task["code_sha256"])
                            product_record = dependencies["products"].write_products(temporary, response=response, result=result,
                                export=export, prediction=prediction, bindings=bindings,
                                static_root=Path(task["local_results"])/"static",
                                static_registry=task.setdefault("_static_registry", {}),
                                renderer=dependencies["renderer"], io=io, guard=boundary_guard,
                                frame_position=index, total_frames=len(frame_ids),
                                solubility_limit_g_l=task["solubility_limit_g_l"],
                                provisional=stage == "provisional", failed_check=task.get("failed_check"),
                                delegation=task.get("provisional_delegation"), write_record=False)
                        del image, masks
                    record = dict(product_record or {}, complete=True, round4_spec_sha256=SPEC_SHA256, stage=stage,
                        run=run, case_id=case_id, frame_index=entry["frame_index"],
                        elapsed_h=entry["elapsed_h"], selected_target_h=entry["selected_target_h"],
                        source_binding=entry["source_binding"], source_frame_unchanged=True,
                        worker=identity, precision=task["settings"]["precision"], backend=task["settings"]["backend"],
                        anchor=anchor, prediction=prediction,
                        scalar_result=None if result is None else {key:value for key,value in result.items()
                            if key not in {"fields", "pixel_backend_facts"}},
                        sandmap=frame.provenance["round4_sandmap"],
                        compute_seconds=time.perf_counter()-started)
                    io.write_json(temporary/"result.json", context.modules["runner"]._jsonable(record))
                    job.commit_frame(case_id, temporary, product_metadata=dict(
                        precision=task["settings"]["precision"], backend=task["settings"]["backend"],
                        round4_spec_sha256=SPEC_SHA256, amendment_sha256=AMENDMENT_SHA256,
                        stage=stage, U_rest_policy="original_capacity_on_corrected_map"))
                    reader._prefetched = {}
                    del response, frame, loaded, row, result, prediction, export, product_record
                    context.captured.clear()
                    gc.collect()
                # At most one disk-only publication overlaps the next frame.
                if pending is not None:
                    records.append(pending.result())
                pending = (publish.submit(job.copy_frame_to_nas, case_id) if publication_callback is None
                           else publish.submit(publication_callback, job, case_id))
                print(json.dumps({"round4_run":run,"stage":stage,"frame":case_id,
                                  "compute_seconds":record["compute_seconds"]}), flush=True)
        if pending is not None:
            records.append(pending.result())
        static_copy = None
        static_path = Path(task["local_results"])/"static"
        if static_path.exists():
            inventory = {path.relative_to(static_path).as_posix(): io.asset(path)
                         for path in sorted(static_path.rglob("*")) if path.is_file()}
            boundary_guard()
            static_callback = dependencies.get("static_publication_callback")
            if static_callback is None:
                io.copy_directory(static_path, Path(task["nas_results"])/"static", inventory)
                static_copy = dict(nas_directory=str(Path(task["nas_results"])/"static"),
                                   files=inventory, verified=True)
            else:
                if not callable(static_callback):
                    raise ValueError("Static publication must supply actual verified NAS evidence")
                static_copy = static_callback(job, static_path, inventory)
        return dict(complete=True, run=run, stage=stage, frames=len(frame_ids), records=records,
            static_copy=static_copy,
            wall_seconds=time.perf_counter()-clock, images_per_hour=len(frame_ids)*3600/(time.perf_counter()-clock),
            actual_worker=identity, precision=task["settings"]["precision"], backend=task["settings"]["backend"])
    finally:
        reader._prefetched = {}
        prefetch.close()
        publish.shutdown(wait=True, cancel_futures=False)
        provider.close() if callable(getattr(provider, "close", None)) else None
