"""Explicit series DATA admission and native/phase/N1 stage adaptation.

No classifier, radiometric fit, geometric fit or mass equation is implemented
here. Frozen measurements and models enter through content-bound DATA. The
runner selects only these fixed library operations, never Python entrypoints
from a configuration file. Historical references are distinct from current
acquisitions and from expected validation products.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
import re
import tomllib

from .job_queue import name, require_unc

SERIES_FORMAT = "darsia-series-v1"
RUN_FORMAT = "darsia-native-phase-n1-run-v1"
PIPELINE = "native-phase-n1"


def pipeline_kind(task):
    """Select only the fixed series route or the compatible original N1 route."""
    value = task.get("pipeline", "n1")
    if value not in {"n1", PIPELINE}:
        raise ValueError("Unsupported explicit pipeline")
    return value


def _no_executable_configuration(value):
    """Reject executable configuration and current expected-product input routes."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "entrypoint",
                "python_module",
                "python_callable",
                "script",
                "expected_phase_products",
                "expected_current_labels",
                "expected_current_classes",
            }:
                raise ValueError(
                    "Configuration must contain DATA, not executable/expected products: "
                    + key
                )
            _no_executable_configuration(item)
    elif isinstance(value, list):
        for item in value:
            _no_executable_configuration(item)


def validate_run(config):
    """Admit a fixed chronological selection and its explicit stage-role data.

    References may identify original source caches and measured parent fields.
    Current labels/classes have no input slot: they are produced by infer_labels.
    """
    if config.get("format") != RUN_FORMAT:
        raise ValueError("Explicit native/phase/N1 run configuration required")
    _no_executable_configuration(config)
    name(config["run"])
    for field in (
        "frames",
        "native",
        "phase",
        "mass",
        "table_fields",
        "history_policy",
        "selection_index_field",
    ):
        if field not in config:
            raise ValueError("Missing run DATA: " + field)
    frames = config["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("A nonempty fixed frame selection is required")
    ids, digests, last = set(), set(), None
    predecessor = config["phase"]["initial_previous"]["source"]["source_sha256"]
    for spec in frames:
        identifier = name(spec["case_id"])
        source = spec["source"]
        digest = source["source_sha256"]
        if (
            identifier in ids
            or digest in digests
            or re.fullmatch("[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("Each acquisition/case must occur once")
        if source["run"] != config["run"] or source.get("selected") is not True:
            raise ValueError("Selected acquisition belongs to another run")
        index = spec["frame_index"]
        elapsed = spec["elapsed_h"]
        if (
            type(index) is not int
            or index < 0
            or type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
        ):
            raise ValueError("Exact acquisition index and finite time required")
        if (
            source[config["selection_index_field"]] != index
            or source["elapsed_h"] != elapsed
        ):
            raise ValueError("Selection/source index or time differs")
        if last is not None and not (index > last[1] and elapsed > last[0]):
            raise ValueError(
                "Selection must retain strictly chronological acquisition order"
            )
        if source["previous_source_sha256"] != predecessor:
            raise ValueError("Selection predecessor chain changed")
        for key in ("native_task", "geometry", "phase_roles", "mass_spec"):
            if key not in spec:
                raise ValueError("Missing frame-stage DATA: " + key)
        if spec["native_task"]["kind"] not in {"rig", "source"}:
            raise ValueError("A fixed native processor kind is required")
        forbidden = {"current", "labels", "classes", "area"} & spec["mass_spec"].keys()
        if forbidden:
            raise ValueError("Mass current/phase/area must be fresh stage outputs")
        area = spec["phase_roles"]["area"]
        expected_role = "measured_parent" if source.get("studio_anchor") else "selected"
        if area["role"] != expected_role:
            raise ValueError("Historical and selected physical-area routes differ")
        if area["role"] == "measured_parent" and not {
            "historical",
            "measured",
        }.issubset(area):
            raise ValueError("A parent needs separate original and measured fields")
        if not math.isfinite(float(area["factor"])) or float(area["factor"]) <= 0:
            raise ValueError("Physical area factor must be positive")
        ids.add(identifier)
        digests.add(digest)
        predecessor, last = digest, (elapsed, index)
    if config["mass"]["template"]["calibration"] != config["mass"]["calibration"]:
        raise ValueError("Mass template/calibration identity differs")
    if (
        not isinstance(config["table_fields"], list)
        or not config["table_fields"]
        or len(set(config["table_fields"])) != len(config["table_fields"])
    ):
        raise ValueError("Explicit distinct scalar-table fields are required")
    return config


def read_series(path, *, expected_sha256):
    """Read a byte-bound TOML selection; no payload or numerical call occurs."""
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Series TOML bytes changed")
    value = tomllib.loads(raw.decode("utf-8"))
    if value.get("format") != SERIES_FORMAT:
        raise ValueError("Explicit series format required")
    if value.get("adopted_for_execution") is not True:
        raise ValueError("Series DATA is inert until explicitly adopted")
    _no_executable_configuration(value)
    from ..mass.n1_pipeline import DataReader

    reader = DataReader(value.get("data_aliases", ()), unc_only=True)
    selected = reader.json(value["selection"])
    if selected.get("complete") is not True or not isinstance(
        selected.get("runs"), list
    ):
        raise ValueError("Complete fixed selection required")
    declared = value["runs"]
    if [row["run"] for row in declared] != [row["run"] for row in selected["runs"]]:
        raise ValueError("Series/selection run order differs")
    if len({name(row["run"]) for row in declared}) != len(declared):
        raise ValueError("Duplicate case-insensitive run")
    configurations = []
    for row, selection in zip(declared, selected["runs"]):
        config = validate_run(reader.json(row["configuration"]))
        if (
            config["run"] != row["run"]
            or [spec["case_id"] for spec in config["frames"]]
            != selection["ordered_case_ids"]
        ):
            raise ValueError("Run configuration differs from fixed selected case order")
        if (
            config["mass"]["calibration"]["sha256"]
            != value["job"]["calibration_sha256"]
        ):
            raise ValueError("Series cannot mix frozen calibrations")
        configurations.append(config)
    return (
        value,
        configurations,
        dict(path=str(path), sha256=expected_sha256, size_bytes=len(raw)),
    )


def series_tasks(series, configurations, *, job_root, series_asset):
    """Build one static whole-run queue task per already admitted selection."""
    require_unc(str(job_root))
    if len(configurations) != len(series["runs"]):
        raise ValueError("Every selected run needs exactly one configuration")
    job = deepcopy(series["job"])
    job["expected_runs"] = [row["run"] for row in series["runs"]]
    job["series"] = dict(series_asset)
    if job.get("precision") not in {"float32", "float64"}:
        raise ValueError("An explicit mass working precision is required")
    if re.fullmatch("[0-9a-f]{40}", job.get("commit", "")) is None:
        raise ValueError("Actual full execution Git commit required")
    lock = job["environment_lock"]
    relative = PureWindowsPath(lock["path"])
    if relative.drive or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Environment lock must remain repository-relative")
    tasks = []
    for row, config in zip(series["runs"], configurations):
        resources = deepcopy(row["resources"])
        if resources.get("backend") not in {"cpu", "cuda", "any"}:
            raise ValueError("One explicit numerical backend is required")
        if type(resources.get("threads")) is not int or resources["threads"] < 1:
            raise ValueError("Positive internal thread count required")
        task = dict(
            resources,
            pipeline=PIPELINE,
            run=config["run"],
            configuration=row["configuration"],
            commit=job["commit"],
            calibration_sha256=job["calibration_sha256"],
            precision=job["precision"],
            job_id=series["job_id"],
            series=series_asset,
        )
        if "output_directory" in resources:
            raise ValueError(
                "Outputs follow canonical job/results/run; no alternate path"
            )
        tasks.append(task)
    return job, tasks


def load_run(task):
    """Read the unchanged input config for one fixed series task."""
    if pipeline_kind(task) != PIPELINE:
        raise ValueError("This adapter requires an explicit series task")
    from ..mass.n1_pipeline import DataReader

    reader = DataReader(unc_only=True)
    config = validate_run(reader.json(task["configuration"]))
    if (
        config["run"] != task["run"]
        or config["mass"]["calibration"]["sha256"] != task["calibration_sha256"]
    ):
        raise ValueError("Task scientific run/calibration binding differs")
    return config


def bind_mass_frame(spec, native, phase, *, baseline):
    """Supply only actual new stage products to the unchanged N1 frame ABI."""
    result = deepcopy(spec["mass_spec"])
    result.update(
        case_id=spec["case_id"],
        frame_index=spec["frame_index"],
        elapsed_h=spec["elapsed_h"],
        source_binding=deepcopy(spec["source"]["source_binding"]),
        current=native["full"],
        baseline=baseline,
        labels=phase["labels"],
        classes=phase["classes"],
        area=phase["area"],
    )
    if (
        native["source_sha256"] != spec["source"]["source_sha256"]
        or phase["source_sha256"] != native["source_sha256"]
    ):
        raise ValueError("Fresh stage products identify another acquisition")
    return result


def restore_history(state, spec):
    """Require the last complete acquisition, never a predicted-label predecessor."""
    if (
        state["source"] != spec["source"]
        or state["source"]["source_sha256"] != spec["source"]["source_sha256"]
    ):
        raise ValueError("History input differs from its selected acquisition")
    if "labels" in state or "classes" in state:
        raise ValueError("Phase outputs cannot provide history features")
    return state


class SeriesInputs:
    """A fixed run's source processors, frozen phase models and acquisition state.

    The instance owns reference arrays and transient caches. It never opens
    current expected phase products, selects another run, or advances history
    until the caller has committed the full native/phase/mass frame.
    """

    def __init__(self, config, output):
        from ..mass.n1_pipeline import DataReader
        from ..phase import features
        from . import phase_inputs as loaders

        self.config, self.output = config, Path(output)
        self.reader = DataReader(config.get("data_aliases", ()), unc_only=True)
        self.aliases = {
            row["logical"]["path"]: row["physical"]["path"]
            for row in config.get("data_aliases", ())
        }
        self.phase = self.reader.json(config["phase"]["configuration"])
        self.parameters = deepcopy(self.phase["parameters"])
        self.native_parameters = self.reader.json(config["native"]["configuration"])
        self.native_parameters["source"]["asset_aliases"] = dict(self.aliases)
        self.processors = {}
        self.model_bundle = self._models()
        reference = config["phase"]["reference"]
        row = self.reader.json(reference["native_row"])
        full, valid = loaders.compose_extended_scientific(
            reference["full"],
            parameters=self.parameters["scientific"],
            aliases=self.aliases,
        )
        native_log = self.reader.array(
            reference["full"]["native_centre"]["log_linear_rgb"]
        )
        native = loaders.original_native_input(
            row,
            scientific_native_log=native_log,
            parameters=self.parameters["inputs"],
            aliases=self.aliases,
        )
        temporal = loaders.original_temporal_input(
            native,
            row,
            parameters=self.parameters["inputs"],
            ring_parameters=self.native_parameters["extended_ring"],
            aliases=self.aliases,
        )
        self.temporal_policy = loaders.temporal_policy(
            row, parameters=self.parameters["inputs"], aliases=self.aliases
        )
        self.model_bundle["baseline"] = dict(
            native_input=native,
            temporal_input=temporal,
            full_log=full,
            full_valid=valid,
        )
        self.model_bundle["native"]["flatfield_reference"] = (
            features.prepare_flatfield_reference(
                native.log,
                source_valid_mask=native.valid,
                tile_rows=self.parameters["native"]["block_rows"],
                parameters=self.parameters["features"],
            )
        )
        self.mass_baseline = reference["full"]
        self.previous = self._saved_acquisition(config["phase"]["initial_previous"])

    def _models(self):
        import numpy as np
        from ..phase import features, heads, rules
        from . import phase_inputs as loaders

        p, data = self.parameters, self.phase["model_bundles"]

        def build(role):
            item = data[role]
            docs = loaders.load_model_documents(
                item["documents"],
                requirements=item["requirements"],
                heads_parameters=p["heads"],
                aliases=self.aliases,
            )
            original = heads.HierarchicalPhaseModel.from_mappings(
                dict(
                    co2_presence=docs["co2"],
                    gas_within_co2=docs["original_generic_gas"],
                ),
                adapter=item["adapter"],
                feature_parameters=p["features"],
                parameters=p["heads"],
            )
            return docs, original

        docs, original = build("native")
        native = dict(
            original_model=original,
            gas_head=docs["g3"],
            profile=docs["original_profile"],
            regional_original=docs["original_regional"],
            regional_refit=docs["regional_refit"],
            bindings=data["native"]["documents"],
            gas_decision_settings=rules.GasDecisionSettings(
                **p["decisions"]["gas_configuration"]["selected_settings"],
                parameters=p["decisions"],
            ),
        )
        native["fixed_baseline_decode_cache"] = loaders.FixedBaselineDecodeCache(
            max_bytes=p["fixed_baseline_cache"]["maximum_bytes"],
            parameters=p["fixed_baseline_cache"],
        )
        native["fixed_baseline_decoder"] = partial(
            features.decode_native_log, parameters=p["features"]
        )
        docs, original = build("edge")
        edge = dict(
            original_model=original,
            gas_head=docs["g3"],
            regional_edge=docs["regional_refit"],
            bindings=data["edge"]["documents"],
        )
        lamp = data["placement"]
        placement = dict(
            original_model=edge["original_model"],
            pixel_model=self.reader.json(lamp["pixel_model"]),
            regional_model=self.reader.json(lamp["regional_model"]),
        )
        if lamp.get("projected_models") is not None:
            placement["joint_pixel_model"] = self.reader.json(
                lamp["projected_models"]["pixel"]
            )
            placement["projected_regional_model"] = self.reader.json(
                lamp["projected_models"]["regional"]
            )
        chain = self.phase["chain_assets"]
        receipt = self.reader.json(chain["envelopes"])
        if (
            receipt["run"] != self.config["run"]
            or receipt.get("contract") != chain["envelope_contract"]
        ):
            raise ValueError("Parent gas-envelope binding differs")
        arrays = []
        for field in ("envelope1", "envelope2"):
            row = receipt[field]
            packed = self.reader.array(row)
            arrays.append(
                np.unpackbits(packed, count=int(np.prod(row["shape_hw"])))
                .reshape(row["shape_hw"])
                .astype(bool)
            )
        binding = receipt["binding"]
        envelopes = dict(
            env1=arrays[0],
            env2=arrays[1],
            u1_elapsed_h=float(binding["u1_elapsed_h"]),
            u2_elapsed_h=float(binding["u2_elapsed_h"]),
            source_receipt=chain["envelopes"],
            source_envelopes={key: receipt[key] for key in ("envelope1", "envelope2")},
            source_u1_index=int(binding["u1_frame_index"]),
            source_u2_index=int(binding["u2_frame_index"]),
            rebuilt=False,
            transported=False,
        )
        return dict(
            native=native,
            edge=edge,
            placement=placement,
            envelopes=envelopes,
            floor_curve=self.reader.array(chain["floor"]),
            zones=self.reader.array(chain["zones"]),
        )

    def _saved_acquisition(self, record):
        """Load separately bound reference/current history arrays without inference."""
        import numpy as np
        from . import phase_inputs as loaders

        native = loaders.original_native_input(
            self.reader.json(record["native_row"]),
            scientific_native_log=self.reader.array(
                record["full"]["native_centre"]["log_linear_rgb"]
            ),
            parameters=self.parameters["inputs"],
            aliases=self.aliases,
        )
        temporal = loaders.original_temporal_input(
            native,
            self.reader.json(record["native_row"]),
            parameters=self.parameters["inputs"],
            ring_parameters=self.native_parameters["extended_ring"],
            aliases=self.aliases,
        )
        full, valid = loaders.compose_extended_scientific(
            record["full"],
            parameters=self.parameters["scientific"],
            aliases=self.aliases,
        )
        return dict(
            source=record["source"],
            full_log=full,
            full_valid=valid,
            current_h=np.asarray(record["current_h"], np.float64),
            temporal_input=temporal,
        )

    def geometry(self, row, native_valid, audit):
        """Prepare the original camera footprint and physical rig context, in order."""
        import numpy as np
        from ..geometry import context as geometry

        gp = self.parameters["geometry"]
        constants = self.phase["geometry_parameters"]
        renderer = audit["renderer"]
        observed = geometry.photographic_observation_mask(
            renderer["review_to_camera_h"],
            renderer["camera_jpeg_wh"],
            row["phase_shape_hw"][::-1],
            native_origin_xy=tuple(gp["rig_origin_xy"]),
            parameters=constants,
        )
        top = self.reader.array(row["frame_photo_top_keep"])
        area = row["area_parameters"]
        fields = self.phase["geometry_fields"]
        metric = np.asarray(area[fields["reference_to_metric"]]) @ np.linalg.inv(
            np.asarray(area[fields["reference_to_frame"]])
        )
        context = geometry.independent_rig_frame_context(
            row["reference_rig"],
            row["H_reference_rig_to_frame_native"],
            tuple(gp["native_shape_hw"]),
            feature_valid_mask=native_valid,
            canvas_wh=row["phase_shape_hw"][::-1],
            content_box_xywh=gp["source_box_xywh"],
            frame_photo_top_keep=top,
            frame_fixture_polygons_native=row["frame_fixture_polygons_native"],
            fixtures_verified_none=row.get("fixtures_verified_none", False),
            photographic_observed_mask=observed,
            H_frame_native_to_metric_pixel=metric,
            parameters=constants,
        )
        return context, observed, top

    def native(self, spec, stage):
        """Execute the explicit frozen RAW processor and camera-ring sampler once."""
        import numpy as np
        from ..native.source import RigSourceProcessor, SourceProcessor
        from ..native import extension, sidecar
        from ..mass.n1_pipeline import asset, _write_json

        task = spec["native_task"]
        geometry_row = self.reader.json(task["geometry"])
        rig_geometry_row = self.reader.json(spec["geometry"])
        kind = task["kind"]
        key = json.dumps(task["processor"], sort_keys=True)
        if key not in self.processors:
            value = task["processor"]
            if kind == "rig":
                self.processors[key] = RigSourceProcessor(
                    self.reader.path(value["native_manifest"]),
                    parameters=self.native_parameters,
                )
            else:
                self.processors[key] = SourceProcessor(
                    self.reader.json(value["source_row"]),
                    calibration_manifest_path=self.reader.path(
                        value["calibration_manifest"]
                    ),
                    photo_aggregate_path=self.reader.path(value["photo_aggregate"]),
                    enclosed_hole_policy=value["enclosed_hole_policy"],
                    display_mode=value["display_mode"],
                    isolated_invalid_policy=value["isolated_invalid_policy"],
                    parameters=self.native_parameters,
                )
        processor = self.processors[key]
        if kind == "rig":
            log, audit = processor.correct(
                geometry_row,
                parent_frame_index=task["parent_frame_index"],
                parent_reference=task["parent_reference"],
            )
        else:
            arguments = deepcopy(task["arguments"])
            for field in ("camera_jpeg", "camera_arw", "review_jpeg"):
                arguments[field] = self.reader.path(arguments[field])
            corrected = processor.correct(**arguments)
            log, certificate = corrected.log_linear_rgb, corrected.certificate
            fields = self.config["native"]["source_certificate_fields"]
            audit = dict(
                processor=processor.identity(),
                renderer=certificate[fields["renderer"]],
                photometry=certificate[fields["photometry"]],
                source_certificate=certificate,
                sources={
                    "camera_arw": dict(
                        task["arguments"]["camera_arw"],
                        path=str(arguments["camera_arw"]),
                    ),
                    "camera_jpeg": dict(
                        task["arguments"]["camera_jpeg"],
                        path=str(arguments["camera_jpeg"]),
                    ),
                    "native_review_jpeg": dict(
                        task["arguments"]["review_jpeg"],
                        path=str(arguments["review_jpeg"]),
                    ),
                },
            )
            h_key = self.native_parameters["extension"]["reference_to_frame_h_key"]
            audit[h_key] = deepcopy(task["reference_to_frame_h"])
            # Adapt the fixed SourceProcessor object protocol, not its maths.
            # The sparse extension uses this actual processor's frozen fit,
            # reference, light field and sensor, with its genuine identity.
            processor_for_ring = SimpleNamespace(
                processor=processor, identity=processor.identity()
            )
        valid = np.isfinite(log).all(0)
        stage.mkdir(parents=True)
        log_path, valid_path = stage / "native_log.npy", stage / "native_valid.npy"
        sidecar.write_log_linear(log_path, log)
        sidecar.write_log_linear(valid_path, valid)
        audit_asset = _write_json(stage / "native_audit.json", audit)
        context, observed, top = self.geometry(rig_geometry_row, valid, audit)
        full, full_valid, ring_audit = extension.extend_prepared_source(
            log,
            audit,
            processor if kind == "rig" else processor_for_ring,
            context.rig_domain & observed,
            parameters=self.native_parameters,
        )
        _write_json(stage / "ring_audit.json", ring_audit)
        centre = dict(
            native_shape_hw=list(log.shape[1:]),
            log_linear_rgb=asset(log_path),
            valid_mask=asset(valid_path),
        )
        windows = []
        for index, window in enumerate(ring_audit["windows"]):
            a, b, c, d = window["canvas_xyxy"]
            path = stage / ("ring_" + str(index) + ".npy")
            sidecar.write_log_linear(path, full[:, b:d, a:c])
            windows.append(dict(canvas_xyxy=[a, b, c, d], log_linear_rgb=asset(path)))
        sp = self.parameters["scientific"]
        current = dict(
            contract=sp["contract"],
            complete=True,
            storage=sp["storage"],
            source_sha256=spec["source"]["source_sha256"],
            phase_shape_hw=list(full.shape[1:]),
            native_origin_xy=list(self.parameters["geometry"]["rig_origin_xy"]),
            native_centre=centre,
            ring_windows=windows,
            binding=dict(geometry=rig_geometry_row, producer_audit=audit_asset),
        )
        native_record = dict(
            complete=True,
            source_sha256=spec["source"]["source_sha256"],
            full=current,
            audit=audit_asset,
            processor_kind=kind,
        )
        native_record_asset = _write_json(stage / "native.json", native_record)
        return dict(
            log=log,
            valid=valid,
            full_log=full,
            full_valid=full_valid,
            audit=audit,
            geometry_row=rig_geometry_row,
            native_geometry_row=geometry_row,
            context=context,
            observed=observed,
            top=top,
            record=native_record,
            record_asset=native_record_asset,
        )

    def frame(self, spec, native):
        """Assemble fresh phase inputs with exact historical/reference/area roles."""
        import numpy as np
        from ..native import support
        from ..phase import inputs, domain
        from ..geometry import areas, barrier
        from . import phase_inputs as loaders

        roles, source, p = spec["phase_roles"], spec["source"], self.parameters
        if self.previous["source"]["source_sha256"] != source["previous_source_sha256"]:
            raise ValueError("Fresh frame has another acquisition predecessor")
        probability = None
        if roles.get("native_row") is not None:
            row = self.reader.json(roles["native_row"])
            probability = self.reader.json(roles["probability_receipt"])
            photo, photo_proof = loaders.original_photo_sand(
                probability, parameters=p["inputs"], aliases=self.aliases
            )
            masked = loaders.original_native_input(
                row,
                scientific_native_log=native["log"],
                parameters=p["inputs"],
                aliases=self.aliases,
            )
            temporal = loaders.original_temporal_input(
                masked,
                row,
                parameters=p["inputs"],
                ring_parameters=self.native_parameters["extended_ring"],
                aliases=self.aliases,
            )
        else:
            photo_row = roles["photo_sand_reference"]
            sand = support.source_photo_grid(
                self.reader.path(photo_row),
                photo_row["sha256"],
                parameters=self.native_parameters["support"],
            )
            photo = support.project_source_photo_safe_core(
                sand,
                roles["reference_to_current_h"],
                shape_hw=tuple(p["geometry"]["native_shape_hw"]),
                guard_px=self.native_parameters["support"]["guard_px"],
                enclosed_hole_policy=self.native_parameters["support"]["fill_policy"],
                parameters=self.native_parameters["support"],
            )
            photo_proof = deepcopy(roles["photo_proof"])
            masked = inputs.new_native_input(
                native["log"],
                native["valid"],
                photo,
                source_binding=source["source_binding"],
                photo_mask_binding=photo_proof,
                parameters=p["inputs"],
            )
            temporal = inputs.new_temporal_input(
                masked,
                native["full_log"],
                native_box_xywh=p["geometry"]["source_box_xywh"],
                frozen_policy=self.temporal_policy,
                parameters=p["inputs"],
            )
        area_data = roles["area"]
        if area_data["role"] == "measured_parent":
            original = self.reader.array(area_data["historical"])
            measured = self.reader.array(area_data["measured"])
            rebuilt, proof = areas.apply_measured_area_revision(
                original,
                area_data["factor"],
                source_binding=area_data["historical"],
                parameters=self.phase["area_parameters"],
            )
            if measured.dtype != rebuilt.dtype or not np.array_equal(measured, rebuilt):
                raise ValueError(
                    "Measured parent area differs from its exact rounding recipe"
                )
        else:
            original = native["context"].pixel_area_mm2
            measured = areas.apply_selected_area_scale(original, area_data["factor"])
            proof = deepcopy(area_data["revision"])
        static = roles["static"]
        values = self.reader.archive(
            static["components"],
            ("open_components", "barrier_mask", "reference_domain"),
        )
        components, ids, static_proof = barrier.prepare_static_components(
            values,
            native["context"].rig_domain,
            native_shape_hw=p["geometry"]["native_shape_hw"],
            source_box_xywh=p["geometry"]["source_box_xywh"],
            source_to_frame_h=roles["reference_to_current_h"],
            source_ids=static["source_ids"],
            unchanged_source=static["unchanged_source"],
            source_binding=static["components"],
            parameters=self.phase["barrier_parameters"],
        )
        prior = None
        if roles.get("retained_source_cache") is not None:
            prior = loaders.prepare_retained_source_cache(
                roles["retained_source_cache"],
                run=self.config["run"],
                frame_index=roles["retained_frame_index"],
                current_source_sha256=source["source_sha256"],
                current_log=masked.log,
                allowed_domain=domain.crop_native_canvas(
                    native["context"].rig_domain, p["geometry"]["source_box_xywh"]
                ),
                producer_audit=native["audit"],
                recompute_old_domain_ring=False,
                recompute_old_domain_interpolation_footprint=False,
                parameters=p["retained"],
                aliases=self.aliases,
            )
        encoding = (
            None
            if probability is None
            else loaders.probability_encoding_contract(
                probability, masked.valid.shape, parameters=p["retained"]
            )
        )
        if encoding != roles["encoding_contract"]:
            raise ValueError("Matched input encoding receipt differs")
        spacing = roles["probability_spacing_m"]
        if probability is not None and list(
            probability["metadata"]["spacing_m"]
        ) != list(spacing):
            raise ValueError("Matched probability axes differ")
        return dict(
            source=source,
            native_input=masked,
            temporal_input=temporal,
            full_log=native["full_log"],
            full_valid=native["full_valid"],
            current_h=np.asarray(roles["current_h"], np.float64),
            scientific=native["record"],
            context=native["context"],
            observed=native["observed"],
            original_area=original,
            pixel_area_mm2=measured,
            components=components,
            source_component_ids=list(ids),
            barrier_proof=static_proof,
            reference_to_current_h=np.asarray(
                roles["reference_to_current_h"], np.float64
            ),
            top_keep=native["top"],
            light_epoch_id=roles["light_epoch_id"],
            placement_metadata=deepcopy(roles["placement_metadata"]),
            inherited_spacing_m=roles["inherited_spacing_m"],
            probability_spacing_m=spacing,
            encoding_contract=encoding,
            labelled_probability_encoding=probability is not None
            and probability["binding"]["labelled"] is True,
            prior_cache=prior,
            physical_scale_revision=proof,
            history_policy=self.config["history_policy"],
            input_provenance=dict(
                source=native["record_asset"],
                historical_native_input=masked.provenance,
                previous_source_sha256=source["previous_source_sha256"],
                photo_domain=photo_proof,
                previous_native_temporal=self.previous["temporal_input"].provenance,
            ),
        )

    def advance(self, frame):
        """Advance only a successfully committed selected acquisition."""
        from ..phase.history import advance_previous

        self.previous = advance_previous(
            {
                key: frame[key]
                for key in (
                    "source",
                    "full_log",
                    "full_valid",
                    "current_h",
                    "temporal_input",
                )
            },
            self.previous,
            selected=frame["source"]["selected"],
            baseline=frame["source"].get("baseline", False),
        )

    def close(self):
        """Release fixed baseline caches and all run-owned array references."""
        self.model_bundle["native"]["fixed_baseline_decode_cache"].clear()
        self.processors.clear()
        self.model_bundle.clear()
        self.previous = None
