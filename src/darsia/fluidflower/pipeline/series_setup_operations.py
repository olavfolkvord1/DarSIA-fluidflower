"""Original corrected Apply and RAW Build under admitted operation writers.

The existing worker owns claims, capacity, publication and retirement. This
module neither enqueues tasks nor adopts geometry or qualifies training data.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path


def _read(bound):
    """Use the original bounded, version-checked reader for exact JSON controls."""
    from .capacity import read_bound
    return read_bound(bound)[0]


@contextmanager
def _native_parameters(bound):
    """Bind the original parameter JSON before loading numerical entry points."""
    from ..native.config import configure_parameters, reset_parameters
    document = _read(bound)
    values = dict(DARSIA_NATIVE_CONFIGURATION=str(Path(bound["path"]).resolve()),
                  DARSIA_NATIVE_CONFIGURATION_SHA256=bound["sha256"])
    previous = {key: os.environ.get(key) for key in values}
    token = configure_parameters(document)
    os.environ.update(values)
    try:
        yield
    finally:
        reset_parameters(token)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sink():
    """Require the actual active bounded metadata writer and live reservation."""
    from .series_setup_sink import SetupMetadataSink, current_sink
    sink = current_sink()
    if not isinstance(sink, SetupMetadataSink):
        raise ValueError("Setup execution requires its active admitted metadata sink")
    return sink


def _write(path, document):
    """Keep immutable control bytes under the active whole-operation ceiling."""
    from ..rig_raw_build import _descriptor
    sink, path = _sink(), Path(path)
    raw = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    sink.admission.require_write(path)
    if path.exists():
        if path.stat().st_size != len(raw):
            raise ValueError("Existing Setup control differs; all artifacts were retained")
        with path.open("rb") as stream:
            existing = stream.read(len(raw) + 1)
        if existing != raw:
            raise ValueError("Existing Setup control differs; all artifacts were retained")
    else:
        sink.mkdir(path.parent, parents=True, exist_ok=True)
        sink.write_bytes(path, raw)
    return _descriptor(path)


def _geometry_sources(project, current, runs):
    """Retain sources returned by original saved-geometry readers, without guessing."""
    from .. import corrected_setup as original
    config, result = current["configuration"], []
    for run in runs:
        registered = original._registered_binding(project.root, current, config, run)
        if registered is None:
            original._require_unedited_adoption(current, config)
            fields = original.selected_fields(config, run)
            row = config["frame_binding"]["runs"][run]
            guarded = [fields[key] for key in ("shape_corrected_baseline", "labels", "facies", "depth")]
            guarded += [row["configured_depth_map"], original._revision_properties(config, row, None)]
        else:
            guarded = [*registered["guarded_sources"], *registered["files"].values()]
        for bound in guarded:
            original._verified(bound)
        result.append(dict(run_id=run, registered=deepcopy(registered), guarded_sources=deepcopy(guarded)))
    return result


def _reverify(records):
    """Recheck the exact retained scientific source descriptors at completion."""
    from .. import corrected_setup as original
    for record in records:
        for bound in record["guarded_sources"]:
            original._verified(bound)


def run_apply(config, plan, revision, project, progress=None):
    """Apply the reference first, then capture requests only after all postimages."""
    from .. import corrected_setup as original
    from ..corrected_setup_all_runs import apply_order
    from ..rig_raw_build import _descriptor, capture_configuration_sources, read_build_request
    from ..series_geometry import SeriesRunInputs
    from .series_setup_protocol import APPLY_COMPLETE_FORMAT
    from .run_io import current_binding
    from .series_setup_run_io import original_geometry_sources, original_apply_run, original_configuration_sources
    _sink()
    frame = revision["configuration"]["frame_binding"]
    if frame["display_reference_run"] != plan["display_reference_run"]:
        raise ValueError("The Setup display reference differs from its saved geometry")
    guarded = (original_geometry_sources(project, revision, plan["runs"])
               if current_binding() is not None else _geometry_sources(project, revision, plan["runs"]))
    completed = {}
    for run in apply_order(plan["runs"], plan["display_reference_run"]):
        if progress:
            progress(dict(state="setup_progress", operation="apply", run=run, step="apply"))
        reference = SeriesRunInputs(run, Path(frame["runs"][run]["configuration"]["path"]))
        settings = next(build for build in plan["build_plans"] if build["run_id"] == run).get("raw_setup_settings")
        if current_binding() is not None:
            exported = original_apply_run(project, revision, reference, raw_setup_settings=settings)
        else:
            exported = original.export_corrected_setup(project, reference=reference)
            options = {} if settings is None else dict(raw_setup_settings=settings)
            original.apply_corrected_selected_run(project, exported, rebuild_facies_by_name=True, **options)
        applied = original._apply_completion(exported.destination, exported.receipt)
        if applied is None:
            raise ValueError("The original Apply did not produce a complete per-run record")
        completed[run] = dict(run_id=run, export=_descriptor(exported.destination / "export.json"),
            apply_completion=_descriptor(exported.destination / "apply-complete.json"))
    # The common TOML can be changed by the reference. No future request is
    # captured until every original selected-run transaction has completed.
    for build in plan["build_plans"]:
        paths = [Path(path) for path in build["configuration_paths"]]
        if current_binding() is not None:
            captured = original_configuration_sources(project, revision, paths)
        else:
            original.require_corrected_export_for_build(paths)
            captured = list(capture_configuration_sources(paths))
        completed[build["run_id"]]["configuration_files"] = captured
    _reverify(guarded)
    coordinator = dict(format=APPLY_COMPLETE_FORMAT, complete=True,
        **{key: deepcopy(config[key]) for key in
           ("setup_plan", "adopted_revision", "project_manifest", "series_manifest")},
        revision_id=revision["revision_id"], current_sha256=config["adopted_revision"]["sha256"],
        runs=[completed[run] for run in plan["runs"]], Build_run=False, optical_calibration_run=False)
    root = Path(plan["control_root"])
    coordinator_bound = _write(root / "apply-complete.json", coordinator)
    children = []
    for build in plan["build_plans"]:
        run = build["run_id"]
        request = dict(deepcopy(build["request_template"]),
                       configuration_files=deepcopy(completed[run]["configuration_files"]))
        request_bound = _write(root / "build" / run / "raw-build-request.json", request)
        read_build_request(request_bound, artifact_kind="raw_rig_baseline")
        operation = dict(format=config["format"], run=run, operation="build",
            **{key: deepcopy(config[key]) for key in
               ("project_root", "project_manifest", "adopted_revision", "series_manifest", "setup_plan")},
            **{key: deepcopy(build[key]) for key in
               ("native_parameters", "target_machine", "metadata_output", "output_scope")},
            configuration_files=deepcopy(completed[run]["configuration_files"]),
            raw_build_request=request_bound, apply_completion=completed[run]["apply_completion"],
            coordinator_completion=coordinator_bound)
        from .series_setup_protocol import validate_setup_operation
        validate_setup_operation(operation)
        operation_bound = _write(root / "build" / run / "operation.json", operation)
        children.append(dict(run_id=run, configuration=operation_bound,
                             raw_build_request=request_bound, target_machine=build["target_machine"]))
    _reverify(guarded)
    return dict(coordinator_completion=coordinator_bound, children=children,
                guarded_geometry=guarded, Build_run=False, optical_calibration_run=False)


def run_build(config, plan, revision, project, raw_request, progress=None):
    """Call the original full RAW setup with its exact post-Apply request."""
    from .. import corrected_setup as original
    from ..rig_raw_build import CERTIFICATE_NAME, _descriptor, read_build_request
    from .run_io import current_binding
    from .series_setup_run_io import original_geometry_sources, original_build
    _sink()
    guarded = (original_geometry_sources(project, revision, [config["run"]])
               if current_binding() is not None else _geometry_sources(project, revision, [config["run"]]))
    request, _pins = read_build_request(config["raw_build_request"], artifact_kind="raw_rig_baseline")
    if request != raw_request:
        raise ValueError("The original RAW request differs from its admitted controls")
    paths = [Path(row["path"]) for row in config["configuration_files"]]
    if progress:
        progress(dict(state="setup_progress", operation="build", run=config["run"], step="build"))
    if current_binding() is not None:
        original_build(project, revision, paths, config["raw_build_request"])
    else:
        original.require_corrected_export_for_build(paths)
        from ..setup_rig import setup_rig
        from darsia.presets.workflows.rig import Rig
        setup_rig(Rig, paths, show=False, raw_build_request=config["raw_build_request"])
    certificate = _descriptor(Path(request["output"]["destination"]) / CERTIFICATE_NAME)
    evidence = _read(certificate)
    _reverify(guarded)
    read_build_request(config["raw_build_request"], artifact_kind="raw_rig_baseline")
    return dict(guarded_geometry=guarded, raw_build_request=deepcopy(config["raw_build_request"]),
        configuration_files=deepcopy(config["configuration_files"]), baseline_renderer=certificate,
        produced_files=deepcopy(evidence["outputs"]), certificate_scope=evidence["certificate_scope"],
        complete_rig_model=evidence["complete_rig_model"],
        production_qualification_claimed=evidence["production_qualification_claimed"],
        training_context_eligibility_claimed=False)


def execute_setup(task, output, backend, precision, progress=None):
    """Execute one original operation under the worker's exact RunIO and sink."""
    from .run_io import for_task, task_without_resume
    from .series_setup_inputs import _same
    from .alias_job import _manifest
    from .series_setup_protocol import validate_setup_operation, validate_setup_plan
    binding = for_task(task)
    if binding is None:
        raise ValueError("Setup requires its exact active per-run I/O binding")
    binding.require_output(output)
    sink = _sink()
    sink.admission.require_write(Path(output))
    config = validate_setup_operation(_read(binding.configuration_descriptor), task=task)
    if (not _same(task_without_resume(sink.admission.task), task_without_resume(task))
            or not _same(sink.admission.configuration, config)):
        raise ValueError("Setup sink and RunIO belong to different operation admissions")
    if backend != "cpu" or precision != task["precision"]:
        raise ValueError("Setup must use its explicit CPU worker and task precision")
    controls = binding.setup_controls
    if controls["scope"] != "series-setup-controls-only":
        raise ValueError("Setup requires its original named control catalogue")
    plan = validate_setup_plan(_read(config["setup_plan"]))
    revision = _read(config["adopted_revision"])
    if not _same(plan, controls["plan"]) or not _same(revision, controls["revision"]):
        raise ValueError("Setup controls changed after admitted replay")
    for key in ("project_manifest", "native_parameters"):
        _read(config[key])
    _manifest(config["series_manifest"])
    with _native_parameters(config["native_parameters"]):
        from ..project import FluidFlowerProject
        project = FluidFlowerProject.load(Path(config["project_root"]))
        if config["operation"] == "apply":
            operation = run_apply(config, plan, revision, project, progress)
        else:
            operation = run_build(config, plan, revision, project, controls["raw_request"], progress)
        for key in ("project_manifest", "adopted_revision", "native_parameters"):
            _read(config[key])
        _manifest(config["series_manifest"])
        result = dict(format="darsia-series-setup-result-v1", complete=True, pipeline="series-setup",
            run=task["run"], operation=config["operation"], backend=backend, precision=precision,
            setup_plan=deepcopy(config["setup_plan"]), worker_references_closed=True, **operation)
        receipt = _write(Path(output) / "operation_result.json", result)
    return dict(result, files={"operation_result.json": receipt})
