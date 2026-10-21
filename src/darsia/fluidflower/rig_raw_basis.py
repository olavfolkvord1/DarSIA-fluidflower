"""Fresh shape-only RAW bootstrap through the original workflow operations.

No map, placement, adoption, full-rig or configuration revision is produced.
All acquisition/configuration/protocol paths enter through one pinned request.
"""

from pathlib import Path

from .rig_raw_build import (
    _descriptor, _verify, begin_raw_rig_build, capture_configuration_sources,
    read_build_request,
)


def protocol_sources(protocols):
    """The exact files/sheets/folders consumed by ProtocolledExperiment."""
    result = []

    def add(role, value, folder=None):
        if value is None:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                add(role, item, key)
            return
        path, sheet = value if isinstance(value, (tuple, list)) else (value, None)
        result.append(dict(role=role, folder=None if folder is None else str(folder),
                           sheet=sheet, source=_descriptor(path)))

    for role in ("imaging", "injection", "pressure_temperature", "blacklist"):
        add(role, getattr(protocols, role))
    return result


def require_protocol_sources(request, protocols):
    """Require the exact original protocol roles, files and sheets."""
    actual = protocol_sources(protocols)
    if actual != request["protocol_inputs"]:
        raise ValueError("RAW operation protocols differ from the exact bound files, sheets or folders")
    if not all(any(row["role"] == role for row in actual)
               for role in ("imaging", "injection", "pressure_temperature")):
        raise ValueError("RAW operation requires its original imaging, injection and pressure protocols")


def build_raw_shape_basis(request_binding):
    """Run the original setup_reading and Rig.save for one bound local basis.

    ``request_binding`` is the exact path/SHA/size of
    ``darsia-raw-shape-basis-request-v1`` DATA. The original TOMLs remain
    immutable; only the request supplies the explicit camera override.
    """
    import darsia
    from darsia._opencv_runtime import scientific_thread, completed_frame
    from darsia.presets.workflows.config.data import DataConfig
    from darsia.presets.workflows.config.protocols import ProtocolsConfig
    from darsia.presets.workflows.config.corrections import CorrectionsConfig
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    from darsia.presets.workflows.rig import Rig

    request, pins = read_build_request(request_binding, artifact_kind="raw_shape_basis")
    paths = [Path(row["path"]) for row in request["configuration_files"]]
    sources = capture_configuration_sources(paths)
    section = _get_section_from_toml(paths, "data")
    if section.get("use_cache", False) is not False or section.get("raw_camera") is not None:
        raise ValueError("Shape bootstrap requires unchanged camera-free TOMLs and disabled data cache")
    # require_results=True avoids the legacy loader's results-directory writer.
    data = DataConfig().load(paths, require_data=False, require_results=True)
    protocols = ProtocolsConfig().load(paths)
    corrections = CorrectionsConfig().load(paths)
    require_protocol_sources(request, protocols)
    if _descriptor(data.baseline) != request["baseline"]:
        raise ValueError("The original configuration baseline differs from the bound RAW request")
    operation = begin_raw_rig_build(paths, data.baseline,
        request["operation_overrides"]["data.raw_camera"], request["output"]["destination"],
        configuration_sources=sources, request_binding=request_binding,
        artifact_kind="raw_shape_basis")
    _verify(pins)
    before = scientific_thread("raw-shape-basis:before")
    experiment = darsia.ProtocolledExperiment(data=data.data,
        imaging_protocol=protocols.imaging, injection_protocol=protocols.injection,
        pressure_temperature_protocol=protocols.pressure_temperature,
        blacklist_protocol=protocols.blacklist)
    rig = Rig()
    rig.baseline_path = data.baseline
    rig.reference_date = experiment.experiment_start
    log = operation.destination / "figures"
    operation.begin_setup(rig, log)
    rig.setup_reading(data.baseline, experiment, corrections_config=corrections,
                      log=log, show_plot=False)
    rig.color_corrections = []
    rig.save(operation.destination)
    rig._raw_basis_opencv = completed_frame(before, "raw-shape-basis:after")
    return operation.publish(rig)
