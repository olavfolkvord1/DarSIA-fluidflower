"""Replay the declared Setup control scope without future configuration pins.

The original series is a source identity, not a fabricated analysis series with
future Build postimages. The immutable Setup plan and selected operation join
those identities. Scientific geometry admission remains the original reader's
separate post-claim boundary and is never claimed by this control catalogue.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path

from .run_io import _Replay
from .store_runtime import InputHashMismatch


def replay_setup_controls(binding):
    """Replay every named control from the verified queue claim exactly once."""
    from .job_inputs import _json_bytes
    from .series_setup_inputs import SCOPE
    from .series_setup_protocol import validate_setup_operation

    task, config = binding.task, binding.configuration
    validate_setup_operation(config, task=task)
    replay = _Replay(binding, ())
    for use, reference in (("run.execution_series", task["series"]),
                           ("run.series_manifest", task["series_manifest"])):
        row = binding.verified.for_use(use)
        if row.reference != reference:
            raise InputHashMismatch("Setup original series identity differs: " + use)
        binding._verified_bytes(row)
        replay.add(dict(path=str(row.path), sha256=reference["sha256"],
                        size_bytes=reference["bytes"]), "configuration", use)
        if (reference["sha256"] != config["series_manifest"]["sha256"]
                or reference["bytes"] != config["series_manifest"]["size_bytes"]):
            raise InputHashMismatch("Setup controls name another original series")
    actual = replay.collect(binding.configuration_descriptor, config)
    for field in ("format", "complete", "run", "configuration", "inputs",
                  "payload_hashes_verified", "numerical_calls_started"):
        if actual[field] != task["input_catalog"][field]:
            raise InputHashMismatch("Setup named controls differ from the prepared catalogue")
    expected = {use for item in task["input_catalog"]["inputs"] for use in item["uses"]}
    if set(replay.used) != expected:
        raise InputHashMismatch("Setup control catalogue has unconsumed or missing roles")
    if binding.verified.for_use("setup.plan").reference != task["setup_plan"]:
        raise InputHashMismatch("Setup task names another immutable plan")
    native = binding._metadata_bytes(binding._descriptor_location(config["native_parameters"]))
    binding._admit_native_code(_json_bytes(native))
    controls = deepcopy(replay.setup_controls)
    if controls["scope"] != SCOPE:
        raise InputHashMismatch("Setup catalogue changed its control-only scope")
    binding.setup_controls = controls
    binding.catalogue_scope = SCOPE
    replay.bind_declared_aliases()
    binding._replayed = True
    return binding


class _OriginalSetupReader:
    """Keep original exact-path data proofs and the admitted native code lease."""

    def __init__(self, binding):
        """Capture only the already replayed operation's code identities."""
        self.binding = binding

    @property
    def series_setup_original_reader(self):
        """Retain original series-relative TOML paths without user remapping."""
        return True

    def native_input_digest(self, path):
        """Use canonical Git identities only for explicitly admitted code."""
        from .run_io import _key
        if _key(path) in self.binding.code_locations:
            return self.binding.native_input_digest(path)
        return None

    def native_metadata_bytes(self, path, expected_sha256):
        """Leave original bound JSON readers responsible for their exact bytes."""
        return None

    def asset_path(self, path):
        """Keep the original locator without relocation or inferred discovery."""
        return Path(path)

    def physical_descriptor(self, descriptor):
        """Leave SHA and size checks with the original scientific reader."""
        return deepcopy(descriptor)

    def reader_verifications(self):
        """Never represent control admission as verified scientific leaves."""
        return {}


@contextmanager
def _original_scope(project, revision):
    """Bind only the fixed original entry points below after real claim admission.

    Scientific inputs are deliberately absent from the control catalogue. The
    original saved-revision readers discover and verify their own explicit
    descriptors, without relocation. Their returned guards remain separate
    completion evidence; none are added to the general RunIO input closure.
    """
    from . import run_io
    from .capacity import read_bound
    from .series_setup_inputs import SCOPE, _same
    from .series_setup_sink import SetupMetadataSink, current_sink
    binding, sink = run_io.current_binding(), current_sink()
    if (not isinstance(binding, run_io.RunIOBinding)
            or run_io.for_task(binding.task) is not binding
            or not isinstance(sink, SetupMetadataSink) or sink.closed
            or not _same(run_io.task_without_resume(sink.admission.task), binding.task)
            or not _same(sink.admission.configuration, binding.configuration)
            or getattr(binding, "catalogue_scope", None) != SCOPE):
        raise InputHashMismatch("Original Setup readers need their exact active operation admission")
    config = binding.configuration
    if (Path(project.root).resolve() != Path(config["project_root"]).resolve()
            or not _same(read_bound(config["adopted_revision"])[0], revision)
            or not _same(binding.setup_controls["revision"], revision)):
        raise InputHashMismatch("Original Setup readers name another admitted saved revision")
    if any(key != run_io._key(row.path) for key, row in binding.locations.items()):
        raise InputHashMismatch("Original Setup scientific readers require unchanged control locations")
    sink.admission.require_write(binding.output)
    for row in binding.verified.rows:
        binding._input_digest(row)
    for code in binding.code_locations.values():
        binding._checked_code_digest(code)
    token = run_io._active.set(_OriginalSetupReader(binding))
    try:
        yield
    finally:
        run_io._active.reset(token)
        for code in binding.code_locations.values():
            binding._checked_code_digest(code)


def original_geometry_sources(project, revision, runs):
    """Run only the original saved-geometry proof and retain its returned guards."""
    from .series_setup_operations import _geometry_sources
    with _original_scope(project, revision):
        return _geometry_sources(project, revision, runs)


def original_apply_run(project, revision, reference, *, raw_setup_settings=None):
    """Export and Apply one saved run through the original guarded transaction."""
    from .. import corrected_setup as original
    with _original_scope(project, revision):
        exported = original.export_corrected_setup(project, reference=reference)
        options = {} if raw_setup_settings is None else dict(raw_setup_settings=raw_setup_settings)
        original.apply_corrected_selected_run(project, exported, rebuild_facies_by_name=True, **options)
        return exported


def original_configuration_sources(project, revision, paths):
    """Verify actual Apply completion before capturing its original TOML bytes."""
    from .. import corrected_setup as original
    from ..rig_raw_build import capture_configuration_sources
    with _original_scope(project, revision):
        original.require_corrected_export_for_build(paths)
        return list(capture_configuration_sources(paths))


def original_build(project, revision, paths, raw_request):
    """Call only the original full RAW setup with its exact captured request."""
    from .. import corrected_setup as original
    from ..setup_rig import setup_rig
    from darsia.presets.workflows.rig import Rig
    with _original_scope(project, revision):
        original.require_corrected_export_for_build(paths)
        return setup_rig(Rig, paths, show=False, raw_build_request=raw_request)
