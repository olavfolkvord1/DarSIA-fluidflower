"""Real tiny Apply/CAS and capped controls; no RAW execution qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import corrected_setup as original
from darsia.fluidflower import rig_raw_build
from darsia.fluidflower.pipeline import series_setup_operations as operations
from darsia.fluidflower.pipeline.series_setup_protocol import FORMAT, PLAN_FORMAT, validate_setup_plan
from darsia.fluidflower.pipeline.series_setup_sink import SetupMetadataSink
from .test_series_setup_admission import _admission
from .test_painted_setup_apply import named_setup  # noqa: F401
from .test_painted_adoption_table import imported, synthetic_payload_admission, prepared_painting, _saved  # noqa: F401


def _file(path, raw):
    """Bind only explicitly created private fixture bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _document(path, value):
    """Save one explicit strict JSON control for the original request reader."""
    return _file(path, (json.dumps(value, allow_nan=False) + "\n").encode())


@pytest.fixture
def applied_plan(named_setup, tmp_path, monkeypatch):
    """Prepare genuine tiny saved geometry with opaque, fully bound RAW controls."""
    data = named_setup
    prepared, *_ = prepared_painting(data, tmp_path)
    _saved(data, prepared)
    current = original.GeometryRevisionStore(data.project).current()
    revision_path = data.project.root / "results/geometry" / current["revision_id"] / "revision.json"
    revision = original._descriptor(revision_path)
    project = _document(data.project.root / "series.json", data.project.manifest)
    native = _document(tmp_path / "native-parameters.json", {"fixture_parameter": "explicit"})
    series = original._descriptor(data.series / "series.toml")
    baseline = _file(tmp_path / "acquisitions/baseline.ARW", b"synthetic RAW; no decode")
    profile = _document(tmp_path / "profile.json", {"fixture_profile": "opaque"})
    basis = _document(tmp_path / "basis.json", {"fixture_basis": "not a renderer qualification"})
    root = tmp_path / "controls"
    root.mkdir()
    caps = dict(maximum_output_bytes=8 * 1024**2, maximum_file_bytes=8 * 1024**2)
    scope = dict(metadata_roots=[str(data.project.root), str(data.series / "config"), str(root)], scratch_roots=[])
    builds = []
    for run in ("two", "one"):
        template = dict(format=rig_raw_build.REQUEST_FORMAT, run_id=run,
            series_context=dict(root=str(data.series), manifest=series), baseline=baseline,
            profile=profile, operation_overrides={}, protocol_inputs=[], shape_basis_certificate=basis,
            output=dict(destination=str(tmp_path / "raw-output" / run), intended_root=str(tmp_path / "raw-output"),
                        forbidden_raw_roots=[str(tmp_path / "acquisitions")], **caps))
        builds.append(dict(run_id=run, target_machine="fixture", native_parameters=native,
            configuration_paths=[str(data.series / "config/common.toml"),
                                 str(data.series / "config/merged" / (run + ".toml"))],
            request_template=template, metadata_output=caps, output_scope=scope))
    plan = dict(format=PLAN_FORMAT, project_root=str(data.project.root), project_manifest=project,
        adopted_revision=revision, series_manifest=series, native_parameters=native,
        runs=["two", "one"], display_reference_run="one", control_root=str(root), build_plans=builds,
        target_machine="fixture", metadata_output=caps, output_scope=scope)
    validate_setup_plan(plan)
    plan_bound = _document(root / "plan.json", plan)
    config = dict(format=FORMAT, run="series-apply", operation="apply", setup_plan=plan_bound,
        **{key: deepcopy(plan[key]) for key in ("project_root", "project_manifest", "adopted_revision",
            "series_manifest", "native_parameters", "runs", "build_plans", "target_machine",
            "metadata_output", "output_scope")})
    admission = _admission(tmp_path / "capacity", monkeypatch, metadata_cap=caps["maximum_output_bytes"],
                           metadata_roots=[data.project.root, data.series / "config", root])
    return SimpleNamespace(data=data, config=config, plan=plan, revision=current,
                           sink=SetupMetadataSink(admission), root=root)


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_original_apply_precedes_capture_and_retains_plan_order(applied_plan, monkeypatch):
    """Actual CAS replay precedes each full request; the shared postimage is stable."""
    case = applied_plan
    expected_root = case.data.project.root / "exports/native_setup" / case.revision["revision_id"]
    capture = rig_raw_build.capture_configuration_sources
    captures = []

    def capture_after_apply(paths):
        assert (expected_root / "apply-complete.json").is_file()
        assert (expected_root / "runs/two/apply-complete.json").is_file()
        captures.append([Path(path) for path in paths])
        return capture(paths)

    monkeypatch.setattr(rig_raw_build, "capture_configuration_sources", capture_after_apply)
    before = {path: path.read_bytes() for path in (case.data.series / "config").rglob("*.toml")}
    with case.sink.activate():
        result = operations.run_apply(case.config, case.plan, case.revision, case.data.project)
    coordinator = operations._read(result["coordinator_completion"])
    assert [row["run_id"] for row in coordinator["runs"]] == ["two", "one"]
    assert [row["run_id"] for row in result["children"]] == ["two", "one"]
    assert coordinator["Build_run"] is False and coordinator["optical_calibration_run"] is False
    assert len(captures) == 2 and all(paths[0].name == "common.toml" for paths in captures)
    for child, template in zip(result["children"], case.plan["build_plans"]):
        request, _ = rig_raw_build.read_build_request(child["raw_build_request"], artifact_kind="raw_rig_baseline")
        assert {key: value for key, value in request.items() if key != "configuration_files"} == template["request_template"]
        assert request["configuration_files"] == list(capture([Path(path) for path in template["configuration_paths"]]))
        operation = operations._read(child["configuration"])
        assert operation["native_parameters"] == case.config["native_parameters"]
        assert operation["native_parameters"]["path"] not in [row["path"] for row in request["configuration_files"]]
        assert operation["coordinator_completion"] == result["coordinator_completion"]
    for record in result["guarded_geometry"]:
        assert record["registered"]["guarded_sources"]
        for bound in record["guarded_sources"]:
            original._verified(bound)
    common = case.data.series / "config/common.toml"
    assert common.read_bytes() != before[common]
    retained = {path: path.read_bytes() for path in case.data.project.root.rglob("*") if path.is_file()}
    written = case.sink.written_bytes
    with case.sink.activate():
        repeated = operations.run_apply(case.config, case.plan, case.revision, case.data.project)
    assert repeated == result and case.sink.written_bytes == written
    assert all(path.read_bytes() == raw for path, raw in retained.items())
    assert not Path(case.plan["build_plans"][0]["request_template"]["output"]["intended_root"]).exists()


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_changed_original_geometry_source_stops_before_any_writer(applied_plan):
    """A changed source fails the original proof before export, Apply or staging."""
    case = applied_plan
    proof = operations._geometry_sources(case.data.project, case.revision, case.plan["runs"])
    measurement = Path(proof[0]["registered"]["registration"]["measurement"]["path"])
    measurement.write_bytes(measurement.read_bytes() + b"\n ")
    before = {path: path.read_bytes() for path in (case.data.series / "config").rglob("*.toml")}
    with case.sink.activate(), pytest.raises((ValueError, RuntimeError)):
        operations.run_apply(case.config, case.plan, case.revision, case.data.project)
    assert case.sink.written_bytes == 0
    assert not (case.root / "apply-complete.json").exists()
    assert not (case.root / "build").exists()
    assert not (case.data.project.root / "exports/native_setup").exists()
    assert all(path.read_bytes() == raw for path, raw in before.items())
