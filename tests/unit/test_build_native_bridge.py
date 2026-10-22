"""Coordinate/provenance integration with real files/readers/exclusive writers.

RAW decode, registration and photometric correction are explicit test doubles.
These cases prove adapter composition and refusal, not scientific RAW parity.
The completed-Build reader's actual archive/ownership contract has its own tests.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass import build_native_bridge as bridge
from darsia.fluidflower.mass.n1_pipeline import asset


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return asset(path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """Real immutable input files surrounding the isolated scientific boundary."""
    raw = tmp_path / "raw"
    raw.mkdir()
    baseline = raw / "baseline.ARW"
    baseline.write_bytes(b"baseline capture contract fixture")
    frame = raw / "wash.ARW"
    frame.write_bytes(b"explicit selected wash capture contract fixture")
    stored = tmp_path / "saved_build"
    stored.mkdir()
    shape = stored / "shape.npz"
    shape.write_bytes(b"new owned shape contract fixture")
    physical = stored / "facies.npz"
    physical.write_bytes(b"saved physical archive contract fixture")
    fit = tmp_path / "fit.npz"
    fit.write_bytes(b"frozen measured photometry contract fixture")
    sidecar = _json(tmp_path / "fit.json", {"source": {"source_arw_sha256": asset(baseline)["sha256"]}})
    old_shape = tmp_path / "old_shape.npz"
    old_shape.write_bytes(b"old shape contract fixture")
    quality = dict(fit_n=10, fit_inlier_fraction=1., sampled_jacobian_min=1.,
        heldout={"held": dict(n=4, p50_px=.1, p95_px=.2, max_px=.3, median_match_correlation=.9)})
    policy = dict(minimum_matches=3, minimum_inlier_fraction=.5,
                  heldout_zones=[["held", [0, 0, 3, 2], 1]],
                  heldout_minimum_matches=2, heldout_min_correlation=.1)
    identity = {"boundary": "a" * 64}
    parameters = dict(source=dict(profiles={"sample": "camera"}, native_shape_hw=[2, 3],
        source_id_pattern="[a-z]+", homography_min_scale=1e-12, homography_match_atol=1e-10,
        raw_execution_identity=identity), registration=dict(common=dict(
            review=policy, sensor=dict(camera_wh=[3, 2]), profiles={"camera": {}})))
    native = _json(tmp_path / "native.json", parameters)
    matrices = dict(
        hb=np.array([[1.05, -.02, .3], [.01, .95, .4], [0., 0., 1.]]),
        hn=np.array([[.9, .05, .1], [.02, 1.1, .2], [0., 0., 1.]]),
        ho=np.array([[1., .1, .2], [0., 1., .1], [0., 0., 1.]]),
        frozen=np.array([[.8, .01, 1.], [.02, .9, 2.], [0., 0., 1.]]),
        k=np.array([[.95, .03, .4], [.04, .98, .5], [0., 0., 1.]]),
        j=np.array([[1.02, .02, .9], [-.01, 1., .7], [0., 0., 1.]]))
    binding = dict(contract=bridge.raw_rig_source.CONTRACT, run="sample", profile_id="camera",
        native_shape_hw=[2, 3], baseline_raw=asset(baseline),
        baseline_reference_to_native_h=matrices["frozen"].tolist(),
        shape_assets=[asset(old_shape)], image_metadata=dict(height=.02, width=.03),
        models=dict(fit=dict(data=asset(fit), sidecar=sidecar),
                    reference=dict(data=asset(fit), sidecar=sidecar), old_field=None),
        execution_identity=identity)
    binding_path = tmp_path / "old_binding.json"
    binding_bound = _json(binding_path, binding)
    request = _json(tmp_path / "request.json", dict(output=dict(forbidden_raw_roots=[str(raw)])))
    baseline_archive = asset(physical)
    renderer = dict(native_to_camera_u_h=matrices["hb"].tolist(), profile_id="camera",
        camera_u_wh=[3, 2], saved_shape_chain_replay_bit_identical=True,
        source_npz=baseline_archive, shape_corrections=[asset(shape)], registration=quality,
        saved_shape_chain_replay=dict(correction_assets=[asset(shape)], normalization=dict(
            black_subtracted_dn_divisor=100., white_balance_applied=False, gamma_applied=False)))
    certificate = dict(renderer=renderer, outputs=[asset(shape), baseline_archive],
        H_rig_corrected_native_to_baseline_camera_u=matrices["hb"].tolist(),
        corrected_baseline=dict(archive=baseline_archive), inputs=dict(uncorrected_grid=dict(
            metadata=dict(type="OpticalImage", color_space="RGB", dimensions=[.02, .03],
                          origin=[0., .02], date="2026-01-01T00:00:00", reference_date="2026-01-01T00:00:00", time=0.))))
    certificate_bound = _json(stored / "renderer.json", certificate)
    operation = _json(tmp_path / "operation.json", {"complete": True})
    context = SimpleNamespace(run="sample", grid=dict(num_voxels=[2, 3]),
        baseline_rgb=np.full((2, 3, 3), .25, np.float32), provenance=dict(
            native_configuration=native, camera_binding=dict(run_id="sample", profile_id="camera"),
            raw_acquisition=asset(baseline), baseline_renderer=certificate_bound,
            raw_build_request=request, physical_archives=dict(shape_corrected_baseline=baseline_archive)))
    calls = SimpleNamespace(loads=0, registrations=[], corrections=[], renders=[], hook=None,
                            correction_h=None, log=None)

    def load(bound, *, reader):
        assert bound == operation
        calls.loads += 1
        for row in (operation, native, context.provenance["baseline_renderer"], request, baseline_archive):
            reader.path(row)
        return context

    def native_renderer(h):
        return dict(native_to_camera_u_h=h.tolist(), camera_u_wh=[3, 2], registration=deepcopy(quality))

    class Processor:
        def __init__(self, path, *, parameters):
            self.binding = json.loads(Path(path).read_bytes())
            self.parameters = parameters
            self.old = Path(path) == binding_path
            self.baseline_sensor = self._sensor(self.binding["baseline_raw"])
            self.baseline_renderer = native_renderer(matrices["ho" if self.old else "hn"])

        def _path(self, path):
            return Path(path)

        def _sensor(self, bound):
            return SimpleNamespace(source=bound, metadata=dict(white_level=110, black_level=10))

        def _render(self, sensor):
            h = matrices["hn"] if sensor.source["sha256"] == asset(baseline)["sha256"] else matrices["k"]
            return np.ones((3, 2, 3), np.float32), native_renderer(h)

        def correct(self, geometry):
            # Exercise the unchanged original frame/binding composition guard.
            bridge.raw_rig_source.validate_frame(geometry, self.binding, self.parameters)
            calls.corrections.append(deepcopy(geometry))
            if calls.hook is not None:
                calls.hook()
            _, rendered = self._render(self._sensor(geometry["acquisition"]))
            if calls.correction_h is not None:
                rendered["native_to_camera_u_h"] = calls.correction_h.tolist()
            log = np.ones((3, 2, 3), np.float32) if calls.log is None else calls.log
            return log, dict(renderer=rendered, new_photometric_fit=False)

    def render(sensor, **kwargs):
        calls.renders.append(kwargs)
        return np.ones((3, 2, 3), np.float32), native_renderer(matrices["hn"])

    def register(saved_green, camera_green, *, parameters):
        calls.registrations.append((saved_green.copy(), camera_green.copy(), parameters))
        return matrices["j"].copy(), deepcopy(quality)

    monkeypatch.setattr(bridge, "load_build_physical_context", load)
    monkeypatch.setattr(bridge.raw_rig_source, "RawRigSourceProcessor", Processor)
    monkeypatch.setattr(bridge.raw_rig_source, "execution_identity", lambda: identity)
    monkeypatch.setattr(bridge.raw_positioning, "render_native", render)
    monkeypatch.setattr(bridge.raw_positioning, "camera_u", lambda *args, **kwargs: np.full((3, 2, 3), .7, np.float32))
    monkeypatch.setattr(bridge.registration, "review_to_camera", register)
    monkeypatch.setattr(bridge, "inspect_local_file", lambda path: SimpleNamespace(allowed=True))
    return SimpleNamespace(root=tmp_path, operation=operation, binding=binding, binding_path=binding_path,
        binding_bound=binding_bound, context=context, calls=calls, matrices=matrices,
        certificate=certificate, certificate_path=Path(certificate_bound["path"]),
        baseline=asset(baseline), frame=asset(frame), fit=fit, shape=shape, physical=physical,
        quality=quality, raw=raw)


def _prepare(value):
    return bridge.BuildNativeBridge(value.operation, value.binding_bound, value.root / "bridge")


def test_saved_build_and_direct_native_grids_remain_distinct(inputs):
    actual = _prepare(inputs)
    m = inputs.matrices
    np.testing.assert_allclose(actual.reference, np.linalg.inv(m["hn"]) @ m["ho"] @ m["frozen"])
    np.testing.assert_allclose(actual.build_to_baseline, np.linalg.inv(m["hn"]) @ m["hb"])
    produced = json.loads(Path(actual.binding["path"]).read_bytes())
    assert produced["models"] == inputs.binding["models"]
    assert produced["shape_assets"] != inputs.binding["shape_assets"]
    assert produced["image_metadata"]["origin"] == [0., .02]
    assert "type" not in produced["image_metadata"] and "color_space" not in produced["image_metadata"]
    assert inputs.calls.renders[0]["image_metadata"]["date"].year == 2026
    assert not actual.review_green.flags.writeable


def test_original_correction_receives_composed_frame_and_real_bound_products(inputs):
    actual = _prepare(inputs)
    result = actual.frame(inputs.frame, inputs.root / "wash_result")
    m = inputs.matrices
    physical = np.linalg.inv(m["k"]) @ m["j"]
    baseline = physical @ np.linalg.inv(np.linalg.inv(m["hn"]) @ m["hb"])
    geometry = inputs.calls.corrections[-1]
    np.testing.assert_allclose(geometry["baseline_to_native_h"], baseline)
    np.testing.assert_allclose(geometry["reference_to_native_h"], baseline @ actual.reference)
    saved = json.loads(Path(result.receipt["path"]).read_bytes())
    np.testing.assert_allclose(saved["build_to_frame_native_h"], physical)
    np.testing.assert_array_equal(inputs.calls.registrations[0][0], np.full((2, 3), 25., np.float32))
    assert inputs.calls.registrations[0][2] == actual.parameters["registration"]["common"]
    assert result.native_task["kind"] == "raw_rig"
    assert result.native_task["processor"]["raw_binding"] == actual.binding
    with np.load(result.products["path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["log_linear_rgb"], np.ones((3, 2, 3), np.float32))
        assert archive["valid_mask"].dtype == bool and archive["valid_mask"].all()
    assert "physical_context_verified" not in saved and "optical_source_contract" not in saved
    assert inputs.calls.loads == 1  # Late checks reuse consumed file identities, not archive decodes.


def test_exact_baseline_keeps_literal_original_identity(inputs):
    actual = _prepare(inputs)
    actual.frame(inputs.baseline, inputs.root / "baseline_result")
    np.testing.assert_array_equal(inputs.calls.corrections[-1]["baseline_to_native_h"], np.eye(3))
    np.testing.assert_allclose(inputs.calls.corrections[-1]["reference_to_native_h"], actual.reference)
    assert inputs.calls.registrations == []


def test_unrepresentable_original_date_metadata_refuses_before_output(inputs):
    inputs.certificate["inputs"]["uncorrected_grid"]["metadata"]["time"] = None
    inputs.context.provenance["baseline_renderer"] = _json(inputs.certificate_path, inputs.certificate)
    with pytest.raises(ValueError, match="explicit time with serialized Build dates"):
        _prepare(inputs)
    assert not (inputs.root / "bridge").exists()


def test_projective_coordinate_joins_preserve_original_homogeneous_normalization(inputs):
    m = inputs.matrices
    for name, last_row in (("ho", [.001, -.002, 1.]), ("hn", [-.002, .001, 1.]),
                           ("j", [.003, .001, 1.]), ("k", [.001, -.003, 1.])):
        m[name][2] = last_row
    normalized = lambda matrix: matrix / matrix[2, 2]
    actual = _prepare(inputs)
    result = actual.frame(inputs.frame, inputs.root / "projective")
    a = normalized(np.linalg.inv(m["hn"]) @ m["hb"])
    p = normalized(np.linalg.inv(m["k"]) @ m["j"])
    b = normalized(p @ np.linalg.inv(a))
    f = normalized(np.linalg.inv(m["hn"]) @ m["ho"] @ m["frozen"])
    np.testing.assert_allclose(actual.reference, f)
    np.testing.assert_allclose(inputs.calls.corrections[-1]["baseline_to_native_h"], b)
    np.testing.assert_allclose(inputs.calls.corrections[-1]["reference_to_native_h"], normalized(b @ f))
    saved = json.loads(Path(result.receipt["path"]).read_bytes())
    np.testing.assert_allclose(saved["build_to_frame_native_h"], p)


@pytest.mark.parametrize("key,value", [("run", "other"), ("profile_id", "other"),
    ("native_shape_hw", [3, 2]), ("baseline_raw", {"sha256": "0" * 64})])
def test_mismatched_source_cannot_bind_a_build(inputs, key, value):
    inputs.binding[key] = value
    changed = _json(inputs.binding_path, inputs.binding)
    with pytest.raises(ValueError, match="differ in run/profile/baseline/grid"):
        bridge.BuildNativeBridge(inputs.operation, changed, inputs.root / "refused")
    assert not (inputs.root / "refused").exists()


@pytest.mark.parametrize("field", ["fit", "shape", "physical"])
def test_changed_consumed_input_refuses_before_frame_publication(inputs, field):
    actual = _prepare(inputs)
    getattr(inputs, field).write_bytes(b"late changed producer input")
    with pytest.raises(ValueError, match="changed|size differs"):
        actual.frame(inputs.frame, inputs.root / "refused")
    assert not (inputs.root / "refused").exists()


def test_model_changed_during_original_correction_is_detected(inputs):
    actual = _prepare(inputs)
    inputs.calls.hook = lambda: inputs.fit.write_bytes(b"changed while correction ran")
    with pytest.raises(ValueError, match="changed|size differs"):
        actual.frame(inputs.frame, inputs.root / "refused")
    assert not (inputs.root / "refused").exists()


def test_registration_gate_and_original_policy_are_retained(inputs):
    actual = _prepare(inputs)
    inputs.quality["fit_n"] = 0
    with pytest.raises(ValueError, match="fit/orientation"):
        actual.frame(inputs.frame, inputs.root / "refused")
    assert inputs.calls.corrections == []


def test_registration_changing_between_geometry_and_correction_refuses(inputs):
    actual = _prepare(inputs)
    inputs.calls.correction_h = inputs.matrices["k"].copy()
    inputs.calls.correction_h[0, 2] += .1
    with pytest.raises(ValueError, match="registration changed"):
        actual.frame(inputs.frame, inputs.root / "refused")


def test_no_new_heldout_residual_ceiling_is_invented(inputs):
    inputs.quality["heldout"]["held"]["p95_px"] = 1000.
    # Original policy has no p95 ceiling; retain that fact rather than qualify science.
    actual = _prepare(inputs)
    assert actual.frame(inputs.frame, inputs.root / "result").receipt


def test_protected_raw_roots_and_existing_outputs_cannot_be_written(inputs):
    actual = _prepare(inputs)
    with pytest.raises(ValueError, match="protected RAW"):
        actual.frame(inputs.frame, inputs.raw / "new_folder")
    with pytest.raises(ValueError, match="fresh absolute"):
        actual.frame(inputs.frame, inputs.root / "bridge")
    assert not (inputs.raw / "new_folder").exists()


def test_empty_original_support_cannot_publish_a_native_frame(inputs):
    actual = _prepare(inputs)
    inputs.calls.log = np.full((3, 2, 3), np.nan, np.float32)
    with pytest.raises(ValueError, match="no finite native support"):
        actual.frame(inputs.frame, inputs.root / "refused")
