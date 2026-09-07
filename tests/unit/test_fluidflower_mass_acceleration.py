from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace

import numpy as np
import pytest

from darsia.fluidflower.mass_acceleration import (
    MassKernelConfiguration,
    NumpyMassComputeSession,
    compare_mass_frame_results,
    create_mass_compute_session,
)


def _inputs(shape: tuple[int, int] = (9, 13)) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(412)
    acid = rng.uniform(0.0, 1.0, shape)
    co2 = rng.uniform(0.2, 1.0, shape)
    gas = co2 * rng.uniform(0.0, 0.8, shape)
    active = np.ones(shape, dtype=bool)
    active[[0, -1], :] = False
    voxel = rng.uniform(1.0e-10, 4.0e-10, shape) * active
    return {
        "active": active,
        "voxel_volume_m3": voxel,
        "acidification": acid,
        "co2_probability": co2,
        "gas_probability": gas,
        "solubility_co2": rng.uniform(1.4, 1.9, shape),
        "density_gaseous_co2": rng.uniform(1.7, 2.2, shape),
        "diffusion_factor": 0.83,
    }


def _configuration(model: str = "titration") -> MassKernelConfiguration:
    signal = np.asarray([0.0, 0.08, 0.23, 0.55, 1.0], dtype=np.float64)
    concentration = np.asarray([0.0, 0.006, 0.03, 0.25, 1.0], dtype=np.float64)
    return MassKernelConfiguration(
        model=model,
        transfer_shape="smoothstep",
        acid_low=0.08,
        acid_high=0.92,
        co2_onset=0.12,
        co2_high=0.88,
        gas_onset=0.18,
        gas_high=0.82,
        max_gas_saturation=0.72,
        lut_signal=signal,
        lut_concentration=concentration,
    )


def _session_values(inputs: dict[str, np.ndarray | float]) -> dict[str, object]:
    return {
        key: inputs[key]
        for key in (
            "acidification",
            "co2_probability",
            "gas_probability",
            "solubility_co2",
            "density_gaseous_co2",
            "diffusion_factor",
        )
    }


def _numpy_session(
    backend: str,
    inputs: dict[str, np.ndarray | float],
    configuration: MassKernelConfiguration | None = None,
) -> NumpyMassComputeSession:
    return NumpyMassComputeSession(
        backend=backend,
        active=np.asarray(inputs["active"]),
        voxel_volume_m3=np.asarray(inputs["voxel_volume_m3"]),
        configuration=configuration or _configuration(),
    )


def test_float32_contract_matches_float64_reference() -> None:
    inputs = _inputs()
    reference = _numpy_session("cuda-float64", inputs).compute_frame(
        **_session_values(inputs)
    )
    candidate = _numpy_session("cuda-float32", inputs).compute_frame(
        **_session_values(inputs)
    )

    report = compare_mass_frame_results(
        candidate,
        reference,
        absolute_tolerance=1.0e-5,
        relative_tolerance=1.0e-4,
    )

    assert report["passed"]
    assert candidate.mass_total.dtype == np.float32
    assert np.all(candidate.mass_total[0] == 0.0)
    assert np.allclose(
        candidate.mass_total,
        candidate.mass_aqueous + candidate.mass_free_gas,
    )


def test_cpu_float32_factory_is_explicit_and_matches_reference() -> None:
    inputs = _inputs()
    session = create_mass_compute_session(
        backend="cpu-float32",
        active=np.asarray(inputs["active"]),
        voxel_volume_m3=np.asarray(inputs["voxel_volume_m3"]),
        configuration=_configuration(),
    )
    candidate = session.compute_frame(**_session_values(inputs))
    reference = _numpy_session("cpu-float64", inputs).compute_frame(
        **_session_values(inputs)
    )

    report = compare_mass_frame_results(
        candidate,
        reference,
        absolute_tolerance=1.0e-5,
        relative_tolerance=1.0e-4,
    )

    assert report["passed"]
    assert candidate.mass_total.dtype == np.float32
    assert session.provenance()["backend"] == "cpu-float32"


def test_simple_flash_uses_transferred_acidification_directly() -> None:
    inputs = _inputs((3, 4))
    config = replace(_configuration("simple_flash"), transfer_shape="linear")
    session = _numpy_session("opencl-float64", inputs, config)
    result = session.compute_frame(**_session_values(inputs))
    expected_acid = np.clip(
        (np.asarray(inputs["acidification"]) - config.acid_low)
        / (config.acid_high - config.acid_low),
        0.0,
        1.0,
    )
    expected_gate = np.clip(
        (np.asarray(inputs["co2_probability"]) - config.co2_onset)
        / (config.co2_high - config.co2_onset),
        0.0,
        1.0,
    )
    expected = expected_acid * expected_gate * float(inputs["diffusion_factor"])
    expected[~np.asarray(inputs["active"], dtype=bool)] = 0.0

    assert np.allclose(result.concentration_aqueous, expected)


def test_parity_report_rejects_changed_phase_field() -> None:
    inputs = _inputs()
    reference = _numpy_session("cuda-float64", inputs).compute_frame(
        **_session_values(inputs)
    )
    candidate = replace(
        reference,
        saturation_free_gas=reference.saturation_free_gas + 0.02,
    )

    report = compare_mass_frame_results(
        candidate,
        reference,
        absolute_tolerance=1.0e-5,
        relative_tolerance=1.0e-4,
    )

    assert not report["passed"]
    assert not report["fields"]["saturation_free_gas"]["passed"]


@pytest.mark.parametrize("backend", ["cuda-float32", "cuda-float64"])
def test_cuda_kernel_matches_numpy_when_available(backend: str) -> None:
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed")
    inputs = _inputs()
    try:
        session = create_mass_compute_session(
            backend=backend,
            active=np.asarray(inputs["active"]),
            voxel_volume_m3=np.asarray(inputs["voxel_volume_m3"]),
            configuration=_configuration(),
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))
    try:
        candidate = session.compute_frame(**_session_values(inputs))
        repeated = session.compute_frame(**_session_values(inputs))
    finally:
        session.close()
    reference = _numpy_session("cuda-float64", inputs).compute_frame(
        **_session_values(inputs)
    )
    tolerance = 1.0e-5 if backend.endswith("float32") else 1.0e-10
    report = compare_mass_frame_results(
        candidate,
        reference,
        absolute_tolerance=tolerance,
        relative_tolerance=10.0 * tolerance,
    )
    assert report["passed"]
    assert np.array_equal(candidate.mass_total, repeated.mass_total)


@pytest.mark.parametrize("backend", ["opencl-float32", "opencl-float64"])
def test_opencl_kernel_matches_numpy_when_available(backend: str) -> None:
    if importlib.util.find_spec("pyopencl") is None:
        pytest.skip("PyOpenCL is not installed")
    inputs = _inputs()
    try:
        session = create_mass_compute_session(
            backend=backend,
            active=np.asarray(inputs["active"]),
            voxel_volume_m3=np.asarray(inputs["voxel_volume_m3"]),
            configuration=_configuration(),
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))
    try:
        candidate = session.compute_frame(**_session_values(inputs))
        repeated = session.compute_frame(**_session_values(inputs))
    finally:
        session.close()
    reference = _numpy_session("opencl-float64", inputs).compute_frame(
        **_session_values(inputs)
    )
    tolerance = 1.0e-5 if backend.endswith("float32") else 1.0e-10
    report = compare_mass_frame_results(
        candidate,
        reference,
        absolute_tolerance=tolerance,
        relative_tolerance=10.0 * tolerance,
    )
    assert report["passed"]
    assert np.array_equal(candidate.mass_total, repeated.mass_total)


def test_metal_kernel_matches_numpy_when_available() -> None:
    if sys.platform != "darwin" or importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch Metal is not installed on Apple Silicon")
    inputs = _inputs()
    try:
        session = create_mass_compute_session(
            backend="metal-float32",
            active=np.asarray(inputs["active"]),
            voxel_volume_m3=np.asarray(inputs["voxel_volume_m3"]),
            configuration=_configuration(),
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))
    try:
        candidate = session.compute_frame(**_session_values(inputs))
    finally:
        session.close()
    reference = _numpy_session("cuda-float64", inputs).compute_frame(
        **_session_values(inputs)
    )
    report = compare_mass_frame_results(
        candidate,
        reference,
        absolute_tolerance=1.0e-5,
        relative_tolerance=1.0e-4,
    )
    assert report["passed"]
