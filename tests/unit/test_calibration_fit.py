"""Unit tests for the calibration fit on synthetic run statistics."""

import dataclasses
import tomllib

import numpy as np
import pytest

import darsia

SETTINGS = darsia.CalibrationFitSettings(
    **tomllib.loads(
        """
        # The values the calibration fit ran with in production.
        [calibration_fit]
        gain_window_h = [0.5, 2.375]
        dilution_start_h = 2.5
        dilution_bins_h = [8.0, 48.0]
        anchor_h = 2.247
        anchor_tolerance_h = 0.13
        gain_grid = [[0.0001, 100000.0, 901]]
        beta_grid = [
            [0.0, 4.0, 801], [4.0, 16.0, 241], [16.0, 64.0, 97],
            [128.0, 128.0, 1], [256.0, 256.0, 1],
        ]
        beta_growth_grid = [
            [0.0, 0.4, 21], [0.6, 0.6, 1], [1.0, 1.0, 1], [2.0, 2.0, 1], [4.0, 4.0, 1],
        ]
        tolerance = 1e-10
        evaluations = 600
        """
    )["calibration_fit"]
)

GAIN, BETA, GROWTH = 1.36, 0.53, 0.11
"""The calibration the synthetic frames follow."""
INJECTION_H = [0.52, 0.67, 0.88, 1.16, 1.41, 1.72, 2.10]
LATER_H = [2.57, 3.83, 6.10, 10.4, 20.0, 36.8, 59.1]
RUNS = ("a", "b", "c")


def _frame(run, index, hours, target, injected, volume, histogram, free=0.0):
    """A frame with one histogram of the dissolved CO2."""
    relative, weight = histogram
    return darsia.CalibrationFrame(
        run=run,
        index=index,
        elapsed_h=hours,
        target_h=target,
        injected_g=injected,
        dissolved={1: darsia.SandHistogram(relative, weight)},
        gas=darsia.GasStatistics(
            free_gas_g=free,
            gas_water_g=0.0,
            co2_pore_volume_m3=volume,
            sands={},
        ),
        column=None,
    )


def _run(run, seed, dtype=np.float64):
    """A run's frames whose injected CO2 is the mass at the calibration above."""
    rng = np.random.default_rng(seed)
    anchor = 3.5e-4 * (1 + 0.1 * seed)
    first = (np.array([0.1], dtype), np.array([0.1], dtype))
    frames = [_frame(run, 0, 2.247, None, 1.0, anchor, first)]
    for index, hours in enumerate(INJECTION_H + LATER_H, start=1):
        relative = np.unique(rng.integers(1, 30000, 200)) / 32768
        weight = rng.gamma(2.0, 1.0, len(relative)) * 1e-3 * hours
        volume = anchor * (hours / 2.25 if hours < 2.25 else 1 + 0.7 * np.log(hours))
        x = max(volume / anchor, 1.0)
        factor = 1.0 if hours <= 2.247 else x ** -(BETA + GROWTH * np.log(x))
        gas = 0.02 * hours
        mass = factor * np.dot(weight, np.clip(GAIN * relative, 0, 1)) + gas
        frames.append(
            _frame(
                run,
                index,
                hours,
                hours,
                mass,
                volume,
                (relative.astype(dtype), weight.astype(dtype)),
                free=gas,
            )
        )
    return frames


@pytest.mark.parametrize("dtype, rel", [(np.float32, 1e-4), (np.float64, 1e-7)])
def test_a_fit_recovers_the_calibration_of_the_frames(dtype, rel):
    frames = [f for seed, run in enumerate("abc") for f in _run(run, seed, dtype)]

    calibration = darsia.fit_calibration(frames, RUNS, SETTINGS)

    assert calibration.gain == pytest.approx(GAIN, rel=rel)
    assert calibration.beta == pytest.approx(BETA, rel=rel)
    assert calibration.beta_growth == pytest.approx(GROWTH, rel=rel)
    anchor = frames[15]  # the anchor frame of run b
    assert calibration.anchors["b"] == darsia.DilutionAnchor(
        anchor.elapsed_h, anchor.gas.co2_pore_volume_m3, anchor.index
    )


def test_dilution_is_one_up_to_the_anchor_and_shrinks_with_the_pore_volume():
    """x ** -(beta + beta_growth ln x) after the anchor, never above one."""
    law = dict(
        anchor_h=2.25,
        anchor_pore_volume_m3=4e-4,
        beta=BETA,
        beta_growth=GROWTH,
        dtype=np.float64,
    )
    assert darsia.dilution_factor(2.0, 1e-3, **law) == 1.0
    assert darsia.dilution_factor(3.0, 2e-4, **law) == 1.0
    x = 2.5
    expected = x ** -(BETA + GROWTH * np.log(x))
    assert darsia.dilution_factor(3.0, x * 4e-4, **law) == pytest.approx(expected)


def test_the_frames_used_and_left_out_are_reported():
    frames = [f for seed, run in enumerate("abc") for f in _run(run, seed)]
    frames[1] = _frame("a", 1, 0.52, 0.52, 0.0, 1e-4, ([0.1], [0.1]))
    frames[2] = _frame("a", 2, 0.67, None, 1.0, 1e-4, ([0.1], [0.1]))

    calibration = darsia.fit_calibration(frames, RUNS, SETTINGS)

    decisions = {(d.run, d.index): d for d in calibration.decisions}
    assert len(decisions) == 3 * 14 - 1  # no anchor-only frame is a fit frame
    assert decisions[("a", 1)].reasons == ("no_positive_dose",)
    assert not decisions[("a", 1)].used
    assert ("a", 2) not in decisions
    assert {d.stage for d in calibration.decisions if d.used} == {"gain", "dilution"}
    assert sum(d.used for d in calibration.decisions) == 3 * 14 - 2


def test_every_run_given_has_the_parameters_of_its_mass():
    frames = [f for seed, run in enumerate("abcd") for f in _run(run, seed)]
    # Run d is not fitted: ten times its CO2 would move any fit that read it.
    frames = [
        dataclasses.replace(f, injected_g=10 * f.injected_g) if f.run == "d" else f
        for f in frames
    ]
    statistics = darsia.RunStatisticsSettings(
        mean_mm=5.0, zero_signal=0.0, saturation_signal=1.2, intervals=32768
    )

    calibration = darsia.fit_calibration(frames, RUNS, SETTINGS)
    runs = darsia.mass_calibrations(calibration, statistics)

    assert calibration.gain == pytest.approx(GAIN, rel=1e-7)
    assert calibration.runs == RUNS
    assert {d.run for d in calibration.decisions} == set(RUNS)
    anchor = frames[45]  # the anchor frame of run d, its frame nearest 2.247 h
    assert runs.keys() == {"a", "b", "c", "d"}
    assert runs["d"] == darsia.MassCalibration(
        gain=calibration.gain,
        zero_signal=0.0,
        saturation_signal=1.2,
        line_share=1.0,
        beta=calibration.beta,
        beta_growth=calibration.beta_growth,
        anchor_h=anchor.elapsed_h,
        anchor_pore_volume_m3=anchor.gas.co2_pore_volume_m3,
    )


def test_no_held_out_run_is_in_its_own_fold():
    frames = [f for seed, run in enumerate("abc") for f in _run(run, seed)]
    off = [f for f in frames if f.run != "b"]
    # The held-out run's frames say nothing about the calibration of its fold.
    off += [
        _frame(
            "b",
            f.index,
            f.elapsed_h,
            f.target_h,
            1e-3,
            f.gas.co2_pore_volume_m3,
            (f.dissolved[1].relative, f.dissolved[1].weight),
        )
        for f in frames
        if f.run == "b"
    ]

    fold, predictions = darsia.calibration_fold(frames, RUNS, SETTINGS, "b")
    other, _ = darsia.calibration_fold(off, RUNS, SETTINGS, "b")

    assert (fold.gain, fold.beta, fold.beta_growth) == (
        other.gain,
        other.beta,
        other.beta_growth,
    )
    assert fold.runs == ("a", "c") and fold.held_out == "b"
    held = [d for d in fold.decisions if d.run == "b"]
    assert held and all(d.reasons == ("leave_one_run_out",) for d in held)
    assert [p.index for p in predictions] == list(range(15))
    assert {p.run for p in predictions} == {"b"}
    assert max(abs(p.residual) for p in predictions[1:]) < 1e-3
