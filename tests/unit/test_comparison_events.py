"""Unit tests for the events of analysis domains on a synthetic run."""

import csv
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.presets.workflows.analysis.domain_mass import RunSeries, write_domain_mass
from darsia.presets.workflows.comparison.comparison_events import (
    COLUMNS,
    EventQuestion,
    event_table,
    injection_answers,
    integrated_events,
    question_states,
    run_answer,
    write_events,
)
from darsia.presets.workflows.config.multi_fluidflower_config import EventConfig

QUESTIONS = {
    row["name"]: EventQuestion(**row)
    for row in tomllib.loads(
        """
        # Questions about the free gas of a box over a run: when it arrives, when it
        # reaches a share of the injected CO2, falls below a mass it reached, peaks
        # and is gone.
        [[question]]
        name = "arrival"
        conditions = [{component = "gas", event = "detected", domain_id = "box2"}]

        [[question]]
        name = "share"
        [[question.conditions]]
        component = "gas"
        event = "reaches"
        domain_id = "box2"
        threshold = 15.0
        unit = "percent_injected"

        [[question]]
        name = "falls"
        [[question.conditions]]
        component = "gas"
        event = "falls_below"
        domain_id = "box2"
        threshold = 0.05
        unit = "g"

        [[question]]
        name = "peak"
        conditions = [{component = "gas", event = "maximum", domain_id = "box2"}]

        [[question]]
        name = "gone"
        conditions = [{component = "gas", event = "gone", domain_id = "box2"}]
        """
    )["question"]
}
HOURS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
GAS = [0.0, 0.0, 0.01, 0.05, 0.08, 0.06, 0.0, 0.0]
INJECTED = [0.0, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4]
INJECTIONS = pd.DataFrame(
    {"number": [1, 2], "start_h": [0.0, 1.0], "end_h": [1.0, 2.0]}
)


def series(dtype=np.float64, gas=GAS, **masses):
    """The synthetic run: free gas in box2 that arrives, peaks and leaves."""
    values = {"gas": np.asarray(gas, dtype), **masses}
    return RunSeries(
        run="run",
        case_ids=tuple(f"run-{n}" for n in range(len(HOURS))),
        elapsed_h=np.asarray(HOURS, dtype),
        injected_g=np.asarray(INJECTED, dtype),
        masses={"box2": values},
    )


def test_injections_are_each_ports_consecutive_steps(tmp_path):
    """Steps at one port are one injection, numbered in start order, in hours since
    the first step's start; a step at another port starts the next one."""
    path = tmp_path / "injection.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "location_x": [0.45, 0.45, 0.74, 0.45],
            "location_y": [0.02, 0.02, 0.16, 0.02],
            "start": [
                "2023-01-31 09:51:00",
                "2023-01-31 09:50:00",
                "2023-01-31 10:20:00",
                "2023-01-31 11:00:00",
            ],
            "end": [
                "2023-01-31 10:05:00",
                "2023-01-31 09:51:00",
                "2023-01-31 11:00:00",
                "2023-01-31 11:30:00",
            ],
            "rate_g/min": [0.1, 0.05, 0.1, 0.1],
        }
    ).to_csv(path, index=False)

    injections = darsia.InjectionProtocol(path).injections()

    assert injections["number"].tolist() == [1, 2, 3]
    assert injections["location_x"].tolist() == [0.45, 0.74, 0.45]
    assert injections["start_h"].tolist() == [0.0, 0.5, 70 / 60]
    assert injections["end_h"].tolist() == [0.25, 70 / 60, 100 / 60]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_events_of_a_synthetic_run(dtype):
    """Arrival, a share of the injected CO2, falling below a mass reached, the peak
    and the end, each at its frame, in single and double precision."""
    run = series(dtype)
    found = {name: run_answer(q, run, INJECTIONS) for name, q in QUESTIONS.items()}

    frames = {name: answer.case_id for name, answer in found.items()}
    assert frames == {
        "arrival": "run-2",
        "share": "run-3",
        "falls": "run-6",
        "peak": "run-4",
        "gone": "run-6",
    }
    arrival = found["arrival"]
    assert (arrival.elapsed_h, arrival.resolution_h) == (1.0, 0.5)
    assert (arrival.injection, arrival.after_injection_h) == (2, 0.0)
    assert found["share"].note == "" and found["falls"].after_injection_h == 3.0


def test_unknown_frames_are_neither_true_nor_false(tmp_path):
    """A domain that is not complete at a frame is unknown there with complete
    masses: an answer after it says so, a question never found says why, and a
    share of the injected CO2 is never reached before anything is injected."""
    rows = [
        dict(
            run="run",
            case_id=f"run-{n}",
            elapsed_h=hours,
            domain_id="box2",
            gas_mass_g=None if n == 1 else gas,
            gas_quantified_mass_g=gas,
            total_mass_g=None,
            total_quantified_mass_g=gas,
            aqueous_mass_g=0.0,
            aqueous_quantified_mass_g=0.0,
        )
        for n, (hours, gas) in reversed(list(enumerate(zip(HOURS, GAS))))
    ]
    path = tmp_path / "domain_mass.csv"
    write_domain_mass(path, rows)
    with path.open(encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    injected = dict(zip((f"run-{n}" for n in range(8)), INJECTED))
    complete = RunSeries.from_domain_mass(table, injected, "complete")
    quantified = RunSeries.from_domain_mass(table, injected, "quantified")

    assert complete.case_ids == series().case_ids
    assert question_states(QUESTIONS["arrival"], complete)[:3].tolist() == [0, 0.5, 1]
    after = run_answer(QUESTIONS["arrival"], complete)
    assert (after.note, after.resolution_h) == ("1 earlier frame(s) unknown", None)
    assert run_answer(QUESTIONS["arrival"], quantified).note == ""
    total = EventQuestion(
        "total", [dict(component="total", event="detected", domain_id="box2")]
    )
    assert run_answer(total, complete).note == "Not found; some frames are unknown"
    early = replace(QUESTIONS["share"].conditions[0], threshold=0.0)
    assert run_answer(EventQuestion("early", [early]), quantified).case_id == "run-1"


def test_thresholds_are_not_rounded_to_the_masses_precision():
    """A threshold just above a single-precision mass is not reached, in grams or
    in percent of the injected CO2."""
    run = series(np.float32, gas=np.ones(8))
    run = RunSeries(
        run.run, run.case_ids, run.elapsed_h, np.ones(8, np.float32), run.masses
    )
    above = 1 + 2**-25
    for threshold, unit in ((above, "g"), (100 * above, "percent_injected")):
        condition = dict(component="gas", event="reaches", domain_id="box2")
        condition.update(threshold=threshold, unit=unit)
        assert run_answer(EventQuestion("q", [condition]), run).case_id is None


def test_conditions_combine_and_hold_with_the_noise_filter():
    """Conditions combine with and and or; the noise limit and frames in a row
    filter flickering mass."""
    gas = [0.0, 0.002, 0.0, 0.01, 0.0, 0.02, 0.03, 0.03]
    run = series(gas=gas, aqueous=np.asarray([0.0, 0, 0, 0, 0.2, 0.3, 0.3, 0.3]))
    arrival = dict(component="gas", event="detected", domain_id="box2")
    dissolved = dict(component="aqueous", event="detected", domain_id="box2")

    def first(**settings):
        return run_answer(EventQuestion("q", **settings), run).case_id

    assert first(conditions=[arrival]) == "run-1"
    assert first(conditions=[arrival], noise_min=0.005) == "run-3"
    share = dict(noise_min=5.0, noise_unit="percent_injected")
    assert first(conditions=[arrival], **share) == "run-6"
    assert first(conditions=[arrival], hold_frames=2) == "run-5"
    assert first(conditions=[arrival, dissolved], noise_min=0.005) == "run-5"
    assert (
        first(conditions=[arrival, dissolved], combine="any", noise_min=0.05) == "run-4"
    )
    assert first(conditions=[arrival], hold_frames=4) is None
    gone = dict(arrival, event="gone")
    assert first(conditions=[gone]) is None
    held = question_states(EventQuestion("q", [arrival], hold_frames=4), run)
    assert held[-3:].tolist() == [0.5, 0.5, 0.5]


def test_answers_per_injection_count_from_each_start(tmp_path):
    """Each injection's answer is the first frame in its time, counted from its
    start; a question that already holds is answered at its first frame, and an
    injection without a frame says so. The table holds both modes."""
    run = series()
    injections = pd.DataFrame(
        {"number": [1, 2, 3], "start_h": [0.0, 1.0, 9.0], "end_h": [1.0, 2.0, 9.5]}
    )
    answers = injection_answers(QUESTIONS["arrival"], run, injections)

    assert [a.case_id for a in answers] == [None, "run-2", None]
    assert answers[0].note == "Never in this injection's time"
    assert answers[2].note == "No frame in this injection's time"
    held = injection_answers(QUESTIONS["share"], run, injections.iloc[[0, 1]])
    assert (held[1].case_id, held[1].after_injection_h) == ("run-3", 0.5)
    boundary = pd.DataFrame({"number": [1, 2], "start_h": [0.0, 1.5]})
    late = injection_answers(QUESTIONS["arrival"], run, boundary)[1]
    assert (late.case_id, late.resolution_h) == ("run-3", None)

    rows = event_table([QUESTIONS["arrival"]], [run], {"run": injections})
    path = tmp_path / "events.csv"
    write_events(path, rows)
    with path.open(encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert list(table[0]) == list(COLUMNS)
    assert [(r["mode"], r["injection"], r["case_id"]) for r in table] == [
        ("run", "3", "run-2"),
        ("injection", "1", ""),
        ("injection", "2", "run-2"),
        ("injection", "3", ""),
    ]


def test_the_official_events_are_unchanged():
    """The official events of an integrated-mass table are the first times their
    region's mass reaches the relative threshold of the largest injected mass."""
    rng = np.random.default_rng(3)
    table = pd.DataFrame(
        {
            "time": np.arange(12) * 0.25,
            "exact_mass": np.linspace(0, 2.0, 12),
            "box_a_detected_mass": np.cumsum(rng.uniform(0, 0.2, 12)),
            "box_a_detected_mass_g": rng.uniform(0, 0.1, 12),
            "box_a_detected_mass_aq": np.full(12, np.nan),
        }
    ).sample(frac=1, random_state=1)
    events = [
        EventConfig().load(key, dict(mode=mode, roi_name="box_a", relative_threshold=t))
        for key, mode, t in (
            ("a", "mass", 0.3),
            ("b", "mass_g", 0.04),
            ("c", "mass_aq", 0.01),
            ("d", "mass", 9.0),
        )
    ]
    found = integrated_events(table, events)

    for event in events:
        key = f"box_a_detected_{event.mode}"
        mass = table[key] >= event.relative_threshold * np.max(table["exact_mass"])
        official = np.min(table["time"][mass])
        assert found[event.event_id] == (None if np.isnan(official) else official)
    assert found["a"] is not None and found["c"] is None and found["d"] is None
    assert integrated_events(table.assign(exact_mass=np.nan), events[:1]) == {"a": None}


def test_the_official_volume_events_are_unchanged():
    """A volume event of a single-phase fluid is the first time its region's volume
    reaches the relative threshold of the largest injected volume; without the
    injected volume it is refused."""
    table = pd.DataFrame(
        {
            "time": np.arange(8) * 0.5,
            "exact_volume_total_m3": np.linspace(0, 4e-6, 8),
            "box_a_detected_volume_m3": np.linspace(0, 3e-6, 8),
        }
    ).sample(frac=1, random_state=2)
    events = [
        EventConfig().load(
            key, dict(mode="volume", roi_name="box_a", relative_threshold=t)
        )
        for key, t in (("v", 0.4), ("w", 0.9))
    ]
    found = integrated_events(table, events)

    volume = table["box_a_detected_volume_m3"]
    largest = np.max(table["exact_volume_total_m3"])
    assert found["v"] == np.min(table["time"][volume >= 0.4 * largest])
    assert found["w"] is None
    with pytest.raises(AssertionError, match="exact_volume_total_m3"):
        integrated_events(table.drop(columns="exact_volume_total_m3"), events)
