"""Unit tests for the ensemble statistics of a synthetic series of runs."""

import csv
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib.image import imread

import darsia
from darsia.presets.workflows.analysis.domain_mass import RunSeries
from darsia.presets.workflows.analysis.domains import AnalysisDomain
from darsia.presets.workflows.comparison.comparison_events import (
    EventQuestion,
    event_table,
    read_events,
    write_events,
)
from darsia.presets.workflows.comparison.ensemble import (
    EnsembleRun,
    EnsembleSettings,
    dose_difference,
    ensemble_tables,
    event_summary,
    summary,
    write_tables,
)
from darsia.presets.workflows.comparison.ensemble_figures import ensemble_figures
from darsia.presets.workflows.media.style import Curve, FigureStyle

SETTINGS = EnsembleSettings(
    **tomllib.loads(
        """
        # Our production's ensemble: the quantified masses of the domains, and the
        # centres of the total CO2 in the trajectory figures.
        masses = "quantified"
        centre_component = "total"
        """
    )
)
STYLE_TABLE = tomllib.loads(
    """
    # Our production's figure style: the ensemble figures' palette, 150 dpi and
    # the 50x rule for a logarithmic time axis; the films' colours and curves.
    dpi = 150
    log_span = 50.0
    surface = "#fcfcfb"
    ink = "#0b0b0b"
    secondary = "#52514e"
    muted = "#898781"
    grid = "#e1e0d9"
    axis = "#c3c2b7"
    washes = ["#ebeae4", "#dcdcd4"]
    series = "#2a78d6"
    frame = "#1e2226"
    empty = "#3e4246"
    no_value = "#e6edf2"
    colour_map = "viridis"
    marker = "#667085"
    curves = [
        {column = "total_mass_g", name = "Total", colour = "#175aad", line = "-"},
        {column = "aqueous_mass_g", name = "Dissolved", colour = "#169879", line = "-"},
        {column = "gas_mass_g", name = "Free gas", colour = "#dc398b", line = "-"},
        {column = "injected_mass_g", name = "Injected", colour = "black", line = "--"},
    ]
    """
)
STYLE = FigureStyle(
    **{
        **STYLE_TABLE,
        "washes": tuple(STYLE_TABLE["washes"]),
        "curves": tuple(Curve(**curve) for curve in STYLE_TABLE["curves"]),
    }
)
DOMAINS = [
    AnalysisDomain(kind="whole", domain_id="whole", name="Whole domain"),
    AnalysisDomain(kind="box", domain_id="low", name="low", bounds_mm=(0, 900, 0, 150)),
    AnalysisDomain(kind="layer", domain_id="layer:3", name="E layer 3", label=3),
]
HOURS = {
    "r1": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "r2": [0.01, 0.51, 1.01, 1.51, 2.01, 3.01],
    "r3": [0.0, 0.5, 0.98, 1.02, 1.5],
}
"""Frame times: the second run is recorded a little late, the third ends early and
has two frames for one target, as far before as after the first injection's end."""
TARGETS = {
    "r1": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "r2": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "r3": [0.0, 0.5, 1.0, 1.0, 1.5],
}
SCALE = {"r1": 1.0, "r2": 1.2, "r3": 0.8}
SHARES = {"total": 1.0, "aqueous": 0.7, "gas": 0.3}


def protocol(tmp_path):
    """0.6 g/h from a low port for an hour, in two steps, then 1.2 g/h from a high
    port for an hour."""
    path = tmp_path / "injection.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "location_x": [0.1, 0.1, 0.3],
            "location_y": [0.05, 0.05, 0.4],
            "start": ["2023-01-31 09:00", "2023-01-31 09:30", "2023-01-31 10:00"],
            "end": ["2023-01-31 09:30", "2023-01-31 10:00", "2023-01-31 11:00"],
            "rate_g/min": [0.01, 0.01, 0.02],
        }
    ).to_csv(path, index=False)
    return darsia.InjectionProtocol(path)


def ensemble(tmp_path, dtype=np.float64):
    """Three runs whose CO2 follows the injection, scaled per run, as rows of their
    tables of masses per domain; the masses are values of ``dtype``. The whole
    domain is never complete, and a mass of zero has no centre."""
    injection = protocol(tmp_path)
    runs = []
    for run, times_h in HOURS.items():
        rows, injected = [], {}
        for index, elapsed in enumerate(times_h):
            case = f"{run}-{index}"
            injected[case] = injection.injected_mass(time=elapsed) * 1000
            low = 0.6 * min(elapsed, 1.0)
            for domain, mass in (
                ("whole", injected[case]),
                ("low", low),
                ("layer:3", 0.5 * low),
            ):
                row = dict(run=run, case_id=case, elapsed_h=elapsed, domain_id=domain)
                for key, share in SHARES.items():
                    grams = float(dtype(SCALE[run] * 0.9 * mass * share))
                    row[key + "_quantified_mass_g"] = grams
                    row[key + "_mass_g"] = None if domain == "whole" else grams
                    if grams > 0:
                        row[key + "_centre_x_mm"] = (100.0 + 10 * index) * SCALE[run]
                        row[key + "_centre_y_mm"] = (50.0 + 5 * index) * SCALE[run]
                        row[key + "_spread_x_mm"], row[key + "_spread_y_mm"] = 3.0, 2.0
                rows.append(row)
        targets = dict(zip((f"{run}-{n}" for n in range(len(times_h))), TARGETS[run]))
        runs.append(EnsembleRun(rows, injected, targets, injection))
    return runs


def select(rows, **keys):
    """The rows with these values."""
    return [row for row in rows if all(row[k] == v for k, v in keys.items())]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_tables_of_a_synthetic_ensemble(dtype, tmp_path):
    """Per target time the summary over the runs' first frames for it; marks at
    each injection's end and the peak; rates in each injection's time; fractions
    of each domain's injected reference, a box's from the ports inside it; the
    whole rig's row, the whole domain's quantified CO2, whatever masses are read."""
    runs = ensemble(tmp_path, dtype)
    for row in runs[0].rows:
        if (row["case_id"], row["domain_id"]) == ("r1-1", "low"):
            row["aqueous_quantified_mass_g"] = float("inf")
    tables = ensemble_tables(runs, DOMAINS, SETTINGS)

    low = dict(domain="low", component="total")
    at_one = select(tables["mass_by_time"], time_h=1.0, quantity="mass_g", **low)[0]
    low_mass = (0.6, 0.6, 0.588)
    masses = [float(dtype(s * 0.9 * m)) for s, m in zip(SCALE.values(), low_mass)]
    assert (at_one["n"], at_one["median"]) == (3, np.median(masses))
    late = select(tables["mass_by_time"], time_h=3.0, **low)
    assert late[0]["n"] == 2 and late[1]["mean"] == pytest.approx(0.99, rel=1e-6)

    marks = tables["mass_at_marks"]
    end = {r["run"]: r for r in select(marks, mark="injection_end:1", **low)}
    assert (end["r2"]["case_id"], end["r3"]["case_id"]) == ("r2-2", "r3-2")
    assert end["r2"]["frame_gap_h"] == pytest.approx(0.01)
    assert end["r3"]["frame_gap_h"] == pytest.approx(-0.02)
    second = {r["run"]: r for r in select(marks, mark="injection_end:2", **low)}
    assert second["r3"]["case_id"] is None and second["r1"]["case_id"] == "r1-4"
    assert select(marks, mark="peak", run="r1", **low)[0]["case_id"] == "r1-2"
    peak = select(marks, mark="peak", run="r1", domain="low", component="aqueous")
    assert peak[0]["case_id"] == "r1-2"
    gap = select(tables["mass_marks_summary"], mark="injection_end:1", **low)[0]
    assert gap["largest_frame_gap_h"] == pytest.approx(0.02)
    assert {r["domain"]: r["injected_rule"] for r in marks} == {
        "full_rig": "whole domain: the mass injected up to the frame",
        "whole": "whole domain: the mass injected up to the frame",
        "low": "ports inside: injection 1",
        "layer:3": "total: this domain is not a box",
    }

    rate = select(tables["mass_rates"], run="r1", period="Injection 1", **low)[0]
    assert rate["n_frames"] == 2
    assert rate["rate_g_per_h"] == pytest.approx(0.54, rel=1e-6)

    write_tables(tmp_path, tables)
    with (tmp_path / "mass_long.csv").open(encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 4 * 3 * 17 and table[0]["fraction_of_injected"] == ""
    assert table[0]["domain"] == "full_rig" and table[0]["domain_label"] == "Whole rig"
    complete = EnsembleSettings(masses="complete", centre_component="total")
    whole = ensemble_tables(runs, DOMAINS[:1], complete)["mass_long"]
    assert {row["mass_g"] for row in select(whole, domain="whole")} == {None}
    # The whole rig keeps the whole domain's quantified CO2 with complete masses.
    rig = select(whole, domain="full_rig", case_id="r2-3", component="gas")[0]
    found = select(runs[1].rows, case_id="r2-3", domain_id="whole")[0]
    assert rig["mass_g"] == found["gas_quantified_mass_g"] > 0
    assert rig["fraction_of_injected"] == rig["mass_g"] / runs[1].injected_g["r2-3"]
    assert "full_rig" not in {row["domain"] for row in tables["centres_long"]}
    layers = ensemble_tables(runs, DOMAINS[2:], complete)["mass_long"]
    assert "full_rig" not in {row["domain"] for row in layers}


def test_the_injected_mass_of_the_frames_is_checked(tmp_path):
    """The largest difference of the frames' injected CO2 from the protocol's."""
    runs = ensemble(tmp_path)
    assert dose_difference(runs) == 0
    runs[0].injected_g["r1-0"] = float("nan")
    runs[1].injected_g["r2-3"] += 0.001
    assert dose_difference(runs) == pytest.approx(0.001)


def test_numpy_numbers_are_written_in_full(tmp_path):
    """A NumPy number in a table is written with all its digits, also under the
    legacy printing that a dependency of darsia sets."""
    with np.printoptions(legacy="1.13"):
        write_tables(
            tmp_path, {"t": [dict(a=np.float64(1 / 3), b=None, c=np.int64(2))]}
        )
    with (tmp_path / "t.csv").open(encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [dict(a=repr(1 / 3), b="", c="2")]


def test_summary_statistics_of_the_known_values():
    """Quartiles interpolate, the deviation is the sample's, and values that are
    not known are left out; a coefficient of variation needs a mean."""
    found = summary([1.0, None, 2.0, np.nan, 4.0, 10.0])
    assert [found[k] for k in ("n", "median", "q25", "q75")] == [4, 3.0, 1.75, 5.5]
    assert found["sd"] == pytest.approx(np.std([1, 2, 4, 10], ddof=1))
    assert summary([1.0])["sd"] is None and summary([-1.0, 1.0])["cv"] is None
    assert summary([None])["n"] == 0


def test_centres_of_a_synthetic_ensemble(tmp_path):
    """Each run's centre per frame where it has one, and per target time the runs'
    mean and median centre and mean spread."""
    tables = ensemble_tables(ensemble(tmp_path), DOMAINS, SETTINGS)

    long = select(tables["centres_long"], domain="low")
    assert len(long) == 3 * 15 and long[0]["case_id"] == "r1-1"
    centres = select(tables["centres_by_time"], domain="low", component="total")
    by_time = {row["time_h"]: row for row in centres}
    assert [by_time[t]["n_runs"] for t in (0.0, 1.0, 3.0)] == [1, 3, 2]
    assert by_time[1.0]["mean_x_mm"] == pytest.approx(120.0)
    assert by_time[1.0]["median_y_mm"] == pytest.approx(60.0)
    assert by_time[3.0]["mean_spread_x_mm"] == 3.0


def test_the_event_times_summarised_over_runs(tmp_path):
    """Per question the event times over the runs answered, per run and per
    injection, with the time resolution; the same from the event table's file."""
    runs = ensemble(tmp_path)
    series = [
        RunSeries.from_domain_mass(r.rows, r.injected_g, "quantified") for r in runs
    ]
    condition = dict(component="gas", event="detected", domain_id="low")
    injections = {s.run: r.protocol.injections() for s, r in zip(series, runs)}
    rows = event_table([EventQuestion("gas", [condition])], series, injections)
    found = event_summary(rows)

    per_run = select(found, mode="run", injection=None)
    assert [r["quantity"] for r in per_run] == ["elapsed_h", "after_injection_h"]
    assert (per_run[0]["n_runs"], per_run[0]["median"]) == (3, 0.5)
    assert per_run[1]["median"] == -0.5
    assert per_run[0]["resolution_max_h"] == pytest.approx(0.5)
    first = select(found, mode="injection", injection=1)[0]
    assert (first["n"], first["min"], first["max"]) == (3, 0.01, 0.5)
    write_events(tmp_path / "events.csv", rows)
    assert event_summary(read_events(tmp_path / "events.csv")) == found


def test_the_figures_are_written(tmp_path):
    """The mass, fraction and trajectory figures, each with and without title, at
    the style's resolution."""
    runs = ensemble(tmp_path)
    tables = ensemble_tables(runs, DOMAINS[:2], SETTINGS)
    files = ensemble_figures(tmp_path, tables, runs, SETTINGS, STYLE)

    assert len(files) == 2 * (3 * 3 + 3 + 2)
    assert "mass_whole-domain_gas_notitle.png" in files
    assert "mass_whole-rig_gas_notitle.png" in files
    assert "trajectory_low_total.png" in files
    assert all((tmp_path / name).stat().st_size > 0 for name in files)
    alike = [*DOMAINS[:2], AnalysisDomain("layer", "layer:3", "Low", label=3)]
    with pytest.raises(ValueError, match="named alike"):
        ensemble_figures(
            tmp_path, ensemble_tables(runs, alike, SETTINGS), runs, SETTINGS, STYLE
        )
    coarse = tmp_path / "coarse"
    coarse.mkdir()
    ensemble_figures(coarse, tables, runs, SETTINGS, replace(STYLE, dpi=75))
    name = "trajectory_low_total_notitle.png"
    heights = [imread(folder / name).shape[0] for folder in (tmp_path, coarse)]
    assert heights[1] == pytest.approx(heights[0] / 2, abs=2)
