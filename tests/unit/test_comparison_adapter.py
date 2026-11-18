"""Small in-process comparisons consume frozen targets and accepted event results."""

import csv
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from darsia.jobs.job import _ordered, _reads
from darsia.jobs.step import Selection, StepContext, Task
from darsia.presets.workflows.analysis.domain_mass import write_domain_mass
from darsia.presets.workflows.analysis.steps import DOMAIN_MASS, DomainSettings
from darsia.presets.workflows.comparison import steps
from darsia.presets.workflows.comparison.comparison_events import write_events
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.utils import results
from darsia.utils.locations import Frame, Locations


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _context(tmp_path, monkeypatch, targets=None):
    """Real accepted CSV inputs, context, input planning and output publication."""
    selection = Selection(
        {run: (Frame("session", "image.ARW"),) for run in ("a", "b")},
        domains=("sand",),
    )
    locations = Locations(tmp_path, Path("results"), {})
    locations.results.mkdir()
    assigned = {run: {"session/image.ARW": 2.0} for run in selection.frames}
    tables = {
        "ensemble": {
            "masses": "quantified",
            "centre_component": "total",
            "target_h": assigned if targets is None else targets,
        },
        "domains": {"domains": {"sand": {"kind": "sand", "name": "Sand"}}},
        "style": {},
    }
    bound = {}
    for run, elapsed, grams in (("a", 1.98, 1.0), ("b", 2.03, 3.0)):
        row = dict(
            run=run, case_id="session/image.ARW", elapsed_h=elapsed, domain_id="sand"
        )
        for component, share in (("total", 1.0), ("aqueous", 0.7), ("gas", 0.3)):
            row[component + "_quantified_mass_g"] = grams * share
        key = f"domain-mass/{run}"
        token = f"mass{run}"
        results.publish(
            locations.result("domain-mass", run=run),
            lambda folder, row=row: write_domain_mass(
                folder / steps.DOMAIN_MASS_TABLE, [row]
            ),
            {},
            token,
            root=locations.results,
        )
        bound[key] = token
    events = [
        dict(
            question="arrival",
            mode="run",
            run=run,
            injection=None,
            case_id="session/image.ARW" if elapsed is not None else None,
            elapsed_h=elapsed,
            after_injection_h=elapsed,
            resolution_h=0.1 if elapsed is not None else None,
            note="" if elapsed is not None else "unknown",
        )
        for run, elapsed in (("a", 1.98), ("b", None))
    ]
    results.publish(
        locations.result("events"),
        lambda folder: write_events(folder / "events.csv", events),
        {},
        "eventsaccepted",
        root=locations.results,
    )
    bound["events"] = "eventsaccepted"
    kinds = {s.name: s for s in (DOMAIN_MASS, steps.EVENTS, steps.ENSEMBLE)}
    inputs = _reads(steps.ENSEMBLE, None, (), None, kinds, selection, tables)
    task = Task("ensemble", "ensemble", None, (), None, tuple(inputs), ())
    protocol = SimpleNamespace(
        injected_mass=lambda time: time * 0.001,
        injections=lambda: pd.DataFrame(
            {"number": [1], "start_h": [0.0], "end_h": [1.0]}
        ),
    )
    monkeypatch.setattr(steps, "run_settings", lambda *args: None)
    monkeypatch.setattr(
        steps,
        "make_experiment",
        lambda config: SimpleNamespace(injection_protocol=protocol),
    )
    monkeypatch.setattr(steps, "ensemble_figures", lambda *args: None)
    settings = {
        "ensemble": settings_from_table(
            steps.EnsembleInputSettings, tables["ensemble"]
        ),
        "domains": DomainSettings(**tables["domains"]),
        "style": None,
    }
    ctx = StepContext(
        steps.ENSEMBLE,
        task,
        selection,
        (),
        tables,
        settings,
        locations,
        "ensembleaccepted",
        {"id": "comparison", "environment": {}, "bound": bound},
        lambda name: frozenset(),
        "testsource",
    )
    return ctx


def test_shared_targets_and_finished_events_survive_adapter_publication(
    tmp_path, monkeypatch
):
    ctx = _context(tmp_path, monkeypatch)
    ordered = _ordered((steps.ENSEMBLE, steps.EVENTS))
    assert [step.name for step in ordered] == ["events", "ensemble"]
    assert [i.key for i in ctx.task.inputs] == [
        "domain-mass/a",
        "domain-mass/b",
        "events",
    ]
    ctx.step.run(ctx)
    folder = ctx.locations.result("ensemble")
    mass = [
        row
        for row in _csv(folder / "mass_by_time.csv")
        if row["component"] == "total" and row["quantity"] == "mass_g"
    ]
    assert len(mass) == 1
    assert float(mass[0]["time_h"]) == 2.0
    assert int(mass[0]["n"]) == 2
    assert float(mass[0]["mean"]) == 2.0
    actual = {
        row["run"]: (
            float(row["elapsed_h"]),
            float(row["target_h"]),
            float(row["injected_g"]),
        )
        for row in _csv(folder / "mass_long.csv")
    }
    assert actual == {"a": (1.98, 2.0, 1.98), "b": (2.03, 2.0, 2.03)}
    event = next(
        row
        for row in _csv(folder / "events_summary.csv")
        if row["quantity"] == "elapsed_h"
    )
    assert event["question"] == "arrival"
    assert int(event["n_runs"]) == 2
    assert int(event["n"]) == 1
    assert float(event["mean"]) == 1.98
    assert float(event["resolution_max_h"]) == 0.1
    stored = results.read_description(folder)
    assert stored["inputs"]["events"] == "eventsaccepted"
    assert (
        stored["settings"]["ensemble"]["target_h"] == ctx.tables["ensemble"]["target_h"]
    )
    changed = deepcopy(stored)
    changed["settings"]["ensemble"]["target_h"]["a"]["session/image.ARW"] = 2.1
    assert results.reusable(stored, changed) is not None


def test_an_unaccepted_events_result_cannot_publish_an_ensemble(tmp_path, monkeypatch):
    ctx = _context(tmp_path, monkeypatch)
    ctx._job["bound"]["events"] = "otherattempt"
    with pytest.raises(PermissionError, match="events holds token"):
        ctx.step.run(ctx)
    assert not ctx.locations.result("ensemble").exists()


@pytest.mark.parametrize("targets", [{}, {"a": {"session/image.ARW": 2.0}}])
def test_adapter_refuses_a_missing_target_instead_of_using_actual_time(
    tmp_path, monkeypatch, targets
):
    ctx = _context(tmp_path, monkeypatch, targets)
    with pytest.raises(ValueError, match="Missing frozen targets"):
        ctx.step.run(ctx)
    assert not ctx.locations.result("ensemble").exists()


def test_legacy_ensemble_configuration_requires_explicit_targets():
    with pytest.raises(ValueError, match="lacks.*target_h"):
        settings_from_table(
            steps.EnsembleInputSettings,
            {"masses": "quantified", "centre_component": "total"},
        )


def _selected(tmp_path):
    frames = (Frame("first", "same.ARW"), Frame("second", "same.ARW"))
    selection = Selection({"run": frames})
    locations = Locations(
        tmp_path,
        Path("results"),
        {"run": {"first": Path("day1"), "second": Path("day2")}},
    )
    matches = {
        "run": [
            (locations.photo("run", frames[0]), 2.0),
            (locations.photo("run", frames[1]), 3.0),
        ]
    }
    return selection, locations, matches


def test_job_authoring_joins_exact_source_identity_not_photo_basename(tmp_path):
    selection, locations, matches = _selected(tmp_path)
    assert steps.ensemble_targets(selection, locations, matches) == {
        "run": {"first/same.ARW": 2.0, "second/same.ARW": 3.0}
    }


@pytest.mark.parametrize(
    "problem",
    ["missing", "duplicate", "conflicting", "unselected", "aliased", "nonfinite"],
)
def test_job_authoring_refuses_missing_or_ambiguous_provenance(tmp_path, problem):
    selection, locations, matches = _selected(tmp_path)
    if problem == "missing":
        matches["run"].pop()
    elif problem in {"duplicate", "conflicting"}:
        path, target = matches["run"][0]
        matches["run"].append((path, target if problem == "duplicate" else 4.0))
    elif problem == "unselected":
        matches["run"][0] = (tmp_path / "different" / "same.ARW", 2.0)
    elif problem == "aliased":
        locations.sessions["run"]["second"] = locations.sessions["run"]["first"]
    else:
        path, _ = matches["run"][0]
        matches["run"][0] = (path, float("nan"))
    with pytest.raises(ValueError, match="Missing|Ambiguous|Unselected|Invalid"):
        steps.ensemble_targets(selection, locations, matches)
