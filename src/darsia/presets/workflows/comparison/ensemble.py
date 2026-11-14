"""Ensemble statistics: the CO2 of a series' runs, domain by domain.

The ensemble reads what the runs' analyses finished (:class:`EnsembleRun`): each
run's table of masses per domain (:class:`DomainMass`), the CO2 injected up to
each frame, each frame's production target and the run's injection protocol. It
computes no map or mass again. Its tables hold one observation per row, the unit
in each column's name:

- ``mass_long``: every run, frame, domain and component: its mass, its injected
  reference and the fraction of it;
- ``mass_by_time``: per domain, component and quantity (the mass, or the fraction
  of the injected reference), the summary over the runs at each production
  target time, from each run's first frame recorded for that target;
- ``mass_at_marks`` and ``mass_marks_summary``: each run's value at each mark, the
  end of each injection and the peak, with the frame used and its gap to the
  mark; and their summary;
- ``mass_rates`` and ``mass_rates_summary``: each run's least-squares rate over
  the frames of each injection's time, from its start to its end; and their
  summary;
- ``centres_long`` and ``centres_by_time``: each run's centre and spread of the
  quantified mass per frame, domain and component, in mm of the grid; and per
  time the runs' mean and median centre, their standard deviation and the mean
  spread;

A fraction divides by the domain's injected reference
(:func:`injected_reference`): the CO2 injected up to the frame for the whole
domain; for a box, the CO2 injected by the ports inside it, from the injection
protocol; for any other domain, the CO2 injected in all. A mark takes the run's
frame nearest to it, while the mark lies within the run's frame times; the peak
is the frame of the largest mass, the first on a tie.

The summaries (:func:`summary`) are the count, median, quartiles (interpolated
linearly), interquartile range, mean, sample standard deviation, coefficient of
variation and range of the known values; they and the rates run in double
precision, as our production summarised. The tables are written by
:func:`write_tables`.

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import darsia
from darsia.presets.workflows.analysis.domain_mass import RunSeries
from darsia.presets.workflows.analysis.domains import (
    COMPONENTS,
    AnalysisDomain,
    injected_mass,
    write_rows,
)

STATISTICS = ("n", "median", "q25", "q75", "iqr", "mean", "sd", "cv", "min", "max")
"""The summary statistics of a set of values, in the unit of the values."""
QUANTITIES = ("mass_g", "fraction_of_injected")
"""The mass [g] and its fraction of the injected reference."""
PEAK = "peak"
"""The mark of a run's largest mass."""


@dataclass(frozen=True)
class EnsembleSettings:
    """Settings of the ensemble statistics."""

    masses: Literal["complete", "quantified"]
    """The masses read from the tables of masses per domain: a domain's total, which
    a domain holding an unknown pixel lacks, or its quantified mass, the known part."""
    centre_component: Literal["total", "aqueous", "gas"]
    """The component whose centres the trajectory figures draw."""


@dataclass(frozen=True)
class EnsembleRun:
    """A run of the ensemble, as its finished analyses give it."""

    rows: Sequence[Mapping]
    """Its rows of the table of masses per domain (:class:`DomainMass`)."""
    injected_g: Mapping[str, float]
    """CO2 injected up to each frame [g], by case id."""
    target_h: Mapping[str, float]
    """Each frame's production target [h since the injection start], by case id."""
    protocol: darsia.InjectionProtocol
    """Its injection protocol."""


def summary(values: Iterable[float | None]) -> dict:
    """The summary statistics (:data:`STATISTICS`) of the known values.

    A statistic that needs more values than there are, or a coefficient of
    variation at zero mean, is None.

    """
    data = np.array([np.nan if v is None else v for v in values], np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return dict(n=0, **dict.fromkeys(STATISTICS[1:]))
    q25, median, q75 = (float(value) for value in np.percentile(data, [25, 50, 75]))
    mean = float(data.mean())
    sd = float(data.std(ddof=1)) if data.size > 1 else None
    return dict(
        n=int(data.size),
        median=median,
        q25=q25,
        q75=q75,
        iqr=q75 - q25,
        mean=mean,
        sd=sd,
        cv=None if sd is None or mean == 0 else sd / mean,
        min=float(data.min()),
        max=float(data.max()),
    )


def linear_rate(
    time_h: Sequence[float], values: Sequence[float | None]
) -> float | None:
    """The least-squares slope [per hour] of the known values against their times
    [h]; None without two distinct times."""
    pairs = [(h, v) for h, v in zip(time_h, values) if v is not None and np.isfinite(v)]
    if len({h for h, _ in pairs}) < 2:
        return None
    x, y = np.asarray(pairs, np.float64).T
    return float(np.polyfit(x, y, 1)[0])


def _value(value) -> float | None:
    """A number of an array as a table value: None where it is not finite."""
    return float(value) if np.isfinite(value) else None


def _ratio(mass: float | None, reference: float | None) -> float | None:
    """mass / reference; None without both, or without a positive reference."""
    if mass is None or reference is None or not reference > 0:
        return None
    return mass / reference


def _mark_frame(elapsed_h: np.ndarray, mark_h: float) -> int | None:
    """The frame nearest to a mark, the earlier on a tie; None for a mark outside
    the run's frame times."""
    if not elapsed_h[0] <= mark_h <= elapsed_h[-1]:
        return None
    return int(np.argmin(np.abs(elapsed_h - mark_h)))


def _injected_g(protocol: darsia.InjectionProtocol, time_h: float) -> float:
    """CO2 [g] the protocol injected up to a time [h]."""
    return float(protocol.injected_mass(time=time_h)) * 1000


def injected_reference(
    series: RunSeries, domain: AnalysisDomain, protocol: darsia.InjectionProtocol
) -> tuple[list[float | None], str]:
    """The CO2 a domain's fractions divide by, at each frame, and its rule.

    Parameters
    ----------
    series : RunSeries
        The run's masses per domain, with the CO2 injected up to each frame.
    domain : AnalysisDomain
        The domain.
    protocol : darsia.InjectionProtocol
        The run's injection protocol.

    Returns
    -------
    tuple
        Per frame the reference [g], None where not known, and the rule: the CO2
        injected up to the frame for the whole domain; for a box, the CO2 injected
        by the ports inside it (the domains' injected mass,
        :func:`~darsia.presets.workflows.analysis.domains.injected_mass`); for a
        domain without an injection of its own the CO2 injected in all.

    """
    injected = [_value(value) for value in series.injected_g]
    if domain.kind == "whole":
        return injected, "whole domain: the mass injected up to the frame"
    own = [injected_mass(domain, protocol, time_h) for time_h in series.elapsed_h]
    if None in own:
        return injected, "total: this domain is not a box"
    values = [float(value) * 1000 for value in own]
    x0, x1, y0, y1 = domain.bounds_m
    ports = protocol.injections()
    inside = (ports["location_x"].between(x0, x1)) & ports["location_y"].between(y0, y1)
    numbers = ", ".join(str(n) for n in ports["number"][inside])
    if not numbers:
        return values, "ports inside: no injection port lies in this box"
    return values, "ports inside: injection " + numbers


def dose_difference(runs: Iterable[EnsembleRun]) -> float | None:
    """The largest difference [g] of the CO2 injected up to the runs' frames from
    their protocols' injected mass (:meth:`darsia.InjectionProtocol.injected_mass`),
    over the frames where both are known; None without such a frame. A difference
    means that the whole domain's fractions and a box's fractions divide by
    different injections."""
    gaps = []
    for run in runs:
        elapsed_h = {str(row["case_id"]): float(row["elapsed_h"]) for row in run.rows}
        for case, elapsed in elapsed_h.items():
            given = run.injected_g.get(case)
            if given is not None:
                gaps.append(abs(_injected_g(run.protocol, elapsed) - given))
    known = [gap for gap in gaps if np.isfinite(gap)]
    return max(known) if known else None


@dataclass(frozen=True)
class _Run:
    """A run as the tables read it."""

    series: RunSeries
    """Its masses and centres per domain."""
    targets: Mapping[str, float]
    """Each frame's production target [h], by case id."""
    references: dict[str, tuple[list[float | None], str]]
    """The injected reference and its rule, by domain identity."""
    periods: list[tuple[str, float, float]]
    """Each injection's label, start and end [h]."""
    marks: list[tuple[str, str, float]]
    """Each injection's end: its key, label and time [h]."""

    @classmethod
    def read(cls, run: EnsembleRun, domains, masses) -> _Run:
        """The run's series, injected references, periods and marks."""
        series = RunSeries.from_domain_mass(run.rows, run.injected_g, masses)
        references = {
            domain.domain_id: injected_reference(series, domain, run.protocol)
            for domain in domains
            if domain.domain_id in series.masses
        }
        injections = list(run.protocol.injections().itertuples())
        return cls(
            series=series,
            targets=run.target_h,
            references=references,
            periods=[(f"Injection {i.number}", i.start_h, i.end_h) for i in injections],
            marks=[
                (f"injection_end:{i.number}", f"End of injection {i.number}", i.end_h)
                for i in injections
            ],
        )


def _axis(runs: Sequence[_Run]) -> dict[float, dict[str, int]]:
    """Each production target time, with each run's first frame recorded for it."""
    found: dict[float, dict[str, int]] = {}
    for run in runs:
        for index, case in enumerate(run.series.case_ids):
            target = run.targets.get(case)
            if target is not None:
                found.setdefault(target, {}).setdefault(run.series.run, index)
    return dict(sorted(found.items()))


def _period(time_h: float, periods: Sequence[tuple[str, float, float]]) -> str:
    """The period holding a time, each half-open; "" for none."""
    return next((label for label, start, end in periods if start <= time_h < end), "")


def ensemble_tables(
    runs: Sequence[EnsembleRun],
    domains: Sequence[AnalysisDomain],
    settings: EnsembleSettings,
) -> dict[str, list[dict]]:
    """The ensemble's tables of the masses and centres of the runs' domains.

    Parameters
    ----------
    runs : sequence of EnsembleRun
        The runs, in the order of the rows.
    domains : sequence of AnalysisDomain
        The domains summarised, in the order of the rows.
    settings : EnsembleSettings
        The masses read.

    Returns
    -------
    dict
        Each table by its name (see the module), as a list of rows; None where a
        value is not defined.

    """
    read = [_Run.read(run, domains, settings.masses) for run in runs]
    axis = _axis(read)
    return dict(
        mass_long=_long(read, domains),
        mass_by_time=_by_time(read, domains, axis),
        **_marks(read, domains),
        **_rates(read, domains),
        **_centres(read, domains, axis),
    )


def _long(runs: Sequence[_Run], domains) -> list[dict]:
    """Every run, frame, domain and component."""
    rows = []
    for run in runs:
        series = run.series
        for index, case in enumerate(series.case_ids):
            elapsed_h = float(series.elapsed_h[index])
            period = _period(elapsed_h, run.periods)
            for domain in domains:
                if domain.domain_id not in run.references:
                    continue
                reference = run.references[domain.domain_id][0][index]
                for component in COMPONENTS:
                    mass = _value(series.masses[domain.domain_id][component][index])
                    rows.append(
                        dict(
                            run=series.run,
                            case_id=case,
                            elapsed_h=elapsed_h,
                            target_h=run.targets.get(case),
                            period=period,
                            domain=domain.domain_id,
                            domain_label=domain.name,
                            component=component,
                            mass_g=mass,
                            injected_g=reference,
                            fraction_of_injected=_ratio(mass, reference),
                        )
                    )
    return rows


def _quantity(run: _Run, domain: str, component: str, frame: int, quantity: str):
    """One run's mass or fraction of the injected reference at a frame."""
    if domain not in run.references:
        return None
    mass = _value(run.series.masses[domain][component][frame])
    if quantity == "mass_g":
        return mass
    return _ratio(mass, run.references[domain][0][frame])


def _by_time(runs: Sequence[_Run], domains, axis) -> list[dict]:
    """The summary per domain, component, quantity and production target time."""
    by_run = {run.series.run: run for run in runs}
    rows = []
    for domain in domains:
        for component in COMPONENTS:
            for quantity in QUANTITIES:
                for time, frames in axis.items():
                    values = [
                        _quantity(
                            by_run[run], domain.domain_id, component, frame, quantity
                        )
                        for run, frame in frames.items()
                    ]
                    rows.append(
                        dict(
                            domain=domain.domain_id,
                            domain_label=domain.name,
                            component=component,
                            quantity=quantity,
                            time_h=time,
                            **summary(values),
                        )
                    )
    return rows


def _marks(runs: Sequence[_Run], domains) -> dict[str, list[dict]]:
    """Each run's value at each mark, and their summary."""
    rows = []
    for run in runs:
        series, elapsed_h = run.series, run.series.elapsed_h
        for domain in domains:
            if domain.domain_id not in run.references:
                continue
            reference, rule = run.references[domain.domain_id]
            for component in COMPONENTS:
                values = series.masses[domain.domain_id][component]
                chosen = [
                    (mark, label, at, _mark_frame(elapsed_h, at))
                    for mark, label, at in run.marks
                ]
                finite = np.isfinite(values)
                if finite.any():
                    peak = int(np.argmax(np.where(finite, values, -np.inf)))
                    chosen.append((PEAK, "Peak", float(elapsed_h[peak]), peak))
                for mark, label, at, frame in chosen:
                    known = frame is not None
                    elapsed = float(elapsed_h[frame]) if known else None
                    mass = _value(values[frame]) if known else None
                    rows.append(
                        dict(
                            run=series.run,
                            mark=mark,
                            mark_label=label,
                            mark_h=at,
                            case_id=series.case_ids[frame] if known else None,
                            elapsed_h=elapsed,
                            frame_gap_h=elapsed - at if known else None,
                            domain=domain.domain_id,
                            domain_label=domain.name,
                            component=component,
                            mass_g=mass,
                            injected_g=reference[frame] if known else None,
                            fraction_of_injected=(
                                _ratio(mass, reference[frame]) if known else None
                            ),
                            injected_rule=rule,
                        )
                    )
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = ("mark", "mark_label", "domain", "domain_label", "component")
        groups.setdefault(tuple(row[k] for k in key), []).append(row)
    summaries = []
    for (mark, label, domain, name, component), found in groups.items():
        gaps = [
            abs(row["frame_gap_h"]) for row in found if row["frame_gap_h"] is not None
        ]
        for quantity in QUANTITIES:
            summaries.append(
                dict(
                    mark=mark,
                    mark_label=label,
                    domain=domain,
                    domain_label=name,
                    component=component,
                    quantity=quantity,
                    n_runs=len(found),
                    **summary(row[quantity] for row in found),
                    largest_frame_gap_h=max(gaps) if gaps else None,
                )
            )
    return dict(mass_at_marks=rows, mass_marks_summary=summaries)


def _rates(runs: Sequence[_Run], domains) -> dict[str, list[dict]]:
    """Each run's rate in each injection's time, and their summary."""
    rows, rates = [], {}
    for run in runs:
        series, elapsed_h = run.series, run.series.elapsed_h
        for label, start, end in run.periods:
            frames = np.flatnonzero((elapsed_h >= start) & (elapsed_h < end))
            for domain in domains:
                if domain.domain_id not in run.references:
                    continue
                for component in COMPONENTS:
                    values = series.masses[domain.domain_id][component][frames]
                    rate = linear_rate(elapsed_h[frames].tolist(), values.tolist())
                    rows.append(
                        dict(
                            run=series.run,
                            period=label,
                            start_h=start,
                            end_h=end,
                            n_frames=int(np.isfinite(values).sum()),
                            domain=domain.domain_id,
                            domain_label=domain.name,
                            component=component,
                            rate_g_per_h=rate,
                        )
                    )
                    key = (label, domain.domain_id, domain.name, component)
                    rates.setdefault(key, []).append(rate)
    summaries = [
        dict(
            period=label,
            domain=domain,
            domain_label=name,
            component=component,
            quantity="rate_g_per_h",
            n_runs=len(values),
            **summary(values),
        )
        for (label, domain, name, component), values in rates.items()
    ]
    return dict(mass_rates=rows, mass_rates_summary=summaries)


def _centres(runs: Sequence[_Run], domains, axis) -> dict[str, list[dict]]:
    """Each run's centres per frame, domain and component, and their summary."""
    rows = []
    for run in runs:
        series = run.series
        for index, case in enumerate(series.case_ids):
            elapsed_h = float(series.elapsed_h[index])
            for domain in domains:
                if domain.domain_id not in series.centres:
                    continue
                for component in COMPONENTS:
                    centre = series.centres[domain.domain_id][component][index]
                    if not np.isfinite(centre[1:]).all():
                        continue
                    mass, x, y, spread_x, spread_y = (_value(v) for v in centre)
                    rows.append(
                        dict(
                            run=series.run,
                            case_id=case,
                            elapsed_h=elapsed_h,
                            target_h=run.targets.get(case),
                            period=_period(elapsed_h, run.periods),
                            domain=domain.domain_id,
                            domain_label=domain.name,
                            component=component,
                            quantified_mass_g=mass,
                            centre_x_mm=x,
                            centre_y_mm=y,
                            spread_x_mm=spread_x,
                            spread_y_mm=spread_y,
                        )
                    )
    frames = {run.series.run: run.series.case_ids for run in runs}
    found = {(r["run"], r["case_id"], r["domain"], r["component"]): r for r in rows}
    keys = dict.fromkeys((r["domain"], r["domain_label"], r["component"]) for r in rows)
    by_time = []
    for domain, name, component in keys:
        for time, chosen in axis.items():
            picked = [
                found.get((run, frames[run][index], domain, component))
                for run, index in chosen.items()
            ]
            picked = [row for row in picked if row is not None]
            if not picked:
                continue
            x, y = (
                summary(row[k] for row in picked)
                for k in ("centre_x_mm", "centre_y_mm")
            )
            by_time.append(
                dict(
                    domain=domain,
                    domain_label=name,
                    component=component,
                    time_h=time,
                    n_runs=len(picked),
                    mean_x_mm=x["mean"],
                    mean_y_mm=y["mean"],
                    median_x_mm=x["median"],
                    median_y_mm=y["median"],
                    sd_x_mm=x["sd"],
                    sd_y_mm=y["sd"],
                    mean_spread_x_mm=summary(r["spread_x_mm"] for r in picked)["mean"],
                    mean_spread_y_mm=summary(r["spread_y_mm"] for r in picked)["mean"],
                )
            )
    return dict(centres_long=rows, centres_by_time=by_time)


def write_tables(folder: Path, tables: Mapping[str, list[dict]]) -> list[Path]:
    """Write each table as ``<name>.csv`` in its rows' columns; None is an empty cell,
    and a NumPy number is written as the Python number it holds, in full
    (:func:`~darsia.presets.workflows.analysis.domains.write_rows`).

    Returns
    -------
    list of Path
        The files, in the order of the tables.

    """
    paths = []
    for name, rows in tables.items():
        path = Path(folder) / f"{name}.csv"
        write_rows(path, rows, list(rows[0]) if rows else [])
        paths.append(path)
    return paths
