"""Read Bronkhorst FlowPlot ``plotdata*.txt`` logs of a mass-flow controller.

FlowPlot writes one line per sample, with a time column for every channel::

    t1;value1;t2;value2;t3;value3

Times are seconds since logging started. With the default FlowPlot setup used
in the FluidFlower labs the channels are *measure*, *setpoint* and *valve
output*, all in percent of full scale. The start time is encoded as
``YYYYMMDDhhmmss`` at the end of the file name, for example
``plotdata20260223175501.txt`` or ``plotdata_eks_120260420103333.txt`` (prefix
``eks_1`` followed by the time stamp).

Note that FlowPlot only logs the controller. A separate shut-off valve
between the controller and the rig is invisible here, so flow in the log is
not necessarily flow into the rig.

The module only needs the standard library, numpy and pandas and runs as a
stand-alone script on Python 3.10::

    python flowplot_log.py summary plotdata20260223175501.txt --full-scale-ml-min 10
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

FORMAT = "darsia-fluidflower-flowplot-v1"
DEFAULT_CHANNELS = ("measure_percent", "setpoint_percent", "valve_percent")
_TRAILING_DIGITS = re.compile(r"(\d+)$")


def start_time_from_name(path: str | Path) -> dt.datetime:
    """Start time encoded in the last 14 digits of the file name."""
    stem = Path(path).stem
    match = _TRAILING_DIGITS.search(stem)
    if match is None or len(match[1]) < 14:
        raise ValueError(
            f"No YYYYMMDDhhmmss time stamp at the end of {Path(path).name}"
        )
    return dt.datetime.strptime(match[1][-14:], "%Y%m%d%H%M%S")


@dataclasses.dataclass(frozen=True, eq=False)
class FlowPlotLog:
    """A parsed FlowPlot log.

    ``frame`` has ``datetime``, ``elapsed_s`` (time column of the first
    channel) and one column per channel. ``max_channel_skew_s`` is the largest
    difference between the time columns of the channels.
    """

    source_path: str
    start_time: dt.datetime
    channels: tuple[str, ...]
    frame: pd.DataFrame
    max_channel_skew_s: float
    full_scale_ml_min: float | None

    @property
    def duration_s(self) -> float:
        return float(self.frame["elapsed_s"].iloc[-1])

    @property
    def sampling_s(self) -> float:
        steps = np.diff(self.frame["elapsed_s"].to_numpy())
        return float(np.median(steps)) if steps.size else float("nan")

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "format": FORMAT,
            "source_path": self.source_path,
            "start_time": self.start_time.isoformat(sep=" "),
            "end_time": (
                self.start_time + dt.timedelta(seconds=self.duration_s)
            ).isoformat(sep=" "),
            "n_samples": int(len(self.frame)),
            "duration_s": self.duration_s,
            "sampling_s": self.sampling_s,
            "max_channel_skew_s": self.max_channel_skew_s,
            "full_scale_ml_min": self.full_scale_ml_min,
        }
        for channel in self.channels:
            values = self.frame[channel]
            info[f"{channel}_min"] = float(values.min())
            info[f"{channel}_max"] = float(values.max())
        return info


def read_plotdata(
    path: str | Path,
    *,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    start_time: dt.datetime | None = None,
    full_scale_ml_min: float | None = None,
) -> FlowPlotLog:
    """Read a FlowPlot log.

    ``start_time`` overrides the time stamp in the file name. With
    ``full_scale_ml_min`` a ``flow_ml_min`` column is added from the first
    channel (the measured flow).
    """
    path = Path(path)
    raw = pd.read_csv(path, sep=";", header=None, decimal=".", engine="c")
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if raw.shape[1] % 2:
        raise ValueError(f"Expected pairs of time and value columns in {path.name}")
    n_channels = raw.shape[1] // 2
    if len(channels) != n_channels:
        raise ValueError(
            f"{path.name} has {n_channels} channels but {len(channels)} names were given"
        )
    values = raw.to_numpy(dtype=float)
    times = values[:, 0::2]
    skew = float(np.max(np.abs(times - times[:, :1]))) if n_channels > 1 else 0.0
    start = start_time if start_time is not None else start_time_from_name(path)
    elapsed = times[:, 0]
    frame = pd.DataFrame(
        {
            "datetime": np.asarray(
                pd.Timestamp(start) + pd.to_timedelta(elapsed, unit="s"),
                dtype="datetime64[ns]",
            ),
            "elapsed_s": elapsed,
        }
    )
    for index, name in enumerate(channels):
        frame[name] = values[:, 2 * index + 1]
    if full_scale_ml_min is not None:
        frame["flow_ml_min"] = frame[channels[0]] / 100.0 * float(full_scale_ml_min)
    return FlowPlotLog(
        source_path=str(path),
        start_time=start,
        channels=tuple(channels),
        frame=frame,
        max_channel_skew_s=skew,
        full_scale_ml_min=full_scale_ml_min,
    )


def episodes(
    log: FlowPlotLog,
    *,
    channel: str | None = None,
    threshold: float = 50.0,
    min_duration_s: float = 30.0,
    merge_gap_s: float = 5.0,
) -> pd.DataFrame:
    """Intervals where ``channel`` (default: measured flow) is at or above ``threshold``."""
    channel = channel or log.channels[0]
    frame = log.frame
    t = frame["elapsed_s"].to_numpy()
    v = frame[channel].to_numpy()
    above = np.flatnonzero(v >= threshold)
    columns = [
        "start",
        "end",
        "duration_s",
        "mean",
        "max",
        "time_of_max",
        "integral_percent_s",
    ]
    if above.size == 0:
        return pd.DataFrame(columns=columns)
    breaks = np.flatnonzero(np.diff(t[above]) > merge_gap_s)
    starts = np.concatenate([[above[0]], above[breaks + 1]])
    ends = np.concatenate([above[breaks], [above[-1]]])
    rows = []
    for a, b in zip(starts, ends):
        if t[b] - t[a] < min_duration_s:
            continue
        segment = slice(a, b + 1)
        peak = a + int(np.argmax(v[segment]))
        rows.append(
            {
                "start": frame["datetime"].iloc[a],
                "end": frame["datetime"].iloc[b],
                "duration_s": float(t[b] - t[a]),
                "mean": float(np.mean(v[segment])),
                "max": float(v[peak]),
                "time_of_max": frame["datetime"].iloc[peak],
                "integral_percent_s": (
                    float(np.trapezoid(v[segment], t[segment]))
                    if hasattr(np, "trapezoid")
                    else float(np.trapz(v[segment], t[segment]))
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def setpoint_changes(
    log: FlowPlotLog, *, channel: str | None = None, min_change: float = 0.5
) -> pd.DataFrame:
    """Time stamps where the setpoint changes by at least ``min_change``."""
    channel = channel or (log.channels[1] if len(log.channels) > 1 else log.channels[0])
    v = log.frame[channel].to_numpy()
    index = np.flatnonzero(np.abs(np.diff(v)) >= min_change) + 1
    return pd.DataFrame(
        {
            "datetime": log.frame["datetime"].to_numpy()[index],
            "elapsed_s": log.frame["elapsed_s"].to_numpy()[index],
            "from": v[index - 1],
            "to": v[index],
        }
    )


def resample(log: FlowPlotLog, period_s: float = 1.0) -> pd.DataFrame:
    """Mean and maximum of every channel in fixed bins (bin start times)."""
    period = pd.Timedelta(seconds=period_s)
    bins = log.frame["datetime"].dt.floor(period)
    columns = [c for c in log.frame.columns if c not in ("datetime", "elapsed_s")]
    grouped = log.frame.groupby(bins, sort=True)[columns]
    mean = grouped.mean()
    peak = grouped.max().add_suffix("_max")
    return pd.concat([mean, peak], axis=1).reset_index(names="datetime")


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser(
        "summary", help="Describe FlowPlot logs and list flow episodes"
    )
    summary.add_argument("paths", nargs="+", type=Path)
    summary.add_argument("--full-scale-ml-min", type=float, default=None)
    summary.add_argument(
        "--threshold", type=float, default=50.0, help="Percent for flow episodes"
    )
    args = parser.parse_args(argv)
    for path in args.paths:
        log = read_plotdata(path, full_scale_ml_min=args.full_scale_ml_min)
        print(json.dumps(log.describe(), ensure_ascii=False))
        print(episodes(log, threshold=args.threshold).to_string(index=False))
        print(setpoint_changes(log).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
