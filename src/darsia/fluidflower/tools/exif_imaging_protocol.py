"""Build DarSIA imaging protocols from the EXIF time stamps of photographs.

File times (ctime/mtime) of FluidFlower photographs are unreliable after
copying, and camera clocks are often off by a constant, for example when
daylight saving time was never set. This module therefore

1. reads ``DateTimeOriginal`` (plus ``SubSecTimeOriginal`` when present) from
   the photographs of one folder, either from every file (``mode="full"``) or
   from a subset (``mode="sampled"``) that is refined until the time between
   any two read photographs is explained by a constant interval per camera
   counter step;
2. reports pauses, skipped shots and missing counter numbers; and
3. writes native DarSIA imaging protocols (``path``, ``image_id``,
   ``datetime``, ``interval_s``) with a documented clock offset added to the
   camera time.

In sampled mode an interval between two read photographs ``i < j`` is
accepted when (a) its duration equals ``(n_j - n_i)`` times the folder's
dominant interval and (b) the photograph in the middle agrees with linear
interpolation, both within ``tolerance_s``. Pauses and skipped shots only add
time, so (a) catches them even when several occur in one interval; (b) guards
against a different but constant cadence. Intervals that fail are split until
the change is located between two neighbouring photographs. A few
interpolated times are read back as an audit.

Only the standard library, numpy, pandas and Pillow are needed; the module
does not import darsia and runs as a stand-alone script on Python 3.10::

    python exif_imaging_protocol.py scan FOLDER --offset-s 3600 --out protocol.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

FORMAT = "darsia-fluidflower-exif-timeline-v1"
IMAGE_SUFFIXES = (".jpg", ".jpeg")
_TRAILING_DIGITS = re.compile(r"(\d+)$")
_EPOCH = dt.datetime(1970, 1, 1)
_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 0x9003
_SUBSEC_ORIGINAL = 0x9291
_DATETIME = 0x0132

Reader = Callable[[Path], "dt.datetime | None"]


def image_number(name: str | Path) -> int | None:
    """Camera counter at the end of the file stem (``DSC42031.JPG`` -> 42031)."""
    match = _TRAILING_DIGITS.search(Path(name).stem)
    return int(match[1]) if match else None


def read_exif_time(path: str | Path) -> dt.datetime | None:
    """``DateTimeOriginal`` with sub-seconds, falling back to ``DateTime``."""
    from PIL import Image

    with Image.open(path) as image:
        exif = image.getexif()
        details = exif.get_ifd(_EXIF_IFD) if exif else {}
        value = details.get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
        if not value:
            return None
        stamp = dt.datetime.strptime(
            str(value).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S"
        )
        subsec = str(details.get(_SUBSEC_ORIGINAL) or "").strip().rstrip("\x00")
        if subsec.isdigit():
            stamp += dt.timedelta(seconds=float(f"0.{subsec}"))
        return stamp


def list_images(
    folder: str | Path, suffixes: Sequence[str] = IMAGE_SUFFIXES
) -> list[str]:
    """Image file names in ``folder`` sorted by camera counter, then name."""
    wanted = tuple(s.casefold() for s in suffixes)
    names = [
        entry.name
        for entry in os.scandir(folder)
        if entry.is_file() and entry.name.casefold().endswith(wanted)
    ]
    return sorted(
        names, key=lambda n: (image_number(n) is None, image_number(n) or 0, n)
    )


def _seconds(stamp: dt.datetime | None) -> float:
    return float("nan") if stamp is None else (stamp - _EPOCH).total_seconds()


def _stamp(seconds: float) -> dt.datetime:
    return _EPOCH + dt.timedelta(seconds=float(seconds))


@dataclasses.dataclass(frozen=True, eq=False)
class FolderTimeline:
    """Camera times of all photographs in one folder.

    ``frame`` has ``path`` (file name), ``image_id``, ``camera_time``,
    ``time_source`` (``exif`` or ``interpolated``; ``unknown`` when no time
    could be found) and ``interval_s`` (camera time since the previous
    photograph in the folder).
    """

    folder: str
    mode: str
    frame: pd.DataFrame
    n_reads: int
    dominant_interval_s: float
    tolerance_s: float
    audit: pd.DataFrame

    def gaps(self, factor: float = 1.5) -> pd.DataFrame:
        return time_gaps(self.frame, factor=factor)

    def describe(self) -> dict[str, Any]:
        frame = self.frame
        ids = frame["image_id"].dropna().astype(int)
        known = frame["camera_time"].dropna()
        intervals = frame["interval_s"].dropna().round(1)
        common = intervals.value_counts().head(3)
        span = int(ids.max() - ids.min() + 1) if len(ids) else 0
        return {
            "format": FORMAT,
            "folder": self.folder,
            "mode": self.mode,
            "n_images": int(len(frame)),
            "n_exif_reads": int(self.n_reads),
            "first_image": frame["path"].iloc[0] if len(frame) else None,
            "last_image": frame["path"].iloc[-1] if len(frame) else None,
            "first_camera_time": str(known.iloc[0]) if len(known) else None,
            "last_camera_time": str(known.iloc[-1]) if len(known) else None,
            "dominant_interval_s": self.dominant_interval_s,
            "common_intervals_s": {str(k): int(v) for k, v in common.items()},
            "missing_image_ids": max(span - int(ids.nunique()), 0),
            "n_unknown_times": int(frame["camera_time"].isna().sum()),
            "n_time_gaps": int(len(self.gaps())),
            "audit_n": int(len(self.audit)),
            "audit_max_abs_error_s": (
                float(self.audit["error_s"].abs().max()) if len(self.audit) else None
            ),
        }


def _read_many(
    folder: Path,
    names: Sequence[str],
    indices: Iterable[int],
    reader: Reader,
    workers: int,
    cache: dict[int, float],
) -> None:
    todo = sorted({int(i) for i in indices if int(i) not in cache})
    if not todo:
        return

    def read(index: int) -> float:
        try:
            return _seconds(reader(folder / names[index]))
        except Exception:  # unreadable or truncated file: treat as missing time
            return float("nan")

    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(read, todo))
    else:
        values = [read(i) for i in todo]
    cache.update(zip(todo, values))


def folder_timeline(
    folder: str | Path,
    *,
    mode: str = "sampled",
    sample_step: int = 16,
    tolerance_s: float = 2.0,
    audit_count: int = 3,
    workers: int = 8,
    reader: Reader = read_exif_time,
    names: Sequence[str] | None = None,
    seed: int = 0,
) -> FolderTimeline:
    """Camera times for every photograph in ``folder`` (see module docstring)."""
    if mode not in ("full", "sampled"):
        raise ValueError("mode must be 'full' or 'sampled'")
    folder = Path(folder)
    names = list(names) if names is not None else list_images(folder)
    n = len(names)
    numbers = [image_number(name) for name in names]
    if n == 0:
        raise FileNotFoundError(f"No photographs in {folder}")
    if any(number is None for number in numbers) and mode == "sampled":
        mode = "full"  # interpolation needs camera counters
    cache: dict[int, float] = {}
    times = np.full(n, np.nan)
    source = np.array(["unknown"] * n, dtype=object)

    if mode == "full":
        _read_many(folder, names, range(n), reader, workers, cache)
        dominant = float("nan")
    else:
        step = max(int(sample_step), 2)
        anchors = sorted(set(range(0, n, step)) | {n - 1})
        _read_many(folder, names, anchors, reader, workers, cache)
        num = np.asarray(numbers, dtype=float)
        rates = [
            (cache[j] - cache[i]) / (num[j] - num[i])
            for i, j in zip(anchors[:-1], anchors[1:])
            if np.isfinite(cache[i]) and np.isfinite(cache[j]) and num[j] > num[i]
        ]
        dominant = float(np.median(rates)) if rates else float("nan")
        pending = list(zip(anchors[:-1], anchors[1:]))
        while pending:
            splits = [(i, j, (i + j) // 2) for i, j in pending if j - i > 1]
            _read_many(folder, names, [m for _, _, m in splits], reader, workers, cache)
            pending = []
            for i, j, m in splits:
                ti, tj, tm = cache[i], cache[j], cache[m]
                ok = (
                    np.isfinite(ti)
                    and np.isfinite(tj)
                    and np.isfinite(tm)
                    and num[j] > num[i]
                )
                if ok:
                    predicted = ti + (tj - ti) * (num[m] - num[i]) / (num[j] - num[i])
                    span_error = (tj - ti) - dominant * (num[j] - num[i])
                    ok = (
                        abs(tm - predicted) <= tolerance_s
                        and abs(span_error) <= tolerance_s
                    )
                if ok:
                    inside = np.arange(i + 1, j)
                    inside = inside[[k not in cache for k in inside]]
                    times[inside] = ti + (tj - ti) * (num[inside] - num[i]) / (
                        num[j] - num[i]
                    )
                    source[inside] = "interpolated"
                else:
                    pending.extend([(i, m), (m, j)])

    for index, value in cache.items():
        if np.isfinite(value):
            times[index] = value
            source[index] = "exif"
    audit_rows = []
    interpolated = [k for k in range(n) if source[k] == "interpolated"]
    if interpolated and audit_count > 0:
        chosen = random.Random(seed).sample(
            interpolated, min(audit_count, len(interpolated))
        )
        probe: dict[int, float] = {}
        _read_many(folder, names, chosen, reader, workers, probe)
        for k in sorted(chosen):
            audit_rows.append(
                {
                    "path": names[k],
                    "interpolated": _stamp(times[k]),
                    "exif": _stamp(probe[k]) if np.isfinite(probe[k]) else None,
                    "error_s": (
                        float(times[k] - probe[k])
                        if np.isfinite(probe[k])
                        else float("nan")
                    ),
                }
            )
    camera_time = pd.Series(
        [_stamp(value) if np.isfinite(value) else pd.NaT for value in times],
        dtype="datetime64[ns]",
    )
    frame = pd.DataFrame(
        {
            "path": names,
            "image_id": pd.array(numbers, dtype="Int64"),
            "camera_time": camera_time.dt.round("ms"),
            "time_source": source.astype(str),
        }
    )
    frame["interval_s"] = frame["camera_time"].diff().dt.total_seconds()
    return FolderTimeline(
        folder=str(folder),
        mode=mode,
        frame=frame,
        n_reads=len(cache),
        dominant_interval_s=dominant,
        tolerance_s=float(tolerance_s),
        audit=pd.DataFrame(
            audit_rows, columns=["path", "interpolated", "exif", "error_s"]
        ),
    )


def time_gaps(
    frame: pd.DataFrame, *, factor: float = 1.5, window: int = 9
) -> pd.DataFrame:
    """Pauses: intervals longer than ``factor`` times the local median interval.

    ``missing_ids`` counts camera numbers absent between the two photographs;
    a pause with no missing numbers means the camera did not shoot.
    """
    intervals = frame["interval_s"]
    local = intervals.rolling(window, center=True, min_periods=3).median()
    flagged = intervals > factor * local
    rows = []
    for index in np.flatnonzero(flagged.to_numpy()):
        before, after = frame.iloc[index - 1], frame.iloc[index]
        missing = None
        if pd.notna(before["image_id"]) and pd.notna(after["image_id"]):
            missing = int(after["image_id"]) - int(before["image_id"]) - 1
        rows.append(
            {
                "before": before["path"],
                "after": after["path"],
                "start": before["camera_time"],
                "end": after["camera_time"],
                "gap_s": float(intervals.iloc[index]),
                "local_interval_s": float(local.iloc[index]),
                "missing_ids": missing,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "before",
            "after",
            "start",
            "end",
            "gap_s",
            "local_interval_s",
            "missing_ids",
        ],
    )


def imaging_protocol(frame: pd.DataFrame, *, offset_s: float = 0.0) -> pd.DataFrame:
    """Native DarSIA imaging protocol: camera time plus a clock offset.

    The offset converts camera time to the reference clock of the series
    (for example the balance PC). Times are rounded to whole seconds.
    Photographs without a time are left out.
    """
    known = frame.dropna(subset=["camera_time"])
    corrected = (known["camera_time"] + pd.to_timedelta(offset_s, unit="s")).dt.round(
        "s"
    )
    return pd.DataFrame(
        {
            "path": known["path"].to_numpy(),
            "image_id": known["image_id"].to_numpy(),
            "datetime": corrected.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "interval_s": known["interval_s"].round(1).to_numpy(),
        }
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="Camera timeline of one folder")
    scan.add_argument("folder", type=Path)
    scan.add_argument("--mode", choices=("sampled", "full"), default="sampled")
    scan.add_argument("--sample-step", type=int, default=16)
    scan.add_argument("--tolerance-s", type=float, default=2.0)
    scan.add_argument("--workers", type=int, default=8)
    scan.add_argument(
        "--offset-s", type=float, default=0.0, help="Added to camera time"
    )
    scan.add_argument("--out", type=Path, help="Write the imaging protocol CSV here")
    scan.add_argument(
        "--timeline", type=Path, help="Write the full camera timeline CSV here"
    )
    args = parser.parse_args(argv)
    timeline = folder_timeline(
        args.folder,
        mode=args.mode,
        sample_step=args.sample_step,
        tolerance_s=args.tolerance_s,
        workers=args.workers,
    )
    print(json.dumps(timeline.describe(), ensure_ascii=False, default=str))
    gaps = timeline.gaps()
    if len(gaps):
        print(gaps.to_string(index=False))
    if args.timeline:
        timeline.frame.to_csv(args.timeline, index=False)
    if args.out:
        imaging_protocol(timeline.frame, offset_s=args.offset_s).to_csv(
            args.out, index=False
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
