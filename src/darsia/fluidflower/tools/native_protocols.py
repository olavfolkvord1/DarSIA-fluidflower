"""Adapt archived camera-ID protocols to native DarSIA filename protocols.

New series TOMLs already reference native CSVs. Archived protocols are converted
only in scratch storage, using actual supplied photograph names. Per-folder
blacklists retain their scope; camera counters are never merged across folders.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pandas as pd

from darsia.fluidflower.series_paths import local_cache_root


def _read(value):
    path, sheet = (Path(value[0]), value[1]) if isinstance(value, (tuple, list)) else (Path(value), None)
    frame = pd.read_csv(path) if path.suffix.casefold() == '.csv' else pd.read_excel(path, sheet_name=sheet)
    frame.columns = [str(c).strip().casefold() for c in frame.columns]
    return path, sheet, frame


def _image_id(path):
    match = re.search(r'(\d+)$', Path(path).stem)
    if not match:
        raise ValueError(f'Photograph has no numeric camera counter: {path}')
    return int(match[1])


def _write(frame, original, source, kind, *, cache_root=None, sheet=None):
    content = frame.to_csv(index=False, date_format='%Y-%m-%dT%H:%M:%S.%f').encode()
    digest = hashlib.sha256(content).hexdigest()
    root = Path(cache_root) if cache_root is not None else local_cache_root() / 'native_protocols'
    target = root / f'{kind}_{digest}.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(target)
    return target, dict(configured=True, source_path=str(source), source_sheet=sheet,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_rows=len(original), valid_rows=len(frame), source_modified=False,
        sanitized=True, sanitizer='camera-protocol-to-native-filenames-v1',
        resolved_path=str(target), resolved_sha256=digest)


def imaging_protocol(value, *, image_paths=(), cache_root=None):
    source, sheet, frame = _read(value)
    if 'datetime' not in frame:
        raise ValueError(f'Imaging protocol lacks datetime: {source}')
    dates = pd.to_datetime(frame['datetime'], errors='coerce')
    if 'path' in frame:
        names = frame['path'].astype('string').str.strip()
        valid = dates.notna() & names.notna() & (names != '')
        result = pd.DataFrame({'path': names[valid], 'datetime': dates[valid]})
        if valid.all():
            return value, dict(source_path=str(source), source_sheet=sheet,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_rows=len(frame), valid_rows=len(frame), dropped_incomplete_rows=0,
                source_modified=False, sanitized=False, resolved_path=str(source))
    else:
        if 'image_id' not in frame or not image_paths:
            raise ValueError(f'Legacy imaging protocol requires actual photograph paths: {source}')
        ids = pd.to_numeric(frame['image_id'], errors='coerce')
        valid = dates.notna() & ids.notna() & (ids % 1 == 0)
        anchors = frame.loc[valid].copy()
        anchors['image_id'] = ids[valid].astype(int)
        anchors['datetime'] = dates[valid]
        anchors = anchors.sort_values('image_id')
        if anchors['image_id'].duplicated().any():
            raise ValueError(f'Ambiguous camera-counter anchors: {source}')
        rows = []
        for path in image_paths:
            number = _image_id(path)
            exact = anchors[anchors.image_id == number]
            if not exact.empty:
                date = exact.iloc[0]['datetime']
            else:
                previous = anchors[anchors.image_id < number]
                if previous.empty or 'interval_s' not in anchors:
                    raise ValueError(f'No recorded acquisition time for {path}')
                anchor = previous.iloc[-1]
                interval = float(anchor['interval_s'])
                if not pd.notna(interval) or interval <= 0:
                    raise ValueError(f'Invalid acquisition interval for {path}')
                date = anchor['datetime'] + pd.to_timedelta((number-int(anchor['image_id']))*interval, unit='s')
            rows.append({'path': str(Path(path).resolve()), 'datetime': date})
        result = pd.DataFrame(rows)
    return _write(result, frame, source, 'imaging', cache_root=cache_root, sheet=sheet)


def blacklist_protocol(value, *, image_paths=(), cache_root=None):
    if value is None:
        return None, {'configured': False}
    source, sheet, frame = _read(value)
    if len(frame.columns) != 1:
        raise ValueError(f'Blacklist must have exactly one column: {source}')
    if 'image_id' not in frame:
        names = frame.iloc[:, 0].astype('string').str.strip()
        if names.isna().any() or (names == '').any():
            raise ValueError(f'Blacklist contains empty photograph names: {source}')
        return value, dict(configured=True, source_path=str(source), source_sheet=sheet,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_rows=len(frame), source_modified=False, sanitized=False, resolved_path=str(source))
    ids = pd.to_numeric(frame['image_id'], errors='raise')
    if (ids % 1 != 0).any():
        raise ValueError(f'Blacklist contains non-integral camera counters: {source}')
    by_id = {}
    for path in image_paths:
        by_id.setdefault(_image_id(path), []).append(Path(path))
    names = []
    for number in ids.astype(int):
        matches = by_id.get(number, [])
        if len(matches) != 1:
            raise ValueError(f'Blacklist camera counter {number} is missing or ambiguous: {source}')
        names.append(str(matches[0].resolve()))
    return _write(pd.DataFrame({'path': names}), frame, source, 'blacklist',
                  cache_root=cache_root, sheet=sheet)
