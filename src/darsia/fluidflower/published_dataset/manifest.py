"""The dataset manifest, ``dataset_manifest.json`` (format ``darsia-published-dataset-v1``).

The manifest is the one description of a record that every DarSIA route
reads: identity, code, environment, plan, runs and frames, expected results,
machine facts, quick-check validity, the reproducibility rule with the known
variation between machines, units, and the record's size and files.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re

from .layout import FILE_LIMIT, FORMAT, MANIFEST, RECORD_TYPES, TIER_BUDGETS, record_relative

# The cross-machine equivalence limits (version 3 of the project's limits).
# On the reference machine a reproduction must be bit-identical; elsewhere
# these limits apply. Masses in g, areas in cm2; a map limit is relative to
# the frame's largest reference value of that array.
DEFAULT_RULE = {
    "name": "cross-machine equivalence limits, version 3",
    "source_sha256": "36aa8e5f5dd4a3e5e3537c5d330aa1f6d6d4b6454d43cd69def9fac8b904404d",
    "same_machine": "bit-identical",
    "labels": {"co2_relative": 0.001, "gas_relative": 0.005, "absolute_cm2": 0.05},
    "masses": {"relative": 0.001, "absolute": 0.0001, "unit": "g"},
    "areas": {"relative": 0.001, "absolute": 0.0001, "unit": "cm2"},
    "maps": {"relative_to_frame_maximum": 0.001, "allowed_fraction_above": 1e-5,
             "label_changed_cells_exempt": True},
}
LABEL_LEGEND = {"0": "water", "1": "dissolved CO2", "2": "free gas", "255": "excluded"}
_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _require(condition, message):
    """Raise ValueError with ``message`` unless ``condition`` holds."""
    if not condition:
        raise ValueError("Dataset manifest: " + message)


def validate_rule(rule: dict) -> dict:
    """Check a reproducibility rule's limits; return it unchanged."""
    _require(isinstance(rule, dict) and rule.get("same_machine") == "bit-identical",
             "the rule must require bit-identical results on the reference machine")
    for group, keys in (("labels", ("co2_relative", "gas_relative", "absolute_cm2")),
                        ("masses", ("relative", "absolute")), ("areas", ("relative", "absolute")),
                        ("maps", ("relative_to_frame_maximum", "allowed_fraction_above"))):
        values = rule.get(group)
        _require(isinstance(values, dict), f"rule lacks {group}")
        for key in keys:
            value = values.get(key)
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                     f"rule {group}.{key} must be a nonnegative number")
    return rule


def validate_manifest(document: dict) -> dict:
    """Check a manifest's structure and cross-references; return it unchanged."""
    _require(isinstance(document, dict) and document.get("format") == FORMAT,
             f"format must be {FORMAT}")
    identity = document.get("identity") or {}
    _require(identity.get("record") in RECORD_TYPES, "identity.record must be R or S")
    _require(identity.get("tier") in TIER_BUDGETS, "identity.tier must be S50, S100 or S200")
    _require(isinstance(identity.get("series_id"), str) and identity["series_id"], "series_id missing")
    code = document.get("code") or {}
    for key in ("expected_results_commit", "published_commit"):
        _require(isinstance(code.get(key), str) and _COMMIT.fullmatch(code[key]), f"code.{key} missing")
    _require(code["expected_results_commit"] == code["published_commit"] or code.get("proof"),
             "a published commit other than the one that made the results needs a proof")
    runs = document.get("runs")
    _require(isinstance(runs, list) and runs, "runs missing")
    names = [row.get("run") for row in runs]
    _require(len(set(names)) == len(names), "a run is listed twice")
    expected = (document.get("expected") or {}).get("runs") or {}
    for row in runs:
        frames = row.get("frames")
        _require(isinstance(frames, list) and frames, f"run {row.get('run')} has no frames")
        for frame in frames:
            raw = frame.get("raw") or {}
            _require(_SHA.fullmatch(str(raw.get("sha256", ""))) is not None,
                     f"{frame.get('case_id')} has no raw SHA-256")
            record_relative(raw.get("path", ""))
        if identity["record"] == "R":
            _require(row["run"] in expected, f"run {row['run']} has no expected results")
            task = (((document.get("plan") or {}).get("tasks") or {}).get(row["run"])) or {}
            _require(_SHA.fullmatch(str(task.get("sha256", ""))) is not None,
                     f"run {row['run']} has no prepared task")
            record_relative(task.get("path", ""))
            cases = {frame["case_id"] for frame in frames}
            _require(set(expected[row["run"]].get("frames", {})) == cases,
                     f"run {row['run']}: expected results and frames differ")
    validate_rule((document.get("reproducibility") or {}).get("other_machines") or {})
    _require(type(document.get("quick_check_valid")) is bool, "quick_check_valid must be true or false")
    for item in document.get("files") or ():
        record_relative(item["name"])
        _require(_SHA.fullmatch(item.get("sha256", "")) is not None, f"{item['name']} has no SHA-256")
    size = document.get("size") or {}
    _require(type(size.get("files")) is int and size["files"] < FILE_LIMIT,
             f"a record must have fewer than {FILE_LIMIT} files")
    return document


def manifest_text(document: dict) -> str:
    """The canonical text of a manifest: UTF-8 JSON, one-space indent, final newline."""
    return json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def write_manifest(root: str | Path, document: dict) -> Path:
    """Validate and write a new manifest into ``root``."""
    validate_manifest(document)
    path = Path(root) / MANIFEST
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(manifest_text(document))
    return path


def load_manifest(root: str | Path) -> dict:
    """Read and validate the manifest of a record or an unpacked dataset folder."""
    path = Path(root) / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"No {MANIFEST} in {root}")
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def is_dataset(root: str | Path) -> bool:
    """True when a folder holds a published dataset manifest."""
    return (Path(root) / MANIFEST).is_file()


def rule_of(document: dict) -> dict:
    """The manifest's cross-machine rule, as a copy."""
    return deepcopy(document["reproducibility"]["other_machines"])
