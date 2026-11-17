"""The stored JSON writer streams bytes while validating before opening."""

import json
import tracemalloc

import pytest

from darsia.utils.results import write_json


def test_streamed_json_keeps_bytes_and_platform_newlines(tmp_path):
    """Nested finite floats, Unicode and all plain types keep the former bytes."""
    values = {
        "frames": [
            {
                "relative": [0.0, -0.0, 1e-308, -1.7976931348623157e308],
                "weight": [1.0000000000000002, 0.3333333333333333],
            }
        ],
        "unicode": "sand \u00e6 \U0001f30d",
        "plain": [None, True, False, 7, "x\ny"],
        "held_out": {"one": [], "two": {}},
    }
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(
        json.dumps(values, indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_json(new, values)
    assert new.read_bytes() == old.read_bytes()


@pytest.mark.parametrize("value", [[float("nan")], [float("inf")], {"x": object()}])
def test_invalid_json_keeps_existing_or_absent_target(tmp_path, value):
    """Invalid data is rejected before the destination is created or truncated."""
    path = tmp_path / "value.json"
    for existing in (False, True):
        if existing:
            path.write_bytes(b"previous valid data\r\n")
        with pytest.raises((ValueError, TypeError)):
            write_json(path, value)
        if existing:
            assert path.read_bytes() == b"previous valid data\r\n"
        else:
            assert not path.exists()


def test_circular_json_keeps_existing_target(tmp_path):
    """Cycle detection also runs before opening the output file."""
    path = tmp_path / "value.json"
    path.write_bytes(b"previous")
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="Circular"):
        write_json(path, value)
    assert path.read_bytes() == b"previous"


def test_streamed_json_does_not_keep_whole_text(tmp_path):
    """The writer's transient allocation stays small for a large numeric row."""
    values = {"relative": [i / 100000 for i in range(30000)]}
    tracemalloc.start()
    try:
        write_json(tmp_path / "values.json", values)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 256 * 1024
