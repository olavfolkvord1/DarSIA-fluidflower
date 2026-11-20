"""Numeric result diagnostics, published without a host or service."""

import json
from types import SimpleNamespace

import pytest

from darsia.jobs.step import StepContext
from darsia.utils import results


def _context(tmp_path):
    request = {
        "step": "phase-maps",
        "settings": {},
        "scope": {},
        "inputs": {},
        "environment": {},
    }
    return SimpleNamespace(
        _output=lambda *args: None,
        _location=lambda output: tmp_path / "frame",
        _configured_files=lambda: {},
        _files={},
        _request=lambda frame: request,
        _job={"id": "job"},
        task=SimpleNamespace(id="phase-maps.run"),
        revision="test-revision",
        device="cpu",
        token="123abc",
        locations=SimpleNamespace(results=tmp_path),
    )


def test_measurements_are_readable_without_arrays_and_cannot_shadow_identity(tmp_path):
    context = _context(tmp_path)
    measurements = {
        "token": "not-the-claim",
        "gas_allocation": {"target_m3": None, "placed_m3": 0.0, "limit_area_mm2": 3.0},
    }
    StepContext.publish(
        context,
        lambda folder: (folder / "arrays").write_bytes(b"not loaded"),
        measurements=measurements,
    )
    stored = json.loads((tmp_path / "frame" / "result.json").read_text())
    assert stored["token"] == context.token
    assert stored["measurements"] == measurements
    assert results.reusable(stored, context._request(None)) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_measurements_cannot_be_published(tmp_path, value):
    with pytest.raises(ValueError):
        StepContext.publish(
            _context(tmp_path), lambda folder: None, measurements={"area": value}
        )
    assert not results.is_complete(tmp_path / "frame")
