"""Unit tests for the injected mass of an injection protocol."""

import pandas as pd
import pytest

import darsia

START = pd.Timestamp(0, unit="s")


def _protocol(tmp_path, rows, rate_column, **kwargs):
    """A protocol file of rows (x, y, start [s], end [s], rate[, density])."""
    lines = [f"id,location_x,location_y,start,end,{rate_column}"]
    for number, (x, y, start, end, *values) in enumerate(rows, 1):
        times = [(START + pd.Timedelta(seconds=s)).isoformat() for s in (start, end)]
        lines.append(",".join(map(str, [number, x, y, *times, *values])))
    path = tmp_path / "injection.csv"
    path.write_text("\n".join(lines) + "\n")
    return darsia.InjectionProtocol(path, **kwargs)


def test_time_mode_counts_each_interval_from_the_first_start(tmp_path):
    protocol = _protocol(
        tmp_path,
        [(0.1, 0.1, 0, 600, 2e-6), (0.5, 0.3, 600, 1800, 5e-6)],
        "rate_kg/s",
    )
    assert protocol.injected_mass(time=0.1) == pytest.approx(360 * 2e-6)
    assert protocol.injected_mass(time=0.25) == pytest.approx(600 * 2e-6 + 300 * 5e-6)
    for hours in (0.1, 0.25, 0.6):
        date = START + pd.Timedelta(hours=hours)
        assert protocol.injected_mass(time=hours) == pytest.approx(
            protocol.injected_mass(date=date), rel=1e-12
        )
