"""Unit tests for the injected mass of an injection protocol."""

from contextlib import nullcontext

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


STEPPED = [
    (0.45, 0.02, 0, 60, 0.1),
    (0.45, 0.02, 60, 120, 0.2),
    (0.45, 0.02, 120, 180, 0.5),
    (0.45, 0.02, 180, 240, 1.0),
    (0.45, 0.02, 240, 300, 1.5),
    (0.45, 0.02, 300, 3000, 2.0),
    (0.45, 0.02, 3000, 3060, 1.5),
    (0.45, 0.02, 3060, 3120, 1.0),
    (0.45, 0.02, 3120, 3180, 0.5),
    (0.45, 0.02, 3180, 3240, 0.2),
    (0.45, 0.02, 3240, 3300, 0.1),
    (0.45, 0.02, 3300, 3330, 0.1),
    (0.45, 0.02, 3330, 3365, 0.24),
    (0.74, 0.16, 3365, 3370, 0.1),
    (0.74, 0.16, 3370, 3520, 0.2),
    (0.74, 0.16, 3520, 3580, 0.1),
    (0.74, 0.16, 3580, 3640, 0.2),
    (0.74, 0.16, 3640, 3700, 0.5),
    (0.74, 0.16, 3700, 3760, 1.0),
    (0.74, 0.16, 3760, 3820, 1.5),
    (0.74, 0.16, 3820, 8320, 2.0),
    (0.74, 0.16, 8320, 8341, 1.5),
    (0.74, 0.16, 8341, 8422, 1.0),
    (0.74, 0.16, 8422, 8482, 0.5),
    (0.74, 0.16, 8482, 8542, 0.2),
    (0.74, 0.16, 8542, 8602, 0.1),
    (0.74, 0.16, 8602, 8607, 0.0),
]
"""A run's stepped injection at two ports as our production read it: x, y [m],
start and end [s] from the first start, and the rate [sccm]."""


def test_volume_rates_take_the_flow_controllers_reference_density(tmp_path):
    expected = {  # our production's injected masses [kg], bit for bit
        0.25: 4.2918600000000006e-05,
        0.305244361: 5.5129813555440009e-05,
        7.436111111111111: 0.00046652948000000017,
    }
    lower = darsia.make_coordinate([[0.0, 0.0], [0.9, 0.1504]])
    cases = ((1.842, 1.842), (1.842, None), (2.0, 1.842))
    for number, (listed, reference) in enumerate(cases):
        folder = tmp_path / str(number)
        folder.mkdir()
        rows = [(*row, listed) for row in STEPPED]
        differs = reference is not None and listed != reference
        with pytest.warns(UserWarning) if differs else nullcontext():
            protocol = _protocol(
                folder, rows, "rate_sccm,density kg/m3", reference_density=reference
            )
        for hours, mass in expected.items():
            assert protocol.injected_mass(time=hours) == mass
        mass = protocol.injected_mass(time=7.436111111111111, roi=lower)
        assert mass == 0.00017828718000000002
