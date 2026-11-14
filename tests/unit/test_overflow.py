"""Unit tests for the overflow reference on synthetic scale readings."""

import datetime
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import darsia

SETTINGS = darsia.OverflowSettings(
    **tomllib.loads(
        """
        # The values the overflow reference ran with in production.
        [overflow]
        statuses = ["Verified", "Corrected"]
        minimum_readings = 30
        supply_check_h = 4.0
        supply_check_span_h = 0.5
        supply_stop_rise_g = 10.0
        late_window_h = [4.0, 0.3]
        minimum_supply_h = 10.0
        maximum_late_sd_g = 1.5
        maximum_pre_injection_offset_ml = 50.0
        pre_injection_span_h = 0.05
        maximum_pre_injection_rate_g_h = 600.0
        recipe_tolerance = {alkalinity_M = 0.03, btb_M = 0.03}
        """
    )["overflow"]
)

START = datetime.datetime(2000, 1, 1, 8, 0)


def _displaced(t):
    """Displaced water [mL]: up to 200 mL at 2 h, back to none at 6 h."""
    return np.interp(t, [0, 2, 6], [0, 200, 0])


def _protocol(tmp_path, stop_h=11.4):
    """Scale readings every 6 minutes: a feed until stop_h and the displaced water."""
    t = np.round(np.arange(-0.5, 14.0, 0.1), 1)
    feed = 30 + 400 * np.minimum(t, stop_h) + 0.2 * np.minimum(t, stop_h) ** 2
    rows = pd.DataFrame(
        {
            "image_time": [START + datetime.timedelta(hours=h) for h in t],
            "weight_g": feed + _displaced(t),
            "status": "Verified",
        }
    )
    rows.loc[::7, "status"] = "Corrected"
    pending = rows.iloc[3::10].assign(weight_g=5000.0, status="Pending")
    path = tmp_path / "overflow.csv"
    pd.concat([rows, pending]).to_csv(path, index=False)
    return darsia.OverflowProtocol(path)


def test_measured_reference_is_the_weight_above_the_feed(tmp_path):
    """The feed from the last reading before injection leaves the displaced water."""
    times, weights = _protocol(tmp_path).readings(START, SETTINGS.statuses)
    volume = darsia.displaced_volume(times, weights, SETTINGS)
    assert volume.reasons == ()
    fed = times <= 11.4
    assert np.allclose(volume.volume_ml[fed], _displaced(times[fed]), atol=1e-6)

    assert darsia.measured_reference(volume, 1.05) == pytest.approx(105e-6)
    assert darsia.measured_reference(volume, 8.0) == pytest.approx(0, abs=1e-12)
    assert darsia.measured_reference(volume, 13.0) == 0.0
    assert darsia.measured_reference(volume, -1.0) is None
    assert darsia.measured_reference(volume, 14.5) is None
    single = replace(volume, times_h=volume.times_h.astype(np.float32))
    single = replace(single, volume_ml=volume.volume_ml.astype(np.float32))
    assert darsia.measured_reference(single, 1.05) == pytest.approx(105e-6, rel=1e-6)

    early = _protocol(tmp_path, stop_h=8.0).readings(START, SETTINGS.statuses)
    assert "the feed stops at 8 h" in darsia.displaced_volume(*early, SETTINGS).reasons


def test_a_sheet_gives_its_own_injection_start(tmp_path):
    """A sheet's readings keep the scale's clock, whose injection start the sheet
    records beside them; a sheet without one has none, and one naming two is
    refused."""
    assert _protocol(tmp_path).injection_start is None
    path = tmp_path / "overflow.csv"
    rows = pd.read_csv(path)
    start = START - datetime.timedelta(hours=1, minutes=5)
    rows.assign(injection_start=[str(start)] + [None] * (len(rows) - 1)).to_csv(
        path, index=False
    )
    assert darsia.OverflowProtocol(path).injection_start == pd.Timestamp(start)
    two = [str(start), str(START)] + [None] * (len(rows) - 2)
    rows.assign(injection_start=two).to_csv(path, index=False)
    with pytest.raises(ValueError, match="2 injection starts"):
        darsia.OverflowProtocol(path)


def _run(volume, offset_bar=0.0, sand_map="map", alkalinity=0.01, cover=(-1.0, 18.0)):
    """A run with a steady pressure record and a recipe."""
    hours = np.arange(cover[0], cover[1] + 0.25, 0.25)
    return darsia.OverflowRun(
        sand_map,
        {"alkalinity_M": alkalinity, "btb_M": 1e-4, "salt": "none"},
        hours,
        1.0 + offset_bar + 0.002 * np.sin(hours),
        (0.0, 3.0),
        volume,
    )


def test_estimated_reference_scales_the_most_similar_run(tmp_path):
    """The compatible run of nearest pressure lends its free-gas share of the mass."""
    volume = darsia.displaced_volume(
        *_protocol(tmp_path).readings(START, SETTINGS.statuses), SETTINGS
    )
    runs = {
        "recipient": _run(None),
        "far": _run(volume, offset_bar=0.01),
        "near": _run(volume, offset_bar=0.002, alkalinity=0.0102),
        "recipe": _run(volume, offset_bar=0.0, alkalinity=0.011),
        "map": _run(volume, offset_bar=0.0, sand_map="other"),
        "short": _run(volume, offset_bar=0.0, cover=(-1.0, 10.0)),
    }
    assert darsia.ranked_donors("recipient", runs, SETTINGS) == ["near", "far"]

    # The donor's 175 mL at 2.5 h, at 1.8 kg/m3, is 0.315 g of its 1 g: 31.5 % of
    # the frame's 2 g at 1.9 kg/m3.
    share = darsia.estimated_reference(volume, 2.5, 1.0, 1.8, 2.0, 1.9)
    assert share == pytest.approx(0.315 * 2.0 / 1.9 / 1000)
    assert darsia.estimated_reference(volume, 2.5, 0.1, 1.8, 2.0, 1.9) is None
    assert darsia.estimated_reference(volume, 15.0, 1.0, 1.8, 2.0, 1.9) is None
