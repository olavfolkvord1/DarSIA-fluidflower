"""Unit tests for the overflow reference on synthetic scale readings."""

import datetime
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.presets.workflows.config.utils import settings_from_table

SETTINGS = settings_from_table(
    darsia.OverflowSettings,
    tomllib.loads(
        """
        # The values the overflow reference ran with in production.
        [overflow]
        statuses = ["Verified", "Corrected"]
        recipe_tolerance = {alkalinity_M = 0.03, btb_M = 0.03}
        variant = "feed"

        [overflow.feed]
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
        """
    )["overflow"],
)
CIRCULATION = replace(
    SETTINGS,
    variant="circulation",
    feed=None,
    circulation=darsia.CirculationSettings(minimum_readings=30),
)
EVAPORATION = darsia.EvaporationSettings(
    minimum_readings=3, rates_g_h={"run": "measured", "other": 0.5}
)

START = datetime.datetime(2000, 1, 1, 8, 0)


def _displaced(t):
    """Displaced water [mL]: up to 200 mL at 2 h, back to none at 6 h."""
    return np.interp(t, [0, 2, 6], [0, 200, 0])


def _protocol(tmp_path, stop_h=11.4, circulation=False, evaporation_g_h=0.0):
    """Scale readings every 6 minutes: displaced water on a feed or on 5 kg."""
    t = np.round(np.arange(-0.5, 14.0, 0.1), 1)
    feed = 30 + 400 * np.minimum(t, stop_h) + 0.2 * np.minimum(t, stop_h) ** 2
    evaporated = evaporation_g_h * (t - t[0])
    rows = pd.DataFrame(
        {
            "image_time": [START + datetime.timedelta(hours=h) for h in t],
            "weight_g": (5000.0 if circulation else feed) + _displaced(t) - evaporated,
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

    # The baseline window scopes a circulation's reference, never the feed's.
    windowed = replace(SETTINGS, baseline_window_h=0.05)
    windowed = darsia.displaced_volume(times, weights, windowed)
    assert np.array_equal(windowed.volume_ml, volume.volume_ml)

    early = _protocol(tmp_path, stop_h=8.0).readings(START, SETTINGS.statuses)
    assert "the feed stops at 8 h" in darsia.displaced_volume(*early, SETTINGS).reasons


def test_a_circulation_reads_the_weight_above_the_baseline(tmp_path):
    """The absolute weight is the displaced water; the configuration names the variant."""
    times, weights = _protocol(tmp_path, circulation=True).readings(
        START, SETTINGS.statuses
    )
    volume = darsia.displaced_volume(times, weights, CIRCULATION)
    assert volume.reasons == ()
    assert np.allclose(volume.volume_ml, _displaced(times), atol=1e-9)
    assert darsia.measured_reference(volume, 13.0) == 0.0

    # The reference is the mean weight of the baseline, within its window if given.
    few = replace(CIRCULATION, circulation=darsia.CirculationSettings(3))
    t, w = [-0.4, -0.3, -0.2, -0.1, 1.0], [5001, 4999, 5002, 5000, 5100]
    assert darsia.displaced_volume(t, w, few).volume_ml[-1] == 99.5
    windowed = replace(few, baseline_window_h=0.25)
    assert darsia.displaced_volume(t, w, windowed).volume_ml[-1] == 99.0

    table = tomllib.loads(
        """
        statuses = ["Verified"]
        recipe_tolerance = {}
        variant = "circulation"
        circulation = {minimum_readings = 30}
        """
    )
    settings = settings_from_table(darsia.OverflowSettings, table)
    assert settings.variant_settings == CIRCULATION.circulation
    with pytest.raises(ValueError, match="does not match the circulation"):
        replace(settings, feed=SETTINGS.feed)
    with pytest.raises(ValueError, match="feed variant needs its"):
        settings_from_table(darsia.OverflowSettings, {**table, "variant": "feed"})


def test_a_steady_evaporation_lies_inside_the_fitted_feed(tmp_path):
    """Without the feed, the baseline proposes almost no rate; a rate cancels."""
    times, weights = _protocol(tmp_path, evaporation_g_h=3.0).readings(
        START, SETTINGS.statuses
    )
    volume = darsia.displaced_volume(times, weights, SETTINGS)
    fed = times <= 11.4
    assert np.allclose(volume.volume_ml[fed], _displaced(times[fed]), atol=1e-6)
    proposal = darsia.proposed_evaporation(volume, EVAPORATION)
    assert proposal.readings == 5 and abs(proposal.rate_g_h) < 1e-6
    corrected = darsia.displaced_volume(times, weights, SETTINGS, 3.0)
    assert np.allclose(corrected.volume_ml, volume.volume_ml, atol=1e-9)


def test_an_evaporation_rate_leaves_the_feed_decisions_to_the_readings(tmp_path):
    """The feed's stop, late window and checks read the weights as read."""
    times, weights = _protocol(tmp_path).readings(START, SETTINGS.statuses)
    feed = replace(SETTINGS.feed, maximum_pre_injection_rate_g_h=401.0)
    tight = replace(SETTINGS, feed=feed)
    volume = darsia.displaced_volume(times, weights, tight)
    corrected = darsia.displaced_volume(times, weights, tight, 3.0)
    assert volume.reasons == corrected.reasons == ()
    assert np.allclose(corrected.volume_ml, volume.volume_ml, atol=1e-9)


def test_a_circulation_proposes_its_evaporation_and_adds_it_back(tmp_path):
    """The weight's fall over the baseline is the rate the approved setting adds."""
    times, weights = _protocol(
        tmp_path, circulation=True, evaporation_g_h=0.8
    ).readings(START, SETTINGS.statuses)
    volume = darsia.displaced_volume(times, weights, CIRCULATION)
    assert volume.volume_ml[-1] == pytest.approx(-0.8 * (13.9 + 0.3))
    proposal = darsia.proposed_evaporation(volume, EVAPORATION)
    assert proposal.rate_g_h == pytest.approx(0.8) and proposal.sd_g_h < 1e-9
    windowed = replace(CIRCULATION, baseline_window_h=0.35)
    windowed = darsia.displaced_volume(times, weights, windowed)
    assert darsia.proposed_evaporation(windowed, EVAPORATION).readings == 3

    # Without the table, or for a run it does not list, nothing is corrected.
    assert darsia.approved_evaporation("run", proposal, CIRCULATION.evaporation) is None
    assert darsia.approved_evaporation("third", proposal, EVAPORATION) is None
    assert darsia.approved_evaporation("other", proposal, EVAPORATION) == 0.5
    with pytest.raises(ValueError, match="No evaporation rate"):
        darsia.approved_evaporation("run", None, EVAPORATION)
    rate = darsia.approved_evaporation("run", proposal, EVAPORATION)
    corrected = darsia.displaced_volume(times, weights, CIRCULATION, rate)
    assert np.allclose(corrected.volume_ml, _displaced(times), atol=1e-9)

    # Minus the slope with its standard error: residuals 1/6, -1/3, 1/6.
    t, v = np.array([-3.0, -2, -1, 1]), np.array([0.0, -1, -1, 5])
    line = darsia.proposed_evaporation(
        darsia.DisplacedVolume(t, v, (), t < 0), EVAPORATION
    )
    assert line.readings == 3 and line.rate_g_h == pytest.approx(0.5)
    assert line.sd_g_h == pytest.approx(np.sqrt(1 / 12))


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
