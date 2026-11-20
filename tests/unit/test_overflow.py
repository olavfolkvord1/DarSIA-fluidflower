"""Unit tests for the overflow reference on synthetic scale readings."""

import datetime
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.experiment.overflow import _volume, _volume_json
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


# Our production's values (its own_target, reviewed_volume and the atmosphere's
# interpolation, under each working precision) for the readings and record below.
OWN_M3 = {"float32": [2.699999868127634e-06, 8.550000529794488e-06]}
OWN_M3["float64"] = [2.7e-06, 8.550000000000001e-06]
DONOR_ML = {"float32": 9.300000190734863, "float64": 9.3}
ATMOSPHERE = {
    "float32": [
        (1.0186875000308182, 20.443181820583245),
        (1.0175260869038802, 20.40434782284194),
        (1.0086391306585085, 19.856521752921754),
    ],
    "float64": [
        (1.0186875, 20.44318181818182),
        (1.0175260869565217, 20.40434782608696),
        (1.0086391304347826, 19.856521739130436),
    ],
}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_frame_volumes_take_the_own_readings_then_the_donor_then_none(dtype):
    """A run's frames take its own readings while they reach, then its donor's
    estimate, and none once the donor's readings end too, without the donor's
    state there, where its pressure record may have ended; interpolated in the
    readings' dtype, they equal our production's in single and double
    precision."""
    first = np.array([True, False, False, False, False])
    own = darsia.DisplacedVolume(
        np.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1.3, 4.1, 7.9, 9.2]),
        (),
        first,
    ).astype(dtype)
    donor = darsia.DisplacedVolume(
        np.array([-1.0, 0.5, 1.5, 2.5, 4.0]),
        np.array([0.0, 2.0, 5.5, 8.1, 9.9]),
        (),
        first,
    ).astype(dtype)

    def state(hours):
        if hours > 4.0:  # the donor's pressure record ends with its readings
            raise ValueError(f"No atmosphere at {hours} h")
        return 25.0, 1.9

    found = darsia.frame_volumes(
        own,
        [0.5, 2.5, 3.5, 4.5],
        ("d", donor),
        [10.0, 20.0, 30.0, 40.0],
        [1.8, 1.8, 1.8, 1.8],
        state,
    )
    assert [v.source for v in found] == [darsia.MEASURED, darsia.MEASURED, "d", None]
    assert [v.volume_m3 for v in found[:2]] == OWN_M3[dtype]
    observed = float(np.dtype(dtype).type(DONOR_ML[dtype]) * 1e-6)
    share = observed * 1.9 * 1000 / 25.0  # the donor's gas share of its CO2
    assert found[2].volume_m3 == share * 30.0 / (1.8 * 1000)
    assert found[3].volume_m3 is None


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_atmosphere_is_interpolated_over_its_record_hours(dtype):
    """The pressure and temperature at a time are linear in the record's hours,
    those in the dtype, as our production interpolated them; a time outside the
    record is refused."""
    record = ([-3.0, 1.4, 3.7, 24.0], [1.013, 1.0207, 1.0061, 1.0153])
    record += ([20.0, 20.6, 19.7, 20.2],)
    found = [
        darsia.interpolated_atmosphere(*record, hours, dtype)
        for hours in (0.25, 1.9, 3.3)
    ]
    assert found == ATMOSPHERE[dtype]
    with pytest.raises(ValueError, match="outside"):
        darsia.interpolated_atmosphere(*record, 24.5, dtype)


def test_the_atmosphere_is_our_productions_at_its_stored_frames():
    """At three frames our production's mass analysis stored, the pressure is the
    one their stored donor difference implies: three rows of the run's stored
    record, linear in their hours, those in float32, the analysis' precision."""
    times = (0.13425194444444444, 0.3009186111111111, 0.4675852777777778)
    pressure = (1.0060873842588398, 1.005886688590725, 1.0056859929226103)
    frames = (  # the frame's hours, its donor's pressure and their stored difference
        (0.24720333333333333, 1.0060943022429933, 0.0001429311154843571),
        (0.30536305555555554, 1.0060873842588398, 0.00020604755557473986),
        (0.3970297222222222, 1.0060873842588398, 0.0003164301796172442),
    )
    for hours, donor, difference in frames:
        found, _ = darsia.interpolated_atmosphere(
            times, pressure, (23.0, 23.0, 23.0), hours, "float32"
        )
        assert abs(donor - found) == difference


# The noise rules: single outliers after the peak, the tail floor, the feed stop.

RULES = replace(
    SETTINGS, outlier_factor=4.0, tail_noise_factor=2.0, end_at_supply_stop=True
)


def _readings(tmp_path, spikes=(), noise_g=0.3, stop_h=11.4):
    """Readings every 6 minutes on a feed with scale noise and added spikes
    (time [h], grams)."""
    t = np.round(np.arange(-0.5, 14.0, 0.1), 1)
    feed = 30 + 400 * np.minimum(t, stop_h) + 0.2 * np.minimum(t, stop_h) ** 2
    weight = feed + _displaced(t) + np.random.default_rng(7).normal(0, noise_g, t.size)
    for h, grams in spikes:
        weight[np.isclose(t, h)] += grams
    rows = pd.DataFrame(
        {
            "image_time": [START + datetime.timedelta(hours=h) for h in t],
            "weight_g": weight,
            "status": "Verified",
        }
    )
    path = tmp_path / "overflow.csv"
    rows.to_csv(path, index=False)
    return darsia.OverflowProtocol(path).readings(START, SETTINGS.statuses)


def test_without_the_rules_the_volume_and_its_record_are_unchanged(tmp_path):
    """Default settings: no rule fields, and the stored record has no new keys."""
    times, weights = _readings(tmp_path)
    volume = darsia.displaced_volume(times, weights, SETTINGS)
    assert volume.excluded == () and volume.end_h is None
    assert volume.peak_h is None and volume.tail_floor_ml is None
    assert set(_volume_json(volume)) == {
        "times_h",
        "volume_ml",
        "reasons",
        "baseline",
        "dtype",
    }


def test_a_single_outlier_after_the_peak_is_left_out(tmp_path):
    """A spike in the decay is replaced by its neighbours' line; one during the
    injection's rise is real and kept."""
    times, weights = _readings(tmp_path, spikes=((4.0, 8.0), (1.0, 8.0)))
    plain = darsia.displaced_volume(times, weights, SETTINGS)
    # The synthetic decay has a corner at 6 h (2.5 mL off its neighbours' line),
    # which a smooth real decay has not; a wider limit keeps it.
    cleaned = darsia.displaced_volume(
        times, weights, replace(RULES, outlier_factor=10.0)
    )
    i = int(np.flatnonzero(np.isclose(times, 4.0))[0])
    rise = int(np.flatnonzero(np.isclose(times, 1.0))[0])
    assert cleaned.excluded == (i,)
    between = (plain.volume_ml[i - 1] + plain.volume_ml[i + 1]) / 2
    assert cleaned.volume_ml[i] == pytest.approx(between)
    assert cleaned.volume_ml[rise] == plain.volume_ml[rise]
    assert cleaned.peak_h == pytest.approx(2.0, abs=0.11)
    others = np.ones(times.size, bool)
    others[i] = False
    assert np.array_equal(cleaned.volume_ml[others], plain.volume_ml[others])


def test_the_tail_within_the_noise_gives_no_reference(tmp_path):
    """After the peak a volume near zero leaves the model's gas; early small
    volumes before the peak still count."""
    times, weights = _readings(tmp_path)
    plain = darsia.displaced_volume(times, weights, SETTINGS)
    ruled = darsia.displaced_volume(times, weights, RULES)
    assert 0 < ruled.tail_floor_ml < 2.0
    assert darsia.measured_reference(plain, 8.0) is not None
    assert darsia.measured_reference(ruled, 8.0) is None
    assert darsia.measured_reference(ruled, 1.05) == pytest.approx(
        darsia.measured_reference(plain, 1.05)
    )
    assert darsia.measured_reference(ruled, 0.05) is not None


def test_a_fed_overflow_ends_at_its_stop(tmp_path):
    """After the feed stops the weight shows no displaced water: no reference."""
    times, weights = _readings(tmp_path, noise_g=0.0)
    plain = darsia.displaced_volume(times, weights, SETTINGS)
    ended = darsia.displaced_volume(
        times, weights, replace(SETTINGS, end_at_supply_stop=True)
    )
    assert ended.end_h == pytest.approx(11.4)
    assert darsia.measured_reference(plain, 13.0) == 0.0
    assert darsia.measured_reference(ended, 13.0) is None
    assert darsia.measured_reference(ended, 3.0) == pytest.approx(
        darsia.measured_reference(plain, 3.0)
    )


def test_own_scale_cutoffs_keep_model_even_with_a_donor(tmp_path):
    times, weights = _readings(tmp_path)
    own = darsia.displaced_volume(times, weights, RULES)
    donor = darsia.DisplacedVolume(
        np.array([-1.0, 15.0]),
        np.array([6.0, 6.0]),
        (),
        np.array([True, False]),
    )

    def donor_state(hour):
        pytest.fail(f"An own-scale decision must not consult the donor at {hour} h.")

    found = darsia.frame_volumes(
        own, [1.0, 8.0, 13.0], ("other", donor), [0.5] * 3, [1.8] * 3, donor_state
    )
    assert found[0].source == darsia.MEASURED and found[0].volume_m3 > 0
    assert all(row.source is None and row.volume_m3 is None for row in found[1:])


def test_a_donor_follows_its_own_rules(tmp_path):
    """An estimate from a donor is withheld where the donor's rules give none."""
    times, weights = _readings(tmp_path)
    donor = darsia.displaced_volume(times, weights, RULES)
    found = darsia.frame_volumes(
        None,
        [1.0, 8.0, 13.0],
        ("other", donor),
        [0.5, 0.5, 0.5],
        [1.8, 1.8, 1.8],
        lambda h: (0.5, 1.8),
    )
    assert found[0].source == "other" and found[0].volume_m3 > 0
    assert found[1].volume_m3 is None and found[2].volume_m3 is None


def test_the_rules_survive_the_stored_record(tmp_path):
    """A donor read back keeps its rules."""
    times, weights = _readings(tmp_path, spikes=((4.0, 8.0),))
    volume = darsia.displaced_volume(times, weights, RULES)
    back = _volume(_volume_json(volume))
    assert back.excluded == volume.excluded
    assert back.end_h == volume.end_h and back.peak_h == volume.peak_h
    assert back.tail_floor_ml == volume.tail_floor_ml
    assert np.array_equal(back.volume_ml, volume.volume_ml)


def test_the_rules_need_a_fed_overflow_and_positive_factors():
    """A circulation has no stop or late window; factors must be positive."""
    with pytest.raises(ValueError, match="feed variant"):
        replace(CIRCULATION, tail_noise_factor=2.0)
    with pytest.raises(ValueError, match="positive"):
        replace(SETTINGS, outlier_factor=0.0)
