"""The real no-link producer must preserve the approved zero-hidden fallback."""

from dataclasses import replace
from dataclasses import asdict
import json

import numpy as np
import pytest

from darsia.multiphase import gas_zone
from darsia.multiphase.gas_allocation import GasAllocationRecord
from darsia.multiphase.phase_rules import PhaseLabel
from tests.unit.test_gas_link_seed_fallback import _allocate, _fixture
from tests.unit.test_gas_limit import RULES, SETTINGS


def _inside_scene():
    scene = _fixture(hidden=False)
    point = (64, 115)
    scene["meeting"] = point
    port = (
        float(scene["positions"].x[point]),
        float(scene["positions"].y[point]),
    )
    scene["protocol"].df.loc[1, ["location_x", "location_y"]] = port
    scene["zone"] = next(
        gas_zone.gas_model_zones(
            [scene["frame"]],
            [scene["date"]],
            scene["image"],
            scene["domain"],
            {1: 1.0, 2: 0.01},
            scene["protocol"],
            RULES,
            SETTINGS,
            
            depth_m=np.full(scene["domain"].shape, 0.005),
        )
    )
    assert scene["zone"].diagnostics["ports"][port]["status"] == "already_inside"
    assert not scene["zone"].links
    assert scene["frame"].labels[point] == PhaseLabel.DISSOLVED
    return scene, port


def test_real_already_inside_producer_meeting_restores_new_component():
    scene, port = _inside_scene()
    contour = scene["zone"].contour.copy()
    maps, record = _allocate(scene)
    assert maps[2][scene["meeting"]] > 0
    assert record.disconnected_added_area_mm2 == 0
    assert record.bridge_volume_m3 == record.bridge_area_mm2 == 0
    assert np.isclose(record.placed_m3, scene["target"], rtol=2e-7, atol=1e-15)
    np.testing.assert_array_equal(scene["zone"].contour, contour)
    np.testing.assert_array_equal(maps[0], scene["frame"].labels)
    np.testing.assert_array_equal(maps[4], scene["frame"].before)
    assert np.all(maps[2] <= 1 - scene["swi"])
    assert record.active_link_diagnostics == [
        {
            "port_m": list(port),
            "status": "already_inside",
            "port_pixel": [64, 115],
        }
    ]
    assert (
        json.loads(json.dumps(asdict(record), allow_nan=False))[
            "active_link_diagnostics"
        ]
        == record.active_link_diagnostics
    )
    # Removing only the affirmative producer diagnostic reproduces the old gap:
    # old global seeds suppress the small new component despite the same contour.
    scene["zone"] = replace(scene["zone"], diagnostics={})
    prior, missing = _allocate(scene)
    assert prior[2][scene["meeting"]] == 0
    assert missing.unplaced_m3 > 0


@pytest.mark.parametrize(
    "status",
    [
        "straight_hidden_segment_blocked",
        "no_current_co2_target",
        "no_porosity_at_port",
        "meeting_below_local_depth",
        "no_finite_path",
    ],
)
def test_other_no_link_status_does_not_invent_zero_hidden_meeting(status):
    scene, port = _inside_scene()
    note = dict(scene["zone"].diagnostics["ports"][port], status=status)
    scene["zone"] = replace(scene["zone"], diagnostics={"ports": {port: note}})
    maps, record = _allocate(scene)
    assert maps[2][scene["meeting"]] == 0 and record.unplaced_m3 > 0


@pytest.mark.parametrize("reason", ["inactive", "unknown", "cap_zero", "zero_score"])
def test_inside_diagnostic_does_not_bypass_final_admission(reason):
    scene, port = _inside_scene()
    if reason == "inactive":
        scene["protocol"].df.loc[1, "end"] = scene["date"]
    elif reason == "unknown":
        labels = scene["frame"].labels.copy()
        labels[scene["meeting"]] = PhaseLabel.UNKNOWN
        scene["frame"] = replace(scene["frame"], labels=labels)
    elif reason == "cap_zero":
        scene["swi"][scene["meeting"]] = 1
    else:
        probability = scene["frame"].gas_probability.copy()
        probability[45:115, 95:145] = 0
        scene["frame"] = replace(scene["frame"], gas_probability=probability)
    maps, record = _allocate(scene)
    assert maps[2][scene["meeting"]] == 0
    assert record.unplaced_m3 > 0


def test_depth_rejection_provenance_survives_record_as_native_json():
    scene, port = _inside_scene()
    note = {
        "status": "meeting_below_local_depth",
        "port_pixel": np.array([64, 115], np.int32),
        "meeting_pixel": np.array([80, 115], np.int64),
        "local_depth_mm": np.float32(5),
        "downward_drop_mm": np.float64(16),
    }
    scene["zone"] = replace(scene["zone"], diagnostics={"ports": {port: note}})
    _, record = _allocate(scene)
    saved = json.loads(json.dumps(asdict(record), allow_nan=False))
    assert saved["active_link_diagnostics"] == [
        {
            "port_m": list(port),
            "status": "meeting_below_local_depth",
            "port_pixel": [64, 115],
            "meeting_pixel": [80, 115],
            "local_depth_mm": 5.0,
            "downward_drop_mm": 16.0,
        }
    ]
    assert GasAllocationRecord(None, 0, 0, 0, 0, 0).active_link_diagnostics == []
