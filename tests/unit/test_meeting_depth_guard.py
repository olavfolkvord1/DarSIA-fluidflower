"""Local injector depth gates a proposed link before any inferred support."""

from types import SimpleNamespace

import numpy as np
import pytest

from darsia.multiphase import gas_zone


def _case(start=(3, 3), meeting=(7, 3)):
    rows, cols = np.indices((13, 13))
    positions = SimpleNamespace(
        x=cols.astype(float) / 1024,
        y=-rows.astype(float) / 1024,
        pitch_mm=(1000 / 1024, 1000 / 1024),
    )
    contour = np.zeros(rows.shape, bool)
    contour[meeting] = True
    current = contour.copy()
    path = np.ones(rows.shape, bool)
    return dict(
        port=(positions.x[start], positions.y[start]),
        contour=contour,
        observed=current.copy(),
        probability=np.ones(rows.shape),
        path=path,
        positions=positions,
        settings=SimpleNamespace(link_smoothing_mm=(0, 0)),
        current_co2=current,
        depth_m=np.full(rows.shape, 1 / 64),
    )


def test_local_injector_depth_overrides_larger_run_and_meeting_depth():
    case = _case()
    case["depth_m"][3, 3] = 3 / 1024
    line, coordinates, note = gas_zone._injector_link(**case)
    assert line is coordinates is None
    assert note["status"] == "meeting_below_local_depth"
    assert note["local_depth_mm"] == 3000 / 1024
    assert note["downward_drop_mm"] == 4000 / 1024
    assert note["meeting_pixel"] == [7, 3]
    case["depth_m"][3, 3] = 5 / 1024
    line, coordinates, note = gas_zone._injector_link(**case)
    assert line is not None and note["status"] == "linked"
    assert tuple(coordinates[0]) == (3, 3)


def test_equal_depth_is_allowed_and_immediately_smaller_depth_refused():
    case = _case()
    case["depth_m"][3, 3] = 4 / 1024
    assert gas_zone._injector_link(**case)[2]["status"] == "linked"
    case["depth_m"][3, 3] = np.nextafter(4 / 1024, 0)
    assert gas_zone._injector_link(**case)[2]["status"] == "meeting_below_local_depth"


def test_actual_protocol_height_sets_drop_not_rounded_cell_height():
    case = _case()
    case["port"] = (case["port"][0], case["port"][1] + 0.25 / 1024)
    case["depth_m"][3, 3] = 4 / 1024
    _, _, note = gas_zone._injector_link(**case)
    assert note["status"] == "meeting_below_local_depth"
    assert note["downward_drop_mm"] == 4250 / 1024


def test_upward_meeting_does_not_receive_a_symmetric_distance_cutoff():
    case = _case(start=(7, 3), meeting=(3, 3))
    case["depth_m"][7, 3] = 1 / 1024
    line, _, note = gas_zone._injector_link(**case)
    assert line is not None and note["downward_drop_mm"] < 0


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf])
def test_invalid_local_depth_refuses_link_clearly(invalid):
    case = _case()
    case["depth_m"][3, 3] = invalid
    with pytest.raises(ValueError, match="finite positive local rig depth"):
        gas_zone._injector_link(**case)


def test_mismatched_depth_grid_is_refused():
    case = _case()
    case["depth_m"] = np.ones((2, 2))
    with pytest.raises(ValueError, match="depth map must share"):
        gas_zone._injector_link(**case)


def test_unknown_or_seal_on_straight_segment_still_refuses_with_sufficient_depth():
    case = _case()
    case["path"][5, 3] = False
    line, coordinates, note = gas_zone._injector_link(**case)
    assert line is coordinates is None
    assert note["status"] == "straight_hidden_segment_blocked"


def test_unknown_injector_cannot_receive_link():
    case = _case()
    case["path"][3, 3] = False
    assert gas_zone._injector_link(**case)[2]["status"] == "no_porosity_at_port"


def test_zero_hidden_length_link_still_requires_valid_local_depth():
    case = _case()
    case["current_co2"][3:8, 3] = True
    case["observed"] = case["current_co2"].copy()
    line, coordinates, note = gas_zone._injector_link(**case)
    assert line is not None and note["meeting_pixel"] == [3, 3]
    assert note["downward_drop_mm"] == 0
    assert tuple(coordinates[0]) == (3, 3)
    case["depth_m"][3, 3] = np.nan
    with pytest.raises(ValueError, match="finite positive local rig depth"):
        gas_zone._injector_link(**case)


def test_already_inside_has_no_link_and_preserves_ordinary_geometry():
    case = _case(start=(7, 3), meeting=(7, 3))
    old = case["contour"].copy()
    line, coordinates, note = gas_zone._injector_link(**case)
    assert line is coordinates is None and note["status"] == "already_inside"
    assert np.array_equal(case["contour"], old)
