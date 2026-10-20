"""Joining an outline never moves the neighbour or mutates an input."""

import copy

import pytest
from shapely.geometry import LineString, Polygon

from darsia.fluidflower.layer_painting_join import JoinEdgesError, join_to_neighbour


def rectangle(top, left, bottom, right):
    return [(top, left), (top, right), (bottom, right), (bottom, left)]


@pytest.mark.parametrize("bottom", [8.0, 12.0])
def test_gap_and_overlap_follow_the_master_without_moving_it(bottom):
    target = rectangle(0, 0, bottom, 20)
    master = [(10, 0), (10, 8), (10, 15), (10, 20), (30, 20), (30, 0)]
    original = copy.deepcopy((target, master))
    plan = join_to_neighbour(target, master, 3)
    assert (target, master) == original
    assert len(plan.stretches) == 1
    assert Polygon(plan.target).is_valid
    assert Polygon(plan.target).intersection(Polygon(master)).area == 0
    assert LineString([*plan.target, plan.target[0]]).intersection(
        LineString([*master, master[0]])
    ).length == pytest.approx(20)
    assert tuple(master) == plan.neighbour
    assert (10, 8) in plan.target and (10, 15) in plan.target


def test_mid_edge_endpoints_are_inserted_in_original_order_at_most_twice():
    target = rectangle(0, 6, 8, 14)
    master = rectangle(10, 0, 30, 20)
    plan = join_to_neighbour(target, master, 3)
    assert len(plan.neighbour) == len(master) + 2
    assert [p for p in plan.neighbour if p in master] == master
    assert plan.neighbour[:4] == ((10, 0), (10, 6), (10, 14), (10, 20))
    assert Polygon(plan.neighbour).equals(Polygon(master))
    assert (10, 6) in plan.target and (10, 14) in plan.target


def test_far_outline_and_zero_distance_leave_every_point_unchanged():
    target, master = rectangle(0, 0, 8, 20), rectangle(20, 0, 40, 20)
    for distance in (0, 5):
        plan = join_to_neighbour(target, master, distance)
        assert plan.target == tuple(target)
        assert plan.neighbour == tuple(master)
        assert not plan.stretches


def test_already_shared_outline_is_not_an_edit():
    target, master = rectangle(0, 0, 10, 20), rectangle(10, 0, 30, 20)
    plan = join_to_neighbour(target, master, 3)
    assert not plan.stretches
    assert plan.target == tuple(target)
    assert plan.neighbour == tuple(master)


def test_a_join_crossing_the_first_vertex_is_one_continuous_stretch():
    target = [(8, 0), (0, 0), (0, 20), (8, 20)]
    master = rectangle(10, 0, 30, 20)
    plan = join_to_neighbour(target, master, 3)
    assert len(plan.stretches) == 1
    assert Polygon(plan.target).is_valid
    assert Polygon(plan.target).intersection(Polygon(master)).area == 0


def test_cannot_collapse_the_whole_polygon_onto_the_neighbour():
    with pytest.raises(JoinEdgesError, match="whole outline"):
        join_to_neighbour(rectangle(0, 0, 1, 1), rectangle(0, 0, 2, 2), 10)


@pytest.mark.parametrize("which", [0, 1])
def test_self_intersection_is_refused_with_its_location(which):
    inputs = [rectangle(0, 0, 8, 20), rectangle(10, 0, 30, 20)]
    inputs[which] = [(0, 0), (10, 10), (0, 10), (10, 0)]
    before = copy.deepcopy(inputs)
    with pytest.raises(JoinEdgesError, match="Self-intersection") as caught:
        join_to_neighbour(*inputs, 3)
    assert caught.value.location == (5, 5)
    assert inputs == before


@pytest.mark.parametrize("distance", [-1, float("nan"), float("inf")])
def test_invalid_distance_is_refused(distance):
    with pytest.raises(JoinEdgesError, match="finite"):
        join_to_neighbour(rectangle(0, 0, 8, 20), rectangle(10, 0, 30, 20), distance)


def test_join_that_would_cross_another_part_of_the_target_is_refused():
    target = [(7.28, 15.31), (10.27, 9.01), (9.92, 1.37), (18.61, -1.35),
              (-7.14, -8.10), (-14.97, -3.25), (-15.70, 6.53), (-1.93, 1.09),
              (-12.19, 7.42), (-11.89, 10.04), (-6.28, 11.96), (-0.84, 6.84)]
    master = [(-30, 10), (30, 10), (30, 40), (-30, 40)]
    assert Polygon(target).is_valid and Polygon(master).is_valid
    before = copy.deepcopy((target, master))
    with pytest.raises(JoinEdgesError, match="Joined polygon: Self-intersection") as caught:
        join_to_neighbour(target, master, 7)
    assert caught.value.location is not None
    assert (target, master) == before


def test_original_master_vertices_are_retained_even_if_two_coincide():
    master = [(10, 0), (10, 0), (10, 20), (30, 20), (30, 0)]
    plan = join_to_neighbour(rectangle(0, 6, 8, 14), master, 3)
    assert [p for p in plan.neighbour if p in master] == master
    assert len(plan.neighbour) == len(master) + 2


def test_transverse_crossing_is_not_collapsed_into_one_shared_point():
    # Almost perpendicular outlines: their projections are only 0.12 px
    # apart, below the existing 0.5 px shared-point tolerance.
    target = [(0, 0), (0, 100), (10, 100), (10, 0)]
    master = [(-10, 40), (30, 40.2), (30, 60.2), (-10, 60)]
    plan = join_to_neighbour(target, master, 3)
    assert not plan.stretches
    assert plan.target == tuple(target)
    assert plan.neighbour == tuple(master)
