"""Tests for math utilities."""

import math
import pytest

from smoother_clothoid_py.utils import (
    normalize_angle, angle_diff, EPSILON, PI,
    world_to_grid, grid_to_world, in_bounds, goal_position_frame_heading,
)


def test_eps_pi_values():
    assert EPSILON > 0
    assert PI == pytest.approx(math.pi)


@pytest.mark.parametrize("angle,expected", [
    (0.0, 0.0),
    (PI, PI),
    (-PI, -PI),  # atan2(sin(-PI), cos(-PI)) = atan2(tiny, -1) ≈ -PI (not +PI)
    (2 * PI, 0.0),
    (-2 * PI, 0.0),
    (PI / 2, PI / 2),
    (-PI / 2, -PI / 2),
    (3 * PI, PI),
    (-3 * PI, -PI),
    (3 * PI / 2, -PI / 2),
    (-3 * PI / 2, PI / 2),
    (5 * PI, PI),
    (-5 * PI, -PI),
    (0.1, 0.1),
    (-0.1, -0.1),
])
def test_normalize_angle(angle, expected):
    assert abs(normalize_angle(angle) - expected) < 1e-10


def test_normalize_angle_result_in_range():
    for a in [-7.0, -3.5, -0.001, 0.0, 0.001, 1.5, 4.7, 12.3]:
        n = normalize_angle(a)
        assert -PI - 1e-12 <= n <= PI + 1e-12


def test_angle_diff_basic():
    assert abs(angle_diff(0.0, 0.0)) < 1e-10
    assert abs(angle_diff(PI, 0.0) - PI) < 1e-10
    assert abs(angle_diff(0.0, PI) - (-PI)) < 1e-10
    assert abs(angle_diff(PI / 4, PI / 4)) < 1e-10
    assert abs(angle_diff(PI / 2, -PI / 2) - PI) < 1e-10


def test_angle_diff_antisymmetric():
    a, b = 1.234, -0.789
    assert abs(angle_diff(a, b) + angle_diff(b, a)) < 1e-10


def test_world_grid_roundtrip():
    from smoother_clothoid_py.costmap2d import Costmap2D
    cm = Costmap2D(100, 100, 0.05, 0.0, 0.0)
    for wx, wy in [(0.0, 0.0), (1.0, 2.0), (4.95, 4.95), (0.075, 0.075)]:
        gx, gy = world_to_grid(cm, wx, wy)
        assert 0 <= gx < 100
        assert 0 <= gy < 100
        rcx, rcy = grid_to_world(cm, gx, gy)
        assert abs(rcx - wx) <= 0.05 + 1e-9
        assert abs(rcy - wy) <= 0.05 + 1e-9


def test_in_bounds():
    assert in_bounds(0, 0, 10, 10)
    assert in_bounds(9, 9, 10, 10)
    assert not in_bounds(-1, 0, 10, 10)
    assert not in_bounds(0, 10, 10, 10)
    assert not in_bounds(10, 0, 10, 10)


def test_goal_position_frame_heading_keep():
    refs = [(0.0, 0.0), (1.0, 0.0)]
    assert goal_position_frame_heading(refs, 0.5, True) == 0.5


def test_goal_position_frame_heading_not_keep():
    refs = [(0.0, 0.0), (1.0, 0.0)]
    assert abs(goal_position_frame_heading(refs, 0.5, False) - 0.0) < 1e-10


def test_goal_position_frame_heading_single_ref():
    assert goal_position_frame_heading([(0.0, 0.0)], 0.7, False) == 0.7


def test_goal_position_frame_heading_zero_dist():
    refs = [(0.0, 0.0), (0.0, 0.0)]
    assert goal_position_frame_heading(refs, 0.7, False) == 0.7
