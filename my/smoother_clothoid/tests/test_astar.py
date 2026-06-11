"""Tests for the A* planner used by the web demo."""

import os
import sys
import math
import numpy as np
import pytest

_WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
if _WEB not in sys.path:
    sys.path.insert(0, _WEB)

from astar import AStarPlanner, downsample_path


def test_world_to_grid():
    cm = np.zeros((100, 100), dtype=np.uint8)
    p = AStarPlanner(cm, 100, 100, 0.05, 0.0, 0.0)
    assert p.world_to_grid(0.0, 0.0) == (0, 0)
    assert p.world_to_grid(0.5, 0.5) == (10, 10)
    assert p.world_to_grid(2.5, 1.0) == (50, 20)


def test_grid_to_world_centers():
    cm = np.zeros((10, 10), dtype=np.uint8)
    p = AStarPlanner(cm, 10, 10, 0.1, 0.0, 0.0)
    wx, wy = p.grid_to_world(0, 0)
    assert (wx, wy) == (0.05, 0.05)
    wx, wy = p.grid_to_world(3, 4)
    assert (wx, wy) == pytest.approx((0.35, 0.45))


def test_in_bounds():
    cm = np.zeros((5, 5), dtype=np.uint8)
    p = AStarPlanner(cm, 5, 5, 0.1, 0.0, 0.0)
    assert p.in_bounds(0, 0)
    assert p.in_bounds(4, 4)
    assert not p.in_bounds(-1, 0)
    assert not p.in_bounds(5, 0)


def test_is_free():
    cm = np.zeros((5, 5), dtype=np.uint8)
    cm[2, 2] = 254
    p = AStarPlanner(cm, 5, 5, 0.1, 0.0, 0.0)
    assert p.is_free(0, 0)
    assert not p.is_free(2, 2)
    assert not p.is_free(10, 10)


def test_plan_straight_line():
    cm = np.zeros((50, 50), dtype=np.uint8)
    p = AStarPlanner(cm, 50, 50, 0.05, 0.0, 0.0)
    path = p.plan(0.05, 0.05, 0.45, 0.45)
    assert path is not None
    assert path[0] == pytest.approx(p.grid_to_world(1, 1))
    assert path[-1] == pytest.approx(p.grid_to_world(9, 9))
    # Path should be monotonic
    for i in range(1, len(path)):
        assert path[i][0] >= path[i-1][0] - 1e-9
        assert path[i][1] >= path[i-1][1] - 1e-9


def test_plan_around_obstacle():
    cm = np.zeros((50, 50), dtype=np.uint8)
    cm[20, :] = 254
    p = AStarPlanner(cm, 50, 50, 0.05, 0.0, 0.0)
    path = p.plan(0.05, 0.05, 0.45, 0.45)
    # All points on one side
    assert path is not None
    assert all(pt[1] < 20 * 0.05 - 1e-6 for pt in path)


def test_plan_no_path_returns_none():
    cm = np.zeros((10, 10), dtype=np.uint8)
    cm[:, 5] = 254
    p = AStarPlanner(cm, 10, 10, 0.1, 0.0, 0.0)
    path = p.plan(0.1, 0.1, 0.9, 0.9)
    assert path is None


def test_plan_blocked_start():
    cm = np.zeros((10, 10), dtype=np.uint8)
    cm[0, 0] = 254
    p = AStarPlanner(cm, 10, 10, 0.1, 0.0, 0.0)
    assert p.plan(0.05, 0.05, 0.5, 0.5) is None


def test_heuristic_is_admissible():
    # Octile distance should be >= true distance for all grid pairs
    cm = np.zeros((10, 10), dtype=np.uint8)
    p = AStarPlanner(cm, 10, 10, 0.1, 0.0, 0.0)
    for ax in range(10):
        for ay in range(10):
            for bx in range(10):
                for by in range(10):
                    h = p._heuristic(ax, ay, bx, by)
                    # Octile >= Chebyshev
                    assert h >= max(abs(ax - bx), abs(ay - by)) - 1e-9


def test_downsample_path_short():
    assert downsample_path([], 0.5) == []
    assert downsample_path([(0, 0)], 0.5) == [(0, 0)]
    assert downsample_path([(0, 0), (0.1, 0.1)], 0.5) == [(0, 0), (0.1, 0.1)]


def test_downsample_path_respects_spacing():
    path = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.6, 0.0), (1.0, 0.0)]
    out = downsample_path(path, 0.5)
    # first and last always kept
    assert out[0] == path[0]
    assert out[-1] == path[-1]
    # All intermediate points should be at least 0.5m apart (except final anchor)
    for i in range(1, len(out) - 1):
        d = math.hypot(out[i][0] - out[i-1][0], out[i][1] - out[i-1][1])
        assert d >= 0.5 - 1e-9


def test_downsample_path_preserves_endpoints():
    path = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0)]
    out = downsample_path(path, 0.5)
    assert out[0] == path[0]
    assert out[-1] == path[-1]
