"""Tests for ProblemBuilder and the ClothoidSmoother integration."""

import math
import numpy as np
import pytest

from smoother_clothoid_py.options import SmootherParams, OptimizerParams
from smoother_clothoid_py.smoother import ClothoidSmoother
from smoother_clothoid_py.smoother_request import SmootherRequest
from smoother_clothoid_py.exceptions import InvalidPath, InvalidCostmap, SmoothingFailureInfo
from smoother_clothoid_py.problem_builder import ProblemBuilder, ProcessedPath
from smoother_clothoid_py.costmap2d import Costmap2D


# ----- build_processed_path -----

def test_simple_path_state_count():
    cm = Costmap2D(20, 20, 0.1)
    path = [np.array([i * 0.5, 1.0, 1.0]) for i in range(5)]
    proc = ProblemBuilder.build_processed_path(
        path, np.array([1, 0]), np.array([1, 0]), SmootherParams(), cm)
    assert proc.state_count == 5
    assert all(g == 1.0 for g in proc.gears)
    assert not any(proc.is_cusp_segment)


def test_reversing_gear_assignment():
    cm = Costmap2D(20, 20, 0.1)
    path = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, -1.0]), np.array([2.0, 0.0, 1.0])]
    proc = ProblemBuilder.build_processed_path(
        path, np.array([1, 0]), np.array([1, 0]), SmootherParams(), cm)
    # A cusp state is inserted in between, so state_count=4 but only 3 gears.
    # The cusp state's gear=0 is a passthrough; the actual transitions are [1, 0, -1].
    assert proc.state_count == 4
    assert proc.gears == [1.0, 0.0, -1.0]
    assert proc.is_cusp_segment[1] is True


def test_reversing_disabled_collapses_gears():
    cm = Costmap2D(20, 20, 0.1)
    path = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, -1.0]), np.array([2.0, 0.0, 1.0])]
    params = SmootherParams()
    params.reversing_enabled = False
    proc = ProblemBuilder.build_processed_path(
        path, np.array([1, 0]), np.array([1, 0]), params, cm)
    assert proc.state_count == 3
    assert all(g == 1.0 for g in proc.gears)
    assert not any(proc.is_cusp_segment)


def test_target_spacing_averaged():
    cm = Costmap2D(20, 20, 0.1)
    path = [np.array([0.0, 0.0, 1.0]), np.array([0.4, 0.0, 1.0]), np.array([0.8, 0.0, 1.0])]
    proc = ProblemBuilder.build_processed_path(
        path, np.array([1, 0]), np.array([1, 0]), SmootherParams(), cm)
    assert proc.target_spacing == pytest.approx(0.4, rel=1e-6)


def test_target_spacing_fallback_no_segments():
    cm = Costmap2D(20, 20, 0.1)
    path = [np.array([0.0, 0.0, 1.0])]
    proc = ProblemBuilder.build_processed_path(
        path, np.array([1, 0]), np.array([1, 0]), SmootherParams(), cm)
    assert proc.target_spacing == pytest.approx(cm.resolution)


def test_keep_orientation_enforced():
    cm = Costmap2D(20, 20, 0.1)
    path = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
    proc = ProblemBuilder.build_processed_path(
        path, np.array([0.0, 1.0]), np.array([0.0, -1.0]), SmootherParams(), cm)
    assert proc.start_theta == pytest.approx(math.pi / 2)
    assert proc.end_theta == pytest.approx(-math.pi / 2)
    assert proc.initial_variables[2] == pytest.approx(math.pi / 2)
    assert proc.initial_variables[-3] == pytest.approx(-math.pi / 2)


def test_no_costmap_target_spacing_default():
    path = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
    proc = ProblemBuilder.build_processed_path(
        path, np.array([1, 0]), np.array([1, 0]), SmootherParams(), None)
    # No costmap => target_spacing is the average segment length (1.0 for this path).
    # The default 0.2 fallback only triggers when there are no segments at all.
    assert proc.target_spacing == pytest.approx(1.0)


def test_no_costmap_target_spacing_fallback_to_default():
    # Path with all coincident points => no segments, no segments avg => 0.2 fallback
    path = [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])]
    proc = ProblemBuilder.build_processed_path(
        path, np.array([1, 0]), np.array([1, 0]), SmootherParams(), None)
    # With coincident points norm=0 < 1e-6, so ds is not accumulated, sp_cnt stays 0
    assert proc.target_spacing == pytest.approx(0.2)


# ----- apply_bounds -----

def test_apply_bounds_curvature_limits():
    lower = np.zeros(10)
    upper = np.zeros(10)
    refs = [(0.0, 0.0), (1.0, 0.0)]
    ProblemBuilder.apply_bounds(lower, upper, refs, 2, mc=1.0, ms=0.0, md=0.0)
    assert lower[3] == pytest.approx(-1.0)
    assert upper[3] == pytest.approx(1.0)
    assert lower[4] == pytest.approx(0.0)
    assert upper[4] == pytest.approx(0.0)


def test_apply_bounds_max_spacing():
    lower = np.zeros(5)
    upper = np.zeros(5)
    refs = [(0.0, 0.0)]
    ProblemBuilder.apply_bounds(lower, upper, refs, 1, mc=1.0, ms=0.5, md=0.0)
    assert lower[4] == pytest.approx(0.0)
    assert upper[4] == pytest.approx(0.5)


def test_apply_bounds_position_deviation():
    lower = np.zeros(5)
    upper = np.zeros(5)
    refs = [(1.0, 2.0)]
    ProblemBuilder.apply_bounds(lower, upper, refs, 1, mc=1.0, ms=0.0, md=0.5)
    assert lower[0] == pytest.approx(0.5)
    assert upper[0] == pytest.approx(1.5)
    assert lower[1] == pytest.approx(1.5)
    assert upper[1] == pytest.approx(2.5)


def test_apply_bounds_min_curvature_floor():
    lower = np.zeros(5)
    upper = np.zeros(5)
    refs = [(0.0, 0.0)]
    ProblemBuilder.apply_bounds(lower, upper, refs, 1, mc=0.0, ms=0.0, md=0.0)
    assert lower[3] == pytest.approx(-1e-6)
    assert upper[3] == pytest.approx(1e-6)


# ----- unpack_path / upsample_path -----

def test_unpack_path_normalizes_yaw():
    vars = [0.0, 0.0, 5 * math.pi, 0.0, 0.0, 1.0, 1.0, 7 * math.pi, 0.0, 0.0]
    out = ProblemBuilder.unpack_path(np.array(vars), 2)
    assert len(out) == 2
    assert abs(out[0][2] - math.pi) < 1e-6
    assert abs(out[1][2] - math.pi) < 1e-6


def test_upsample_no_factor():
    proc = _make_proc(2)
    vars = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    out = ProblemBuilder.upsample_path(vars, proc, SmootherParams())
    assert len(out) == 2


def test_upsample_distribution():
    proc = _make_proc(2)
    vars = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.4, 0.2, 0.0, 0.0])
    params = SmootherParams()
    params.path_upsampling_factor = 4
    out = ProblemBuilder.upsample_path(vars, proc, params)
    assert len(out) == 5
    assert abs(out[1][0] - 0.15) < 1e-6
    assert abs(out[2][0] - 0.30) < 1e-6
    assert abs(out[3][0] - 0.45) < 1e-6


def test_upsample_cusp_passthrough():
    cm = Costmap2D(20, 20, 0.1)
    proc = ProblemBuilder.build_processed_path(
        [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, -1.0]), np.array([2.0, 0.0, 1.0])],
        np.array([1, 0]), np.array([1, 0]), SmootherParams(), cm)
    assert proc.state_count == 4
    vars = np.array([0.0, 0.0, 0.0, 0.0, 1.0,
                     1.0, 0.0, 0.0, 0.0, 0.0,
                     1.0, 0.0, math.pi, 0.0, 0.0,
                     2.0, 0.0, 0.0, 0.0, 0.0])
    params = SmootherParams()
    params.path_upsampling_factor = 4
    out = ProblemBuilder.upsample_path(vars, proc, params)
    assert np.allclose(out[-1][:2], np.array([2.0, 0.0]), atol=1e-6)


# ----- helpers -----

def _make_proc(n: int) -> ProcessedPath:
    p = ProcessedPath()
    p.state_count = n
    p.gears = [1.0] * (n - 1)
    p.is_cusp_segment = [False] * (n - 1)
    p.reference_points = [(float(i), 0.0) for i in range(n)]
    p.start_theta = 0.0
    p.end_theta = 0.0
    p.target_spacing = 0.2
    p.initial_variables = [v for i in range(n) for v in [float(i), 0.0, 0.0, 0.0, 1.0]]
    return p

