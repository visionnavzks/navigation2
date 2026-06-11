"""Tests for the Python smoother_clothoid module."""

import math
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smoother_clothoid_py.options import SmootherParams, OptimizerParams, LinearSolver
from smoother_clothoid_py.exceptions import (
    ErrorCode, SmoothingFailureReason, SmoothingFailureInfo,
    InvalidPath, InvalidCostmap, FailedToSmoothPath,
    to_error_code_string, to_failure_reason_string,
    build_smoothing_failure_message, throw_or_store_smoothing_failure,
)
from smoother_clothoid_py.utils import normalize_angle, angle_diff, EPSILON, PI
from smoother_clothoid_py.costmap2d import Costmap2D
from smoother_clothoid_py.problem_builder import ProcessedPath, ProblemBuilder
from smoother_clothoid_py.smoother import ClothoidSmoother
from smoother_clothoid_py.smoother_request import SmootherResult, SmootherRequest


def test_error_code_strings():
    assert to_error_code_string(ErrorCode.InvalidPath) == "SC_INVALID_PATH"
    assert to_error_code_string(ErrorCode.FailedToSmoothPath) == "SC_SMOOTHING_FAILED"


def test_failure_reason_strings():
    assert to_failure_reason_string(SmoothingFailureReason.CurvatureConstraint) == "curvature_constraint"


def test_smoothing_failure_message():
    msg = build_smoothing_failure_message(SmoothingFailureReason.GoalOrientationConstraint, "test", 7)
    assert msg == "goal_orientation_constraint@7: test"


def test_invalid_path_exception():
    e = InvalidPath("test")
    assert e.code == ErrorCode.InvalidPath
    assert e.code_string == "SC_INVALID_PATH"


def test_throw_or_store_with_failure():
    f = SmoothingFailureInfo()
    assert throw_or_store_smoothing_failure(f, SmoothingFailureReason.CurvatureConstraint, "kappa", 3) is False
    assert f.reason == SmoothingFailureReason.CurvatureConstraint
    assert f.failed_index == 3


def test_throw_or_store_without_failure():
    with pytest.raises(FailedToSmoothPath):
        throw_or_store_smoothing_failure(None, SmoothingFailureReason.SolverRejectedSolution, "fail")


def test_normalize_angle():
    assert abs(normalize_angle(0.0)) < 1e-10
    assert abs(normalize_angle(PI) - PI) < 1e-10
    assert abs(normalize_angle(2 * PI)) < 1e-10


def test_angle_diff():
    assert abs(angle_diff(PI, 0.0) - PI) < 1e-10
    assert abs(angle_diff(PI/4, PI/4)) < 1e-10


def test_costmap2d():
    c = Costmap2D(100, 100, 0.05, 0.0, 0.0)
    assert c.size_x == 100
    c.set_cost(5, 5, Costmap2D.LETHAL_OBSTACLE)
    assert c.get_cost(5, 5) == Costmap2D.LETHAL_OBSTACLE


def test_smoother_params_defaults():
    p = SmootherParams()
    assert p.model_weight_sqrt == 0.0
    assert p.fix_weight == 100.0
    assert p.reversing_enabled is True


def test_obstacle_terms():
    assert SmootherParams().obstacle_terms_enabled() is False
    assert SmootherParams(costmap_weight_sqrt=1.0).obstacle_terms_enabled() is True


def test_build_processed_path_cusp():
    costmap = Costmap2D(40, 40, 0.05, 0.0, 0.0)
    path = [np.array([0,0,1.0]), np.array([1,0,-1.0]), np.array([0.5,0,-1.0])]
    params = SmootherParams()
    p = ProblemBuilder.build_processed_path(path, np.array([1,0]), np.array([-1,0]), params, costmap)
    assert p.state_count == 4
    assert p.gears[0] == 1.0
    assert p.gears[1] == 0.0
    assert p.gears[2] == -1.0
    assert p.is_cusp_segment[1] is True


def test_build_processed_path_no_reversing():
    costmap = Costmap2D(40, 40, 0.05, 0.0, 0.0)
    path = [np.array([0,0,1.0]), np.array([1,0,-1.0]), np.array([0.5,0,-1.0])]
    params = SmootherParams()
    params.reversing_enabled = False
    p = ProblemBuilder.build_processed_path(path, np.array([1,0]), np.array([1,0]), params, costmap)
    assert p.state_count == 3
    assert p.gears[0] == 1.0
    assert p.gears[1] == 1.0


def test_smoother_init():
    s = ClothoidSmoother()
    s.initialize(OptimizerParams())


def test_smooth_straight_path():
    costmap = Costmap2D(100, 100, 0.05, 0.0, 0.0)
    path = [np.array([0.5 + i*0.1, 2.5, 1.0]) for i in range(10)]
    params = SmootherParams()
    params.model_weight_sqrt = math.sqrt(20.0)
    params.reference_path_weight_sqrt = math.sqrt(1.0)
    params.kinematic_curvature_weight_sqrt = math.sqrt(30.0)
    params.kinematic_curvature_rate_weight_sqrt = math.sqrt(5.0)
    params.max_curvature = 1.0 / 0.4
    params.max_time = 1.0
    opt = OptimizerParams()
    opt.max_iterations = 30
    s = ClothoidSmoother()
    s.initialize(opt)
    r = s.smooth(SmootherRequest(path=path, start_dir=np.array([1,0]), end_dir=np.array([1,0]),
                                  costmap=costmap, params=params))
    assert r.success
    assert len(r.smoothed_path) >= 2


def test_smoother_result_defaults():
    r = SmootherResult()
    assert r.success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
