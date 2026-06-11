"""Basic tests for the Python constrained_smoother rewrite."""

import math
import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constrained_smoother.options import SmootherParams, OptimizerParams, LinearSolver
from constrained_smoother.exceptions import (
    ErrorCode,
    SmoothingFailureReason,
    SmoothingFailureInfo,
    InvalidPath,
    InvalidCostmap,
    FailedToSmoothPath,
    to_error_code_string,
    to_failure_reason_string,
    build_smoothing_failure_message,
    throw_or_store_smoothing_failure,
)
from constrained_smoother.utils import normalize_angle, angle_diff, EPSILON, PI
from constrained_smoother.costmap2d import Costmap2D
from constrained_smoother.problem_builder import (
    KinematicProcessedPath,
    KinematicSmootherProblemBuilder,
)
from constrained_smoother.smoother import KinematicSmoother
from constrained_smoother.smoother_request import SmootherResult, SmootherRequest


# ---- Error code tests ----

def test_error_code_strings():
    assert to_error_code_string(ErrorCode.InvalidPath) == "CS_INVALID_PATH"
    assert to_error_code_string(ErrorCode.FailedToSmoothPath) == "CS_SMOOTHING_FAILED"
    assert to_error_code_string(ErrorCode.InvalidCostmap) == "CS_INVALID_COSTMAP"
    assert to_error_code_string(ErrorCode.PrecomputedEsdfSizeMismatch) == "CS_PRECOMPUTED_ESDF_SIZE_MISMATCH"


def test_failure_reason_strings():
    assert to_failure_reason_string(SmoothingFailureReason.SolverRejectedSolution) == "solver_rejected_solution"
    assert to_failure_reason_string(SmoothingFailureReason.NoCostImprovement) == "no_cost_improvement"
    assert to_failure_reason_string(SmoothingFailureReason.InvalidStateVector) == "invalid_state_vector"
    assert to_failure_reason_string(SmoothingFailureReason.NonFiniteState) == "nonfinite_state"
    assert to_failure_reason_string(SmoothingFailureReason.CurvatureConstraint) == "curvature_constraint"


def test_smoothing_failure_message_format():
    msg = build_smoothing_failure_message(
        SmoothingFailureReason.GoalOrientationConstraint,
        "test smoothing failure",
        7,
    )
    assert msg == "goal_orientation_constraint@7: test smoothing failure"


def test_smoothing_failure_message_no_index():
    msg = build_smoothing_failure_message(
        SmoothingFailureReason.SolverRejectedSolution,
        "solver failed",
    )
    assert msg == "solver_rejected_solution: solver failed"


def test_invalid_path_exception():
    error = InvalidPath("test invalid path")
    assert error.code == ErrorCode.InvalidPath
    assert error.code_string == "CS_INVALID_PATH"
    assert str(error) == "test invalid path"


def test_invalid_costmap_exception():
    error = InvalidCostmap("test invalid costmap")
    assert error.code == ErrorCode.InvalidCostmap
    assert error.code_string == "CS_INVALID_COSTMAP"


def test_failed_to_smooth_exception():
    error = FailedToSmoothPath("test failure")
    assert error.code == ErrorCode.FailedToSmoothPath
    assert error.code_string == "CS_SMOOTHING_FAILED"


def test_throw_or_store_with_failure():
    failure = SmoothingFailureInfo()
    result = throw_or_store_smoothing_failure(
        failure,
        SmoothingFailureReason.CurvatureConstraint,
        "curvature too high",
        3,
    )
    assert result is False
    assert failure.reason == SmoothingFailureReason.CurvatureConstraint
    assert failure.message == "curvature too high"
    assert failure.failed_index == 3


def test_throw_or_store_without_failure():
    with pytest.raises(FailedToSmoothPath):
        throw_or_store_smoothing_failure(
            None,
            SmoothingFailureReason.SolverRejectedSolution,
            "solver failed",
        )


# ---- Math utility tests ----

def test_normalize_angle():
    assert abs(normalize_angle(0.0) - 0.0) < 1e-10
    assert abs(normalize_angle(PI) - PI) < 1e-10
    assert abs(normalize_angle(-PI) + PI) < 1e-10  # atan2(-0.0, -1.0) = -PI
    assert abs(normalize_angle(2 * PI) - 0.0) < 1e-10
    assert abs(normalize_angle(3 * PI / 2) - (-PI / 2)) < 1e-10


def test_angle_diff():
    assert abs(angle_diff(PI, 0.0) - PI) < 1e-10
    assert abs(angle_diff(0.0, PI) - (-PI)) < 1e-10
    assert abs(angle_diff(PI / 4, PI / 4)) < 1e-10


# ---- Costmap2D tests ----

def test_costmap2d_basic():
    costmap = Costmap2D(100, 100, 0.05, 0.0, 0.0)
    assert costmap.size_x == 100
    assert costmap.size_y == 100
    assert costmap.resolution == 0.05
    assert costmap.origin_x == 0.0
    assert costmap.origin_y == 0.0
    assert costmap.get_cost(50, 50) == 0


def test_costmap2d_set_cost():
    costmap = Costmap2D(10, 10, 0.1, 0.0, 0.0)
    costmap.set_cost(5, 5, Costmap2D.LETHAL_OBSTACLE)
    assert costmap.get_cost(5, 5) == Costmap2D.LETHAL_OBSTACLE


# ---- SmootherParams tests ----

def test_smoother_params_defaults():
    params = SmootherParams()
    assert params.model_weight_sqrt == 0.0
    assert params.fix_weight == 100.0
    assert params.kinematic_spacing_weight_sqrt == 1.0
    assert params.reversing_enabled is True
    assert params.keep_start_orientation is True
    assert params.keep_goal_orientation is True


def test_obstacle_terms_disabled_by_default():
    params = SmootherParams()
    assert params.obstacle_terms_enabled() is False


def test_obstacle_terms_enabled():
    params = SmootherParams(costmap_weight_sqrt=1.0)
    assert params.obstacle_terms_enabled() is True


def test_obstacle_terms_enabled_cusp():
    params = SmootherParams(cusp_costmap_weight_sqrt=1.0)
    assert params.obstacle_terms_enabled() is True


# ---- Processed path tests ----

def test_build_processed_path_inserts_cusp():
    costmap = Costmap2D(40, 40, 0.05, 0.0, 0.0)
    path = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([0.5, 0.0, -1.0]),
    ]

    params = SmootherParams()
    params.keep_start_orientation = True
    params.keep_goal_orientation = True

    processed = KinematicSmootherProblemBuilder.build_processed_path(
        path,
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        params,
        costmap,
    )

    assert processed.state_count == 4
    assert len(processed.gears) == 3
    assert len(processed.is_cusp_segment) == 3
    assert processed.gears[0] == 1.0
    assert processed.gears[1] == 0.0
    assert processed.gears[2] == -1.0
    assert processed.is_cusp_segment[0] is False
    assert processed.is_cusp_segment[1] is True
    assert processed.is_cusp_segment[2] is False
    assert abs(processed.reference_points[1][0] - 1.0) < 1e-9
    assert abs(processed.reference_points[2][0] - 1.0) < 1e-9
    assert len(processed.initial_variables) == processed.state_count * 5


def test_build_processed_path_disabled_reversing():
    costmap = Costmap2D(40, 40, 0.05, 0.0, 0.0)
    path = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([0.5, 0.0, -1.0]),
    ]

    params = SmootherParams()
    params.reversing_enabled = False
    params.keep_start_orientation = True
    params.keep_goal_orientation = True

    processed = KinematicSmootherProblemBuilder.build_processed_path(
        path,
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        params,
        costmap,
    )

    assert processed.state_count == 3
    assert len(processed.gears) == 2
    assert len(processed.is_cusp_segment) == 2
    assert processed.gears[0] == 1.0
    assert processed.gears[1] == 1.0
    assert processed.is_cusp_segment[0] is False
    assert processed.is_cusp_segment[1] is False


# ---- Solver tests ----

def test_kinematic_smoother_initialization():
    smoother = KinematicSmoother()
    opt_params = OptimizerParams()
    smoother.initialize(opt_params)
    # Should not raise


def test_kinematic_smoother_smooth_straight_path():
    costmap = Costmap2D(100, 100, 0.05, 0.0, 0.0)

    path = []
    for i in range(10):
        x = 0.5 + i * 0.1
        y = 2.5
        path.append(np.array([x, y, 1.0]))

    params = SmootherParams()
    params.model_weight_sqrt = math.sqrt(20.0)
    params.reference_path_weight_sqrt = math.sqrt(1.0)
    params.kinematic_curvature_weight_sqrt = math.sqrt(30.0)
    params.kinematic_curvature_rate_weight_sqrt = math.sqrt(5.0)
    params.max_curvature = 1.0 / 0.4
    params.max_time = 1.0
    params.obstacle_safe_distance = 0.5

    opt_params = OptimizerParams()
    opt_params.max_iterations = 30

    smoother = KinematicSmoother()
    smoother.initialize(opt_params)

    start_dir = np.array([1.0, 0.0])
    end_dir = np.array([1.0, 0.0])

    request = SmootherRequest(
        path=path,
        start_dir=start_dir,
        end_dir=end_dir,
        costmap=costmap,
        params=params,
    )

    result = smoother.smooth(request)

    assert result.success is True
    assert len(result.candidate_path) > 0
    assert len(result.smoothed_path) >= 2
    assert result.optimized_knot_count > 0
    assert result.target_spacing > 0.0


# ---- SmootherRequest and SmootherResult tests ----

def test_smoother_result_defaults():
    result = SmootherResult()
    assert result.success is False
    assert result.optimized_knot_count == 0
    assert result.target_spacing == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
