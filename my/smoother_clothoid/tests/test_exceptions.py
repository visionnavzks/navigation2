"""Tests for error codes, failure reasons, and exceptions."""

import math
import pytest

from smoother_clothoid_py.exceptions import (
    ErrorCode, SmoothingFailureReason, SmoothingFailureInfo,
    InvalidPath, InvalidCostmap, PrecomputedEsdfSizeMismatch, FailedToSmoothPath,
    to_error_code_string, to_failure_reason_string,
    build_smoothing_failure_message, throw_or_store_smoothing_failure,
)


def test_all_error_codes_have_string():
    for code in ErrorCode:
        s = to_error_code_string(code)
        assert isinstance(s, str) and s.startswith("SC_")


def test_all_failure_reasons_have_string():
    for reason in SmoothingFailureReason:
        s = to_failure_reason_string(reason)
        assert isinstance(s, str) and len(s) > 0


def test_error_code_values_unique():
    codes = list(ErrorCode)
    assert len(codes) == len(set(codes))


def test_failure_reason_values_unique():
    reasons = list(SmoothingFailureReason)
    assert len(reasons) == len(set(reasons))


def test_build_message_format_with_index():
    msg = build_smoothing_failure_message(SmoothingFailureReason.GoalOrientationConstraint, "test", 7)
    assert msg == "goal_orientation_constraint@7: test"


def test_build_message_format_no_index():
    msg = build_smoothing_failure_message(SmoothingFailureReason.SolverRejectedSolution, "fail")
    assert msg == "solver_rejected_solution: fail"


def test_invalid_path_code_string():
    e = InvalidPath("x")
    assert e.code == ErrorCode.InvalidPath
    assert e.code_string == "SC_INVALID_PATH"
    assert str(e) == "x"


def test_invalid_costmap_code_string():
    e = InvalidCostmap("y")
    assert e.code == ErrorCode.InvalidCostmap
    assert e.code_string == "SC_INVALID_COSTMAP"


def test_precomputed_esdf_size_mismatch_code():
    e = PrecomputedEsdfSizeMismatch("z")
    assert e.code == ErrorCode.PrecomputedEsdfSizeMismatch
    assert e.code_string == "SC_PRECOMPUTED_ESDF_SIZE_MISMATCH"


def test_failed_to_smooth_code():
    e = FailedToSmoothPath("w")
    assert e.code == ErrorCode.FailedToSmoothPath
    assert e.code_string == "SC_SMOOTHING_FAILED"


def test_throw_or_store_with_failure_returns_false():
    f = SmoothingFailureInfo()
    assert throw_or_store_smoothing_failure(f, SmoothingFailureReason.CurvatureConstraint, "kappa", 3) is False
    assert f.reason == SmoothingFailureReason.CurvatureConstraint
    assert f.message == "kappa"
    assert f.failed_index == 3


def test_throw_or_store_with_failure_default_index():
    f = SmoothingFailureInfo()
    throw_or_store_smoothing_failure(f, SmoothingFailureReason.Unknown, "msg")
    assert f.failed_index == -1


def test_throw_or_store_without_failure_raises():
    with pytest.raises(FailedToSmoothPath) as exc_info:
        throw_or_store_smoothing_failure(None, SmoothingFailureReason.SolverRejectedSolution, "fail", 2)
    assert "solver_rejected_solution@2" in str(exc_info.value)


def test_smoothing_failure_info_defaults():
    f = SmoothingFailureInfo()
    assert f.reason == SmoothingFailureReason.Unknown
    assert f.message == ""
    assert f.failed_index == -1
    assert math.isnan(f.actual_curvature)
    assert math.isnan(f.max_curvature)
    assert math.isnan(f.turning_radius)


def test_smoothing_failure_info_formatted():
    f = SmoothingFailureInfo()
    f.reason = SmoothingFailureReason.NonFiniteState
    f.message = "bad"
    f.failed_index = 5
    assert f.formatted_message() == "nonfinite_state@5: bad"
