"""Error codes, failure reasons, and exceptions."""

from __future__ import annotations
import math
from enum import IntEnum
from typing import Optional


class ErrorCode(IntEnum):
    InvalidPath = 1001
    FailedToSmoothPath = 2001
    InvalidCostmap = 3001
    PrecomputedEsdfSizeMismatch = 3002


class SmoothingFailureReason(IntEnum):
    Unknown = 0
    SolverRejectedSolution = 1
    NoCostImprovement = 2
    InvalidStateVector = 3
    NonFiniteState = 4
    StartPositionConstraint = 5
    StartOrientationConstraint = 6
    GoalPositionConstraint = 7
    GoalOrientationConstraint = 8
    CuspHoldConstraint = 9
    CollapsedSegment = 10
    MotionDirectionConstraint = 11
    PathOutOfBounds = 12
    FootprintCollision = 13
    CurvatureConstraint = 14


_ERROR_STRINGS = {
    ErrorCode.InvalidPath: "SC_INVALID_PATH",
    ErrorCode.FailedToSmoothPath: "SC_SMOOTHING_FAILED",
    ErrorCode.InvalidCostmap: "SC_INVALID_COSTMAP",
    ErrorCode.PrecomputedEsdfSizeMismatch: "SC_PRECOMPUTED_ESDF_SIZE_MISMATCH",
}

_REASON_STRINGS = {
    SmoothingFailureReason.SolverRejectedSolution: "solver_rejected_solution",
    SmoothingFailureReason.NoCostImprovement: "no_cost_improvement",
    SmoothingFailureReason.InvalidStateVector: "invalid_state_vector",
    SmoothingFailureReason.NonFiniteState: "nonfinite_state",
    SmoothingFailureReason.StartPositionConstraint: "start_position_constraint",
    SmoothingFailureReason.StartOrientationConstraint: "start_orientation_constraint",
    SmoothingFailureReason.GoalPositionConstraint: "goal_position_constraint",
    SmoothingFailureReason.GoalOrientationConstraint: "goal_orientation_constraint",
    SmoothingFailureReason.CuspHoldConstraint: "cusp_hold_constraint",
    SmoothingFailureReason.CollapsedSegment: "collapsed_segment",
    SmoothingFailureReason.MotionDirectionConstraint: "motion_direction_constraint",
    SmoothingFailureReason.PathOutOfBounds: "path_out_of_bounds",
    SmoothingFailureReason.FootprintCollision: "footprint_collision",
    SmoothingFailureReason.CurvatureConstraint: "curvature_constraint",
    SmoothingFailureReason.Unknown: "unknown",
}


def to_error_code_string(code: ErrorCode) -> str:
    return _ERROR_STRINGS.get(code, "SC_UNKNOWN_ERROR")


def to_failure_reason_string(reason: SmoothingFailureReason) -> str:
    return _REASON_STRINGS.get(reason, "unknown")


def build_smoothing_failure_message(reason: SmoothingFailureReason, message: str, failed_index: int = -1) -> str:
    s = to_failure_reason_string(reason)
    if failed_index >= 0:
        s += f"@{failed_index}"
    return f"{s}: {message}"


class SmoothingFailureInfo:
    __slots__ = (
        "reason", "message", "failed_index",
        "actual_curvature", "max_curvature", "turning_radius",
        "goal_longitudinal_error", "goal_lateral_error",
        "goal_longitudinal_tolerance", "goal_lateral_tolerance",
    )

    def __init__(self) -> None:
        self.reason = SmoothingFailureReason.Unknown
        self.message = ""
        self.failed_index = -1
        self.actual_curvature = math.nan
        self.max_curvature = math.nan
        self.turning_radius = math.nan
        self.goal_longitudinal_error = math.nan
        self.goal_lateral_error = math.nan
        self.goal_longitudinal_tolerance = math.nan
        self.goal_lateral_tolerance = math.nan

    def formatted_message(self) -> str:
        return build_smoothing_failure_message(self.reason, self.message, self.failed_index)


class InvalidPath(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self._code = ErrorCode.InvalidPath
    @property
    def code(self) -> ErrorCode: return self._code
    @property
    def code_string(self) -> str: return to_error_code_string(self._code)


class InvalidCostmap(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self._code = ErrorCode.InvalidCostmap
    @property
    def code(self) -> ErrorCode: return self._code
    @property
    def code_string(self) -> str: return to_error_code_string(self._code)


class PrecomputedEsdfSizeMismatch(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self._code = ErrorCode.PrecomputedEsdfSizeMismatch
    @property
    def code(self) -> ErrorCode: return self._code
    @property
    def code_string(self) -> str: return to_error_code_string(self._code)


class FailedToSmoothPath(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self._code = ErrorCode.FailedToSmoothPath
    @property
    def code(self) -> ErrorCode: return self._code
    @property
    def code_string(self) -> str: return to_error_code_string(self._code)


def throw_or_store_smoothing_failure(
    failure: Optional[SmoothingFailureInfo],
    reason: SmoothingFailureReason,
    message: str,
    failed_index: int = -1,
) -> bool:
    if failure is not None:
        failure.reason = reason
        failure.message = message
        failure.failed_index = failed_index
        return False
    raise FailedToSmoothPath(build_smoothing_failure_message(reason, message, failed_index))
