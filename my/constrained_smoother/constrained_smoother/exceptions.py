# Copyright (c) 2021 RoboTech Vision
# Copyright (c) 2020, Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stable error codes, failure reason enums, and exception classes.

Mirrors the C++ exceptions.hpp.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Optional


class ErrorCode(IntEnum):
    """Stable public error codes."""

    InvalidPath = 1001
    FailedToSmoothPath = 2001
    InvalidCostmap = 3001
    PrecomputedEsdfSizeMismatch = 3002


class SmoothingFailureReason(IntEnum):
    """Stable failure reason enum for post-validation failures."""

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


# ---- String conversion tables ----

_ERROR_CODE_STRINGS = {
    ErrorCode.InvalidPath: "CS_INVALID_PATH",
    ErrorCode.FailedToSmoothPath: "CS_SMOOTHING_FAILED",
    ErrorCode.InvalidCostmap: "CS_INVALID_COSTMAP",
    ErrorCode.PrecomputedEsdfSizeMismatch: "CS_PRECOMPUTED_ESDF_SIZE_MISMATCH",
}

_FAILURE_REASON_STRINGS = {
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
    """Convert an ErrorCode to its stable string representation."""
    return _ERROR_CODE_STRINGS.get(code, "CS_UNKNOWN_ERROR")


def to_failure_reason_string(reason: SmoothingFailureReason) -> str:
    """Convert a SmoothingFailureReason to its stable string representation."""
    return _FAILURE_REASON_STRINGS.get(reason, "unknown")


def build_smoothing_failure_message(
    reason: SmoothingFailureReason,
    message: str,
    failed_index: int = -1,
) -> str:
    """Format a structured failure message."""
    formatted = to_failure_reason_string(reason)
    if failed_index >= 0:
        formatted += f"@{failed_index}"
    formatted += f": {message}"
    return formatted


class SmoothingFailureInfo:
    """Structured failure information for non-exception error propagation."""

    __slots__ = (
        "reason",
        "message",
        "failed_index",
        "actual_curvature",
        "max_curvature",
        "turning_radius",
        "goal_longitudinal_error",
        "goal_lateral_error",
        "goal_longitudinal_tolerance",
        "goal_lateral_tolerance",
    )

    def __init__(self) -> None:
        self.reason: SmoothingFailureReason = SmoothingFailureReason.Unknown
        self.message: str = ""
        self.failed_index: int = -1
        self.actual_curvature: float = math.nan
        self.max_curvature: float = math.nan
        self.turning_radius: float = math.nan
        self.goal_longitudinal_error: float = math.nan
        self.goal_lateral_error: float = math.nan
        self.goal_longitudinal_tolerance: float = math.nan
        self.goal_lateral_tolerance: float = math.nan

    def formatted_message(self) -> str:
        """Return the formatted failure message."""
        return build_smoothing_failure_message(
            self.reason, self.message, self.failed_index
        )


# ---- Exceptions ----


class InvalidPath(Exception):
    """Thrown when the input path is invalid (e.g. too short)."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self._code = ErrorCode.InvalidPath

    @property
    def code(self) -> ErrorCode:
        return self._code

    @property
    def code_string(self) -> str:
        return to_error_code_string(self._code)


class InvalidCostmap(Exception):
    """Thrown when the costmap is invalid or missing."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self._code = ErrorCode.InvalidCostmap

    @property
    def code(self) -> ErrorCode:
        return self._code

    @property
    def code_string(self) -> str:
        return to_error_code_string(self._code)


class PrecomputedEsdfSizeMismatch(Exception):
    """Thrown when the precomputed ESDF size does not match the costmap."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self._code = ErrorCode.PrecomputedEsdfSizeMismatch

    @property
    def code(self) -> ErrorCode:
        return self._code

    @property
    def code_string(self) -> str:
        return to_error_code_string(self._code)


class FailedToSmoothPath(Exception):
    """Thrown when the optimizer fails to produce a usable solution."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self._code = ErrorCode.FailedToSmoothPath

    @property
    def code(self) -> ErrorCode:
        return self._code

    @property
    def code_string(self) -> str:
        return to_error_code_string(self._code)


def throw_or_store_smoothing_failure(
    failure: Optional[SmoothingFailureInfo],
    reason: SmoothingFailureReason,
    message: str,
    failed_index: int = -1,
) -> bool:
    """Unified dispatch between exception and structured failure paths.

    - If failure is not None, writes the failure info and returns False.
    - If failure is None, raises FailedToSmoothPath.
    """
    if failure is not None:
        failure.reason = reason
        failure.message = message
        failure.failed_index = failed_index
        return False

    raise FailedToSmoothPath(
        build_smoothing_failure_message(reason, message, failed_index)
    )
