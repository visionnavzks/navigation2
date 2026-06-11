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
Post-solution hard validator for the kinematic smoother.

Mirrors the C++ smoother_validator.hpp.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from constrained_smoother.options import SmootherParams
from constrained_smoother.costmap2d import Costmap2D
from constrained_smoother.exceptions import (
    SmoothingFailureInfo,
    SmoothingFailureReason,
    throw_or_store_smoothing_failure,
)
from constrained_smoother.problem_builder import KinematicProcessedPath
from constrained_smoother.utils import (
    normalize_angle,
    angle_diff,
    world_to_grid,
    in_bounds,
    EPSILON,
    PI,
)


class SmootherValidator:
    """Post-solution hard validator.

    Separates "numerically converged" from "engineering deliverable".
    """

    @staticmethod
    def normalized_direction(dx: float, dy: float) -> tuple[float, float]:
        norm = math.hypot(dx, dy)
        if norm <= EPSILON:
            return 1.0, 0.0
        return dx / norm, dy / norm

    def validate_kinematic_solution(
        self,
        variables: np.ndarray,
        processed: KinematicProcessedPath,
        costmap: Optional[Costmap2D],
        params: SmootherParams,
        esdf_values: list[float],
        failure: Optional[SmoothingFailureInfo] = None,
    ) -> bool:
        """Run all post-validation checks on the kinematic solution.

        Returns True if the solution is accepted, False otherwise.
        """
        n = processed.state_count

        # 1) State vector shape check
        if len(variables) != n * 5:
            return throw_or_store_smoothing_failure(
                failure,
                SmoothingFailureReason.InvalidStateVector,
                "Kinematic smoother returned an invalid state vector size",
            )

        # 2) Finite states
        if not self._validate_finite_states(variables, n, failure):
            return False

        # 3) Boundary states
        if not self._validate_boundary_states(
            variables, processed, costmap, params, failure
        ):
            return False

        # 4) Segment consistency
        if not self._validate_segment_consistency(
            variables, processed, costmap, failure
        ):
            return False

        # 5) Curvature constraint
        if not self._validate_curvature_constraint(
            variables, processed, costmap, params, failure
        ):
            return False

        # 6) Obstacle clearance
        if not self._validate_obstacle_clearance(
            variables, processed, costmap, params, esdf_values, failure
        ):
            return False

        return True

    # ---- Private validation methods ----

    @staticmethod
    def _validate_finite_states(
        variables: np.ndarray,
        state_count: int,
        failure: Optional[SmoothingFailureInfo],
    ) -> bool:
        for index in range(state_count):
            state = variables[5 * index: 5 * index + 5]
            if not all(math.isfinite(v) for v in state):
                return throw_or_store_smoothing_failure(
                    failure,
                    SmoothingFailureReason.NonFiniteState,
                    f"Kinematic smoother returned a non-finite state at index {index}",
                    index,
                )
        return True

    @staticmethod
    def _position_tolerance(costmap: Optional[Costmap2D]) -> float:
        if costmap is not None:
            return max(costmap.resolution * 0.5, 1e-3)
        return 1e-3

    @staticmethod
    def _orientation_tolerance() -> float:
        return 0.1

    @staticmethod
    def _displacement_tolerance(costmap: Optional[Costmap2D]) -> float:
        if costmap is not None:
            return max(costmap.resolution * 0.25, 1e-4)
        return 1e-4

    def _validate_boundary_states(
        self,
        variables: np.ndarray,
        processed: KinematicProcessedPath,
        costmap: Optional[Costmap2D],
        params: SmootherParams,
        failure: Optional[SmoothingFailureInfo],
    ) -> bool:
        n = processed.state_count
        position_tol = self._position_tolerance(costmap)
        angle_tol = self._orientation_tolerance()

        # Start position check
        start_state = variables[0:5]
        start_dx = start_state[0] - processed.reference_points[0][0]
        start_dy = start_state[1] - processed.reference_points[0][1]
        if math.hypot(start_dx, start_dy) > position_tol:
            return throw_or_store_smoothing_failure(
                failure,
                SmoothingFailureReason.StartPositionConstraint,
                "Kinematic smoother violated the fixed start position constraint",
                0,
            )

        # Start orientation check
        if params.keep_start_orientation:
            if abs(angle_diff(start_state[2], processed.start_theta)) > angle_tol:
                return throw_or_store_smoothing_failure(
                    failure,
                    SmoothingFailureReason.StartOrientationConstraint,
                    "Kinematic smoother violated the fixed start orientation constraint",
                    0,
                )

        # Goal position check
        goal_state = variables[5 * (n - 1): 5 * (n - 1) + 5]
        goal_dx = goal_state[0] - processed.reference_points[-1][0]
        goal_dy = goal_state[1] - processed.reference_points[-1][1]

        from constrained_smoother.utils import goal_position_frame_heading
        goal_position_theta = goal_position_frame_heading(
            processed.reference_points,
            processed.end_theta,
            params.keep_goal_orientation,
        )

        cos_goal = math.cos(goal_position_theta)
        sin_goal = math.sin(goal_position_theta)
        goal_lon = cos_goal * goal_dx + sin_goal * goal_dy
        goal_lat = -sin_goal * goal_dx + cos_goal * goal_dy
        goal_lon_tol = max(params.goal_longitudinal_tolerance, position_tol)
        goal_lat_tol = max(params.goal_lateral_tolerance, position_tol)
        convergence_epsilon = 5e-4

        if (abs(goal_lon) > goal_lon_tol + convergence_epsilon or
                abs(goal_lat) > goal_lat_tol + convergence_epsilon):
            uses_goal_box = (
                params.goal_longitudinal_tolerance > 1e-9 or
                params.goal_lateral_tolerance > 1e-9
            )
            prefix = (
                "Kinematic smoother violated the goal position tolerance box"
                if uses_goal_box
                else "Kinematic smoother violated the fixed goal position constraint"
            )
            message = (
                f"{prefix}: lon error {goal_lon:.4f} m (tol {goal_lon_tol:.4f} m), "
                f"lat error {goal_lat:.4f} m (tol {goal_lat_tol:.4f} m)"
            )

            if failure is not None:
                failure.reason = SmoothingFailureReason.GoalPositionConstraint
                failure.message = message
                failure.failed_index = n - 1
                failure.goal_longitudinal_error = goal_lon
                failure.goal_lateral_error = goal_lat
                failure.goal_longitudinal_tolerance = goal_lon_tol
                failure.goal_lateral_tolerance = goal_lat_tol
                return False

            return throw_or_store_smoothing_failure(
                failure,
                SmoothingFailureReason.GoalPositionConstraint,
                message,
                n - 1,
            )

        # Goal orientation check
        if params.keep_goal_orientation:
            goal_heading_error = abs(angle_diff(goal_state[2], processed.end_theta))
            goal_heading_tol = max(params.goal_orientation_tolerance, angle_tol)
            if goal_heading_error > goal_heading_tol:
                return throw_or_store_smoothing_failure(
                    failure,
                    SmoothingFailureReason.GoalOrientationConstraint,
                    f"Kinematic smoother violated the fixed goal orientation constraint",
                    n - 1,
                )

        return True

    def _validate_segment_consistency(
        self,
        variables: np.ndarray,
        processed: KinematicProcessedPath,
        costmap: Optional[Costmap2D],
        failure: Optional[SmoothingFailureInfo],
    ) -> bool:
        n = processed.state_count
        position_tol = self._position_tolerance(costmap)
        displacement_tol = self._displacement_tolerance(costmap)
        angle_tol = self._orientation_tolerance()

        for index in range(n - 1):
            current = variables[5 * index: 5 * index + 5]
            next_state = variables[5 * (index + 1): 5 * (index + 1) + 5]
            dx = next_state[0] - current[0]
            dy = next_state[1] - current[1]
            displacement = math.hypot(dx, dy)

            if processed.is_cusp_segment[index]:
                if (displacement > position_tol or
                        abs(angle_diff(next_state[2], current[2])) > angle_tol):
                    return throw_or_store_smoothing_failure(
                        failure,
                        SmoothingFailureReason.CuspHoldConstraint,
                        "Kinematic smoother violated the cusp hold constraint during post-validation",
                        index,
                    )
                continue

            if displacement <= displacement_tol:
                return throw_or_store_smoothing_failure(
                    failure,
                    SmoothingFailureReason.CollapsedSegment,
                    "Kinematic smoother collapsed a non-cusp segment during post-validation",
                    index,
                )

            # Motion direction check
            heading_x = math.cos(current[2])
            heading_y = math.sin(current[2])
            signed_projection = dx * heading_x + dy * heading_y
            gear = processed.gears[index]
            if ((gear >= 0.0 and signed_projection <= 0.0) or
                    (gear < 0.0 and signed_projection >= 0.0)):
                return throw_or_store_smoothing_failure(
                    failure,
                    SmoothingFailureReason.MotionDirectionConstraint,
                    "Kinematic smoother returned a path whose motion direction violates the input gear and endpoint constraints",
                    index,
                )

        return True

    def _validate_curvature_constraint(
        self,
        variables: np.ndarray,
        processed: KinematicProcessedPath,
        costmap: Optional[Costmap2D],
        params: SmootherParams,
        failure: Optional[SmoothingFailureInfo],
    ) -> bool:
        n = processed.state_count
        max_curvature = max(params.max_curvature, 1e-6)
        curvature_tolerance = 1e-4
        displacement_tol = self._displacement_tolerance(costmap)

        def _report(index: int, actual_curvature: float) -> bool:
            turning_radius = (
                1.0 / actual_curvature
                if actual_curvature > 1e-9
                else float("inf")
            )
            message = (
                f"Kinematic smoother violated the maximum curvature constraint during post-validation"
                f": actual curvature {actual_curvature:.4f} 1/m"
                f", limit {max_curvature:.4f} 1/m"
                f", excess {actual_curvature - max_curvature:.4f} 1/m"
                f", turning radius {turning_radius:.3f} m"
            )
            if failure is not None:
                failure.reason = SmoothingFailureReason.CurvatureConstraint
                failure.message = message
                failure.failed_index = index
                failure.actual_curvature = actual_curvature
                failure.max_curvature = max_curvature
                failure.turning_radius = turning_radius
                return False

            return throw_or_store_smoothing_failure(
                failure,
                SmoothingFailureReason.CurvatureConstraint,
                message,
                index,
            )

        # 1) Check explicit kappa
        for index in range(n):
            state = variables[5 * index: 5 * index + 5]
            abs_kappa = abs(state[3])
            if abs_kappa > max_curvature + curvature_tolerance:
                if not _report(index, abs_kappa):
                    return False

        # 2) Check geometric curvature from adjacent poses
        for index in range(n - 1):
            if processed.is_cusp_segment[index]:
                continue
            current = variables[5 * index: 5 * index + 5]
            next_state = variables[5 * (index + 1): 5 * (index + 1) + 5]
            dx = next_state[0] - current[0]
            dy = next_state[1] - current[1]
            displacement = math.hypot(dx, dy)
            if displacement <= displacement_tol:
                continue
            delta_theta = angle_diff(next_state[2], current[2])
            geometric_curvature = abs(delta_theta) / displacement
            if geometric_curvature > max_curvature + curvature_tolerance:
                if not _report(index, geometric_curvature):
                    return False

        return True

    @staticmethod
    def _validate_obstacle_clearance(
        variables: np.ndarray,
        processed: KinematicProcessedPath,
        costmap: Optional[Costmap2D],
        params: SmootherParams,
        esdf_values: list[float],
        failure: Optional[SmoothingFailureInfo],
    ) -> bool:
        if not params.obstacle_terms_enabled() or costmap is None:
            return True

        radius = max(params.cost_check_radius, 0.0)
        if radius <= 1e-9 and not params.cost_check_points:
            return True

        for state_index in range(processed.state_count):
            state = variables[5 * state_index: 5 * state_index + 5]
            x, y, theta = state[0], state[1], state[2]
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            def _validate_checkpoint(local_x: float, local_y: float) -> bool:
                world_x = x + cos_theta * local_x - sin_theta * local_y
                world_y = y + sin_theta * local_x + cos_theta * local_y
                mx, my = world_to_grid(costmap, world_x, world_y)
                if not in_bounds(mx, my, costmap.size_x, costmap.size_y):
                    return throw_or_store_smoothing_failure(
                        failure,
                        SmoothingFailureReason.PathOutOfBounds,
                        "Kinematic smoother returned a path that leaves the map bounds during footprint validation",
                        state_index,
                    )
                flat_index = my * costmap.size_x + mx
                if flat_index < 0 or flat_index >= len(esdf_values):
                    return throw_or_store_smoothing_failure(
                        failure,
                        SmoothingFailureReason.PathOutOfBounds,
                        "Kinematic smoother returned a path that leaves the map bounds during footprint validation",
                        state_index,
                    )
                clearance = esdf_values[flat_index]
                if clearance < radius:
                    return throw_or_store_smoothing_failure(
                        failure,
                        SmoothingFailureReason.FootprintCollision,
                        "Kinematic smoother returned a path that collides with obstacles during footprint validation",
                        state_index,
                    )
                return True

            if not params.cost_check_points:
                if not _validate_checkpoint(0.0, 0.0):
                    return False
                continue

            for offset in range(0, len(params.cost_check_points) - 2, 3):
                if not _validate_checkpoint(
                    params.cost_check_points[offset],
                    params.cost_check_points[offset + 1],
                ):
                    return False

        return True
