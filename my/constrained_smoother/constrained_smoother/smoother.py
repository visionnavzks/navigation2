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
Kinematic smoother - main entry point.

Mirrors the C++ kinematic_smoother.hpp.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from constrained_smoother.options import SmootherParams, OptimizerParams
from constrained_smoother.exceptions import (
    InvalidPath,
    InvalidCostmap,
    SmoothingFailureInfo,
)
from constrained_smoother.smoother_request import SmootherResult, SmootherRequest
from constrained_smoother.costmap2d import Costmap2D
from constrained_smoother.problem_builder import KinematicSmootherProblemBuilder
from constrained_smoother.validator import SmootherValidator
from constrained_smoother.solver_utils import solve_problem_or_report_failure


class KinematicSmoother:
    """Kinematic path smoother using nonlinear least-squares optimization.

    Each state is explicitly represented as (x, y, theta, kappa, ds) and
    constrained by kinematic transition residuals between consecutive states.
    """

    def __init__(self) -> None:
        self._debug: bool = False
        self._max_iterations: int = 50
        self._function_tolerance: float = 1e-6
        self._gradient_tolerance: float = 1e-10
        self._parameter_tolerance: float = 1e-8
        self._linear_solver: str = "SPARSE_NORMAL_CHOLESKY"
        self._esdf_values: list[float] = []
        self._validator = SmootherValidator()

    def initialize(self, params: OptimizerParams) -> None:
        """Initialize solver configuration from OptimizerParams."""
        self._debug = params.debug
        self._linear_solver = params.linear_solver.value
        self._max_iterations = params.max_iterations
        self._function_tolerance = params.function_tolerance
        self._gradient_tolerance = params.gradient_tolerance
        self._parameter_tolerance = params.parameter_tolerance

    def smooth(self, request: SmootherRequest) -> SmootherResult:
        """Smooth a path using the kinematic backend.

        Input path uses (x, y, direction_sign) in the third component.
        Output paths use (x, y, yaw).

        Raises InvalidPath, InvalidCostmap, or FailedToSmoothPath on failure
        (when request.failure is None).
        """
        # 1) Input validation
        if len(request.path) < 2:
            raise InvalidPath("Kinematic smoother: Path must have at least 2 points")
        if request.params.obstacle_terms_enabled() and request.costmap is None:
            raise InvalidCostmap("Kinematic smoother: Costmap must not be null")

        result = SmootherResult()

        # 2) Build and solve
        builder = KinematicSmootherProblemBuilder(self._esdf_values)
        builder.initialize_esdf_values(
            request.costmap, request.params, request.precomputed_esdf
        )

        processed = KinematicSmootherProblemBuilder.build_processed_path(
            request.path,
            request.start_dir,
            request.end_dir,
            request.params,
            request.costmap,
        )

        result.optimized_knot_count = processed.state_count
        result.target_spacing = processed.target_spacing

        # Build residual function
        residual_fn, num_params = builder.build_residual_fn(
            processed, request.costmap, request.params
        )

        x0 = np.array(processed.initial_variables, dtype=np.float64)

        # Apply bounds
        n = processed.state_count
        lower = np.full(num_params, -np.inf)
        upper = np.full(num_params, np.inf)

        KinematicSmootherProblemBuilder.apply_bounds(
            lower, upper,
            processed.reference_points,
            n,
            request.params.max_curvature,
            request.params.kinematic_max_spacing,
            request.params.reference_point_max_deviation_m,
        )

        bounds = (lower, upper)

        # Solve
        success, x_opt, _ = solve_problem_or_report_failure(
            residual_fn,
            x0,
            bounds=bounds,
            max_iterations=self._max_iterations,
            max_time=request.params.max_time,
            function_tolerance=self._function_tolerance,
            parameter_tolerance=self._parameter_tolerance,
            gradient_tolerance=self._gradient_tolerance,
            debug=self._debug,
            smoother_name="Kinematic smoother",
            failure=request.failure,
        )

        if not success:
            return result

        # 5) Unpack path
        result.candidate_path = KinematicSmootherProblemBuilder.unpack_path(
            x_opt, n
        )

        # 6) Post-validation
        accepted = self._validator.validate_kinematic_solution(
            x_opt,
            processed,
            request.costmap,
            request.params,
            self._esdf_values,
            request.failure,
        )

        if not accepted:
            return result

        # 7) Upsample
        result.smoothed_path = KinematicSmootherProblemBuilder.upsample_path_kinematic(
            x_opt, processed, request.params
        )
        result.success = True
        return result

    def smooth_with_planner_esdf(
        self,
        path: list[np.ndarray],
        start_dir: np.ndarray,
        end_dir: np.ndarray,
        costmap: Costmap2D,
        params: SmootherParams,
        planner_esdf: list[float],
    ) -> SmootherResult:
        """Smooth while reusing a precomputed ESDF from an A* planner."""
        request = SmootherRequest(
            path=path,
            start_dir=start_dir,
            end_dir=end_dir,
            costmap=costmap,
            params=params,
            precomputed_esdf=planner_esdf,
        )
        return self.smooth(request)

    # ---- Convenience wrappers matching pybind interface ----

    def try_smooth(
        self,
        path: list,
        start_dir: list,
        end_dir: list,
        costmap: Optional[Costmap2D] = None,
        params: Optional[SmootherParams] = None,
    ) -> dict:
        """Try to smooth and return a structured result dict (safe API).

        Returns
        -------
        dict with keys: ok, path, smoothed_path, candidate_path,
        optimized_knot_count, target_spacing_m, error_code, error_message,
        error_reason, error_details
        """
        if params is None:
            params = SmootherParams()

        request = SmootherRequest(
            path=[np.array(p, dtype=np.float64) for p in path],
            start_dir=np.array(start_dir, dtype=np.float64),
            end_dir=np.array(end_dir, dtype=np.float64),
            costmap=costmap,
            params=params,
        )

        failure = SmoothingFailureInfo()
        request.failure = failure

        try:
            result = self.smooth(request)
        except Exception as e:
            return {
                "ok": False,
                "path": None,
                "smoothed_path": None,
                "candidate_path": None,
                "optimized_knot_count": 0,
                "target_spacing_m": 0.0,
                "error_code": None,
                "error_message": str(e),
                "error_reason": None,
                "error_details": None,
            }

        if not result.success:
            from constrained_smoother.exceptions import (
                to_error_code_string,
                to_failure_reason_string,
                ErrorCode,
            )
            details = {}
            if failure.failed_index >= 0:
                details["failed_index"] = failure.failed_index
            if np.isfinite(failure.actual_curvature):
                details["actual_curvature"] = failure.actual_curvature
            if np.isfinite(failure.max_curvature):
                details["max_curvature"] = failure.max_curvature
            if np.isfinite(failure.turning_radius):
                details["turning_radius"] = failure.turning_radius
            if (np.isfinite(failure.actual_curvature) and
                    np.isfinite(failure.max_curvature)):
                details["curvature_excess"] = (
                    failure.actual_curvature - failure.max_curvature
                )
            if np.isfinite(failure.goal_longitudinal_error):
                details["goal_longitudinal_error"] = failure.goal_longitudinal_error
            if np.isfinite(failure.goal_lateral_error):
                details["goal_lateral_error"] = failure.goal_lateral_error
            if np.isfinite(failure.goal_longitudinal_tolerance):
                details["goal_longitudinal_tolerance"] = failure.goal_longitudinal_tolerance
            if np.isfinite(failure.goal_lateral_tolerance):
                details["goal_lateral_tolerance"] = failure.goal_lateral_tolerance

            return {
                "ok": False,
                "path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
                "smoothed_path": None,
                "candidate_path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
                "optimized_knot_count": result.optimized_knot_count,
                "target_spacing_m": result.target_spacing,
                "error_code": to_error_code_string(ErrorCode.FailedToSmoothPath),
                "error_message": failure.message,
                "error_reason": to_failure_reason_string(failure.reason),
                "error_details": details if details else None,
            }

        return {
            "ok": True,
            "path": [p.tolist() for p in result.smoothed_path],
            "smoothed_path": [p.tolist() for p in result.smoothed_path],
            "candidate_path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
            "optimized_knot_count": result.optimized_knot_count,
            "target_spacing_m": result.target_spacing,
            "error_code": None,
            "error_message": None,
            "error_reason": None,
            "error_details": None,
        }

    def try_smooth_with_planner_esdf(
        self,
        path: list,
        start_dir: list,
        end_dir: list,
        costmap: Costmap2D,
        params: SmootherParams,
        planner_esdf: list[float],
    ) -> dict:
        """Try to smooth with planner ESDF and return structured result dict."""
        request = SmootherRequest(
            path=[np.array(p, dtype=np.float64) for p in path],
            start_dir=np.array(start_dir, dtype=np.float64),
            end_dir=np.array(end_dir, dtype=np.float64),
            costmap=costmap,
            params=params,
            precomputed_esdf=planner_esdf,
        )

        failure = SmoothingFailureInfo()
        request.failure = failure

        try:
            result = self.smooth(request)
        except Exception as e:
            return {
                "ok": False,
                "path": None,
                "smoothed_path": None,
                "candidate_path": None,
                "optimized_knot_count": 0,
                "target_spacing_m": 0.0,
                "error_code": None,
                "error_message": str(e),
                "error_reason": None,
                "error_details": None,
            }

        if not result.success:
            from constrained_smoother.exceptions import (
                to_error_code_string,
                to_failure_reason_string,
                ErrorCode,
            )
            details = {}
            if failure.failed_index >= 0:
                details["failed_index"] = failure.failed_index
            if np.isfinite(failure.actual_curvature):
                details["actual_curvature"] = failure.actual_curvature
            if np.isfinite(failure.max_curvature):
                details["max_curvature"] = failure.max_curvature
            if np.isfinite(failure.turning_radius):
                details["turning_radius"] = failure.turning_radius
            if (np.isfinite(failure.actual_curvature) and
                    np.isfinite(failure.max_curvature)):
                details["curvature_excess"] = (
                    failure.actual_curvature - failure.max_curvature
                )
            if np.isfinite(failure.goal_longitudinal_error):
                details["goal_longitudinal_error"] = failure.goal_longitudinal_error
            if np.isfinite(failure.goal_lateral_error):
                details["goal_lateral_error"] = failure.goal_lateral_error
            if np.isfinite(failure.goal_longitudinal_tolerance):
                details["goal_longitudinal_tolerance"] = failure.goal_longitudinal_tolerance
            if np.isfinite(failure.goal_lateral_tolerance):
                details["goal_lateral_tolerance"] = failure.goal_lateral_tolerance

            return {
                "ok": False,
                "path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
                "smoothed_path": None,
                "candidate_path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
                "optimized_knot_count": result.optimized_knot_count,
                "target_spacing_m": result.target_spacing,
                "error_code": to_error_code_string(ErrorCode.FailedToSmoothPath),
                "error_message": failure.message,
                "error_reason": to_failure_reason_string(failure.reason),
                "error_details": details if details else None,
            }

        return {
            "ok": True,
            "path": [p.tolist() for p in result.smoothed_path],
            "smoothed_path": [p.tolist() for p in result.smoothed_path],
            "candidate_path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
            "optimized_knot_count": result.optimized_knot_count,
            "target_spacing_m": result.target_spacing,
            "error_code": None,
            "error_message": None,
            "error_reason": None,
            "error_details": None,
        }
