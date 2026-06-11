"""ClothoidSmoother - main entry point."""

from __future__ import annotations
import numpy as np
from typing import Optional

from smoother_clothoid_py.options import SmootherParams, OptimizerParams
from smoother_clothoid_py.exceptions import InvalidPath, InvalidCostmap, SmoothingFailureInfo
from smoother_clothoid_py.smoother_request import SmootherResult, SmootherRequest
from smoother_clothoid_py.costmap2d import Costmap2D
from smoother_clothoid_py.problem_builder import ProblemBuilder, ProcessedPath
from smoother_clothoid_py.solver_utils import solve_problem_or_report_failure


class _Validator:
    """Minimal post-solution validator."""

    @staticmethod
    def _pos_tol(c): return max(c.resolution * 0.5, 1e-3) if c else 1e-3

    def validate(self, variables: np.ndarray, proc: ProcessedPath,
                 costmap: Optional[Costmap2D], params: SmootherParams,
                 esdf: list[float], failure: Optional[SmoothingFailureInfo]) -> bool:
        n = proc.state_count
        if len(variables) != n * 5:
            from smoother_clothoid_py.exceptions import throw_or_store_smoothing_failure, SmoothingFailureReason
            return throw_or_store_smoothing_failure(failure, SmoothingFailureReason.InvalidStateVector, "Invalid state vector size")

        import math
        from smoother_clothoid_py.utils import angle_diff, normalize_angle

        tol = self._pos_tol(costmap)
        atol = 0.1

        # Finite check
        for i in range(n):
            if not all(math.isfinite(v) for v in variables[5*i:5*i+5]):
                from smoother_clothoid_py.exceptions import throw_or_store_smoothing_failure, SmoothingFailureReason
                return throw_or_store_smoothing_failure(failure, SmoothingFailureReason.NonFiniteState,
                    f"Non-finite state at {i}", i)

        # Start boundary
        s0 = variables[0:5]
        if math.hypot(s0[0]-proc.reference_points[0][0], s0[1]-proc.reference_points[0][1]) > tol:
            from smoother_clothoid_py.exceptions import throw_or_store_smoothing_failure, SmoothingFailureReason
            return throw_or_store_smoothing_failure(failure, SmoothingFailureReason.StartPositionConstraint, "Start pos", 0)
        if params.keep_start_orientation and abs(angle_diff(s0[2], proc.start_theta)) > atol:
            from smoother_clothoid_py.exceptions import throw_or_store_smoothing_failure, SmoothingFailureReason
            return throw_or_store_smoothing_failure(failure, SmoothingFailureReason.StartOrientationConstraint, "Start ori", 0)

        # Goal boundary
        sg = variables[5*(n-1):5*(n-1)+5]
        from smoother_clothoid_py.utils import goal_position_frame_heading
        gth = goal_position_frame_heading(proc.reference_points, proc.end_theta, params.keep_goal_orientation)
        dx, dy = sg[0]-proc.reference_points[-1][0], sg[1]-proc.reference_points[-1][1]
        cg, sg_ = math.cos(gth), math.sin(gth)
        lon, lat = cg*dx+sg_*dy, -sg_*dx+cg*dy
        lt = max(params.goal_longitudinal_tolerance, tol)
        bt = max(params.goal_lateral_tolerance, tol)
        if abs(lon) > lt + 5e-4 or abs(lat) > bt + 5e-4:
            from smoother_clothoid_py.exceptions import throw_or_store_smoothing_failure, SmoothingFailureReason
            if failure:
                failure.reason = SmoothingFailureReason.GoalPositionConstraint
                failure.failed_index = n-1
                failure.goal_longitudinal_error = lon
                failure.goal_lateral_error = lat
                failure.goal_longitudinal_tolerance = lt
                failure.goal_lateral_tolerance = bt
                failure.message = f"Goal pos: lon={lon} lat={lat}"
                return False
            return throw_or_store_smoothing_failure(failure, SmoothingFailureReason.GoalPositionConstraint, "Goal pos", n-1)
        if params.keep_goal_orientation and abs(angle_diff(sg[2], proc.end_theta)) > max(params.goal_orientation_tolerance, atol):
            from smoother_clothoid_py.exceptions import throw_or_store_smoothing_failure, SmoothingFailureReason
            return throw_or_store_smoothing_failure(failure, SmoothingFailureReason.GoalOrientationConstraint, "Goal ori", n-1)

        # Curvature
        mc = max(params.max_curvature, 1e-6)
        for i in range(n):
            ak = abs(variables[5*i+3])
            if ak > mc + 1e-4:
                from smoother_clothoid_py.exceptions import throw_or_store_smoothing_failure, SmoothingFailureReason
                if failure:
                    failure.reason = SmoothingFailureReason.CurvatureConstraint
                    failure.actual_curvature = ak; failure.max_curvature = mc
                    failure.turning_radius = 1.0/ak if ak > 1e-9 else float("inf")
                    failure.failed_index = i; failure.message = f"kappa={ak}>{mc}"
                    return False
                return throw_or_store_smoothing_failure(failure, SmoothingFailureReason.CurvatureConstraint, "Curvature", i)

        return True


class ClothoidSmoother:
    def __init__(self) -> None:
        self._debug = False
        self._max_iter = 50
        self._ftol = 1e-6
        self._gtol = 1e-10
        self._ptol = 1e-8
        self._esdf: list[float] = []
        self._validator = _Validator()

    def initialize(self, params: OptimizerParams) -> None:
        self._debug = params.debug
        self._max_iter = params.max_iterations
        self._ftol = params.function_tolerance
        self._gtol = params.gradient_tolerance
        self._ptol = params.parameter_tolerance

    def smooth(self, request: SmootherRequest) -> SmootherResult:
        if len(request.path) < 2:
            raise InvalidPath("Clothoid smoother: Path must have at least 2 points")
        if request.params.obstacle_terms_enabled() and request.costmap is None:
            raise InvalidCostmap("Clothoid smoother: Costmap must not be null")

        result = SmootherResult()
        builder = ProblemBuilder(self._esdf)
        builder.initialize_esdf_values(request.costmap, request.params, request.precomputed_esdf)

        proc = ProblemBuilder.build_processed_path(
            request.path, request.start_dir, request.end_dir, request.params, request.costmap)
        result.optimized_knot_count = proc.state_count
        result.target_spacing = proc.target_spacing

        fn, nparams = builder.build_residual_fn(proc, request.costmap, request.params)
        x0 = np.array(proc.initial_variables, dtype=np.float64)
        lower = np.full(nparams, -np.inf)
        upper = np.full(nparams, np.inf)
        ProblemBuilder.apply_bounds(lower, upper, proc.reference_points, proc.state_count,
            request.params.max_curvature, request.params.kinematic_max_spacing,
            request.params.reference_point_max_deviation_m)

        ok, x_opt, _ = solve_problem_or_report_failure(
            fn, x0, bounds=(lower, upper), max_iterations=self._max_iter,
            max_time=request.params.max_time, function_tolerance=self._ftol,
            parameter_tolerance=self._ptol, gradient_tolerance=self._gtol,
            debug=self._debug, smoother_name="Clothoid smoother", failure=request.failure)

        if not ok: return result

        result.candidate_path = ProblemBuilder.unpack_path(x_opt, proc.state_count)

        if not self._validator.validate(x_opt, proc, request.costmap, request.params, self._esdf, request.failure):
            return result

        result.smoothed_path = ProblemBuilder.upsample_path(x_opt, proc, request.params)
        result.success = True
        return result

    def try_smooth(self, path, start_dir, end_dir, costmap=None, params=None) -> dict:
        if params is None: params = SmootherParams()
        request = SmootherRequest(
            path=[np.array(p, dtype=np.float64) for p in path],
            start_dir=np.array(start_dir, dtype=np.float64),
            end_dir=np.array(end_dir, dtype=np.float64),
            costmap=costmap, params=params)
        failure = SmoothingFailureInfo()
        request.failure = failure
        try:
            result = self.smooth(request)
        except Exception as e:
            return {"ok": False, "path": None, "smoothed_path": None, "candidate_path": None,
                    "optimized_knot_count": 0, "target_spacing_m": 0.0,
                    "error_code": None, "error_message": str(e), "error_reason": None, "error_details": None}
        if not result.success:
            from smoother_clothoid_py.exceptions import to_error_code_string, to_failure_reason_string, ErrorCode
            det = {}
            if failure.failed_index >= 0: det["failed_index"] = failure.failed_index
            if np.isfinite(failure.actual_curvature): det["actual_curvature"] = failure.actual_curvature
            return {"ok": False,
                    "path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
                    "smoothed_path": None,
                    "candidate_path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
                    "optimized_knot_count": result.optimized_knot_count,
                    "target_spacing_m": result.target_spacing,
                    "error_code": to_error_code_string(ErrorCode.FailedToSmoothPath),
                    "error_message": failure.message,
                    "error_reason": to_failure_reason_string(failure.reason),
                    "error_details": det or None}
        return {"ok": True,
                "path": [p.tolist() for p in result.smoothed_path],
                "smoothed_path": [p.tolist() for p in result.smoothed_path],
                "candidate_path": [p.tolist() for p in result.candidate_path] if result.candidate_path else None,
                "optimized_knot_count": result.optimized_knot_count,
                "target_spacing_m": result.target_spacing,
                "error_code": None, "error_message": None, "error_reason": None, "error_details": None}
