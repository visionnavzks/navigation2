"""Solver utilities wrapping scipy.optimize.least_squares."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import least_squares

from smoother_clothoid_py.exceptions import (
    SmoothingFailureInfo, SmoothingFailureReason, throw_or_store_smoothing_failure,
)


@dataclass
class SolverSummary:
    is_solution_usable: bool = False
    initial_cost: float = float("inf")
    final_cost: float = float("inf")
    cost_improved: bool = False
    num_iterations: int = 0
    message: str = ""


def solve_problem(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
    max_iterations: int = 50,
    max_time: float = 10.0,
    function_tolerance: float = 1e-6,
    parameter_tolerance: float = 1e-8,
    gradient_tolerance: float = 1e-10,
    debug: bool = False,
) -> tuple[np.ndarray, SolverSummary]:
    s = SolverSummary()
    try:
        s.initial_cost = float(np.sum(residual_fn(x0) ** 2))
    except Exception as e:
        s.message = f"Initial cost evaluation failed: {e}"
        return x0, s

    try:
        result = least_squares(
            residual_fn, x0,
            bounds=bounds if bounds else (-np.inf, np.inf),
            max_nfev=max_iterations * 10,
            ftol=function_tolerance, xtol=parameter_tolerance,
            gtol=gradient_tolerance, verbose=2 if debug else 0,
        )
        x_opt = result.x
        s.final_cost = float(np.sum(residual_fn(x_opt) ** 2))
        s.num_iterations = result.nfev
        s.is_solution_usable = result.success
        if s.initial_cost < 1e-20:
            s.cost_improved = s.final_cost < 1e-6
        else:
            s.cost_improved = (s.final_cost - s.initial_cost) <= 0.0
        s.message = result.message
    except Exception as e:
        s.message = f"Solver exception: {e}"
        return x0, s

    return x_opt, s


def solve_problem_or_report_failure(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
    max_iterations: int = 50,
    max_time: float = 10.0,
    function_tolerance: float = 1e-6,
    parameter_tolerance: float = 1e-8,
    gradient_tolerance: float = 1e-10,
    debug: bool = False,
    smoother_name: str = "Clothoid smoother",
    failure: Optional[SmoothingFailureInfo] = None,
) -> tuple[bool, np.ndarray, Optional[SmoothingFailureInfo]]:
    x_opt, s = solve_problem(residual_fn, x0, bounds, max_iterations, max_time,
        function_tolerance, parameter_tolerance, gradient_tolerance, debug)
    if not s.is_solution_usable:
        throw_or_store_smoothing_failure(failure, SmoothingFailureReason.SolverRejectedSolution,
            f"{smoother_name} rejected the solution as unusable")
        return False, x_opt, failure
    if not s.cost_improved:
        throw_or_store_smoothing_failure(failure, SmoothingFailureReason.NoCostImprovement,
            f"{smoother_name} did not improve the objective cost")
        return False, x_opt, failure
    return True, x_opt, failure
