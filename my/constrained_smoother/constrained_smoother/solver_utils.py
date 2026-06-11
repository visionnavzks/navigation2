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
Solver utilities wrapping scipy.optimize.least_squares.

Mirrors the C++ solver_utils.hpp (Ceres wrapper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import least_squares

from constrained_smoother.exceptions import (
    SmoothingFailureInfo,
    SmoothingFailureReason,
    throw_or_store_smoothing_failure,
)


@dataclass
class SolverSummary:
    """Result from the solver, analogous to Ceres Solver::Summary."""

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
    """Solve a nonlinear least-squares problem using scipy.

    Parameters
    ----------
    residual_fn : callable
        Function that takes a flat parameter vector and returns a flat residual vector.
    x0 : ndarray
        Initial parameter vector.
    bounds : tuple of (lower, upper) ndarrays or None
        Parameter bounds.
    max_iterations : int
    max_time : float
        Not used (scipy doesn't have wall-clock timeout).
    function_tolerance, parameter_tolerance, gradient_tolerance : float
    debug : bool

    Returns
    -------
    x_opt : ndarray
        Optimized parameter vector.
    summary : SolverSummary
    """
    summary = SolverSummary()

    # Compute initial cost
    try:
        initial_residuals = residual_fn(x0)
        summary.initial_cost = float(np.sum(initial_residuals ** 2))
    except Exception as e:
        summary.message = f"Failed to evaluate initial cost: {e}"
        return x0, summary

    try:
        result = least_squares(
            residual_fn,
            x0,
            bounds=bounds if bounds else (-np.inf, np.inf),
            max_nfev=max_iterations * 10,
            ftol=function_tolerance,
            xtol=parameter_tolerance,
            gtol=gradient_tolerance,
            verbose=2 if debug else 0,
        )

        x_opt = result.x
        final_residuals = residual_fn(x_opt)
        summary.final_cost = float(np.sum(final_residuals ** 2))
        summary.num_iterations = result.nfev
        summary.is_solution_usable = result.success

        # Match C++ behavior: cost must not increase.
        # When initial cost is essentially zero, small numerical perturbations
        # from the optimizer can make final_cost slightly larger.  Treat the
        # solution as acceptable when the absolute change is negligible.
        cost_change = summary.final_cost - summary.initial_cost
        if summary.initial_cost < 1e-20:
            # Near-zero initial cost: accept if final cost is also small
            summary.cost_improved = summary.final_cost < 1e-6
        else:
            summary.cost_improved = cost_change <= 0.0

        summary.message = result.message

    except Exception as e:
        summary.message = f"Solver exception: {e}"
        return x0, summary

    return x_opt, summary


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
    smoother_name: str = "Kinematic smoother",
    failure: Optional[SmoothingFailureInfo] = None,
) -> tuple[bool, np.ndarray, Optional[SmoothingFailureInfo]]:
    """Solve and report failures using the unified failure dispatch.

    Returns
    -------
    success : bool
    x_opt : ndarray
    failure : SmoothingFailureInfo or None
    """
    x_opt, summary = solve_problem(
        residual_fn, x0, bounds, max_iterations, max_time,
        function_tolerance, parameter_tolerance, gradient_tolerance, debug,
    )

    if not summary.is_solution_usable:
        throw_or_store_smoothing_failure(
            failure,
            SmoothingFailureReason.SolverRejectedSolution,
            f"{smoother_name} rejected the solution as unusable",
        )
        return False, x_opt, failure

    if not summary.cost_improved:
        throw_or_store_smoothing_failure(
            failure,
            SmoothingFailureReason.NoCostImprovement,
            f"{smoother_name} did not improve the objective cost",
        )
        return False, x_opt, failure

    return True, x_opt, failure
