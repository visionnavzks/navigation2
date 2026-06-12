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
Runtime configuration structs for the constrained smoother.

Mirrors the C++ options.hpp with SmootherParams and OptimizerParams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SmootherParams:
    """Runtime configuration for the kinematic smoother.

    Most weights are in square-root form: they multiply the residual directly,
    and Ceres squares them in the objective function.
    """

    # --- Kinematic and reference-path weights ---

    model_weight_sqrt: float = 0.0
    costmap_weight_sqrt: float = 0.0
    reference_path_weight_sqrt: float = 0.0
    reference_point_max_deviation_m: float = 0.0
    kinematic_curvature_weight_sqrt: float = 0.0
    kinematic_curvature_rate_weight_sqrt: float = 0.0
    kinematic_spacing_weight_sqrt: float = 1.0
    kinematic_max_spacing: float = 0.0
    path_length_weight_sqrt: float = 0.0
    fix_weight: float = 100.0
    max_curvature: float = 0.0
    max_time: float = 10.0

    # --- Obstacle and footprint handling ---

    use_exact_esdf: bool = True
    obstacle_safe_distance: float = 0.5
    cost_check_radius: float = 0.0
    cost_check_points: list[float] = field(default_factory=list)

    # --- Path resampling and direction semantics ---

    path_target_spacing: float = 0.0
    path_downsampling_factor: int = 1
    path_upsampling_factor: int = 1
    reversing_enabled: bool = True

    # --- Goal and boundary handling ---

    goal_longitudinal_tolerance: float = 0.0
    goal_lateral_tolerance: float = 0.0
    goal_orientation_tolerance: float = 0.0
    keep_goal_orientation: bool = True
    keep_start_orientation: bool = True

    def obstacle_terms_enabled(self) -> bool:
        """Return whether any obstacle residual is actually enabled."""
        return self.costmap_weight_sqrt > 1e-9


class LinearSolver(Enum):
    """Linear solver selection for Ceres."""

    DenseQr = "DENSE_QR"
    SparseNormalCholesky = "SPARSE_NORMAL_CHOLESKY"


@dataclass
class OptimizerParams:
    """Ceres solver-level configuration."""

    debug: bool = False
    linear_solver: LinearSolver = LinearSolver.SparseNormalCholesky
    max_iterations: int = 50
    parameter_tolerance: float = 1e-8
    function_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-10
