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
Kinematic smoother problem builder.

Mirrors the C++ kinematic_smoother_problem_builder.hpp.
Handles state expansion, initial variable generation, residual assembly,
explicit bounds, and result unpacking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from constrained_smoother.options import SmootherParams
from constrained_smoother.costmap2d import Costmap2D
from constrained_smoother.exceptions import PrecomputedEsdfSizeMismatch
from constrained_smoother.esdf import compute_esdf
from constrained_smoother.costs import (
    transition_residuals,
    boundary_residuals,
    reference_residuals,
    obstacle_residuals,
)
from constrained_smoother.utils import (
    normalize_angle,
    angle_diff,
    world_to_grid,
    in_bounds,
    goal_position_frame_heading,
    EPSILON,
    PI,
)


@dataclass
class KinematicProcessedPath:
    """Expanded state chain with cusp metadata."""

    reference_points: list[tuple[float, float]] = field(default_factory=list)
    gears: list[float] = field(default_factory=list)
    is_cusp_segment: list[bool] = field(default_factory=list)
    initial_variables: list[float] = field(default_factory=list)
    state_count: int = 0
    start_theta: float = 0.0
    end_theta: float = 0.0
    target_spacing: float = 0.2


class KinematicSmootherProblemBuilder:
    """Problem builder for the kinematic smoother."""

    def __init__(self, esdf_values: list[float]) -> None:
        self._esdf_values: list[float] = esdf_values
        self._costmap: Optional[Costmap2D] = None
        self._params: Optional[SmootherParams] = None

    def initialize_esdf_values(
        self,
        costmap: Optional[Costmap2D],
        params: SmootherParams,
        precomputed_esdf: Optional[list[float]] = None,
    ) -> None:
        """Prepare ESDF values for obstacle residuals and post-validation."""
        if not params.obstacle_terms_enabled():
            self._esdf_values.clear()
            self._costmap = None
            self._params = params
            return

        if costmap is None:
            self._costmap = None
            self._params = params
            return

        expected_size = costmap.size_x * costmap.size_y
        if precomputed_esdf is not None:
            if len(precomputed_esdf) != expected_size:
                raise PrecomputedEsdfSizeMismatch(
                    "Precomputed ESDF size does not match costmap dimensions"
                )
            self._esdf_values = list(precomputed_esdf)
        else:
            self._esdf_values = compute_esdf(
                costmap,
                Costmap2D.LETHAL_OBSTACLE,
            )

        self._costmap = costmap
        self._params = params

    @staticmethod
    def build_processed_path(
        path: list[np.ndarray],
        start_dir: np.ndarray,
        end_dir: np.ndarray,
        params: SmootherParams,
        costmap: Optional[Costmap2D] = None,
    ) -> KinematicProcessedPath:
        """Expand the input path into a kinematic state chain.

        Handles downsampling, gear assignment, cusp insertion, and initial
        variable generation.
        """
        processed = KinematicProcessedPath()
        processed.start_theta = math.atan2(start_dir[1], start_dir[0])
        processed.end_theta = math.atan2(end_dir[1], end_dir[0])

        sampled_path = _downsample_input_path(path, params)

        # Compute gear directions
        gear_directions: list[float] = []
        for index in range(len(sampled_path) - 1):
            if params.reversing_enabled:
                gear_directions.append(-1.0 if sampled_path[index][2] < 0.0 else 1.0)
            else:
                gear_directions.append(1.0)

        # Expand states with cusp insertion
        processed.reference_points.append(
            (sampled_path[0][0], sampled_path[0][1])
        )
        for index in range(len(sampled_path) - 1):
            current_gear = gear_directions[index]
            next_gear = (
                gear_directions[index + 1]
                if index + 1 < len(gear_directions)
                else current_gear
            )

            processed.gears.append(current_gear)
            processed.is_cusp_segment.append(False)
            processed.reference_points.append(
                (sampled_path[index + 1][0], sampled_path[index + 1][1])
            )

            # Insert cusp state if gear changes
            if index + 2 < len(sampled_path) and current_gear != next_gear:
                processed.gears.append(0.0)
                processed.is_cusp_segment.append(True)
                processed.reference_points.append(
                    (sampled_path[index + 1][0], sampled_path[index + 1][1])
                )

        processed.state_count = len(processed.reference_points)

        # Initialize theta, kappa, ds
        theta = [0.0] * processed.state_count
        kappa = [0.0] * processed.state_count
        ds = [0.0] * processed.state_count

        spacing_sum = 0.0
        spacing_count = 0
        for index in range(processed.state_count - 1):
            rx = processed.reference_points[index + 1][0] - processed.reference_points[index][0]
            ry = processed.reference_points[index + 1][1] - processed.reference_points[index][1]
            segment_norm = math.hypot(rx, ry)

            if processed.is_cusp_segment[index]:
                theta[index] = theta[index - 1] if index > 0 else processed.start_theta
                ds[index] = 0.0
                continue

            if segment_norm > 1e-6:
                heading = math.atan2(ry, rx)
                if processed.gears[index] < 0.0:
                    heading += PI
                theta[index] = normalize_angle(heading)
                ds[index] = segment_norm
                spacing_sum += segment_norm
                spacing_count += 1
            else:
                theta[index] = theta[index - 1] if index > 0 else processed.start_theta

        # Last theta
        theta[-1] = theta[-2] if len(theta) > 1 else processed.start_theta
        if params.keep_start_orientation:
            theta[0] = processed.start_theta
        if params.keep_goal_orientation:
            theta[-1] = processed.end_theta

        # Target spacing
        if params.path_target_spacing > 1e-9:
            processed.target_spacing = params.path_target_spacing
        elif spacing_count > 0:
            processed.target_spacing = spacing_sum / spacing_count
        elif costmap is not None:
            processed.target_spacing = max(costmap.resolution, 1e-3)
        else:
            processed.target_spacing = 0.2

        # Flatten into initial_variables: [x0,y0,theta0,kappa0,ds0, x1,y1,...]
        processed.initial_variables = []
        for index in range(processed.state_count):
            processed.initial_variables.extend([
                processed.reference_points[index][0],
                processed.reference_points[index][1],
                theta[index],
                kappa[index],
                ds[index],
            ])

        return processed

    def build_residual_fn(
        self,
        processed: KinematicProcessedPath,
        costmap: Optional[Costmap2D],
        params: SmootherParams,
    ) -> tuple[callable, int]:
        """Build the residual function for scipy.optimize.least_squares.

        Returns (residual_fn, num_parameters).
        """
        n = processed.state_count
        num_params = n * 5

        # Pre-compute weights
        model_weight = max(params.model_weight_sqrt, 0.0)
        curvature_weight = max(params.kinematic_curvature_weight_sqrt, 0.0)
        curvature_rate_weight = max(params.kinematic_curvature_rate_weight_sqrt, 0.0)
        spacing_weight = max(params.kinematic_spacing_weight_sqrt, 0.0)
        length_weight = max(params.path_length_weight_sqrt, 0.0)
        fix_weight = max(params.fix_weight, 0.0)
        reference_weight = max(params.reference_path_weight_sqrt, 0.0)
        has_obstacle_cost = params.obstacle_terms_enabled()

        obstacle_weight = max(params.costmap_weight_sqrt, 0.0)

        # Pre-compute goal heading
        goal_position_theta = goal_position_frame_heading(
            processed.reference_points,
            processed.end_theta,
            params.keep_goal_orientation,
        )

        def residual_fn(variables: np.ndarray) -> np.ndarray:
            residuals = []

            # 1) Transition residuals: 7 per consecutive pair
            for index in range(n - 1):
                current = variables[5 * index: 5 * index + 5]
                next_state = variables[5 * (index + 1): 5 * (index + 1) + 5]
                r = transition_residuals(
                    current, next_state,
                    processed.gears[index],
                    processed.is_cusp_segment[index],
                    model_weight,
                    curvature_weight,
                    curvature_rate_weight,
                    spacing_weight,
                    length_weight,
                    fix_weight,
                    processed.target_spacing,
                )
                residuals.extend(r)

            # 2) Start boundary: 3 residuals
            start_state = variables[0:5]
            start_ref = np.array(processed.reference_points[0])
            r_start = boundary_residuals(
                start_state, start_ref,
                processed.start_theta,
                params.keep_start_orientation,
                0.0, 0.0, 0.0,
                fix_weight,
            )
            residuals.extend(r_start)

            # 3) Goal boundary: 3 residuals
            goal_state = variables[5 * (n - 1): 5 * (n - 1) + 5]
            goal_ref = np.array(processed.reference_points[-1])
            r_goal = boundary_residuals(
                goal_state, goal_ref,
                goal_position_theta,
                params.keep_goal_orientation,
                params.goal_longitudinal_tolerance,
                params.goal_lateral_tolerance,
                params.goal_orientation_tolerance,
                fix_weight,
            )
            residuals.extend(r_goal)

            # 4) Reference path residuals: 2 per state
            if reference_weight > 1e-9:
                for index in range(n):
                    state = variables[5 * index: 5 * index + 5]
                    ref_pt = np.array(processed.reference_points[index])
                    r_ref = reference_residuals(state, ref_pt, reference_weight)
                    residuals.extend(r_ref)

            # 5) Obstacle residuals: variable per state
            if has_obstacle_cost and costmap is not None:
                for index in range(n):
                    state = variables[5 * index: 5 * index + 5]
                    r_obs = obstacle_residuals(
                        state,
                        self._esdf_values,
                        costmap.size_x,
                        costmap.size_y,
                        costmap.origin_x,
                        costmap.origin_y,
                        costmap.resolution,
                        params.obstacle_safe_distance,
                        params.cost_check_radius,
                        obstacle_weight,
                        params.cost_check_points if params.cost_check_points else None,
                    )
                    residuals.extend(r_obs)

            return np.array(residuals, dtype=np.float64)

        return residual_fn, num_params

    @staticmethod
    def apply_bounds(
        lower: np.ndarray,
        upper: np.ndarray,
        reference_points: list[tuple[float, float]],
        is_cusp_segment: list[bool],
        state_count: int,
        max_curvature: float,
        max_spacing: float,
        reference_point_max_deviation_m: float,
    ) -> None:
        """Apply explicit variable bounds in-place."""
        clamped_max_curvature = max(max_curvature, 1e-6)

        for index in range(state_count):
            base = 5 * index

            if reference_point_max_deviation_m > 1e-9:
                lower[base + 0] = reference_points[index][0] - reference_point_max_deviation_m
                upper[base + 0] = reference_points[index][0] + reference_point_max_deviation_m
                lower[base + 1] = reference_points[index][1] - reference_point_max_deviation_m
                upper[base + 1] = reference_points[index][1] + reference_point_max_deviation_m

            lower[base + 3] = -clamped_max_curvature
            upper[base + 3] = clamped_max_curvature
            ds_is_used = index + 1 < state_count
            is_cusp_ds = index < len(is_cusp_segment) and is_cusp_segment[index]
            lower[base + 4] = 1e-6 if ds_is_used and not is_cusp_ds else 0.0
            if max_spacing > 1e-9:
                upper[base + 4] = max_spacing

    @staticmethod
    def unpack_path(variables: np.ndarray, state_count: int) -> list[np.ndarray]:
        """Convert flat variables back to (x, y, yaw) path."""
        path = []
        for index in range(state_count):
            base = 5 * index
            path.append(np.array([
                variables[base + 0],
                variables[base + 1],
                normalize_angle(variables[base + 2]),
            ]))
        return path

    @staticmethod
    def upsample_path_kinematic(
        variables: np.ndarray,
        processed: KinematicProcessedPath,
        params: SmootherParams,
    ) -> list[np.ndarray]:
        """Upsample the path using kinematic interpolation."""
        upsample_factor = max(params.path_upsampling_factor, 1)
        path = KinematicSmootherProblemBuilder.unpack_path(variables, processed.state_count)

        if upsample_factor <= 1 or processed.state_count < 2:
            return path

        upsampled = [path[0]]

        for index in range(processed.state_count - 1):
            is_cusp_seg = (
                index < len(processed.is_cusp_segment) and
                processed.is_cusp_segment[index]
            )
            gear = processed.gears[index] if index < len(processed.gears) else 1.0

            base = 5 * index
            x = variables[base + 0]
            y = variables[base + 1]
            theta = normalize_angle(variables[base + 2])
            kappa = variables[base + 3]
            ds = max(variables[base + 4], 0.0)
            next_kappa = variables[5 * (index + 1) + 3]

            next_pose = path[index + 1]

            if is_cusp_seg or abs(gear) < 1e-9 or ds <= 1e-6:
                upsampled.append(next_pose)
                continue

            direction = 1.0 if gear >= 0.0 else -1.0
            step = ds / upsample_factor

            interp_x = x
            interp_y = y
            interp_theta = theta
            segment_samples = []

            for step_index in range(1, upsample_factor):
                t0 = (step_index - 1) / upsample_factor
                t1 = step_index / upsample_factor
                kappa0 = kappa + (next_kappa - kappa) * t0
                kappa1 = kappa + (next_kappa - kappa) * t1

                theta_mid = interp_theta + direction * step * 0.5 * kappa0
                interp_x += direction * step * math.cos(theta_mid)
                interp_y += direction * step * math.sin(theta_mid)
                interp_theta = normalize_angle(
                    interp_theta + direction * step * 0.5 * (kappa0 + kappa1)
                )
                segment_samples.append(np.array([interp_x, interp_y, interp_theta]))

            # Predict end position
            final_t0 = (upsample_factor - 1) / upsample_factor
            final_kappa0 = kappa + (next_kappa - kappa) * final_t0
            final_theta_mid = interp_theta + direction * step * 0.5 * final_kappa0
            predicted_end_x = interp_x + direction * step * math.cos(final_theta_mid)
            predicted_end_y = interp_y + direction * step * math.sin(final_theta_mid)
            predicted_end_theta = normalize_angle(
                interp_theta + direction * step * 0.5 * (final_kappa0 + next_kappa)
            )

            closure_x = next_pose[0] - predicted_end_x
            closure_y = next_pose[1] - predicted_end_y
            closure_theta = normalize_angle(next_pose[2] - predicted_end_theta)

            # Distribute closure error uniformly
            for step_index in range(1, upsample_factor):
                t = step_index / upsample_factor
                sample = segment_samples[step_index - 1]
                upsampled.append(np.array([
                    sample[0] + t * closure_x,
                    sample[1] + t * closure_y,
                    normalize_angle(sample[2] + t * closure_theta),
                ]))

            upsampled.append(next_pose)

        return upsampled


def _downsample_input_path(
    path: list[np.ndarray],
    params: SmootherParams,
) -> list[np.ndarray]:
    """Downsample the input path, preserving cusp points."""
    if params.path_target_spacing > 1e-9:
        return _resample_input_path_by_spacing(path, params)

    downsample_factor = max(params.path_downsampling_factor, 1)
    if downsample_factor <= 1 or len(path) <= 2:
        return list(path)

    sampled = [path[0]]
    last_kept_index = 0

    def direction_sign(index: int) -> float:
        if not params.reversing_enabled:
            return 1.0
        return -1.0 if path[index][2] < 0.0 else 1.0

    for index in range(1, len(path) - 1):
        prev_sign = direction_sign(index - 1)
        current_sign = direction_sign(index)
        next_sign = direction_sign(index + 1)
        around_cusp = current_sign != prev_sign or current_sign != next_sign

        if around_cusp or (index - last_kept_index) >= downsample_factor:
            sampled.append(path[index])
            last_kept_index = index

    # Ensure last point is included
    if len(sampled) < 2 or not np.allclose(sampled[-1], path[-1], atol=1e-9):
        sampled.append(path[-1])

    if len(sampled) < 2:
        sampled = [path[0], path[-1]]

    return sampled


def _resample_input_path_by_spacing(
    path: list[np.ndarray],
    params: SmootherParams,
) -> list[np.ndarray]:
    """Resample the input path to a metric target spacing, preserving cusp points."""
    target_spacing = max(float(params.path_target_spacing), 0.0)
    if target_spacing <= 1e-9 or len(path) <= 2:
        return list(path)

    sampled = [np.array(path[0], dtype=float)]

    def direction_sign(index: int) -> float:
        if not params.reversing_enabled:
            return 1.0
        return -1.0 if path[index][2] < 0.0 else 1.0

    def append_or_update(point: np.ndarray) -> None:
        point = np.array(point, dtype=float)
        if sampled and np.linalg.norm(sampled[-1][:2] - point[:2]) <= 1e-9:
            sampled[-1][2] = point[2]
        else:
            sampled.append(point)

    distance_since_keep = 0.0
    for index in range(len(path) - 1):
        start = np.array(path[index], dtype=float)
        end = np.array(path[index + 1], dtype=float)
        start_sign = direction_sign(index)
        end_sign = direction_sign(index + 1)
        delta = end[:2] - start[:2]
        segment_length = float(np.linalg.norm(delta))

        if segment_length <= 1e-9:
            if start_sign != end_sign:
                append_or_update(end)
                distance_since_keep = 0.0
            continue

        traversed = 0.0
        while distance_since_keep + (segment_length - traversed) >= target_spacing:
            step = target_spacing - distance_since_keep
            traversed += step
            ratio = min(1.0, max(0.0, traversed / segment_length))
            sample = start + ratio * (end - start)
            sample[2] = start[2]
            append_or_update(sample)
            distance_since_keep = 0.0

        distance_since_keep += segment_length - traversed
        if start_sign != end_sign:
            append_or_update(end)
            distance_since_keep = 0.0

    append_or_update(np.array(path[-1], dtype=float))

    if len(sampled) < 2:
        sampled = [np.array(path[0], dtype=float), np.array(path[-1], dtype=float)]

    return sampled
