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
Cost functors for the kinematic smoother.

Mirrors the C++ kinematic_smoother_costs.hpp.

Each functor computes residuals as plain numpy operations (no autodiff).
The numerical Jacobian is computed by scipy.optimize.least_squares.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from constrained_smoother.utils import normalize_angle, angle_diff, EPSILON, PI


# ---------------------------------------------------------------------------
# Transition cost: 7 residuals for a pair of consecutive states
# ---------------------------------------------------------------------------

def transition_residuals(
    current: np.ndarray,
    next_state: np.ndarray,
    gear: float,
    is_cusp_segment: bool,
    model_weight: float,
    curvature_weight: float,
    curvature_rate_weight: float,
    spacing_weight: float,
    length_weight: float,
    fix_weight: float,
    target_spacing: float,
) -> np.ndarray:
    """Compute 7 residuals for the kinematic transition between two states.

    Parameters
    ----------
    current, next_state : ndarray of shape (5,)
        State vectors [x, y, theta, kappa, ds].
    gear : float
        Direction of travel: +1 forward, -1 reverse, 0 cusp.
    is_cusp_segment : bool
        Whether this is a cusp (reversal) segment.
    model_weight, curvature_weight, curvature_rate_weight, spacing_weight,
    length_weight, fix_weight : float
        Weights (already in non-sqrt form for use in residuals).
    target_spacing : float
        Desired knot spacing in meters.

    Returns
    -------
    residuals : ndarray of shape (7,)
    """
    residuals = np.zeros(7)

    if is_cusp_segment:
        residuals[0] = fix_weight * (next_state[0] - current[0])
        residuals[1] = fix_weight * (next_state[1] - current[1])
        residuals[2] = fix_weight * angle_diff(next_state[2], current[2])
        residuals[5] = spacing_weight * 10.0 * current[4]
        residuals[6] = length_weight * current[4]
        return residuals

    x, y, theta, kappa, ds = current
    next_x, next_y, next_theta, next_kappa = next_state[:4]

    direction = 1.0 if gear >= 0.0 else -1.0

    # Trapezoidal curvature integration for predicted heading
    theta_pred = theta + direction * ds * (kappa + next_kappa) * 0.5
    # Euler midpoint for predicted position, matching the C++ functor.
    theta_mid = (theta + theta_pred) * 0.5
    x_pred = x + direction * ds * math.cos(theta_mid)
    y_pred = y + direction * ds * math.sin(theta_mid)

    # Curvature rate normalization denominator
    denom = math.sqrt(ds) if ds > 1e-3 else 0.03

    residuals[0] = model_weight * (next_x - x_pred)
    residuals[1] = model_weight * (next_y - y_pred)
    residuals[2] = model_weight * angle_diff(next_theta, theta_pred)
    residuals[3] = curvature_weight * (kappa + next_kappa) * 0.5
    residuals[4] = curvature_rate_weight * (next_kappa - kappa) / denom

    spacing_ref = max(target_spacing, 1e-3)
    residuals[5] = spacing_weight * (ds - spacing_ref) / spacing_ref
    residuals[6] = length_weight * ds

    return residuals


# ---------------------------------------------------------------------------
# Boundary cost: 3 residuals for a start or end state
# ---------------------------------------------------------------------------

def boundary_residuals(
    state: np.ndarray,
    reference_point: np.ndarray,
    target_theta: float,
    keep_orientation: bool,
    longitudinal_tolerance: float,
    lateral_tolerance: float,
    orientation_tolerance: float,
    fix_weight: float,
) -> np.ndarray:
    """Compute 3 residuals for boundary (start/goal) constraints.

    Parameters
    ----------
    state : ndarray of shape (5,)
        The state vector [x, y, theta, kappa, ds].
    reference_point : ndarray of shape (2,)
        The target position.
    target_theta : float
        The target heading angle.
    keep_orientation : bool
        Whether to enforce heading constraint.
    longitudinal_tolerance, lateral_tolerance : float
        Position tolerances in the goal frame (meters).
    orientation_tolerance : float
        Heading tolerance (radians).
    fix_weight : float
        Constraint weight.
    Returns
    -------
    residuals : ndarray of shape (3,)
    """
    dx = state[0] - reference_point[0]
    dy = state[1] - reference_point[1]
    cos_theta = math.cos(target_theta)
    sin_theta = math.sin(target_theta)
    lon_error = cos_theta * dx + sin_theta * dy
    lat_error = -sin_theta * dx + cos_theta * dy

    lon_violation = abs(lon_error) - max(longitudinal_tolerance, 0.0)
    lat_violation = abs(lat_error) - max(lateral_tolerance, 0.0)

    residuals = np.zeros(3)
    residuals[0] = fix_weight * lon_violation if lon_violation > 0.0 else 0.0
    residuals[1] = fix_weight * lat_violation if lat_violation > 0.0 else 0.0

    if keep_orientation:
        heading_error = abs(angle_diff(state[2], target_theta))
        heading_violation = heading_error - max(orientation_tolerance, 0.0)
        residuals[2] = fix_weight * heading_violation if heading_violation > 0.0 else 0.0
    else:
        residuals[2] = 0.0

    return residuals


# ---------------------------------------------------------------------------
# Reference path cost: 2 residuals per state
# ---------------------------------------------------------------------------

def reference_residuals(
    state: np.ndarray,
    reference_point: np.ndarray,
    reference_weight: float,
) -> np.ndarray:
    """Compute 2 residuals pulling state toward the reference path point."""
    dx = state[0] - reference_point[0]
    dy = state[1] - reference_point[1]
    return np.array([reference_weight * dx, reference_weight * dy])


# ---------------------------------------------------------------------------
# Obstacle cost: 1 or N residuals per state
# ---------------------------------------------------------------------------

def obstacle_residuals(
    state: np.ndarray,
    esdf_values: list[float],
    costmap_size_x: int,
    costmap_size_y: int,
    costmap_origin_x: float,
    costmap_origin_y: float,
    costmap_resolution: float,
    obstacle_safe_distance: float,
    cost_check_radius: float,
    pose_obstacle_weight: float,
    cost_check_points: Optional[list[float]] = None,
) -> np.ndarray:
    """Compute obstacle penalty residuals for a single state.

    Parameters
    ----------
    state : ndarray of shape (5,)
    esdf_values : list[float]
        Flat ESDF values (row-major, size_x * size_y).
    costmap_size_x, costmap_size_y : int
    costmap_origin_x, costmap_origin_y, costmap_resolution : float
    obstacle_safe_distance : float
    cost_check_radius : float
    pose_obstacle_weight : float
        Obstacle residual weight already resolved for this state.
    cost_check_points : list[float] or None
        Local (x, y, weight) triples for multi-point footprint check.

    Returns
    -------
    residuals : ndarray of shape (N,)
        N = len(cost_check_points)/3 if cost_check_points else 1.
    """
    x, y, theta = state[0], state[1], state[2]
    pose_weight = max(pose_obstacle_weight, 0.0)

    def _obstacle_penalty(wx: float, wy: float) -> float:
        grid_x = (wx - costmap_origin_x) / costmap_resolution
        grid_y = (wy - costmap_origin_y) / costmap_resolution

        if (grid_x < 1.5 or grid_y < 1.5 or
                grid_x >= costmap_size_x - 1.5 or
                grid_y >= costmap_size_y - 1.5):
            return 1.0

        # Bilinear interpolation for ESDF value
        gx = grid_x - 0.5
        gy = grid_y - 0.5
        ix = int(math.floor(gx))
        iy = int(math.floor(gy))
        fx = gx - ix
        fy = gy - iy

        def _esdf_at(r: int, c: int) -> float:
            if 0 <= c < costmap_size_x and 0 <= r < costmap_size_y:
                idx = r * costmap_size_x + c
                if 0 <= idx < len(esdf_values):
                    return esdf_values[idx]
            return float("inf")

        v00 = _esdf_at(iy, ix)
        v10 = _esdf_at(iy + 1, ix)
        v01 = _esdf_at(iy, ix + 1)
        v11 = _esdf_at(iy + 1, ix + 1)
        distance = (v00 * (1 - fx) * (1 - fy) +
                    v10 * fx * (1 - fy) +
                    v01 * (1 - fx) * fy +
                    v11 * fx * fy)

        surface_distance = distance - cost_check_radius
        safe_dist = max(obstacle_safe_distance, 1e-6)
        if surface_distance >= safe_dist:
            return 0.0

        return (safe_dist - surface_distance) / safe_dist

    if not cost_check_points:
        return np.array([pose_weight * _obstacle_penalty(x, y)])

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    residuals = []
    for offset in range(0, len(cost_check_points) - 2, 3):
        local_x = cost_check_points[offset + 0]
        local_y = cost_check_points[offset + 1]
        point_weight = cost_check_points[offset + 2]
        world_x = x + cos_theta * local_x - sin_theta * local_y
        world_y = y + sin_theta * local_x + cos_theta * local_y
        residuals.append(pose_weight * point_weight * _obstacle_penalty(world_x, world_y))
    return np.array(residuals)
