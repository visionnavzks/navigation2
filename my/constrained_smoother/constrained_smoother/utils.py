# Copyright (c) 2021 RoboTech Vision
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
Shared math and coordinate helpers for the constrained smoother.

Mirrors the C++ utils.hpp.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constrained_smoother.costmap2d import Costmap2D

EPSILON = 0.0001
PI = math.pi


def normalize_angle(angle: float) -> float:
    """Normalize angle to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(a: float, b: float) -> float:
    """Signed difference of two angles, normalized to (-pi, pi]."""
    return normalize_angle(a - b)


def world_to_grid(costmap: Costmap2D, wx: float, wy: float) -> tuple[int, int]:
    """Convert world coordinates to grid cell indices."""
    resolution = costmap.resolution
    mx = int(math.floor((wx - costmap.origin_x) / resolution))
    my = int(math.floor((wy - costmap.origin_y) / resolution))
    return mx, my


def grid_to_world(costmap: Costmap2D, mx: int, my: int) -> tuple[float, float]:
    """Convert grid cell indices to world coordinates (center of cell)."""
    return (
        costmap.origin_x + (mx + 0.5) * costmap.resolution,
        costmap.origin_y + (my + 0.5) * costmap.resolution,
    )


def in_bounds(mx: int, my: int, size_x: int, size_y: int) -> bool:
    """Check if grid indices are within map bounds."""
    return 0 <= mx < size_x and 0 <= my < size_y


def in_bounds_costmap(costmap: Costmap2D, mx: int, my: int) -> bool:
    """Check if grid indices are within costmap bounds."""
    return in_bounds(mx, my, costmap.size_x, costmap.size_y)


def goal_position_frame_heading(
    reference_points: list[tuple[float, float]],
    end_theta: float,
    keep_goal_orientation: bool,
) -> float:
    """Compute the reference heading for the goal boundary cost.

    If keep_goal_orientation is True or there are fewer than 2 points,
    returns end_theta. Otherwise returns the geometric heading of the
    last segment.
    """
    if keep_goal_orientation or len(reference_points) < 2:
        return end_theta

    goal_dx = reference_points[-1][0] - reference_points[-2][0]
    goal_dy = reference_points[-1][1] - reference_points[-2][1]
    if math.hypot(goal_dx, goal_dy) <= EPSILON:
        return end_theta

    return math.atan2(goal_dy, goal_dx)
