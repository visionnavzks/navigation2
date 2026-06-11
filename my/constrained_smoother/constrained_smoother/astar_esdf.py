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
A* path planner with ESDF integration.

Mirrors the C++ astar_esdf.hpp.
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from constrained_smoother.costmap2d import Costmap2D
from constrained_smoother.esdf import compute_esdf
from constrained_smoother.exceptions import InvalidCostmap
from constrained_smoother.utils import world_to_grid, grid_to_world, in_bounds


@dataclass
class AStarPlannerParams:
    """Parameters for the A* planner."""

    lethal_cost: int = Costmap2D.LETHAL_OBSTACLE
    use_exact_esdf: bool = True
    safe_distance: float = 0.5
    cost_penalty_weight: float = 1.0
    point_radius: float = 0.0
    collision_check_radius: float = 0.0
    collision_check_points: list[float] = field(default_factory=list)
    use_rectangular_footprint: bool = False
    rectangular_length: float = 0.0
    rectangular_width: float = 0.0


# 8-connected neighbourhood
_NEIGHBORS = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
]


class AStarPlanner:
    """Grid-based A* path planner with ESDF integration."""

    def __init__(self) -> None:
        self._esdf: list[float] = []

    @property
    def esdf(self) -> list[float]:
        """Get the last computed ESDF values."""
        return self._esdf

    def get_esdf(self) -> list[float]:
        """Get the last computed ESDF values (compatibility alias)."""
        return self._esdf

    @staticmethod
    def compute_esdf(
        costmap: Costmap2D,
        lethal_cost: int = Costmap2D.LETHAL_OBSTACLE,
        use_exact: bool = True,
    ) -> list[float]:
        """Compute ESDF for the given costmap."""
        return compute_esdf(costmap, lethal_cost)

    @staticmethod
    def evaluate_penalty(distance: float, safe_distance: float) -> float:
        """Evaluate obstacle penalty based on distance."""
        if not math.isfinite(distance) or distance >= safe_distance:
            return 0.0
        clamped_safe = max(safe_distance, 1e-6)
        normalized_gap = (clamped_safe - distance) / clamped_safe
        return normalized_gap * normalized_gap

    def plan(
        self,
        costmap: Costmap2D,
        start_wx: float,
        start_wy: float,
        goal_wx: float,
        goal_wy: float,
        params: AStarPlannerParams,
    ) -> list[np.ndarray]:
        """Plan a path from start to goal in world coordinates.

        Returns list of np.ndarray([x, y]) waypoints, or empty list if no path.
        """
        if costmap is None:
            raise InvalidCostmap("AStarPlanner requires a valid costmap")

        size_x = costmap.size_x
        size_y = costmap.size_y
        if size_x <= 0 or size_y <= 0:
            return []

        # Compute ESDF
        self._esdf = compute_esdf(
            costmap,
            params.lethal_cost,
        )

        start = world_to_grid(costmap, start_wx, start_wy)
        goal = world_to_grid(costmap, goal_wx, goal_wy)

        if (not in_bounds(start[0], start[1], size_x, size_y) or
                not in_bounds(goal[0], goal[1], size_x, size_y)):
            return []

        nominal_yaw = math.atan2(goal_wy - start_wy, goal_wx - start_wx)

        if (not self._is_traversable(costmap, start[0], start[1], nominal_yaw, params) or
                not self._is_traversable(costmap, goal[0], goal[1], nominal_yaw, params)):
            return []

        cell_count = size_x * size_y
        start_index = self._to_index(start[0], start[1], size_x)
        goal_index = self._to_index(goal[0], goal[1], size_x)

        g_score = [float("inf")] * cell_count
        came_from = [-1] * cell_count

        # Priority queue: (f_score, g_score, flat_index)
        open_set: list[tuple[float, float, int]] = []
        g_score[start_index] = 0.0
        initial_f = self._heuristic(start[0], start[1], goal[0], goal[1]) * costmap.resolution
        heapq.heappush(open_set, (initial_f, 0.0, start_index))

        while open_set:
            f, g, current = heapq.heappop(open_set)

            if g > g_score[current]:
                continue

            if current == goal_index:
                return self._reconstruct_path(costmap, came_from, goal_index, start_index, size_x)

            cx = current % size_x
            cy = current // size_x

            for dx, dy, dist in _NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                traversal_yaw = math.atan2(float(dy), float(dx))

                if (not in_bounds(nx, ny, size_x, size_y) or
                        not self._is_traversable(costmap, nx, ny, traversal_yaw, params)):
                    continue

                next_index = self._to_index(nx, ny, size_x)
                step_cost = dist * costmap.resolution

                next_center = grid_to_world(costmap, nx, ny)
                footprint_distance = self._evaluate_footprint_distance(
                    costmap, next_center[0], next_center[1], traversal_yaw, params, next_index
                )
                surface_distance = footprint_distance - max(params.collision_check_radius, 0.0)
                penalty = params.cost_penalty_weight * self.evaluate_penalty(
                    surface_distance, params.safe_distance
                )
                tentative_g = g + step_cost + penalty * costmap.resolution

                if tentative_g < g_score[next_index]:
                    g_score[next_index] = tentative_g
                    came_from[next_index] = current
                    h = self._heuristic(nx, ny, goal[0], goal[1]) * costmap.resolution
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, next_index))

        return []

    # ---- Private helpers ----

    @staticmethod
    def _to_index(mx: int, my: int, size_x: int) -> int:
        return my * size_x + mx

    def _is_traversable(
        self,
        costmap: Costmap2D,
        mx: int,
        my: int,
        yaw: float,
        params: AStarPlannerParams,
    ) -> bool:
        if costmap.get_cost(mx, my) >= params.lethal_cost:
            return False

        if params.collision_check_points and params.collision_check_radius > 1e-9:
            return self._is_multi_circle_traversable(costmap, mx, my, yaw, params)

        if params.use_rectangular_footprint:
            return self._is_axis_aligned_rectangle_traversable(costmap, mx, my, params)

        return self._is_point_robot_traversable(costmap, mx, my, params)

    def _is_point_robot_traversable(
        self,
        costmap: Costmap2D,
        mx: int,
        my: int,
        params: AStarPlannerParams,
    ) -> bool:
        point_radius = max(params.point_radius, 0.0)
        if point_radius <= 1e-9:
            return True

        center = grid_to_world(costmap, mx, my)
        if not self._is_footprint_inside_map_bounds(
            costmap, center[0], center[1], point_radius, point_radius
        ):
            return False

        index = self._to_index(mx, my, costmap.size_x)
        return 0 <= index < len(self._esdf) and self._esdf[index] >= point_radius

    def _is_multi_circle_traversable(
        self,
        costmap: Costmap2D,
        mx: int,
        my: int,
        yaw: float,
        params: AStarPlannerParams,
    ) -> bool:
        center = grid_to_world(costmap, mx, my)
        radius = max(params.collision_check_radius, 0.0)
        clearance = self._evaluate_footprint_distance(
            costmap, center[0], center[1], yaw, params,
            self._to_index(mx, my, costmap.size_x),
        )
        return math.isfinite(clearance) and clearance >= radius

    def _is_axis_aligned_rectangle_traversable(
        self,
        costmap: Costmap2D,
        mx: int,
        my: int,
        params: AStarPlannerParams,
    ) -> bool:
        half_length = max(params.rectangular_length, 0.0) * 0.5
        half_width = max(params.rectangular_width, 0.0) * 0.5
        if half_length <= 1e-9 and half_width <= 1e-9:
            return True

        center = grid_to_world(costmap, mx, my)
        if not self._is_footprint_inside_map_bounds(
            costmap, center[0], center[1], half_length, half_width
        ):
            return False

        resolution = costmap.resolution
        origin_x = costmap.origin_x
        origin_y = costmap.origin_y
        min_mx = int(math.floor((center[0] - half_length - origin_x) / resolution))
        max_mx = int(math.ceil((center[0] + half_length - origin_x) / resolution)) - 1
        min_my = int(math.floor((center[1] - half_width - origin_y) / resolution))
        max_my = int(math.ceil((center[1] + half_width - origin_y) / resolution)) - 1

        size_x = costmap.size_x
        size_y = costmap.size_y
        if min_mx < 0 or min_my < 0 or max_mx >= size_x or max_my >= size_y:
            return False

        for check_my in range(min_my, max_my + 1):
            for check_mx in range(min_mx, max_mx + 1):
                if costmap.get_cost(check_mx, check_my) >= params.lethal_cost:
                    return False

        return True

    @staticmethod
    def _is_footprint_inside_map_bounds(
        costmap: Costmap2D,
        center_wx: float,
        center_wy: float,
        half_extent_x: float,
        half_extent_y: float,
    ) -> bool:
        min_wx = center_wx - half_extent_x
        max_wx = center_wx + half_extent_x
        min_wy = center_wy - half_extent_y
        max_wy = center_wy + half_extent_y
        map_min_x = costmap.origin_x
        map_min_y = costmap.origin_y
        map_max_x = map_min_x + costmap.size_x * costmap.resolution
        map_max_y = map_min_y + costmap.size_y * costmap.resolution
        return (min_wx >= map_min_x and min_wy >= map_min_y and
                max_wx <= map_max_x and max_wy <= map_max_y)

    def _evaluate_footprint_distance(
        self,
        costmap: Costmap2D,
        center_wx: float,
        center_wy: float,
        yaw: float,
        params: AStarPlannerParams,
        fallback_index: int,
    ) -> float:
        if not params.collision_check_points or params.collision_check_radius <= 1e-9:
            if 0 <= fallback_index < len(self._esdf):
                return self._esdf[fallback_index]
            return float("-inf")

        size_x = costmap.size_x
        size_y = costmap.size_y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        min_distance = float("inf")

        for offset in range(0, len(params.collision_check_points) - 1, 2):
            local_x = params.collision_check_points[offset]
            local_y = params.collision_check_points[offset + 1]
            world_x = center_wx + cos_yaw * local_x - sin_yaw * local_y
            world_y = center_wy + sin_yaw * local_x + cos_yaw * local_y
            grid = world_to_grid(costmap, world_x, world_y)
            if not in_bounds(grid[0], grid[1], size_x, size_y):
                return float("-inf")

            index = self._to_index(grid[0], grid[1], size_x)
            if index < 0 or index >= len(self._esdf):
                return float("-inf")

            min_distance = min(min_distance, self._esdf[index])

        return min_distance

    @staticmethod
    def _heuristic(ax: int, ay: int, bx: int, by: int) -> float:
        """Octile distance heuristic."""
        dx = abs(ax - bx)
        dy = abs(ay - by)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    @staticmethod
    def _reconstruct_path(
        costmap: Costmap2D,
        came_from: list[int],
        goal_index: int,
        start_index: int,
        size_x: int,
    ) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        current = goal_index
        while current >= 0:
            mx = current % size_x
            my = current // size_x
            wx, wy = grid_to_world(costmap, mx, my)
            path.append(np.array([wx, wy]))
            if current == start_index:
                break
            current = came_from[current]

        if not path or current < 0:
            return []

        path.reverse()
        return path
