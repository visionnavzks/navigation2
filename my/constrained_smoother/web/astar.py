"""A* path planner on a 2D costmap grid."""

import heapq
import math
import numpy as np


class AStarPlanner:
    """Grid-based A* path planner that operates on a 2D costmap."""

    # 8-connected neighbourhood: (dx, dy, cost_multiplier)
    NEIGHBORS = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (-1, -1, math.sqrt(2)),
    ]

    LETHAL = 254
    INSCRIBED = 253

    def __init__(self, costmap_data, size_x, size_y, resolution, origin_x, origin_y,
                 lethal_cost=LETHAL, cost_factor=0.5):
        """
        Parameters
        ----------
        costmap_data : 2-D numpy uint8 array (size_y, size_x) – row-major grid.
        size_x, size_y : grid dimensions in cells.
        resolution : metres per cell.
        origin_x, origin_y : world origin of cell (0, 0).
        lethal_cost : costs >= this value are impassable.
        cost_factor : multiplier applied to cell costs for traversal weight.
        """
        self.grid = costmap_data
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.lethal_cost = lethal_cost
        self.cost_factor = cost_factor

    # -- coordinate helpers --------------------------------------------------

    def world_to_grid(self, wx, wy):
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def grid_to_world(self, mx, my):
        wx = self.origin_x + (mx + 0.5) * self.resolution
        wy = self.origin_y + (my + 0.5) * self.resolution
        return wx, wy

    def in_bounds(self, mx, my):
        return 0 <= mx < self.size_x and 0 <= my < self.size_y

    def is_free(self, mx, my):
        return self.in_bounds(mx, my) and self.grid[my, mx] < self.lethal_cost

    # -- A* ------------------------------------------------------------------

    @staticmethod
    def _heuristic(ax, ay, bx, by):
        """Octile distance heuristic (admissible for 8-connected grid)."""
        dx = abs(ax - bx)
        dy = abs(ay - by)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def plan(self, start_wx, start_wy, goal_wx, goal_wy):
        """Plan a path from (start_wx, start_wy) to (goal_wx, goal_wy) in world coordinates.

        Returns
        -------
        path : list of (wx, wy) or None if no path found.
        """
        sx, sy = self.world_to_grid(start_wx, start_wy)
        gx, gy = self.world_to_grid(goal_wx, goal_wy)

        if not self.is_free(sx, sy) or not self.is_free(gx, gy):
            return None

        open_set = []  # (f, g, x, y)
        heapq.heappush(open_set, (0.0, 0.0, sx, sy))
        came_from = {}
        g_score = {(sx, sy): 0.0}

        while open_set:
            f, g, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                return self._reconstruct(came_from, (gx, gy))

            if g > g_score.get((cx, cy), float('inf')):
                continue

            for dx, dy, base_cost in self.NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if not self.is_free(nx, ny):
                    continue

                cell_cost = float(self.grid[ny, nx])
                move_cost = base_cost * self.resolution + self.cost_factor * cell_cost * self.resolution
                tentative_g = g + move_cost

                if tentative_g < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = tentative_g
                    h = self._heuristic(nx, ny, gx, gy) * self.resolution
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)

        return None  # no path

    def _reconstruct(self, came_from, current):
        path = [self.grid_to_world(*current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.grid_to_world(*current))
        path.reverse()
        return path


def downsample_path(path, target_ds):
    """Reduce a dense grid path to roughly uniform spacing *target_ds* metres."""
    if not path or len(path) < 2:
        return path
    result = [path[0]]
    acc = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        seg = math.hypot(dx, dy)
        acc += seg
        if acc >= target_ds:
            result.append(path[i])
            acc = 0.0
    # Always include the last point
    if result[-1] != path[-1]:
        result.append(path[-1])
    return result
