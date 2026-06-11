"""Math and coordinate helpers."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smoother_clothoid_py.costmap2d import Costmap2D

EPSILON = 0.0001
PI = math.pi


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(a: float, b: float) -> float:
    return normalize_angle(a - b)


def world_to_grid(costmap: Costmap2D, wx: float, wy: float) -> tuple[int, int]:
    r = costmap.resolution
    return int(math.floor((wx - costmap.origin_x) / r)), int(math.floor((wy - costmap.origin_y) / r))


def grid_to_world(costmap: Costmap2D, mx: int, my: int) -> tuple[float, float]:
    return (costmap.origin_x + (mx + 0.5) * costmap.resolution,
            costmap.origin_y + (my + 0.5) * costmap.resolution)


def in_bounds(mx: int, my: int, size_x: int, size_y: int) -> bool:
    return 0 <= mx < size_x and 0 <= my < size_y


def goal_position_frame_heading(
    refs: list[tuple[float, float]], end_theta: float, keep: bool
) -> float:
    if keep or len(refs) < 2:
        return end_theta
    dx = refs[-1][0] - refs[-2][0]
    dy = refs[-1][1] - refs[-2][1]
    return math.atan2(dy, dx) if math.hypot(dx, dy) > EPSILON else end_theta
