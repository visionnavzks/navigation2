"""ESDF computation."""

from __future__ import annotations
import math
import heapq
from enum import Enum

import numpy as np

from smoother_clothoid_py.costmap2d import Costmap2D
from smoother_clothoid_py.exceptions import InvalidCostmap


class ESDFAlgorithm(Enum):
    Exact = "exact"
    Approximate = "approximate"


_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def _unsigned_esdf(costmap: Costmap2D, lethal: int, zero_is_obstacle: bool) -> list[float]:
    sx, sy = costmap.size_x, costmap.size_y
    esdf = [float("inf")] * (sx * sy)
    q: list[tuple[float, int]] = []
    for my in range(sy):
        for mx in range(sx):
            idx = my * sx + mx
            is_obs = costmap.get_cost(mx, my) >= lethal
            if zero_is_obstacle == is_obs:
                esdf[idx] = 0.0
                heapq.heappush(q, (0.0, idx))
    while q:
        d, i = heapq.heappop(q)
        if d > esdf[i]: continue
        cx, cy = i % sx, i // sx
        for dx, dy, nd in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < sx and 0 <= ny < sy:
                ni = ny * sx + nx
                c = d + nd * costmap.resolution
                if c < esdf[ni]:
                    esdf[ni] = c
                    heapq.heappush(q, (c, ni))
    return esdf


def compute_esdf(costmap: Costmap2D, lethal: int = Costmap2D.LETHAL_OBSTACLE,
                 use_exact: bool = True) -> list[float]:
    if costmap is None:
        raise InvalidCostmap("compute_esdf received a null costmap")
    return compute_approximate_esdf(costmap, lethal)


def compute_approximate_esdf(costmap: Costmap2D, lethal: int = Costmap2D.LETHAL_OBSTACLE) -> list[float]:
    sx, sy = costmap.size_x, costmap.size_y
    outside = _unsigned_esdf(costmap, lethal, True)
    inside = _unsigned_esdf(costmap, lethal, False)
    result = list(outside)
    for my in range(sy):
        for mx in range(sx):
            if costmap.get_cost(mx, my) >= lethal:
                result[my * sx + mx] = -inside[my * sx + mx]
    return result
