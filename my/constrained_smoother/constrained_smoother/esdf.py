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
2D Euclidean Signed Distance Field computation.

Mirrors the C++ esdf_core::ESDF.
"""

from __future__ import annotations

import math
import heapq
from enum import Enum
from typing import Optional

import numpy as np

from constrained_smoother.costmap2d import Costmap2D
from constrained_smoother.exceptions import InvalidCostmap


class ESDFAlgorithm(Enum):
    Exact = "exact"
    Approximate = "approximate"


# 8-connected neighbourhood offsets: (dx, dy, distance)
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


def _to_index(mx: int, my: int, size_x: int) -> int:
    return my * size_x + mx


def _in_bounds(mx: int, my: int, size_x: int, size_y: int) -> bool:
    return 0 <= mx < size_x and 0 <= my < size_y


def _compute_approximate_unsigned_esdf(
    costmap: Costmap2D,
    lethal_cost: int,
    treat_obstacles_as_zero: bool,
) -> list[float]:
    """Dijkstra-style 8-neighborhood approximate unsigned distance transform."""
    size_x = costmap.size_x
    size_y = costmap.size_y
    cell_count = size_x * size_y
    esdf = [float("inf")] * cell_count

    # Priority queue: (distance, flat_index)
    queue: list[tuple[float, int]] = []
    for my in range(size_y):
        for mx in range(size_x):
            index = _to_index(mx, my, size_x)
            is_obstacle = costmap.get_cost(mx, my) >= lethal_cost
            is_zero_seed = treat_obstacles_as_zero == is_obstacle
            if is_zero_seed:
                esdf[index] = 0.0
                heapq.heappush(queue, (0.0, index))

    while queue:
        dist, idx = heapq.heappop(queue)
        if dist > esdf[idx]:
            continue
        cx = idx % size_x
        cy = idx // size_x
        for dx, dy, ndist in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not _in_bounds(nx, ny, size_x, size_y):
                continue
            nidx = _to_index(nx, ny, size_x)
            candidate = dist + ndist * costmap.resolution
            if candidate < esdf[nidx]:
                esdf[nidx] = candidate
                heapq.heappush(queue, (candidate, nidx))

    return esdf


def compute_esdf(
    costmap: Costmap2D,
    lethal_cost: int = Costmap2D.LETHAL_OBSTACLE,
    use_exact: bool = True,
) -> list[float]:
    """Compute the signed ESDF for the given costmap.

    Returns a flat list of length size_x * size_y (row-major), where:
      - positive: distance to nearest obstacle surface
      - negative: distance from inside obstacle to nearest free cell
      - 0.0: on the obstacle boundary
    """
    if costmap is None:
        raise InvalidCostmap("compute_esdf received a null costmap")
    return compute_approximate_esdf(costmap, lethal_cost)


def compute_approximate_esdf(
    costmap: Costmap2D,
    lethal_cost: int = Costmap2D.LETHAL_OBSTACLE,
) -> list[float]:
    """Compute signed ESDF using approximate Dijkstra."""
    size_x = costmap.size_x
    size_y = costmap.size_y

    outside_esdf = _compute_approximate_unsigned_esdf(costmap, lethal_cost, True)
    inside_esdf = _compute_approximate_unsigned_esdf(costmap, lethal_cost, False)

    signed_esdf = list(outside_esdf)
    for my in range(size_y):
        for mx in range(size_x):
            index = _to_index(mx, my, size_x)
            if costmap.get_cost(mx, my) >= lethal_cost:
                signed_esdf[index] = -inside_esdf[index]

    return signed_esdf
