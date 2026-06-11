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
Lightweight 2D occupancy costmap, independent of any ROS/Ceres dependency.

Mirrors the C++ esdf_core::Costmap2D.
"""

from __future__ import annotations

import numpy as np


class Costmap2D:
    """Lightweight 2D occupancy costmap.

    Stores an occupancy grid as a flat uint8 array in row-major order.
    """

    NO_INFORMATION: int = 255
    LETHAL_OBSTACLE: int = 254
    INSCRIBED_INFLATED_OBSTACLE: int = 253
    FREE_SPACE: int = 0

    def __init__(
        self,
        size_x: int = 0,
        size_y: int = 0,
        resolution: float = 1.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ) -> None:
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.data: np.ndarray = np.zeros((size_y, size_x), dtype=np.uint8)

    def get_cost(self, mx: int, my: int) -> int:
        """Get cost at grid cell (mx, my)."""
        return int(self.data[my, mx])

    def set_cost(self, mx: int, my: int, cost: int) -> None:
        """Set cost at grid cell (mx, my)."""
        self.data[my, mx] = np.uint8(cost)

    def get_char_map(self) -> np.ndarray:
        """Return the raw data array (row-major, shape (size_y, size_x))."""
        return self.data
