"""Lightweight 2D costmap."""

from __future__ import annotations
import numpy as np


class Costmap2D:
    NO_INFORMATION: int = 255
    LETHAL_OBSTACLE: int = 254
    INSCRIBED_INFLATED_OBSTACLE: int = 253
    FREE_SPACE: int = 0

    def __init__(self, size_x: int = 0, size_y: int = 0, resolution: float = 1.0,
                 origin_x: float = 0.0, origin_y: float = 0.0) -> None:
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.data: np.ndarray = np.zeros((size_y, size_x), dtype=np.uint8)

    def get_cost(self, mx: int, my: int) -> int:
        return int(self.data[my, mx])

    def set_cost(self, mx: int, my: int, cost: int) -> None:
        self.data[my, mx] = np.uint8(max(0, min(255, int(cost))))

    def get_char_map(self) -> np.ndarray:
        return self.data
