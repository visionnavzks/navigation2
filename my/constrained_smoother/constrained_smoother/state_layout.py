"""Shared layout helpers for flattened kinematic smoother states."""

from __future__ import annotations

import numpy as np

STATE_SIZE = 5
X_INDEX = 0
Y_INDEX = 1
THETA_INDEX = 2
KAPPA_INDEX = 3
DS_INDEX = 4

ENABLED_EPS = 1e-9
GEOMETRY_EPS = 1e-6
POINT_EPS = 1e-9


def state_offset(index: int) -> int:
    return STATE_SIZE * index


def state_view(variables: np.ndarray, index: int) -> np.ndarray:
    base = state_offset(index)
    return variables[base: base + STATE_SIZE]
