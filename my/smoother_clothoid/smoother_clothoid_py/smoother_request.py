"""SmootherResult and SmootherRequest."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from smoother_clothoid_py.options import SmootherParams
from smoother_clothoid_py.exceptions import SmoothingFailureInfo


@dataclass
class SmootherResult:
    candidate_path: list[np.ndarray] = field(default_factory=list)
    smoothed_path: list[np.ndarray] = field(default_factory=list)
    optimized_knot_count: int = 0
    target_spacing: float = 0.0
    success: bool = False


@dataclass
class SmootherRequest:
    path: list[np.ndarray]
    start_dir: np.ndarray
    end_dir: np.ndarray
    costmap: object
    params: SmootherParams
    precomputed_esdf: Optional[list[float]] = None
    failure: Optional[SmoothingFailureInfo] = None
