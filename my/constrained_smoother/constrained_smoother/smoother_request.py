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
SmootherResult and SmootherRequest data structures.

Mirrors the C++ smoother_request.hpp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from constrained_smoother.options import SmootherParams
from constrained_smoother.exceptions import SmoothingFailureInfo


@dataclass
class SmootherResult:
    """Result of a single smooth() call.

    candidate_path: path after solving, before validation (x, y, yaw)
    smoothed_path: final path after validation and upsampling (x, y, yaw)
    optimized_knot_count: number of state points that participated in optimization
    target_spacing: target knot spacing used (meters)
    success: whether a deliverable smoothed path was obtained
    """

    candidate_path: list[np.ndarray] = field(default_factory=list)
    smoothed_path: list[np.ndarray] = field(default_factory=list)
    optimized_knot_count: int = 0
    target_spacing: float = 0.0
    success: bool = False


@dataclass
class SmootherRequest:
    """Single smooth() call context (non-owning view).

    Input path uses (x, y, direction_sign) in the third component.
    """

    path: list[np.ndarray]
    start_dir: np.ndarray
    end_dir: np.ndarray
    costmap: object  # Costmap2D or None
    params: SmootherParams
    precomputed_esdf: Optional[list[float]] = None
    failure: Optional[SmoothingFailureInfo] = None
