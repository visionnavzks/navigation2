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
constrained_smoother - Pure Python kinematic path smoother.

Based on the C++ KinematicSmoother, providing nonlinear least-squares
path optimization with ESDF integration, A* planning, and a Flask web app.
"""

from constrained_smoother.options import SmootherParams, OptimizerParams, LinearSolver
from constrained_smoother.exceptions import (
    ErrorCode,
    SmoothingFailureReason,
    SmoothingFailureInfo,
    InvalidPath,
    InvalidCostmap,
    PrecomputedEsdfSizeMismatch,
    FailedToSmoothPath,
)
from constrained_smoother.smoother_request import SmootherResult, SmootherRequest
from constrained_smoother.costmap2d import Costmap2D
from constrained_smoother.smoother import KinematicSmoother

__all__ = [
    "SmootherParams",
    "OptimizerParams",
    "LinearSolver",
    "ErrorCode",
    "SmoothingFailureReason",
    "SmoothingFailureInfo",
    "InvalidPath",
    "InvalidCostmap",
    "PrecomputedEsdfSizeMismatch",
    "FailedToSmoothPath",
    "SmootherResult",
    "SmootherRequest",
    "Costmap2D",
    "KinematicSmoother",
]
