// Copyright (c) 2021 RoboTech Vision
// Copyright (c) 2020, Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONSTRAINED_SMOOTHER__COSTMAP2D_HPP_
#define CONSTRAINED_SMOOTHER__COSTMAP2D_HPP_

// This header is a thin shim that re-exports esdf_core::Costmap2D into the
// constrained_smoother namespace. The actual implementation lives in the
// esdf_core package so it can be shared with other consumers (e.g.
// hybrid_astar) without pulling in Ceres.

#include "esdf_core/costmap2d.hpp"

namespace constrained_smoother
{

using Costmap2D = esdf_core::Costmap2D;

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__COSTMAP2D_HPP_
