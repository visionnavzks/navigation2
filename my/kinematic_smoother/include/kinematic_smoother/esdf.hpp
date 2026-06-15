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

#ifndef CONSTRAINED_SMOOTHER__ESDF_HPP_
#define CONSTRAINED_SMOOTHER__ESDF_HPP_

// This header is a thin shim that re-exports the ESDF computation from the
// esdf_core package. The actual implementation lives there so that consumers
// without a Ceres dependency (e.g. hybrid_astar) can use it directly.

#include "esdf_core/esdf.hpp"
#include "esdf_core/exceptions.hpp"

namespace kinematic_smoother
{

using ESDF = esdf_core::ESDF;
using ESDFAlgorithm = esdf_core::ESDFAlgorithm;
using InvalidCostmap = esdf_core::InvalidCostmap;
using PrecomputedEsdfSizeMismatch = esdf_core::PrecomputedEsdfSizeMismatch;

}  // namespace kinematic_smoother

#endif  // CONSTRAINED_SMOOTHER__ESDF_HPP_
