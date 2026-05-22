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

#ifndef KINEMATIC_SMOOTHER__SOLVER_UTILS_HPP_
#define KINEMATIC_SMOOTHER__SOLVER_UTILS_HPP_

#include <iostream>
#include <string>

#include "ceres/ceres.h"

#include "kinematic_smoother/exceptions.hpp"

namespace kinematic_smoother
{

/// 共享的 Ceres 求解入口。
///
/// 顶层 smoother 不直接操作 `ceres::Solve(...)` 的返回细节，而是统一经由这里把
/// “求解器拒绝结果”和“目标函数未改善”转换成稳定的 failure reason。
inline bool solveProblemOrReportFailure(
  ceres::Problem & problem,
  const ceres::Solver::Options & options,
  bool debug,
  const char * smoother_name,
  SmoothingFailureInfo * failure)
{
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (debug) {
    std::cout << summary.FullReport() << std::endl;
  }

  if (!summary.IsSolutionUsable()) {
    return throwOrStoreSmoothingFailure(
      failure,
      SmoothingFailureReason::SolverRejectedSolution,
      std::string(smoother_name) + " rejected the Ceres solution as unusable");
  }
  if (summary.initial_cost - summary.final_cost < 0.0) {
    return throwOrStoreSmoothingFailure(
      failure,
      SmoothingFailureReason::NoCostImprovement,
      std::string(smoother_name) + " did not improve the objective cost");
  }

  return true;
}

}  // namespace kinematic_smoother

#endif  // KINEMATIC_SMOOTHER__SOLVER_UTILS_HPP_