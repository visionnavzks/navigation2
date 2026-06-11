#ifndef KINEMATIC_PATH_SMOOTHER__SOLVER_UTILS_HPP_
#define KINEMATIC_PATH_SMOOTHER__SOLVER_UTILS_HPP_

#include <iostream>
#include <string>

#include "ceres/ceres.h"

#include "kinematic_path_smoother/exceptions.hpp"

namespace kinematic_path_smoother
{

/// 调用 Ceres 并把求解失败统一映射到 FailureInfo / FailedToSmoothPath。
///
/// 这里不做路径几何判断；几何可交付性由 SmootherValidator 负责。
inline bool solveOrReport(
  ceres::Problem & problem,
  const ceres::Solver::Options & options,
  bool debug,
  FailureInfo * failure)
{
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (debug) {
    std::cout << summary.FullReport() << std::endl;
  }

  if (!summary.IsSolutionUsable()) {
    return failOrThrow(failure, FailureReason::SolverFailure, summary.BriefReport());
  }
  if (summary.final_cost > summary.initial_cost + 1e-12) {
    return failOrThrow(failure, FailureReason::SolverFailure, "Ceres increased the objective cost");
  }
  return true;
}

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__SOLVER_UTILS_HPP_
