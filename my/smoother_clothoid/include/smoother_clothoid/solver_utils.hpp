#ifndef SMOOTHER_CLOTHOID__SOLVER_UTILS_HPP_
#define SMOOTHER_CLOTHOID__SOLVER_UTILS_HPP_

#include <iostream>
#include <string>

#include "ceres/ceres.h"

#include "smoother_clothoid/exceptions.hpp"

namespace smoother_clothoid
{

inline bool solveProblemOrReportFailure(
  ceres::Problem & problem,
  const ceres::Solver::Options & options,
  bool debug,
  const char * smoother_name,
  SmoothingFailureInfo * failure)
{
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (debug) std::cout << summary.FullReport() << std::endl;

  if (!summary.IsSolutionUsable())
    return throwOrStoreSmoothingFailure(failure, SmoothingFailureReason::SolverRejectedSolution,
      std::string(smoother_name) + " rejected the Ceres solution as unusable");
  if (summary.initial_cost - summary.final_cost < 0.0)
    return throwOrStoreSmoothingFailure(failure, SmoothingFailureReason::NoCostImprovement,
      std::string(smoother_name) + " did not improve the objective cost");
  return true;
}

}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__SOLVER_UTILS_HPP_
