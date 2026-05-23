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

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_BASE_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_BASE_HPP_

#include <string>

#include "ceres/ceres.h"

#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/options.hpp"
#include "constrained_smoother/solver_utils.hpp"

namespace constrained_smoother
{

class SolverBackedSmootherBase
{
protected:
  SolverBackedSmootherBase() = default;
  ~SolverBackedSmootherBase() = default;

  void initializeOptimizer(const OptimizerParams & params)
  {
    debug_ = params.debug;
    options_.linear_solver_type = params.solver_types.at(params.linear_solver_type);
    options_.max_num_iterations = params.max_iterations;
    options_.function_tolerance = params.fn_tol;
    options_.gradient_tolerance = params.gradient_tol;
    options_.parameter_tolerance = params.param_tol;
    options_.minimizer_progress_to_stdout = debug_;

    if (debug_) {
      options_.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
    } else {
      options_.logging_type = ceres::SILENT;
    }
  }

  template<typename PathT>
  void validateCommonInputs(
    const PathT & path,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const char * smoother_name) const
  {
    if (path.size() < 2) {
      throw InvalidPath(std::string(smoother_name) + ": Path must have at least 2 points");
    }
    if (params.obstacleTermsEnabled() && costmap == nullptr) {
      throw InvalidCostmap(std::string(smoother_name) + ": Costmap must not be null");
    }
  }

  void setMaxSolverTime(double max_time)
  {
    options_.max_solver_time_in_seconds = max_time;
  }

  bool solvePreparedProblem(
    ceres::Problem & problem,
    const char * smoother_name,
    SmoothingFailureInfo * failure) const
  {
    return solveProblemOrReportFailure(problem, options_, debug_, smoother_name, failure);
  }

  bool isDebugEnabled() const
  {
    return debug_;
  }

private:
  bool debug_{false};
  ceres::Solver::Options options_{};
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_BASE_HPP_