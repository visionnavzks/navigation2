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
// limitations under the License. Reserved.

#ifndef CONSTRAINED_SMOOTHER__OPTIONS_HPP_
#define CONSTRAINED_SMOOTHER__OPTIONS_HPP_

#include <map>
#include <string>
#include <vector>
#include "ceres/ceres.h"
#include "constrained_smoother/astar_esdf.hpp"

namespace constrained_smoother
{

/**
 * @struct constrained_smoother::SmootherParams
 * @brief Parameters for the smoother cost function
 */
struct SmootherParams
{
  SmootherParams() {}

  double smooth_weight_sqrt{0.0};
  double costmap_weight_sqrt{0.0};
  double cusp_costmap_weight_sqrt{0.0};
  double cusp_zone_length{0.0};
  double distance_weight_sqrt{0.0};
  double curvature_weight_sqrt{0.0};
  double max_curvature{0.0};
  double max_time{10.0};
  double obstacle_safe_distance{0.5};
  double obstacle_decay_distance{0.25};
  double obstacle_reciprocal_epsilon{0.05};
  PlannerPenaltyType obstacle_penalty_type{PlannerPenaltyType::QuadraticHinge};
  int path_downsampling_factor{1};
  int path_upsampling_factor{1};
  bool reversing_enabled{true};
  bool keep_goal_orientation{true};
  bool keep_start_orientation{true};
  std::vector<double> cost_check_points{};
};

/**
 * @struct constrained_smoother::OptimizerParams
 * @brief Parameters for the ceres optimizer
 */
struct OptimizerParams
{
  OptimizerParams()
  : debug(false),
    linear_solver_type("SPARSE_NORMAL_CHOLESKY"),
    max_iterations(50),
    param_tol(1e-8),
    fn_tol(1e-6),
    gradient_tol(1e-10)
  {
  }

  const std::map<std::string, ceres::LinearSolverType> solver_types = {
    {"DENSE_QR", ceres::DENSE_QR},
    {"SPARSE_NORMAL_CHOLESKY", ceres::SPARSE_NORMAL_CHOLESKY}};

  bool debug;
  std::string linear_solver_type;
  int max_iterations;     // Ceres default: 50

  double param_tol;       // Ceres default: 1e-8
  double fn_tol;          // Ceres default: 1e-6
  double gradient_tol;    // Ceres default: 1e-10
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__OPTIONS_HPP_
