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

#ifndef CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_
#define CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_

#include <algorithm>
#include <cmath>
#include <vector>

#include "ceres/ceres.h"
#include "Eigen/Core"

#include "constrained_smoother/esdf.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/kinematic_smoother_costs.hpp"
#include "constrained_smoother/options.hpp"

namespace constrained_smoother
{

/// 运动学版 smoother 展开的内部状态链。
///
/// 与几何版直接优化路径点不同，这里会把参考路径扩展成显式的
/// (x, y, theta, kappa, ds) 状态序列，并保留 cusp 段的附加元数据。
struct KinematicProcessedPath
{
  std::vector<Eigen::Vector2d> reference_points{};
  std::vector<double> gears{};
  std::vector<bool> is_cusp_segment{};
  std::vector<double> initial_variables{};
  size_t state_count{0};
  double start_theta{0.0};
  double end_theta{0.0};
  double target_spacing{0.2};
};

/// 运动学版 smoother 的问题构建器。
///
/// 它把 ESDF 准备、状态展开、残差拼接、显式边界约束和结果解包集中在一起，
/// 让顶层 smoother 只保留编排与后验校验职责。
class KinematicSmootherProblemBuilder
{
public:
  explicit KinematicSmootherProblemBuilder(std::vector<double> & esdf_values)
  : esdf_values_(esdf_values)
  {
  }

  void initializeEsdfValues(
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
    if (!params.obstacleTermsEnabled()) {
      esdf_values_.clear();
      return;
    }

    // 这份 ESDF 会被障碍物残差和最终后验校验共同复用。
    const size_t expected_esdf_size =
      static_cast<size_t>(costmap->getSizeInCellsX()) * costmap->getSizeInCellsY();
    if (precomputed_esdf != nullptr) {
      if (precomputed_esdf->size() != expected_esdf_size) {
        throw PrecomputedEsdfSizeMismatch(
                "Precomputed ESDF size does not match costmap dimensions");
      }
      esdf_values_ = *precomputed_esdf;
    } else {
      esdf_values_ = ESDF::ComputeESDF(
        costmap,
        Costmap2D::LETHAL_OBSTACLE,
        params.use_exact_esdf ? ESDFAlgorithm::Exact : ESDFAlgorithm::Approximate);
    }
  }

  static KinematicProcessedPath buildProcessedPath(
    const std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params,
    const Costmap2D * costmap)
  {
    // 先展开 gear / cusp 元数据，再把参考几何转成求解器可用的状态初值。
    KinematicProcessedPath processed;
    processed.start_theta = std::atan2(start_dir.y(), start_dir.x());
    processed.end_theta = std::atan2(end_dir.y(), end_dir.x());

    std::vector<double> gear_directions;
    gear_directions.reserve(path.size() - 1);
    for (size_t index = 0; index + 1 < path.size(); ++index) {
      if (params.reversing_enabled) {
        gear_directions.push_back(path[index].z() < 0.0 ? -1.0 : 1.0);
      } else {
        gear_directions.push_back(1.0);
      }
    }

    processed.reference_points.emplace_back(path.front().x(), path.front().y());
    for (size_t index = 0; index + 1 < path.size(); ++index) {
      const double current_gear = gear_directions[index];
      const double next_gear = index + 1 < gear_directions.size() ? gear_directions[index + 1] : current_gear;

      processed.gears.push_back(current_gear);
      processed.is_cusp_segment.push_back(false);
      processed.reference_points.emplace_back(path[index + 1].x(), path[index + 1].y());

      if (index + 2 < path.size() && current_gear != next_gear) {
        processed.gears.push_back(0.0);
        processed.is_cusp_segment.push_back(true);
        processed.reference_points.emplace_back(path[index + 1].x(), path[index + 1].y());
      }
    }

    processed.state_count = processed.reference_points.size();
    std::vector<double> theta(processed.state_count, 0.0);
    std::vector<double> kappa(processed.state_count, 0.0);
    std::vector<double> ds(processed.state_count, 0.0);

    double spacing_sum = 0.0;
    size_t spacing_count = 0;
    for (size_t index = 0; index + 1 < processed.state_count; ++index) {
      const Eigen::Vector2d delta = processed.reference_points[index + 1] - processed.reference_points[index];
      const double segment_norm = delta.norm();
      if (processed.is_cusp_segment[index]) {
        theta[index] = index > 0 ? theta[index - 1] : processed.start_theta;
        ds[index] = 0.0;
        continue;
      }

      if (segment_norm > 1e-6) {
        double heading = std::atan2(delta.y(), delta.x());
        if (processed.gears[index] < 0.0) {
          heading += M_PI;
        }
        theta[index] = normalizeAngle(heading);
        ds[index] = segment_norm;
        spacing_sum += segment_norm;
        ++spacing_count;
      } else {
        theta[index] = index > 0 ? theta[index - 1] : processed.start_theta;
      }
    }

    theta.back() = theta.size() > 1 ? theta[theta.size() - 2] : processed.start_theta;
    if (params.keep_start_orientation) {
      theta.front() = processed.start_theta;
    }
    if (params.keep_goal_orientation) {
      theta.back() = processed.end_theta;
    }

    processed.target_spacing = spacing_count > 0 ?
      spacing_sum / static_cast<double>(spacing_count) :
      (costmap != nullptr ? std::max(costmap->getResolution(), 1e-3) : processed.target_spacing);

    processed.initial_variables.reserve(processed.state_count * 5);
    for (size_t index = 0; index < processed.state_count; ++index) {
      processed.initial_variables.push_back(processed.reference_points[index].x());
      processed.initial_variables.push_back(processed.reference_points[index].y());
      processed.initial_variables.push_back(theta[index]);
      processed.initial_variables.push_back(kappa[index]);
      processed.initial_variables.push_back(ds[index]);
    }

    return processed;
  }

  void buildProblem(
    const KinematicProcessedPath & processed,
    const Costmap2D * costmap,
    const SmootherParams & params,
    std::vector<double> & variables,
    ceres::Problem & problem) const
  {
    // 调用方必须先用 buildProcessedPath() 生成 processed，并把 variables 初始化为状态初值。
    const double model_weight = std::max(params.model_weight_sqrt, 1.0);
    const double curvature_weight = std::max(params.kinematic_curvature_weight_sqrt, 0.0);
    const double curvature_rate_weight =
      std::max(params.kinematic_curvature_rate_weight_sqrt, 0.0);
    const double spacing_weight = 1.0;
    const double fix_weight = 100.0;
    const double reference_weight = std::max(params.distance_weight_sqrt, 0.0);
    const bool has_obstacle_cost = params.obstacleTermsEnabled();

    for (size_t index = 0; index + 1 < processed.state_count; ++index) {
      auto * transition_cost = new kinematic_smoother_detail::TransitionCostFunctor(
        processed.gears[index],
        processed.is_cusp_segment[index],
        model_weight,
        curvature_weight,
        curvature_rate_weight,
        spacing_weight,
        fix_weight,
        processed.target_spacing);
      problem.AddResidualBlock(
        transition_cost->AutoDiff(),
        nullptr,
        stateData(variables, index),
        stateData(variables, index + 1));
    }

    auto * start_boundary_cost = new kinematic_smoother_detail::BoundaryCostFunctor(
      processed.reference_points.front(),
      processed.start_theta,
      params.keep_start_orientation,
      0.0,
      0.0,
      0.0,
      fix_weight,
      false);
    problem.AddResidualBlock(start_boundary_cost->AutoDiff(), nullptr, stateData(variables, 0));

    auto * goal_boundary_cost = new kinematic_smoother_detail::BoundaryCostFunctor(
      processed.reference_points.back(),
      processed.end_theta,
      params.keep_goal_orientation,
      params.goal_longitudinal_tolerance,
      params.goal_lateral_tolerance,
      params.goal_orientation_tolerance,
      fix_weight,
      true);
    problem.AddResidualBlock(
      goal_boundary_cost->AutoDiff(),
      nullptr,
      stateData(variables, processed.state_count - 1));

    if (reference_weight > 1e-9) {
      for (size_t index = 0; index < processed.state_count; ++index) {
        auto * reference_cost = new kinematic_smoother_detail::ReferenceCostFunctor(
          processed.reference_points[index], reference_weight);
        problem.AddResidualBlock(reference_cost->AutoDiff(), nullptr, stateData(variables, index));
      }
    }

    if (has_obstacle_cost) {
      for (size_t index = 0; index < processed.state_count; ++index) {
        const bool is_cusp_pose =
          (index < processed.is_cusp_segment.size() && processed.is_cusp_segment[index]) ||
          (index > 0 && processed.is_cusp_segment[index - 1]);
        auto * obstacle_cost = new kinematic_smoother_detail::ObstacleCostFunctor(
          is_cusp_pose, costmap, params, esdf_values_);
        problem.AddResidualBlock(obstacle_cost->AutoDiff(), nullptr, stateData(variables, index));
      }
    }
  }

  static void applyBounds(
    ceres::Problem & problem,
    double * variables,
    size_t state_count,
    double max_curvature)
  {
    const double clamped_max_curvature = std::max(max_curvature, 1e-6);
    for (size_t index = 0; index < state_count; ++index) {
      double * state = variables + 5 * index;
      problem.SetParameterLowerBound(state, 3, -clamped_max_curvature);
      problem.SetParameterUpperBound(state, 3, clamped_max_curvature);
      problem.SetParameterLowerBound(state, 4, 0.0);
    }
  }

  static std::vector<Eigen::Vector3d> unpackPath(const std::vector<double> & variables, size_t state_count)
  {
    std::vector<Eigen::Vector3d> path;
    path.reserve(state_count);
    for (size_t index = 0; index < state_count; ++index) {
      path.emplace_back(
        variables[5 * index + 0],
        variables[5 * index + 1],
        normalizeAngle(variables[5 * index + 2]));
    }
    return path;
  }

private:
  static double normalizeAngle(double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  static double * stateData(std::vector<double> & variables, size_t index)
  {
    return variables.data() + 5 * index;
  }

  std::vector<double> & esdf_values_;
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_