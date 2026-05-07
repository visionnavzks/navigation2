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

#ifndef CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_
#define CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "Eigen/Core"

#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/options.hpp"

namespace constrained_smoother
{
namespace kinematic_smoother_detail
{

class TransitionCostFunctor
{
public:
  TransitionCostFunctor(
    double gear,
    bool is_cusp_segment,
    double model_weight,
    double curvature_weight,
    double curvature_rate_weight,
    double spacing_weight,
    double fix_weight,
    double target_spacing)
  : gear_(gear),
    is_cusp_segment_(is_cusp_segment),
    model_weight_(model_weight),
    curvature_weight_(curvature_weight),
    curvature_rate_weight_(curvature_rate_weight),
    spacing_weight_(spacing_weight),
    fix_weight_(fix_weight),
    target_spacing_(target_spacing)
  {
  }

  ceres::CostFunction * AutoDiff()
  {
    return new ceres::AutoDiffCostFunction<TransitionCostFunctor, 6, 5, 5>(this);
  }

  template<typename T>
  bool operator()(const T * const current, const T * const next, T * residuals) const
  {
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residuals);
    residual.setZero();

    const T x = current[0];
    const T y = current[1];
    const T theta = current[2];
    const T kappa = current[3];
    const T ds = current[4];

    const T next_x = next[0];
    const T next_y = next[1];
    const T next_theta = next[2];
    const T next_kappa = next[3];

    if (is_cusp_segment_) {
      residual[0] = T(fix_weight_) * (next_x - x);
      residual[1] = T(fix_weight_) * (next_y - y);
      residual[2] = T(fix_weight_) * angleDiff(next_theta, theta);
      residual[5] = T(spacing_weight_) * T(10.0) * ds;
      return true;
    }

    const T direction = gear_ >= 0.0 ? T(1.0) : T(-1.0);
    const T theta_pred = theta + direction * ds * (kappa + next_kappa) * T(0.5);
    const T theta_mid = theta + direction * ds * kappa * T(0.5);
    const T x_pred = x + direction * ds * cosValue(theta_mid);
    const T y_pred = y + direction * ds * sinValue(theta_mid);
    const T denom = ds > T(1e-3) ? sqrtValue(ds) : T(0.03);

    residual[0] = T(model_weight_) * (next_x - x_pred);
    residual[1] = T(model_weight_) * (next_y - y_pred);
    residual[2] = T(model_weight_) * angleDiff(next_theta, theta_pred);
    residual[3] = T(curvature_weight_) * (kappa + next_kappa) * T(0.5);
    residual[4] = T(curvature_rate_weight_) * (next_kappa - kappa) / denom;
    residual[5] = T(spacing_weight_) * (ds - T(target_spacing_)) / T(target_spacing_);
    return true;
  }

private:
  template<typename T>
  static T normalizeAngle(T angle)
  {
    using std::atan2;
    using std::cos;
    using std::sin;
    return atan2(sin(angle), cos(angle));
  }

  template<typename T>
  static T angleDiff(T a, T b)
  {
    return normalizeAngle(a - b);
  }

  template<typename T>
  static T sinValue(T value)
  {
    using std::sin;
    return sin(value);
  }

  template<typename T>
  static T cosValue(T value)
  {
    using std::cos;
    return cos(value);
  }

  template<typename T>
  static T sqrtValue(T value)
  {
    using std::sqrt;
    return sqrt(value);
  }

  double gear_;
  bool is_cusp_segment_;
  double model_weight_;
  double curvature_weight_;
  double curvature_rate_weight_;
  double spacing_weight_;
  double fix_weight_;
  double target_spacing_;
};

class BoundaryCostFunctor
{
public:
  BoundaryCostFunctor(
    const Eigen::Vector2d & reference_point,
    double target_theta,
    bool keep_orientation,
    double fix_weight,
    bool constrain_stop)
  : reference_point_(reference_point),
    target_theta_(target_theta),
    keep_orientation_(keep_orientation),
    fix_weight_(fix_weight),
    constrain_stop_(constrain_stop)
  {
  }

  ceres::CostFunction * AutoDiff()
  {
    return new ceres::AutoDiffCostFunction<BoundaryCostFunctor, 4, 5>(this);
  }

  template<typename T>
  bool operator()(const T * const state, T * residuals) const
  {
    residuals[0] = T(fix_weight_) * (state[0] - T(reference_point_.x()));
    residuals[1] = T(fix_weight_) * (state[1] - T(reference_point_.y()));
    residuals[2] =
      keep_orientation_ ? T(fix_weight_) * angleDiff(state[2], T(target_theta_)) : T(0.0);
    residuals[3] = constrain_stop_ ? T(fix_weight_) * state[4] : T(0.0);
    return true;
  }

private:
  template<typename T>
  static T normalizeAngle(T angle)
  {
    using std::atan2;
    using std::cos;
    using std::sin;
    return atan2(sin(angle), cos(angle));
  }

  template<typename T>
  static T angleDiff(T a, T b)
  {
    return normalizeAngle(a - b);
  }

  Eigen::Vector2d reference_point_;
  double target_theta_;
  bool keep_orientation_;
  double fix_weight_;
  bool constrain_stop_;
};

class ReferenceCostFunctor
{
public:
  ReferenceCostFunctor(const Eigen::Vector2d & reference_point, double reference_weight)
  : reference_point_(reference_point), reference_weight_(reference_weight)
  {
  }

  ceres::CostFunction * AutoDiff()
  {
    return new ceres::AutoDiffCostFunction<ReferenceCostFunctor, 2, 5>(this);
  }

  template<typename T>
  bool operator()(const T * const state, T * residuals) const
  {
    const T dx = state[0] - T(reference_point_.x());
    const T dy = state[1] - T(reference_point_.y());
    residuals[0] = T(reference_weight_) * dx;
    residuals[1] = T(reference_weight_) * dy;
    return true;
  }

private:
  Eigen::Vector2d reference_point_;
  double reference_weight_;
};

class ObstacleCostFunctor
{
public:
  ObstacleCostFunctor(
    bool is_cusp_pose,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> & esdf_values)
  : costmap_origin_(costmap->getOriginX(), costmap->getOriginY()),
    costmap_resolution_(costmap->getResolution()),
    size_x_(costmap->getSizeInCellsX()),
    size_y_(costmap->getSizeInCellsY()),
    obstacle_safe_distance_(std::max(params.obstacle_safe_distance, 1e-6)),
    cost_check_radius_(std::max(params.cost_check_radius, 0.0)),
    obstacle_weight_(std::max(params.costmap_weight_sqrt, 0.0)),
    cusp_obstacle_weight_(std::max(params.cusp_costmap_weight_sqrt, params.costmap_weight_sqrt)),
    is_cusp_pose_(is_cusp_pose),
    cost_check_points_(params.cost_check_points),
    esdf_grid_(std::make_shared<ceres::Grid2D<double>>(esdf_values.data(), 0, size_y_, 0, size_x_)),
    esdf_interpolator_(
      std::make_shared<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>(*esdf_grid_))
  {
  }

  int numResiduals() const
  {
    return cost_check_points_.empty() ? 1 : static_cast<int>(cost_check_points_.size() / 3);
  }

  ceres::CostFunction * AutoDiff()
  {
    auto * cost_function = new ceres::DynamicAutoDiffCostFunction<ObstacleCostFunctor>(this);
    cost_function->AddParameterBlock(5);
    cost_function->SetNumResiduals(numResiduals());
    return cost_function;
  }

  template<typename T>
  bool operator()(const T * const * parameters, T * residuals) const
  {
    const T * state = parameters[0];
    const T x = state[0];
    const T y = state[1];
    const T theta = state[2];
    const T pose_weight = T(is_cusp_pose_ ? cusp_obstacle_weight_ : obstacle_weight_);

    if (cost_check_points_.empty()) {
      residuals[0] = pose_weight * obstaclePenalty(x, y);
      return true;
    }

    const T cos_theta = cosValue(theta);
    const T sin_theta = sinValue(theta);
    int residual_index = 0;
    for (size_t offset = 0; offset + 2 < cost_check_points_.size(); offset += 3) {
      const T local_x = T(cost_check_points_[offset + 0]);
      const T local_y = T(cost_check_points_[offset + 1]);
      const T point_weight = T(cost_check_points_[offset + 2]);
      const T world_x = x + cos_theta * local_x - sin_theta * local_y;
      const T world_y = y + sin_theta * local_x + cos_theta * local_y;
      residuals[residual_index++] = pose_weight * point_weight * obstaclePenalty(world_x, world_y);
    }
    return true;
  }

private:
  template<typename T>
  T obstaclePenalty(T world_x, T world_y) const
  {
    const T grid_x = (world_x - T(costmap_origin_.x())) / T(costmap_resolution_);
    const T grid_y = (world_y - T(costmap_origin_.y())) / T(costmap_resolution_);
    if (grid_x < T(0.0) || grid_y < T(0.0) ||
      grid_x >= T(static_cast<double>(size_x_)) || grid_y >= T(static_cast<double>(size_y_)))
    {
      return T(1.0);
    }

    T distance = T(0.0);
    esdf_interpolator_->Evaluate(grid_y - T(0.5), grid_x - T(0.5), &distance);
    const T surface_distance = distance - T(cost_check_radius_);
    if (surface_distance >= T(obstacle_safe_distance_)) {
      return T(0.0);
    }

    const T normalized_gap =
      (T(obstacle_safe_distance_) - surface_distance) / T(obstacle_safe_distance_);
    return normalized_gap * normalized_gap;
  }

  template<typename T>
  static T sinValue(T value)
  {
    using std::sin;
    return sin(value);
  }

  template<typename T>
  static T cosValue(T value)
  {
    using std::cos;
    return cos(value);
  }

  Eigen::Vector2d costmap_origin_;
  double costmap_resolution_;
  unsigned int size_x_;
  unsigned int size_y_;
  double obstacle_safe_distance_;
  double cost_check_radius_;
  double obstacle_weight_;
  double cusp_obstacle_weight_;
  bool is_cusp_pose_;
  std::vector<double> cost_check_points_;
  std::shared_ptr<ceres::Grid2D<double>> esdf_grid_;
  std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> esdf_interpolator_;
};

}  // namespace kinematic_smoother_detail
}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_