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

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_COST_FUNCTION_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_COST_FUNCTION_HPP_

#include <cmath>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <queue>
#include <utility>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "Eigen/Core"
#include "constrained_smoother/astar_esdf.hpp"
#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/options.hpp"
#include "constrained_smoother/utils.hpp"

namespace constrained_smoother
{

/**
 * @struct constrained_smoother::SmootherCostFunction
 * @brief Cost function for path smoothing with multiple terms
 * including curvature, smoothness, distance from original and obstacle avoidance.
 */
class SmootherCostFunction
{
public:
  /**
   * @brief A constructor for constrained_smoother::SmootherCostFunction
   * @param original_pos Original position of the path node
    * @param last_to_current_length_ratio Ratio of the previous segment length to the
    * current segment length. Negative if the previous/current transition crosses a cusp.
   * @param reversing Whether the path segment after this node represents reversing motion.
   * @param costmap A costmap to get values for collision and obstacle avoidance
   * @param esdf_interpolator Bicubic interpolator over ESDF grid
   * @param params Optimization weights and parameters
   * @param costmap_weight_sqrt Costmap cost weight (sqrt)
   */
  SmootherCostFunction(
    const Eigen::Vector2d & original_pos,
    double last_to_current_length_ratio,
    bool reversing,
    const Costmap2D * costmap,
    const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> & esdf_interpolator,
    const SmootherParams & params,
    double costmap_weight_sqrt)
  : original_pos_(original_pos),
    last_to_current_length_ratio_(last_to_current_length_ratio),
    reversing_(reversing),
    params_(params),
    costmap_weight_sqrt_(costmap_weight_sqrt),
    costmap_origin_(costmap->getOriginX(), costmap->getOriginY()),
    costmap_resolution_(costmap->getResolution()),
    esdf_interpolator_(esdf_interpolator)
  {
  }

  ceres::CostFunction * AutoDiff()
  {
    return new ceres::AutoDiffCostFunction<SmootherCostFunction, 6, 2, 2, 2>(this);
  }

  void setCostmapWeightSqrt(double costmap_weight_sqrt)
  {
    costmap_weight_sqrt_ = costmap_weight_sqrt;
  }

  double getCostmapWeightSqrt()
  {
    return costmap_weight_sqrt_;
  }

  /**
   * @brief Smoother cost function evaluation
   * @param pt X, Y coords of current point
   * @param pt_next X, Y coords of next point
   * @param pt_prev X, Y coords of previous point
   * @param pt_residual array of output residuals (smoothing, curvature, distance, cost)
   * @return if successful in computing values
   */
  template<typename T>
  bool operator()(
    const T * const pt, const T * const pt_next, const T * const pt_prev,
    T * pt_residual) const
  {
    Eigen::Map<const Eigen::Matrix<T, 2, 1>> xi(pt);
    Eigen::Map<const Eigen::Matrix<T, 2, 1>> xi_next(pt_next);
    Eigen::Map<const Eigen::Matrix<T, 2, 1>> xi_prev(pt_prev);
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(pt_residual);
    residual.setZero();

    // compute cost
    addSmoothingResidual<T>(
      params_.smooth_weight_sqrt, xi, xi_next, xi_prev, residual[0],
      residual[1]);
    addCurvatureResidual<T>(params_.curvature_weight_sqrt, xi, xi_next, xi_prev, residual[2]);
    addDistanceResidual<T>(
      params_.distance_weight_sqrt, xi,
      original_pos_.template cast<T>(), residual[3], residual[4]);
    addCostResidual<T>(costmap_weight_sqrt_, xi, xi_next, xi_prev, residual[5]);

    return true;
  }

protected:
  /**
   * @brief Cost function term for smooth paths
   */
  template<typename T>
  inline void addSmoothingResidual(
    const double & weight_sqrt,
    const Eigen::Matrix<T, 2, 1> & pt,
    const Eigen::Matrix<T, 2, 1> & pt_next,
    const Eigen::Matrix<T, 2, 1> & pt_prev,
    T & r1, T & r2) const
  {
    Eigen::Matrix<T, 2, 1> d_next = pt_next - pt;
    Eigen::Matrix<T, 2, 1> d_prev = pt - pt_prev;
    Eigen::Matrix<T, 2, 1> d_diff = last_to_current_length_ratio_ * d_next - d_prev;
    r1 += (T)weight_sqrt * d_diff(0, 0);
    r2 += (T)weight_sqrt * d_diff(1, 0);
  }

  /**
   * @brief Cost function term for maximum curved paths
   */
  template<typename T>
  inline void addCurvatureResidual(
    const double & weight_sqrt,
    const Eigen::Matrix<T, 2, 1> & pt,
    const Eigen::Matrix<T, 2, 1> & pt_next,
    const Eigen::Matrix<T, 2, 1> & pt_prev,
    T & r) const
  {
    Eigen::Matrix<T, 2, 1> center = arcCenter(
      pt_prev, pt, pt_next,
      last_to_current_length_ratio_ < 0);
    if (CERES_ISINF(center[0])) {
      return;
    }
    T turning_rad = (pt - center).norm();
    T ki_minus_kmax = (T)1.0 / turning_rad - params_.max_curvature;

    if (ki_minus_kmax <= (T)EPSILON) {
      return;
    }

    r += (T)weight_sqrt * ki_minus_kmax;
  }

  /**
   * @brief Cost function term for steering away changes in pose
   */
  template<typename T>
  inline void addDistanceResidual(
    const double & weight_sqrt,
    const Eigen::Matrix<T, 2, 1> & xi,
    const Eigen::Matrix<T, 2, 1> & xi_original,
    T & r1, T & r2) const
  {
    Eigen::Matrix<T, 2, 1> diff = xi - xi_original;
    r1 += (T)weight_sqrt * diff(0, 0);
    r2 += (T)weight_sqrt * diff(1, 0);
  }

  /**
   * @brief Cost function term for steering away from costs
   */
  template<typename T>
  inline void addCostResidual(
    const double & weight_sqrt,
    const Eigen::Matrix<T, 2, 1> & pt,
    const Eigen::Matrix<T, 2, 1> & pt_next,
    const Eigen::Matrix<T, 2, 1> & pt_prev,
    T & r) const
  {
    if (params_.cost_check_points.empty()) {
      Eigen::Matrix<T, 2, 1> interp_pos =
        (pt - costmap_origin_.template cast<T>()) / (T)costmap_resolution_;
      T distance;
      esdf_interpolator_->Evaluate(interp_pos[1] - (T)0.5, interp_pos[0] - (T)0.5, &distance);
      const T penalty = evaluateObstaclePenalty(distance);
      r += (T)weight_sqrt * penalty;
    } else {
      Eigen::Matrix<T, 2, 1> dir = tangentDir(
        pt_prev, pt, pt_next,
        last_to_current_length_ratio_ < 0);
      dir.normalize();
      if (((pt_next - pt).dot(dir) < (T)0) != reversing_) {
        dir = -dir;
      }
      Eigen::Matrix<T, 3, 3> transform;
      transform << dir[0], -dir[1], pt[0],
        dir[1], dir[0], pt[1],
        (T)0, (T)0, (T)1;
      for (size_t i = 0; i < params_.cost_check_points.size(); i += 3) {
        Eigen::Matrix<T, 3, 1> ccpt((T)params_.cost_check_points[i],
          (T)params_.cost_check_points[i + 1], (T)1);
        auto ccpt_world = (transform * ccpt).template block<2, 1>(0, 0);
        Eigen::Matrix<T, 2,
          1> interp_pos = (ccpt_world - costmap_origin_.template cast<T>()) /
          (T)costmap_resolution_;
        T distance;
        esdf_interpolator_->Evaluate(interp_pos[1] - (T)0.5, interp_pos[0] - (T)0.5, &distance);
        const T penalty = evaluateObstaclePenalty(distance);
        r += (T)weight_sqrt * (T)params_.cost_check_points[i + 2] * penalty;
      }
    }
  }

  template<typename T>
  inline T evaluateObstaclePenalty(const T & distance) const
  {
    const T safe_distance = (T)std::max(params_.obstacle_safe_distance, 1e-6);
    if (distance >= safe_distance) {
      return (T)0.0;
    }

    const T normalized_gap = (safe_distance - distance) / safe_distance;
    return normalized_gap * normalized_gap;
  }

  const Eigen::Vector2d original_pos_;
  double last_to_current_length_ratio_;
  bool reversing_;
  SmootherParams params_;
  double costmap_weight_sqrt_;
  Eigen::Vector2d costmap_origin_;
  double costmap_resolution_;
  std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> esdf_interpolator_;
};

/**
 * @brief D3 finite-difference proxy for curvature-rate smoothness.
 */
class CurvatureRateCostFunction
{
public:
  explicit CurvatureRateCostFunction(double weight_sqrt)
  : weight_sqrt_(weight_sqrt)
  {
  }

  ceres::CostFunction * AutoDiff()
  {
    return new ceres::AutoDiffCostFunction<CurvatureRateCostFunction, 2, 2, 2, 2, 2>(this);
  }

  template<typename T>
  bool operator()(
    const T * const pt_prev,
    const T * const pt,
    const T * const pt_next,
    const T * const pt_next2,
    T * pt_residual) const
  {
    Eigen::Map<const Eigen::Matrix<T, 2, 1>> xi_prev(pt_prev);
    Eigen::Map<const Eigen::Matrix<T, 2, 1>> xi(pt);
    Eigen::Map<const Eigen::Matrix<T, 2, 1>> xi_next(pt_next);
    Eigen::Map<const Eigen::Matrix<T, 2, 1>> xi_next2(pt_next2);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(pt_residual);

    residual = (T)weight_sqrt_ * (xi_next2 - (T)3.0 * xi_next + (T)3.0 * xi - xi_prev);
    return true;
  }

protected:
  double weight_sqrt_;
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_COST_FUNCTION_HPP_
