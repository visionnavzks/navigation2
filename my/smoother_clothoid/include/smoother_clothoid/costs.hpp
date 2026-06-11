#ifndef SMOOTHER_CLOTHOID__COSTS_HPP_
#define SMOOTHER_CLOTHOID__COSTS_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "Eigen/Core"

#include "smoother_clothoid/costmap2d.hpp"
#include "smoother_clothoid/options.hpp"

namespace smoother_clothoid
{
namespace detail
{

class TransitionCostFunctor
{
public:
  TransitionCostFunctor(
    double gear, bool is_cusp_segment,
    double model_weight, double curvature_weight, double curvature_rate_weight,
    double spacing_weight, double length_weight, double fix_weight,
    double target_spacing)
  : gear_(gear), is_cusp_segment_(is_cusp_segment),
    model_weight_(model_weight), curvature_weight_(curvature_weight),
    curvature_rate_weight_(curvature_rate_weight), spacing_weight_(spacing_weight),
    length_weight_(length_weight), fix_weight_(fix_weight),
    target_spacing_(target_spacing) {}

  static ceres::CostFunction * Create(
    double gear, bool is_cusp_segment,
    double model_weight, double curvature_weight, double curvature_rate_weight,
    double spacing_weight, double length_weight, double fix_weight,
    double target_spacing)
  {
    return new ceres::AutoDiffCostFunction<TransitionCostFunctor, 7, 5, 5>(
      new TransitionCostFunctor(
        gear, is_cusp_segment, model_weight, curvature_weight, curvature_rate_weight,
        spacing_weight, length_weight, fix_weight, target_spacing));
  }

  template<typename T>
  bool operator()(const T * const current, const T * const next, T * residuals) const
  {
    Eigen::Map<Eigen::Matrix<T, 7, 1>> residual(residuals);
    residual.setZero();

    const T x = current[0], y = current[1], theta = current[2], kappa = current[3], ds = current[4];
    const T next_x = next[0], next_y = next[1], next_theta = next[2], next_kappa = next[3];

    if (is_cusp_segment_) {
      residual[0] = T(fix_weight_) * (next_x - x);
      residual[1] = T(fix_weight_) * (next_y - y);
      residual[2] = T(fix_weight_) * angleDiff(next_theta, theta);
      residual[5] = T(spacing_weight_) * T(10.0) * ds;
      residual[6] = T(length_weight_) * ds;
      return true;
    }

    const T direction = gear_ >= 0.0 ? T(1.0) : T(-1.0);
    const T theta_pred = theta + direction * ds * (kappa + next_kappa) * T(0.5);
    const T theta_mid = (theta + theta_pred) * T(0.5);
    const T x_pred = x + direction * ds * cosValue(theta_mid);
    const T y_pred = y + direction * ds * sinValue(theta_mid);
    const T denom = ds > T(1e-3) ? sqrtValue(ds) : T(0.03);

    residual[0] = T(model_weight_) * (next_x - x_pred);
    residual[1] = T(model_weight_) * (next_y - y_pred);
    residual[2] = T(model_weight_) * angleDiff(next_theta, theta_pred);
    residual[3] = T(curvature_weight_) * (kappa + next_kappa) * T(0.5);
    residual[4] = T(curvature_rate_weight_) * (next_kappa - kappa) / denom;
    const T spacing_ref = T(std::max(target_spacing_, 1e-3));
    residual[5] = T(spacing_weight_) * (ds - spacing_ref) / spacing_ref;
    residual[6] = T(length_weight_) * ds;
    return true;
  }

private:
  template<typename T> static T normalizeAngle(T angle) { using std::atan2; using std::cos; using std::sin; return atan2(sin(angle), cos(angle)); }
  template<typename T> static T angleDiff(T a, T b) { return normalizeAngle(a - b); }
  template<typename T> static T sinValue(T v) { using std::sin; return sin(v); }
  template<typename T> static T cosValue(T v) { using std::cos; return cos(v); }
  template<typename T> static T sqrtValue(T v) { using std::sqrt; return sqrt(v); }

  double gear_, model_weight_, curvature_weight_, curvature_rate_weight_,
         spacing_weight_, length_weight_, fix_weight_, target_spacing_;
  bool is_cusp_segment_;
};

class BoundaryCostFunctor
{
public:
  BoundaryCostFunctor(
    const Eigen::Vector2d & ref, double target_theta, bool keep_orientation,
    double lon_tol, double lat_tol, double ori_tol, double fix_weight, bool constrain_stop)
  : reference_point_(ref), target_theta_(target_theta), keep_orientation_(keep_orientation),
    longitudinal_tolerance_(std::max(lon_tol, 0.0)), lateral_tolerance_(std::max(lat_tol, 0.0)),
    orientation_tolerance_(std::max(ori_tol, 0.0)), fix_weight_(fix_weight),
    constrain_stop_(constrain_stop) {}

  static ceres::CostFunction * Create(
    const Eigen::Vector2d & ref, double target_theta, bool keep_orientation,
    double lon_tol, double lat_tol, double ori_tol, double fix_weight, bool constrain_stop)
  {
    return new ceres::AutoDiffCostFunction<BoundaryCostFunctor, 4, 5>(
      new BoundaryCostFunctor(
        ref, target_theta, keep_orientation, lon_tol, lat_tol, ori_tol, fix_weight, constrain_stop));
  }

  template<typename T>
  bool operator()(const T * const state, T * residuals) const
  {
    using std::abs;
    const T dx = state[0] - T(reference_point_.x());
    const T dy = state[1] - T(reference_point_.y());
    const T cos_theta = T(std::cos(target_theta_));
    const T sin_theta = T(std::sin(target_theta_));
    const T lon_error = cos_theta * dx + sin_theta * dy;
    const T lat_error = -sin_theta * dx + cos_theta * dy;
    const T lon_violation = abs(lon_error) - T(longitudinal_tolerance_);
    const T lat_violation = abs(lat_error) - T(lateral_tolerance_);

    residuals[0] = lon_violation > T(0.0) ? T(fix_weight_) * lon_violation : T(0.0);
    residuals[1] = lat_violation > T(0.0) ? T(fix_weight_) * lat_violation : T(0.0);

    if (keep_orientation_) {
      const T heading_error = abs(angleDiff(state[2], T(target_theta_)));
      const T heading_violation = heading_error - T(orientation_tolerance_);
      residuals[2] = heading_violation > T(0.0) ? T(fix_weight_) * heading_violation : T(0.0);
    } else {
      residuals[2] = T(0.0);
    }
    residuals[3] = constrain_stop_ ? T(fix_weight_) * state[4] : T(0.0);
    return true;
  }

private:
  template<typename T> static T normalizeAngle(T angle) { using std::atan2; using std::cos; using std::sin; return atan2(sin(angle), cos(angle)); }
  template<typename T> static T angleDiff(T a, T b) { return normalizeAngle(a - b); }

  Eigen::Vector2d reference_point_;
  double target_theta_, longitudinal_tolerance_, lateral_tolerance_,
         orientation_tolerance_, fix_weight_;
  bool keep_orientation_, constrain_stop_;
};

class ReferenceCostFunctor
{
public:
  ReferenceCostFunctor(const Eigen::Vector2d & ref, double weight)
  : reference_point_(ref), reference_weight_(weight) {}

  static ceres::CostFunction * Create(const Eigen::Vector2d & ref, double weight)
  {
    return new ceres::AutoDiffCostFunction<ReferenceCostFunctor, 2, 5>(
      new ReferenceCostFunctor(ref, weight));
  }

  template<typename T>
  bool operator()(const T * const state, T * residuals) const
  {
    residuals[0] = T(reference_weight_) * (state[0] - T(reference_point_.x()));
    residuals[1] = T(reference_weight_) * (state[1] - T(reference_point_.y()));
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
    bool is_cusp_pose, const Costmap2D * costmap, const SmootherParams & params,
    const std::shared_ptr<ceres::Grid2D<double>> & grid,
    const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> & interp)
  : costmap_origin_(costmap->getOriginX(), costmap->getOriginY()),
    costmap_resolution_(costmap->getResolution()),
    size_x_(costmap->getSizeInCellsX()), size_y_(costmap->getSizeInCellsY()),
    obstacle_safe_distance_(std::max(params.obstacle_safe_distance, 1e-6)),
    cost_check_radius_(std::max(params.cost_check_radius, 0.0)),
    obstacle_weight_(std::max(params.costmap_weight_sqrt, 0.0)),
    cusp_obstacle_weight_(std::max(params.cusp_costmap_weight_sqrt, params.costmap_weight_sqrt)),
    is_cusp_pose_(is_cusp_pose), cost_check_points_(params.cost_check_points),
    esdf_grid_(grid), esdf_interpolator_(interp)
  {
    if (!cost_check_points_.empty() && cost_check_points_.size() % 3 != 0) {
      throw std::invalid_argument("cost_check_points size must be a multiple of 3");
    }
  }

  int numResiduals() const
  {
    return cost_check_points_.empty() ? 1 : static_cast<int>(cost_check_points_.size() / 3);
  }

  static ceres::CostFunction * Create(
    bool is_cusp_pose, const Costmap2D * costmap, const SmootherParams & params,
    const std::shared_ptr<ceres::Grid2D<double>> & grid,
    const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> & interp)
  {
    auto * functor = new ObstacleCostFunctor(is_cusp_pose, costmap, params, grid, interp);
    auto * cf = new ceres::DynamicAutoDiffCostFunction<ObstacleCostFunctor>(functor);
    cf->AddParameterBlock(5);
    cf->SetNumResiduals(functor->numResiduals());
    return cf;
  }

  template<typename T>
  bool operator()(const T * const * parameters, T * residuals) const
  {
    const T * state = parameters[0];
    const T x = state[0], y = state[1], theta = state[2];
    const T pose_weight = T(is_cusp_pose_ ? cusp_obstacle_weight_ : obstacle_weight_);

    if (cost_check_points_.empty()) {
      residuals[0] = pose_weight * obstaclePenalty(x, y);
      return true;
    }

    const T cos_theta = cosValue(theta);
    const T sin_theta = sinValue(theta);
    int idx = 0;
    for (size_t off = 0; off + 2 < cost_check_points_.size(); off += 3) {
      const T lx = T(cost_check_points_[off]);
      const T ly = T(cost_check_points_[off + 1]);
      const T pw = T(cost_check_points_[off + 2]);
      const T wx = x + cos_theta * lx - sin_theta * ly;
      const T wy = y + sin_theta * lx + cos_theta * ly;
      residuals[idx++] = pose_weight * pw * obstaclePenalty(wx, wy);
    }
    return true;
  }

private:
  template<typename T>
  T obstaclePenalty(T wx, T wy) const
  {
    const T gx = (wx - T(costmap_origin_.x())) / T(costmap_resolution_);
    const T gy = (wy - T(costmap_origin_.y())) / T(costmap_resolution_);
    if (gx < T(1.5) || gy < T(1.5) || gx >= T(size_x_) - T(1.5) || gy >= T(size_y_) - T(1.5))
      return T(1.0);
    T distance = T(0.0);
    esdf_interpolator_->Evaluate(gy - T(0.5), gx - T(0.5), &distance);
    const T surface_distance = distance - T(cost_check_radius_);
    if (surface_distance >= T(obstacle_safe_distance_)) return T(0.0);
    return (T(obstacle_safe_distance_) - surface_distance) / T(obstacle_safe_distance_);
  }

  template<typename T> static T sinValue(T v) { using std::sin; return sin(v); }
  template<typename T> static T cosValue(T v) { using std::cos; return cos(v); }

  Eigen::Vector2d costmap_origin_;
  double costmap_resolution_, obstacle_safe_distance_, cost_check_radius_,
         obstacle_weight_, cusp_obstacle_weight_;
  unsigned int size_x_, size_y_;
  bool is_cusp_pose_;
  std::vector<double> cost_check_points_;
  std::shared_ptr<ceres::Grid2D<double>> esdf_grid_;
  std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> esdf_interpolator_;
};

}  // namespace detail
}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__COSTS_HPP_
