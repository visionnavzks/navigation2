// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0
//
// Ceres cost functors for the kinematic path smoother.
// Each functor defines a residual block for the least-squares problem.

#ifndef KINEMATIC_SMOOTHER__COST_FUNCTIONS_HPP_
#define KINEMATIC_SMOOTHER__COST_FUNCTIONS_HPP_

#include <cmath>
#include <memory>
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "Eigen/Core"

namespace kinematic_smoother
{

// ---------------------------------------------------------------------------
// 1. KinematicDefectCost
//    Enforces the bicycle-model kinematic constraints between consecutive
//    (non-virtual) path points.
//    Parameter blocks: state_i[4], state_ip1[4], control_i[2]
//    Residuals: 4  (x, y, theta, kappa defects)
// ---------------------------------------------------------------------------
struct KinematicDefectCost
{
  double gear;     // +1.0 (forward) or -1.0 (reverse)
  double weight;   // penalty weight (sqrt)

  KinematicDefectCost(double gear, double weight)
  : gear(gear), weight(weight) {}

  template<typename T>
  bool operator()(const T * const state_i,
                  const T * const state_ip1,
                  const T * const control_i,
                  T * residual) const
  {
    // State: [x, y, theta, kappa]
    const T & x_i     = state_i[0];
    const T & y_i     = state_i[1];
    const T & theta_i = state_i[2];
    const T & kappa_i = state_i[3];

    const T & x_ip1     = state_ip1[0];
    const T & y_ip1     = state_ip1[1];
    const T & theta_ip1 = state_ip1[2];
    const T & kappa_ip1 = state_ip1[3];

    // Control: [ds, dkappa]
    const T & ds_i     = control_i[0];
    const T & dkappa_i = control_i[1];

    const T g = T(gear);

    // Curvature propagation
    T kappa_next = kappa_i + ds_i * dkappa_i;

    // Heading propagation (exact integration for piecewise-constant dkappa)
    T theta_next = theta_i + g * (ds_i * kappa_i + T(0.5) * ds_i * ds_i * dkappa_i);

    // Simpson's rule for position integration
    T theta_mid = theta_i + g * (T(0.5) * ds_i * kappa_i +
                                  T(0.125) * ds_i * ds_i * dkappa_i);
    T x_next = x_i + g * (ds_i / T(6.0)) *
      (ceres::cos(theta_i) + T(4.0) * ceres::cos(theta_mid) + ceres::cos(theta_next));
    T y_next = y_i + g * (ds_i / T(6.0)) *
      (ceres::sin(theta_i) + T(4.0) * ceres::sin(theta_mid) + ceres::sin(theta_next));

    const T w = T(weight);
    residual[0] = w * (x_ip1 - x_next);
    residual[1] = w * (y_ip1 - y_next);
    residual[2] = w * (theta_ip1 - theta_next);
    residual[3] = w * (kappa_ip1 - kappa_next);

    return true;
  }

  static ceres::CostFunction * Create(double gear, double weight)
  {
    return new ceres::AutoDiffCostFunction<KinematicDefectCost, 4, 4, 4, 2>(
      new KinematicDefectCost(gear, weight));
  }
};

// ---------------------------------------------------------------------------
// 2. VirtualSegmentCost
//    At cusps, x/y/theta must match and ds must be 0.
//    Parameter blocks: state_i[4], state_ip1[4]
//    Residuals: 3  (x, y, theta equality)
// ---------------------------------------------------------------------------
struct VirtualSegmentCost
{
  double weight;

  explicit VirtualSegmentCost(double weight) : weight(weight) {}

  template<typename T>
  bool operator()(const T * const state_i,
                  const T * const state_ip1,
                  T * residual) const
  {
    const T w = T(weight);
    residual[0] = w * (state_ip1[0] - state_i[0]);
    residual[1] = w * (state_ip1[1] - state_i[1]);
    residual[2] = w * (state_ip1[2] - state_i[2]);
    return true;
  }

  static ceres::CostFunction * Create(double weight)
  {
    return new ceres::AutoDiffCostFunction<VirtualSegmentCost, 3, 4, 4>(
      new VirtualSegmentCost(weight));
  }
};

// ---------------------------------------------------------------------------
// 3. ReferenceTrackingCost
//    Penalises deviation from reference position and heading.
//    Parameter blocks: state_i[4]
//    Residuals: 3  (dx, dy, heading_error)
// ---------------------------------------------------------------------------
struct ReferenceTrackingCost
{
  double x_ref, y_ref, theta_ref;
  double weight_xy;
  double weight_theta;

  ReferenceTrackingCost(double x_ref, double y_ref, double theta_ref,
                        double weight_xy, double weight_theta)
  : x_ref(x_ref), y_ref(y_ref), theta_ref(theta_ref),
    weight_xy(weight_xy), weight_theta(weight_theta) {}

  template<typename T>
  bool operator()(const T * const state, T * residual) const
  {
    residual[0] = T(weight_xy) * (state[0] - T(x_ref));
    residual[1] = T(weight_xy) * (state[1] - T(y_ref));
    // 1 - cos(delta) is a smooth, wrapping-safe heading error
    residual[2] = T(weight_theta) * (T(1.0) - ceres::cos(state[2] - T(theta_ref)));
    return true;
  }

  static ceres::CostFunction * Create(double x_ref, double y_ref, double theta_ref,
                                       double weight_xy, double weight_theta)
  {
    return new ceres::AutoDiffCostFunction<ReferenceTrackingCost, 3, 4>(
      new ReferenceTrackingCost(x_ref, y_ref, theta_ref, weight_xy, weight_theta));
  }
};

// ---------------------------------------------------------------------------
// 4. SmoothnessCost
//    Penalises large curvature-rate (dkappa), weighted by arc length.
//    Parameter blocks: control_i[2]
//    Residuals: 1
// ---------------------------------------------------------------------------
struct SmoothnessCost
{
  double weight;

  explicit SmoothnessCost(double weight) : weight(weight) {}

  template<typename T>
  bool operator()(const T * const control, T * residual) const
  {
    // control = [ds, dkappa]
    // objective contribution:  w * dkappa^2 * ds
    // residual:  sqrt(w) * dkappa * sqrt(ds + eps)
    T ds_safe = control[0] + T(1e-8);
    residual[0] = T(weight) * control[1] * ceres::sqrt(ds_safe);
    return true;
  }

  static ceres::CostFunction * Create(double weight)
  {
    return new ceres::AutoDiffCostFunction<SmoothnessCost, 1, 2>(
      new SmoothnessCost(weight));
  }
};

// ---------------------------------------------------------------------------
// 5. CurvatureCost
//    Penalises large curvature magnitude.
//    Parameter blocks: state_i[4]
//    Residuals: 1
// ---------------------------------------------------------------------------
struct CurvatureCost
{
  double weight;

  explicit CurvatureCost(double weight) : weight(weight) {}

  template<typename T>
  bool operator()(const T * const state, T * residual) const
  {
    residual[0] = T(weight) * state[3];   // kappa
    return true;
  }

  static ceres::CostFunction * Create(double weight)
  {
    return new ceres::AutoDiffCostFunction<CurvatureCost, 1, 4>(
      new CurvatureCost(weight));
  }
};

// ---------------------------------------------------------------------------
// 6. StepSizeUniformityCost
//    Penalises deviation from the target step size.
//    Parameter blocks: control_i[2]
//    Residuals: 1
// ---------------------------------------------------------------------------
struct StepSizeUniformityCost
{
  double target_ds;
  double weight;

  StepSizeUniformityCost(double target_ds, double weight)
  : target_ds(target_ds), weight(weight) {}

  template<typename T>
  bool operator()(const T * const control, T * residual) const
  {
    residual[0] = T(weight) * (control[0] - T(target_ds));
    return true;
  }

  static ceres::CostFunction * Create(double target_ds, double weight)
  {
    return new ceres::AutoDiffCostFunction<StepSizeUniformityCost, 1, 2>(
      new StepSizeUniformityCost(target_ds, weight));
  }
};

// ---------------------------------------------------------------------------
// 7. ESDFCost
//    Penalises path points that are too close to obstacles using ESDF.
//    Uses BiCubicInterpolator on an ESDF grid for smooth gradients.
//    Parameter blocks: state_i[4]
//    Residuals: 1
// ---------------------------------------------------------------------------
struct ESDFCost
{
  using Grid = ceres::Grid2D<double>;
  using Interpolator = ceres::BiCubicInterpolator<Grid>;

  double weight;
  double safe_distance;
  double origin_x, origin_y;
  double resolution;
  const Interpolator * interpolator;   // non-owning; must outlive this cost

  ESDFCost(double weight, double safe_distance,
           double origin_x, double origin_y, double resolution,
           const Interpolator * interpolator)
  : weight(weight), safe_distance(safe_distance),
    origin_x(origin_x), origin_y(origin_y),
    resolution(resolution), interpolator(interpolator) {}

  template<typename T>
  bool operator()(const T * const state, T * residual) const
  {
    // Convert world coords to grid coords
    T gx = (state[0] - T(origin_x)) / T(resolution);
    T gy = (state[1] - T(origin_y)) / T(resolution);

    T dist;
    interpolator->Evaluate(gy - T(0.5), gx - T(0.5), &dist);

    // Smooth penalty: max(0, safe_distance - dist)
    // Use softplus for differentiability: log(1 + exp(k * violation)) / k
    T violation = T(safe_distance) - dist;
    const T k = T(10.0);   // sharpness
    residual[0] = T(weight) * ceres::log(T(1.0) + ceres::exp(k * violation)) / k;
    return true;
  }

  static ceres::CostFunction * Create(
    double weight, double safe_distance,
    double origin_x, double origin_y, double resolution,
    const Interpolator * interpolator)
  {
    return new ceres::AutoDiffCostFunction<ESDFCost, 1, 4>(
      new ESDFCost(weight, safe_distance, origin_x, origin_y, resolution, interpolator));
  }
};

}  // namespace kinematic_smoother

#endif  // KINEMATIC_SMOOTHER__COST_FUNCTIONS_HPP_
