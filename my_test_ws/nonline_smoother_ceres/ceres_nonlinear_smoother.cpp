#include "ceres_nonlinear_smoother.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

#include "ceres/ceres.h"

namespace nonline_smoother
{
namespace
{

template<typename T>
T normalizeAngle(const T & angle)
{
  using std::atan2;
  using std::cos;
  using std::sin;
  return atan2(sin(angle), cos(angle));
}

template<typename T>
T safeSqrt(const T & value)
{
  using std::sqrt;
  return sqrt(value + T(1e-9));
}

struct AugmentedReference
{
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> theta;
  std::vector<double> gears;
  std::vector<bool> is_virtual;
};

AugmentedReference augmentReference(
  const std::vector<double> & x_ref,
  const std::vector<double> & y_ref,
  const std::vector<double> & theta_ref,
  const std::vector<double> & gears)
{
  if (x_ref.size() != y_ref.size() || x_ref.size() != theta_ref.size()) {
    throw std::invalid_argument("reference state vectors must have identical sizes");
  }
  if (x_ref.size() < 2) {
    throw std::invalid_argument("reference path must contain at least two states");
  }
  if (gears.size() != x_ref.size() - 1) {
    throw std::invalid_argument("gear vector must have size N-1");
  }

  AugmentedReference augmented;
  augmented.x.reserve(x_ref.size() + gears.size());
  augmented.y.reserve(y_ref.size() + gears.size());
  augmented.theta.reserve(theta_ref.size() + gears.size());
  augmented.gears.reserve(gears.size() * 2);
  augmented.is_virtual.reserve(gears.size() * 2);

  augmented.x.push_back(x_ref.front());
  augmented.y.push_back(y_ref.front());
  augmented.theta.push_back(theta_ref.front());

  for (std::size_t i = 0; i + 1 < x_ref.size(); ++i) {
    if (i > 0 && ((gears[i] >= 0.0) != (gears[i - 1] >= 0.0))) {
      augmented.x.push_back(x_ref[i]);
      augmented.y.push_back(y_ref[i]);
      augmented.theta.push_back(theta_ref[i]);
      augmented.gears.push_back(gears[i]);
      augmented.is_virtual.push_back(true);
    }

    augmented.x.push_back(x_ref[i + 1]);
    augmented.y.push_back(y_ref[i + 1]);
    augmented.theta.push_back(theta_ref[i + 1]);
    augmented.gears.push_back(gears[i]);
    augmented.is_virtual.push_back(false);
  }

  return augmented;
}

double estimateTargetSpacing(const AugmentedReference & ref, double requested_target_ds)
{
  if (requested_target_ds > 0.01) {
    return requested_target_ds;
  }

  double distance_sum = 0.0;
  int distance_count = 0;
  for (std::size_t i = 0; i < ref.gears.size(); ++i) {
    if (ref.is_virtual[i]) {
      continue;
    }
    const double dx = ref.x[i + 1] - ref.x[i];
    const double dy = ref.y[i + 1] - ref.y[i];
    const double distance = std::hypot(dx, dy);
    if (distance > 1e-4) {
      distance_sum += distance;
      ++distance_count;
    }
  }

  if (distance_count == 0) {
    throw std::runtime_error("failed to estimate target spacing from reference path");
  }
  return distance_sum / static_cast<double>(distance_count);
}

struct ReferenceCost
{
  ReferenceCost(double x_ref, double y_ref, double theta_ref, double w_ref)
  : x_ref_(x_ref), y_ref_(y_ref), theta_ref_(theta_ref), sqrt_w_ref_(std::sqrt(w_ref)),
    sqrt_2w_ref_(std::sqrt(2.0 * w_ref))
  {
  }

  template<typename T>
  bool operator()(const T * const pose, T * residuals) const
  {
    residuals[0] = T(sqrt_w_ref_) * (pose[0] - T(x_ref_));
    residuals[1] = T(sqrt_w_ref_) * (pose[1] - T(y_ref_));
    residuals[2] = T(sqrt_2w_ref_) * sin(T(0.5) * normalizeAngle(pose[2] - T(theta_ref_)));
    return true;
  }

  double x_ref_;
  double y_ref_;
  double theta_ref_;
  double sqrt_w_ref_;
  double sqrt_2w_ref_;
};

struct CurvatureCost
{
  explicit CurvatureCost(double w_kappa)
  : sqrt_w_kappa_(std::sqrt(w_kappa))
  {
  }

  template<typename T>
  bool operator()(const T * const kappa, T * residuals) const
  {
    residuals[0] = T(sqrt_w_kappa_) * kappa[0];
    return true;
  }

  double sqrt_w_kappa_;
};

struct TransitionCost
{
  TransitionCost(double gear, double dynamic_weight, double w_dkappa, double w_ds, double target_ds)
  : gear_(gear >= 0.0 ? 1.0 : -1.0),
    dynamic_weight_(dynamic_weight),
    sqrt_w_dkappa_(std::sqrt(w_dkappa)),
    sqrt_w_ds_(std::sqrt(w_ds)),
    target_ds_(target_ds)
  {
  }

  template<typename T>
  bool operator()(
    const T * const pose_i,
    const T * const kappa_i,
    const T * const pose_j,
    const T * const kappa_j,
    const T * const control,
    T * residuals) const
  {
    const T ds = control[0];
    const T dkappa = control[1];
    const T kappa_next = kappa_i[0] + ds * dkappa;
    const T theta_next = pose_i[2] + T(gear_) * (ds * kappa_i[0] + T(0.5) * ds * ds * dkappa);
    const T theta_mid = pose_i[2] + T(gear_) * (T(0.5) * ds * kappa_i[0] + T(0.125) * ds * ds * dkappa);
    const T x_next = pose_i[0] + T(gear_) * (ds / T(6.0)) *
      (cos(pose_i[2]) + T(4.0) * cos(theta_mid) + cos(theta_next));
    const T y_next = pose_i[1] + T(gear_) * (ds / T(6.0)) *
      (sin(pose_i[2]) + T(4.0) * sin(theta_mid) + sin(theta_next));

    residuals[0] = T(dynamic_weight_) * (pose_j[0] - x_next);
    residuals[1] = T(dynamic_weight_) * (pose_j[1] - y_next);
    residuals[2] = T(dynamic_weight_) * normalizeAngle(pose_j[2] - theta_next);
    residuals[3] = T(dynamic_weight_) * (kappa_j[0] - kappa_next);
    residuals[4] = T(sqrt_w_dkappa_) * dkappa * safeSqrt(ds);
    residuals[5] = T(sqrt_w_ds_) * (ds - T(target_ds_));
    return true;
  }

  double gear_;
  double dynamic_weight_;
  double sqrt_w_dkappa_;
  double sqrt_w_ds_;
  double target_ds_;
};

struct VirtualTransitionCost
{
  explicit VirtualTransitionCost(double dynamic_weight)
  : dynamic_weight_(dynamic_weight)
  {
  }

  template<typename T>
  bool operator()(const T * const pose_i, const T * const pose_j, const T * const control, T * residuals) const
  {
    residuals[0] = T(dynamic_weight_) * (pose_j[0] - pose_i[0]);
    residuals[1] = T(dynamic_weight_) * (pose_j[1] - pose_i[1]);
    residuals[2] = T(dynamic_weight_) * normalizeAngle(pose_j[2] - pose_i[2]);
    residuals[3] = T(dynamic_weight_) * control[0];
    return true;
  }

  double dynamic_weight_;
};

}  // namespace

CeresPathSmoother::CeresPathSmoother(SmootherParams params)
: params_(std::move(params))
{
  if (params_.num_threads <= 0) {
    params_.num_threads = std::max(1u, std::thread::hardware_concurrency());
  }
}

SmootherResult CeresPathSmoother::solve(
  const std::vector<double> & x_ref,
  const std::vector<double> & y_ref,
  const std::vector<double> & theta_ref,
  const std::vector<double> & gears) const
{
  AugmentedReference ref = augmentReference(x_ref, y_ref, theta_ref, gears);
  const double target_ds = estimateTargetSpacing(ref, params_.target_ds);
  const std::size_t state_count = ref.x.size();
  const std::size_t segment_count = ref.gears.size();

  std::vector<std::array<double, 3>> poses(state_count);
  std::vector<std::array<double, 2>> controls(segment_count);
  std::vector<double> kappas(state_count, 0.0);

  for (std::size_t i = 0; i < state_count; ++i) {
    poses[i] = {ref.x[i], ref.y[i], ref.theta[i]};
  }
  for (std::size_t i = 0; i < segment_count; ++i) {
    controls[i] = {ref.is_virtual[i] ? 0.0 : target_ds, 0.0};
  }

  ceres::Problem problem;

  for (std::size_t i = 0; i < state_count; ++i) {
    problem.AddParameterBlock(poses[i].data(), 3);
    problem.AddParameterBlock(&kappas[i], 1);

    auto * reference_cost =
      new ceres::AutoDiffCostFunction<ReferenceCost, 3, 3>(
      new ReferenceCost(ref.x[i], ref.y[i], ref.theta[i], params_.w_ref));
    problem.AddResidualBlock(reference_cost, nullptr, poses[i].data());

    auto * curvature_cost =
      new ceres::AutoDiffCostFunction<CurvatureCost, 1, 1>(new CurvatureCost(params_.w_kappa));
    problem.AddResidualBlock(curvature_cost, nullptr, &kappas[i]);
    problem.SetParameterLowerBound(&kappas[i], 0, -params_.max_kappa);
    problem.SetParameterUpperBound(&kappas[i], 0, params_.max_kappa);
  }

  for (std::size_t i = 0; i < segment_count; ++i) {
    problem.AddParameterBlock(controls[i].data(), 2);

    if (ref.is_virtual[i]) {
      controls[i][0] = 0.0;
      controls[i][1] = 0.0;
      problem.SetParameterBlockConstant(controls[i].data());
      auto * virtual_cost =
        new ceres::AutoDiffCostFunction<VirtualTransitionCost, 4, 3, 3, 2>(
        new VirtualTransitionCost(params_.dynamic_weight));
      problem.AddResidualBlock(virtual_cost, nullptr, poses[i].data(), poses[i + 1].data(), controls[i].data());
      continue;
    }

    problem.SetParameterLowerBound(controls[i].data(), 0, params_.ds_min_ratio * target_ds);
    problem.SetParameterUpperBound(controls[i].data(), 0, params_.ds_max_ratio * target_ds);

    auto * transition_cost =
      new ceres::AutoDiffCostFunction<TransitionCost, 6, 3, 1, 3, 1, 2>(
      new TransitionCost(ref.gears[i], params_.dynamic_weight, params_.w_dkappa, params_.w_ds, target_ds));
    problem.AddResidualBlock(
      transition_cost,
      nullptr,
      poses[i].data(),
      &kappas[i],
      poses[i + 1].data(),
      &kappas[i + 1],
      controls[i].data());
  }

  problem.SetParameterBlockConstant(poses.front().data());
  problem.SetParameterBlockConstant(poses.back().data());
  if (params_.has_kappa_start) {
    kappas.front() = params_.kappa_start;
    problem.SetParameterBlockConstant(&kappas.front());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = params_.max_num_iterations;
  options.num_threads = params_.num_threads;
  options.minimizer_progress_to_stdout = params_.verbose;
  options.logging_type = params_.verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;

  ceres::Solver::Summary summary;
  const auto start_time = std::chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);
  const auto end_time = std::chrono::steady_clock::now();

  SmootherResult result;
  result.success = summary.IsSolutionUsable();
  result.target_ds = target_ds;
  result.solve_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  result.final_cost = summary.final_cost;
  result.iterations = static_cast<int>(summary.iterations.size());
  result.brief_report = summary.BriefReport();
  result.is_virtual = ref.is_virtual;

  result.x.reserve(state_count);
  result.y.reserve(state_count);
  result.theta.reserve(state_count);
  result.kappa.reserve(state_count);
  result.ds.reserve(segment_count);
  result.dkappa.reserve(segment_count);
  result.gears.reserve(segment_count);

  for (std::size_t i = 0; i < state_count; ++i) {
    result.x.push_back(poses[i][0]);
    result.y.push_back(poses[i][1]);
    result.theta.push_back(poses[i][2]);
    result.kappa.push_back(kappas[i]);
  }
  for (std::size_t i = 0; i < segment_count; ++i) {
    result.ds.push_back(controls[i][0] * ref.gears[i]);
    result.dkappa.push_back(controls[i][1]);
    result.gears.push_back(ref.gears[i] >= 0.0 ? 1.0 : -1.0);
  }

  return result;
}

}  // namespace nonline_smoother