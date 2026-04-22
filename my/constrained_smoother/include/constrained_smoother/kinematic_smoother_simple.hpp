#ifndef CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_SIMPLE_HPP_
#define CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_SIMPLE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "Eigen/Core"

#include "constrained_smoother/esdf.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/options.hpp"

namespace constrained_smoother
{

class SimpleKinematicSmoother
{
public:
  SimpleKinematicSmoother() = default;
  ~SimpleKinematicSmoother() = default;

  void initialize(const OptimizerParams & params)
  {
    debug_ = params.debug;
    options_.max_num_iterations = params.max_iterations;
    options_.function_tolerance = params.fn_tol;
    options_.gradient_tolerance = params.gradient_tol;
    options_.parameter_tolerance = params.param_tol;
    options_.linear_solver_type = params.solver_types.at(params.linear_solver_type);

    if (debug_) {
      options_.minimizer_progress_to_stdout = true;
      options_.logging_type = ceres::PER_MINIMIZER_ITERATION;
    } else {
      options_.logging_type = ceres::SILENT;
    }
  }

  size_t getLastOptimizedKnotCount() const
  {
    return last_optimized_knot_count_;
  }

  bool smooth(
    std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const Costmap2D * costmap,
    const SmootherParams & params)
  {
    return smooth(path, start_dir, end_dir, costmap, params, nullptr);
  }

  bool smooth(
    std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
    if (path.size() < 2) {
      throw InvalidPath("Simple kinematic smoother: Path must have at least 2 points");
    }

    options_.max_solver_time_in_seconds = params.max_time;

    const size_t expected_esdf_size =
      static_cast<size_t>(costmap->getSizeInCellsX()) * costmap->getSizeInCellsY();
    if (precomputed_esdf != nullptr) {
      if (precomputed_esdf->size() != expected_esdf_size) {
        throw std::runtime_error("Precomputed ESDF size does not match costmap dimensions");
      }
      esdf_values_ = *precomputed_esdf;
    } else {
      esdf_values_ = ESDF::ComputeESDF(
        costmap,
        Costmap2D::LETHAL_OBSTACLE,
        params.use_exact_esdf ? ESDFAlgorithm::Exact : ESDFAlgorithm::Approximate);
    }

    const ProcessedPath processed = buildProcessedPath(path, start_dir, end_dir, params, costmap);
    std::vector<double> variables = processed.initial_variables;

    ceres::Problem problem;
    buildProblem(processed, costmap, params, variables, problem);

    applyBounds(problem, variables.data(), processed.state_count, params.max_curvature);

    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);
    if (debug_) {
      std::cout << summary.FullReport() << std::endl;
    }
    if (!summary.IsSolutionUsable() || summary.initial_cost - summary.final_cost < 0.0) {
      throw FailedToSmoothPath("Simple kinematic smoother failed to produce a usable solution");
    }

    path = unpackPath(variables, processed.state_count);
    last_optimized_knot_count_ = processed.state_count;
    return true;
  }

private:
  struct ProcessedPath
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

  class TransitionCostFunctor
  {
  public:
    TransitionCostFunctor(
      double gear,
      bool is_cusp_segment,
      double model_weight,
      double smooth_weight,
      double spacing_weight,
      double fix_weight,
      double target_spacing)
    : gear_(gear),
      is_cusp_segment_(is_cusp_segment),
      model_weight_(model_weight),
      smooth_weight_(smooth_weight),
      spacing_weight_(spacing_weight),
      fix_weight_(fix_weight),
      target_spacing_(target_spacing)
    {
    }

    ceres::CostFunction * AutoDiff()
    {
      return new ceres::AutoDiffCostFunction<TransitionCostFunctor, 5, 5, 5>(this);
    }

    template<typename T>
    bool operator()(const T * const current, const T * const next, T * residuals) const
    {
      Eigen::Map<Eigen::Matrix<T, 5, 1>> residual(residuals);
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
        residual[4] = T(spacing_weight_) * T(10.0) * ds;
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
      residual[3] = T(smooth_weight_) * (next_kappa - kappa) / denom;
      residual[4] = T(spacing_weight_) * (ds - T(target_spacing_)) / T(target_spacing_);
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
    double smooth_weight_;
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
      esdf_interpolator_(std::make_shared<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>(*esdf_grid_))
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

  static double normalizeAngle(double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  static Eigen::Vector2d normalizedDirection(const Eigen::Vector2d & dir)
  {
    if (dir.norm() <= 1e-9) {
      return Eigen::Vector2d(1.0, 0.0);
    }
    return dir.normalized();
  }

  ProcessedPath buildProcessedPath(
    const std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params,
    const Costmap2D * costmap) const
  {
    ProcessedPath processed;
    processed.start_theta = std::atan2(start_dir.y(), start_dir.x());
    processed.end_theta = std::atan2(end_dir.y(), end_dir.x());

    std::vector<double> gear_directions;
    gear_directions.reserve(path.size() - 1);
    for (size_t index = 0; index + 1 < path.size(); ++index) {
      gear_directions.push_back(path[index].z() < 0.0 ? -1.0 : 1.0);
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
      std::max(costmap->getResolution(), 1e-3);

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
    const ProcessedPath & processed,
    const Costmap2D * costmap,
    const SmootherParams & params,
    std::vector<double> & variables,
    ceres::Problem & problem) const
  {
    const double model_weight = std::max(params.smooth_weight_sqrt, 1.0);
    const double smooth_weight =
      std::max(std::max(params.curvature_rate_weight_sqrt, params.curvature_weight_sqrt), 1.0);
    const double spacing_weight = 1.0;
    const double fix_weight = 100.0;
    const double reference_weight = std::max(params.distance_weight_sqrt, 0.0);
    const bool has_obstacle_cost = std::max(params.costmap_weight_sqrt, 0.0) > 1e-9;

    for (size_t index = 0; index + 1 < processed.state_count; ++index) {
      auto * transition_cost = new TransitionCostFunctor(
        processed.gears[index],
        processed.is_cusp_segment[index],
        model_weight,
        smooth_weight,
        spacing_weight,
        fix_weight,
        processed.target_spacing);
      problem.AddResidualBlock(
        transition_cost->AutoDiff(),
        nullptr,
        stateData(variables, index),
        stateData(variables, index + 1));
    }

    auto * start_boundary_cost = new BoundaryCostFunctor(
      processed.reference_points.front(),
      processed.start_theta,
      params.keep_start_orientation,
      fix_weight,
      false);
    problem.AddResidualBlock(start_boundary_cost->AutoDiff(), nullptr, stateData(variables, 0));

    auto * goal_boundary_cost = new BoundaryCostFunctor(
      processed.reference_points.back(),
      processed.end_theta,
      params.keep_goal_orientation,
      fix_weight,
      true);
    problem.AddResidualBlock(
      goal_boundary_cost->AutoDiff(),
      nullptr,
      stateData(variables, processed.state_count - 1));

    if (reference_weight > 1e-9) {
      for (size_t index = 0; index < processed.state_count; ++index) {
        auto * reference_cost = new ReferenceCostFunctor(processed.reference_points[index], reference_weight);
        problem.AddResidualBlock(reference_cost->AutoDiff(), nullptr, stateData(variables, index));
      }
    }

    if (has_obstacle_cost) {
      for (size_t index = 0; index < processed.state_count; ++index) {
        const bool is_cusp_pose =
          (index < processed.is_cusp_segment.size() && processed.is_cusp_segment[index]) ||
          (index > 0 && processed.is_cusp_segment[index - 1]);
        auto * obstacle_cost = new ObstacleCostFunctor(is_cusp_pose, costmap, params, esdf_values_);
        problem.AddResidualBlock(obstacle_cost->AutoDiff(), nullptr, stateData(variables, index));
      }
    }
  }

  void applyBounds(
    ceres::Problem & problem,
    double * variables,
    size_t state_count,
    double max_curvature) const
  {
    const double clamped_max_curvature = std::max(max_curvature, 1e-6);
    for (size_t index = 0; index < state_count; ++index) {
      double * state = variables + 5 * index;
      problem.SetParameterLowerBound(state, 3, -clamped_max_curvature);
      problem.SetParameterUpperBound(state, 3, clamped_max_curvature);
      problem.SetParameterLowerBound(state, 4, 0.0);
    }
  }

  static double * stateData(std::vector<double> & variables, size_t index)
  {
    return variables.data() + 5 * index;
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

  bool debug_{false};
  ceres::Solver::Options options_{};
  std::vector<double> esdf_values_{};
  size_t last_optimized_knot_count_{0};
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_SIMPLE_HPP_