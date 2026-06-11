#ifndef SMOOTHER_CLOTHOID__PROBLEM_BUILDER_HPP_
#define SMOOTHER_CLOTHOID__PROBLEM_BUILDER_HPP_

#include <algorithm>
#include <cmath>
#include <vector>

#include "ceres/ceres.h"
#include "Eigen/Core"

#include "smoother_clothoid/costs.hpp"
#include "smoother_clothoid/esdf.hpp"
#include "smoother_clothoid/exceptions.hpp"
#include "smoother_clothoid/options.hpp"
#include "smoother_clothoid/utils.hpp"

namespace smoother_clothoid
{

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

class ProblemBuilder
{
public:
  using EsdfGrid = ceres::Grid2D<double>;
  using EsdfInterpolator = ceres::BiCubicInterpolator<EsdfGrid>;

  explicit ProblemBuilder(std::vector<double> & esdf_values) : esdf_values_(esdf_values) {}

  void initializeEsdfValues(
    const Costmap2D * costmap, const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
    if (!params.obstacleTermsEnabled()) {
      esdf_values_.clear();
      esdf_grid_.reset();
      esdf_interpolator_.reset();
      return;
    }
    const size_t expected = static_cast<size_t>(costmap->getSizeInCellsX()) * costmap->getSizeInCellsY();
    if (precomputed_esdf != nullptr) {
      if (precomputed_esdf->size() != expected)
        throw PrecomputedEsdfSizeMismatch("Precomputed ESDF size does not match costmap dimensions");
      esdf_values_ = *precomputed_esdf;
    } else {
      esdf_values_ = ESDF::ComputeESDF(costmap, Costmap2D::LETHAL_OBSTACLE,
        params.use_exact_esdf ? ESDFAlgorithm::Exact : ESDFAlgorithm::Approximate);
    }
    esdf_grid_ = std::make_shared<EsdfGrid>(
      esdf_values_.data(), 0, costmap->getSizeInCellsY(), 0, costmap->getSizeInCellsX());
    esdf_interpolator_ = std::make_shared<EsdfInterpolator>(*esdf_grid_);
  }

  static ProcessedPath buildProcessedPath(
    const std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir, const Eigen::Vector2d & end_dir,
    const SmootherParams & params, const Costmap2D * costmap)
  {
    ProcessedPath p;
    p.start_theta = std::atan2(start_dir.y(), start_dir.x());
    p.end_theta = std::atan2(end_dir.y(), end_dir.x());

    const auto sampled = downsampleInputPath(path, params);

    std::vector<double> gears;
    gears.reserve(sampled.size() - 1);
    for (size_t i = 0; i + 1 < sampled.size(); ++i)
      gears.push_back(params.reversing_enabled ? (sampled[i].z() < 0.0 ? -1.0 : 1.0) : 1.0);

    p.reference_points.emplace_back(sampled.front().x(), sampled.front().y());
    for (size_t i = 0; i + 1 < sampled.size(); ++i) {
      const double cg = gears[i];
      const double ng = i + 1 < gears.size() ? gears[i + 1] : cg;
      p.gears.push_back(cg);
      p.is_cusp_segment.push_back(false);
      p.reference_points.emplace_back(sampled[i + 1].x(), sampled[i + 1].y());
      if (i + 2 < sampled.size() && cg != ng) {
        p.gears.push_back(0.0);
        p.is_cusp_segment.push_back(true);
        p.reference_points.emplace_back(sampled[i + 1].x(), sampled[i + 1].y());
      }
    }

    p.state_count = p.reference_points.size();
    std::vector<double> theta(p.state_count, 0.0);
    std::vector<double> kappa(p.state_count, 0.0);
    std::vector<double> ds(p.state_count, 0.0);

    double spacing_sum = 0.0;
    size_t spacing_count = 0;
    for (size_t i = 0; i + 1 < p.state_count; ++i) {
      const Eigen::Vector2d d = p.reference_points[i + 1] - p.reference_points[i];
      const double norm = d.norm();
      if (p.is_cusp_segment[i]) {
        theta[i] = i > 0 ? theta[i - 1] : p.start_theta;
        ds[i] = 0.0;
        continue;
      }
      if (norm > 1e-6) {
        double heading = std::atan2(d.y(), d.x());
        if (p.gears[i] < 0.0) heading += PI;
        theta[i] = normalizeAngle(heading);
        ds[i] = norm;
        spacing_sum += norm;
        ++spacing_count;
      } else {
        theta[i] = i > 0 ? theta[i - 1] : p.start_theta;
      }
    }

    theta.back() = theta.size() > 1 ? theta[theta.size() - 2] : p.start_theta;
    if (params.keep_start_orientation) theta.front() = p.start_theta;
    if (params.keep_goal_orientation) theta.back() = p.end_theta;

    p.target_spacing = spacing_count > 0 ? spacing_sum / static_cast<double>(spacing_count)
      : (costmap != nullptr ? std::max(costmap->getResolution(), 1e-3) : p.target_spacing);

    p.initial_variables.reserve(p.state_count * 5);
    for (size_t i = 0; i < p.state_count; ++i) {
      p.initial_variables.push_back(p.reference_points[i].x());
      p.initial_variables.push_back(p.reference_points[i].y());
      p.initial_variables.push_back(theta[i]);
      p.initial_variables.push_back(kappa[i]);
      p.initial_variables.push_back(ds[i]);
    }
    return p;
  }

  void buildProblem(
    const ProcessedPath & processed, const Costmap2D * costmap,
    const SmootherParams & params, std::vector<double> & variables,
    ceres::Problem & problem) const
  {
    const double mw = std::max(params.model_weight_sqrt, 0.0);
    const double cw = std::max(params.kinematic_curvature_weight_sqrt, 0.0);
    const double crw = std::max(params.kinematic_curvature_rate_weight_sqrt, 0.0);
    const double sw = std::max(params.kinematic_spacing_weight_sqrt, 0.0);
    const double lw = std::max(params.path_length_weight_sqrt, 0.0);
    const double fw = std::max(params.fix_weight, 0.0);
    const double rw = std::max(params.reference_path_weight_sqrt, 0.0);
    const bool has_obs = params.obstacleTermsEnabled();

    for (size_t i = 0; i + 1 < processed.state_count; ++i) {
      auto * cost = new detail::TransitionCostFunctor(
        processed.gears[i], processed.is_cusp_segment[i],
        mw, cw, crw, sw, lw, fw, processed.target_spacing);
      problem.AddResidualBlock(cost->AutoDiff(), nullptr,
        stateData(variables, i), stateData(variables, i + 1));
    }

    auto * start_cost = new detail::BoundaryCostFunctor(
      processed.reference_points.front(), processed.start_theta,
      params.keep_start_orientation, 0.0, 0.0, 0.0, fw, false);
    problem.AddResidualBlock(start_cost->AutoDiff(), nullptr, stateData(variables, 0));

    const double goal_theta = goalPositionFrameHeading(
      processed.reference_points, processed.end_theta, params.keep_goal_orientation);
    auto * goal_cost = new detail::BoundaryCostFunctor(
      processed.reference_points.back(), goal_theta, params.keep_goal_orientation,
      params.goal_longitudinal_tolerance, params.goal_lateral_tolerance,
      params.goal_orientation_tolerance, fw, true);
    problem.AddResidualBlock(goal_cost->AutoDiff(), nullptr,
      stateData(variables, processed.state_count - 1));

    if (rw > 1e-9) {
      for (size_t i = 0; i < processed.state_count; ++i) {
        auto * ref_cost = new detail::ReferenceCostFunctor(processed.reference_points[i], rw);
        problem.AddResidualBlock(ref_cost->AutoDiff(), nullptr, stateData(variables, i));
      }
    }

    if (has_obs) {
      for (size_t i = 0; i < processed.state_count; ++i) {
        const bool is_cusp = (i < processed.is_cusp_segment.size() && processed.is_cusp_segment[i])
          || (i > 0 && processed.is_cusp_segment[i - 1]);
        auto * obs_cost = new detail::ObstacleCostFunctor(
          is_cusp, costmap, params, esdf_grid_, esdf_interpolator_);
        problem.AddResidualBlock(obs_cost->AutoDiff(), nullptr, stateData(variables, i));
      }
    }
  }

  static void applyBounds(
    ceres::Problem & problem, double * variables,
    const std::vector<Eigen::Vector2d> & refs, size_t n,
    double max_curvature, double max_spacing, double max_deviation)
  {
    const double mc = std::max(max_curvature, 1e-6);
    for (size_t i = 0; i < n; ++i) {
      double * s = variables + 5 * i;
      if (max_deviation > 1e-9) {
        problem.SetParameterLowerBound(s, 0, refs[i].x() - max_deviation);
        problem.SetParameterUpperBound(s, 0, refs[i].x() + max_deviation);
        problem.SetParameterLowerBound(s, 1, refs[i].y() - max_deviation);
        problem.SetParameterUpperBound(s, 1, refs[i].y() + max_deviation);
      }
      problem.SetParameterLowerBound(s, 3, -mc);
      problem.SetParameterUpperBound(s, 3, mc);
      problem.SetParameterLowerBound(s, 4, 0.0);
      if (max_spacing > 1e-9) problem.SetParameterUpperBound(s, 4, max_spacing);
    }
  }

  static std::vector<Eigen::Vector3d> unpackPath(const std::vector<double> & vars, size_t n)
  {
    std::vector<Eigen::Vector3d> path;
    path.reserve(n);
    for (size_t i = 0; i < n; ++i)
      path.emplace_back(vars[5*i], vars[5*i+1], normalizeAngle(vars[5*i+2]));
    return path;
  }

  static std::vector<Eigen::Vector3d> upsamplePath(
    const std::vector<double> & vars, const ProcessedPath & p, const SmootherParams & params)
  {
    const int factor = std::max(params.path_upsampling_factor, 1);
    auto path = unpackPath(vars, p.state_count);
    if (factor <= 1 || p.state_count < 2) return path;

    std::vector<Eigen::Vector3d> up;
    up.reserve(static_cast<size_t>(factor) * (p.state_count - 1) + 1);
    up.push_back(path.front());

    for (size_t i = 0; i + 1 < p.state_count; ++i) {
      const bool cusp = i < p.is_cusp_segment.size() && p.is_cusp_segment[i];
      const double gear = i < p.gears.size() ? p.gears[i] : 1.0;
      const double x = vars[5*i], y = vars[5*i+1], theta = normalizeAngle(vars[5*i+2]);
      const double kappa = vars[5*i+3], ds = std::max(vars[5*i+4], 0.0);
      const double nk = vars[5*(i+1)+3];
      const auto & next = path[i + 1];

      if (cusp || std::abs(gear) < 1e-9 || ds <= 1e-6) { up.push_back(next); continue; }

      const double dir = gear >= 0.0 ? 1.0 : -1.0;
      const double step = ds / static_cast<double>(factor);
      double ix = x, iy = y, it = theta;
      std::vector<Eigen::Vector3d> seg;
      seg.reserve(static_cast<size_t>(factor - 1));

      for (int j = 1; j < factor; ++j) {
        const double t0 = static_cast<double>(j - 1) / factor;
        const double t1 = static_cast<double>(j) / factor;
        const double k0 = kappa + (nk - kappa) * t0;
        const double k1 = kappa + (nk - kappa) * t1;
        const double tm = it + dir * step * 0.5 * k0;
        ix += dir * step * std::cos(tm);
        iy += dir * step * std::sin(tm);
        it = normalizeAngle(it + dir * step * 0.5 * (k0 + k1));
        seg.emplace_back(ix, iy, it);
      }

      const double ft0 = static_cast<double>(factor - 1) / factor;
      const double fk0 = kappa + (nk - kappa) * ft0;
      const double ftm = it + dir * step * 0.5 * fk0;
      const double px = ix + dir * step * std::cos(ftm);
      const double py = iy + dir * step * std::sin(ftm);
      const double pt = normalizeAngle(it + dir * step * 0.5 * (fk0 + nk));
      const double cx = next.x() - px, cy = next.y() - py;
      const double ct = normalizeAngle(next.z() - pt);

      for (int j = 1; j < factor; ++j) {
        const double t = static_cast<double>(j) / factor;
        const auto & s = seg[static_cast<size_t>(j - 1)];
        up.emplace_back(s.x() + t * cx, s.y() + t * cy, normalizeAngle(s.z() + t * ct));
      }
      up.push_back(next);
    }
    return up;
  }

private:
  static std::vector<Eigen::Vector3d> downsampleInputPath(
    const std::vector<Eigen::Vector3d> & path, const SmootherParams & params)
  {
    const int factor = std::max(params.path_downsampling_factor, 1);
    if (factor <= 1 || path.size() <= 2) return path;

    std::vector<Eigen::Vector3d> s;
    s.reserve(path.size());
    s.push_back(path.front());
    size_t last = 0;
    auto ds = [&](size_t i) { return params.reversing_enabled ? (path[i].z() < 0.0 ? -1.0 : 1.0) : 1.0; };

    for (size_t i = 1; i + 1 < path.size(); ++i) {
      const bool cusp = ds(i) != ds(i - 1) || ds(i) != ds(i + 1);
      if (cusp || static_cast<int>(i - last) >= factor) { s.push_back(path[i]); last = i; }
    }
    if (!s.back().isApprox(path.back(), 1e-9)) s.push_back(path.back());
    if (s.size() < 2) s = {path.front(), path.back()};
    return s;
  }

  static double normalizeAngle(double a) { return std::atan2(std::sin(a), std::cos(a)); }
  static double * stateData(std::vector<double> & v, size_t i) { return v.data() + 5 * i; }

  std::vector<double> & esdf_values_;
  std::shared_ptr<EsdfGrid> esdf_grid_{};
  std::shared_ptr<EsdfInterpolator> esdf_interpolator_{};
};

}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__PROBLEM_BUILDER_HPP_
