// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0
//
// Kinematic path smoother using Ceres solver.
// Converts the Python CasADi/IPOPT nonlinear smoother to a C++ Ceres
// least-squares formulation, following the pattern of nav2_constrained_smoother.

#ifndef KINEMATIC_SMOOTHER__KINEMATIC_SMOOTHER_HPP_
#define KINEMATIC_SMOOTHER__KINEMATIC_SMOOTHER_HPP_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "ceres/ceres.h"
#include "Eigen/Core"

#include "kinematic_smoother/cost_functions.hpp"
#include "kinematic_smoother/dubins_curve.hpp"
#include "kinematic_smoother/esdf.hpp"

namespace kinematic_smoother
{

/// Algorithm parameters.
struct SmootherParams
{
  double max_kappa      = 0.5;    ///< Maximum curvature [1/m]
  double w_ref          = 10.0;   ///< Reference tracking weight
  double w_dkappa       = 10.0;   ///< Smoothness (curvature-rate) weight
  double w_kappa        = 0.1;    ///< Curvature magnitude weight
  double w_ds           = 1.0;    ///< Step-size uniformity weight
  double w_kinematic    = 1000.0; ///< Kinematic constraint penalty weight
  double target_ds      = 0.0;    ///< Desired spacing (0 = auto from ref)
  double ds_min_ratio   = 0.05;   ///< Lower bound on ds / target_ds
  double ds_max_ratio   = 2.0;    ///< Upper bound on ds / target_ds
  bool   fix_start_kappa = true;
  double kappa_start    = 0.0;    ///< Initial curvature (used when fix_start_kappa)

  // ESDF
  double w_esdf           = 0.0;  ///< ESDF obstacle-avoidance weight
  double esdf_safe_distance = 0.5; ///< Distance [m] below which penalty applies

  // Solver
  int    max_iterations = 500;
  double tolerance      = 1e-6;
  bool   debug          = false;
};

/// Result returned by the solver.
struct SmootherResult
{
  bool success = false;
  double solve_time_ms = 0.0;
  double target_ds_mag = 0.0;

  std::vector<double> x_opt, y_opt, theta_opt;
  std::vector<double> kappa_opt, ds_opt, dkappa_opt;
  std::vector<double> gears_opt;

  struct Costs
  {
    double total     = 0.0;
    double ref       = 0.0;
    double smooth    = 0.0;
    double kappa     = 0.0;
    double ds        = 0.0;
    double kinematic = 0.0;
    double esdf      = 0.0;
  } costs;
};

/// Main kinematic smoother.
class KinematicSmoother
{
public:
  KinematicSmoother() = default;

  /// Solve the smoothing problem.
  SmootherResult solve(
    const std::vector<double> & x_ref_in,
    const std::vector<double> & y_ref_in,
    const std::vector<double> & theta_ref_in,
    const std::vector<double> & gears_in,
    const SmootherParams & params,
    const ESDF * esdf = nullptr) const
  {
    SmootherResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------------
    // 1. Augment path: insert virtual segments at cusps
    // ---------------------------------------------------------------
    std::vector<double> x_ref, y_ref, theta_ref;
    std::vector<double> gears;
    std::vector<bool>   is_virtual;

    x_ref.push_back(x_ref_in[0]);
    y_ref.push_back(y_ref_in[0]);
    theta_ref.push_back(theta_ref_in[0]);

    for (size_t i = 0; i < x_ref_in.size() - 1; ++i) {
      if (i > 0 && gears_in[i] != gears_in[i - 1]) {
        // Cusp: duplicate the point to create a virtual zero-length segment
        x_ref.push_back(x_ref_in[i]);
        y_ref.push_back(y_ref_in[i]);
        theta_ref.push_back(theta_ref_in[i]);
        gears.push_back(gears_in[i]);
        is_virtual.push_back(true);
      }
      x_ref.push_back(x_ref_in[i + 1]);
      y_ref.push_back(y_ref_in[i + 1]);
      theta_ref.push_back(theta_ref_in[i + 1]);
      gears.push_back(gears_in[i]);
      is_virtual.push_back(false);
    }

    const int N = static_cast<int>(x_ref.size());
    const int M = N - 1;   // number of segments

    // ---------------------------------------------------------------
    // 2. Determine target step size
    // ---------------------------------------------------------------
    double target_ds_mag = params.target_ds;
    if (target_ds_mag < 0.01) {
      double sum = 0.0;
      int cnt = 0;
      for (int i = 0; i < M; ++i) {
        if (is_virtual[i]) {continue;}
        double d = std::hypot(x_ref[i + 1] - x_ref[i], y_ref[i + 1] - y_ref[i]);
        if (d > 1e-4) { sum += d; ++cnt; }
      }
      target_ds_mag = cnt > 0 ? sum / cnt : 0.3;
    }
    result.target_ds_mag = target_ds_mag;

    // ---------------------------------------------------------------
    // 3. Allocate decision variables and set initial values
    // ---------------------------------------------------------------
    // state[i] = [x, y, theta, kappa]   (4 per point)
    // control[i] = [ds, dkappa]          (2 per segment)
    std::vector<std::array<double, 4>> states(N);
    std::vector<std::array<double, 2>> controls(M);

    for (int i = 0; i < N; ++i) {
      states[i] = {x_ref[i], y_ref[i], theta_ref[i], 0.0};
    }
    for (int i = 0; i < M; ++i) {
      if (is_virtual[i]) {
        controls[i] = {0.0, 0.0};
      } else {
        double d = std::hypot(x_ref[i + 1] - x_ref[i], y_ref[i + 1] - y_ref[i]);
        controls[i] = {std::max(d, 1e-4), 0.0};
      }
    }

    // ---------------------------------------------------------------
    // 4. Build Ceres problem
    // ---------------------------------------------------------------
    ceres::Problem problem;

    // Pre-compute sqrt-weights (Ceres minimises 0.5 * sum(r^2))
    const double sw_kin   = std::sqrt(2.0 * params.w_kinematic);
    const double sw_ref   = std::sqrt(2.0 * params.w_ref);
    const double sw_dk    = std::sqrt(2.0 * params.w_dkappa);
    const double sw_kappa = std::sqrt(2.0 * params.w_kappa);
    const double sw_ds    = std::sqrt(2.0 * params.w_ds);
    const double sw_esdf  = std::sqrt(2.0 * params.w_esdf);

    // --- 4a. Kinematic / virtual defects ---
    for (int i = 0; i < M; ++i) {
      if (is_virtual[i]) {
        problem.AddResidualBlock(
          VirtualSegmentCost::Create(sw_kin),
          nullptr,
          states[i].data(), states[i + 1].data());
      } else {
        problem.AddResidualBlock(
          KinematicDefectCost::Create(gears[i], sw_kin),
          nullptr,
          states[i].data(), states[i + 1].data(), controls[i].data());
      }
    }

    // --- 4b. Reference tracking ---
    for (int i = 0; i < N; ++i) {
      problem.AddResidualBlock(
        ReferenceTrackingCost::Create(
          x_ref[i], y_ref[i], theta_ref[i], sw_ref, sw_ref),
        nullptr,
        states[i].data());
    }

    // --- 4c. Smoothness, step-size uniformity ---
    for (int i = 0; i < M; ++i) {
      if (is_virtual[i]) {continue;}
      problem.AddResidualBlock(
        SmoothnessCost::Create(sw_dk),
        nullptr, controls[i].data());
      problem.AddResidualBlock(
        StepSizeUniformityCost::Create(target_ds_mag, sw_ds),
        nullptr, controls[i].data());
    }

    // --- 4d. Curvature ---
    for (int i = 0; i < N; ++i) {
      problem.AddResidualBlock(
        CurvatureCost::Create(sw_kappa),
        nullptr, states[i].data());
    }

    // --- 4e. ESDF obstacle avoidance ---
    if (esdf && esdf->valid() && params.w_esdf > 1e-8) {
      for (int i = 0; i < N; ++i) {
        problem.AddResidualBlock(
          ESDFCost::Create(
            sw_esdf, params.esdf_safe_distance,
            esdf->originX(), esdf->originY(), esdf->resolution(),
            esdf->interpolator()),
          nullptr, states[i].data());
      }
    }

    // ---------------------------------------------------------------
    // 5. Parameter bounds and fixed blocks
    // ---------------------------------------------------------------
    // Fix start state
    if (params.fix_start_kappa) {
      problem.SetParameterBlockConstant(states[0].data());
    } else {
      auto * manifold = new ceres::SubsetManifold(4, {0, 1, 2});
      problem.SetManifold(states[0].data(), manifold);
    }

    // Fix goal x, y, theta (kappa free)
    {
      auto * manifold = new ceres::SubsetManifold(4, {0, 1, 2});
      problem.SetManifold(states[N - 1].data(), manifold);
    }

    // Curvature bounds on all non-fixed states
    for (int i = 0; i < N; ++i) {
      if (problem.IsParameterBlockConstant(states[i].data())) {continue;}
      problem.SetParameterLowerBound(states[i].data(), 3, -params.max_kappa);
      problem.SetParameterUpperBound(states[i].data(), 3,  params.max_kappa);
    }

    // DS bounds on non-virtual controls
    for (int i = 0; i < M; ++i) {
      if (is_virtual[i]) {
        // Fix virtual control to [0, 0]
        problem.SetParameterBlockConstant(controls[i].data());
      } else {
        double ds_min = params.ds_min_ratio * target_ds_mag;
        double ds_max = params.ds_max_ratio * target_ds_mag;
        problem.SetParameterLowerBound(controls[i].data(), 0, ds_min);
        problem.SetParameterUpperBound(controls[i].data(), 0, ds_max);
      }
    }

    // ---------------------------------------------------------------
    // 6. Solve
    // ---------------------------------------------------------------
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = params.max_iterations;
    options.function_tolerance = params.tolerance;
    options.parameter_tolerance = params.tolerance * 0.01;
    options.gradient_tolerance = params.tolerance * 0.1;
    if (params.debug) {
      options.minimizer_progress_to_stdout = true;
      options.logging_type = ceres::PER_MINIMIZER_ITERATION;
    } else {
      options.logging_type = ceres::SILENT;
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.solve_time_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (params.debug) {
      std::cerr << summary.FullReport() << "\n";
    }

    result.success = summary.IsSolutionUsable();

    // ---------------------------------------------------------------
    // 7. Extract solution
    // ---------------------------------------------------------------
    result.x_opt.resize(N);
    result.y_opt.resize(N);
    result.theta_opt.resize(N);
    result.kappa_opt.resize(N);
    for (int i = 0; i < N; ++i) {
      result.x_opt[i]     = states[i][0];
      result.y_opt[i]     = states[i][1];
      result.theta_opt[i] = states[i][2];
      result.kappa_opt[i] = states[i][3];
    }

    result.ds_opt.resize(M);
    result.dkappa_opt.resize(M);
    result.gears_opt.resize(M);
    for (int i = 0; i < M; ++i) {
      double ds_val = controls[i][0];
      result.ds_opt[i]     = ds_val * gears[i];   // signed
      result.dkappa_opt[i] = controls[i][1];
      result.gears_opt[i]  = gears[i];
    }

    // ---------------------------------------------------------------
    // 8. Compute individual cost contributions
    // ---------------------------------------------------------------
    result.costs.total = summary.final_cost * 2.0;  // Ceres stores 0.5*sum(r^2)

    // Evaluate individual cost groups
    auto evalGroup = [&](
      const std::vector<ceres::ResidualBlockId> & ids) -> double
    {
      double cost = 0.0;
      for (auto id : ids) {
        double c = 0.0;
        problem.EvaluateResidualBlock(id, false, &c, nullptr, nullptr);
        cost += c;
      }
      return cost * 2.0;
    };
    (void)evalGroup;  // suppress unused warning if not used

    return result;
  }
};

/// Helper: generate a reference path using the Dubins planner.
inline void generateReferencePath(
  double start_x, double start_y, double start_theta,
  double goal_x, double goal_y, double goal_theta,
  double target_ds, double turning_radius,
  std::vector<double> & x_ref,
  std::vector<double> & y_ref,
  std::vector<double> & theta_ref,
  std::vector<double> & gears,
  std::vector<DubinsCommand> & commands)
{
  DubinsPlanner planner(turning_radius);
  auto result = planner.plan(start_x, start_y, start_theta,
                             goal_x, goal_y, goal_theta);
  if (result.hasSolution()) {
    auto & best = result.best();
    commands = best.commands;
    best.generateTrajectory(start_x, start_y, start_theta, target_ds,
                            x_ref, y_ref, theta_ref, gears);
    // Unwrap angles
    for (size_t i = 1; i < theta_ref.size(); ++i) {
      double d = theta_ref[i] - theta_ref[i - 1];
      while (d > M_PI) { theta_ref[i] -= 2 * M_PI; d -= 2 * M_PI; }
      while (d < -M_PI) { theta_ref[i] += 2 * M_PI; d += 2 * M_PI; }
    }
    return;
  }

  // Fallback: straight line
  double dist = std::hypot(goal_x - start_x, goal_y - start_y);
  int steps = std::max(1, static_cast<int>(dist / target_ds));
  for (int i = 0; i <= steps; ++i) {
    double t = static_cast<double>(i) / steps;
    x_ref.push_back(start_x + t * (goal_x - start_x));
    y_ref.push_back(start_y + t * (goal_y - start_y));
    double move_dir = std::atan2(goal_y - start_y, goal_x - start_x);
    theta_ref.push_back(i == 0 ? start_theta : (i == steps ? goal_theta : move_dir));
    if (i < steps) { gears.push_back(1.0); }
  }
  commands.push_back({dist, 0.0});
}

}  // namespace kinematic_smoother

#endif  // KINEMATIC_SMOOTHER__KINEMATIC_SMOOTHER_HPP_
