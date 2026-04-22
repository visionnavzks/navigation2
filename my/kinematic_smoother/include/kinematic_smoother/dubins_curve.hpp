// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0
//
// Dubins curve planner – C++ port from the Python implementation.

#ifndef KINEMATIC_SMOOTHER__DUBINS_CURVE_HPP_
#define KINEMATIC_SMOOTHER__DUBINS_CURVE_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace kinematic_smoother
{

struct DubinsCommand
{
  double length;
  double curvature;
};

struct DubinsPath
{
  std::vector<DubinsCommand> commands;

  double cost() const
  {
    double c = 0.0;
    for (auto & cmd : commands) {c += std::abs(cmd.length);}
    return c;
  }

  static double mod2pi(double theta)
  {
    return theta - 2.0 * M_PI * std::floor(theta / (2.0 * M_PI));
  }

  /// Generate a trajectory along this Dubins path.
  void generateTrajectory(
    double sx, double sy, double syaw, double step_size,
    std::vector<double> & xs, std::vector<double> & ys,
    std::vector<double> & yaws, std::vector<double> & gears) const
  {
    double x = sx, y = sy, yaw = syaw;
    xs.push_back(x);
    ys.push_back(y);
    yaws.push_back(yaw);

    for (auto & cmd : commands) {
      if (std::abs(cmd.length) < 1e-6) {continue;}
      double direction = cmd.length > 0 ? 1.0 : -1.0;
      int steps = std::max(1, static_cast<int>(std::abs(cmd.length) / step_size));
      double ds = cmd.length / steps;
      for (int j = 0; j < steps; ++j) {
        if (std::abs(cmd.curvature) < 1e-6) {
          x += ds * std::cos(yaw);
          y += ds * std::sin(yaw);
        } else {
          double new_yaw = yaw + ds * cmd.curvature;
          x += (std::sin(new_yaw) - std::sin(yaw)) / cmd.curvature;
          y += (std::cos(yaw) - std::cos(new_yaw)) / cmd.curvature;
          yaw = new_yaw;
        }
        yaw = mod2pi(yaw);
        xs.push_back(x);
        ys.push_back(y);
        yaws.push_back(yaw);
        gears.push_back(direction);
      }
    }
  }
};

struct DubinsResult
{
  std::vector<DubinsPath> all_paths;
  bool hasSolution() const { return !all_paths.empty(); }

  const DubinsPath & best() const
  {
    return *std::min_element(all_paths.begin(), all_paths.end(),
      [](const DubinsPath & a, const DubinsPath & b) { return a.cost() < b.cost(); });
  }
};

class DubinsPlanner
{
public:
  explicit DubinsPlanner(double turning_radius)
  : r_(turning_radius) {}

  DubinsResult plan(double sx, double sy, double syaw,
                    double ex, double ey, double eyaw) const
  {
    DubinsResult result;
    result.all_paths = getAllPaths(sx, sy, syaw, ex, ey, eyaw);
    return result;
  }

private:
  double r_;

  static double mod2pi(double theta)
  {
    return theta - 2.0 * M_PI * std::floor(theta / (2.0 * M_PI));
  }

  using Eval = std::optional<std::tuple<double, double, double, std::string>>;

  Eval evalLSL(double a, double b, double d) const
  {
    double psq = 2.0 + d * d - 2.0 * std::cos(a - b) + 2.0 * d * (std::sin(a) - std::sin(b));
    if (psq < 0) {return std::nullopt;}
    double tmp = std::atan2(std::cos(b) - std::cos(a), d + std::sin(a) - std::sin(b));
    return std::make_tuple(mod2pi(-a + tmp), std::sqrt(psq), mod2pi(b - tmp), std::string("LSL"));
  }

  Eval evalRSR(double a, double b, double d) const
  {
    double psq = 2.0 + d * d - 2.0 * std::cos(a - b) + 2.0 * d * (-std::sin(a) + std::sin(b));
    if (psq < 0) {return std::nullopt;}
    double tmp = std::atan2(std::cos(a) - std::cos(b), d - std::sin(a) + std::sin(b));
    return std::make_tuple(mod2pi(a - tmp), std::sqrt(psq), mod2pi(-b + tmp), std::string("RSR"));
  }

  Eval evalLSR(double a, double b, double d) const
  {
    double psq = -2.0 + d * d + 2.0 * std::cos(a - b) + 2.0 * d * (std::sin(a) + std::sin(b));
    if (psq < 0) {return std::nullopt;}
    double tmp = std::atan2(-std::cos(a) - std::cos(b),
                             d + std::sin(a) + std::sin(b)) - std::atan2(-2.0, std::sqrt(psq));
    return std::make_tuple(mod2pi(-a + tmp), std::sqrt(psq), mod2pi(-b + tmp), std::string("LSR"));
  }

  Eval evalRSL(double a, double b, double d) const
  {
    double psq = -2.0 + d * d + 2.0 * std::cos(a - b) - 2.0 * d * (std::sin(a) + std::sin(b));
    if (psq < 0) {return std::nullopt;}
    double tmp = std::atan2(std::cos(a) + std::cos(b),
                             d - std::sin(a) - std::sin(b)) - std::atan2(2.0, std::sqrt(psq));
    return std::make_tuple(mod2pi(a - tmp), std::sqrt(psq), mod2pi(b - tmp), std::string("RSL"));
  }

  Eval evalRLR(double a, double b, double d) const
  {
    double tmp = (6.0 - d * d + 2.0 * std::cos(a - b) +
                  2.0 * d * (std::sin(a) - std::sin(b))) / 8.0;
    if (std::abs(tmp) > 1.0) {return std::nullopt;}
    double p = mod2pi(std::acos(tmp));
    double t = mod2pi(a - std::atan2(std::cos(a) - std::cos(b),
                                      d - std::sin(a) + std::sin(b)) + p / 2.0);
    return std::make_tuple(t, p, mod2pi(a - b - t + p), std::string("RLR"));
  }

  Eval evalLRL(double a, double b, double d) const
  {
    double tmp = (6.0 - d * d + 2.0 * std::cos(a - b) +
                  2.0 * d * (-std::sin(a) + std::sin(b))) / 8.0;
    if (std::abs(tmp) > 1.0) {return std::nullopt;}
    double p = mod2pi(std::acos(tmp));
    double t = mod2pi(-a + std::atan2(-std::cos(a) + std::cos(b),
                                       d + std::sin(a) - std::sin(b)) + p / 2.0);
    return std::make_tuple(t, p, mod2pi(b - a - t + p), std::string("LRL"));
  }

  std::vector<DubinsPath> getAllPaths(
    double sx, double sy, double syaw,
    double ex, double ey, double eyaw) const
  {
    double dx = ex - sx, dy = ey - sy;
    double d_val = std::hypot(dx, dy) / r_;
    double theta = mod2pi(std::atan2(dy, dx));
    double alpha = mod2pi(syaw - theta);
    double beta  = mod2pi(eyaw - theta);

    using EvalFn = Eval (DubinsPlanner::*)(double, double, double) const;
    EvalFn evaluators[] = {
      &DubinsPlanner::evalLSL, &DubinsPlanner::evalRSR,
      &DubinsPlanner::evalLSR, &DubinsPlanner::evalRSL,
      &DubinsPlanner::evalRLR, &DubinsPlanner::evalLRL};

    std::vector<DubinsPath> paths;
    for (auto eval : evaluators) {
      auto res = (this->*eval)(alpha, beta, d_val);
      if (!res) {continue;}
      auto & [t, p, q, types] = *res;
      DubinsPath path;
      double lengths[] = {t, p, q};
      for (int k = 0; k < 3; ++k) {
        double kappa = 0.0;
        if (types[k] == 'L') {kappa = 1.0 / r_;}
        else if (types[k] == 'R') {kappa = -1.0 / r_;}
        path.commands.push_back({lengths[k] * r_, kappa});
      }
      paths.push_back(std::move(path));
    }
    return paths;
  }
};

}  // namespace kinematic_smoother

#endif  // KINEMATIC_SMOOTHER__DUBINS_CURVE_HPP_
