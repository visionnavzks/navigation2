#pragma once

#include <limits>

namespace ceres_smoother_2d
{

// ========================================================================
// Smoother Parameters
// ========================================================================
struct SmootherParams
{
  int max_iterations{100};
  double max_time_seconds{0.5};
  bool verbose{false};

  // Smoothness: penalizes second-order difference
  double w_smooth{10.0};

  // Max-curvature: soft constraint on maximum curvature
  double w_max_curvature{1000.0};
  double min_turning_radius{0.2};  // meters

  // Reference tracking: penalty for deviating from the A* reference path.
  // Prevents the optimizer from pulling the path too far from the original
  // route when obstacle/length weights are strong. A nonzero default keeps
  // the path anchored near the planner's intent.
  double w_reference{5.0};

  // Elastic-band length: weight on minimizing Σ‖p_next - p_curr‖² (sum of
  // squared inter-point distances). Combined with smoothness + obstacle +
  // reference, this acts as a uniform-spacing force without the nonlinearity
  // or rest-length conflicts of a target_spacing spring.
  // Lowered from 10.0 to avoid over-shrinking the path and overpowering
  // obstacle avoidance — reference + smoothness now carry the shape.
  double w_length{2.0};
  // Desired inter-point spacing in meters — used ONLY by the resample
  // stages (resample_before_smooth / resample_after_smooth), not by the
  // optimization loop. Default 0.3 m gives ~34 points on a 10 m path.
  double target_spacing{0.3};

  // Obstacle (ESDF): two separate terms so the optimizer can trade off
  // "stay out of the safety zone" (soft hinge) vs. "absolutely don't be
  // inside a wall" (deeper penalty that grows monotonically as the point
  // penetrates further). The first term alone (a symmetric hinge around
  // the safety boundary) has a flat plateau on the obstacle side: if the
  // optimizer ends up at any point with dist < 0, the gradient magnitude
  // is constant and the smoother may stall in a wall. w_penetration fixes
  // that by adding a quadratic in -dist that grows the deeper you go.
  double w_obstacle{10.0};
  // Weight on the inside-obstacle penalty. The hinge term alone (w_obstacle)
  // has a flat plateau inside the obstacle where the gradient is constant but
  // small; a point stuck deep inside may never escape. The penetration term
  // adds a cost that grows with depth (-dist), pulling the optimizer out.
  // Nonzero by default so this defense is always active.
  double w_penetration{1000.0};
  double safety_margin{1.0};       // meters, desired minimum clearance (from robot edge)
  double robot_radius{0.5};        // meters, robot inscribed radius; effective clearance
                                  // threshold = safety_margin + robot_radius

  // Post-processing: resample the smoothed path along its arc length so
  // adjacent output points are ~target_spacing meters apart. OFF by default:
  // the returned path then matches the exact discrete points optimized by
  // Ceres, which keeps solver cost diagnostics easier to interpret.
  bool resample_after_smooth{false};

  // Pre-processing: resample the *input* reference path to uniform spacing
  // before optimization. ON by default: when the upstream path (e.g. A*)
  // has uneven point density — dense clusters near walls and sparse points
  // in open space — the optimizer's parameter blocks would otherwise
  // inherit that unevenness and w_length can only push existing points
  // apart, never insert new ones.
  bool resample_before_smooth{true};

  double maxCurvature() const
  {
    return min_turning_radius > 0 ?
           1.0 / min_turning_radius : std::numeric_limits<double>::infinity();
  }

  double obstacleCostDistance() const
  {
    return safety_margin + robot_radius;
  }
};

}  // namespace ceres_smoother_2d
