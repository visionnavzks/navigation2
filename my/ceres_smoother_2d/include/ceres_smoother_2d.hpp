#pragma once

/**
 * Ceres-based 2D path smoother with ESDF obstacle avoidance.
 *
 * Costs:
 *   - Smoothness:  jerk penalty  (second-order finite difference)
 *   - Curvature:   hinge penalty on local turning angle (≤ min_turning_radius)
 *   - Reference:   spring toward the A* reference path
 *   - Length:      elastic-band squared segment length (uniform-spacing force)
 *   - Obstacle:    ESDF-based; pushes path away from obstacles
 *
 * All gradients via Ceres AutoDiff (Jet<double,N>). No ROS dependency.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <vector>

#include "ceres/ceres.h"

#include "esdf_map.hpp"

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
    return min_turning_radius > 0 ? 1.0 / min_turning_radius : std::numeric_limits<double>::infinity();
  }

  double obstacleCostDistance() const
  {
    return safety_margin + robot_radius;
  }

};

// NOTE on constructor conventions:
// All cost structs accept a pre-computed sqrt_w in their constructor
// and store it directly.  The caller is responsible for computing
// sqrt(w) *once*; the struct must NOT call std::sqrt again internally.

// Smoothness cost: penalizes second-order finite difference (= discrete
// acceleration, not jerk).  Jerk would be a 4-point cubic difference
// p[i+2]-3*p[i+1]+3*p[i]-p[i-1]; we keep this 3-point form because it
// still yields a tridiagonal Hessian and a closed-form Ceres AutoDiff.
//   residual = sqrt_w * (p_next - 2*p_curr + p_prev)
struct SmoothnessCost
{
  explicit SmoothnessCost(double sqrt_w) : sqrt_w_(sqrt_w) {}

  template<typename T>
  bool operator()(const T * p_prev, const T * p_curr, const T * p_next, T * r) const
  {
    r[0] = sqrt_w_ * (p_next[0] - T(2.0) * p_curr[0] + p_prev[0]);
    r[1] = sqrt_w_ * (p_next[1] - T(2.0) * p_curr[1] + p_prev[1]);
    return true;
  }

  double sqrt_w_;
};

// Curvature cost: denominator-free turning-angle constraint via dot-product
// hinge loss.  Replaces Menger curvature (which has a·b·c denominator that
// explodes near degenerate triangles).  No division → immune to NaN.
//
// Given segments v1=(p_curr-p_prev) and v2=(p_next-p_curr), we compute
// the local step ds = ‖v1‖+‖v2‖)/2 and the maximum allowed turning angle
// θ_max = κ_max · ds.  The hinge penalty fires when the dot product falls
// below ‖v1‖·‖v2‖·cos(θ_max), i.e. the turn is too sharp.
struct CurvatureCost
{
  CurvatureCost(double sqrt_w, double max_kappa)
  : sqrt_w_(sqrt_w), max_kappa_(max_kappa) {}

  template<typename T>
  bool operator()(const T * p_prev, const T * p_curr, const T * p_next, T * r) const
  {
    T v1[2] = {p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]};
    T v2[2] = {p_next[0] - p_curr[0], p_next[1] - p_curr[1]};

    T dot = v1[0] * v2[0] + v1[1] * v2[1];
    T eps(1e-12);
    T norm_v1 = ceres::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + eps);
    T norm_v2 = ceres::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + eps);

    T current_ds = T(0.5) * (norm_v1 + norm_v2);
    // Clamp to π so cos() stays in [-1,1] even for unrealistically large κ·ds.
    T max_theta = ceres::fmin(T(M_PI), T(max_kappa_) * current_ds);
    T target_dot = norm_v1 * norm_v2 * ceres::cos(max_theta);

    T deficit = target_dot - dot;
    r[0] = deficit > T(0.0) ? sqrt_w_ * deficit : T(0.0);
    return true;
  }

  double sqrt_w_;
  double max_kappa_;
};

// Reference cost: penalty for deviating from the reference path.
//   residual = sqrt_w * (p - p_ref)
struct ReferenceCost
{
  ReferenceCost(double x_ref, double y_ref, double sqrt_w)
  : x_ref_(x_ref), y_ref_(y_ref), sqrt_w_(sqrt_w) {}

  template<typename T>
  bool operator()(const T * p, T * r) const
  {
    r[0] = sqrt_w_ * (p[0] - T(x_ref_));
    r[1] = sqrt_w_ * (p[1] - T(y_ref_));
    return true;
  }

  double x_ref_, y_ref_;
  double sqrt_w_;
};

// Elastic-band length cost: minimize Σ‖p_next - p_curr‖² (sum of squared
// inter-point distances). Equivalent to the "rubber band" force used in
// TEB / E-band planners — no rest length, no sqrt nonlinearity.
//
//   residual = sqrt_w * (p_next - p_curr)        // 2 components (dx, dy)
//   Ceres reports 0.5 * sum(residual²), so final_cost contribution is
//   0.5 * w * (dx² + dy²) per segment.
//
// Why this over a target_spacing spring:
//   - Pure linear residual → constant Jacobian → Ceres converges in only
//     a few iterations. No sqrt(.) nonlinearity, no 1/||Δp|| singularities.
//   - No fixed rest length → no conflict with the locked start/goal points
//     (target_spacing × (N-1) rarely equals the actual path length).
//   - When the total length is bounded by other costs (smoothness, reference,
//     obstacle), minimizing Σ‖Δs‖² also evens out segment lengths, so the
//     resample_after_smooth step finishes the job deterministically.
struct PathLengthSquareCost
{
  explicit PathLengthSquareCost(double sqrt_w) : sqrt_w_(sqrt_w) {}

  template<typename T>
  bool operator()(const T * p_curr, const T * p_next, T * r) const
  {
    r[0] = sqrt_w_ * (p_next[0] - p_curr[0]);
    r[1] = sqrt_w_ * (p_next[1] - p_curr[1]);
    return true;
  }
  double sqrt_w_;
};

// Obstacle cost: two terms for cleaner wall behavior.
//   residual[0] = sqrt_w_obstacle  * max(0, safe_dist - dist)   // soft hinge outside
//   residual[1] = sqrt_w_penetrate * max(0, -dist)             // hard penalty inside
//
// The first term is the standard symmetric hinge: as the path approaches
// the safety boundary from outside, residual grows; past the boundary
// (dist < 0) it stays at a flat plateau. The second term fills in that
// plateau so the optimizer keeps pushing OUT the deeper it goes — a
// point 0.3m inside a wall pays `0.5 * w_penetration * 0.09`, vs. a
// point at the boundary which pays only `0.5 * w_obstacle * safe_dist^2`.
//
// Both weights default so w_penetration=0 reproduces the old single-term
// behavior exactly. Setting w_penetration > 0 makes inside-obstacle
// states strictly suboptimal, eliminating the "stuck inside a wall"
// local minimum that the hinge alone cannot escape.
//
// BILINEAR (not BiCubic) lookup — see the long comment on
// ESDFMap::bilinearJet for why. The BiCubic kernel overshoots across
// the sharp ESDF discontinuity at obstacle boundaries, producing wildly
// wrong distances (and gradients that point INTO walls). Bilinear is
// C^0 only but always bounded by neighbor min/max — so the push direction
// is always correct.
struct ObstacleCostCeres
{
  ObstacleCostCeres(
    const ESDFMap * map,
    double safe_dist,
    double sqrt_w_obstacle,
    double sqrt_w_penetrate)
  : map_(map),
    safe_dist_(safe_dist),
    sqrt_w_obstacle_(sqrt_w_obstacle),
    sqrt_w_penetrate_(sqrt_w_penetrate)
  {
  }

  template<typename T>
  bool operator()(const T * p, T * residual) const
  {
    // Bilinear ESDF lookup with analytical Jet derivatives.
    const T dist = map_->bilinearJet<T>(p[0], p[1]);
    const T diff = T(safe_dist_) - dist;
    // Gate the hinge on the scalar part. Ceres Jet<T,N> has no implicit
    // conversion to T, so peek at the .a member; for plain double this
    // branch is taken and the result is identity.
    double diff_scalar;
    if constexpr (std::is_same<T, double>::value) {
      diff_scalar = diff;
    } else {
      diff_scalar = diff.a;
    }
    residual[0] = diff_scalar > 0.0 ? sqrt_w_obstacle_ * diff : T(0.0);
    // Penetration cost: -dist > 0 only when inside an obstacle. Active
    // on the SECOND residual so the AutoDiff slot count matches what
    // we declare in AddResidualBlock<ObstacleCostCeres, 2, 2>.
    const T pen = -dist;
    double pen_scalar;
    if constexpr (std::is_same<T, double>::value) {
      pen_scalar = pen;
    } else {
      pen_scalar = pen.a;
    }
    residual[1] = pen_scalar > 0.0 ? sqrt_w_penetrate_ * pen : T(0.0);
    return true;
  }

  const ESDFMap * map_;
  double safe_dist_, sqrt_w_obstacle_, sqrt_w_penetrate_;
};

// ========================================================================
// Smoother Result
// ========================================================================
struct SmootherResult
{
  bool success{false};
  std::vector<double> x;
  std::vector<double> y;
  double final_cost{0.0};
  double solve_time_ms{0.0};
  int iterations{0};
  std::string report;
};

// ========================================================================
// resamplePathByArcLength
// ========================================================================
// Uniformly resample a polyline along its arc length so consecutive output
// points are ~`target_spacing` meters apart. Keeps the first and last points
// exactly (no endpoint drift) and linearly interpolates within each input
// segment. Used as an optional post-processing step after smooth().
//
// Args:
//   xs_in, ys_in:    input polyline (N >= 2)
//   target_spacing:  desired average inter-point distance (meters)
//   xs_out, ys_out:  resampled output polyline
//
// Notes:
//   - Output count M = max(2, round(L / target_spacing) + 1) where L is the
//     total arc length, guaranteeing average spacing <= target.
//   - If N < 2, target_spacing <= 0, or total arc length is ~0, the input
//     is returned unchanged (degenerate case: no resampling possible).
inline void resamplePathByArcLength(
  const std::vector<double> & xs_in,
  const std::vector<double> & ys_in,
  double target_spacing,
  std::vector<double> & xs_out,
  std::vector<double> & ys_out)
{
  xs_out.clear();
  ys_out.clear();
  const int N = static_cast<int>(xs_in.size());
  if (N < 2 || target_spacing <= 0.0) {
    xs_out = xs_in;
    ys_out = ys_in;
    return;
  }

  // Cumulative arc length at each input vertex.
  std::vector<double> cum(N, 0.0);
  for (int i = 1; i < N; ++i) {
    const double dx = xs_in[i] - xs_in[i - 1];
    const double dy = ys_in[i] - ys_in[i - 1];
    cum[i] = cum[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  const double total = cum.back();
  if (total < 1e-12) {
    // All input points coincide: nothing to resample.
    xs_out = xs_in;
    ys_out = ys_in;
    return;
  }

  int M = static_cast<int>(std::round(total / target_spacing)) + 1;
  if (M < 2) {M = 2;}

  xs_out.resize(M);
  ys_out.resize(M);
  // Anchor endpoints to original vertices (preserves start/goal exactly).
  xs_out[0] = xs_in.front();
  ys_out[0] = ys_in.front();
  xs_out[M - 1] = xs_in.back();
  ys_out[M - 1] = ys_in.back();

  // For each intermediate output, walk cum[] to find the enclosing segment
  // and linearly interpolate by the local arc-length fraction.
  for (int j = 1; j < M - 1; ++j) {
    const double s = static_cast<double>(j) * total / static_cast<double>(M - 1);
    int i = 1;
    while (i < N && cum[i] < s) {++i;}
    if (i >= N) {i = N - 1;}
    const double seg_len = cum[i] - cum[i - 1];
    const double t = (seg_len > 1e-12) ? (s - cum[i - 1]) / seg_len : 0.0;
    xs_out[j] = xs_in[i - 1] + t * (xs_in[i] - xs_in[i - 1]);
    ys_out[j] = ys_in[i - 1] + t * (ys_in[i] - ys_in[i - 1]);
  }
}

// ========================================================================
// PathSmoother2D: Ceres-based 2D path smoother
// ========================================================================
class PathSmoother2D
{
public:
  explicit PathSmoother2D(SmootherParams params = {})
  : params_(std::move(params)) {}

  SmootherResult smooth(
    const std::vector<double> & x_in,
    const std::vector<double> & y_in,
    const ESDFMap & map) const
  {
    SmootherResult result;

    const int N_in = static_cast<int>(x_in.size());
    if (x_in.size() != y_in.size()) {
      result.success = false;
      result.report = "x_in and y_in size mismatch";
      return result;
    }
    if (N_in < 3) {
      result.x = x_in;
      result.y = y_in;
      result.success = true;
      return result;
    }

    // Pre-processing: optionally resample the input to uniform spacing so
    // the optimizer starts from an evenly distributed initial guess. Without
    // this, A*-style inputs with dense clusters near walls and sparse points
    // in open space would force the optimizer to fight that unevenness.
    std::vector<double> xs = x_in;
    std::vector<double> ys = y_in;
    if (params_.resample_before_smooth && params_.target_spacing > 0.0) {
      std::vector<double> rx, ry;
      resamplePathByArcLength(xs, ys, params_.target_spacing, rx, ry);
      xs = std::move(rx);
      ys = std::move(ry);
    }
    const int N = static_cast<int>(xs.size());
    if (N < 3) {
      result.x = xs;
      result.y = ys;
      result.success = true;
      return result;
    }

    std::vector<std::array<double, 2>> path_optim(N);
    for (int i = 0; i < N; ++i) {
      path_optim[i] = {xs[i], ys[i]};
    }

    const double sqrt_w_ref = std::sqrt(params_.w_reference);
    const double sqrt_w_smooth = std::sqrt(params_.w_smooth);
    const double sqrt_w_curv = std::sqrt(params_.w_max_curvature);
    const double sqrt_w_length = std::sqrt(params_.w_length);
    const double max_kappa = params_.maxCurvature();

    std::vector<double> obstacle_weight_stages;
    if (params_.w_obstacle <= 0.0) {
      obstacle_weight_stages.push_back(0.0);
    } else {
      double stage_weight = std::min(params_.w_obstacle, 2.0);
      obstacle_weight_stages.push_back(stage_weight);
      while (stage_weight * 10.0 < params_.w_obstacle) {
        stage_weight *= 10.0;
        obstacle_weight_stages.push_back(stage_weight);
      }
      if (obstacle_weight_stages.back() < params_.w_obstacle) {
        obstacle_weight_stages.push_back(params_.w_obstacle);
      }
    }

    ceres::Solver::Summary final_summary;
    double total_solve_time_ms = 0.0;
    int total_iterations = 0;
    bool all_stages_usable = true;
    std::ostringstream report;

    auto solve_stage = [&](double obstacle_weight) {
        ceres::Problem problem;
        for (int i = 0; i < N; ++i) {
          problem.AddParameterBlock(path_optim[i].data(), 2);
        }
        problem.SetParameterBlockConstant(path_optim[0].data());
        problem.SetParameterBlockConstant(path_optim[N - 1].data());

        const double sqrt_w_obs = std::sqrt(obstacle_weight);
        // Penetration cost uses its own (decoupled) sqrt weight so the user
        // can tune "soft obstacle" vs. "hard wall" independently. We do
        // NOT scale it across obstacle_weight_stages — the goal of the
        // staged ramp is to let the smoother find a good shape first
        // (with a low w_obstacle), then progressively tighten. The
        // penetration term should always be on at full strength: even at
        // stage 0, we don't want the optimizer sitting inside a wall.
        const double sqrt_w_pen = std::sqrt(params_.w_penetration);
        // Obstacle cost uses the ESDFMap's Jet-aware bilinear lookup directly
        // (see ObstacleCostCeres comment for why bilinear instead of BiCubic).
        for (int i = 0; i < N; ++i) {
          // Intermediate node costs (position + obstacle)
          if (i > 0 && i < N - 1) {
            if (params_.w_reference > 0.0) {
              problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ReferenceCost, 2, 2>(
                  new ReferenceCost(xs[i], ys[i], sqrt_w_ref)),
                nullptr, path_optim[i].data());
            }
            if (obstacle_weight > 0.0 || sqrt_w_pen > 0.0) {
              problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ObstacleCostCeres, 2, 2>(
                  new ObstacleCostCeres(
                    &map, params_.obstacleCostDistance(), sqrt_w_obs, sqrt_w_pen)),
                nullptr, path_optim[i].data());
            }
          }

          // Elastic-band length cost: squared inter-point distance.
          // 2 residuals (dx, dy) -> constant Jacobian -> fast convergence.
          if (params_.w_length > 0.0 && i < N - 1) {
            problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<PathLengthSquareCost, 2, 2, 2>(
                new PathLengthSquareCost(sqrt_w_length)),
              nullptr, path_optim[i].data(), path_optim[i + 1].data());
          }

          // Three-point geometric constraints (i = 1 .. N-2)
          if (i > 0 && i < N - 1) {
            if (params_.w_smooth > 0.0) {
              problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SmoothnessCost, 2, 2, 2, 2>(
                  new SmoothnessCost(sqrt_w_smooth)),
                nullptr,
                path_optim[i - 1].data(), path_optim[i].data(), path_optim[i + 1].data());
            }
            if (params_.w_max_curvature > 0.0) {
              problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CurvatureCost, 1, 2, 2, 2>(
                  new CurvatureCost(sqrt_w_curv, max_kappa)),
                nullptr,
                path_optim[i - 1].data(), path_optim[i].data(), path_optim[i + 1].data());
            }
          }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = params_.max_iterations;
        options.max_solver_time_in_seconds = params_.max_time_seconds;
        options.minimizer_progress_to_stdout = params_.verbose;
        options.logging_type = params_.verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
        // Threading overhead exceeds the speedup for sub-2k-variable problems.
        options.num_threads = 1;

        ceres::Solver::Summary summary;
        auto t0 = std::chrono::steady_clock::now();
        ceres::Solve(options, &problem, &summary);
        auto t1 = std::chrono::steady_clock::now();

        total_solve_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_iterations += static_cast<int>(summary.iterations.size());
        final_summary = summary;
        all_stages_usable = all_stages_usable && summary.IsSolutionUsable();

        double min_dist = std::numeric_limits<double>::infinity();
        double min_margin = std::numeric_limits<double>::infinity();
        int active_count = 0;
        const double obstacle_threshold = params_.obstacleCostDistance();
        for (int i = 1; i < N - 1; ++i) {
          const double d = map.bilinearJet<double>(path_optim[i][0], path_optim[i][1]);
          const double margin = d - obstacle_threshold;
          min_dist = std::min(min_dist, d);
          min_margin = std::min(min_margin, margin);
          if (d < obstacle_threshold) {
            ++active_count;
          }
        }

        std::ostringstream stage_report;
        stage_report << "stage w_obstacle=" << obstacle_weight
                     << ", final_cost=" << summary.final_cost
                     << ", min_d=" << min_dist
                     << ", min_margin=" << min_margin
                     << ", active_count=" << active_count
                     << ", " << summary.BriefReport() << '\n';
        report << stage_report.str();
        if (params_.verbose) {
          std::cout << stage_report.str();
        }
      };

    for (double obstacle_weight : obstacle_weight_stages) {
      solve_stage(obstacle_weight);
      if (!all_stages_usable) {
        break;
      }
    }

    result.solve_time_ms = total_solve_time_ms;
    result.final_cost = final_summary.final_cost;
    result.iterations = total_iterations;
    result.report = report.str();
    result.success = all_stages_usable && final_summary.IsSolutionUsable();

    result.x.resize(N);
    result.y.resize(N);
    for (int i = 0; i < N; ++i) {
      result.x[i] = path_optim[i][0];
      result.y[i] = path_optim[i][1];
    }

    // Optional post-processing: enforce uniform inter-point spacing along
    // the smoothed path.  When enabled, the result may contain more (or
    // fewer) points than the input; start/goal positions are preserved.
    if (params_.resample_after_smooth && params_.target_spacing > 0.0) {
      std::vector<double> rx, ry;
      resamplePathByArcLength(result.x, result.y, params_.target_spacing, rx, ry);
      result.x = std::move(rx);
      result.y = std::move(ry);
    }

    return result;
  }

private:
  SmootherParams params_;
};

}  // namespace ceres_smoother_2d
