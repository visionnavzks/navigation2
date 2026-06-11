#ifndef SMOOTHER_CLOTHOID__VALIDATOR_HPP_
#define SMOOTHER_CLOTHOID__VALIDATOR_HPP_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "smoother_clothoid/costmap2d.hpp"
#include "smoother_clothoid/exceptions.hpp"
#include "smoother_clothoid/options.hpp"
#include "smoother_clothoid/utils.hpp"

namespace smoother_clothoid
{

class SmootherValidator
{
public:
  struct KinematicRequest
  {
    const std::vector<double> & variables;
    const std::vector<Eigen::Vector2d> & reference_points;
    const std::vector<double> & gears;
    const std::vector<bool> & is_cusp_segment;
    size_t state_count;
    double start_theta, end_theta;
    const Costmap2D * costmap;
    const SmootherParams & params;
    const std::vector<double> & esdf_values;
  };

  bool validateSolution(const KinematicRequest & req, SmoothingFailureInfo * failure) const
  {
    if (req.variables.size() != req.state_count * 5)
      return throwOrStoreSmoothingFailure(failure, SmoothingFailureReason::InvalidStateVector,
        "Invalid state vector size");
    return validateFinite(req.variables, req.state_count, failure)
      && validateBoundary(req, failure)
      && validateSegments(req, failure)
      && validateCurvature(req, failure)
      && validateObstacles(req, failure);
  }

private:
  static double normAngle(double a) { return std::atan2(std::sin(a), std::cos(a)); }
  static double angleDiff(double a, double b) { return normAngle(a - b); }
  static double posTol(const Costmap2D * c) { return c ? std::max(c->getResolution() * 0.5, 1e-3) : 1e-3; }
  static double oriTol() { return 0.1; }
  static double dispTol(const Costmap2D * c) { return c ? std::max(c->getResolution() * 0.25, 1e-4) : 1e-4; }

  bool validateFinite(const std::vector<double> & v, size_t n, SmoothingFailureInfo * f) const
  {
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < 5; ++j)
        if (!std::isfinite(v[5*i+j]))
          return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::NonFiniteState,
            "Non-finite state at index " + std::to_string(i), static_cast<int>(i));
    return true;
  }

  bool validateBoundary(const KinematicRequest & r, SmoothingFailureInfo * f) const
  {
    const double pt = posTol(r.costmap), at = oriTol();
    const double * s0 = r.variables.data();
    if (std::hypot(s0[0] - r.reference_points.front().x(), s0[1] - r.reference_points.front().y()) > pt)
      return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::StartPositionConstraint,
        "Start position constraint violated", 0);
    if (r.params.keep_start_orientation && std::abs(angleDiff(s0[2], r.start_theta)) > at)
      return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::StartOrientationConstraint,
        "Start orientation constraint violated", 0);

    const double * sg = r.variables.data() + 5 * (r.state_count - 1);
    const double dx = sg[0] - r.reference_points.back().x();
    const double dy = sg[1] - r.reference_points.back().y();
    const double gth = goalPositionFrameHeading(r.reference_points, r.end_theta, r.params.keep_goal_orientation);
    const double cg = std::cos(gth), sg_ = std::sin(gth);
    const double lon = cg*dx + sg_*dy, lat = -sg_*dx + cg*dy;
    const double lt = std::max(r.params.goal_longitudinal_tolerance, pt);
    const double bt = std::max(r.params.goal_lateral_tolerance, pt);
    constexpr double eps = 5e-4;
    if (std::abs(lon) > lt + eps || std::abs(lat) > bt + eps) {
      if (f) {
        f->reason = SmoothingFailureReason::GoalPositionConstraint;
        f->message = "Goal position violated: lon=" + std::to_string(lon) + " lat=" + std::to_string(lat);
        f->failed_index = static_cast<int>(r.state_count - 1);
        f->goal_longitudinal_error = lon;
        f->goal_lateral_error = lat;
        f->goal_longitudinal_tolerance = lt;
        f->goal_lateral_tolerance = bt;
        return false;
      }
      return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::GoalPositionConstraint,
        "Goal position violated", static_cast<int>(r.state_count - 1));
    }
    if (r.params.keep_goal_orientation &&
        std::abs(angleDiff(sg[2], r.end_theta)) > std::max(r.params.goal_orientation_tolerance, at))
      return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::GoalOrientationConstraint,
        "Goal orientation violated", static_cast<int>(r.state_count - 1));
    return true;
  }

  bool validateSegments(const KinematicRequest & r, SmoothingFailureInfo * f) const
  {
    const double pt = posTol(r.costmap), dt = dispTol(r.costmap), at = oriTol();
    for (size_t i = 0; i + 1 < r.state_count; ++i) {
      const double * c = r.variables.data() + 5*i;
      const double * n = r.variables.data() + 5*(i+1);
      const double d = std::hypot(n[0]-c[0], n[1]-c[1]);
      if (r.is_cusp_segment[i]) {
        if (d > pt || std::abs(angleDiff(n[2], c[2])) > at)
          return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::CuspHoldConstraint,
            "Cusp hold violated", static_cast<int>(i));
        continue;
      }
      if (d <= dt)
        return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::CollapsedSegment,
          "Collapsed segment", static_cast<int>(i));
      const double proj = (n[0]-c[0])*std::cos(c[2]) + (n[1]-c[1])*std::sin(c[2]);
      const double g = r.gears[i];
      if ((g >= 0 && proj <= 0) || (g < 0 && proj >= 0))
        return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::MotionDirectionConstraint,
          "Motion direction violated", static_cast<int>(i));
    }
    return true;
  }

  bool validateCurvature(const KinematicRequest & r, SmoothingFailureInfo * f) const
  {
    const double mc = std::max(r.params.max_curvature, 1e-6);
    constexpr double tol = 1e-4;
    for (size_t i = 0; i < r.state_count; ++i) {
      const double ak = std::abs(r.variables[5*i+3]);
      if (ak > mc + tol) {
        const double tr = ak > 1e-9 ? 1.0/ak : std::numeric_limits<double>::infinity();
        if (f) { f->reason = SmoothingFailureReason::CurvatureConstraint; f->failed_index = i;
          f->actual_curvature = ak; f->max_curvature = mc; f->turning_radius = tr;
          f->message = "Curvature " + std::to_string(ak) + " > " + std::to_string(mc); return false; }
        return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::CurvatureConstraint,
          "Curvature violated", static_cast<int>(i));
      }
    }
    const double dt = dispTol(r.costmap);
    for (size_t i = 0; i + 1 < r.state_count; ++i) {
      if (r.is_cusp_segment[i]) continue;
      const double * c = r.variables.data() + 5*i;
      const double * n = r.variables.data() + 5*(i+1);
      const double d = std::hypot(n[0]-c[0], n[1]-c[1]);
      if (d <= dt) continue;
      const double gc = std::abs(angleDiff(n[2], c[2])) / d;
      if (gc > mc + tol) {
        if (f) { f->reason = SmoothingFailureReason::CurvatureConstraint; f->failed_index = i;
          f->actual_curvature = gc; f->max_curvature = mc; f->turning_radius = 1.0/gc;
          f->message = "Geometric curvature " + std::to_string(gc) + " > " + std::to_string(mc); return false; }
        return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::CurvatureConstraint,
          "Geometric curvature violated", static_cast<int>(i));
      }
    }
    return true;
  }

  bool validateObstacles(const KinematicRequest & r, SmoothingFailureInfo * f) const
  {
    if (!r.params.obstacleTermsEnabled() || !r.costmap) return true;
    const double radius = std::max(r.params.cost_check_radius, 0.0);
    if (radius <= 1e-9 && r.params.cost_check_points.empty()) return true;
    const double res = r.costmap->getResolution();
    const double ox = r.costmap->getOriginX(), oy = r.costmap->getOriginY();
    const int sx = r.costmap->getSizeInCellsX(), sy = r.costmap->getSizeInCellsY();

    for (size_t si = 0; si < r.state_count; ++si) {
      const double * st = r.variables.data() + 5*si;
      const double x = st[0], y = st[1], th = st[2];
      const double ct = std::cos(th), sth = std::sin(th);
      auto check = [&](double lx, double ly) {
        const double wx = x + ct*lx - sth*ly, wy = y + sth*lx + ct*ly;
        const int mx = static_cast<int>(std::floor((wx-ox)/res));
        const int my = static_cast<int>(std::floor((wy-oy)/res));
        if (mx < 0 || my < 0 || mx >= sx || my >= sy)
          return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::PathOutOfBounds,
            "Path out of bounds", static_cast<int>(si));
        const size_t idx = static_cast<size_t>(my)*sx + static_cast<size_t>(mx);
        if (idx >= r.esdf_values.size())
          return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::PathOutOfBounds,
            "Path out of bounds", static_cast<int>(si));
        if (r.esdf_values[idx] < radius)
          return throwOrStoreSmoothingFailure(f, SmoothingFailureReason::FootprintCollision,
            "Footprint collision", static_cast<int>(si));
        return true;
      };
      if (r.params.cost_check_points.empty()) { if (!check(0,0)) return false; continue; }
      for (size_t o = 0; o + 2 < r.params.cost_check_points.size(); o += 3)
        if (!check(r.params.cost_check_points[o], r.params.cost_check_points[o+1])) return false;
    }
    return true;
  }
};

}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__VALIDATOR_HPP_
