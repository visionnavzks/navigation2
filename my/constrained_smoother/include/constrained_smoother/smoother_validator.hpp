// Copyright (c) 2021 RoboTech Vision
// Copyright (c) 2020, Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. Reserved.

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_VALIDATOR_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_VALIDATOR_HPP_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"

#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/options.hpp"
#include "constrained_smoother/utils.hpp"

namespace constrained_smoother
{

/// 求解后统一执行的硬性校验器。
///
/// 这个对象故意放在求解器之外，目的是把“数值上收敛”与“工程上可交付”分开：
/// builder 负责构建优化问题，validator 负责决定解是否真的能被接受。
class SmootherValidator
{
public:
  static Eigen::Vector2d normalizedDirection(const Eigen::Vector2d & dir)
  {
    const double norm = dir.norm();
    if (norm <= EPSILON) {
      return Eigen::Vector2d(1.0, 0.0);
    }
    return dir / norm;
  }

  struct KinematicRequest
  {
    /// 展平后的 (x, y, theta, kappa, ds) 状态数组。
    const std::vector<double> & variables;
    /// 展开后的参考点链，与状态索引一一对应。
    const std::vector<Eigen::Vector2d> & reference_points;
    /// 每个状态转移段的 gear 方向，前进为 1，倒车为 -1，cusp 为 0。
    const std::vector<double> & gears;
    /// 标记哪些状态转移段是 cusp 保持段。
    const std::vector<bool> & is_cusp_segment;
    /// 当前状态链长度，调用方需保证与 variables 维度一致。
    size_t state_count;
    /// 起点边界姿态目标值。
    double start_theta;
    /// 终点边界姿态目标值。
    double end_theta;
    /// 与本次平滑对应的 costmap；用于坐标容差和障碍物净空检查。
    const Costmap2D * costmap;
    /// 本次平滑的约束和权重参数。
    const SmootherParams & params;
    /// 与优化阶段共享的 ESDF 扁平化存储。
    const std::vector<double> & esdf_values;
  };

  /// 几何版 smoother 的总校验入口。
  ///
  /// 这里按“有限值 -> 边界 -> 曲率 -> 障碍物净空”的顺序短路执行，
  /// 一旦发现首个不可接受条件，就通过 failure 或异常向上层汇报。
  bool validateKinematicSolution(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    if (request.variables.size() != request.state_count * 5) {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::InvalidStateVector,
        "Kinematic smoother returned an invalid state vector size");
    }

    if (!validateFiniteStates(request.variables, request.state_count, failure)) {
      return false;
    }
    if (!validateKinematicBoundaryStates(request, failure)) {
      return false;
    }
    if (!validateKinematicSegmentConsistency(request, failure)) {
      return false;
    }
    if (!validateKinematicObstacleClearance(request, failure)) {
      return false;
    }

    return true;
  }

private:
  // ---- Shared numeric helpers ----

  static double normalizeAngle(double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  static double angleDifference(double a, double b)
  {
    return normalizeAngle(a - b);
  }

  static double positionTolerance(const Costmap2D * costmap)
  {
    return costmap != nullptr ? std::max(costmap->getResolution() * 0.5, 1e-3) : 1e-3;
  }

  static double orientationTolerance()
  {
    return 0.1;
  }

  static double radiansToDegrees(double radians)
  {
    return radians * 180.0 / M_PI;
  }

  static std::string describeOrientationViolation(
    const std::string & prefix,
    double actual_angle,
    double expected_angle,
    double tolerance)
  {
    const double error = std::abs(angleDifference(actual_angle, expected_angle));
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2)
           << prefix
           << " by " << radiansToDegrees(error) << " deg ("
           << std::setprecision(4) << error << " rad); tolerance "
           << radiansToDegrees(tolerance) << " deg (" << tolerance << " rad); expected "
           << std::setprecision(2) << radiansToDegrees(expected_angle) << " deg, got "
           << radiansToDegrees(actual_angle) << " deg";
    return stream.str();
  }

  static std::string describeCurvatureViolation(
    double actual_curvature,
    double max_curvature,
    double turning_radius)
  {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(4)
           << "Constrained smoother violated the maximum curvature constraint during post-validation"
           << ": actual curvature " << actual_curvature << " 1/m"
           << ", limit " << max_curvature << " 1/m"
           << ", excess " << (actual_curvature - max_curvature) << " 1/m"
           << ", turning radius " << std::setprecision(3) << turning_radius << " m";
    return stream.str();
  }

  static std::string describeGoalPositionViolation(
    const std::string & prefix,
    double goal_lon,
    double goal_lat,
    double goal_lon_tol,
    double goal_lat_tol)
  {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(4)
           << prefix
           << ": lon error " << goal_lon << " m"
           << " (tol " << goal_lon_tol << " m), lat error " << goal_lat << " m"
           << " (tol " << goal_lat_tol << " m)";
    return stream.str();
  }

  static double displacementTolerance(const Costmap2D * costmap)
  {
    return costmap != nullptr ? std::max(costmap->getResolution() * 0.25, 1e-4) : 1e-4;
  }

  static std::pair<int, int> worldToGrid(const Costmap2D * costmap, double wx, double wy)
  {
    const double resolution = costmap->getResolution();
    const int mx = static_cast<int>(std::floor((wx - costmap->getOriginX()) / resolution));
    const int my = static_cast<int>(std::floor((wy - costmap->getOriginY()) / resolution));
    return {mx, my};
  }

  static bool inBounds(const Costmap2D * costmap, int mx, int my)
  {
    return mx >= 0 && my >= 0 &&
           mx < static_cast<int>(costmap->getSizeInCellsX()) &&
           my < static_cast<int>(costmap->getSizeInCellsY());
  }

  static double clearanceAtWorldPoint(
    const Costmap2D * costmap,
    const std::vector<double> & esdf_values,
    double world_x,
    double world_y)
  {
    const auto grid = worldToGrid(costmap, world_x, world_y);
    if (!inBounds(costmap, grid.first, grid.second)) {
      return -std::numeric_limits<double>::infinity();
    }

    const size_t flat_index = static_cast<size_t>(grid.second) * costmap->getSizeInCellsX() +
      static_cast<size_t>(grid.first);
    if (flat_index >= esdf_values.size()) {
      return -std::numeric_limits<double>::infinity();
    }

    return esdf_values[flat_index];
  }

  static bool isFiniteState(const double * state)
  {
    for (size_t index = 0; index < 5; ++index) {
      if (!std::isfinite(state[index])) {
        return false;
      }
    }
    return true;
  }

  // ---- Kinematic smoother validation ----

  bool validateFiniteStates(
    const std::vector<double> & variables,
    size_t state_count,
    SmoothingFailureInfo * failure) const
  {
    for (size_t index = 0; index < state_count; ++index) {
      const double * state = variables.data() + 5 * index;
      if (!isFiniteState(state)) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::NonFiniteState,
          "Kinematic smoother returned a non-finite state at index " + std::to_string(index),
          static_cast<int>(index));
      }
    }

    return true;
  }

  bool validateKinematicBoundaryStates(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    const double position_tol = positionTolerance(request.costmap);
    const double angle_tol = orientationTolerance();

    const double * start_state = request.variables.data();
    const double start_dx = start_state[0] - request.reference_points.front().x();
    const double start_dy = start_state[1] - request.reference_points.front().y();
    if (std::hypot(start_dx, start_dy) > position_tol) {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::StartPositionConstraint,
        "Kinematic smoother violated the fixed start position constraint",
        0);
    }
    if (request.params.keep_start_orientation &&
      std::abs(angleDifference(start_state[2], request.start_theta)) > angle_tol)
    {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::StartOrientationConstraint,
        "Kinematic smoother violated the fixed start orientation constraint",
        0);
    }

    const double * goal_state = request.variables.data() + 5 * (request.state_count - 1);
    const double goal_dx = goal_state[0] - request.reference_points.back().x();
    const double goal_dy = goal_state[1] - request.reference_points.back().y();
    double goal_position_theta = request.end_theta;
    if (!request.params.keep_goal_orientation && request.reference_points.size() >= 2) {
      const Eigen::Vector2d goal_delta =
        request.reference_points.back() - request.reference_points[request.reference_points.size() - 2];
      if (goal_delta.norm() > EPSILON) {
        goal_position_theta = std::atan2(goal_delta.y(), goal_delta.x());
      }
    }
    const double cos_goal = std::cos(goal_position_theta);
    const double sin_goal = std::sin(goal_position_theta);
    const double goal_lon = cos_goal * goal_dx + sin_goal * goal_dy;
    const double goal_lat = -sin_goal * goal_dx + cos_goal * goal_dy;
    const double goal_lon_tol = std::max(request.params.goal_longitudinal_tolerance, position_tol);
    const double goal_lat_tol = std::max(request.params.goal_lateral_tolerance, position_tol);
    constexpr double convergence_epsilon = 5e-4;
    if (std::abs(goal_lon) > goal_lon_tol + convergence_epsilon ||
        std::abs(goal_lat) > goal_lat_tol + convergence_epsilon) {
      const bool uses_goal_box =
        request.params.goal_longitudinal_tolerance > 1e-9 ||
        request.params.goal_lateral_tolerance > 1e-9;
      const std::string message = describeGoalPositionViolation(
        uses_goal_box ?
        "Kinematic smoother violated the goal position tolerance box" :
        "Kinematic smoother violated the fixed goal position constraint",
        goal_lon,
        goal_lat,
        goal_lon_tol,
        goal_lat_tol);
      if (failure != nullptr) {
        failure->reason = SmoothingFailureReason::GoalPositionConstraint;
        failure->message = message;
        failure->failed_index = static_cast<int>(request.state_count - 1);
        failure->goal_longitudinal_error = goal_lon;
        failure->goal_lateral_error = goal_lat;
        failure->goal_longitudinal_tolerance = goal_lon_tol;
        failure->goal_lateral_tolerance = goal_lat_tol;
        return false;
      }
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::GoalPositionConstraint,
        message,
        static_cast<int>(request.state_count - 1));
    }
    if (request.params.keep_goal_orientation &&
      std::abs(angleDifference(goal_state[2], request.end_theta)) >
      std::max(request.params.goal_orientation_tolerance, angle_tol))
    {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::GoalOrientationConstraint,
        describeOrientationViolation(
          "Kinematic smoother violated the fixed goal orientation constraint",
          goal_state[2],
          request.end_theta,
          angle_tol),
        static_cast<int>(request.state_count - 1));
    }

    return true;
  }

  bool validateKinematicSegmentConsistency(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    const double position_tol = positionTolerance(request.costmap);
    const double displacement_tol = displacementTolerance(request.costmap);
    const double angle_tol = orientationTolerance();

    for (size_t index = 0; index + 1 < request.state_count; ++index) {
      const double * current = request.variables.data() + 5 * index;
      const double * next = request.variables.data() + 5 * (index + 1);
      const double dx = next[0] - current[0];
      const double dy = next[1] - current[1];
      const double displacement = std::hypot(dx, dy);

      if (request.is_cusp_segment[index]) {
        if (
          displacement > position_tol ||
          std::abs(angleDifference(next[2], current[2])) > angle_tol)
        {
          return throwOrStoreSmoothingFailure(
            failure,
            SmoothingFailureReason::CuspHoldConstraint,
            "Kinematic smoother violated the cusp hold constraint during post-validation",
            static_cast<int>(index));
        }
        continue;
      }

      if (displacement <= displacement_tol) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::CollapsedSegment,
          "Kinematic smoother collapsed a non-cusp segment during post-validation",
          static_cast<int>(index));
      }

      const Eigen::Vector2d heading(std::cos(current[2]), std::sin(current[2]));
      const double signed_projection = dx * heading.x() + dy * heading.y();
      const double gear = request.gears[index];
      if ((gear >= 0.0 && signed_projection <= 0.0) || (gear < 0.0 && signed_projection >= 0.0)) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::MotionDirectionConstraint,
          "Kinematic smoother returned a path whose motion direction violates the input gear and endpoint constraints",
          static_cast<int>(index));
      }
    }

    return true;
  }

  bool validateKinematicObstacleClearance(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    if (!request.params.obstacleTermsEnabled() || request.costmap == nullptr) {
      return true;
    }

    const double radius = std::max(request.params.cost_check_radius, 0.0);
    if (radius <= 1e-9) {
      return true;
    }

    for (size_t state_index = 0; state_index < request.state_count; ++state_index) {
      const double * state = request.variables.data() + 5 * state_index;
      const double x = state[0];
      const double y = state[1];
      const double theta = state[2];
      const double cos_theta = std::cos(theta);
      const double sin_theta = std::sin(theta);

      auto validate_checkpoint = [&](double local_x, double local_y) {
          const double world_x = x + cos_theta * local_x - sin_theta * local_y;
          const double world_y = y + sin_theta * local_x + cos_theta * local_y;
          const double clearance = clearanceAtWorldPoint(
            request.costmap, request.esdf_values, world_x, world_y);
          if (!std::isfinite(clearance)) {
            return throwOrStoreSmoothingFailure(
              failure,
              SmoothingFailureReason::PathOutOfBounds,
              "Kinematic smoother returned a path that leaves the map bounds during footprint validation",
              static_cast<int>(state_index));
          }
          if (clearance < radius) {
            return throwOrStoreSmoothingFailure(
              failure,
              SmoothingFailureReason::FootprintCollision,
              "Kinematic smoother returned a path that collides with obstacles during footprint validation",
              static_cast<int>(state_index));
          }
          return true;
        };

      if (request.params.cost_check_points.empty()) {
        if (!validate_checkpoint(0.0, 0.0)) {
          return false;
        }
        continue;
      }

      for (size_t offset = 0; offset + 2 < request.params.cost_check_points.size(); offset += 3) {
        if (!validate_checkpoint(
            request.params.cost_check_points[offset + 0],
            request.params.cost_check_points[offset + 1]))
        {
          return false;
        }
      }
    }

    return true;
  }
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_VALIDATOR_HPP_
