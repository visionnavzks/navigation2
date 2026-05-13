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

  struct SmoothedPathRequest
  {
    /// 求解并重建后的公共路径表示，第三个分量已经是 yaw。
    const std::vector<Eigen::Vector3d> & path;
    /// 进入优化前的参考路径快照，用于固定边界位置与拓扑语义。
    const std::vector<Eigen::Vector3d> & reference_path;
    /// 起点目标切向方向，仅在启用起点朝向约束时参与校验。
    const Eigen::Vector2d & start_dir;
    /// 终点目标切向方向，仅在启用终点朝向约束时参与校验。
    const Eigen::Vector2d & end_dir;
    /// 与本次平滑对应的 costmap；用于坐标容差和障碍物净空检查。
    const Costmap2D * costmap;
    /// 本次平滑的约束和权重参数。
    const SmootherParams & params;
    /// 与优化阶段共享的 ESDF 扁平化存储。
    const std::vector<double> & esdf_values;
  };

  /// 运动学版后验校验所需的状态视图。
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
  bool validateSmoothedPath(
    const SmoothedPathRequest & request,
    SmoothingFailureInfo * failure) const
  {
    if (!validateFinitePath(request.path, failure)) {
      return false;
    }
    if (!validateSmoothedBoundaryStates(request, failure)) {
      return false;
    }
    if (!validateSmoothedCurvature(request, failure)) {
      return false;
    }
    if (!validateSmoothedObstacleClearance(request, failure)) {
      return false;
    }

    return true;
  }

  /// 运动学版 smoother 的总校验入口。
  ///
  /// 它先检查状态向量形状和有限值，再检查边界、段一致性和障碍物净空。
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

  // ---- Geometric smoother validation ----

  bool validateFinitePath(
    const std::vector<Eigen::Vector3d> & path,
    SmoothingFailureInfo * failure) const
  {
    for (size_t index = 0; index < path.size(); ++index) {
      const auto & pose = path[index];
      if (!std::isfinite(pose.x()) || !std::isfinite(pose.y()) || !std::isfinite(pose.z())) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::NonFiniteState,
          "Constrained smoother returned a non-finite pose during post-validation",
          static_cast<int>(index));
      }
    }

    return true;
  }

  bool validateSmoothedBoundaryStates(
    const SmoothedPathRequest & request,
    SmoothingFailureInfo * failure) const
  {
    const double position_tol = positionTolerance(request.costmap);
    const double angle_tol = orientationTolerance();

    const auto & start_pose = request.path.front();
    const auto & reference_start = request.reference_path.front();
    if ((start_pose.template head<2>() - reference_start.template head<2>()).norm() > position_tol) {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::StartPositionConstraint,
        "Constrained smoother violated the fixed start position constraint",
        0);
    }
    if (request.params.keep_start_orientation) {
      const Eigen::Vector2d start_heading = normalizedDirection(request.start_dir);
      const double expected_start_yaw = std::atan2(start_heading.y(), start_heading.x());
      if (std::abs(angleDifference(start_pose.z(), expected_start_yaw)) > angle_tol) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::StartOrientationConstraint,
          "Constrained smoother violated the fixed start orientation constraint",
          0);
      }
    }

    const auto & goal_pose = request.path.back();
    const auto & reference_goal = request.reference_path.back();
    Eigen::Vector2d goal_frame = normalizedDirection(request.end_dir);
    if (!request.params.keep_goal_orientation && request.reference_path.size() >= 2) {
      goal_frame =
        (request.reference_path.back() - request.reference_path[request.reference_path.size() - 2])
        .template head<2>();
      if (goal_frame.norm() <= EPSILON) {
        goal_frame = normalizedDirection(request.end_dir);
      } else {
        goal_frame.normalize();
      }
    }

    const Eigen::Vector2d goal_delta = goal_pose.template head<2>() - reference_goal.template head<2>();
    const double goal_lon = goal_frame.x() * goal_delta.x() + goal_frame.y() * goal_delta.y();
    const double goal_lat = -goal_frame.y() * goal_delta.x() + goal_frame.x() * goal_delta.y();
    const double goal_lon_tol = std::max(request.params.goal_longitudinal_tolerance, position_tol);
    const double goal_lat_tol = std::max(request.params.goal_lateral_tolerance, position_tol);
    if (std::abs(goal_lon) > goal_lon_tol || std::abs(goal_lat) > goal_lat_tol) {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::GoalPositionConstraint,
        "Constrained smoother violated the fixed goal position constraint",
        static_cast<int>(request.path.size() - 1));
    }
    if (request.params.keep_goal_orientation) {
      const Eigen::Vector2d goal_heading = normalizedDirection(request.end_dir);
      const double expected_goal_yaw = std::atan2(goal_heading.y(), goal_heading.x());
      if (std::abs(angleDifference(goal_pose.z(), expected_goal_yaw)) > angle_tol) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::GoalOrientationConstraint,
          "Constrained smoother violated the fixed goal orientation constraint",
          static_cast<int>(request.path.size() - 1));
      }
    }

    return true;
  }

  bool validateSmoothedCurvature(
    const SmoothedPathRequest & request,
    SmoothingFailureInfo * failure) const
  {
    if (
      request.path.size() < 3 ||
      !std::isfinite(request.params.max_curvature) ||
      request.params.max_curvature <= 0.0)
    {
      return true;
    }

    const double displacement_tol = displacementTolerance(request.costmap);

    for (size_t index = 1; index + 1 < request.path.size(); ++index) {
      const Eigen::Vector2d prev = request.path[index - 1].template head<2>();
      const Eigen::Vector2d current = request.path[index].template head<2>();
      const Eigen::Vector2d next = request.path[index + 1].template head<2>();
      const Eigen::Vector2d prev_delta = current - prev;
      const Eigen::Vector2d next_delta = next - current;

      if (prev_delta.norm() <= displacement_tol || next_delta.norm() <= displacement_tol) {
        continue;
      }

      const Eigen::Vector2d heading(
        std::cos(request.path[index].z()),
        std::sin(request.path[index].z()));
      const bool is_cusp = prev_delta.dot(heading) * next_delta.dot(heading) < 0.0;
      const Eigen::Vector2d center = arcCenter<double>(prev, current, next, is_cusp);
      if (!std::isfinite(center.x()) || !std::isfinite(center.y())) {
        continue;
      }

      const double turning_radius = (current - center).norm();
      if (turning_radius <= EPSILON) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::CurvatureConstraint,
          "Constrained smoother collapsed the turning radius during post-validation",
          static_cast<int>(index));
      }

      const double curvature = 1.0 / turning_radius;
      if (curvature - request.params.max_curvature > 1e-3) {
        const std::string message = describeCurvatureViolation(
          curvature,
          request.params.max_curvature,
          turning_radius);
        if (failure != nullptr) {
          failure->reason = SmoothingFailureReason::CurvatureConstraint;
          failure->message = message;
          failure->failed_index = static_cast<int>(index);
          failure->actual_curvature = curvature;
          failure->max_curvature = request.params.max_curvature;
          failure->turning_radius = turning_radius;
          return false;
        }

        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::CurvatureConstraint,
          message,
          static_cast<int>(index));
      }
    }

    return true;
  }

  bool validateSmoothedObstacleClearance(
    const SmoothedPathRequest & request,
    SmoothingFailureInfo * failure) const
  {
    if (!request.params.obstacleTermsEnabled() || request.costmap == nullptr) {
      return true;
    }

    const double radius = std::max(request.params.cost_check_radius, 0.0);

    for (size_t pose_index = 0; pose_index < request.path.size(); ++pose_index) {
      const auto & pose = request.path[pose_index];
      const double cos_theta = std::cos(pose.z());
      const double sin_theta = std::sin(pose.z());

      auto validate_checkpoint = [&](double local_x, double local_y) {
          const double world_x = pose.x() + cos_theta * local_x - sin_theta * local_y;
          const double world_y = pose.y() + sin_theta * local_x + cos_theta * local_y;
          const double clearance = clearanceAtWorldPoint(
            request.costmap, request.esdf_values, world_x, world_y);

          if (!std::isfinite(clearance)) {
            return throwOrStoreSmoothingFailure(
              failure,
              SmoothingFailureReason::PathOutOfBounds,
              "Constrained smoother returned a path that leaves the map bounds during footprint validation",
              static_cast<int>(pose_index));
          }
          if (clearance < radius || (radius <= 1e-9 && clearance <= 0.0)) {
            return throwOrStoreSmoothingFailure(
              failure,
              SmoothingFailureReason::FootprintCollision,
              "Constrained smoother returned a path that collides with obstacles during footprint validation",
              static_cast<int>(pose_index));
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
            request.params.cost_check_points[offset],
            request.params.cost_check_points[offset + 1]))
        {
          return false;
        }
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
    const double cos_goal = std::cos(request.end_theta);
    const double sin_goal = std::sin(request.end_theta);
    const double goal_lon = cos_goal * goal_dx + sin_goal * goal_dy;
    const double goal_lat = -sin_goal * goal_dx + cos_goal * goal_dy;
    const double goal_lon_tol = std::max(request.params.goal_longitudinal_tolerance, position_tol);
    const double goal_lat_tol = std::max(request.params.goal_lateral_tolerance, position_tol);
    if (std::abs(goal_lon) > goal_lon_tol || std::abs(goal_lat) > goal_lat_tol) {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::GoalPositionConstraint,
        "Kinematic smoother violated the fixed goal position constraint",
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
