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
#include <limits>
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
    const std::vector<Eigen::Vector3d> & path;
    const std::vector<Eigen::Vector3d> & reference_path;
    const Eigen::Vector2d & start_dir;
    const Eigen::Vector2d & end_dir;
    const Costmap2D * costmap;
    const SmootherParams & params;
    const std::vector<double> & esdf_values;
  };

  struct KinematicRequest
  {
    const std::vector<double> & variables;
    const std::vector<Eigen::Vector2d> & reference_points;
    const std::vector<double> & gears;
    const std::vector<bool> & is_cusp_segment;
    size_t state_count;
    double start_theta;
    double end_theta;
    const Costmap2D * costmap;
    const SmootherParams & params;
    const std::vector<double> & esdf_values;
  };

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
    return std::max(costmap->getResolution() * 0.5, 1e-3);
  }

  static double orientationTolerance()
  {
    return 0.1;
  }

  static double displacementTolerance(const Costmap2D * costmap)
  {
    return std::max(costmap->getResolution() * 0.25, 1e-4);
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
    if ((goal_pose.template head<2>() - reference_goal.template head<2>()).norm() > position_tol) {
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
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::CurvatureConstraint,
          "Constrained smoother violated the maximum curvature constraint during post-validation",
          static_cast<int>(index));
      }
    }

    return true;
  }

  bool validateSmoothedObstacleClearance(
    const SmoothedPathRequest & request,
    SmoothingFailureInfo * failure) const
  {
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
    if (std::hypot(goal_dx, goal_dy) > position_tol) {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::GoalPositionConstraint,
        "Kinematic smoother violated the fixed goal position constraint",
        static_cast<int>(request.state_count - 1));
    }
    if (request.params.keep_goal_orientation &&
      std::abs(angleDifference(goal_state[2], request.end_theta)) > angle_tol)
    {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::GoalOrientationConstraint,
        "Kinematic smoother violated the fixed goal orientation constraint",
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
    const double radius = std::max(request.params.cost_check_radius, 0.0);
    if (request.params.cost_check_points.empty() || radius <= 1e-9) {
      return true;
    }

    for (size_t state_index = 0; state_index < request.state_count; ++state_index) {
      const double * state = request.variables.data() + 5 * state_index;
      const double x = state[0];
      const double y = state[1];
      const double theta = state[2];
      const double cos_theta = std::cos(theta);
      const double sin_theta = std::sin(theta);

      for (size_t offset = 0; offset + 2 < request.params.cost_check_points.size(); offset += 3) {
        const double local_x = request.params.cost_check_points[offset + 0];
        const double local_y = request.params.cost_check_points[offset + 1];
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
      }
    }

    return true;
  }
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_VALIDATOR_HPP_
