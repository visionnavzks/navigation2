#ifndef KINEMATIC_PATH_SMOOTHER__SMOOTHER_VALIDATOR_HPP_
#define KINEMATIC_PATH_SMOOTHER__SMOOTHER_VALIDATOR_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"

#include "kinematic_path_smoother/costmap2d.hpp"
#include "kinematic_path_smoother/exceptions.hpp"
#include "kinematic_path_smoother/kinematic_smoother_problem_builder.hpp"
#include "kinematic_path_smoother/math_utils.hpp"
#include "kinematic_path_smoother/options.hpp"

namespace kinematic_path_smoother
{

/// 求解后的硬性校验器。
///
/// Ceres 收敛只代表目标函数局部可接受，不代表路径能交付使用。这里统一检查
/// 数值有限性、起终点约束、cusp 保持、运动方向、曲率和 footprint 净空。
class SmootherValidator
{
public:
  /// 后验校验所需的不可拥有视图。
  struct Request
  {
    /// 优化后的扁平状态数组 [x, y, theta, kappa, ds] * N。
    const std::vector<double> & variables;
    /// 与变量数组对应的拓扑和参考路径元数据。
    const ProcessedPath & path;
    /// 可选 costmap，用于分辨率容差和 footprint 检查。
    const Costmap2D * costmap;
    /// 本次 smooth() 参数。
    const SmootherParams & params;
    /// 与优化阶段共享的 ESDF。
    const std::vector<double> & esdf;
  };

  /// 执行完整后验校验；失败时按 failure 是否为空选择回传或抛异常。
  bool validate(const Request & request, FailureInfo * failure) const
  {
    if (request.variables.size() != request.path.size * 5) {
      return failOrThrow(
        failure, FailureReason::InvalidStateVector, "state vector size does not match knot count");
    }
    if (!finiteStates(request, failure)) {
      return false;
    }
    if (!boundaryStates(request, failure)) {
      return false;
    }
    if (!segments(request, failure)) {
      return false;
    }
    if (!curvature(request, failure)) {
      return false;
    }
    if (!footprint(request, failure)) {
      return false;
    }
    return true;
  }

private:
  /// 位置约束容差跟随地图分辨率，避免对低分辨率地图做不现实的毫米级检查。
  static double positionTolerance(const Costmap2D * costmap)
  {
    return costmap != nullptr ? std::max(0.5 * costmap->getResolution(), 1e-3) : 1e-3;
  }

  static double displacementTolerance(const Costmap2D * costmap)
  {
    return costmap != nullptr ? std::max(0.25 * costmap->getResolution(), 1e-4) : 1e-4;
  }

  static bool finiteState(const double * state)
  {
    for (int i = 0; i < 5; ++i) {
      if (!std::isfinite(state[i])) {
        return false;
      }
    }
    return true;
  }

  static std::pair<int, int> worldToGrid(const Costmap2D & costmap, double x, double y)
  {
    const int mx = static_cast<int>(std::floor((x - costmap.getOriginX()) / costmap.getResolution()));
    const int my = static_cast<int>(std::floor((y - costmap.getOriginY()) / costmap.getResolution()));
    return {mx, my};
  }

  static bool inBounds(const Costmap2D & costmap, int mx, int my)
  {
    return mx >= 0 && my >= 0 &&
           mx < static_cast<int>(costmap.getSizeInCellsX()) &&
           my < static_cast<int>(costmap.getSizeInCellsY());
  }

  static double clearanceAt(
    const Costmap2D & costmap,
    const std::vector<double> & esdf,
    double x,
    double y)
  {
    const auto [mx, my] = worldToGrid(costmap, x, y);
    if (!inBounds(costmap, mx, my)) {
      return -std::numeric_limits<double>::infinity();
    }
    const std::size_t index =
      static_cast<std::size_t>(my) * costmap.getSizeInCellsX() + static_cast<std::size_t>(mx);
    return index < esdf.size() ? esdf[index] : -std::numeric_limits<double>::infinity();
  }

  bool finiteStates(const Request & request, FailureInfo * failure) const
  {
    for (std::size_t i = 0; i < request.path.size; ++i) {
      if (!finiteState(request.variables.data() + 5 * i)) {
        return failOrThrow(failure, FailureReason::NonFiniteState, "non-finite optimized state", i);
      }
    }
    return true;
  }

  bool boundaryStates(const Request & request, FailureInfo * failure) const
  {
    constexpr double angle_tolerance = 0.1;
    constexpr double convergence_slack = 5e-4;
    const double pos_tolerance = positionTolerance(request.costmap);
    const auto & references = request.path.references;

    // 起点是硬锚定点，保证平滑器不会改变规划起点。
    const double * start = request.variables.data();
    if (std::hypot(start[0] - references.front().x(), start[1] - references.front().y()) > pos_tolerance) {
      return failOrThrow(failure, FailureReason::StartConstraint, "start position moved too far", 0);
    }
    if (request.params.keep_start_orientation &&
      std::abs(angleDifference(start[2], request.path.start_heading)) > angle_tolerance)
    {
      return failOrThrow(failure, FailureReason::StartConstraint, "start heading constraint violated", 0);
    }

    // 终点位置按 goal frame 拆成 lon/lat，支持非对称停车容差。
    const std::size_t last = request.path.size - 1;
    const double * goal = request.variables.data() + 5 * last;
    const double goal_heading =
      goalFrameHeading(references, request.path.goal_heading, request.params.keep_goal_orientation);
    const double dx = goal[0] - references.back().x();
    const double dy = goal[1] - references.back().y();
    const double c = std::cos(goal_heading);
    const double s = std::sin(goal_heading);
    const double lon = c * dx + s * dy;
    const double lat = -s * dx + c * dy;
    const double lon_tolerance = std::max(request.params.goal_longitudinal_tolerance, pos_tolerance);
    const double lat_tolerance = std::max(request.params.goal_lateral_tolerance, pos_tolerance);
    if (std::abs(lon) > lon_tolerance + convergence_slack ||
      std::abs(lat) > lat_tolerance + convergence_slack)
    {
      return failOrThrow(failure, FailureReason::GoalConstraint, "goal position tolerance violated", last);
    }

    const double heading_tolerance =
      std::max(request.params.goal_orientation_tolerance, angle_tolerance);
    if (request.params.keep_goal_orientation &&
      std::abs(angleDifference(goal[2], request.path.goal_heading)) > heading_tolerance)
    {
      return failOrThrow(failure, FailureReason::GoalConstraint, "goal heading constraint violated", last);
    }
    return true;
  }

  bool segments(const Request & request, FailureInfo * failure) const
  {
    const double pos_tolerance = positionTolerance(request.costmap);
    const double min_motion = displacementTolerance(request.costmap);
    constexpr double angle_tolerance = 0.1;

    for (std::size_t i = 0; i + 1 < request.path.size; ++i) {
      const double * current = request.variables.data() + 5 * i;
      const double * next = request.variables.data() + 5 * (i + 1);
      const double dx = next[0] - current[0];
      const double dy = next[1] - current[1];
      const double length = std::hypot(dx, dy);

      // cusp 段表示方向切换时车辆静止，必须保持位置和朝向连续。
      if (request.path.cusp_segments[i]) {
        if (length > pos_tolerance || std::abs(angleDifference(next[2], current[2])) > angle_tolerance) {
          return failOrThrow(failure, FailureReason::CuspConstraint, "cusp hold segment moved", i);
        }
        continue;
      }

      if (length <= min_motion) {
        return failOrThrow(failure, FailureReason::CollapsedSegment, "non-cusp segment collapsed", i);
      }

      const Eigen::Vector2d forward(std::cos(current[2]), std::sin(current[2]));
      const double projection = dx * forward.x() + dy * forward.y();
      const double gear = request.path.gears[i];
      if ((gear >= 0.0 && projection <= 0.0) || (gear < 0.0 && projection >= 0.0)) {
        return failOrThrow(
          failure, FailureReason::MotionDirection, "segment direction conflicts with requested gear", i);
      }
    }
    return true;
  }

  bool curvature(const Request & request, FailureInfo * failure) const
  {
    const double limit = std::max(request.params.max_curvature, 1e-6);
    constexpr double slack = 1e-4;
    auto report = [&](std::size_t index, double actual) {
        if (failure != nullptr) {
          failure->reason = FailureReason::CurvatureLimit;
          failure->message = "curvature limit violated";
          failure->index = static_cast<int>(index);
          failure->actual_curvature = actual;
          failure->max_curvature = limit;
          return false;
        }
        return failOrThrow(failure, FailureReason::CurvatureLimit, "curvature limit violated", index);
      };

    // 先检查显式 kappa，再检查相邻姿态形成的几何曲率。
    for (std::size_t i = 0; i < request.path.size; ++i) {
      const double actual = std::abs(request.variables[5 * i + 3]);
      if (actual > limit + slack) {
        return report(i, actual);
      }
    }

    const double min_motion = displacementTolerance(request.costmap);
    for (std::size_t i = 0; i + 1 < request.path.size; ++i) {
      if (request.path.cusp_segments[i]) {
        continue;
      }
      const double * current = request.variables.data() + 5 * i;
      const double * next = request.variables.data() + 5 * (i + 1);
      const double length = std::hypot(next[0] - current[0], next[1] - current[1]);
      if (length <= min_motion) {
        continue;
      }
      const double geometric_curvature = std::abs(angleDifference(next[2], current[2])) / length;
      if (geometric_curvature > limit + slack) {
        return report(i, geometric_curvature);
      }
    }
    return true;
  }

  bool footprint(const Request & request, FailureInfo * failure) const
  {
    if (!request.params.obstacleTermsEnabled() || request.costmap == nullptr) {
      return true;
    }
    if (request.params.footprint_radius <= 1e-9 && request.params.footprint_points.empty()) {
      return true;
    }

    const double radius = std::max(request.params.footprint_radius, 0.0);
    for (std::size_t i = 0; i < request.path.size; ++i) {
      const double * state = request.variables.data() + 5 * i;
      const double c = std::cos(state[2]);
      const double s = std::sin(state[2]);

      // 与障碍物残差一致：无检查点时检查中心点，有检查点时检查每个局部点。
      auto check = [&](double local_x, double local_y) {
          const double world_x = state[0] + c * local_x - s * local_y;
          const double world_y = state[1] + s * local_x + c * local_y;
          const double clearance = clearanceAt(*request.costmap, request.esdf, world_x, world_y);
          if (!std::isfinite(clearance)) {
            return failOrThrow(failure, FailureReason::OutOfBounds, "footprint left costmap bounds", i);
          }
          if (clearance < radius) {
            return failOrThrow(failure, FailureReason::Collision, "footprint intersects obstacle", i);
          }
          return true;
        };

      if (request.params.footprint_points.empty()) {
        if (!check(0.0, 0.0)) {
          return false;
        }
        continue;
      }
      for (std::size_t offset = 0; offset + 2 < request.params.footprint_points.size(); offset += 3) {
        if (!check(request.params.footprint_points[offset], request.params.footprint_points[offset + 1])) {
          return false;
        }
      }
    }
    return true;
  }
};

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__SMOOTHER_VALIDATOR_HPP_
