#ifndef KINEMATIC_PATH_SMOOTHER__MATH_UTILS_HPP_
#define KINEMATIC_PATH_SMOOTHER__MATH_UTILS_HPP_

#include <cmath>
#include <vector>

#include "Eigen/Core"

namespace kinematic_path_smoother
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-6;

/// 将角度归一化到 (-pi, pi] 附近，避免跨 pi 边界的跳变。
inline double normalizeAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

/// 计算 lhs - rhs 的最短角度差。
inline double angleDifference(double lhs, double rhs)
{
  return normalizeAngle(lhs - rhs);
}

/// 从方向向量生成 yaw；零向量时返回调用方提供的 fallback。
inline double headingFromVector(const Eigen::Vector2d & direction, double fallback = 0.0)
{
  return direction.norm() > kEpsilon ? std::atan2(direction.y(), direction.x()) : fallback;
}

/// 终点位置容差框的坐标系朝向。
///
/// 固定终点朝向时使用用户给定目标方向；不固定时使用末段参考路径方向，
/// 让纵向/横向容差跟随停车段几何方向。
inline double goalFrameHeading(
  const std::vector<Eigen::Vector2d> & points,
  double requested_goal_heading,
  bool keep_goal_orientation)
{
  if (keep_goal_orientation || points.size() < 2) {
    return requested_goal_heading;
  }

  const Eigen::Vector2d delta = points.back() - points[points.size() - 2];
  return delta.norm() > kEpsilon ? std::atan2(delta.y(), delta.x()) : requested_goal_heading;
}

/// 自动微分版本的角度归一化。
template<typename T>
inline T normalizedAngle(T angle)
{
  using std::atan2;
  using std::cos;
  using std::sin;
  return atan2(sin(angle), cos(angle));
}

/// 自动微分版本的最短角度差。
template<typename T>
inline T angularError(T lhs, T rhs)
{
  return normalizedAngle(lhs - rhs);
}

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__MATH_UTILS_HPP_
