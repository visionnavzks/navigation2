#ifndef SMOOTHER_CLOTHOID__UTILS_HPP_
#define SMOOTHER_CLOTHOID__UTILS_HPP_

#include <ceres/ceres.h>
#include <cmath>
#include <limits>
#include <vector>
#include "Eigen/Core"

#if defined(USE_OLD_CERES_API)
  #define CERES_ISINF(x) ceres::IsInfinite(x)
#else
  #define CERES_ISINF(x) ceres::isinf(x)
#endif

namespace smoother_clothoid
{

constexpr double EPSILON = 0.0001;
constexpr double PI = 3.14159265358979323846;

inline double goalPositionFrameHeading(
  const std::vector<Eigen::Vector2d> & reference_points,
  double end_theta,
  bool keep_goal_orientation)
{
  if (keep_goal_orientation || reference_points.size() < 2) return end_theta;
  const Eigen::Vector2d delta = reference_points.back() - reference_points[reference_points.size() - 2];
  if (delta.norm() <= EPSILON) return end_theta;
  return std::atan2(delta.y(), delta.x());
}

template<typename T>
inline Eigen::Matrix<T, 2, 1> arcCenter(
  Eigen::Matrix<T, 2, 1> pt_prev,
  Eigen::Matrix<T, 2, 1> pt,
  Eigen::Matrix<T, 2, 1> pt_next,
  bool is_cusp)
{
  Eigen::Matrix<T, 2, 1> d1 = pt - pt_prev;
  Eigen::Matrix<T, 2, 1> d2 = pt_next - pt;
  if (is_cusp) { d2 = -d2; pt_next = pt + d2; }
  T det = d1[0] * d2[1] - d1[1] * d2[0];
  if (ceres::abs(det) < (T)1e-4)
    return Eigen::Matrix<T, 2, 1>((T)std::numeric_limits<double>::infinity(), (T)std::numeric_limits<double>::infinity());
  Eigen::Matrix<T, 2, 1> mid1 = (pt_prev + pt) / (T)2;
  Eigen::Matrix<T, 2, 1> mid2 = (pt + pt_next) / (T)2;
  Eigen::Matrix<T, 2, 1> n1(-d1[1], d1[0]);
  Eigen::Matrix<T, 2, 1> n2(-d2[1], d2[0]);
  T det1 = (mid1[0] + n1[0]) * mid1[1] - (mid1[1] + n1[1]) * mid1[0];
  T det2 = (mid2[0] + n2[0]) * mid2[1] - (mid2[1] + n2[1]) * mid2[0];
  return Eigen::Matrix<T, 2, 1>((det1 * n2[0] - det2 * n1[0]) / det, (det1 * n2[1] - det2 * n1[1]) / det);
}

template<typename T>
inline Eigen::Matrix<T, 2, 1> tangentDir(
  Eigen::Matrix<T, 2, 1> pt_prev,
  Eigen::Matrix<T, 2, 1> pt,
  Eigen::Matrix<T, 2, 1> pt_next,
  bool is_cusp)
{
  Eigen::Matrix<T, 2, 1> center = arcCenter(pt_prev, pt, pt_next, is_cusp);
  if (CERES_ISINF(center[0])) {
    Eigen::Matrix<T, 2, 1> d1 = pt - pt_prev;
    Eigen::Matrix<T, 2, 1> d2 = pt_next - pt;
    if (is_cusp) { d2 = -d2; pt_next = pt + d2; }
    Eigen::Matrix<T, 2, 1> result(pt_next[0] - pt_prev[0], pt_next[1] - pt_prev[1]);
    if (result[0] == 0.0 && result[1] == 0.0) return Eigen::Matrix<T, 2, 1>(d1[1], -d1[0]);
    return result;
  }
  return Eigen::Matrix<T, 2, 1>(center[1] - pt[1], pt[0] - center[0]);
}

}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__UTILS_HPP_
