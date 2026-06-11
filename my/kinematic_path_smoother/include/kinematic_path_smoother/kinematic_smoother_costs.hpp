#ifndef KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_
#define KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

#include "kinematic_path_smoother/costmap2d.hpp"
#include "kinematic_path_smoother/math_utils.hpp"
#include "kinematic_path_smoother/options.hpp"

namespace kinematic_path_smoother
{
namespace detail
{

/// 相邻两个运动学状态之间的转移残差。
///
/// 参数块均为 5 维：[x, y, theta, kappa, ds]。
/// 输出 7 个残差：
/// 0-2 为运动学模型位置/朝向误差，3 为曲率正则，4 为曲率变化率，
/// 5 为目标间距约束，6 为路径长度惩罚。cusp 段会退化为位置/朝向保持约束。
class MotionCost
{
public:
  MotionCost(
    double gear,
    bool cusp,
    double model_weight,
    double curvature_weight,
    double curvature_rate_weight,
    double spacing_weight,
    double length_weight,
    double fix_weight,
    double target_spacing)
  : gear_(gear),
    cusp_(cusp),
    model_weight_(model_weight),
    curvature_weight_(curvature_weight),
    curvature_rate_weight_(curvature_rate_weight),
    spacing_weight_(spacing_weight),
    length_weight_(length_weight),
    fix_weight_(fix_weight),
    target_spacing_(std::max(target_spacing, 1e-3))
  {
  }

  static ceres::CostFunction * Create(
    double gear,
    bool cusp,
    double model_weight,
    double curvature_weight,
    double curvature_rate_weight,
    double spacing_weight,
    double length_weight,
    double fix_weight,
    double target_spacing)
  {
    return new ceres::AutoDiffCostFunction<MotionCost, 7, 5, 5>(
      new MotionCost(
        gear,
        cusp,
        model_weight,
        curvature_weight,
        curvature_rate_weight,
        spacing_weight,
        length_weight,
        fix_weight,
        target_spacing));
  }

  template<typename T>
  bool operator()(const T * const current, const T * const next, T * residuals) const
  {
    for (int i = 0; i < 7; ++i) {
      residuals[i] = T(0.0);
    }

    const T x = current[0];
    const T y = current[1];
    const T theta = current[2];
    const T kappa = current[3];
    const T ds = current[4];

    const T next_x = next[0];
    const T next_y = next[1];
    const T next_theta = next[2];
    const T next_kappa = next[3];

    // 前进/倒车切换处不应该产生实际位移，只约束两个 knot 保持重合。
    if (cusp_) {
      residuals[0] = T(fix_weight_) * (next_x - x);
      residuals[1] = T(fix_weight_) * (next_y - y);
      residuals[2] = T(fix_weight_) * angularError(next_theta, theta);
      residuals[5] = T(spacing_weight_) * T(10.0) * ds;
      residuals[6] = T(length_weight_) * ds;
      return true;
    }

    using std::cos;
    using std::sin;
    using std::sqrt;
    const T direction = gear_ < 0.0 ? T(-1.0) : T(1.0);
    // 位置和朝向都使用同一条梯形曲率积分近似。
    const T theta_next = theta + direction * ds * (kappa + next_kappa) * T(0.5);
    const T theta_mid = (theta + theta_next) * T(0.5);
    const T predicted_x = x + direction * ds * cos(theta_mid);
    const T predicted_y = y + direction * ds * sin(theta_mid);
    const T rate_scale = ds > T(1e-3) ? sqrt(ds) : T(0.03);

    residuals[0] = T(model_weight_) * (next_x - predicted_x);
    residuals[1] = T(model_weight_) * (next_y - predicted_y);
    residuals[2] = T(model_weight_) * angularError(next_theta, theta_next);
    residuals[3] = T(curvature_weight_) * (kappa + next_kappa) * T(0.5);
    residuals[4] = T(curvature_rate_weight_) * (next_kappa - kappa) / rate_scale;
    residuals[5] = T(spacing_weight_) * (ds - T(target_spacing_)) / T(target_spacing_);
    residuals[6] = T(length_weight_) * ds;
    return true;
  }

private:
  double gear_;
  bool cusp_;
  double model_weight_;
  double curvature_weight_;
  double curvature_rate_weight_;
  double spacing_weight_;
  double length_weight_;
  double fix_weight_;
  double target_spacing_;
};

/// 起点/终点边界约束。
///
/// 位置误差会投影到 frame_heading 定义的 lon/lat 坐标系，并按容差做 hinge
/// loss：在容差内残差为 0，超出后线性增长。朝向约束同理。
class EndpointCost
{
public:
  EndpointCost(
    const Eigen::Vector2d & reference,
    double frame_heading,
    bool keep_heading,
    double longitudinal_tolerance,
    double lateral_tolerance,
    double heading_tolerance,
    double weight,
    bool stop)
  : reference_(reference),
    frame_heading_(frame_heading),
    keep_heading_(keep_heading),
    longitudinal_tolerance_(std::max(longitudinal_tolerance, 0.0)),
    lateral_tolerance_(std::max(lateral_tolerance, 0.0)),
    heading_tolerance_(std::max(heading_tolerance, 0.0)),
    weight_(weight),
    stop_(stop)
  {
  }

  static ceres::CostFunction * Create(
    const Eigen::Vector2d & reference,
    double frame_heading,
    bool keep_heading,
    double longitudinal_tolerance,
    double lateral_tolerance,
    double heading_tolerance,
    double weight,
    bool stop)
  {
    return new ceres::AutoDiffCostFunction<EndpointCost, 4, 5>(
      new EndpointCost(
        reference,
        frame_heading,
        keep_heading,
        longitudinal_tolerance,
        lateral_tolerance,
        heading_tolerance,
        weight,
        stop));
  }

  template<typename T>
  bool operator()(const T * const state, T * residuals) const
  {
    using std::abs;
    const T dx = state[0] - T(reference_.x());
    const T dy = state[1] - T(reference_.y());
    const T c = T(std::cos(frame_heading_));
    const T s = T(std::sin(frame_heading_));
    const T lon = c * dx + s * dy;
    const T lat = -s * dx + c * dy;
    const T lon_over = abs(lon) - T(longitudinal_tolerance_);
    const T lat_over = abs(lat) - T(lateral_tolerance_);

    residuals[0] = lon_over > T(0.0) ? T(weight_) * lon_over : T(0.0);
    residuals[1] = lat_over > T(0.0) ? T(weight_) * lat_over : T(0.0);

    if (keep_heading_) {
      const T heading_over = abs(angularError(state[2], T(frame_heading_))) - T(heading_tolerance_);
      residuals[2] = heading_over > T(0.0) ? T(weight_) * heading_over : T(0.0);
    } else {
      residuals[2] = T(0.0);
    }

    residuals[3] = stop_ ? T(weight_) * state[4] : T(0.0);
    return true;
  }

private:
  Eigen::Vector2d reference_;
  double frame_heading_;
  bool keep_heading_;
  double longitudinal_tolerance_;
  double lateral_tolerance_;
  double heading_tolerance_;
  double weight_;
  bool stop_;
};

/// 优化点到原始参考点的软吸附残差。
class ReferenceCost
{
public:
  ReferenceCost(const Eigen::Vector2d & reference, double weight)
  : reference_(reference), weight_(weight)
  {
  }

  static ceres::CostFunction * Create(const Eigen::Vector2d & reference, double weight)
  {
    return new ceres::AutoDiffCostFunction<ReferenceCost, 2, 5>(
      new ReferenceCost(reference, weight));
  }

  template<typename T>
  bool operator()(const T * const state, T * residuals) const
  {
    residuals[0] = T(weight_) * (state[0] - T(reference_.x()));
    residuals[1] = T(weight_) * (state[1] - T(reference_.y()));
    return true;
  }

private:
  Eigen::Vector2d reference_;
  double weight_;
};

/// 基于 ESDF 的障碍物残差。
///
/// 若 footprint_points 为空，则检查状态中心点；否则按 [lx, ly, weight]
/// 三元组把局部检查点变换到世界坐标，每个检查点产生一个残差。
/// penalty() 返回线性 hinge residual，Ceres 平方后形成二次净空代价。
class ObstacleCost
{
public:
  using Grid = ceres::Grid2D<double>;
  using Interpolator = ceres::BiCubicInterpolator<Grid>;

  ObstacleCost(
    bool cusp_pose,
    const Costmap2D & costmap,
    const SmootherParams & params,
    std::shared_ptr<Grid> grid,
    std::shared_ptr<Interpolator> interpolator)
  : origin_(costmap.getOriginX(), costmap.getOriginY()),
    resolution_(costmap.getResolution()),
    size_x_(costmap.getSizeInCellsX()),
    size_y_(costmap.getSizeInCellsY()),
    safe_distance_(std::max(params.obstacle_safe_distance, 1e-6)),
    footprint_radius_(std::max(params.footprint_radius, 0.0)),
    weight_(std::sqrt(std::max(params.obstacle_weight, 0.0))),
    cusp_weight_(std::sqrt(std::max(params.cusp_obstacle_weight, params.obstacle_weight))),
    cusp_pose_(cusp_pose),
    footprint_points_(params.footprint_points),
    grid_(std::move(grid)),
    interpolator_(std::move(interpolator))
  {
    if (!footprint_points_.empty() && footprint_points_.size() % 3 != 0) {
      throw std::invalid_argument("footprint_points size must be a multiple of 3");
    }
  }

  int residualCount() const
  {
    return footprint_points_.empty() ? 1 : static_cast<int>(footprint_points_.size() / 3);
  }

  /// 使用 DynamicAutoDiffCostFunction，因为 footprint 检查点数量运行期才确定。
  static ceres::CostFunction * Create(
    bool cusp_pose,
    const Costmap2D & costmap,
    const SmootherParams & params,
    std::shared_ptr<Grid> grid,
    std::shared_ptr<Interpolator> interpolator)
  {
    auto * functor = new ObstacleCost(
      cusp_pose, costmap, params, std::move(grid), std::move(interpolator));
    auto * cost = new ceres::DynamicAutoDiffCostFunction<ObstacleCost>(functor);
    cost->AddParameterBlock(5);
    cost->SetNumResiduals(functor->residualCount());
    return cost;
  }

  template<typename T>
  bool operator()(const T * const * parameters, T * residuals) const
  {
    const T * state = parameters[0];
    const T pose_weight = T(cusp_pose_ ? cusp_weight_ : weight_);

    if (footprint_points_.empty()) {
      residuals[0] = pose_weight * penalty(state[0], state[1]);
      return true;
    }

    using std::cos;
    using std::sin;
    const T c = cos(state[2]);
    const T s = sin(state[2]);
    int out = 0;
    for (std::size_t i = 0; i + 2 < footprint_points_.size(); i += 3) {
      const T local_x = T(footprint_points_[i]);
      const T local_y = T(footprint_points_[i + 1]);
      const T point_weight = T(footprint_points_[i + 2]);
      const T world_x = state[0] + c * local_x - s * local_y;
      const T world_y = state[1] + s * local_x + c * local_y;
      residuals[out++] = pose_weight * point_weight * penalty(world_x, world_y);
    }
    return true;
  }

private:
  template<typename T>
  T penalty(T world_x, T world_y) const
  {
    // Ceres Grid2D 以格坐标插值；ESDF 数值单位是米。
    const T grid_x = (world_x - T(origin_.x())) / T(resolution_);
    const T grid_y = (world_y - T(origin_.y())) / T(resolution_);
    if (grid_x < T(1.5) || grid_y < T(1.5) ||
      grid_x >= T(static_cast<double>(size_x_) - 1.5) ||
      grid_y >= T(static_cast<double>(size_y_) - 1.5))
    {
      return T(1.0);
    }

    T distance = T(0.0);
    interpolator_->Evaluate(grid_y - T(0.5), grid_x - T(0.5), &distance);
    // 单圆 footprint 模式下，ESDF 到障碍物中心点距离需要扣除半径。
    const T clearance = distance - T(footprint_radius_);
    if (clearance >= T(safe_distance_)) {
      return T(0.0);
    }

    return (T(safe_distance_) - clearance) / T(safe_distance_);
  }

  Eigen::Vector2d origin_;
  double resolution_;
  unsigned int size_x_;
  unsigned int size_y_;
  double safe_distance_;
  double footprint_radius_;
  double weight_;
  double cusp_weight_;
  bool cusp_pose_;
  std::vector<double> footprint_points_;
  std::shared_ptr<Grid> grid_;
  std::shared_ptr<Interpolator> interpolator_;
};

}  // namespace detail
}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_
