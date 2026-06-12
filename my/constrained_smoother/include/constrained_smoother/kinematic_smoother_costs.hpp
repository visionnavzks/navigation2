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
// limitations under the License.

#ifndef CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_
#define CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "Eigen/Core"

#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/options.hpp"

namespace constrained_smoother
{
namespace kinematic_smoother_detail
{

/**
 * @brief 相邻路径点之间的运动学过渡代价函数
 *
 * 该代价函数约束路径平滑器中相邻两个状态点之间的运动学一致性。
 * 每个状态向量包含 5 个分量：[x, y, theta, kappa, ds]
 *   - x, y    ：世界坐标系下的位置
 *   - theta   ：朝向角（弧度）
 *   - kappa   ：当前点的曲率（1/转弯半径）
 *   - ds      ：从当前点到下一个点的弧长步长
 *
 * 运动学模型（正向/倒车均适用）：
 *   采用梯形曲率积分（当前曲率与下一曲率的均值）预测朝向角变化，
 *   采用中间朝向角预测位置（Euler midpoint 近似）。
 *
 * 输出 7 个残差：
 *   [0] 位置 x 误差（运动学模型约束）
 *   [1] 位置 y 误差（运动学模型约束）
 *   [2] 朝向角误差（运动学模型约束）
 *   [3] 平均曲率惩罚（鼓励路径趋向直行）
 *   [4] 曲率变化率惩罚（鼓励曲率平滑，以弧长归一化）
 *   [5] 步长误差（约束相邻点间距接近目标步长）
 *   [6] 总长度惩罚（直接压缩 ds，鼓励整条路径更短）
 *
 * 尖点（cusp）段特殊处理：
 *   尖点处为前进/倒退切换点，此段强制相邻点保持相同位置和朝向，
 *   并惩罚非零步长（约束尖点处车辆静止）。
 */
class TransitionCostFunctor
{
public:
  /**
   * @param gear                 档位，>=0 为前进，<0 为倒退
   * @param is_cusp_segment      是否为前进/倒退切换的尖点段
   * @param model_weight         运动学模型约束权重（位置/朝向残差的系数）
   * @param curvature_weight     曲率大小惩罚权重（鼓励路径趋直）
   * @param curvature_rate_weight 曲率变化率惩罚权重（鼓励曲率连续平滑）
   * @param spacing_weight       步长误差惩罚权重
   * @param length_weight        总长度惩罚权重（直接压缩 ds）
   * @param fix_weight           尖点段强固定约束权重
   * @param target_spacing       期望的相邻点弧长步长（米）
   */
  TransitionCostFunctor(
    double gear,
    bool is_cusp_segment,
    double model_weight,
    double curvature_weight,
    double curvature_rate_weight,
    double spacing_weight,
      double length_weight,
    double fix_weight,
    double target_spacing)
  : gear_(gear),
    is_cusp_segment_(is_cusp_segment),
    model_weight_(model_weight),
    curvature_weight_(curvature_weight),
    curvature_rate_weight_(curvature_rate_weight),
    spacing_weight_(spacing_weight),
    length_weight_(length_weight),
    fix_weight_(fix_weight),
    target_spacing_(target_spacing)
  {
  }

  /**
   * @brief 创建用于 Ceres 自动微分的代价函数对象
   *        模板参数：7 个残差，两个参数块各 5 个分量（current 和 next）
   */
  static ceres::CostFunction * Create(
    double gear,
    bool is_cusp_segment,
    double model_weight,
    double curvature_weight,
    double curvature_rate_weight,
    double spacing_weight,
    double length_weight,
    double fix_weight,
    double target_spacing)
  {
    return new ceres::AutoDiffCostFunction<TransitionCostFunctor, 7, 5, 5>(
      new TransitionCostFunctor(
        gear,
        is_cusp_segment,
        model_weight,
        curvature_weight,
        curvature_rate_weight,
        spacing_weight,
        length_weight,
        fix_weight,
        target_spacing));
  }

  /**
   * @brief 计算相邻两点的运动学过渡残差
   * @param current  当前点状态 [x, y, theta, kappa, ds]
   * @param next     下一点状态 [x, y, theta, kappa, ds]（ds 分量不使用）
   * @param residuals 输出残差数组，长度为 7
   */
  template<typename T>
  bool operator()(const T * const current, const T * const next, T * residuals) const
  {
    // 将输出残差映射为 Eigen 向量，便于操作，初始化为零
    Eigen::Map<Eigen::Matrix<T, 7, 1>> residual(residuals);
    residual.setZero();

    // 解包当前点状态
    const T x = current[0];      // 当前点 x 坐标
    const T y = current[1];      // 当前点 y 坐标
    const T theta = current[2];  // 当前点朝向角
    const T kappa = current[3];  // 当前点曲率
    const T ds = current[4];     // 当前点到下一点的弧长步长

    // 解包下一点状态（ds 不参与计算）
    const T next_x = next[0];
    const T next_y = next[1];
    const T next_theta = next[2];
    const T next_kappa = next[3];

    // ---- 尖点段特殊处理 ----
    // 前进/倒退切换处，强制相邻点静止（位置/朝向不变，步长为零）
    if (is_cusp_segment_) {
      residual[0] = T(fix_weight_) * (next_x - x);                      // 强制 x 不变
      residual[1] = T(fix_weight_) * (next_y - y);                      // 强制 y 不变
      residual[2] = T(fix_weight_) * angleDiff(next_theta, theta);      // 强制朝向不变
      residual[5] = T(spacing_weight_) * T(10.0) * ds;                  // 强惩罚非零步长
      residual[6] = T(length_weight_) * ds;                             // 直接压缩总长度
      return true;
    }

    // ---- 正常运动学模型 ----
    // 行驶方向：前进为 +1，倒退为 -1
    const T direction = gear_ >= 0.0 ? T(1.0) : T(-1.0);

    // 用梯形曲率积分预测下一点朝向角（使用当前和下一曲率的均值）
    // theta_pred = theta + direction * ds * (kappa + next_kappa) / 2
    const T theta_pred = theta + direction * ds * (kappa + next_kappa) * T(0.5);

    // 中间朝向角，用于 Euler midpoint 位置预测；与梯形曲率积分保持一致。
    const T theta_mid = (theta + theta_pred) * T(0.5);

    // 用中间朝向角预测下一点位置（Euler midpoint 近似）
    const T x_pred = x + direction * ds * cosValue(theta_mid);
    const T y_pred = y + direction * ds * sinValue(theta_mid);

    // 曲率变化率归一化分母：以弧长平方根归一化，避免小步长时数值爆炸
    const T denom = ds > T(1e-3) ? sqrtValue(ds) : T(0.03);

    // 残差[0][1][2]：运动学模型约束——预测位置/朝向与实际下一点的偏差
    residual[0] = T(model_weight_) * (next_x - x_pred);
    residual[1] = T(model_weight_) * (next_y - y_pred);
    residual[2] = T(model_weight_) * angleDiff(next_theta, theta_pred);

    // 残差[3]：平均曲率惩罚——使路径尽量趋向直行（曲率趋零）
    residual[3] = T(curvature_weight_) * (kappa + next_kappa) * T(0.5);

    // 残差[4]：曲率变化率惩罚——使曲率沿弧长平滑变化（避免急剧转向）
    residual[4] = T(curvature_rate_weight_) * (next_kappa - kappa) / denom;

    // 残差[5]：步长误差——约束相邻点间距接近目标步长，归一化后无量纲
    // 对目标步长做下限保护，避免极端配置导致除零或梯度异常放大。
    const T spacing_ref = T(std::max(target_spacing_, 1e-3));
    residual[5] = T(spacing_weight_) * (ds - spacing_ref) / spacing_ref;

    // 残差[6]：长度惩罚——对每一段 ds 直接施加代价，使总路径长度更短
    residual[6] = T(length_weight_) * ds;
    return true;
  }

private:
  /**
   * @brief 将角度归一化到 (-pi, pi] 区间
   *        使用 atan2(sin, cos) 实现，保证可微性（适用于自动微分）
   */
  template<typename T>
  static T normalizeAngle(T angle)
  {
    using std::atan2;
    using std::cos;
    using std::sin;
    return atan2(sin(angle), cos(angle));
  }

  /**
   * @brief 计算两角度之差并归一化到 (-pi, pi]
   * @return a - b 的归一化角度差
   */
  template<typename T>
  static T angleDiff(T a, T b)
  {
    return normalizeAngle(a - b);
  }

  // 以下三个辅助函数封装 std 数学函数，确保模板 T 可正确推导
  template<typename T>
  static T sinValue(T value)
  {
    using std::sin;
    return sin(value);
  }

  template<typename T>
  static T cosValue(T value)
  {
    using std::cos;
    return cos(value);
  }

  template<typename T>
  static T sqrtValue(T value)
  {
    using std::sqrt;
    return sqrt(value);
  }

  double gear_;                  ///< 档位（前进/倒退）
  bool is_cusp_segment_;         ///< 是否为前进/倒退切换的尖点段
  double model_weight_;          ///< 运动学模型约束权重
  double curvature_weight_;      ///< 曲率大小惩罚权重
  double curvature_rate_weight_; ///< 曲率变化率惩罚权重
  double spacing_weight_;        ///< 步长误差惩罚权重
  double length_weight_;         ///< 总长度惩罚权重
  double fix_weight_;            ///< 尖点段固定约束权重
  double target_spacing_;        ///< 期望相邻点弧长（米）
};

/**
 * @brief 路径端点（起点/终点）的边界约束代价函数
 *
 * 该代价函数用于将路径的起点或终点锚定到参考位置（和朝向）。
 * 对终点还可配置 lon / lat / theta 容差，从而表达“范围停”而不是绝对硬锚定。
 * 每个状态向量包含 5 个分量：[x, y, theta, kappa, ds]
 *
 * 输出 3 个残差：
 *   [0] 目标坐标系 lon 方向位置误差（超出容差才惩罚）
 *   [1] 目标坐标系 lat 方向位置误差（超出容差才惩罚）
 *   [2] 朝向角误差（仅在 keep_orientation=true 时生效，且超出容差才惩罚）
 */
class BoundaryCostFunctor
{
public:
  /**
   * @param reference_point   端点的参考位置（世界坐标系，单位：米）
   * @param target_theta      端点的参考朝向角（弧度）
   * @param keep_orientation  是否强制朝向角与参考一致
   * @param fix_weight        位置/朝向约束权重
   */
  BoundaryCostFunctor(
    const Eigen::Vector2d & reference_point,
    double target_theta,
    bool keep_orientation,
    double longitudinal_tolerance,
    double lateral_tolerance,
    double orientation_tolerance,
    double fix_weight)
  : reference_point_(reference_point),
    target_theta_(target_theta),
    keep_orientation_(keep_orientation),
    longitudinal_tolerance_(std::max(longitudinal_tolerance, 0.0)),
    lateral_tolerance_(std::max(lateral_tolerance, 0.0)),
    orientation_tolerance_(std::max(orientation_tolerance, 0.0)),
    fix_weight_(fix_weight)
  {
  }

  /**
   * @brief 创建用于 Ceres 自动微分的代价函数对象
   *        模板参数：3 个残差，1 个参数块（5 个分量）
   */
  static ceres::CostFunction * Create(
    const Eigen::Vector2d & reference_point,
    double target_theta,
    bool keep_orientation,
    double longitudinal_tolerance,
    double lateral_tolerance,
    double orientation_tolerance,
    double fix_weight)
  {
    return new ceres::AutoDiffCostFunction<BoundaryCostFunctor, 3, 5>(
      new BoundaryCostFunctor(
        reference_point,
        target_theta,
        keep_orientation,
        longitudinal_tolerance,
        lateral_tolerance,
        orientation_tolerance,
        fix_weight));
  }

  /**
   * @brief 计算端点约束残差
   * @param state     端点状态 [x, y, theta, kappa, ds]
   * @param residuals 输出残差数组，长度为 3
   */
  template<typename T>
  bool operator()(const T * const state, T * residuals) const
  {
    using std::abs;

    const T dx = state[0] - T(reference_point_.x());
    const T dy = state[1] - T(reference_point_.y());
    const T cos_theta = T(std::cos(target_theta_));
    const T sin_theta = T(std::sin(target_theta_));
    const T lon_error = cos_theta * dx + sin_theta * dy;
    const T lat_error = -sin_theta * dx + cos_theta * dy;
    const T lon_violation = abs(lon_error) - T(longitudinal_tolerance_);
    const T lat_violation = abs(lat_error) - T(lateral_tolerance_);

    residuals[0] = lon_violation > T(0.0) ? T(fix_weight_) * lon_violation : T(0.0);
    residuals[1] = lat_violation > T(0.0) ? T(fix_weight_) * lat_violation : T(0.0);

    // 残差[2]：若启用朝向约束，只在超出允许朝向容差时惩罚
    if (keep_orientation_) {
      const T heading_error = abs(angleDiff(state[2], T(target_theta_)));
      const T heading_violation = heading_error - T(orientation_tolerance_);
      residuals[2] = heading_violation > T(0.0) ? T(fix_weight_) * heading_violation : T(0.0);
    } else {
      residuals[2] = T(0.0);
    }
    return true;
  }

private:
  /// @brief 将角度归一化到 (-pi, pi]
  template<typename T>
  static T normalizeAngle(T angle)
  {
    using std::atan2;
    using std::cos;
    using std::sin;
    return atan2(sin(angle), cos(angle));
  }

  /// @brief 计算两角度之差并归一化到 (-pi, pi]
  template<typename T>
  static T angleDiff(T a, T b)
  {
    return normalizeAngle(a - b);
  }

  Eigen::Vector2d reference_point_; ///< 参考位置（世界坐标）
  double target_theta_;             ///< 参考朝向角（弧度）
  bool keep_orientation_;           ///< 是否约束朝向角
  double longitudinal_tolerance_;   ///< lon 方向允许容差（米）
  double lateral_tolerance_;        ///< lat 方向允许容差（米）
  double orientation_tolerance_;    ///< 朝向允许容差（弧度）
  double fix_weight_;               ///< 约束权重
};

/**
 * @brief 参考路径点吸引代价函数
 *
 * 该代价函数使平滑后的路径点在位置上靠近原始参考路径点，
 * 防止优化过程中路径发生大幅度漂移。
 *
 * 输出 2 个残差：
 *   [0] x 方向偏差（加权）
 *   [1] y 方向偏差（加权）
 */
class ReferenceCostFunctor
{
public:
  /**
   * @param reference_point   原始参考路径点的位置（世界坐标，单位：米）
   * @param reference_weight  吸引权重（越大则平滑路径越靠近原始路径）
   */
  ReferenceCostFunctor(const Eigen::Vector2d & reference_point, double reference_weight)
  : reference_point_(reference_point), reference_weight_(reference_weight)
  {
  }

  /**
   * @brief 创建用于 Ceres 自动微分的代价函数对象
   *        模板参数：2 个残差，1 个参数块（5 个分量）
   */
  static ceres::CostFunction * Create(
    const Eigen::Vector2d & reference_point,
    double reference_weight)
  {
    return new ceres::AutoDiffCostFunction<ReferenceCostFunctor, 2, 5>(
      new ReferenceCostFunctor(reference_point, reference_weight));
  }

  /**
   * @brief 计算路径点与参考点的位置偏差残差
   * @param state     路径点状态 [x, y, theta, kappa, ds]
   * @param residuals 输出残差数组，长度为 2
   */
  template<typename T>
  bool operator()(const T * const state, T * residuals) const
  {
    // 计算当前路径点与参考点在 x、y 方向上的偏差
    const T dx = state[0] - T(reference_point_.x());
    const T dy = state[1] - T(reference_point_.y());
    // 施加参考权重，鼓励路径点靠近参考位置
    residuals[0] = T(reference_weight_) * dx;
    residuals[1] = T(reference_weight_) * dy;
    return true;
  }

private:
  Eigen::Vector2d reference_point_; ///< 参考路径点位置（世界坐标）
  double reference_weight_;         ///< 参考吸引权重
};

/**
 * @brief 障碍物避让代价函数（基于 ESDF 代价地图）
 *
 * 该代价函数利用预先计算的欧几里得有符号距离场（ESDF）对路径点与障碍物的
 * 距离进行惩罚，确保平滑后的路径与障碍物保持安全距离。
 *
 * 距离惩罚模型：
 *   - 当到障碍物表面的距离 >= obstacle_safe_distance 时，残差为 0
 *   - 当距离 < obstacle_safe_distance 时，残差为一次 hinge；
 *     Ceres 对 residual 平方后得到二次净空代价
 *   - 当路径点超出代价地图范围时，返回常数边界残差
 *
 * 支持多检测点模式（cost_check_points）：
 *   可指定多个相对于路径点坐标系的检测点（如机器人轮廓上的多个点），
 *   每个检测点独立计算障碍物惩罚，输出为多个残差。
 *   若 cost_check_points 为空，则仅对路径点本身计算一个残差。
 *
 * 调用方传入统一的障碍物权重；该 functor 只负责按权重计算残差。
 */
class ObstacleCostFunctor
{
public:
  /**
   * @param obstacle_weight 该路径点解析后的障碍物残差权重
   * @param costmap         代价地图指针，提供地图元数据（原点、分辨率、尺寸）
   * @param params          平滑器参数（障碍物权重、安全距离、检测点等）
   * @param esdf_grid       共享的 ESDF Grid2D 存储，保证插值器引用的底层网格生命周期
   * @param esdf_interpolator 共享的 ESDF 双三次插值器
   */
  ObstacleCostFunctor(
    double obstacle_weight,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::shared_ptr<ceres::Grid2D<double>> & esdf_grid,
    const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> & esdf_interpolator)
  : costmap_origin_(costmap->getOriginX(), costmap->getOriginY()),
    costmap_resolution_(costmap->getResolution()),
    size_x_(costmap->getSizeInCellsX()),
    size_y_(costmap->getSizeInCellsY()),
    obstacle_safe_distance_(std::max(params.obstacle_safe_distance, 1e-6)),
    cost_check_radius_(std::max(params.cost_check_radius, 0.0)),
    obstacle_weight_(std::max(obstacle_weight, 0.0)),
    cost_check_points_(params.cost_check_points),
    esdf_grid_(esdf_grid),
    esdf_interpolator_(esdf_interpolator)
  {
    if (!cost_check_points_.empty() && cost_check_points_.size() % 3 != 0) {
      throw std::invalid_argument("cost_check_points size must be a multiple of 3");
    }
  }

  /**
   * @brief 返回该代价函数的残差数量
   *        若无检测点则为 1，否则为检测点数量（每个检测点一个残差）
   */
  int numResiduals() const
  {
    return cost_check_points_.empty() ? 1 : static_cast<int>(cost_check_points_.size() / 3);
  }

  /**
   * @brief 创建用于 Ceres 自动微分的动态代价函数对象
   *        残差数量在运行时由 numResiduals() 决定（动态残差）
   */
  static ceres::CostFunction * Create(
    double obstacle_weight,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::shared_ptr<ceres::Grid2D<double>> & esdf_grid,
    const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> & esdf_interpolator)
  {
    auto * functor = new ObstacleCostFunctor(
      obstacle_weight, costmap, params, esdf_grid, esdf_interpolator);
    auto * cost_function = new ceres::DynamicAutoDiffCostFunction<ObstacleCostFunctor>(functor);
    cost_function->AddParameterBlock(5);  // 状态向量维度为 5
    cost_function->SetNumResiduals(functor->numResiduals());
    return cost_function;
  }

  /**
   * @brief 计算障碍物代价残差
   * @param parameters  参数块数组，parameters[0] 为路径点状态 [x, y, theta, kappa, ds]
   * @param residuals   输出残差数组，长度为 numResiduals()
   */
  template<typename T>
  bool operator()(const T * const * parameters, T * residuals) const
  {
    const T * state = parameters[0];
    const T x = state[0];      // 路径点 x 坐标
    const T y = state[1];      // 路径点 y 坐标
    const T theta = state[2];  // 路径点朝向角

    const T pose_weight = T(obstacle_weight_);

    // 若无多点检测，直接对路径点坐标计算单个障碍物残差
    if (cost_check_points_.empty()) {
      residuals[0] = pose_weight * obstaclePenalty(x, y);
      return true;
    }

    // 多点检测：将局部坐标系下的检测点旋转到世界坐标系，分别计算障碍物惩罚
    // cost_check_points_ 格式：[local_x, local_y, weight, local_x, local_y, weight, ...]
    const T cos_theta = cosValue(theta);
    const T sin_theta = sinValue(theta);
    int residual_index = 0;
    for (size_t offset = 0; offset + 2 < cost_check_points_.size(); offset += 3) {
      const T local_x = T(cost_check_points_[offset + 0]);  // 局部坐标 x（沿车辆前方）
      const T local_y = T(cost_check_points_[offset + 1]);  // 局部坐标 y（沿车辆左方）
      const T point_weight = T(cost_check_points_[offset + 2]);  // 该检测点的额外权重
      // 旋转变换：局部坐标 -> 世界坐标
      const T world_x = x + cos_theta * local_x - sin_theta * local_y;
      const T world_y = y + sin_theta * local_x + cos_theta * local_y;
      residuals[residual_index++] = pose_weight * point_weight * obstaclePenalty(world_x, world_y);
    }
    return true;
  }

private:
  /**
   * @brief 计算给定世界坐标点处的障碍物惩罚值
   *
   * 从 ESDF 插值器中查询该点到最近障碍物的距离，
   * 并根据距离与安全距离的关系计算一次 hinge residual：
   *   residual = (safe_dist - (esdf_dist - robot_radius)) / safe_dist
   *              当 esdf_dist - robot_radius < safe_dist 时生效，否则为 0
   *
   * @param world_x  世界坐标 x（米）
   * @param world_y  世界坐标 y（米）
   * @return 障碍物 residual；Ceres 会进一步平方形成二次代价
   */
  template<typename T>
  T obstaclePenalty(T world_x, T world_y) const
  {
    // 将世界坐标转换为代价地图格坐标
    const T grid_x = (world_x - T(costmap_origin_.x())) / T(costmap_resolution_);
    const T grid_y = (world_y - T(costmap_origin_.y())) / T(costmap_resolution_);

    // 若超出可插值边界，返回常数边界残差。
    if (grid_x < T(1.5) || grid_y < T(1.5) ||
      grid_x >= T(static_cast<double>(size_x_) - 1.5) || grid_y >= T(static_cast<double>(size_y_) - 1.5))
    {
      return T(1.0);
    }

    // 通过双三次插值查询该格坐标处的 ESDF 距离值（单位：格）
    // 注意：插值器的坐标约定为 (row, col)，且原点偏移 0.5 格
    T distance = T(0.0);
    esdf_interpolator_->Evaluate(grid_y - T(0.5), grid_x - T(0.5), &distance);

    // 减去机器人半径（cost_check_radius_），得到到障碍物表面的真实距离
    const T surface_distance = distance - T(cost_check_radius_);

    // 若到障碍物表面距离已满足安全要求，惩罚为零
    if (surface_distance >= T(obstacle_safe_distance_)) {
      return T(0.0);
    }

    // 一次 hinge residual；Ceres 平方 residual 后得到二次净空代价。
    return (T(obstacle_safe_distance_) - surface_distance) / T(obstacle_safe_distance_);
  }

  // 以下两个辅助函数封装 std 数学函数，确保模板 T 可正确推导
  template<typename T>
  static T sinValue(T value)
  {
    using std::sin;
    return sin(value);
  }

  template<typename T>
  static T cosValue(T value)
  {
    using std::cos;
    return cos(value);
  }

  Eigen::Vector2d costmap_origin_;   ///< 代价地图原点（世界坐标，米）
  double costmap_resolution_;        ///< 代价地图分辨率（米/格）
  unsigned int size_x_;              ///< 代价地图 x 方向格数
  unsigned int size_y_;              ///< 代价地图 y 方向格数
  double obstacle_safe_distance_;    ///< 障碍物安全距离阈值（米），低于此值时施加惩罚
  double cost_check_radius_;         ///< 机器人检测半径（米），从 ESDF 距离中减去
  double obstacle_weight_;           ///< 当前路径点解析后的障碍物惩罚权重（平方根形式）
  std::vector<double> cost_check_points_; ///< 多检测点列表，格式：[lx,ly,w, lx,ly,w, ...]
  std::shared_ptr<ceres::Grid2D<double>> esdf_grid_;  ///< 保持插值器底层网格存活
  std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> esdf_interpolator_; ///< 双三次插值器
};

}  // namespace kinematic_smoother_detail
}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_COSTS_HPP_
