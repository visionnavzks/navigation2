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

#include "kinematic_smoother/costmap2d.hpp"
#include "kinematic_smoother/exceptions.hpp"
#include "kinematic_smoother/options.hpp"
#include "kinematic_smoother/state_layout.hpp"
#include "kinematic_smoother/utils.hpp"

namespace kinematic_smoother
{

/// 求解后统一执行的硬性校验器。
///
/// 这个对象故意放在求解器之外，目的是把“数值上收敛”与“工程上可交付”分开：
/// builder 负责构建优化问题，validator 负责决定解是否真的能被接受。
class SmootherValidator
{
public:
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

  /// 运动学版 smoother 的总校验入口。
  ///
  /// 串行执行六道硬性检查，任一失败立刻返回 `false` 并把原因写入
  /// `SmoothingFailureInfo`（若提供）：
  ///   1. 形状 —— `variables.size()` 是否等于 `state_count * 5`。
  ///   2. 有限值 —— 每个状态 (x, y, theta, kappa, ds) 是否都是有限数。
  ///   3. 边界 —— 起终点位置/朝向是否落在调用方声明的容差内。
  ///   4. 段一致性 —— 相邻状态间的位移、方向与 gear/cusp 语义是否自洽。
  ///   5. 曲率 —— 显式 `kappa` 与相邻姿态推导出的几何曲率是否都
  ///      不超过 `params.max_curvature`。
  ///   6. 净空 —— 在启用障碍物项的前提下，每个状态的足迹采样点
  ///      是否都满足 `costmap` 的最小 ESDF 净空。
  bool validateKinematicSolution(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    if (request.variables.size() != request.state_count * KinematicStateLayout::Size) {
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
    if (!validateKinematicCurvatureConstraint(request, failure)) {
      return false;
    }
    if (!validateKinematicObstacleClearance(request, failure)) {
      return false;
    }

    return true;
  }

private:
  // ---- Shared numeric helpers ----

  static double radiansToDegrees(double radians)
  {
    return radians * 180.0 / PI;
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
          << "Kinematic smoother violated the maximum curvature constraint during post-validation"
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
    for (size_t index = 0; index < KinematicStateLayout::Size; ++index) {
      if (!std::isfinite(state[index])) {
        return false;
      }
    }
    return true;
  }

  // ---- Kinematic smoother validation ----

  /// 逐状态检查解向量中每个分量 (x, y, theta, kappa, ds) 是否都是有限数。
  ///
  /// 求解器在极端工况下（例如约束两两冲突、线搜索失败、回退步被截断）会
  /// 把状态值写成 `NaN` 或 `Inf`，后续消费方（位姿插值、碰撞检查）一旦
  /// 碰到这种值会直接连锁崩。所以这一关必须先于其它几何/物理检查执行：
  /// 一旦数值本身不可信，再讨论它是否合法已经没有意义。
  ///
  /// 失败时上报 `NonFiniteState`，并把首个出问题的状态索引写进
  /// `failure->failed_index`，便于调用层定位是哪个点把整个解拖垮的。
  bool validateFiniteStates(
    const std::vector<double> & variables,
    size_t state_count,
    SmoothingFailureInfo * failure) const
  {
    for (size_t index = 0; index < state_count; ++index) {
      const double * state = KinematicStateLayout::data(variables, index);
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

  /// 校验起终点是否与输入参考路径在调用方声明的容差内对齐。
  ///
  /// 这一关分四块，每块独立判断：
  ///   * 起点位置：`|p_start - ref_front|` 不得超过 `position_tol`
  ///     （= `max(0.5 * costmap_resolution, 1e-3)`），否则视为起点漂移。
  ///   * 起点朝向：仅当 `params.keep_start_orientation = true` 时启用，
  ///     把绝对差折到 `(-π, π]` 再与 `orientation_tol`（0.1 rad ≈ 5.7°）
  ///     比较。
  ///   * 终点位置：把 `p_goal - ref_back` 投影到「目标坐标系」(lon, lat)，
  ///     并分别与 `goal_longitudinal_tolerance` / `goal_lateral_tolerance`
  ///     比较。这样可以同时支持「严格固定」和「目标容差盒」两种使用
  ///     模式：当两个容差都 ~0 时退化为点固定；否则允许在矩形盒内
  ///     自由调整。验收比较还会加 `goal_position_numerical_slack_m` 的纯
  ///     数值余量，吸收软 hinge 在边界附近的正常收敛残差；错误报告仍
  ///     显示调用方声明的容差。目标坐标系的 x 轴是 `goal_position_theta`：
  ///     当 `keep_goal_orientation` 为真时取 `end_theta`（终点姿态）；
  ///     否则用参考路径最后一段的切向，让容差盒方向与参考方向一致。
  ///   * 终点朝向：仅当 `params.keep_goal_orientation = true` 时启用，
  ///     容差取用户声明的 `goal_orientation_tolerance` 加一个很小的
  ///     收敛余量；当配置为 0 时仍按严格终点朝向约束处理。
  ///
  /// 起点失败报 `StartPositionConstraint` / `StartOrientationConstraint`；
  /// 终点失败报 `GoalPositionConstraint` / `GoalOrientationConstraint`。
  /// 终点位置这一支会额外把 lon/lat 误差和容差写进 `failure`，方便
  /// 上游做日志/可视化。
  bool validateKinematicBoundaryStates(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    const ValidationTolerances & tol = request.params.validation;

    const double * start_state = KinematicStateLayout::data(request.variables, 0);
    const double start_dx =
      start_state[KinematicStateLayout::X] - request.reference_points.front().x();
    const double start_dy =
      start_state[KinematicStateLayout::Y] - request.reference_points.front().y();
    if (std::hypot(start_dx, start_dy) > tol.start_position_m) {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::StartPositionConstraint,
        "Kinematic smoother violated the fixed start position constraint",
        0);
    }
    if (request.params.keep_start_orientation &&
      std::abs(angleDifference(start_state[KinematicStateLayout::Theta], request.start_theta)) >
      tol.start_orientation_rad)
    {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::StartOrientationConstraint,
        "Kinematic smoother violated the fixed start orientation constraint",
        0);
    }

    const double * goal_state =
      KinematicStateLayout::data(request.variables, request.state_count - 1);
    const double goal_dx =
      goal_state[KinematicStateLayout::X] - request.reference_points.back().x();
    const double goal_dy =
      goal_state[KinematicStateLayout::Y] - request.reference_points.back().y();
    const double goal_position_theta = goalPositionFrameHeading(
      request.reference_points,
      request.end_theta,
      request.params.keep_goal_orientation);
    const double cos_goal = std::cos(goal_position_theta);
    const double sin_goal = std::sin(goal_position_theta);
    const double goal_lon = cos_goal * goal_dx + sin_goal * goal_dy;
    const double goal_lat = -sin_goal * goal_dx + cos_goal * goal_dy;
    const double goal_lon_tol = std::max(request.params.goal_longitudinal_tolerance, tol.goal_position_m);
    const double goal_lat_tol = std::max(request.params.goal_lateral_tolerance, tol.goal_position_m);
    const double goal_position_slack = tol.goal_position_numerical_slack_m;
    if (std::abs(goal_lon) > goal_lon_tol + goal_position_slack ||
        std::abs(goal_lat) > goal_lat_tol + goal_position_slack) {
      const bool uses_goal_box =
        request.params.goal_longitudinal_tolerance > KinematicStateLayout::EnabledEpsilon ||
        request.params.goal_lateral_tolerance > KinematicStateLayout::EnabledEpsilon;
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
    // 终点朝向验收容差：取「优化阶段声明的容差」与「验收表 floor（默认 1°）」的较大者。
    // 软 hinge 约束的终点姿态总有 ~1e-4 rad 量级残差，floor 用来吸收这类正常噪声。
    const double goal_angle_tol =
      std::max(request.params.goal_orientation_tolerance, tol.goal_orientation_rad);
    if (request.params.keep_goal_orientation &&
      std::abs(angleDifference(goal_state[KinematicStateLayout::Theta], request.end_theta)) >
      goal_angle_tol)
    {
      return throwOrStoreSmoothingFailure(
        failure,
        SmoothingFailureReason::GoalOrientationConstraint,
        describeOrientationViolation(
          "Kinematic smoother violated the fixed goal orientation constraint",
          goal_state[KinematicStateLayout::Theta],
          request.end_theta,
          goal_angle_tol),
        static_cast<int>(request.state_count - 1));
    }
    if (request.params.keep_goal_orientation && request.state_count >= 2) {
      const size_t segment_index = request.state_count - 2;
      const bool is_cusp_segment =
        segment_index < request.is_cusp_segment.size() && request.is_cusp_segment[segment_index];
      if (!is_cusp_segment) {
        // 只依据优化后解的末段运动方向来判断终点朝向是否自相矛盾。
        // 不能拿输入参考路径的末段几何方向做判据:keep_goal_orientation 的语义
        // 恰恰允许请求朝向与参考末段方向明显不同(让平滑器把末段弯到请求朝向),
        // 否则夹角≥90°(或垂直)的合法请求会被误判为冲突而拒绝。
        const double * previous_state = KinematicStateLayout::data(request.variables, segment_index);
        const double terminal_dx =
          goal_state[KinematicStateLayout::X] - previous_state[KinematicStateLayout::X];
        const double terminal_dy =
          goal_state[KinematicStateLayout::Y] - previous_state[KinematicStateLayout::Y];
        if (std::hypot(terminal_dx, terminal_dy) > tol.min_segment_displacement_m) {
          const double goal_heading_x = std::cos(goal_state[KinematicStateLayout::Theta]);
          const double goal_heading_y = std::sin(goal_state[KinematicStateLayout::Theta]);
          const double terminal_projection =
            terminal_dx * goal_heading_x + terminal_dy * goal_heading_y;
          const double terminal_gear =
            segment_index < request.gears.size() ? request.gears[segment_index] : 1.0;
          if (
            (terminal_gear >= 0.0 && terminal_projection <= 0.0) ||
            (terminal_gear < 0.0 && terminal_projection >= 0.0))
          {
            return throwOrStoreSmoothingFailure(
              failure,
              SmoothingFailureReason::GoalOrientationConstraint,
              "Kinematic smoother fixed goal orientation conflicts with the terminal motion direction",
              static_cast<int>(request.state_count - 1));
          }
        }
      }
    }

    return true;
  }

  /// 校验相邻状态之间的几何与运动学语义是否自洽。
  ///
  /// 遍历每一对相邻状态 (i, i+1)，按该段是否被标记为 cusp 分两种判据：
  ///
  ///   * **cusp 段**（`is_cusp_segment[i] = true`）：
  ///     这一段是「原地换向」段，理论上位置和朝向都不应变化。
  ///     因此要求 `|p_{i+1} - p_i|` 不超过 `position_tol`，且
  ///     `|angle(θ_{i+1} - θ_i)|` 不超过 `orientation_tol`。
  ///     越界时上报 `CuspHoldConstraint`。
  ///
  ///   * **非 cusp 段**：
  ///     1) 位移下限：`||p_{i+1} - p_i||` 必须大于 `displacement_tol`
  ///        （= `max(0.25 * costmap_resolution, 1e-4)`），否则视为该段
  ///        被求解器压扁成同一点（`CollapsedSegment`）。
  ///     2) 运动方向：用 `θ_i` 的单位向量与 `p_{i+1} - p_i` 做点积，
  ///        得到「沿当前朝向的有符号投影」。当 `gears[i] >= 0`（前进）
  ///        时投影必须 > 0；`gears[i] < 0`（倒车）时投影必须 < 0。
  ///        这能拦下求解器在倒车段把车头调转方向这种
  ///        「数值上能收敛、语义上不可用」的解
  ///        （`MotionDirectionConstraint`）。
  bool validateKinematicSegmentConsistency(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    const ValidationTolerances & tol = request.params.validation;

    for (size_t index = 0; index + 1 < request.state_count; ++index) {
      const double * current = KinematicStateLayout::data(request.variables, index);
      const double * next = KinematicStateLayout::data(request.variables, index + 1);
      const double dx = next[KinematicStateLayout::X] - current[KinematicStateLayout::X];
      const double dy = next[KinematicStateLayout::Y] - current[KinematicStateLayout::Y];
      const double displacement = std::hypot(dx, dy);

      if (request.is_cusp_segment[index]) {
        if (
          displacement > tol.cusp_position_m ||
          std::abs(angleDifference(
            next[KinematicStateLayout::Theta],
            current[KinematicStateLayout::Theta])) > tol.cusp_orientation_rad)
        {
          return throwOrStoreSmoothingFailure(
            failure,
            SmoothingFailureReason::CuspHoldConstraint,
            "Kinematic smoother violated the cusp hold constraint during post-validation",
            static_cast<int>(index));
        }
        continue;
      }

      if (displacement <= tol.min_segment_displacement_m) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::CollapsedSegment,
          "Kinematic smoother collapsed a non-cusp segment during post-validation",
          static_cast<int>(index));
      }

      const Eigen::Vector2d heading(
        std::cos(current[KinematicStateLayout::Theta]),
        std::sin(current[KinematicStateLayout::Theta]));
      const double signed_projection = dx * heading.x() + dy * heading.y();
      const double gear = request.gears[index];
      if ((gear >= 0.0 && signed_projection <= 0.0) || (gear < 0.0 && signed_projection >= 0.0)) {
        if (request.params.keep_goal_orientation && index + 2 == request.state_count) {
          return throwOrStoreSmoothingFailure(
            failure,
            SmoothingFailureReason::GoalOrientationConstraint,
            "Kinematic smoother fixed goal orientation conflicts with the terminal motion direction",
            static_cast<int>(request.state_count - 1));
        }
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::MotionDirectionConstraint,
          "Kinematic smoother returned a path whose motion direction violates the input gear and endpoint constraints",
          static_cast<int>(index));
      }
    }

    return true;
  }

  /// 校验曲率上限 `params.max_curvature` 是否被违反。
  ///
  /// 仅看「显式 kappa」不够——求解器可能在权重配置偏小或残差互相
  /// 拉扯时输出 kappa 合法但实际姿态轨迹弯曲过大的结果。所以这一关
  /// 用两把尺子同时量：
  ///
  ///   1. **状态曲率**：对每个 `state`，取 `|state[KinematicStateLayout::Kappa]|`（即 kappa）
  ///      直接与 `max_curvature` 比较。`1e-4` 的容差用来吸收求解器
  ///      末次迭代的舍入噪声，避免「刚刚越线」就被打回。
  ///
  ///   2. **几何曲率**：遍历每一对非 cusp 相邻状态，用
  ///      `|normalize(θ_{i+1} - θ_i)| / ||p_{i+1} - p_i||` 估算离散
  ///      曲率。这一项会捕获「kappa 看似合规，但相邻两点的连线
  ///      拐弯过急」的情形（例如 kappa 在两状态间被人为拉平）。如果
  ///      位移低于 `displacement_tol`（基本是同一点），几何曲率本身
  ///      没有定义，直接跳过。
  ///
  /// 失败时上报 `CurvatureConstraint`，并把 `actual_curvature`、
  /// `max_curvature` 与对应转弯半径写进 `failure`，便于排障时一眼
  /// 看到偏差数量级。
  bool validateKinematicCurvatureConstraint(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    const double max_curvature =
      std::max(request.params.max_curvature, KinematicStateLayout::GeometryEpsilon);
    const double curvature_tolerance = request.params.validation.curvature_tolerance;

    auto report_curvature_violation =
      [&](size_t index, double actual_curvature) {
        const double turning_radius =
          actual_curvature > KinematicStateLayout::EnabledEpsilon ?
          1.0 / actual_curvature :
          std::numeric_limits<double>::infinity();
        const std::string message = describeCurvatureViolation(
          actual_curvature,
          max_curvature,
          turning_radius);
        if (failure != nullptr) {
          failure->reason = SmoothingFailureReason::CurvatureConstraint;
          failure->message = message;
          failure->failed_index = static_cast<int>(index);
          failure->actual_curvature = actual_curvature;
          failure->max_curvature = max_curvature;
          failure->turning_radius = turning_radius;
          return false;
        }
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::CurvatureConstraint,
          message,
          static_cast<int>(index));
      };

    // 1) 检查显式状态曲率 kappa 是否越界。
    for (size_t index = 0; index < request.state_count; ++index) {
      const double * state = KinematicStateLayout::data(request.variables, index);
      const double abs_kappa = std::abs(state[KinematicStateLayout::Kappa]);
      if (abs_kappa > max_curvature + curvature_tolerance) {
        return report_curvature_violation(index, abs_kappa);
      }
    }

    // 2) 再检查由相邻姿态形成的几何曲率，覆盖“kappa 合法但输出轨迹几何超限”的情形。
    const double displacement_tol = request.params.validation.min_segment_displacement_m;
    for (size_t index = 0; index + 1 < request.state_count; ++index) {
      if (request.is_cusp_segment[index]) {
        continue;
      }
      const double * current = KinematicStateLayout::data(request.variables, index);
      const double * next = KinematicStateLayout::data(request.variables, index + 1);
      const double dx = next[KinematicStateLayout::X] - current[KinematicStateLayout::X];
      const double dy = next[KinematicStateLayout::Y] - current[KinematicStateLayout::Y];
      const double displacement = std::hypot(dx, dy);
      if (displacement <= displacement_tol) {
        continue;
      }
      const double delta_theta = angleDifference(
        next[KinematicStateLayout::Theta],
        current[KinematicStateLayout::Theta]);
      const double geometric_curvature = std::abs(delta_theta) / displacement;
      if (geometric_curvature > max_curvature + curvature_tolerance) {
        return report_curvature_violation(index, geometric_curvature);
      }
    }

    return true;
  }

  /// 校验路径上每个状态的足迹采样点是否发生硬碰撞或离开地图。
  ///
  /// 这是一道「只在工程上需要时才执行」的检查：调用层通过
  /// `params.obstacleTermsEnabled()`（`obstacle_weight > KinematicStateLayout::EnabledEpsilon`）
  /// 显式声明要不要障碍物项；没声明就整段直接放行（返回 `true`），
  /// 不会污染纯几何调参场景。
  ///
  /// 走到这里意味着既需要足迹检查、也提供了 costmap。剩下的判据是
  /// 「必须配置了至少一种足迹模型」：当 `cost_check_radius` 接近 0 且
  /// `cost_check_points` 为空时，没有可检查的几何形状，同样直接放行，
  /// 以兼容「我只想要运动学一致、不在乎车体是否撞墙」的退化用法。
  ///
  /// 否则对每个状态执行：
  ///   * 单圆模型（`cost_check_points` 为空）：只检查 `state` 本身
  ///     一点。
  ///   * 多点模型（`cost_check_points` 非空）：按 `(x_local, y_local, w)`
  ///     三元组遍历，把局部坐标用 `state.theta` 旋到世界坐标系后
  ///     逐点采样 ESDF。`w` 在构造代价时参与权重，但后验阶段只判断
  ///     点位置是否满足净空，不重复使用 `w`。
  ///
  /// 每个采样点用 `clearanceAtWorldPoint` 查 ESDF。这里的后验检查只
  /// 执行硬安全验收：`obstacle_safe_distance` 是优化阶段的软净空 margin，
  /// 低于它会产生代价，但不会单独导致后验失败。
  ///
  /// 具体判据：
  ///   * 查不到（地图外 / 索引越界）→ 返回 `-Inf`，本步判为越界
  ///     失败（`PathOutOfBounds`），避免对「地图未覆盖区域」误判成
  ///     「安全」。
  ///   * `clearance < radius` → 命中障碍物（`FootprintCollision`）。
  ///
  /// 失败时 `failed_index` 写的是出问题的状态序号（不是采样点序号），
  /// 上层要更细定位可以结合轨迹回放判断是哪一段。
  bool validateKinematicObstacleClearance(
    const KinematicRequest & request,
    SmoothingFailureInfo * failure) const
  {
    if (!request.params.obstacleTermsEnabled() || request.costmap == nullptr) {
      return true;
    }

    const double radius = std::max(request.params.cost_check_radius, 0.0);
    if (
      radius <= KinematicStateLayout::EnabledEpsilon &&
      request.params.cost_check_points.empty())
    {
      return true;
    }

    for (size_t state_index = 0; state_index < request.state_count; ++state_index) {
      const double * state = KinematicStateLayout::data(request.variables, state_index);
      const double x = state[KinematicStateLayout::X];
      const double y = state[KinematicStateLayout::Y];
      const double theta = state[KinematicStateLayout::Theta];
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

}  // namespace kinematic_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_VALIDATOR_HPP_
