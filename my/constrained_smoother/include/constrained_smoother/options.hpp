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

#ifndef CONSTRAINED_SMOOTHER__OPTIONS_HPP_
#define CONSTRAINED_SMOOTHER__OPTIONS_HPP_

#include <map>
#include <string>
#include <vector>
#include "ceres/ceres.h"
#include "constrained_smoother/astar_esdf.hpp"

namespace constrained_smoother
{

/**
 * @struct constrained_smoother::SmootherParams
 * @brief 几何约束平滑器的运行时配置。
 *
 * 独立版平滑器会在二维路径点上最小化一个非线性最小二乘目标。
 * 大多数权重采用平方根形式，是因为它们会先直接乘到残差上，随后再由
 * Ceres 在目标函数中完成平方。
 */
struct SmootherParams
{
  SmootherParams() {}

  bool obstacleTermsEnabled() const
  {
    return std::max(costmap_weight_sqrt, cusp_costmap_weight_sqrt) > 1e-9;
  }

  /// 三点平滑残差的平方根权重。
  double smooth_weight_sqrt{0.0};
  /// 运动学状态转移一致性残差的平方根权重。
  double model_weight_sqrt{0.0};
  /// 障碍物净空残差的基础平方根权重。
  double costmap_weight_sqrt{0.0};
  /// cusp 邻域内使用的增强障碍物权重。
  double cusp_costmap_weight_sqrt{0.0};
  /// cusp 周围用于过渡障碍物权重的弧长范围。
  double cusp_zone_length{0.0};
  /// 约束优化后控制点贴近参考路径的平方根权重。
  double distance_weight_sqrt{0.0};
  /// 每个优化点相对对应参考点的最大 x/y 偏移半径，单位米；<= 0 表示关闭。
  double reference_point_max_deviation{0.0};
  /// 几何版中“超出最大曲率阈值”的平方根惩罚权重。
  double curvature_weight_sqrt{0.0};
  /// 几何版可选四点曲率变化率代理项的平方根权重。
  double curvature_rate_weight_sqrt{0.0};
  /// 运动学版显式曲率状态 kappa 的平方根正则权重。
  double kinematic_curvature_weight_sqrt{0.0};
  /// 运动学版显式曲率变化率项的平方根权重。
  double kinematic_curvature_rate_weight_sqrt{0.0};
  /// 允许的最大曲率，单位为 1 / m。
  double max_curvature{0.0};
  /// 传给 Ceres 求解器的最大墙钟时间。
  double max_time{10.0};
  /// 为 true 时使用精确有符号距离场后端。
  bool use_exact_esdf{true};
  /// 对障碍物距离场期望满足的最小有符号净空。
  double obstacle_safe_distance{0.5};
  /// 当 cost_check_points 为空时使用的圆形足迹采样半径。
  double cost_check_radius{0.0};
  /// 在连接残差块之前应用的路径下采样步长。
  int path_downsampling_factor{1};
  /// 重建最终路径时使用的插值倍数。
  int path_upsampling_factor{1};
  /// 为保持 API 兼容而保留；当前独立版求解器并未实际使用。
  bool reversing_enabled{true};
  /// 终点在目标坐标系前向轴上的允许位置容差，单位米；0 表示严格固定。
  double goal_longitudinal_tolerance{0.0};
  /// 终点在目标坐标系横向轴上的允许位置容差，单位米；0 表示严格固定。
  double goal_lateral_tolerance{0.0};
  /// 终点朝向允许容差，单位弧度；仅在 keep_goal_orientation=true 时生效。
  double goal_orientation_tolerance{0.0};
  /// 通过锚定终点前一个点来固定终点切向方向。
  bool keep_goal_orientation{true};
  /// 通过锚定第二个点来固定起点切向方向。
  bool keep_start_orientation{true};
  /// 用于障碍物足迹检查的局部坐标三元组 (x, y, weight)。
  std::vector<double> cost_check_points{};
};

/**
 * @struct constrained_smoother::OptimizerParams
 * @brief 传递给 Ceres 的求解器级配置。
 */
struct OptimizerParams
{
  OptimizerParams()
  : debug(false),
    linear_solver_type("SPARSE_NORMAL_CHOLESKY"),
    max_iterations(50),
    param_tol(1e-8),
    fn_tol(1e-6),
    gradient_tol(1e-10)
  {
  }

  const std::map<std::string, ceres::LinearSolverType> solver_types = {
    {"DENSE_QR", ceres::DENSE_QR},
    {"SPARSE_NORMAL_CHOLESKY", ceres::SPARSE_NORMAL_CHOLESKY}};

  /// 开启逐迭代详细日志和最终摘要输出。
  bool debug;
  /// solver_types 中的键，用于选择 Ceres 线性求解器后端。
  std::string linear_solver_type;
  int max_iterations;     // Ceres default: 50

  double param_tol;       // Ceres default: 1e-8
  double fn_tol;          // Ceres default: 1e-6
  double gradient_tol;    // Ceres default: 1e-10
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__OPTIONS_HPP_
