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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace kinematic_smoother
{

/**
 * @struct kinematic_smoother::SmootherParams
 * @brief 独立平滑器的运行时配置。
 *
 * 当前公开后端是 `KinematicSmoother`。
 * 大多数权重由调用方传入平方后的值（即实际权重），代码内部自动开方后再
 * 乘到残差上，随后由 Ceres 在目标函数中完成平方。
 *
 * 这组参数可以按四类来理解：
 * 1. 运动学和参考路径权重。
 * 2. 障碍物与足迹检查配置。
 * 3. 路径重采样与倒车语义。
 * 4. 起终点位置 / 朝向约束。
 */
struct SmootherParams
{
  /// 当前请求是否真的启用了任何障碍物残差。
  ///
  /// 这让调用层可以在 costmap 为 null 时区分“完全不依赖障碍物”与
  /// “必须提供地图才能继续”的两种情况。
  bool obstacleTermsEnabled() const
  {
    return obstacle_weight > 1e-9;
  }

  // --- Kinematic and reference-path weights ---

  /// 运动学状态转移一致性残差的权重（传入的是平方后的值，内部自动开方）。
  double model_weight{20.0};
  /// 障碍物净空残差的基础权重（传入的是平方后的值，内部自动开方）。
  double obstacle_weight{1.0};
  /// 约束优化后控制点贴近参考路径的权重（传入的是平方后的值，内部自动开方）。
  /// 设为 0 时，路径只受运动学与障碍物项驱动。
  double reference_path_weight{0.0};
  /// 每个优化点相对对应参考点的最大 x/y 偏移半径，单位米；<= 0 表示关闭。
  double reference_point_max_deviation_m{0.0};
  /// 运动学版显式曲率状态 kappa 的正则权重（传入的是平方后的值，内部自动开方）。
  double kinematic_curvature_weight{1.0};
  /// 运动学版显式曲率变化率项的权重（传入的是平方后的值，内部自动开方）。
  double kinematic_curvature_rate_weight{1.0};
  /// 运动学版显式弧长步长 ds 贴近目标间距的正则权重（传入的是平方后的值，内部自动开方）。
  /// 默认保留为 1.0，用于避免步长变量在无约束时完全漂移。
  double kinematic_spacing_weight{0.0};
  /// 运动学版显式弧长步长 ds 的上界，单位米；<= 0 表示不启用上界。
  double kinematic_max_spacing{0.0};
  /// 总长度惩罚的权重（传入的是平方后的值，内部自动开方）；值越大，越倾向于压缩整条路径的总弧长。
  double path_length_weight{1.0};
  /// cusp 保持段和起终点边界残差共用的直接约束权重。
  /// 与 `*_sqrt` 参数不同，这个值不会再开方，直接乘到残差上。
  double fix_weight{100.0};
  /// 允许的最大曲率，单位为 1 / m。
  double max_curvature{10.0};
  /// 传给 Ceres 求解器的最大墙钟时间，单位秒。
  double max_time{10.0};

  // --- Obstacle and footprint handling ---

  /// 为 true 时使用精确有符号距离场后端。
  /// 为 false 时允许调用层回退到近似距离场实现。
  bool use_exact_esdf{true};
  /// 对障碍物距离场期望满足的最小有符号净空，单位米。
  double obstacle_safe_distance{0.5};
  /// 当 cost_check_points 为空时使用的圆形足迹采样半径，单位米。
  double cost_check_radius{0.0};
  /// 用于障碍物足迹检查的局部坐标三元组 (x, y, weight)。
  ///
  /// 每 3 个数表示一个局部检查点。若该数组为空，则退回到以
  /// `cost_check_radius` 为半径的单圆检查模型。
  std::vector<double> cost_check_points{};

  // --- Path resampling and direction semantics ---

  /// 在连接残差块之前按目标间距重采样路径，单位米；<= 0 时使用旧的倍率下采样。
  /// 该值同时作为运动学 spacing residual 的目标步长。
  double path_target_spacing{0.0};
  /// 在连接残差块之前应用的路径下采样步长。
  /// 值越大，参与求解的状态数越少；仅在 path_target_spacing <= 0 时生效。
  int path_downsampling_factor{1};
  /// 重建最终路径时使用的插值倍数。
  /// 值越大，输出路径越密；仅在 path_output_spacing <= 0 时生效。
  int path_upsampling_factor{1};
  /// 重建最终路径时使用的目标间距，单位米；<= 0 时使用旧的插值倍数。
  double path_output_spacing{0.0};
  /// 为 false 时忽略输入路径第三分量的倒车语义，整条路径按前进段处理。
  /// 这只影响 gear 推断，不会改写调用方传入的原始路径符号。
  bool reversing_enabled{true};

  // --- Goal and boundary handling ---

  /// 终点在目标坐标系前向轴上的允许位置容差，单位米；0 表示严格固定。
  double goal_longitudinal_tolerance{0.0};
  /// 终点在目标坐标系横向轴上的允许位置容差，单位米；0 表示严格固定。
  double goal_lateral_tolerance{0.0};
  /// 终点朝向允许容差，单位弧度；仅在 keep_goal_orientation=true 时生效。
  double goal_orientation_tolerance{0.0};
  /// 通过锚定终点前一个点来固定终点切向方向。
  /// 为 false 时，终点朝向可在目标位置容差内自由调整。
  bool keep_goal_orientation{true};
  /// 通过锚定第二个点来固定起点切向方向。
  bool keep_start_orientation{true};
};

/**
 * @struct kinematic_smoother::OptimizerParams
 * @brief 传递给 Ceres 的求解器级配置。
 *
 * 这个结构只保存核心求解配置。面向 Python / Web 的字符串形式线性求解器
 * 选择会在边界层转换成 `LinearSolver` 枚举，再由核心实现映射成
 * `ceres::LinearSolverType`。
 */
struct OptimizerParams
{
  /// 当前公开的 Ceres 线性求解器选择。
  enum class LinearSolver
  {
    /// 适合小型稠密问题的 QR 分解后端。
    DenseQr,
    /// 默认后端，适合当前稀疏结构的运动学优化问题。
    SparseNormalCholesky,
  };

  /// 把内部求解器枚举转成稳定字符串，供 pybind / Web 边界复用。
  static const char * linearSolverToString(LinearSolver solver)
  {
    switch (solver) {
      case LinearSolver::DenseQr:
        return "DENSE_QR";
      case LinearSolver::SparseNormalCholesky:
        return "SPARSE_NORMAL_CHOLESKY";
    }
    return "SPARSE_NORMAL_CHOLESKY";
  }

  /// 从边界层字符串恢复内部枚举；非法值直接抛异常。
  static LinearSolver linearSolverFromString(const std::string & solver_name)
  {
    if (solver_name == "DENSE_QR") {
      return LinearSolver::DenseQr;
    }
    if (solver_name == "SPARSE_NORMAL_CHOLESKY") {
      return LinearSolver::SparseNormalCholesky;
    }
    throw std::invalid_argument("Unsupported linear_solver_type: " + solver_name);
  }

  /// 开启逐迭代详细日志和最终摘要输出。
  bool debug{false};
  /// Ceres 线性求解器选择。
  /// 这项配置主要影响每次非线性迭代内部的线性子问题求解方式。
  LinearSolver linear_solver{LinearSolver::SparseNormalCholesky};
  /// 最大非线性迭代次数。
  int max_iterations{50};

  /// 参数步长收敛阈值。
  /// 当连续迭代的参数更新足够小，Ceres 可以提前停止。
  double parameter_tolerance{1e-8};
  /// 目标函数值收敛阈值。
  /// 当目标函数改善幅度足够小，Ceres 可以提前停止。
  double function_tolerance{1e-6};
  /// 梯度收敛阈值。
  /// 当梯度足够接近零，说明当前解已经接近局部驻点。
  double gradient_tolerance{1e-10};
};

}  // namespace kinematic_smoother

#endif  // CONSTRAINED_SMOOTHER__OPTIONS_HPP_
