#ifndef KINEMATIC_PATH_SMOOTHER__OPTIONS_HPP_
#define KINEMATIC_PATH_SMOOTHER__OPTIONS_HPP_

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace kinematic_path_smoother
{

/// 单次路径平滑请求的算法参数。
///
/// 除 fix_weight 外，公开的 *_weight 表示目标函数中对应代价项的权重。
/// Ceres 会最小化 1/2 * sum(residual^2)，因此实现层会先对这些权重取 sqrt，
/// 再乘到残差上。fix_weight 是硬约束残差尺度，保持直接使用。
/// 输入路径第三维表示方向符号：>=0 为前进，<0 为倒车。
struct SmootherParams
{
  /// 当前参数是否启用了任何障碍物残差。
  bool obstacleTermsEnabled() const
  {
    return std::max(obstacle_weight, cusp_obstacle_weight) > 1e-9;
  }

  /// 运动学状态转移一致性权重：约束 x/y/yaw 与曲率积分模型一致。
  double model_weight{1.0};
  /// 参考路径吸附权重：防止优化点离原始路径过远。
  double reference_weight{0.0};
  /// 普通路径点的 ESDF 障碍物权重；0 表示关闭障碍物项。
  double obstacle_weight{0.0};
  /// cusp 邻域点的 ESDF 障碍物权重；默认不小于普通障碍物权重。
  double cusp_obstacle_weight{0.0};
  /// 显式曲率 kappa 正则权重。
  double curvature_weight{0.0};
  /// 曲率变化率正则权重，抑制曲率突变。
  double curvature_rate_weight{0.0};
  /// 相邻优化点弧长 ds 接近目标间距的权重。
  double spacing_weight{1.0};
  /// 总路径长度惩罚权重，越大越偏向短路径。
  double length_weight{0.0};
  /// 起终点和 cusp 保持段的强约束权重。
  double fix_weight{100.0};

  /// 最大曲率，单位 1/m；同时用于 Ceres 参数边界和后验校验。
  double max_curvature{1.0};
  /// 单段 ds 上界，<=0 表示不限制。
  double max_segment_length{0.0};
  /// 每个优化点相对参考点的最大 xy 偏移，<=0 表示不限制。
  double max_reference_deviation{0.0};
  /// 单次 Ceres 求解的墙钟时间上限，单位秒。
  double max_time{10.0};

  /// true 使用 exact ESDF，false 使用近似 ESDF。
  bool use_exact_esdf{true};
  /// 期望路径 footprint 与障碍物保持的最小净空，单位米。
  double obstacle_safe_distance{0.5};
  /// 单圆 footprint 半径；当 footprint_points 为空时用于障碍物检查。
  double footprint_radius{0.0};
  /// 多检查点 footprint，格式为 [local_x, local_y, weight, ...]。
  std::vector<double> footprint_points{};

  /// 优化前路径下采样步长；cusp 附近会强制保留。
  int path_downsampling_factor{1};
  /// 求解后按运动学模型插值的上采样倍数。
  int path_upsampling_factor{1};
  /// false 时忽略输入路径第三维，整条路径按前进处理。
  bool reversing_enabled{true};

  /// 是否固定起点朝向为 start_direction。
  bool keep_start_orientation{true};
  /// 是否固定终点朝向为 goal_direction。
  bool keep_goal_orientation{true};
  /// 终点在目标坐标系纵向上的允许误差，单位米。
  double goal_longitudinal_tolerance{0.0};
  /// 终点在目标坐标系横向上的允许误差，单位米。
  double goal_lateral_tolerance{0.0};
  /// 终点朝向允许误差，单位弧度。
  double goal_orientation_tolerance{0.0};
};

/// Ceres 求解器级参数，与单次路径几何语义无关。
struct OptimizerParams
{
  /// 暴露两种常用线性求解器，避免调用侧直接依赖 Ceres 枚举。
  enum class LinearSolver
  {
    /// 小规模问题可用，配置简单。
    DenseQr,
    /// 默认选择，适合当前相邻状态链带来的稀疏结构。
    SparseNormalCholesky,
  };

  /// 将内部枚举转为稳定字符串，便于配置文件或绑定层使用。
  static const char * toString(LinearSolver solver)
  {
    return solver == LinearSolver::DenseQr ? "DENSE_QR" : "SPARSE_NORMAL_CHOLESKY";
  }

  /// 从配置字符串恢复枚举；未知值直接抛出，避免静默退化。
  static LinearSolver fromString(const std::string & name)
  {
    if (name == "DENSE_QR") {
      return LinearSolver::DenseQr;
    }
    if (name == "SPARSE_NORMAL_CHOLESKY") {
      return LinearSolver::SparseNormalCholesky;
    }
    throw std::invalid_argument("Unsupported linear solver: " + name);
  }

  /// 是否输出 Ceres 详细迭代日志。
  bool debug{false};
  /// Ceres 线性子问题求解器选择。
  LinearSolver linear_solver{LinearSolver::SparseNormalCholesky};
  /// 最大非线性迭代次数。
  int max_iterations{50};
  /// 目标函数变化收敛阈值。
  double function_tolerance{1e-6};
  /// 梯度收敛阈值。
  double gradient_tolerance{1e-10};
  /// 参数更新收敛阈值。
  double parameter_tolerance{1e-8};
};

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__OPTIONS_HPP_
