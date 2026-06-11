#ifndef KINEMATIC_PATH_SMOOTHER__SMOOTHER_REQUEST_HPP_
#define KINEMATIC_PATH_SMOOTHER__SMOOTHER_REQUEST_HPP_

#include <cstddef>
#include <vector>

#include "Eigen/Core"

#include "kinematic_path_smoother/costmap2d.hpp"
#include "kinematic_path_smoother/exceptions.hpp"
#include "kinematic_path_smoother/options.hpp"

namespace kinematic_path_smoother
{

/// 一次 smooth() 调用的显式结果。
///
/// optimized_path 是 Ceres knot 直接解包结果；path 是通过后验校验后、
/// 按 path_upsampling_factor 重建的最终输出路径。
struct SmoothingResult
{
  /// 优化 knot 路径，格式为 (x, y, yaw)。
  std::vector<Eigen::Vector3d> optimized_path;
  /// 最终输出路径，格式为 (x, y, yaw)。
  std::vector<Eigen::Vector3d> path;
  /// 参与优化的 knot 数，包含为 cusp 插入的保持状态。
  std::size_t optimized_knot_count{0};
  /// 本次优化估计出的目标 knot 间距，单位米。
  double target_spacing{0.0};
  /// true 表示求解和后验校验都通过。
  bool success{false};
};

/// 单次 smooth() 的只读请求视图。
///
/// 该结构不拥有 path、方向向量、costmap、params 或 ESDF，调用方需保证这些对象
/// 的生命周期覆盖 smooth() 调用。输入 path 的 z 分量是方向符号，不是 yaw。
struct SmoothingRequest
{
  /// 输入参考路径，格式为 (x, y, direction_sign)。
  const std::vector<Eigen::Vector3d> & path;
  /// 起点切向方向向量，用于生成起点 yaw 约束。
  const Eigen::Vector2d & start_direction;
  /// 终点切向方向向量，用于生成终点 yaw 约束。
  const Eigen::Vector2d & goal_direction;
  /// 可选 costmap；启用障碍物项时必须非空。
  const Costmap2D * costmap{nullptr};
  /// 单次平滑的算法参数。
  const SmootherParams & params;
  /// 可选预计算 ESDF，大小必须等于 costmap 宽高乘积。
  const std::vector<double> * precomputed_esdf{nullptr};
  /// 可选失败回传槽；为空时失败通过异常抛出。
  FailureInfo * failure{nullptr};
};

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__SMOOTHER_REQUEST_HPP_
