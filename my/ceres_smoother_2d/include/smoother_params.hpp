#pragma once

#include <limits>

namespace ceres_smoother_2d
{

// ========================================================================
// 平滑器参数
// ========================================================================
struct SmootherParams
{
  int max_iterations{100};
  double max_time_seconds{0.5};
  bool verbose{false};

  // 平滑度：惩罚二阶差分。
  double w_smooth{10.0};

  // 最大曲率：对最大曲率施加软约束。
  double w_max_curvature{1000.0};
  double min_turning_radius{0.2};  // 米

  // 参考路径跟踪：惩罚偏离 A* 参考路径。
  // 当障碍/长度权重较强时，防止优化器把路径拉得离原始路线太远。
  // 默认关闭；需要更贴近 A* 路径时增大该权重。
  double w_reference{0.0};

  // 弹性带长度：最小化 Σ‖p_next - p_curr‖²（点间距离平方和）的权重。
  // 与平滑、障碍和参考项结合后，它起到均匀间距作用力的效果，同时避免
  // target_spacing 弹簧的非线性和静止长度冲突。
  // 默认值较小，避免长度项过度收缩路径并压过障碍避让。
  double w_length{1.0};
  // 障碍物（ESDF）：拆成两个独立项，使优化器能权衡“待在安全区外”
  // （soft hinge）与“绝不能在墙内”（点越深入惩罚越大的项）。第一项本身
  // 是安全边界附近的对称 hinge，在障碍侧存在平坦平台：如果某点最终
  // dist < 0，梯度幅值恒定，平滑器可能停在墙内。w_penetration 通过加入
  // 随 -dist 二次增长的项修复这个问题，点越深入惩罚越大。
  double w_obstacle{1.0};
  // 障碍内部惩罚权重。仅 hinge 项（w_obstacle）在障碍内部有梯度恒定且
  // 较小的平台；卡在深处的点可能无法逃出。穿透项加入随深度（-dist）
  // 增长的代价，把优化器拉出障碍。默认非零，使这个防护始终生效。
  double w_penetration{1000.0};
  double safety_margin{1.0};       // 米，期望最小间隙（从机器人边缘算起）
  double robot_radius{0.5};        // 米，机器人内切圆半径；有效间隙阈值 =
                                  // safety_margin + robot_radius

  // 平滑前/后的可选弧长重采样。两个阶段共用一个间距，减少对外调参面。
  bool resample_before_smooth{true};
  bool resample_after_smooth{false};
  double resample_spacing{0.3};  // 米

  double maxCurvature() const
  {
    return min_turning_radius > 0 ?
           1.0 / min_turning_radius : std::numeric_limits<double>::infinity();
  }

  double obstacleCostDistance() const
  {
    return safety_margin + robot_radius;
  }
};

}  // namespace ceres_smoother_2d
