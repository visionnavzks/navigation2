// Copyright (c) 2026
// Licensed under the Apache License, Version 2.0

#ifndef KINEMATIC_SMOOTHER__FOOTPRINT_HPP_
#define KINEMATIC_SMOOTHER__FOOTPRINT_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace kinematic_smoother
{

/// 足迹形状（高层语义）。`Point` 等价于"单圆 + 原心单点"；`Capsule` 等价于
/// "矩形 + 两端半圆"，并被离散成一组等距分布的圆心 + 一个等于车宽一半的
/// 单圆半径。两种模式都会进一步被压成 `(radius, check_points)`，作为
/// smoother / validator / A* 的统一输入。
enum class FootprintMode
{
  Point = 0,
  Capsule = 1,
};

/// Capsule 模式下，中心点沿车长方向的覆盖范围。
/// - `Conservative`（默认）：中心点最远到 `±L/2`（含两端半圆），更保守；
/// - `Exact`：中心点最远到 `±(L/2 - r)`，与"长 L + 宽 W + 圆角 W/2"的
///   capsule 严格匹配。
enum class CapsuleMode
{
  Conservative = 0,
  Exact = 1,
};

/// `buildFootprintModel` 的高层输入参数。所有字段都按米计。
struct FootprintSpec
{
  FootprintMode mode{FootprintMode::Capsule};
  CapsuleMode capsule_mode{CapsuleMode::Conservative};

  /// 车身长度（米），仅 Capsule 模式使用。
  double length_m{0.0};
  /// 车身宽度（米），仅 Capsule 模式使用；`check_radius = width_m / 2`。
  double width_m{0.0};
  /// Point 模式下的等价圆盘半径（米）。`0` 表示用 `min_resolution_m` 作下界。
  double point_radius_m{0.0};
  /// Capsule 模式下的腰部采样容差（米）。`0` 表示按 `min_resolution_m` 推导。
  /// 实际最大凹陷被夹到 `[min_resolution_m, radius/2]`。
  double sampling_tolerance_m{0.0};

  /// 防止尺寸 / 间距退化的最小分辨率下界（米）。通常等于调用方的
  /// costmap resolution；不可为 0 或负数。
  double min_resolution_m{0.05};
};

/// 离散化后的足迹模型，是 smoother / validator / A* 三者共享的"几何原料"。
struct FootprintModel
{
  /// 等价单圆半径（米）。`Point` 模式 = `max(point_radius, 0.5 * min_res)`；
  /// `Capsule` 模式 = `width_m / 2`。
  double check_radius{0.0};
  /// 平铺的检测点列表，每 3 个数一组：`(x_local, y_local, weight)`。
  /// `x_local` 沿车体前向、`y_local` 沿车体左向，单位米；
  /// `weight` 是该点的额外残差权重，`Point` / `Capsule` 模式下都填 1.0。
  /// 空数组意味着"无检测点"（不应出现——`buildFootprintModel` 总会至少输出 1 个）。
  std::vector<double> check_points{};
};

namespace footprint_detail
{

/// 把 `capsule_mode` 规范化为内部枚举；用于给 pybind 提供字符串/枚举互转。
inline CapsuleMode normalizeCapsuleMode(CapsuleMode mode) { return mode; }

/// Capsule 中心点沿车长方向的最大偏移（绝对值，米）。
inline double resolveCapsuleCenterLimit(
  double half_length, double radius, CapsuleMode capsule_mode)
{
  if (capsule_mode == CapsuleMode::Exact) {
    return std::max(half_length - radius, 0.0);
  }
  return half_length;
}

/// 计算 capsule 圆心序列——参考 Python 版 `_build_capsule_center_offsets`。
///
/// 几何含义：每个圆心配半径 `radius` 覆盖一段圆盘；两两圆心之间腰部
/// 的最大凹陷不超过 `tolerance`。由勾股定理反解出圆心间最大间距
/// `max_spacing = 2 * sqrt(radius² − (radius − d)²)`，再 `np.linspace`
/// 等距分布 `2*limit / max_spacing + 1` 个点。
inline std::vector<double> buildCapsuleCenterOffsets(
  double limit_x, double radius, double tolerance, double min_resolution)
{
  std::vector<double> offsets;
  if (limit_x <= 1e-6) {
    offsets.push_back(0.0);
    return offsets;
  }

  const double max_gap_depth = std::min(
    std::max(tolerance, 1e-3),
    std::max(radius * 0.5, 1e-3));
  const double min_val = radius * radius -
    std::pow(std::max(radius - max_gap_depth, 0.0), 2);
  double max_spacing = 2.0 * std::sqrt(std::max(min_val, 1e-9));
  max_spacing = std::max(max_spacing, min_resolution * 0.5);

  const int interval_count = std::max(
    1, static_cast<int>(std::ceil((2.0 * limit_x) / max_spacing)));
  const int total = interval_count + 1;
  offsets.reserve(static_cast<std::size_t>(total));

  const double step = (2.0 * limit_x) / static_cast<double>(interval_count);
  for (int index = 0; index < total; ++index) {
    const double t = static_cast<double>(index);
    offsets.push_back(-limit_x + t * step);
  }
  return offsets;
}

}  // namespace footprint_detail

/// 把高层几何规格（`FootprintSpec`）展开为离散的 `(radius, check_points)`。
/// 纯几何预计算，**不读 costmap / ESDF**——是 smoother / validator / A*
/// 三处碰撞检查统一的几何原料。
inline FootprintModel buildFootprintModel(const FootprintSpec & spec)
{
  const double min_res = std::max(spec.min_resolution_m, 1e-6);
  const double half_length = std::max(spec.length_m * 0.5, min_res * 0.5);
  const double half_width = std::max(spec.width_m * 0.5, min_res * 0.5);
  const double sampling_tolerance = std::max(spec.sampling_tolerance_m, 0.0);

  FootprintModel model;

  if (spec.mode == FootprintMode::Point) {
    model.check_radius = std::max(spec.point_radius_m, min_res * 0.5);
    model.check_points = {0.0, 0.0, 1.0};
    return model;
  }

  // Capsule 模式
  model.check_radius = half_width;
  const double center_limit = footprint_detail::resolveCapsuleCenterLimit(
    half_length, model.check_radius, spec.capsule_mode);

  const std::vector<double> offsets = footprint_detail::buildCapsuleCenterOffsets(
    center_limit, model.check_radius, sampling_tolerance, min_res);

  model.check_points.reserve(offsets.size() * 3);
  for (double offset_x : offsets) {
    model.check_points.push_back(offset_x);
    model.check_points.push_back(0.0);
    model.check_points.push_back(1.0);
  }
  return model;
}

}  // namespace kinematic_smoother

#endif  // KINEMATIC_SMOOTHER__FOOTPRINT_HPP_
