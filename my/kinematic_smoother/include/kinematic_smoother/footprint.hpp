// Copyright (c) 2026
// Licensed under the Apache License, Version 2.0

#ifndef KINEMATIC_SMOOTHER__FOOTPRINT_HPP_
#define KINEMATIC_SMOOTHER__FOOTPRINT_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
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

/// Capsule 模式下的几何语义。**两种模式都不是严格的安全模型，只是常用近似**。
///
/// - `Conservative`（默认）：用半径 `W/2` 的圆沿车体中心线扫描，**用来保守
///   包络一个 L×W 矩形车体**。中心点最远到 `±L/2`，几何外包络总长 = `L + W`。
///   适合：真实车体是矩形、或对安全裕度有要求的场景。
///
/// - `Exact`：中心点最远到 `±(L/2 - r)`，代表**真实几何就是总长 L、总宽 W
///   的 capsule**。**不适合用来近似矩形车体**——会漏掉矩形的四个角。
///   适合：真实车体本身就是 capsule 形状（如某些 AGV 的圆角车壳）。
enum class CapsuleMode
{
  Conservative = 0,
  Exact = 1,
};

/// `buildFootprintModel` 的高层输入参数。所有长度字段都按米计。
struct FootprintSpec
{
  FootprintMode mode{FootprintMode::Capsule};
  CapsuleMode capsule_mode{CapsuleMode::Conservative};

  /// 车身长度（米），仅 Capsule 模式使用；必须 > 0。
  double length_m{0.0};
  /// 车身宽度（米），仅 Capsule 模式使用；`check_radius = width_m / 2`，必须 > 0。
  double width_m{0.0};
  /// Point 模式下的等价圆盘半径（米）。`0` 表示用 `0.5 * min_resolution_m`
  /// 作下界（半个栅格作为最小点机器人半径）。
  double point_radius_m{0.0};
  /// Capsule 模式下两圆之间腰部允许的最大凹陷（米）。
  /// `0` 表示用 `min_resolution_m` 兜底，避免 1mm 超密采样。实际值
  /// 会被夹到 `[min_resolution_m, max(radius / 2, min_resolution_m)]` 区间
  /// （当 `min_resolution_m > radius / 2` 时上界退化为 `min_resolution_m`）。
  double sampling_tolerance_m{0.0};

  /// 防止尺寸 / 间距退化的最小分辨率下界（米）。通常等于调用方的
  /// costmap resolution；必须 > 0。
  double min_resolution_m{0.05};
};

/// 离散化后的足迹模型，是 smoother / validator / A* 三者共享的"几何原料"。
///
/// **重要语义提醒**：离散的圆心序列只是连续 capsule 的**采样近似**，不是严格
/// 的安全模型。相邻两个圆之间会有 `sampling_tolerance` 量级的腰部凹陷——
/// smoother 端用的可微 hinge 不感知这个凹陷，validator 用的最近邻硬判也只看
/// 圆心位置。**生产环境对安全裕度有要求时，应在 `cost_check_radius` 上额外
/// 加 `sampling_tolerance` 的余量**，让凹陷区也落在硬判半径之外。
struct FootprintModel
{
  /// 等价单圆半径（米）。`Point` 模式 = `max(point_radius, 0.5 * min_res)`；
  /// `Capsule` 模式 = `width_m / 2`。
  double check_radius{0.0};
  /// 平铺的检测点列表，每 3 个数一组：`(x_local, y_local, weight)`。
  /// `x_local` 沿车体前向、`y_local` 沿车体左向，单位米。
  /// `weight` 已是**归一化权重** = `1 / sqrt(N)`（N 为检测点总数），
  /// 让 Ceres 平方后总代价 ≈ `pose_weight² · mean(hinge_i²)`，与采样密度 N
  /// 解耦——同一个 `obstacle_weight` 永远代表同一物理强度的安全约束。
  /// 空数组意味着"无检测点"（不应出现——`buildFootprintModel` 总会至少输出 1 个）。
  std::vector<double> check_points{};
};

namespace footprint_detail
{

struct CapsuleSampling
{
  int interval_count{1};
  int total{1};
  double step{0.0};
};

/// Capsule 中心点沿车长方向的最大偏移（绝对值，米）。
inline double resolveCapsuleCenterLimit(
  double half_length, double radius, CapsuleMode capsule_mode)
{
  if (capsule_mode == CapsuleMode::Exact) {
    return std::max(half_length - radius, 0.0);
  }
  return half_length;
}

inline CapsuleSampling resolveCapsuleSampling(
  double limit_x, double radius, double max_gap_depth, double min_resolution)
{
  if (limit_x <= 1e-6) {
    return {};
  }

  const double clamped_depth = std::clamp(
    max_gap_depth, min_resolution, std::max(radius * 0.5, min_resolution));
  const double min_val = radius * radius -
    std::pow(std::max(radius - clamped_depth, 0.0), 2);
  double max_spacing = 2.0 * std::sqrt(std::max(min_val, 1e-9));
  max_spacing = std::max(max_spacing, min_resolution * 0.5);

  const int interval_count = std::max(
    1, static_cast<int>(std::ceil((2.0 * limit_x) / max_spacing)));
  return {
    interval_count,
    interval_count + 1,
    (2.0 * limit_x) / static_cast<double>(interval_count)};
}

/// 计算 capsule 圆心序列——参考 Python 版 `_build_capsule_center_offsets`。
///
/// 几何含义：每个圆心配半径 `radius` 覆盖一段圆盘；两两圆心之间腰部
/// 的最大凹陷不超过 `max_gap_depth`。由勾股定理反解出圆心间最大间距
/// `max_spacing = 2 * sqrt(radius² − (radius − d)²)`，再等距分布
/// `2 * limit / max_spacing + 1` 个点。
///
/// `max_gap_depth` 会被夹到 `[min_resolution, max(radius/2, min_resolution)]`
/// 区间（当 `min_resolution > radius/2` 时上界退化为 `min_resolution`），
/// 防止凹陷过深导致圆心过密、退化。
inline std::vector<double> buildCapsuleCenterOffsets(
  double limit_x, double radius, double max_gap_depth, double min_resolution)
{
  const CapsuleSampling sampling = resolveCapsuleSampling(
    limit_x, radius, max_gap_depth, min_resolution);
  std::vector<double> offsets(static_cast<std::size_t>(sampling.total));

  for (int index = 0; index < sampling.total; ++index) {
    offsets[static_cast<std::size_t>(index)] =
      sampling.total == 1 ? 0.0 : -limit_x + static_cast<double>(index) * sampling.step;
  }
  return offsets;
}

}  // namespace footprint_detail

/// 把高层几何规格（`FootprintSpec`）展开为离散的 `(radius, check_points)`。
/// 纯几何预计算，**不读 costmap / ESDF**——是 smoother / validator / A*
/// 三处碰撞检查统一的几何原料。
///
/// **入参校验**（抛 `std::invalid_argument` 当）：
///   * `min_resolution_m <= 0`
///   * Capsule 模式下 `length_m <= 0` 或 `width_m <= 0`
///   * `Exact` Capsule 模式下 `length_m < width_m`
///   * Point 模式下 `point_radius_m < 0`
///   * 任何字段为 NaN / Inf
///
/// **weight 归一化**：返回的 `check_points` 中 `weight = 1 / sqrt(N)`，
/// 让 smoother 端 Ceres 平方后的总代价与采样密度 N 解耦。
inline FootprintModel buildFootprintModel(const FootprintSpec & spec)
{
  // ---- 基础校验 ----
  // 先查非有限值（NaN / Inf），否则 `NaN > 0.0 == false` 会让下面的
  // `> 0` 校验抢先抛出误导性的 "must be > 0"。
  if (!std::isfinite(spec.min_resolution_m) ||
    !std::isfinite(spec.length_m) || !std::isfinite(spec.width_m) ||
    !std::isfinite(spec.point_radius_m) || !std::isfinite(spec.sampling_tolerance_m))
  {
    throw std::invalid_argument("FootprintSpec contains non-finite numeric values");
  }
  if (!(spec.min_resolution_m > 0.0)) {
    throw std::invalid_argument("FootprintSpec.min_resolution_m must be > 0");
  }
  if (spec.mode == FootprintMode::Capsule) {
    if (!(spec.length_m > 0.0)) {
      throw std::invalid_argument("FootprintSpec.length_m must be > 0 in Capsule mode");
    }
    if (!(spec.width_m > 0.0)) {
      throw std::invalid_argument("FootprintSpec.width_m must be > 0 in Capsule mode");
    }
    if (spec.capsule_mode == CapsuleMode::Exact && spec.length_m < spec.width_m) {
      throw std::invalid_argument(
        "FootprintSpec.length_m must be >= width_m in CapsuleMode::Exact "
        "(a capsule cannot be shorter than wide)");
    }
  } else {
    if (!(spec.point_radius_m >= 0.0)) {
      throw std::invalid_argument("FootprintSpec.point_radius_m must be >= 0 in Point mode");
    }
  }

  const double min_res = spec.min_resolution_m;

  FootprintModel model;

  if (spec.mode == FootprintMode::Point) {
    model.check_radius = std::max(spec.point_radius_m, min_res * 0.5);
    model.check_points = {0.0, 0.0, 1.0};
    return model;
  }

  // ---- Capsule 模式 ----
  const double half_length = spec.length_m * 0.5;
  const double half_width = spec.width_m * 0.5;
  model.check_radius = half_width;

  const double center_limit = footprint_detail::resolveCapsuleCenterLimit(
    half_length, model.check_radius, spec.capsule_mode);

  // `sampling_tolerance_m == 0` 兜底成 min_res，避免 1mm 超密采样。
  const double requested_tolerance = spec.sampling_tolerance_m > 0.0
    ? spec.sampling_tolerance_m
    : min_res;

  const footprint_detail::CapsuleSampling sampling = footprint_detail::resolveCapsuleSampling(
    center_limit, model.check_radius, requested_tolerance, min_res);

  // 归一化权重 1/sqrt(N)：Ceres 平方后总代价 ≈ pose_weight² · mean(hinge_i²)，
  // 与采样密度 N 解耦。
  const double weight = 1.0 / std::sqrt(static_cast<double>(sampling.total));

  model.check_points.reserve(static_cast<std::size_t>(sampling.total) * 3u);
  for (int index = 0; index < sampling.total; ++index) {
    const double offset_x = sampling.total == 1 ?
      0.0 : -center_limit + static_cast<double>(index) * sampling.step;
    model.check_points.push_back(offset_x);
    model.check_points.push_back(0.0);
    model.check_points.push_back(weight);
  }
  return model;
}

}  // namespace kinematic_smoother

#endif  // KINEMATIC_SMOOTHER__FOOTPRINT_HPP_
