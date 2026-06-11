#ifndef KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_
#define KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "Eigen/Core"
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

#include "kinematic_path_smoother/esdf.hpp"
#include "kinematic_path_smoother/exceptions.hpp"
#include "kinematic_path_smoother/kinematic_smoother_costs.hpp"
#include "kinematic_path_smoother/math_utils.hpp"
#include "kinematic_path_smoother/options.hpp"

namespace kinematic_path_smoother
{

/// 输入参考路径展开后的内部状态链。
///
/// references、gears、cusp_segments 用于描述优化拓扑；variables 是 Ceres 直接使用的
/// 扁平化状态数组，按 [x, y, theta, kappa, ds] 连续存放。
struct ProcessedPath
{
  /// 每个优化 knot 对应的参考 xy 点。
  std::vector<Eigen::Vector2d> references;
  /// 每段运动方向：1 前进，-1 倒车，0 表示 cusp 保持段。
  std::vector<double> gears;
  /// 每段是否为前进/倒车切换处的保持段。
  std::vector<bool> cusp_segments;
  /// Ceres 参数数组，长度为 size * 5。
  std::vector<double> variables;
  /// 优化 knot 数量。
  std::size_t size{0};
  /// 起点目标 yaw。
  double start_heading{0.0};
  /// 终点目标 yaw。
  double goal_heading{0.0};
  /// 非 cusp 段平均间距，用作 ds 正则参考值。
  double target_spacing{0.2};
};

/// 负责把公共请求转换成 Ceres 问题。
///
/// 顶层 smoother 只做流程编排；ESDF 准备、状态展开、残差拼装、边界和结果重建
/// 都集中在这里，方便单独测试。
class KinematicSmootherProblemBuilder
{
public:
  using Grid = ceres::Grid2D<double>;
  using Interpolator = ceres::BiCubicInterpolator<Grid>;

  explicit KinematicSmootherProblemBuilder(std::vector<double> & esdf)
  : esdf_(esdf)
  {
  }

  /// 准备障碍物残差需要的 ESDF 数组和 Ceres 双三次插值器。
  ///
  /// 未启用障碍物项时会清空缓存，避免后续校验误用旧地图。
  void prepareEsdf(
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed)
  {
    if (!params.obstacleTermsEnabled()) {
      esdf_.clear();
      grid_.reset();
      interpolator_.reset();
      return;
    }
    if (costmap == nullptr) {
      throw InvalidCostmap("Obstacle costs require a costmap");
    }

    const std::size_t expected =
      static_cast<std::size_t>(costmap->getSizeInCellsX()) * costmap->getSizeInCellsY();
    if (precomputed != nullptr) {
      if (precomputed->size() != expected) {
        throw PrecomputedEsdfSizeMismatch("Precomputed ESDF size does not match costmap dimensions");
      }
      esdf_ = *precomputed;
    } else {
      esdf_ = ESDF::ComputeESDF(
        costmap,
        Costmap2D::LETHAL_OBSTACLE,
        params.use_exact_esdf ? ESDFAlgorithm::Exact : ESDFAlgorithm::Approximate);
    }

    grid_ = std::make_shared<Grid>(
      esdf_.data(), 0, costmap->getSizeInCellsY(), 0, costmap->getSizeInCellsX());
    interpolator_ = std::make_shared<Interpolator>(*grid_);
  }

  static ProcessedPath buildProcessedPath(
    const std::vector<Eigen::Vector3d> & input,
    const Eigen::Vector2d & start_direction,
    const Eigen::Vector2d & goal_direction,
    const SmootherParams & params,
    const Costmap2D * costmap)
  {
    ProcessedPath out;
    out.start_heading = headingFromVector(start_direction, 0.0);
    out.goal_heading = headingFromVector(goal_direction, out.start_heading);

    const auto sampled = downsample(input, params);
    std::vector<double> segment_gears;
    segment_gears.reserve(sampled.size() > 0 ? sampled.size() - 1 : 0);
    for (std::size_t i = 0; i + 1 < sampled.size(); ++i) {
      segment_gears.push_back(params.reversing_enabled && sampled[i].z() < 0.0 ? -1.0 : 1.0);
    }

    // 遇到 gear 切换时插入一个重复参考点，形成零位移 cusp 保持段。
    out.references.emplace_back(sampled.front().x(), sampled.front().y());
    for (std::size_t i = 0; i + 1 < sampled.size(); ++i) {
      const double gear = segment_gears[i];
      const double next_gear = i + 1 < segment_gears.size() ? segment_gears[i + 1] : gear;

      out.gears.push_back(gear);
      out.cusp_segments.push_back(false);
      out.references.emplace_back(sampled[i + 1].x(), sampled[i + 1].y());

      if (i + 2 < sampled.size() && gear != next_gear) {
        out.gears.push_back(0.0);
        out.cusp_segments.push_back(true);
        out.references.emplace_back(sampled[i + 1].x(), sampled[i + 1].y());
      }
    }

    out.size = out.references.size();
    std::vector<double> heading(out.size, out.start_heading);
    std::vector<double> ds(out.size, 0.0);

    // 从参考几何初始化 theta/ds。倒车段的车身朝向与路径切线相差 pi。
    double spacing_sum = 0.0;
    std::size_t spacing_count = 0;
    for (std::size_t i = 0; i + 1 < out.size; ++i) {
      if (out.cusp_segments[i]) {
        heading[i] = i > 0 ? heading[i - 1] : out.start_heading;
        continue;
      }

      const Eigen::Vector2d delta = out.references[i + 1] - out.references[i];
      const double length = delta.norm();
      if (length <= kEpsilon) {
        heading[i] = i > 0 ? heading[i - 1] : out.start_heading;
        continue;
      }

      double segment_heading = std::atan2(delta.y(), delta.x());
      if (out.gears[i] < 0.0) {
        segment_heading += kPi;
      }
      heading[i] = normalizeAngle(segment_heading);
      ds[i] = length;
      spacing_sum += length;
      ++spacing_count;
    }

    if (out.size > 1) {
      heading.back() = heading[out.size - 2];
    }
    if (params.keep_start_orientation) {
      heading.front() = out.start_heading;
    }
    if (params.keep_goal_orientation) {
      heading.back() = out.goal_heading;
    }

    // 目标间距使用非 cusp 段平均长度；全零路径时退回到地图分辨率或默认值。
    if (spacing_count > 0) {
      out.target_spacing = spacing_sum / static_cast<double>(spacing_count);
    } else if (costmap != nullptr) {
      out.target_spacing = std::max(costmap->getResolution(), 1e-3);
    }

    // Ceres 参数块采用 AoS 布局：每个 knot 一个 5 维参数块。
    out.variables.reserve(out.size * 5);
    for (std::size_t i = 0; i < out.size; ++i) {
      out.variables.push_back(out.references[i].x());
      out.variables.push_back(out.references[i].y());
      out.variables.push_back(heading[i]);
      out.variables.push_back(0.0);
      out.variables.push_back(ds[i]);
    }
    return out;
  }

  void addResiduals(
    const ProcessedPath & path,
    const Costmap2D * costmap,
    const SmootherParams & params,
    std::vector<double> & variables,
    ceres::Problem & problem) const
  {
    // Ceres 会平方 residual。公开参数用目标函数权重表达，进入 residual 前取 sqrt。
    const double model_weight = sqrtWeight(params.model_weight);
    const double curvature_weight = sqrtWeight(params.curvature_weight);
    const double curvature_rate_weight = sqrtWeight(params.curvature_rate_weight);
    const double spacing_weight = sqrtWeight(params.spacing_weight);
    const double length_weight = sqrtWeight(params.length_weight);
    const double reference_weight = sqrtWeight(params.reference_weight);
    const double fix_weight = std::max(params.fix_weight, 0.0);

    // 每个相邻状态对添加一个运动学转移残差。
    for (std::size_t i = 0; i + 1 < path.size; ++i) {
      problem.AddResidualBlock(
        detail::MotionCost::Create(
          path.gears[i],
          path.cusp_segments[i],
          model_weight,
          curvature_weight,
          curvature_rate_weight,
          spacing_weight,
          length_weight,
          fix_weight,
          path.target_spacing),
        nullptr,
        state(variables, i),
        state(variables, i + 1));
    }

    // 起点严格固定位置；朝向是否固定由 params 控制。
    problem.AddResidualBlock(
      detail::EndpointCost::Create(
        path.references.front(), path.start_heading, params.keep_start_orientation,
        0.0, 0.0, 0.0, fix_weight, false),
      nullptr,
      state(variables, 0));

    // 终点支持 lon/lat/yaw 容差，便于表达“停在目标范围内”。
    const double goal_position_heading =
      goalFrameHeading(path.references, path.goal_heading, params.keep_goal_orientation);
    problem.AddResidualBlock(
      detail::EndpointCost::Create(
        path.references.back(),
        goal_position_heading,
        params.keep_goal_orientation,
        params.goal_longitudinal_tolerance,
        params.goal_lateral_tolerance,
        params.goal_orientation_tolerance,
        fix_weight,
        true),
      nullptr,
      state(variables, path.size - 1));

    // 参考吸附是软约束，通常与 max_reference_deviation 硬边界配合使用。
    if (reference_weight > 1e-9) {
      for (std::size_t i = 0; i < path.size; ++i) {
        problem.AddResidualBlock(
          detail::ReferenceCost::Create(path.references[i], reference_weight),
          nullptr,
          state(variables, i));
      }
    }

    // 障碍物残差复用 prepareEsdf() 创建的插值器。
    if (params.obstacleTermsEnabled()) {
      for (std::size_t i = 0; i < path.size; ++i) {
        const bool cusp_pose =
          (i < path.cusp_segments.size() && path.cusp_segments[i]) ||
          (i > 0 && path.cusp_segments[i - 1]);
        problem.AddResidualBlock(
          detail::ObstacleCost::Create(cusp_pose, *costmap, params, grid_, interpolator_),
          nullptr,
          state(variables, i));
      }
    }
  }

  static void applyBounds(
    ceres::Problem & problem,
    std::vector<double> & variables,
    const ProcessedPath & path,
    const SmootherParams & params)
  {
    // kappa 和 ds 使用硬边界；xy 偏移边界可选。
    const double max_curvature = std::max(params.max_curvature, 1e-6);
    for (std::size_t i = 0; i < path.size; ++i) {
      double * block = state(variables, i);
      if (params.max_reference_deviation > 1e-9) {
        problem.SetParameterLowerBound(block, 0, path.references[i].x() - params.max_reference_deviation);
        problem.SetParameterUpperBound(block, 0, path.references[i].x() + params.max_reference_deviation);
        problem.SetParameterLowerBound(block, 1, path.references[i].y() - params.max_reference_deviation);
        problem.SetParameterUpperBound(block, 1, path.references[i].y() + params.max_reference_deviation);
      }
      problem.SetParameterLowerBound(block, 3, -max_curvature);
      problem.SetParameterUpperBound(block, 3, max_curvature);
      const bool ds_is_used = i + 1 < path.size;
      const bool is_cusp_ds = i < path.cusp_segments.size() && path.cusp_segments[i];
      problem.SetParameterLowerBound(block, 4, ds_is_used && !is_cusp_ds ? 1e-6 : 0.0);
      if (params.max_segment_length > 1e-9) {
        problem.SetParameterUpperBound(block, 4, params.max_segment_length);
      }
    }
  }

  static std::vector<Eigen::Vector3d> unpack(
    const std::vector<double> & variables,
    std::size_t count)
  {
    // 对外路径只暴露 (x, y, yaw)，不暴露内部 kappa/ds。
    std::vector<Eigen::Vector3d> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      result.emplace_back(variables[5 * i], variables[5 * i + 1], normalizeAngle(variables[5 * i + 2]));
    }
    return result;
  }

  static std::vector<Eigen::Vector3d> upsample(
    const std::vector<double> & variables,
    const ProcessedPath & path,
    const SmootherParams & params)
  {
    // 求解 knot 可能较稀疏；输出阶段按同一运动学模型做段内插值。
    const int factor = std::max(params.path_upsampling_factor, 1);
    auto knots = unpack(variables, path.size);
    if (factor <= 1 || path.size < 2) {
      return knots;
    }

    std::vector<Eigen::Vector3d> dense;
    dense.reserve(static_cast<std::size_t>(factor) * (path.size - 1) + 1);
    dense.push_back(knots.front());

    for (std::size_t i = 0; i + 1 < path.size; ++i) {
      if (path.cusp_segments[i] || std::abs(path.gears[i]) < 1e-9) {
        dense.push_back(knots[i + 1]);
        continue;
      }

      // 按线性曲率变化积分生成中间点，再把闭合误差均匀摊回段内。
      const double direction = path.gears[i] < 0.0 ? -1.0 : 1.0;
      const double step = std::max(variables[5 * i + 4], 0.0) / static_cast<double>(factor);
      if (step <= 1e-9) {
        dense.push_back(knots[i + 1]);
        continue;
      }

      double x = variables[5 * i];
      double y = variables[5 * i + 1];
      double theta = normalizeAngle(variables[5 * i + 2]);
      const double kappa0 = variables[5 * i + 3];
      const double kappa1 = variables[5 * (i + 1) + 3];

      std::vector<Eigen::Vector3d> samples;
      samples.reserve(static_cast<std::size_t>(factor - 1));
      for (int j = 1; j < factor; ++j) {
        const double t0 = static_cast<double>(j - 1) / static_cast<double>(factor);
        const double t1 = static_cast<double>(j) / static_cast<double>(factor);
        const double k0 = kappa0 + (kappa1 - kappa0) * t0;
        const double k1 = kappa0 + (kappa1 - kappa0) * t1;
        const double theta_mid = theta + direction * step * k0 * 0.5;
        x += direction * step * std::cos(theta_mid);
        y += direction * step * std::sin(theta_mid);
        theta = normalizeAngle(theta + direction * step * (k0 + k1) * 0.5);
        samples.emplace_back(x, y, theta);
      }

      const double final_t = static_cast<double>(factor - 1) / static_cast<double>(factor);
      const double final_kappa = kappa0 + (kappa1 - kappa0) * final_t;
      const double predicted_x = x + direction * step * std::cos(theta + direction * step * final_kappa * 0.5);
      const double predicted_y = y + direction * step * std::sin(theta + direction * step * final_kappa * 0.5);
      const double predicted_theta =
        normalizeAngle(theta + direction * step * (final_kappa + kappa1) * 0.5);
      const Eigen::Vector3d end = knots[i + 1];
      const double close_x = end.x() - predicted_x;
      const double close_y = end.y() - predicted_y;
      const double close_theta = angleDifference(end.z(), predicted_theta);

      for (int j = 1; j < factor; ++j) {
        const double t = static_cast<double>(j) / static_cast<double>(factor);
        const auto & sample = samples[static_cast<std::size_t>(j - 1)];
        dense.emplace_back(
          sample.x() + t * close_x,
          sample.y() + t * close_y,
          normalizeAngle(sample.z() + t * close_theta));
      }
      dense.push_back(end);
    }
    return dense;
  }

private:
  static std::vector<Eigen::Vector3d> downsample(
    const std::vector<Eigen::Vector3d> & path,
    const SmootherParams & params)
  {
    const int stride = std::max(params.path_downsampling_factor, 1);
    if (stride <= 1 || path.size() <= 2) {
      return path;
    }

    std::vector<Eigen::Vector3d> sampled;
    sampled.reserve(path.size());
    sampled.push_back(path.front());
    std::size_t last = 0;

    auto sign_at = [&](std::size_t i) {
      return params.reversing_enabled && path[i].z() < 0.0 ? -1.0 : 1.0;
    };

    // 下采样时保留 cusp 邻域点，否则可能丢失方向切换语义。
    for (std::size_t i = 1; i + 1 < path.size(); ++i) {
      const bool near_cusp = sign_at(i - 1) != sign_at(i) || sign_at(i) != sign_at(i + 1);
      if (near_cusp || static_cast<int>(i - last) >= stride) {
        sampled.push_back(path[i]);
        last = i;
      }
    }
    if (!sampled.back().isApprox(path.back(), 1e-12)) {
      sampled.push_back(path.back());
    }
    return sampled;
  }

  static double * state(std::vector<double> & variables, std::size_t index)
  {
    return variables.data() + 5 * index;
  }

  static double sqrtWeight(double weight)
  {
    return std::sqrt(std::max(weight, 0.0));
  }

  std::vector<double> & esdf_;
  std::shared_ptr<Grid> grid_{};
  std::shared_ptr<Interpolator> interpolator_{};
};

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_
