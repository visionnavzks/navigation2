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

#ifndef CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_
#define CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ceres/ceres.h"
#include "Eigen/Core"

#include "kinematic_smoother/esdf.hpp"
#include "kinematic_smoother/exceptions.hpp"
#include "kinematic_smoother/kinematic_smoother_costs.hpp"
#include "kinematic_smoother/options.hpp"
#include "kinematic_smoother/smoother_request.hpp"
#include "kinematic_smoother/state_layout.hpp"
#include "kinematic_smoother/utils.hpp"

namespace kinematic_smoother
{

/// 运动学版 smoother 展开的内部状态链。
///
/// 这里会把参考路径扩展成显式的
/// (x, y, theta, kappa, ds) 状态序列，并保留 cusp 段的附加元数据。
struct KinematicProcessedPath
{
  std::vector<Eigen::Vector2d> reference_points{};
  std::vector<double> gears{};
  std::vector<bool> is_cusp_segment{};
  std::vector<double> initial_variables{};
  size_t state_count{0};
  double start_theta{0.0};
  double end_theta{0.0};
  double target_spacing{0.2};
};

/// 运动学输出上采样结果及其同源曲率诊断 profile。
struct KinematicUpsampledPathProfile
{
  std::vector<Eigen::Vector3d> path{};
  std::vector<double> curvatures{};
  std::vector<double> curvature_rates{};
};

/// buildProblem() 期间记录的各代价类别残差块 ID。
///
/// 求解前后可以据此用 ceres::Problem::Evaluate 按类别拆分代价，
/// 供诊断 / UI 展示每个代价项对总代价的贡献。
struct KinematicResidualCatalog
{
  std::vector<ceres::ResidualBlockId> transition_blocks{};
  std::vector<ceres::ResidualBlockId> start_boundary_blocks{};
  std::vector<ceres::ResidualBlockId> goal_boundary_blocks{};
  std::vector<ceres::ResidualBlockId> reference_blocks{};
  std::vector<ceres::ResidualBlockId> obstacle_blocks{};
};

/// 运动学版 smoother 的问题构建器。
///
/// 它把 ESDF 准备、状态展开、残差拼接、显式边界约束和结果解包集中在一起，
/// 让顶层 smoother 只保留编排与后验校验职责。
class KinematicSmootherProblemBuilder
{
public:
  using EsdfGrid = ceres::Grid2D<double>;

  explicit KinematicSmootherProblemBuilder(std::vector<double> & esdf_values)
  : esdf_values_(esdf_values)
  {
  }

  void initializeEsdfValues(
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
    if (!params.obstacleTermsEnabled()) {
      esdf_values_.clear();
      esdf_grid_.reset();
      return;
    }

    // 这份 ESDF 会被障碍物残差和最终后验校验共同复用。
    const size_t expected_esdf_size =
      static_cast<size_t>(costmap->getSizeInCellsX()) * costmap->getSizeInCellsY();
    if (precomputed_esdf != nullptr) {
      if (precomputed_esdf->size() != expected_esdf_size) {
        throw PrecomputedEsdfSizeMismatch(
                "Precomputed ESDF size does not match costmap dimensions");
      }
      esdf_values_ = *precomputed_esdf;
    } else {
      esdf_values_ = ESDF::ComputeESDF(
        costmap,
        Costmap2D::LETHAL_OBSTACLE,
        params.use_exact_esdf ? ESDFAlgorithm::Exact : ESDFAlgorithm::Approximate);
    }

    esdf_grid_ = std::make_shared<EsdfGrid>(
      esdf_values_.data(), 0, costmap->getSizeInCellsY(), 0, costmap->getSizeInCellsX());
  }

  static KinematicProcessedPath buildProcessedPath(
    const std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params,
    const Costmap2D * costmap)
  {
    // 先展开 gear / cusp 元数据，再把参考几何转成求解器可用的状态初值。
    KinematicProcessedPath processed;
    processed.start_theta = std::atan2(start_dir.y(), start_dir.x());
    processed.end_theta = std::atan2(end_dir.y(), end_dir.x());

    const std::vector<Eigen::Vector3d> sampled_path = downsampleInputPath(path, params);

    std::vector<double> gear_directions;
    gear_directions.reserve(sampled_path.size() - 1);
    for (size_t index = 0; index + 1 < sampled_path.size(); ++index) {
      gear_directions.push_back(directionSign(sampled_path[index], params.reversing_enabled));
    }

    processed.reference_points.emplace_back(sampled_path.front().x(), sampled_path.front().y());
    for (size_t index = 0; index + 1 < sampled_path.size(); ++index) {
      const double current_gear = gear_directions[index];
      const double next_gear = index + 1 < gear_directions.size() ? gear_directions[index + 1] : current_gear;
      const Eigen::Vector2d current_point(sampled_path[index].x(), sampled_path[index].y());
      const Eigen::Vector2d next_point(sampled_path[index + 1].x(), sampled_path[index + 1].y());

      if (
        index + 2 < sampled_path.size() &&
        current_gear != next_gear &&
        (next_point - current_point).norm() <= KinematicStateLayout::PointEpsilon)
      {
        processed.gears.push_back(0.0);
        processed.is_cusp_segment.push_back(true);
        processed.reference_points.push_back(next_point);
        continue;
      }

      processed.gears.push_back(current_gear);
      processed.is_cusp_segment.push_back(false);
      processed.reference_points.push_back(next_point);

      if (index + 2 < sampled_path.size() && current_gear != next_gear) {
        processed.gears.push_back(0.0);
        processed.is_cusp_segment.push_back(true);
        processed.reference_points.push_back(next_point);
      }
    }

    processed.state_count = processed.reference_points.size();
    std::vector<double> theta(processed.state_count, 0.0);
    std::vector<double> kappa(processed.state_count, 0.0);
    std::vector<double> ds(processed.state_count, 0.0);

    double spacing_sum = 0.0;
    size_t spacing_count = 0;
    for (size_t index = 0; index + 1 < processed.state_count; ++index) {
      const Eigen::Vector2d delta = processed.reference_points[index + 1] - processed.reference_points[index];
      const double segment_norm = delta.norm();
      if (processed.is_cusp_segment[index]) {
        theta[index] = index > 0 ? theta[index - 1] : processed.start_theta;
        ds[index] = 0.0;
        continue;
      }

      if (segment_norm > KinematicStateLayout::GeometryEpsilon) {
        double heading = std::atan2(delta.y(), delta.x());
        if (processed.gears[index] < 0.0) {
          heading += kinematic_smoother::PI;
        }
        theta[index] = normalizeAngle(heading);
        ds[index] = segment_norm;
        spacing_sum += segment_norm;
        ++spacing_count;
      } else {
        theta[index] = index > 0 ? theta[index - 1] : processed.start_theta;
      }
    }

    theta.back() = theta.size() > 1 ? theta[theta.size() - 2] : processed.start_theta;
    if (params.keep_start_orientation) {
      theta.front() = processed.start_theta;
    }
    if (params.keep_goal_orientation) {
      theta.back() = processed.end_theta;
    }

    if (params.path_target_spacing > KinematicStateLayout::EnabledEpsilon) {
      processed.target_spacing = params.path_target_spacing;
    } else {
      processed.target_spacing = spacing_count > 0 ?
        spacing_sum / static_cast<double>(spacing_count) :
        (costmap != nullptr ? std::max(costmap->getResolution(), 1e-3) : processed.target_spacing);
    }

    processed.initial_variables.reserve(processed.state_count * KinematicStateLayout::Size);
    for (size_t index = 0; index < processed.state_count; ++index) {
      processed.initial_variables.push_back(processed.reference_points[index].x());
      processed.initial_variables.push_back(processed.reference_points[index].y());
      processed.initial_variables.push_back(theta[index]);
      processed.initial_variables.push_back(kappa[index]);
      processed.initial_variables.push_back(ds[index]);
    }

    return processed;
  }

  KinematicResidualCatalog buildProblem(
    const KinematicProcessedPath & processed,
    const Costmap2D * costmap,
    const SmootherParams & params,
    std::vector<double> & variables,
    ceres::Problem & problem) const
  {
    // 调用方必须先用 buildProcessedPath() 生成 processed，并把 variables 初始化为状态初值。
    // 大多数权重由调用方传入平方后的值，代码内部自动开方；fix_weight 是直接约束系数，不再额外开方。
    KinematicResidualCatalog catalog;
    const double model_weight = std::sqrt(std::max(params.model_weight, 0.0));
    const double curvature_weight = std::sqrt(std::max(params.kinematic_curvature_weight, 0.0));
    const double curvature_rate_weight =
      std::sqrt(std::max(params.kinematic_curvature_rate_weight, 0.0));
    const double spacing_weight = std::sqrt(std::max(params.kinematic_spacing_weight, 0.0));
    const double length_weight = std::sqrt(std::max(params.path_length_weight, 0.0));
    const double fix_weight = std::max(params.fix_weight, 0.0);
    const double reference_weight = std::sqrt(std::max(params.reference_path_weight, 0.0));
    const bool has_obstacle_cost = params.obstacleTermsEnabled();

    // 邻接状态过渡残差：约束运动学一致性、曲率、曲率变化率与间距均匀性。
    catalog.transition_blocks.reserve(
      processed.state_count > 0 ? processed.state_count - 1 : 0);
    for (size_t index = 0; index + 1 < processed.state_count; ++index) {
      // 均匀间距需要比较两个真实路径段 ds[index] 和 ds[index + 1]。
      // 末状态没有对应路径段，cusp 段的 ds 则固定为 0，都不参与差分。
      const bool compare_next_spacing =
        index + 1 < processed.is_cusp_segment.size() &&
        !processed.is_cusp_segment[index] &&
        !processed.is_cusp_segment[index + 1];
      catalog.transition_blocks.push_back(
        problem.AddResidualBlock(
          kinematic_smoother_detail::TransitionCostFunctor::Create(
            processed.gears[index],
            processed.is_cusp_segment[index],
            model_weight,
            curvature_weight,
            curvature_rate_weight,
            spacing_weight,
            length_weight,
            fix_weight,
            params.max_curvature,
            processed.target_spacing,
            compare_next_spacing),
          nullptr,
          KinematicStateLayout::data(variables, index),
          KinematicStateLayout::data(variables, index + 1)));
    }

    // 起点边界残差：位置固定，朝向是否固定由 keep_start_orientation 控制。
    catalog.start_boundary_blocks.push_back(
      problem.AddResidualBlock(
        kinematic_smoother_detail::BoundaryCostFunctor::Create(
          processed.reference_points.front(),
          processed.start_theta,
          params.keep_start_orientation,
          0.0,
          0.0,
          0.0,
          fix_weight),
        nullptr,
        KinematicStateLayout::data(variables, 0)));

    // 终点位置容差框所用的参考朝向：
    // keep_goal_orientation=true 时采用 end_theta，否则采用末段几何朝向。
    const double goal_position_theta = goalPositionFrameHeading(
      processed.reference_points,
      processed.end_theta,
      params.keep_goal_orientation);

    // 终点边界残差：支持纵向/横向容差与可选朝向固定。
    catalog.goal_boundary_blocks.push_back(
      problem.AddResidualBlock(
        kinematic_smoother_detail::BoundaryCostFunctor::Create(
          processed.reference_points.back(),
          goal_position_theta,
          params.keep_goal_orientation,
          params.goal_longitudinal_tolerance,
          params.goal_lateral_tolerance,
          params.goal_orientation_tolerance,
          fix_weight),
        nullptr,
        KinematicStateLayout::data(variables, processed.state_count - 1)));

    // 参考路径吸附残差：仅在 reference_weight>0 时启用。
    if (reference_weight > KinematicStateLayout::EnabledEpsilon) {
      catalog.reference_blocks.reserve(processed.state_count);
      for (size_t index = 0; index < processed.state_count; ++index) {
        catalog.reference_blocks.push_back(
          problem.AddResidualBlock(
            kinematic_smoother_detail::ReferenceCostFunctor::Create(
              processed.reference_points[index], reference_weight),
            nullptr,
            KinematicStateLayout::data(variables, index)));
      }
    }

    // 障碍物残差：所有状态使用统一的 ESDF 障碍物权重。
    if (has_obstacle_cost) {
      const double obstacle_weight = std::sqrt(std::max(params.obstacle_weight, 0.0));
      catalog.obstacle_blocks.reserve(processed.state_count);
      for (size_t index = 0; index < processed.state_count; ++index) {
        catalog.obstacle_blocks.push_back(
          problem.AddResidualBlock(
            kinematic_smoother_detail::ObstacleCostFunctor::Create(
              obstacle_weight, costmap, params, esdf_grid_),
            nullptr,
            KinematicStateLayout::data(variables, index)));
      }
    }

    return catalog;
  }

  /// 代价分项的稳定名称表；顺序与 evaluateCostTermValues() 返回值一一对应。
  ///
  /// 前 5 项对应 TransitionCostFunctor 的 7 个残差按语义拆分：
  /// [0..2] 运动学模型（尖点段为固定约束）、[3] 曲率、[4] 曲率变化率、
  /// [5] 间距（尖点段为零步长约束）、[6] 路径长度。
  static const std::array<const char *, 9> & costTermNames()
  {
    static const std::array<const char *, 9> names = {
      "kinematic_model",
      "curvature",
      "curvature_rate",
      "spacing",
      "path_length",
      "start_boundary",
      "goal_boundary",
      "reference_path",
      "obstacle",
    };
    return names;
  }

  /// 按 costTermNames() 顺序评估当前变量下各代价分项的代价。
  ///
  /// 代价口径与 Ceres 总代价一致（0.5·Σr²），因此所有分项之和等于问题总代价。
  /// 可在求解前后各调用一次，得到初始 / 最终代价对比。
  static std::vector<double> evaluateCostTermValues(
    ceres::Problem & problem,
    const KinematicResidualCatalog & catalog)
  {
    std::vector<double> values(costTermNames().size(), 0.0);

    // Transition 块的 7 个残差按索引拆分为前 5 个代价分项。
    if (!catalog.transition_blocks.empty()) {
      ceres::Problem::EvaluateOptions options;
      options.residual_blocks = catalog.transition_blocks;
      options.apply_loss_function = true;
      double cost = 0.0;
      std::vector<double> residuals;
      problem.Evaluate(options, &cost, &residuals, nullptr, nullptr);
      constexpr size_t kTransitionResidualCount = 7;
      for (size_t offset = 0; offset + kTransitionResidualCount <= residuals.size();
        offset += kTransitionResidualCount)
      {
        auto half_squared = [&](size_t index) {
            const double residual = residuals[offset + index];
            return 0.5 * residual * residual;
          };
        values[0] += half_squared(0) + half_squared(1) + half_squared(2);
        values[1] += half_squared(3);
        values[2] += half_squared(4);
        values[3] += half_squared(5);
        values[4] += half_squared(6);
      }
    }

    auto blocks_cost = [&problem](const std::vector<ceres::ResidualBlockId> & blocks) {
        if (blocks.empty()) {
          return 0.0;
        }
        ceres::Problem::EvaluateOptions options;
        options.residual_blocks = blocks;
        options.apply_loss_function = true;
        double cost = 0.0;
        problem.Evaluate(options, &cost, nullptr, nullptr, nullptr);
        return cost;
      };
    values[5] = blocks_cost(catalog.start_boundary_blocks);
    values[6] = blocks_cost(catalog.goal_boundary_blocks);
    values[7] = blocks_cost(catalog.reference_blocks);
    values[8] = blocks_cost(catalog.obstacle_blocks);
    return values;
  }

  static void applyBounds(
    ceres::Problem & problem,
    double * variables,
    const std::vector<Eigen::Vector2d> & reference_points,
    const std::vector<bool> & is_cusp_segment,
    size_t state_count,
    double max_curvature,
    double max_spacing,
    double reference_point_max_deviation_m)
  {
    // 显式参数边界：
    // x/y 可选地限制在参考点邻域；kappa 与 ds 始终受物理边界约束。
    const double clamped_max_curvature =
      std::max(max_curvature, KinematicStateLayout::GeometryEpsilon);
    for (size_t index = 0; index < state_count; ++index) {
      double * state = KinematicStateLayout::data(variables, index);
      if (reference_point_max_deviation_m > KinematicStateLayout::EnabledEpsilon) {
        problem.SetParameterLowerBound(
          state,
          KinematicStateLayout::X,
          reference_points[index].x() - reference_point_max_deviation_m);
        problem.SetParameterUpperBound(
          state,
          KinematicStateLayout::X,
          reference_points[index].x() + reference_point_max_deviation_m);
        problem.SetParameterLowerBound(
          state,
          KinematicStateLayout::Y,
          reference_points[index].y() - reference_point_max_deviation_m);
        problem.SetParameterUpperBound(
          state,
          KinematicStateLayout::Y,
          reference_points[index].y() + reference_point_max_deviation_m);
      }
      problem.SetParameterLowerBound(state, KinematicStateLayout::Kappa, -clamped_max_curvature);
      problem.SetParameterUpperBound(state, KinematicStateLayout::Kappa, clamped_max_curvature);
      const bool ds_is_used = index + 1 < state_count;
      const bool is_cusp_ds = index < is_cusp_segment.size() && is_cusp_segment[index];
      const double ds_lower =
        ds_is_used && !is_cusp_ds ? KinematicStateLayout::GeometryEpsilon : 0.0;
      problem.SetParameterLowerBound(state, KinematicStateLayout::Ds, ds_lower);
      double ds_upper = std::numeric_limits<double>::infinity();
      if (ds_is_used && !is_cusp_ds && max_spacing > KinematicStateLayout::EnabledEpsilon) {
        // 上界绝不能低于下界（例如 max_spacing 配成亚微米级），否则可行域为空，
        // Ceres 会直接把问题判为 infeasible。
        ds_upper = std::max(max_spacing, ds_lower);
        problem.SetParameterUpperBound(state, KinematicStateLayout::Ds, ds_upper);
      }
      // 近零长度的非 cusp 段初值 ds 为 0，会落在刚设置的下界 (GeometryEpsilon) 之外。
      // 把初值夹回 [ds_lower, ds_upper]，否则 Ceres 以 "infeasible initial point"
      // 直接拒绝整次求解（重复/近重合的输入路点即可触发）。
      double & ds_value = state[KinematicStateLayout::Ds];
      ds_value = std::min(std::max(ds_value, ds_lower), ds_upper);
    }
  }

  static std::vector<Eigen::Vector3d> unpackPath(
    const std::vector<double> & variables,
    size_t state_count)
  {
    // 将求解变量回写为公共路径格式：(x, y, yaw)。
    std::vector<Eigen::Vector3d> path;
    path.reserve(state_count);
    for (size_t index = 0; index < state_count; ++index) {
      const size_t offset = KinematicStateLayout::offset(index);
      path.emplace_back(
        variables[offset + KinematicStateLayout::X],
        variables[offset + KinematicStateLayout::Y],
        normalizeAngle(variables[offset + KinematicStateLayout::Theta]));
    }
    return path;
  }

  static KinematicUpsampledPathProfile upsamplePathKinematicProfile(
    const std::vector<double> & variables,
    const KinematicProcessedPath & processed,
    const SmootherParams & params)
  {
    std::vector<Eigen::Vector3d> path = unpackPath(variables, processed.state_count);
    KinematicUpsampledPathProfile profile;
    if (processed.state_count == 0) {
      return profile;
    }

    const double undefined_rate = std::numeric_limits<double>::quiet_NaN();
    auto segment_curvature_rate = [&](size_t index) {
        if (index + 1 >= processed.state_count) {
          return undefined_rate;
        }
        const bool is_cusp_segment =
          index < processed.is_cusp_segment.size() && processed.is_cusp_segment[index];
        const double gear = index < processed.gears.size() ? processed.gears[index] : 1.0;
        const double * state = KinematicStateLayout::data(variables, index);
        const double * next_state = KinematicStateLayout::data(variables, index + 1);
        const double ds = std::max(state[KinematicStateLayout::Ds], 0.0);
        if (
          is_cusp_segment ||
          std::abs(gear) < KinematicStateLayout::EnabledEpsilon ||
          ds <= KinematicStateLayout::GeometryEpsilon)
        {
          return undefined_rate;
        }
        return (next_state[KinematicStateLayout::Kappa] - state[KinematicStateLayout::Kappa]) / ds;
      };

    auto append_sample = [&](const Eigen::Vector3d & pose, double curvature, double rate) {
        profile.path.push_back(pose);
        profile.curvatures.push_back(curvature);
        profile.curvature_rates.push_back(rate);
      };

    const double first_curvature =
      KinematicStateLayout::data(variables, 0)[KinematicStateLayout::Kappa];
    append_sample(path.front(), first_curvature, segment_curvature_rate(0));

    const bool use_output_spacing =
      params.path_output_spacing > KinematicStateLayout::EnabledEpsilon;
    // 上采样倍率未做范围校验；夹到一个合理上限，避免巨大取值在下面的
    // reserve()（factor * (state_count-1)）里整型溢出或触发 OOM。
    const int fallback_upsample_factor =
      std::min(std::max(params.path_upsampling_factor, 1), kMaxStepsPerSegment);
    if (processed.state_count < 2) {
      return profile;
    }

    profile.path.reserve(
      static_cast<size_t>(fallback_upsample_factor) * (processed.state_count - 1) + 1);
    profile.curvatures.reserve(profile.path.capacity());
    profile.curvature_rates.reserve(profile.path.capacity());

    for (size_t index = 0; index + 1 < processed.state_count; ++index) {
      const bool is_cusp_segment =
        index < processed.is_cusp_segment.size() && processed.is_cusp_segment[index];
      const double gear = index < processed.gears.size() ? processed.gears[index] : 1.0;

      const double * state = KinematicStateLayout::data(variables, index);
      const double * next_state = KinematicStateLayout::data(variables, index + 1);
      const double x = state[KinematicStateLayout::X];
      const double y = state[KinematicStateLayout::Y];
      const double theta = normalizeAngle(state[KinematicStateLayout::Theta]);
      const double kappa = state[KinematicStateLayout::Kappa];
      const double ds = std::max(state[KinematicStateLayout::Ds], 0.0);
      const double next_kappa = next_state[KinematicStateLayout::Kappa];
      const double curvature_rate = segment_curvature_rate(index);

      const Eigen::Vector3d & next_pose = path[index + 1];

      if (
        is_cusp_segment ||
        std::abs(gear) < KinematicStateLayout::EnabledEpsilon ||
        ds <= KinematicStateLayout::GeometryEpsilon)
      {
        append_sample(next_pose, next_kappa, undefined_rate);
        continue;
      }

      const int segment_step_count = segmentStepCount(
        ds, params.path_output_spacing, fallback_upsample_factor, use_output_spacing);
      if (segment_step_count <= 1) {
        append_sample(next_pose, next_kappa, curvature_rate);
        continue;
      }

      const double direction = gear >= 0.0 ? 1.0 : -1.0;
      const double step = ds / static_cast<double>(segment_step_count);

      double interp_x = x;
      double interp_y = y;
      double interp_theta = theta;
      std::vector<Eigen::Vector3d> segment_samples;
      segment_samples.reserve(static_cast<size_t>(segment_step_count - 1));

      for (int step_index = 1; step_index < segment_step_count; ++step_index) {
        const double t0 = static_cast<double>(step_index - 1) /
          static_cast<double>(segment_step_count);
        const double t1 = static_cast<double>(step_index) /
          static_cast<double>(segment_step_count);
        const double kappa0 = kappa + (next_kappa - kappa) * t0;
        const double kappa1 = kappa + (next_kappa - kappa) * t1;

        const double theta_mid = interp_theta + direction * step * 0.5 * kappa0;
        interp_x += direction * step * std::cos(theta_mid);
        interp_y += direction * step * std::sin(theta_mid);
        interp_theta = normalizeAngle(interp_theta + direction * step * 0.5 * (kappa0 + kappa1));
        segment_samples.emplace_back(interp_x, interp_y, interp_theta);
      }

      const double final_t0 = static_cast<double>(segment_step_count - 1) /
        static_cast<double>(segment_step_count);
      const double final_kappa0 = kappa + (next_kappa - kappa) * final_t0;
      const double final_theta_mid = interp_theta + direction * step * 0.5 * final_kappa0;
      const double predicted_end_x = interp_x + direction * step * std::cos(final_theta_mid);
      const double predicted_end_y = interp_y + direction * step * std::sin(final_theta_mid);
      const double predicted_end_theta = normalizeAngle(
        interp_theta + direction * step * 0.5 * (final_kappa0 + next_kappa));

      const double closure_x = next_pose.x() - predicted_end_x;
      const double closure_y = next_pose.y() - predicted_end_y;
      const double closure_theta = normalizeAngle(next_pose.z() - predicted_end_theta);

      // 优化后的相邻状态只在有限权重下逼近运动学一致性；
      // 将端点闭合误差沿整段均匀摊开，避免最后一个插值点硬跳到 next_pose。
      for (int step_index = 1; step_index < segment_step_count; ++step_index) {
        const double t = static_cast<double>(step_index) / static_cast<double>(segment_step_count);
        const Eigen::Vector3d & sample = segment_samples[static_cast<size_t>(step_index - 1)];
        append_sample(
          Eigen::Vector3d(
            sample.x() + t * closure_x,
            sample.y() + t * closure_y,
            normalizeAngle(sample.z() + t * closure_theta)),
          kappa + (next_kappa - kappa) * t,
          curvature_rate);
      }

      append_sample(next_pose, next_kappa, curvature_rate);
    }

    return profile;
  }

private:
  static std::vector<Eigen::Vector3d> downsampleInputPath(
    const std::vector<Eigen::Vector3d> & path,
    const SmootherParams & params)
  {
    if (params.path_target_spacing > KinematicStateLayout::EnabledEpsilon) {
      return resampleInputPathBySpacing(path, params);
    }

    const int downsample_factor = std::max(params.path_downsampling_factor, 1);
    if (downsample_factor <= 1 || path.size() <= 2) {
      return path;
    }

    std::vector<Eigen::Vector3d> sampled;
    sampled.reserve(path.size());
    sampled.push_back(path.front());

    size_t last_kept_index = 0;
    for (size_t index = 1; index + 1 < path.size(); ++index) {
      const double prev_sign = directionSign(path[index - 1], params.reversing_enabled);
      const double current_sign = directionSign(path[index], params.reversing_enabled);
      const double next_sign = directionSign(path[index + 1], params.reversing_enabled);
      const bool around_cusp = (current_sign != prev_sign) || (current_sign != next_sign);

      if (around_cusp || static_cast<int>(index - last_kept_index) >= downsample_factor) {
        sampled.push_back(path[index]);
        last_kept_index = index;
      }
    }

    if (!sampled.back().isApprox(path.back(), KinematicStateLayout::PointEpsilon)) {
      sampled.push_back(path.back());
    }

    if (sampled.size() < 2) {
      sampled = {path.front(), path.back()};
    }

    return sampled;
  }

  static std::vector<Eigen::Vector3d> resampleInputPathBySpacing(
    const std::vector<Eigen::Vector3d> & path,
    const SmootherParams & params)
  {
    const double target_spacing = std::max(params.path_target_spacing, 0.0);
    if (target_spacing <= KinematicStateLayout::EnabledEpsilon || path.size() <= 2) {
      return path;
    }

    std::vector<Eigen::Vector3d> sampled;
    sampled.reserve(path.size());
    sampled.push_back(path.front());

    auto append_or_update = [&](const Eigen::Vector3d & point) {
      if (
        !sampled.empty() &&
        (sampled.back().head<2>() - point.head<2>()).norm() <= KinematicStateLayout::PointEpsilon)
      {
        const double previous_sign = directionSign(sampled.back(), params.reversing_enabled);
        const double point_sign = directionSign(point, params.reversing_enabled);
        if (previous_sign != point_sign) {
          sampled.push_back(point);
        } else {
          sampled.back().z() = point.z();
        }
      } else {
        sampled.push_back(point);
      }
    };

    double distance_since_keep = 0.0;
    for (size_t index = 0; index + 1 < path.size(); ++index) {
      const Eigen::Vector3d & start = path[index];
      const Eigen::Vector3d & end = path[index + 1];
      const double start_sign = directionSign(start, params.reversing_enabled);
      const double end_sign = directionSign(end, params.reversing_enabled);
      const Eigen::Vector2d delta = end.head<2>() - start.head<2>();
      const double segment_length = delta.norm();

      if (segment_length <= KinematicStateLayout::PointEpsilon) {
        if (start_sign != end_sign) {
          append_or_update(end);
          distance_since_keep = 0.0;
        }
        continue;
      }

      double traversed = 0.0;
      while (distance_since_keep + (segment_length - traversed) >= target_spacing) {
        const double step = target_spacing - distance_since_keep;
        traversed += step;
        const double ratio = std::min(1.0, std::max(0.0, traversed / segment_length));
        Eigen::Vector3d sample = start + ratio * (end - start);
        sample.z() = start.z();
        append_or_update(sample);
        distance_since_keep = 0.0;
      }

      distance_since_keep += segment_length - traversed;
      if (start_sign != end_sign) {
        append_or_update(end);
        distance_since_keep = 0.0;
      }
    }

    append_or_update(path.back());

    if (sampled.size() < 2) {
      sampled = {path.front(), path.back()};
    }

    return sampled;
  }

  static double directionSign(const Eigen::Vector3d & point, bool reversing_enabled)
  {
    if (!reversing_enabled) {
      return 1.0;
    }
    return point.z() < 0.0 ? -1.0 : 1.0;
  }

  // 单段上采样步数上限：防止极端的 path_output_spacing / path_upsampling_factor
  // 配置导致 ceil(ds/spacing) 超过 INT_MAX（整型溢出 UB）或天量内存分配。
  static constexpr int kMaxStepsPerSegment = 100000;

  static int segmentStepCount(
    double ds,
    double output_spacing,
    int fallback_upsample_factor,
    bool use_output_spacing)
  {
    if (!use_output_spacing) {
      return std::min(std::max(fallback_upsample_factor, 1), kMaxStepsPerSegment);
    }
    const double steps = std::ceil(ds / output_spacing);
    if (!(steps > 1.0)) {
      return 1;
    }
    if (steps >= static_cast<double>(kMaxStepsPerSegment)) {
      return kMaxStepsPerSegment;
    }
    return static_cast<int>(steps);
  }

  std::vector<double> & esdf_values_;
  std::shared_ptr<EsdfGrid> esdf_grid_{};
};

}  // namespace kinematic_smoother

#endif  // CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_PROBLEM_BUILDER_HPP_
