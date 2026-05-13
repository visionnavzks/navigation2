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

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_PROBLEM_BUILDER_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_PROBLEM_BUILDER_HPP_

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "ceres/ceres.h"
#include "Eigen/Core"

#include "constrained_smoother/esdf.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/options.hpp"
#include "constrained_smoother/smoother_cost_function.hpp"
#include "constrained_smoother/utils.hpp"

namespace constrained_smoother
{

/// 几何版 smoother 的问题构建器。
///
/// 负责 ESDF 准备、局部残差连接、cusp 邻域障碍物重赋权与边界冻结。
/// 详细的遍历与重赋权流程说明放在 docs/SMOOTHER_DESIGN.md。
class SmootherProblemBuilder
{
public:
  using EsdfInterpolator = ceres::BiCubicInterpolator<ceres::Grid2D<double>>;

  SmootherProblemBuilder(
    std::vector<double> & esdf_values,
    std::shared_ptr<ceres::Grid2D<double>> & esdf_grid)
  : esdf_values_(esdf_values), esdf_grid_(esdf_grid)
  {
  }

  bool buildProblem(
    const std::vector<Eigen::Vector3d> & path,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf,
    ceres::Problem & problem,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized)
  {
    // 调用方必须先准备好 path_optim / optimized；这里专注于 ESDF 和残差构建。
    auto esdf_interpolator = initializeEsdfInterpolator(costmap, params, precomputed_esdf);
    addPathResidualBlocks(path, costmap, params, esdf_interpolator, problem, path_optim, optimized);
    addGoalPositionResidualBlock(path, path_optim, params, problem);
    return finalizeOptimizationProblem(problem, path_optim, params);
  }

private:
  /// 单次遍历路径并拼接几何残差时维护的滚动窗口状态。
  struct BuildProblemState
  {
    explicit BuildProblemState(double initial_direction)
    : last_direction(initial_direction)
    {
    }

    // 最近几个被采纳控制点的索引，用于拼接三点/四点局部残差。
    int preprelast_i{-1};
    int prelast_i{-1};
    int last_i{0};

    // 最近一段路径的方向符号；异号翻转表示穿过 cusp。
    double last_direction;

    // 上一条三点残差是否跨越 cusp。
    bool last_was_cusp{false};

    // 当前局部段是否处于倒车方向。
    bool last_is_reversing{false};

    // 候选 cusp 回溯窗口：(段长, 已接入 problem 的三点残差对象)。
    std::deque<std::pair<double, SmootherCostFunction *>> potential_cusp_funcs{};

    // 上一条被采纳 segment 的长度。
    double last_segment_len{EPSILON};

    // 回溯窗口覆盖的累计弧长，用于把窗口裁剪到 cusp_half_length 以内。
    double potential_cusp_funcs_len{0.0};

    // 与最近一个 cusp 的前向弧长距离，用于 cusp 后半区的权重插值。
    double len_since_cusp{std::numeric_limits<double>::infinity()};
  };

  static double interpolateCuspZoneWeight(
    double distance_from_cusp,
    double cusp_half_length,
    const SmootherParams & params)
  {
    // 距离 cusp 越近越接近 cusp_costmap_weight_sqrt，越远越退回普通权重。
    return params.cusp_costmap_weight_sqrt * (1.0 - distance_from_cusp / cusp_half_length) +
           params.costmap_weight_sqrt * distance_from_cusp / cusp_half_length;
  }

  std::shared_ptr<EsdfInterpolator> initializeEsdfInterpolator(
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
    if (!params.obstacleTermsEnabled()) {
      esdf_values_.clear();
      esdf_grid_.reset();
      return nullptr;
    }

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

    esdf_grid_ = std::make_shared<ceres::Grid2D<double>>(
      esdf_values_.data(), 0, costmap->getSizeInCellsY(), 0, costmap->getSizeInCellsX());
    return std::make_shared<EsdfInterpolator>(*esdf_grid_);
  }

  void addPathResidualBlocks(
    const std::vector<Eigen::Vector3d> & path,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::shared_ptr<EsdfInterpolator> & esdf_interpolator,
    ceres::Problem & problem,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized) const
  {
    const double cusp_half_length = params.cusp_zone_length / 2;
    ceres::LossFunction * loss_function = nullptr;
    BuildProblemState state(path_optim[0][2]);

    for (size_t i = 1; i < path_optim.size(); i++) {
      auto & pt = path_optim[i];

      bool is_cusp = false;
      if (i != path_optim.size() - 1) {
        // 方向符号翻转表示这里穿过 cusp。
        is_cusp = pt[2] * state.last_direction < 0;
        state.last_direction = pt[2];

        if (!is_cusp &&
          i > (params.keep_start_orientation ? 1 : 0) &&
          i < path_optim.size() - (params.keep_goal_orientation ? 2 : 1) &&
          static_cast<int>(i - state.last_i) < params.path_downsampling_factor)
        {
          // 普通点允许按 downsampling factor 跳过；cusp 必须保留。
          continue;
        }
      }

      // 当前采纳 segment 的长度，同时用于几何比例和 cusp 邻域计数。
      double current_segment_len =
        (path_optim[i] - path_optim[state.last_i]).block<2, 1>(0, 0).norm();

      // 把当前段长计入候选 cusp 回溯窗口的累计长度。
      state.potential_cusp_funcs_len += current_segment_len;

      // 只保留 cusp 前半区可能用到的最近残差。
      while (!state.potential_cusp_funcs.empty() &&
        state.potential_cusp_funcs_len > cusp_half_length)
      {
        state.potential_cusp_funcs_len -= state.potential_cusp_funcs.front().first;
        state.potential_cusp_funcs.pop_front();
      }

      if (is_cusp) {
        // 回溯修改 cusp 前半区的 obstacle 权重。
        double len_to_cusp = current_segment_len;
        for (int i_cusp = state.potential_cusp_funcs.size() - 1; i_cusp >= 0; i_cusp--) {
          auto & f = state.potential_cusp_funcs[i_cusp];
          double new_weight = interpolateCuspZoneWeight(len_to_cusp, cusp_half_length, params);

          // 多个 cusp zone 重叠时保留更强的那次提权。
          if (std::abs(new_weight - params.cusp_costmap_weight_sqrt) <
            std::abs(f.second->getCostmapWeightSqrt() - params.cusp_costmap_weight_sqrt))
          {
            f.second->setCostmapWeightSqrt(new_weight);
          }

          len_to_cusp += f.first;
        }

        // 清空回溯窗口，并开始记录 cusp 后半区的前向距离。
        state.potential_cusp_funcs_len = 0;
        state.potential_cusp_funcs.clear();
        state.len_since_cusp = 0;
      }

      optimized[i] = true;
      if (state.prelast_i != -1) {
        double costmap_weight_sqrt = params.costmap_weight_sqrt;

        // cusp 后半区内的新残差使用插值后的 obstacle 权重。
        if (state.len_since_cusp <= cusp_half_length) {
          costmap_weight_sqrt =
            interpolateCuspZoneWeight(state.len_since_cusp, cusp_half_length, params);
        }

        // 三点主 residual：smooth / curvature / distance / obstacle 打包在一起。
        SmootherCostFunction * cost_function = new SmootherCostFunction(
          path[state.last_i].template block<2, 1>(0, 0),
          (state.last_was_cusp ? -1 : 1) * state.last_segment_len / current_segment_len,
          state.last_is_reversing,
          costmap,
          esdf_interpolator,
          params,
          costmap_weight_sqrt);
        problem.AddResidualBlock(
          cost_function->AutoDiff(), loss_function,
          path_optim[state.last_i].data(), pt.data(), path_optim[state.prelast_i].data());

        // 记入候选回溯窗口，供未来 cusp 可能回头提权。
        state.potential_cusp_funcs.emplace_back(current_segment_len, cost_function);

        // 可选的四点 curvature-rate 附加约束。
        if (params.curvature_rate_weight_sqrt > 0.0 &&
          state.preprelast_i != -1 &&
          path_optim[state.preprelast_i][2] * path_optim[state.prelast_i][2] > 0.0 &&
          path_optim[state.prelast_i][2] * path_optim[state.last_i][2] > 0.0 &&
          path_optim[state.last_i][2] * pt[2] > 0.0)
        {
          CurvatureRateCostFunction * curvature_rate_cost_function =
            new CurvatureRateCostFunction(params.curvature_rate_weight_sqrt);
          problem.AddResidualBlock(
            curvature_rate_cost_function->AutoDiff(), loss_function,
            path_optim[state.preprelast_i].data(), path_optim[state.prelast_i].data(),
            path_optim[state.last_i].data(), pt.data());
        }
      }

      // 更新滚动窗口状态，供下一次迭代使用。
      state.last_was_cusp = is_cusp;
      state.last_is_reversing = state.last_direction < 0;
      state.preprelast_i = state.prelast_i;
      state.prelast_i = state.last_i;
      state.last_i = i;

      // 沿路径前向累加与最近 cusp 的距离。
      state.len_since_cusp += current_segment_len;
      state.last_segment_len = std::max(EPSILON, current_segment_len);
    }
  }

  static bool finalizeOptimizationProblem(
    ceres::Problem & problem,
    const std::vector<Eigen::Vector3d> & path_optim,
    const SmootherParams & params)
  {
    const bool goal_position_fixed =
      params.goal_longitudinal_tolerance <= 1e-9 && params.goal_lateral_tolerance <= 1e-9;

    int posesToOptimize = problem.NumParameterBlocks() - 1;
    if (params.keep_goal_orientation) {
      posesToOptimize -= 1;
    }
    if (params.keep_start_orientation) {
      posesToOptimize -= 1;
    }
    if (goal_position_fixed) {
      posesToOptimize -= 1;
    }
    if (posesToOptimize <= 0) {
      return false;
    }

    problem.SetParameterBlockConstant(path_optim.front().data());
    if (params.keep_start_orientation) {
      problem.SetParameterBlockConstant(path_optim[1].data());
    }
    if (params.keep_goal_orientation) {
      problem.SetParameterBlockConstant(path_optim[path_optim.size() - 2].data());
    }
    if (goal_position_fixed) {
      problem.SetParameterBlockConstant(path_optim.back().data());
    }
    return true;
  }

  static void addGoalPositionResidualBlock(
    const std::vector<Eigen::Vector3d> & path,
    std::vector<Eigen::Vector3d> & path_optim,
    const SmootherParams & params,
    ceres::Problem & problem)
  {
    if (
      path_optim.size() < 2 ||
      (params.goal_longitudinal_tolerance <= 1e-9 && params.goal_lateral_tolerance <= 1e-9))
    {
      return;
    }

    Eigen::Vector2d goal_direction =
      (path_optim.back() - path_optim[path_optim.size() - 2]).template block<2, 1>(0, 0);
    if (goal_direction.norm() <= EPSILON && path.size() >= 2) {
      goal_direction = (path.back() - path[path.size() - 2]).template block<2, 1>(0, 0);
    }
    if (goal_direction.norm() <= EPSILON) {
      goal_direction = Eigen::Vector2d(1.0, 0.0);
    }

    auto * goal_position_cost = new GoalPositionCostFunction(
      path.back().template block<2, 1>(0, 0),
      goal_direction,
      params.goal_longitudinal_tolerance,
      params.goal_lateral_tolerance,
      100.0);
    problem.AddResidualBlock(goal_position_cost->AutoDiff(), nullptr, path_optim.back().data());
  }

  std::vector<double> & esdf_values_;
  std::shared_ptr<ceres::Grid2D<double>> & esdf_grid_;
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_PROBLEM_BUILDER_HPP_