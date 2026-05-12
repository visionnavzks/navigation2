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
/// 它负责把 ESDF 准备、三点/四点残差连接、cusp 邻域障碍物重赋权和边界冻结
/// 聚合成一个可复用的构建阶段，而不是把这些步骤继续留在顶层 smoother 里。
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

    int preprelast_i{-1};
    int prelast_i{-1};
    int last_i{0};
    double last_direction;
    bool last_was_cusp{false};
    bool last_is_reversing{false};
    // 最近创建出的几何残差块，按路径遍历顺序保存为
    // (该残差对应的段长, 残差对象指针)。
    // 一旦后续检测到 cusp，会回头只修改这批“离 cusp 足够近”的旧残差块，
    // 把它们的 obstacle 权重向 cusp_costmap_weight_sqrt 提升。
    std::deque<std::pair<double, SmootherCostFunction *>> potential_cusp_funcs{};
    double last_segment_len{EPSILON};
    // potential_cusp_funcs 中所有段长的累计和。
    // 它用于把队列裁剪成一个滑动窗口：窗口总弧长始终不超过 cusp_half_length。
    // 这样当遍历到 cusp 时，只会回溯修改 cusp 前半区的残差块，而不会影响更早的路径。
    double potential_cusp_funcs_len{0.0};
    // 从最近一个 cusp 往后累计的弧长。
    // 用来给 cusp 后半区新创建的残差块做前向插值重赋权。
    double len_since_cusp{std::numeric_limits<double>::infinity()};
  };

  static double interpolateCuspZoneWeight(
    double distance_from_cusp,
    double cusp_half_length,
    const SmootherParams & params)
  {
    return params.cusp_costmap_weight_sqrt * (1.0 - distance_from_cusp / cusp_half_length) +
           params.costmap_weight_sqrt * distance_from_cusp / cusp_half_length;
  }

  std::shared_ptr<EsdfInterpolator> initializeEsdfInterpolator(
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
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
        is_cusp = pt[2] * state.last_direction < 0;
        state.last_direction = pt[2];

        if (!is_cusp &&
          i > (params.keep_start_orientation ? 1 : 0) &&
          i < path_optim.size() - (params.keep_goal_orientation ? 2 : 1) &&
          static_cast<int>(i - state.last_i) < params.path_downsampling_factor)
        {
          continue;
        }
      }

      double current_segment_len =
        (path_optim[i] - path_optim[state.last_i]).block<2, 1>(0, 0).norm();

      // 先把当前段长计入“候选 cusp 回溯窗口”的累计长度。
      // 这个窗口保存的是最近创建过的 obstacle 残差块；如果稍后发现当前位置是 cusp，
      // 我们会回头把窗口里的旧残差块重新加权。
      state.potential_cusp_funcs_len += current_segment_len;

      // 维护一个长度受限的滑动窗口：只保留距离当前位置不超过 cusp_half_length
      // 的那部分旧残差块。队头是最老、离当前位置最远的残差；超出范围就弹掉。
      while (!state.potential_cusp_funcs.empty() &&
        state.potential_cusp_funcs_len > cusp_half_length)
      {
        state.potential_cusp_funcs_len -= state.potential_cusp_funcs.front().first;
        state.potential_cusp_funcs.pop_front();
      }

      if (is_cusp) {
        // 已经确认当前位置穿过了一个 cusp。
        // 现在反向遍历窗口中的旧残差块，按它们距离 cusp 的弧长做线性插值，
        // 让 cusp 前半区的 obstacle 权重从普通值逐渐过渡到 cusp 增强值。
        double len_to_cusp = current_segment_len;
        for (int i_cusp = state.potential_cusp_funcs.size() - 1; i_cusp >= 0; i_cusp--) {
          auto & f = state.potential_cusp_funcs[i_cusp];
          double new_weight = interpolateCuspZoneWeight(len_to_cusp, cusp_half_length, params);
          if (std::abs(new_weight - params.cusp_costmap_weight_sqrt) <
            std::abs(f.second->getCostmapWeightSqrt() - params.cusp_costmap_weight_sqrt))
          {
            f.second->setCostmapWeightSqrt(new_weight);
          }
          len_to_cusp += f.first;
        }

        // cusp 前半区已经处理完；清空回溯窗口，并把“距最近 cusp 的距离”置零。
        // 后续新创建的残差块会用 len_since_cusp 负责 cusp 后半区的前向重赋权。
        state.potential_cusp_funcs_len = 0;
        state.potential_cusp_funcs.clear();
        state.len_since_cusp = 0;
      }

      optimized[i] = true;
      if (state.prelast_i != -1) {
        double costmap_weight_sqrt = params.costmap_weight_sqrt;

        // 如果当前残差块处在最近一个 cusp 之后的半个作用区间内，直接按与 cusp 的
        // 弧长距离插值得到更高的 obstacle 权重；超过该范围则退回普通权重。
        if (state.len_since_cusp <= cusp_half_length) {
          costmap_weight_sqrt =
            interpolateCuspZoneWeight(state.len_since_cusp, cusp_half_length, params);
        }

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

        // 这个新残差块从现在开始进入“可能被未来某个 cusp 回溯修改”的候选集合。
        // 与之配套的 segment length 会被 potential_cusp_funcs_len 累计，用于维持窗口大小。
        state.potential_cusp_funcs.emplace_back(current_segment_len, cost_function);
      }

      state.last_was_cusp = is_cusp;
      state.last_is_reversing = state.last_direction < 0;
      state.preprelast_i = state.prelast_i;
      state.prelast_i = state.last_i;
      state.last_i = i;
      state.len_since_cusp += current_segment_len;
      state.last_segment_len = std::max(EPSILON, current_segment_len);
    }
  }

  static bool finalizeOptimizationProblem(
    ceres::Problem & problem,
    const std::vector<Eigen::Vector3d> & path_optim,
    const SmootherParams & params)
  {
    int posesToOptimize = problem.NumParameterBlocks() - 2;
    if (params.keep_goal_orientation) {
      posesToOptimize -= 1;
    }
    if (params.keep_start_orientation) {
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
    problem.SetParameterBlockConstant(path_optim.back().data());
    return true;
  }

  std::vector<double> & esdf_values_;
  std::shared_ptr<ceres::Grid2D<double>> & esdf_grid_;
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_PROBLEM_BUILDER_HPP_