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

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_HPP_

#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <queue>
#include <utility>
#include <deque>
#include <limits>
#include <algorithm>

#include "constrained_smoother/smoother_cost_function.hpp"
#include "constrained_smoother/utils.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/smoother_validator.hpp"

#include "ceres/ceres.h"
#include "Eigen/Core"

namespace constrained_smoother
{

/**
 * @class constrained_smoother::Smoother
 * @brief 基于 Ceres 的独立约束路径平滑器，不依赖 ROS。
 *
 * 独立版流程会保留原始参考路径拓扑，在下采样后的控制点子集上构建
 * 非线性最小二乘问题，求解得到更优的二维位置，然后再重建 yaw，必要时
 * 还会补回上采样后的中间姿态。障碍物距离来自 costmap 生成的 ESDF，
 * 也可以由调用方直接注入。
 *
 * 输入约定：
 * - path[i] 初始表示为 (x, y, direction_sign)，第三个分量通常用
 *   +1 表示前进，用 -1 表示倒车。
 * - start_dir 和 end_dir 表示切向方向，而不是 yaw 标量。
 *
 * 输出约定：
 * - 成功时，返回路径中每个点的第三个分量都会被改写为弧度制 yaw。
 * - getLastOptimizedKnotCount() 返回最近一次优化中实际参与求解的
 *   下采样控制点数量。
 */
class Smoother
{
public:
  Smoother() {}
  ~Smoother() {}

  /**
    * @brief 返回最近一次 smooth() 调用中被激活的控制点数量。
   *
    * 当下采样跳过了大多数内部点，或路径过短无法形成非平凡优化问题时，
    * 这个值有助于判断本次求解实际优化了多少点。
   */
  size_t getLastOptimizedKnotCount() const
  {
    return last_optimized_knot_count_;
  }

  /**
    * @brief 初始化后续求解共用的 Ceres 配置。
    * @param params 控制日志、容差以及线性求解器选择的 OptimizerParams。
   */
  void initialize(const OptimizerParams params)
  {
    // 第 1 步：缓存调试开关，后续日志策略和求解报告输出都会依赖它。
    debug_ = params.debug;

    // 第 2 步：按外部配置写入线性求解器、迭代上限和收敛容差。
    options_.linear_solver_type = params.solver_types.at(params.linear_solver_type);

    options_.max_num_iterations = params.max_iterations;

    options_.function_tolerance = params.fn_tol;
    options_.gradient_tolerance = params.gradient_tol;
    options_.parameter_tolerance = params.param_tol;

    if (debug_) {
      options_.minimizer_progress_to_stdout = true;
      options_.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
    } else {
      options_.logging_type = ceres::SILENT;
    }
  }

  /**
    * @brief 使用内部生成的 ESDF 对参考路径进行平滑。
    * @param path 原地修改的路径缓冲区。输入时保存
    * (x, y, direction_sign)；输出时第三个分量改写为 yaw。
    * @param start_dir 起点期望切向方向。
    * @param end_dir 终点期望切向方向。
    * @param costmap 用于构建障碍物残差和 ESDF 的 costmap。
    * @param params 平滑配置及各残差项权重。
    * @return 当优化器和后验校验都成功时返回 true。
   */
  bool smooth(
    std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const Costmap2D * costmap,
    const SmootherParams & params)
  {
    return smooth(path, start_dir, end_dir, costmap, params, nullptr, nullptr);
  }

  bool smooth(
    std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf,
    SmoothingFailureInfo * failure = nullptr)
  {
    // 第 1 步：先校验输入路径至少包含起点和终点。
    if (path.size() < 2) {
      throw InvalidPath("Constrained smoother: Path must have at least 2 points");
    }

    // 第 2 步：保留原始参考路径，供后续边界约束和结果校验对照使用。
    const std::vector<Eigen::Vector3d> reference_path = path;

    // 第 3 步：把本次运行的最大求解时长写入 Ceres 配置。
    options_.max_solver_time_in_seconds = params.max_time;

    ceres::Problem problem;
    std::vector<Eigen::Vector3d> path_optim;
    std::vector<bool> optimized;
    // 第 4 步：构建优化问题，包括 ESDF、端点锚定、下采样和残差连接。
    if (buildProblem(
      path, start_dir, end_dir, costmap, params, precomputed_esdf, problem, path_optim,
      optimized))
    {
      last_optimized_knot_count_ = std::count(optimized.begin(), optimized.end(), true);
      // 第 5 步：调用 Ceres 求解，并在调试模式下输出完整收敛报告。
      ceres::Solver::Summary summary;
      ceres::Solve(options_, &problem, &summary);
      if (debug_) {
        std::cout << summary.FullReport() << std::endl;
      }
      // 第 6 步：显式拒绝不可用结果，以及没有带来目标函数改进的结果。
      if (!summary.IsSolutionUsable()) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::SolverRejectedSolution,
          "Constrained smoother rejected the Ceres solution as unusable");
      }
      if (summary.initial_cost - summary.final_cost < 0.0) {
        return throwOrStoreSmoothingFailure(
          failure,
          SmoothingFailureReason::NoCostImprovement,
          "Constrained smoother did not improve the objective cost");
      }
    } else {
      // 第 5 步（退化路径）：若没有可优化自由度，直接跳过求解，但保持后续输出流程一致。
      last_optimized_knot_count_ = std::count(optimized.begin(), optimized.end(), true);
      if (debug_) {
        std::cout << "[smoother] Path too short to optimize" << std::endl;
      }
    }

    // 第 7 步：根据保留下来的控制点重建完整输出路径，并恢复 yaw。
    upsampleAndPopulate(path_optim, optimized, start_dir, end_dir, params, path);

    // 第 8 步：统一执行后验校验，确认边界、曲率和净空约束真正成立。
    if (!validator_.validateSmoothedPath(
        {path, reference_path, start_dir, end_dir, costmap, params, esdf_values_}, failure))
    {
      return false;
    }

    return true;
  }

private:
  using EsdfInterpolator = ceres::BiCubicInterpolator<ceres::Grid2D<double>>;

  /// 单次遍历路径并构建残差时维护的滚动状态。
  struct BuildProblemState
  {
    explicit BuildProblemState(double initial_direction)
    : last_direction(initial_direction)
    {
    }

    int preprelast_i{-1};
    int prelast_i{-1};
    int last_i{0};
    /// 最近一次接受的控制点对应的方向符号。
    double last_direction;
    /// 上一个被接受的段是否以 cusp 结束。
    bool last_was_cusp{false};
    /// 当前开放段缓存下来的运动方向。
    bool last_is_reversing{false};
    /// 近期障碍物残差；若出现 cusp，它们可能被重新赋权。
    std::deque<std::pair<double, SmootherCostFunction *>> potential_cusp_funcs{};
    /// 最近一次接受的路径段长度。
    double last_segment_len{EPSILON};
    /// potential_cusp_funcs 中累计保存的弧长。
    double potential_cusp_funcs_len{0.0};
    /// 自最近一个 cusp 起累计的弧长。
    double len_since_cusp{std::numeric_limits<double>::infinity()};
  };

  /// 在常规障碍物权重与 cusp 强化权重之间做线性插值。
  static double interpolateCuspZoneWeight(
    double distance_from_cusp,
    double cusp_half_length,
    const SmootherParams & params)
  {
    return params.cusp_costmap_weight_sqrt * (1.0 - distance_from_cusp / cusp_half_length) +
           params.costmap_weight_sqrt * distance_from_cusp / cusp_half_length;
  }

  /**
    * @brief 构建 ESDF 存储和障碍物代价使用的插值器。
   *
    * 当提供 precomputed_esdf 时，它的大小必须与 costmap 维度完全一致，
    * 因为后续后验校验会复用这份扁平化存储。
   */
  std::shared_ptr<EsdfInterpolator> initializeEsdfInterpolator(
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
    // 构建或校验障碍物残差背后的 ESDF 数据。
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

  /// 复制参考路径，并应用可选的起终点切向锚定。
  void initializeOptimizationPath(
    const std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized) const
  {
    // 第 1 步：复制参考路径，保证原始输入不会在问题构建阶段被提前改写。
    path_optim = path;
    // 第 2 步：在工作副本上施加起终点朝向锚定。
    applyEndpointOrientationAnchors(path_optim, start_dir, end_dir, params);
    // 第 3 步：初始化优化标记数组，记录哪些点真正进入主优化链路。
    optimized = std::vector<bool>(path.size(), false);
    optimized[0] = true;
  }

  /**
    * @brief 在下采样和检测 cusp 的同时连接所有残差块。
   *
    * 整条路径只遍历一次。内部点可能依据下采样因子被跳过，但 cusp 必须
    * 被保留下来，这样重建出来的 yaw 序列才能反映真实的换向信息。
   */
  void addPathResidualBlocks(
    const std::vector<Eigen::Vector3d> & path,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::shared_ptr<EsdfInterpolator> & esdf_interpolator,
    ceres::Problem & problem,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized) const
  {
    // 第 1 步：单次遍历路径，同时完成下采样、cusp 检测和残差连接。
    const double cusp_half_length = params.cusp_zone_length / 2;
    ceres::LossFunction * loss_function = nullptr;
    BuildProblemState state(path_optim[0][2]);

    for (size_t i = 1; i < path_optim.size(); i++) {
      auto & pt = path_optim[i];

      // 第 1.1 步：先识别当前点是否触发方向翻转；cusp 点必须保留。
      bool is_cusp = false;
      if (i != path_optim.size() - 1) {
        is_cusp = pt[2] * state.last_direction < 0;
        state.last_direction = pt[2];

        // 第 1.2 步：仅对普通内部点应用下采样跳过逻辑。
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

      // 第 2 步：维护一个“可被未来 cusp 回溯重赋权”的近期残差窗口。
      state.potential_cusp_funcs_len += current_segment_len;
      while (!state.potential_cusp_funcs.empty() &&
        state.potential_cusp_funcs_len > cusp_half_length)
      {
        state.potential_cusp_funcs_len -= state.potential_cusp_funcs.front().first;
        state.potential_cusp_funcs.pop_front();
      }

      // 第 3 步：若检测到 cusp，就回溯提高附近历史残差的障碍物权重。
      if (is_cusp) {
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
        state.potential_cusp_funcs_len = 0;
        state.potential_cusp_funcs.clear();
        // cusp 之后重新开始累计前向弧长，用于前半段权重渐变。
        state.len_since_cusp = 0;
      }

      // 第 4 步：一旦前驱点足够，就连接主三点残差，并按需连接四点残差。
      optimized[i] = true;
      if (state.prelast_i != -1) {
        double costmap_weight_sqrt = params.costmap_weight_sqrt;
        if (state.len_since_cusp <= cusp_half_length) {
          // 第 4.1 步：处于 cusp 邻域时，继续沿弧长平滑过渡障碍物权重。
          costmap_weight_sqrt =
            interpolateCuspZoneWeight(state.len_since_cusp, cusp_half_length, params);
        }

        // 第 4.2 步：主代价函数统一编码平滑、贴合、曲率和净空约束。
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
          // 第 4.3 步：四点曲率变化率残差不能跨越 cusp，只能用于方向一致的连续段。
          CurvatureRateCostFunction * curvature_rate_cost_function =
            new CurvatureRateCostFunction(params.curvature_rate_weight_sqrt);
          problem.AddResidualBlock(
            curvature_rate_cost_function->AutoDiff(), loss_function,
            path_optim[state.preprelast_i].data(), path_optim[state.prelast_i].data(),
            path_optim[state.last_i].data(), pt.data());
        }

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

  /// 冻结端点锚点，并报告是否仍存在非平凡优化问题。
  bool finalizeOptimizationProblem(
    ceres::Problem & problem,
    const std::vector<Eigen::Vector3d> & path_optim,
    const SmootherParams & params) const
  {
    // 第 1 步：先统计剩余内部自由度，判断是否还存在非平凡优化问题。
    int posesToOptimize = problem.NumParameterBlocks() - 2;  // minus start and goal
    if (params.keep_goal_orientation) {
      posesToOptimize -= 1;
    }
    if (params.keep_start_orientation) {
      posesToOptimize -= 1;
    }
    if (posesToOptimize <= 0) {
      return false;  // 没有任何点需要优化
    }

    // 第 2 步：在残差全部连接完成后，再统一冻结边界相关参数块。
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

  /**
    * @brief 执行完整的问题构建流水线。
    * @return 当至少还存在一个内部自由度可供求解时返回 true。
   */
  bool buildProblem(
    const std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf,
    ceres::Problem & problem,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized)
  {
    // 第 1 步：准备 ESDF 插值器。
    auto esdf_interpolator = initializeEsdfInterpolator(costmap, params, precomputed_esdf);
    // 第 2 步：复制路径并施加端点锚定。
    initializeOptimizationPath(path, start_dir, end_dir, params, path_optim, optimized);
    // 第 3 步：遍历路径并连接全部局部残差。
    addPathResidualBlocks(path, costmap, params, esdf_interpolator, problem, path_optim, optimized);
    // 第 4 步：冻结边界参数，并判断是否存在可求解自由度。
    return finalizeOptimizationProblem(problem, path_optim, params);
  }

  /**
    * @brief 重新放置第二个点和或倒数第二个点，以满足端点朝向约束。
   *
    * 起点和终点位置保持固定。当路径只有三个姿态且同时启用起终点朝向约束时，
    * 中间那个共享点会在两个锚定建议之间做折中。
   */
  void applyEndpointOrientationAnchors(
    std::vector<Eigen::Vector3d> & path_optim,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params) const
  {
    if (path_optim.size() < 3) {
      // 第 1 步：少于三个点时没有中间点可调整，直接返回。
      return;
    }

    if (params.keep_start_orientation) {
      // 第 2 步：保持首段长度不变，只调整第二个点来满足起点朝向。
      const double start_segment_len =
        (path_optim[1] - path_optim[0]).template block<2, 1>(0, 0).norm();
      path_optim[1].template block<2, 1>(0, 0) =
        path_optim[0].template block<2, 1>(0, 0) +
        SmootherValidator::normalizedDirection(start_dir) * start_segment_len;
    }

    if (params.keep_goal_orientation) {
      // 第 3 步：对称地保持末段长度不变，只调整倒数第二个点满足终点朝向。
      const size_t goal_index = path_optim.size() - 1;
      const size_t pregoal_index = goal_index - 1;
      const double goal_segment_len =
        (path_optim[goal_index] - path_optim[pregoal_index]).template block<2, 1>(0, 0).norm();
      Eigen::Vector2d anchored_pregoal =
        path_optim[goal_index].template block<2, 1>(0, 0) -
        SmootherValidator::normalizedDirection(end_dir) * goal_segment_len;

      if (params.keep_start_orientation && pregoal_index == 1) {
        // 第 3.1 步：若三点路径同时锚定两端，中间点只能在两侧建议之间折中。
        path_optim[pregoal_index].template block<2, 1>(0, 0) =
          0.5 * (path_optim[pregoal_index].template block<2, 1>(0, 0) + anchored_pregoal);
      } else {
        path_optim[pregoal_index].template block<2, 1>(0, 0) = anchored_pregoal;
      }
    }
  }

  /**
    * @brief 根据优化后的控制点重建最终对外输出路径。
   *
    * 重建得到的 yaw 来自局部切向方向；若开启上采样，则会使用三次 Bezier
    * 插值补回下采样阶段跳过的姿态点。
   */
  void upsampleAndPopulate(
    const std::vector<Eigen::Vector3d> & path_optim,
    const std::vector<bool> & optimized,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params,
    std::vector<Eigen::Vector3d> & path)
  {
    // 第 1 步：清空输出，并按上采样倍率预留容量。
    path.clear();
    if (params.path_upsampling_factor > 1) {
      path.reserve(params.path_upsampling_factor * (path_optim.size() - 1) + 1);
    }
    int last_i = 0;
    int prelast_i = -1;
    Eigen::Vector2d prelast_dir = {0, 0};
    for (int i = 1; i <= static_cast<int>(path_optim.size()); i++) {
      if (i == static_cast<int>(path_optim.size()) || optimized[i]) {
        if (prelast_i != -1) {
          Eigen::Vector2d last_dir;
          auto & prelast = path_optim[prelast_i];
          auto & last = path_optim[last_i];

          // 第 2 步：先恢复当前关键点的切向方向，再把它转成 yaw。
          if (i < static_cast<int>(path_optim.size())) {
            auto & current = path_optim[i];
            Eigen::Vector2d tangent_dir_val = tangentDir<double>(
              prelast.block<2, 1>(0, 0),
              last.block<2, 1>(0, 0),
              current.block<2, 1>(0, 0),
              prelast[2] * last[2] < 0);

            last_dir =
              tangent_dir_val.dot((current - last).block<2, 1>(0, 0) * last[2]) >= 0 ?
              tangent_dir_val :
              -tangent_dir_val;
            last_dir.normalize();
          } else if (params.keep_goal_orientation) {
            last_dir = end_dir;
          } else {
            last_dir = (last - prelast).block<2, 1>(0, 0) * last[2];
            last_dir.normalize();
          }
          double last_angle = atan2(last_dir[1], last_dir[0]);

          // 第 3 步：若中间有被下采样跳过的点，则用 Bezier 曲线补点。
          int interp_cnt = (last_i - prelast_i) * params.path_upsampling_factor - 1;
          if (interp_cnt > 0) {
            Eigen::Vector2d last_pt = last.block<2, 1>(0, 0);
            Eigen::Vector2d prelast_pt = prelast.block<2, 1>(0, 0);
            double dist = (last_pt - prelast_pt).norm();
            Eigen::Vector2d pt1 = prelast_pt + prelast_dir * dist * 0.4 * prelast[2];
            Eigen::Vector2d pt2 = last_pt - last_dir * dist * 0.4 * prelast[2];
            for (int j = 1; j <= interp_cnt; j++) {
              double interp = j / static_cast<double>(interp_cnt + 1);
              Eigen::Vector2d pt = cubicBezier(prelast_pt, pt1, pt2, last_pt, interp);
              path.emplace_back(pt[0], pt[1], 0.0);
            }
          }
          path.emplace_back(last[0], last[1], last_angle);

          // 第 4 步：对补出来的中间点逐个恢复 yaw。
          for (size_t j = path.size() - 1 - interp_cnt; j < path.size() - 1; j++) {
            Eigen::Vector2d tangent_dir_val = tangentDir<double>(
              path[j - 1].block<2, 1>(0, 0),
              path[j].block<2, 1>(0, 0),
              path[j + 1].block<2, 1>(0, 0),
              false);
            tangent_dir_val =
              tangent_dir_val.dot(
              (path[j + 1] - path[j]).block<2, 1>(0, 0) * prelast[2]) >= 0 ?
              tangent_dir_val :
              -tangent_dir_val;
            path[j][2] = atan2(tangent_dir_val[1], tangent_dir_val[0]);
          }

          prelast_dir = last_dir;
        } else {  // 第 2 步（起点分支）：先为重建链路初始化第一个切向方向。
          auto & start = path_optim[0];
          Eigen::Vector2d dir = params.keep_start_orientation ?
            start_dir :
            ((path_optim[i] - start).block<2, 1>(0, 0) * start[2]).normalized();
          path.emplace_back(start[0], start[1], atan2(dir[1], dir[0]));
          prelast_dir = dir;
        }
        prelast_i = last_i;
        last_i = i;
      }
    }
  }

  /**
    * @brief 计算用于路径上采样的三次 Bezier 曲线点。
    * @param mu 位于 [0, 1] 区间内的插值参数。
   */
  static Eigen::Vector2d cubicBezier(
    Eigen::Vector2d & pt0, Eigen::Vector2d & pt1,
    Eigen::Vector2d & pt2, Eigen::Vector2d & pt3, double mu)
  {
    Eigen::Vector2d a, b, c, pt;

    // 第 1 步：先把 Bezier 曲线改写成三次多项式系数。
    c[0] = 3 * (pt1[0] - pt0[0]);
    c[1] = 3 * (pt1[1] - pt0[1]);
    b[0] = 3 * (pt2[0] - pt1[0]) - c[0];
    b[1] = 3 * (pt2[1] - pt1[1]) - c[1];
    a[0] = pt3[0] - pt0[0] - c[0] - b[0];
    a[1] = pt3[1] - pt0[1] - c[1] - b[1];

    // 第 2 步：把 mu 代入多项式，得到插值点。
    pt[0] = a[0] * mu * mu * mu + b[0] * mu * mu + c[0] * mu + pt0[0];
    pt[1] = a[1] * mu * mu * mu + b[1] * mu * mu + c[1] * mu + pt0[1];

    return pt;
  }

  /// 为开发和调试开启详细 Ceres 日志。
  bool debug_;
  /// 由 initialize() 填充并复用的 Ceres 求解配置。
  ceres::Solver::Options options_;
  /// 残差和后验校验共用的扁平化 ESDF 存储。
  std::vector<double> esdf_values_;
  /// 提供给双三次插值器使用的 esdf_values_ 网格视图。
  std::shared_ptr<ceres::Grid2D<double>> esdf_grid_;
  /// 集中执行边界、曲率和净空约束的求解后校验器。
  SmootherValidator validator_{};
  /// 最近一次求解尝试中保持激活状态的控制点数量。
  size_t last_optimized_knot_count_{0};
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_HPP_
