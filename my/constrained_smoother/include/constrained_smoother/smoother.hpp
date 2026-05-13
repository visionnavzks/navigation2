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

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "constrained_smoother/smoother_path_ops.hpp"
#include "constrained_smoother/smoother_problem_builder.hpp"
#include "constrained_smoother/smoother_request.hpp"
#include "constrained_smoother/smoother_run_base.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/smoother_base.hpp"
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
 *
 * 当前实现分成三层：
 * - `Smoother` 持有跨多次调用复用的长期状态，例如 ESDF 缓存、validator 和
 *   最近一次优化点数。
 * - 内部 `Run` 表示单次 `smooth()` 调用的生命周期。
 * - `SmootherPathOps` 与 `SmootherProblemBuilder` 分别负责路径重建和问题构建。
 */
class Smoother : public SolverBackedSmootherBase
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
    initializeOptimizer(params);
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
    const SmootherRequest request{path, start_dir, end_dir, costmap, params, precomputed_esdf, failure};
    return Run(*this, request).execute();
  }

private:

  /// 几何版 smoother 的单次执行对象。
  ///
  /// 顶层 Smoother 只保留长期状态；一次 smooth() 的参考路径快照、工作路径、
  /// 问题对象和求解流程都收口在这里。
  class Run : public SmootherRunBase<Run, Smoother, SmootherRequest>
  {
  public:
    Run(Smoother & smoother, const SmootherRequest & request)
    : SmootherRunBase<Run, Smoother, SmootherRequest>(smoother, request)
    {
    }

    /// 第 1 阶段：校验请求、准备工作路径，并构建待求解的问题。
    void prepare()
    {
      this->owner().validateCommonInputs(
        this->request().path,
        this->request().costmap,
        this->request().params,
        "Constrained smoother");
      reference_path_ = this->request().path;
      path_ops_ = std::make_unique<SmootherPathOps>(
        this->request().start_dir, this->request().end_dir, this->request().params);
      path_ops_->initializeOptimizationPath(this->request().path, path_optim_, optimized_);
      this->owner().setMaxSolverTime(this->request().params.max_time);
      auto builder = this->owner().makeProblemBuilder();
      has_optimizable_problem_ = builder.buildProblem(
        this->request().path,
        this->request().costmap,
        this->request().params,
        this->request().precomputed_esdf,
        problem_,
        path_optim_,
        optimized_);
      this->owner().last_optimized_knot_count_ =
        std::count(optimized_.begin(), optimized_.end(), true);
    }

    /// 第 2 阶段：若仍存在自由度，则调用共享求解器执行优化。
    bool solve() const
    {
      if (!has_optimizable_problem_) {
        if (this->owner().isDebugEnabled()) {
          std::cout << "[smoother] Path too short to optimize" << std::endl;
        }
        return true;
      }

      return this->owner().solvePreparedProblem(
        problem_, "Constrained smoother", this->request().failure);
    }

    /// 第 3 阶段：把优化结果重建成对外路径，并执行统一后验校验。
    bool finalize()
    {
      path_ops_->populateOutput(path_optim_, optimized_, this->request().path);
      return this->owner().validator_.validateSmoothedPath(
        {
          this->request().path,
          reference_path_,
          this->request().start_dir,
          this->request().end_dir,
          this->request().costmap,
          this->request().params,
          this->owner().esdf_values_,
        },
        this->request().failure);
    }

    std::unique_ptr<SmootherPathOps> path_ops_{};
    std::vector<Eigen::Vector3d> reference_path_{};
    std::vector<Eigen::Vector3d> path_optim_{};
    std::vector<bool> optimized_{};
    mutable ceres::Problem problem_{};
    bool has_optimizable_problem_{false};
  };

  /// 返回绑定到当前 Smoother 长期状态的几何问题构建器。
  ///
  /// 这让 Run 不需要知道 ESDF 缓存和网格视图具体由哪些成员支撑。
  SmootherProblemBuilder makeProblemBuilder()
  {
    return SmootherProblemBuilder(esdf_values_, esdf_grid_);
  }

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
