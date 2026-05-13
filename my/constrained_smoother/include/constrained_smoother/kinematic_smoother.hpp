#ifndef CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_HPP_
#define CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_HPP_

#include <vector>

#include "ceres/ceres.h"
#include "Eigen/Core"

#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/kinematic_smoother_problem_builder.hpp"
#include "constrained_smoother/options.hpp"
#include "constrained_smoother/smoother_base.hpp"
#include "constrained_smoother/smoother_request.hpp"
#include "constrained_smoother/smoother_run_base.hpp"
#include "constrained_smoother/smoother_validator.hpp"

namespace constrained_smoother
{

/**
 * @class constrained_smoother::KinematicSmoother
 * @brief 基于简化运动学状态的路径平滑器。
 *
 * 与几何版 smoother 不同，这个版本把每个状态显式表示为
 * (x, y, theta, kappa, ds)，并通过相邻状态之间的运动学过渡残差来约束
 * 路径演化。方向切换处会显式插入 cusp 段，并在求解后进行硬性后验校验。
 *
 * 当前实现同样分成三层：
 * - `KinematicSmoother` 持有跨多次调用复用的长期状态，例如 ESDF 缓存、
 *   validator 和最近一次优化状态数。
 * - 内部 `Run` 表示单次 `smooth()` 调用的生命周期。
 * - `KinematicSmootherProblemBuilder` 负责 ESDF 准备、状态展开、问题拼接和
 *   输出解包。
 */
class KinematicSmoother : public SolverBackedSmootherBase
{
public:
  KinematicSmoother() = default;
  ~KinematicSmoother() = default;

  /**
   * @brief 初始化后续求解共用的 Ceres 配置。
   */
  void initialize(const OptimizerParams & params)
  {
    initializeOptimizer(params);
  }

  /// 返回最近一次运动学优化中参与求解的状态数量。
  size_t getLastOptimizedKnotCount() const
  {
    return last_optimized_knot_count_;
  }

  /**
   * @brief 使用内部生成的 ESDF 对路径做运动学平滑。
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
  };

private:
  using ProcessedPath = KinematicProcessedPath;

  /// 运动学版 smoother 的单次执行对象。
  ///
  /// 它持有一次优化的展开状态、变量数组和问题对象，并把 prepare / solve /
  /// finalize 三段生命周期与顶层长期状态分离开。
  class Run : public SmootherRunBase<Run, KinematicSmoother, SmootherRequest>
  {
  public:
    Run(KinematicSmoother & smoother, const SmootherRequest & request)
    : SmootherRunBase<Run, KinematicSmoother, SmootherRequest>(smoother, request)
    {
    }

    /// 第 1 阶段：准备 ESDF、展开状态链，并构建运动学优化问题。
    void prepare()
    {
      auto builder = this->owner().makeProblemBuilder();
      this->owner().validateCommonInputs(
        this->request().path,
        this->request().costmap,
        this->request().params,
        "Kinematic smoother");
      this->owner().setMaxSolverTime(this->request().params.max_time);
      builder.initializeEsdfValues(
        this->request().costmap, this->request().params, this->request().precomputed_esdf);
      processed_ = KinematicSmootherProblemBuilder::buildProcessedPath(
        this->request().path,
        this->request().start_dir,
        this->request().end_dir,
        this->request().params,
        this->request().costmap);
      variables_ = processed_.initial_variables;
      builder.buildProblem(
        processed_, this->request().costmap, this->request().params, variables_, problem_);
      KinematicSmootherProblemBuilder::applyBounds(
        problem_,
        variables_.data(),
        processed_.state_count,
        this->request().params.max_curvature);
      this->owner().last_optimized_knot_count_ = processed_.state_count;
    }

    /// 第 2 阶段：调用共享求解器执行运动学状态优化。
    bool solve() const
    {
      return this->owner().solvePreparedProblem(
        problem_, "Kinematic smoother", this->request().failure);
    }

    /// 第 3 阶段：执行硬性后验校验，并把状态链回写成公共路径表示。
    bool finalize()
    {
      this->request().path =
        KinematicSmootherProblemBuilder::unpackPath(variables_, processed_.state_count);

      if (!this->owner().validator_.validateKinematicSolution(
          {
            variables_,
            processed_.reference_points,
            processed_.gears,
            processed_.is_cusp_segment,
            processed_.state_count,
            processed_.start_theta,
            processed_.end_theta,
            this->request().costmap,
            this->request().params,
            this->owner().esdf_values_,
          }, this->request().failure))
      {
        return false;
      }
      return true;
    }

    ProcessedPath processed_{};
    std::vector<double> variables_{};
    mutable ceres::Problem problem_{};
  };

  std::vector<double> esdf_values_{};
  SmootherValidator validator_{};
  size_t last_optimized_knot_count_{0};

  /// 返回绑定到当前 KinematicSmoother 长期状态的运动学问题构建器。
  ///
  /// 这让 Run 只依赖一个稳定的构建入口，而不直接操作长期 ESDF 存储细节。
  KinematicSmootherProblemBuilder makeProblemBuilder()
  {
    return KinematicSmootherProblemBuilder(esdf_values_);
  }
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_HPP_