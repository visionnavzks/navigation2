#ifndef KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_HPP_
#define KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_HPP_

#include <string>
#include <vector>

#include "ceres/ceres.h"

#include "kinematic_path_smoother/exceptions.hpp"
#include "kinematic_path_smoother/kinematic_smoother_problem_builder.hpp"
#include "kinematic_path_smoother/options.hpp"
#include "kinematic_path_smoother/smoother_request.hpp"
#include "kinematic_path_smoother/smoother_validator.hpp"
#include "kinematic_path_smoother/solver_utils.hpp"

namespace kinematic_path_smoother
{

/// 基于显式运动学状态的路径平滑器。
///
/// 每个优化 knot 表示为 (x, y, theta, kappa, ds)。相邻 knot 通过自行车式
/// 曲率积分模型连接，倒车段通过 gear 符号处理，前进/倒车切换处显式插入
/// cusp 保持段。输入 path 的 z 是方向符号，输出路径 z 是 yaw。
class KinematicPathSmoother
{
public:
  /// 初始化可复用的 Ceres 求解器配置。
  ///
  /// max_time 属于单次请求参数，因此会在 smooth() 内动态覆盖。
  void initialize(const OptimizerParams & params)
  {
    debug_ = params.debug;
    solver_options_ = ceres::Solver::Options{};
    solver_options_.linear_solver_type =
      params.linear_solver == OptimizerParams::LinearSolver::DenseQr ?
      ceres::DENSE_QR : ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options_.max_num_iterations = params.max_iterations;
    solver_options_.function_tolerance = params.function_tolerance;
    solver_options_.gradient_tolerance = params.gradient_tolerance;
    solver_options_.parameter_tolerance = params.parameter_tolerance;
    solver_options_.minimizer_progress_to_stdout = debug_;
    solver_options_.logging_type = debug_ ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
  }

  /// 执行一次完整平滑。
  ///
  /// 失败语义：
  /// - request.failure 非空：写入失败原因并返回 success=false。
  /// - request.failure 为空：输入错误或求解/校验失败直接抛异常。
  [[nodiscard]] SmoothingResult smooth(const SmoothingRequest & request)
  {
    if (request.path.size() < 2) {
      throw InvalidPath("KinematicPathSmoother requires at least two input points");
    }
    if (request.params.obstacleTermsEnabled() && request.costmap == nullptr) {
      throw InvalidCostmap("KinematicPathSmoother obstacle terms require a costmap");
    }

    SmoothingResult result;
    solver_options_.max_solver_time_in_seconds = request.params.max_time;

    // ESDF 由 builder 持有插值器视图，底层数组保存在 smoother 实例中以复用内存。
    KinematicSmootherProblemBuilder builder(esdf_);
    builder.prepareEsdf(request.costmap, request.params, request.precomputed_esdf);

    // 将公共路径展开成求解器状态链，并保留 gear/cusp 元数据用于建模和校验。
    ProcessedPath processed = KinematicSmootherProblemBuilder::buildProcessedPath(
      request.path,
      request.start_direction,
      request.goal_direction,
      request.params,
      request.costmap);
    std::vector<double> variables = processed.variables;

    // Ceres 问题在本次调用内构建，不跨请求复用，避免残差和参数块生命周期复杂化。
    ceres::Problem problem;
    builder.addResiduals(processed, request.costmap, request.params, variables, problem);
    KinematicSmootherProblemBuilder::applyBounds(problem, variables, processed, request.params);

    result.optimized_knot_count = processed.size;
    result.target_spacing = processed.target_spacing;

    if (!solveOrReport(problem, solver_options_, debug_, request.failure)) {
      return result;
    }

    // 先保留候选解，后验校验失败时调用方仍可用于诊断。
    result.optimized_path = KinematicSmootherProblemBuilder::unpack(variables, processed.size);
    const bool valid = validator_.validate(
      {variables, processed, request.costmap, request.params, esdf_},
      request.failure);
    if (!valid) {
      return result;
    }

    result.path = KinematicSmootherProblemBuilder::upsample(variables, processed, request.params);
    result.success = true;
    return result;
  }

private:
  bool debug_{false};
  ceres::Solver::Options solver_options_{};
  std::vector<double> esdf_{};
  SmootherValidator validator_{};
};

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__KINEMATIC_SMOOTHER_HPP_
