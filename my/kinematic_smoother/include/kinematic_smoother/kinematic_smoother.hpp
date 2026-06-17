#ifndef CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_HPP_
#define CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_HPP_

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "ceres/ceres.h"

#include "kinematic_smoother/exceptions.hpp"
#include "kinematic_smoother/kinematic_smoother_problem_builder.hpp"
#include "kinematic_smoother/options.hpp"
#include "kinematic_smoother/smoother_request.hpp"
#include "kinematic_smoother/solver_utils.hpp"
#include "kinematic_smoother/smoother_validator.hpp"

namespace kinematic_smoother
{

/**
 * @class kinematic_smoother::KinematicSmoother
 * @brief 基于简化运动学状态的路径平滑器。
 *
 * 这个版本把每个状态显式表示为
 * (x, y, theta, kappa, ds)，并通过相邻状态之间的运动学过渡残差来约束
 * 路径演化。方向切换处会显式插入 cusp 段，并在求解后进行硬性后验校验。
 */
class KinematicSmoother
{
public:
  KinematicSmoother() = default;
  ~KinematicSmoother() = default;

  /**
   * @brief 初始化后续求解共用的 Ceres 配置。
   *
   * 该配置会在每次 smooth() 中复用；仅 max_solver_time 会按请求动态覆盖。
   */
  void initialize(const OptimizerParams & params)
  {
    validateOptimizerParams(params);
    debug_ = params.debug;
    // 当前仅公开两种线性求解器：DenseQr（小规模稠密）与
    // SparseNormalCholesky（当前默认，适合本问题稀疏结构）。
    solver_options_.linear_solver_type =
      params.linear_solver == OptimizerParams::LinearSolver::DenseQr
      ? ceres::DENSE_QR
      : ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options_.max_num_iterations = params.max_iterations;
    solver_options_.function_tolerance = params.function_tolerance;
    solver_options_.gradient_tolerance = params.gradient_tolerance;
    solver_options_.parameter_tolerance = params.parameter_tolerance;
    solver_options_.minimizer_progress_to_stdout = debug_;
    solver_options_.logging_type = debug_
      ? ceres::LoggingType::PER_MINIMIZER_ITERATION
      : ceres::LoggingType::SILENT;
  }

  /// 使用结构化请求入口执行一次完整平滑。
  ///
  /// 生命周期约定：request 内部引用（path/start_dir/end_dir/params 等）
  /// 必须在本次调用结束前保持有效。
  /// 输入约定：request.path 的第三维在输入时表示方向符号（+1/-1）。
  /// 输出约定：返回值中 `candidate_path` 保存解包后的候选结果，
  /// `smoothed_path` 保存通过后验校验后的最终输出，第三维均为 yaw（弧度）。
  /// 失败语义：
  /// - 若 request.failure 非空，失败原因会写入该结构。
  /// - 若失败发生在后验校验之后，返回值中的 `candidate_path` 仍会保留诊断候选。
  /// - 若 request.failure 为空，求解失败或后验校验失败会抛异常。
  [[nodiscard]] SmootherResult smooth(const SmootherRequest & request)
  {
    // 1) 基础输入约束：至少两点；启用障碍项时必须有 costmap。
    constexpr const char * smoother_name = "Kinematic smoother";
    if (request.path.size() < 2) {
      throw InvalidPath(std::string(smoother_name) + ": Path must have at least 2 points");
    }
    validateFiniteInput(request, smoother_name);
    validateFiniteParams(request.params, smoother_name);
    if (request.params.obstacleTermsEnabled() && request.costmap == nullptr) {
      throw InvalidCostmap(std::string(smoother_name) + ": Costmap must not be null");
    }

    SmootherResult result;

    // 2) 本次调用可覆盖全局默认的求解时间预算。
    solver_options_.max_solver_time_in_seconds = request.params.max_time;

    // 3) 构建并初始化问题：ESDF、状态展开、残差与边界约束。
    KinematicSmootherProblemBuilder builder(esdf_values_);
    builder.initializeEsdfValues(request.costmap, request.params, request.precomputed_esdf);

    // processed 保存展开后的状态链、gear/cusp 元数据和边界姿态信息。
    const auto processed = KinematicSmootherProblemBuilder::buildProcessedPath(
      request.path,
      request.start_dir,
      request.end_dir,
      request.params,
      request.costmap);
    // variables 是 Ceres 优化变量的连续存储，按 (x, y, theta, kappa, ds) 扁平化。
    std::vector<double> variables = processed.initial_variables;

    // problem 在本次调用内构建并求解，不跨调用复用。
    ceres::Problem problem;
    builder.buildProblem(processed, request.costmap, request.params, variables, problem);

    KinematicSmootherProblemBuilder::applyBounds(
      problem,
      variables.data(),
      processed.reference_points,
      processed.is_cusp_segment,
      processed.state_count,
      request.params.max_curvature,
      request.params.kinematic_max_spacing,
      request.params.reference_point_max_deviation_m);

    // 记录本次参与优化的诊断元数据，供外层诊断 / UI 使用。
    result.optimized_knot_count = processed.state_count;
    result.target_spacing = processed.target_spacing;

    // 4) 调用 Ceres 求解，失败原因统一写入 failure（如提供）。
    if (!solveProblemOrReportFailure(
        problem,
        solver_options_,
        debug_,
        smoother_name,
        request.failure))
    {
      return result;
    }

    // 5) 将内部变量解包为公共路径表示，并执行后验硬校验。
    result.candidate_path = KinematicSmootherProblemBuilder::unpackPath(
      variables,
      processed.state_count);

    // 6) 后验硬校验：过滤数值上收敛但不满足工程约束的结果。
    // 字段顺序需与 SmootherValidator::KinematicRequest 定义严格一致。
    // 这里显式传入优化变量、参考链和 ESDF 缓存，避免校验阶段重新推导。
    const bool accepted = validator_.validateKinematicSolution(
      {
        variables,
        processed.reference_points,
        processed.gears,
        processed.is_cusp_segment,
        processed.state_count,
        processed.start_theta,
        processed.end_theta,
        request.costmap,
        request.params,
        esdf_values_,
      },
      request.failure);

    if (!accepted) {
      return result;
    }

    // 7) 校验通过后按运动学状态做段内插值，并同步生成同源曲率诊断 profile。
    const auto output_profile = KinematicSmootherProblemBuilder::upsamplePathKinematicProfile(
      variables,
      processed,
      request.params);
    result.smoothed_path = output_profile.path;
    result.smoothed_curvatures = output_profile.curvatures;
    result.smoothed_curvature_rates = output_profile.curvature_rates;
    result.success = true;
    return result;
  }

private:
  static void validateOptimizerParams(const OptimizerParams & params)
  {
    if (params.max_iterations <= 0) {
      throw std::invalid_argument("OptimizerParams.max_iterations must be positive");
    }

    auto require_nonnegative_finite = [&](double value, const char * field_name) {
        if (!std::isfinite(value) || value < 0.0) {
          throw std::invalid_argument(
                  std::string("OptimizerParams.") + field_name +
                  " must be finite and non-negative");
        }
      };
    require_nonnegative_finite(params.parameter_tolerance, "parameter_tolerance");
    require_nonnegative_finite(params.function_tolerance, "function_tolerance");
    require_nonnegative_finite(params.gradient_tolerance, "gradient_tolerance");
  }

  static void validateFiniteInput(
    const SmootherRequest & request,
    const char * smoother_name)
  {
    for (size_t index = 0; index < request.path.size(); ++index) {
      const Eigen::Vector3d & point = request.path[index];
      if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z())) {
        throw InvalidPath(
                std::string(smoother_name) +
                ": Path contains a non-finite point at index " + std::to_string(index));
      }
    }
    if (
      !std::isfinite(request.start_dir.x()) ||
      !std::isfinite(request.start_dir.y()) ||
      !std::isfinite(request.end_dir.x()) ||
      !std::isfinite(request.end_dir.y()))
    {
      throw InvalidPath(std::string(smoother_name) + ": Start or goal direction is non-finite");
    }
  }

  static void validateFiniteParams(
    const SmootherParams & params,
    const char * smoother_name)
  {
    auto require_finite = [&](double value, const char * field_name) {
        if (!std::isfinite(value)) {
          throw std::invalid_argument(
                  std::string(smoother_name) +
                  ": SmootherParams." + field_name + " must be finite");
        }
      };
    auto require_positive = [&](double value, const char * field_name) {
        if (!std::isfinite(value) || value <= 0.0) {
          throw std::invalid_argument(
                  std::string(smoother_name) +
                  ": SmootherParams." + field_name + " must be finite and positive");
        }
      };

    require_finite(params.model_weight, "model_weight");
    require_finite(params.obstacle_weight, "obstacle_weight");
    require_finite(params.reference_path_weight, "reference_path_weight");
    require_finite(params.reference_point_max_deviation_m, "reference_point_max_deviation_m");
    require_finite(params.kinematic_curvature_weight, "kinematic_curvature_weight");
    require_finite(params.kinematic_curvature_rate_weight, "kinematic_curvature_rate_weight");
    require_finite(params.kinematic_spacing_weight, "kinematic_spacing_weight");
    require_finite(params.kinematic_max_spacing, "kinematic_max_spacing");
    require_finite(params.path_length_weight, "path_length_weight");
    require_finite(params.fix_weight, "fix_weight");
    // max_curvature<=0 会被 functor/边界/校验统一夹到 1e-6，把路径钉成近似直线，
    // 表现为一个含糊的曲率/求解失败；max_time<=0 会让 Ceres 立刻超时返回未优化的
    // 初值。两者都应在入口处直接拒绝，而不是放行后产生误导性结果。
    require_positive(params.max_curvature, "max_curvature");
    require_positive(params.max_time, "max_time");
    require_finite(params.obstacle_safe_distance, "obstacle_safe_distance");
    require_finite(params.cost_check_radius, "cost_check_radius");
    require_finite(params.path_target_spacing, "path_target_spacing");
    require_finite(params.path_output_spacing, "path_output_spacing");
    require_finite(params.goal_longitudinal_tolerance, "goal_longitudinal_tolerance");
    require_finite(params.goal_lateral_tolerance, "goal_lateral_tolerance");
    require_finite(params.goal_orientation_tolerance, "goal_orientation_tolerance");

    // 后验验收容差表（ValidationTolerances）：非有限值会让 ">" 比较恒为 false，
    // 把校验静默旁路，因此同样要求有限。
    require_finite(params.validation.start_position_m, "validation.start_position_m");
    require_finite(params.validation.goal_position_m, "validation.goal_position_m");
    require_finite(params.validation.cusp_position_m, "validation.cusp_position_m");
    require_finite(
      params.validation.min_segment_displacement_m, "validation.min_segment_displacement_m");
    require_finite(params.validation.start_orientation_rad, "validation.start_orientation_rad");
    require_finite(params.validation.goal_orientation_rad, "validation.goal_orientation_rad");
    require_finite(params.validation.cusp_orientation_rad, "validation.cusp_orientation_rad");
    require_finite(params.validation.curvature_tolerance, "validation.curvature_tolerance");

    if (!params.cost_check_points.empty() && params.cost_check_points.size() % 3 != 0) {
      throw std::invalid_argument(
              std::string(smoother_name) +
              ": SmootherParams.cost_check_points size must be a multiple of 3");
    }
    for (size_t index = 0; index < params.cost_check_points.size(); ++index) {
      if (!std::isfinite(params.cost_check_points[index])) {
        throw std::invalid_argument(
                std::string(smoother_name) +
                ": SmootherParams.cost_check_points contains a non-finite value at index " +
                std::to_string(index));
      }
    }
  }

  // 与本实例生命周期绑定的 ESDF 缓存，用于构建器与后验校验共享读取。
  std::vector<double> esdf_values_{};
  // 后验硬约束校验器：用于拒绝数值收敛但工程不可交付的结果。
  SmootherValidator validator_{};

  // 初始化阶段固定配置：是否打印详细求解日志。
  bool debug_{false};
  // Ceres 通用求解配置；每次 smooth() 会补充 max_solver_time。
  // 该对象由 initialize() 建立基线，避免每次请求重复设置静态项。
  ceres::Solver::Options solver_options_{};
};

}  // namespace kinematic_smoother

#endif  // CONSTRAINED_SMOOTHER__KINEMATIC_SMOOTHER_HPP_
