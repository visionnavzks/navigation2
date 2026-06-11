#ifndef SMOOTHER_CLOTHOID__SMOOTHER_HPP_
#define SMOOTHER_CLOTHOID__SMOOTHER_HPP_

#include <string>
#include <vector>

#include "ceres/ceres.h"

#include "smoother_clothoid/exceptions.hpp"
#include "smoother_clothoid/options.hpp"
#include "smoother_clothoid/problem_builder.hpp"
#include "smoother_clothoid/smoother_request.hpp"
#include "smoother_clothoid/solver_utils.hpp"
#include "smoother_clothoid/validator.hpp"

namespace smoother_clothoid
{

class ClothoidSmoother
{
public:
  ClothoidSmoother() = default;
  ~ClothoidSmoother() = default;

  void initialize(const OptimizerParams & params)
  {
    debug_ = params.debug;
    solver_options_.linear_solver_type =
      params.linear_solver == OptimizerParams::LinearSolver::DenseQr
      ? ceres::DENSE_QR : ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options_.max_num_iterations = params.max_iterations;
    solver_options_.function_tolerance = params.function_tolerance;
    solver_options_.gradient_tolerance = params.gradient_tolerance;
    solver_options_.parameter_tolerance = params.parameter_tolerance;
    solver_options_.minimizer_progress_to_stdout = debug_;
    solver_options_.logging_type = debug_
      ? ceres::LoggingType::PER_MINIMIZER_ITERATION : ceres::LoggingType::SILENT;
  }

  [[nodiscard]] SmootherResult smooth(const SmootherRequest & request)
  {
    constexpr const char * name = "Clothoid smoother";
    if (request.path.size() < 2)
      throw InvalidPath(std::string(name) + ": Path must have at least 2 points");
    if (request.params.obstacleTermsEnabled() && request.costmap == nullptr)
      throw InvalidCostmap(std::string(name) + ": Costmap must not be null");

    SmootherResult result;
    solver_options_.max_solver_time_in_seconds = request.params.max_time;

    ProblemBuilder builder(esdf_values_);
    builder.initializeEsdfValues(request.costmap, request.params, request.precomputed_esdf);

    const auto processed = ProblemBuilder::buildProcessedPath(
      request.path, request.start_dir, request.end_dir, request.params, request.costmap);
    std::vector<double> variables = processed.initial_variables;

    ceres::Problem problem;
    builder.buildProblem(processed, request.costmap, request.params, variables, problem);
    ProblemBuilder::applyBounds(problem, variables.data(), processed.reference_points,
      processed.is_cusp_segment, processed.state_count, request.params.max_curvature,
      request.params.kinematic_max_spacing, request.params.reference_point_max_deviation_m);

    result.optimized_knot_count = processed.state_count;
    result.target_spacing = processed.target_spacing;

    if (!solveProblemOrReportFailure(problem, solver_options_, debug_, name, request.failure))
      return result;

    result.candidate_path = ProblemBuilder::unpackPath(variables, processed.state_count);

    const bool accepted = validator_.validateSolution({
      variables, processed.reference_points, processed.gears, processed.is_cusp_segment,
      processed.state_count, processed.start_theta, processed.end_theta,
      request.costmap, request.params, esdf_values_
    }, request.failure);

    if (!accepted) return result;

    result.smoothed_path = ProblemBuilder::upsamplePath(variables, processed, request.params);
    result.success = true;
    return result;
  }

private:
  std::vector<double> esdf_values_{};
  SmootherValidator validator_{};
  bool debug_{false};
  ceres::Solver::Options solver_options_{};
};

}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__SMOOTHER_HPP_
