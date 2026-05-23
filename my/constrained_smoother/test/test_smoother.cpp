// Copyright (c) 2021 RoboTech Vision
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

#include <vector>
#include <cmath>
#include <string>

#include "constrained_smoother/kinematic_smoother_problem_builder.hpp"
#include "constrained_smoother/kinematic_smoother.hpp"
#include "constrained_smoother/smoother_base.hpp"
#include "constrained_smoother/smoother_run_base.hpp"
#include "constrained_smoother/smoother_validator.hpp"
#include "gtest/gtest.h"

namespace
{

template<typename CallableT>
std::string expectFailedToSmoothPath(CallableT && callable)
{
  try {
    callable();
  } catch (const constrained_smoother::FailedToSmoothPath & error) {
    return error.what();
  } catch (const std::exception & error) {
    ADD_FAILURE() << "Expected FailedToSmoothPath, got: " << error.what();
    return "";
  }

  ADD_FAILURE() << "Expected FailedToSmoothPath to be thrown";
  return "";
}

}  // namespace

// NOTE: This file currently keeps helper-layer, backend-behavior, and stable-error
// contract tests together because the standalone project still validates most slices
// through one executable target. If it grows further, the cleanest split boundary is:
//   1. math / cost-function / helper tests,
//   2. geometric smoother behavior and failure-surface tests,
//   3. kinematic smoother behavior and failure-surface tests,
//   4. shared error-code / costmap sanity tests.

// ---- Testable subclass to expose protected methods ----

;

class TestSolverBackedSmootherBase : public constrained_smoother::SolverBackedSmootherBase
{
public:
  using constrained_smoother::SolverBackedSmootherBase::initializeOptimizer;
  using constrained_smoother::SolverBackedSmootherBase::isDebugEnabled;
  using constrained_smoother::SolverBackedSmootherBase::setMaxSolverTime;
  using constrained_smoother::SolverBackedSmootherBase::solvePreparedProblem;
  using constrained_smoother::SolverBackedSmootherBase::validateCommonInputs;
};

struct TestRunOwner
{
  int prepare_calls{0};
  int solve_calls{0};
  int finalize_calls{0};
};

struct TestRunRequest
{
  int token{0};
};

class TestRunSuccess : public constrained_smoother::SmootherRunBase<TestRunSuccess, TestRunOwner, TestRunRequest>
{
public:
  TestRunSuccess(TestRunOwner & owner, const TestRunRequest & request)
  : constrained_smoother::SmootherRunBase<TestRunSuccess, TestRunOwner, TestRunRequest>(owner, request)
  {
  }

  void prepare()
  {
    owner().prepare_calls += request().token;
  }

  bool solve()
  {
    owner().solve_calls += request().token;
    return true;
  }

  bool finalize()
  {
    owner().finalize_calls += request().token;
    return true;
  }
};

class TestRunFailure : public constrained_smoother::SmootherRunBase<TestRunFailure, TestRunOwner, TestRunRequest>
{
public:
  TestRunFailure(TestRunOwner & owner, const TestRunRequest & request)
  : constrained_smoother::SmootherRunBase<TestRunFailure, TestRunOwner, TestRunRequest>(owner, request)
  {
  }

  void prepare()
  {
    owner().prepare_calls += request().token;
  }

  bool solve()
  {
    owner().solve_calls += request().token;
    return false;
  }

  bool finalize()
  {
    owner().finalize_calls += request().token;
    return true;
  }
};

struct QuadraticResidual
{
  template<typename T>
  bool operator()(const T * const x, T * residual) const
  {
    residual[0] = x[0];
    return true;
  }
};

// ---- Low-level math and cost-function tests ----

TEST(SmootherRunBaseTest, ExecuteCallsPrepareSolveFinalizeInOrder)
{
  TestRunOwner owner;
  const TestRunRequest request{2};

  TestRunSuccess run(owner, request);

  EXPECT_TRUE(run.execute());
  EXPECT_EQ(owner.prepare_calls, 2);
  EXPECT_EQ(owner.solve_calls, 2);
  EXPECT_EQ(owner.finalize_calls, 2);
}

TEST(SmootherRunBaseTest, ExecuteShortCircuitsFinalizeWhenSolveFails)
{
  TestRunOwner owner;
  const TestRunRequest request{3};

  TestRunFailure run(owner, request);

  EXPECT_FALSE(run.execute());
  EXPECT_EQ(owner.prepare_calls, 3);
  EXPECT_EQ(owner.solve_calls, 3);
  EXPECT_EQ(owner.finalize_calls, 0);
}

TEST(SolverBackedSmootherBaseTest, ValidateCommonInputsRejectsShortPathAndOnlyRequiresCostmapForObstacleSlices)
{
  TestSolverBackedSmootherBase base;
  constrained_smoother::Costmap2D costmap(10, 10, 0.05, 0.0, 0.0);
  constrained_smoother::SmootherParams params;
  const std::vector<Eigen::Vector3d> short_path = {{0.0, 0.0, 1.0}};
  const std::vector<Eigen::Vector3d> valid_path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
  };

  EXPECT_THROW(
    base.validateCommonInputs(short_path, &costmap, params, "Test smoother"),
    constrained_smoother::InvalidPath);

  EXPECT_NO_THROW(base.validateCommonInputs(valid_path, nullptr, params, "Test smoother"));

  params.costmap_weight_sqrt = 1.0;
  EXPECT_THROW(
    base.validateCommonInputs(valid_path, nullptr, params, "Test smoother"),
    constrained_smoother::InvalidCostmap);
}

TEST(SolverBackedSmootherBaseTest, InitializeOptimizerAndSolvePreparedProblemSucceedForSimpleProblem)
{
  TestSolverBackedSmootherBase base;
  constrained_smoother::OptimizerParams params;
  params.debug = true;
  params.max_iterations = 20;
  base.initializeOptimizer(params);
  EXPECT_TRUE(base.isDebugEnabled());

  double x = 1.0;
  ceres::Problem problem;
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<QuadraticResidual, 1, 1>(new QuadraticResidual()),
    nullptr,
    &x);

  constrained_smoother::SmoothingFailureInfo failure;
  base.setMaxSolverTime(1.0);

  EXPECT_TRUE(base.solvePreparedProblem(problem, "Test smoother", &failure));
  EXPECT_NEAR(x, 0.0, 1e-6);
  EXPECT_EQ(failure.reason, constrained_smoother::SmoothingFailureReason::Unknown);
}

// ---- Extracted helper-layer tests ----

TEST(KinematicSmootherProblemBuilderTest, BuildProcessedPathInsertsCuspState)
{
  constrained_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, -1.0},
    {0.5, 0.0, -1.0},
  };

  constrained_smoother::SmootherParams params;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  const auto processed = constrained_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(-1.0, 0.0),
    params,
    &costmap);

  EXPECT_EQ(processed.state_count, 4u);
  ASSERT_EQ(processed.gears.size(), 3u);
  ASSERT_EQ(processed.is_cusp_segment.size(), 3u);
  EXPECT_DOUBLE_EQ(processed.gears[0], 1.0);
  EXPECT_DOUBLE_EQ(processed.gears[1], 0.0);
  EXPECT_DOUBLE_EQ(processed.gears[2], -1.0);
  EXPECT_FALSE(processed.is_cusp_segment[0]);
  EXPECT_TRUE(processed.is_cusp_segment[1]);
  EXPECT_FALSE(processed.is_cusp_segment[2]);
  EXPECT_NEAR(processed.reference_points[1].x(), 1.0, 1e-9);
  EXPECT_NEAR(processed.reference_points[2].x(), 1.0, 1e-9);
  EXPECT_EQ(processed.initial_variables.size(), processed.state_count * 5);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProcessedPathHonorsDisabledReversing)
{
  constrained_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, -1.0},
    {0.5, 0.0, -1.0},
  };

  constrained_smoother::SmootherParams params;
  params.reversing_enabled = false;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  const auto processed = constrained_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
    params,
    &costmap);

  EXPECT_EQ(processed.state_count, 3u);
  ASSERT_EQ(processed.gears.size(), 2u);
  ASSERT_EQ(processed.is_cusp_segment.size(), 2u);
  EXPECT_DOUBLE_EQ(processed.gears[0], 1.0);
  EXPECT_DOUBLE_EQ(processed.gears[1], 1.0);
  EXPECT_FALSE(processed.is_cusp_segment[0]);
  EXPECT_FALSE(processed.is_cusp_segment[1]);
}

TEST(KinematicSmootherCostTest, TransitionCostKeepsCurvatureWeightsIndependent)
{
  constrained_smoother::kinematic_smoother_detail::TransitionCostFunctor curvature_cost(
    1.0, false, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0);
  const double current_state[5] = {0.0, 0.0, 0.0, 2.0, 1.0};
  const double next_state[5] = {0.0, 0.0, 0.0, 2.0, 0.0};
  double curvature_residuals[6] = {};

  EXPECT_TRUE(curvature_cost(current_state, next_state, curvature_residuals));
  EXPECT_DOUBLE_EQ(curvature_residuals[3], 6.0);
  EXPECT_DOUBLE_EQ(curvature_residuals[4], 0.0);

  constrained_smoother::kinematic_smoother_detail::TransitionCostFunctor curvature_rate_cost(
    1.0, false, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0);
  const double rate_current_state[5] = {0.0, 0.0, 0.0, 1.0, 4.0};
  const double rate_next_state[5] = {0.0, 0.0, 0.0, 3.0, 0.0};
  double curvature_rate_residuals[6] = {};

  EXPECT_TRUE(curvature_rate_cost(rate_current_state, rate_next_state, curvature_rate_residuals));
  EXPECT_DOUBLE_EQ(curvature_rate_residuals[3], 0.0);
  EXPECT_DOUBLE_EQ(curvature_rate_residuals[4], 4.0);
}

TEST(KinematicSmootherCostTest, BoundaryCostUsesGoalFrameTolerances)
{
  constrained_smoother::kinematic_smoother_detail::BoundaryCostFunctor goal_cost(
    Eigen::Vector2d(0.0, 0.0),
    M_PI / 2.0,
    true,
    0.2,
    0.1,
    0.05,
    10.0,
    false);

  double state_within_tolerance[5] = {0.05, 0.15, M_PI / 2.0 + 0.03, 0.0, 0.0};
  double residuals_within[4] = {};
  EXPECT_TRUE(goal_cost(state_within_tolerance, residuals_within));
  EXPECT_DOUBLE_EQ(residuals_within[0], 0.0);
  EXPECT_DOUBLE_EQ(residuals_within[1], 0.0);
  EXPECT_DOUBLE_EQ(residuals_within[2], 0.0);

  double state_outside_tolerance[5] = {0.05, 0.25, M_PI / 2.0 + 0.07, 0.0, 0.0};
  double residuals_outside[4] = {};
  EXPECT_TRUE(goal_cost(state_outside_tolerance, residuals_outside));
  EXPECT_NEAR(residuals_outside[0], 0.5, 1e-9);
  EXPECT_DOUBLE_EQ(residuals_outside[1], 0.0);
  EXPECT_NEAR(residuals_outside[2], 0.2, 1e-9);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProblemAddsTransitionAndBoundaryBlocks)
{
  constrained_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
    {1.0, 0.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(1.0);
  params.model_weight_sqrt = std::sqrt(1.0);
  params.costmap_weight_sqrt = 0.0;
  params.cusp_costmap_weight_sqrt = 0.0;
  params.distance_weight_sqrt = 0.0;
  params.curvature_weight_sqrt = 0.0;
  params.curvature_rate_weight_sqrt = 0.0;
  params.curvature_weight_sqrt = 0.0;
  params.curvature_rate_weight_sqrt = 0.0;

  std::vector<double> esdf_values;
  constrained_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
  builder.initializeEsdfValues(&costmap, params, nullptr);
  const auto processed = constrained_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
    params,
    &costmap);

  std::vector<double> variables = processed.initial_variables;
  ceres::Problem problem;
  builder.buildProblem(processed, &costmap, params, variables, problem);

  EXPECT_EQ(problem.NumParameterBlocks(), static_cast<int>(processed.state_count));
  EXPECT_EQ(problem.NumResidualBlocks(), 4);

  const auto unpacked = constrained_smoother::KinematicSmootherProblemBuilder::unpackPath(
    variables, processed.state_count);
  ASSERT_EQ(unpacked.size(), processed.state_count);
  EXPECT_NEAR(unpacked.front().x(), path.front().x(), 1e-9);
  EXPECT_NEAR(unpacked.back().x(), path.back().x(), 1e-9);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProblemUsesDedicatedModelWeight)
{
  constrained_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
    {1.0, 0.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = 0.0;
  params.model_weight_sqrt = 3.0;
  params.costmap_weight_sqrt = 0.0;
  params.cusp_costmap_weight_sqrt = 0.0;
  params.distance_weight_sqrt = 0.0;
  params.curvature_weight_sqrt = 0.0;
  params.curvature_rate_weight_sqrt = 0.0;
  params.curvature_weight_sqrt = 0.0;
  params.curvature_rate_weight_sqrt = 0.0;

  std::vector<double> esdf_values;
  constrained_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
  builder.initializeEsdfValues(&costmap, params, nullptr);
  const auto processed = constrained_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
    params,
    &costmap);

  std::vector<double> variables = processed.initial_variables;
  variables[5] += 0.1;

  ceres::Problem problem;
  builder.buildProblem(processed, &costmap, params, variables, problem);

  ceres::Problem::EvaluateOptions options;
  double cost = 0.0;
  EXPECT_TRUE(problem.Evaluate(options, &cost, nullptr, nullptr, nullptr));
  EXPECT_NEAR(cost, 0.09, 1e-6);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProblemUsesDedicatedKinematicCurvatureWeights)
{
  constrained_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {std::cos(1.0), std::sin(1.0), 1.0},
  };

  auto evaluate_cost = [&](double geometric_curvature_weight, double curvature_weight) {
    constrained_smoother::SmootherParams params;
    params.smooth_weight_sqrt = 0.0;
    params.model_weight_sqrt = 0.0;
    params.costmap_weight_sqrt = 0.0;
    params.cusp_costmap_weight_sqrt = 0.0;
    params.distance_weight_sqrt = 0.0;
    params.curvature_weight_sqrt = geometric_curvature_weight;
    params.curvature_rate_weight_sqrt = 0.0;
    params.curvature_weight_sqrt = curvature_weight;
    params.curvature_rate_weight_sqrt = 0.0;
    params.keep_start_orientation = false;
    params.keep_goal_orientation = false;

    std::vector<double> esdf_values;
    constrained_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
    builder.initializeEsdfValues(&costmap, params, nullptr);
    const auto processed = constrained_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
      path,
      Eigen::Vector2d(1.0, 0.0),
      Eigen::Vector2d(1.0, 0.0),
      params,
      &costmap);

    std::vector<double> variables = processed.initial_variables;
    variables[0] = 0.0;
    variables[1] = 0.0;
    variables[2] = 0.0;
    variables[3] = 2.0;
    variables[4] = 1.0;
    variables[5] = std::cos(1.0);
    variables[6] = std::sin(1.0);
    variables[7] = 2.0;
    variables[8] = 2.0;
    variables[9] = 0.0;

    ceres::Problem problem;
    builder.buildProblem(processed, &costmap, params, variables, problem);

    ceres::Problem::EvaluateOptions options;
    double cost = 0.0;
    EXPECT_TRUE(problem.Evaluate(options, &cost, nullptr, nullptr, nullptr));
    return cost;
  };

  const double geometric_cost = evaluate_cost(3.0, 0.0);
  const double kinematic_cost = evaluate_cost(0.0, 3.0);

  EXPECT_NEAR(geometric_cost, 0.0, 1e-9);
  EXPECT_GT(kinematic_cost, 1.0);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProblemUsesDedicatedKinematicSpacingWeight)
{
  constrained_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
  };

  auto evaluate_cost = [&](double spacing_weight) {
    constrained_smoother::SmootherParams params;
    params.smooth_weight_sqrt = 0.0;
    params.model_weight_sqrt = 0.0;
    params.costmap_weight_sqrt = 0.0;
    params.cusp_costmap_weight_sqrt = 0.0;
    params.distance_weight_sqrt = 0.0;
    params.curvature_weight_sqrt = 0.0;
    params.curvature_rate_weight_sqrt = 0.0;
    params.curvature_weight_sqrt = 0.0;
    params.curvature_rate_weight_sqrt = 0.0;
    params.kinematic_spacing_weight_sqrt = spacing_weight;
    params.keep_start_orientation = false;
    params.keep_goal_orientation = false;
    params.goal_longitudinal_tolerance = 2.0;
    params.goal_lateral_tolerance = 2.0;

    std::vector<double> esdf_values;
    constrained_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
    builder.initializeEsdfValues(&costmap, params, nullptr);
    const auto processed = constrained_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
      path,
      Eigen::Vector2d(1.0, 0.0),
      Eigen::Vector2d(1.0, 0.0),
      params,
      &costmap);

    std::vector<double> variables = processed.initial_variables;
    variables[4] = 2.0;
    variables[5] = 2.0;
    variables[6] = 0.0;
    variables[7] = 0.0;
    variables[8] = 0.0;
    variables[9] = 0.0;

    ceres::Problem problem;
    builder.buildProblem(processed, &costmap, params, variables, problem);

    ceres::Problem::EvaluateOptions options;
    double cost = 0.0;
    EXPECT_TRUE(problem.Evaluate(options, &cost, nullptr, nullptr, nullptr));
    return cost;
  };

  EXPECT_NEAR(evaluate_cost(0.0), 0.0, 1e-9);
  EXPECT_NEAR(evaluate_cost(3.0), 4.5, 1e-9);
}

// ---- Geometric smoother behavior and error-surface tests ----

// ---- Stable error-code and failure-message contract tests ----

TEST(ErrorTest, InvalidPathCarriesStableCode)
{
  const constrained_smoother::InvalidPath error("test invalid path");

  EXPECT_EQ(error.code(), constrained_smoother::ErrorCode::InvalidPath);
  EXPECT_STREQ(error.codeString(), "CS_INVALID_PATH");
  EXPECT_STREQ(error.what(), std::string("test invalid path").c_str());
}

TEST(ErrorTest, SmoothingFailureMessageCarriesReasonAndIndex)
{
  const std::string message = constrained_smoother::buildSmoothingFailureMessage(
    constrained_smoother::SmoothingFailureReason::GoalOrientationConstraint,
    "test smoothing failure",
    7);

  EXPECT_EQ(message, "goal_orientation_constraint@7: test smoothing failure");
}

// ---- Kinematic smoother behavior and error-surface tests ----

TEST(KinematicSmootherTest, SmoothStraightPath)
{
  constrained_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path;
  for (int i = 0; i < 10; ++i) {
    const double x = 0.5 + i * 0.1;
    const double y = 2.5 + (i == 5 ? 0.04 : 0.0);
    path.emplace_back(x, y, 1.0);
  }

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(20.0);
  params.model_weight_sqrt = std::sqrt(20.0);
  params.costmap_weight_sqrt = std::sqrt(0.5);
  params.cusp_costmap_weight_sqrt = std::sqrt(0.75);
  params.distance_weight_sqrt = std::sqrt(1.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.obstacle_safe_distance = 0.5;

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 30;

  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  EXPECT_NO_THROW(smoother.smooth(path, start_dir, end_dir, &costmap, params));
  EXPECT_GE(path.size(), 2u);
  EXPECT_GT(smoother.getLastOptimizedKnotCount(), 0u);
}

TEST(KinematicSmootherTest, ReferencePointMaxDeviationDefaultsOffAndBoundsOptimizedPoint)
{
  constrained_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);

  const std::vector<Eigen::Vector3d> reference_path = {
    {0.5, 0.5, 1.0},
    {1.0, 1.0, 1.0},
    {1.5, 0.5, 1.0},
    {2.0, 0.5, 1.0},
  };

  auto run_case = [&](double max_deviation) {
      std::vector<Eigen::Vector3d> path = reference_path;

      constrained_smoother::SmootherParams params;
      params.smooth_weight_sqrt = std::sqrt(20.0);
      params.model_weight_sqrt = std::sqrt(20.0);
      params.costmap_weight_sqrt = 0.0;
      params.cusp_costmap_weight_sqrt = 0.0;
      params.distance_weight_sqrt = 0.0;
      params.curvature_weight_sqrt = 0.0;
      params.curvature_rate_weight_sqrt = 0.0;
      params.curvature_weight_sqrt = 0.0;
      params.curvature_rate_weight_sqrt = 0.0;
      params.max_curvature = 1.0 / 0.4;
      params.max_time = 1.0;
      params.keep_start_orientation = false;
      params.keep_goal_orientation = false;
      params.reference_point_max_deviation = max_deviation;

      constrained_smoother::OptimizerParams opt_params;
      opt_params.max_iterations = 60;

      constrained_smoother::KinematicSmoother smoother;
      smoother.initialize(opt_params);
      smoother.smooth(path, Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(1.0, 0.0), &costmap, params);
      return path;
    };

  const auto unbounded = run_case(0.0);
  const auto bounded = run_case(0.1);

  EXPECT_GT(std::abs(unbounded[1].y() - reference_path[1].y()), 0.100001);
  EXPECT_LE(std::abs(bounded[1].y() - reference_path[1].y()), 0.100001);
}

TEST(KinematicSmootherTest, SmoothCuspPath)
{
  constrained_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path;
  constexpr double spacing = 0.2;
  for (double x = 1.0; x <= 6.0 + 1e-9; x += spacing) {
    path.emplace_back(x, 2.0, 1.0);
  }
  for (double x = 6.0 - spacing; x >= 1.4 - 1e-9; x -= spacing) {
    path.emplace_back(x, 2.0, -1.0);
  }
  const auto input_size = path.size();

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(20.0);
  params.model_weight_sqrt = std::sqrt(20.0);
  params.costmap_weight_sqrt = 0.0;
  params.cusp_costmap_weight_sqrt = 0.0;
  params.distance_weight_sqrt = std::sqrt(0.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 40;

  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  EXPECT_NO_THROW(smoother.smooth(path, start_dir, end_dir, &costmap, params));
  EXPECT_GE(path.size(), 2u);
  EXPECT_GT(smoother.getLastOptimizedKnotCount(), input_size);
}

TEST(KinematicSmootherTest, NullCostmapAllowedWhenObstacleTermsDisabled)
{
  std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
  };

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  constrained_smoother::SmootherParams params;
  constrained_smoother::OptimizerParams opt_params;
  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  EXPECT_NO_THROW(smoother.smooth(path, start_dir, end_dir, nullptr, params));
}

TEST(KinematicSmootherTest, NullCostmapStillRejectedWhenObstacleTermsEnabled)
{
  std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
  };

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  constrained_smoother::SmootherParams params;
  params.costmap_weight_sqrt = 1.0;
  constrained_smoother::OptimizerParams opt_params;
  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  EXPECT_THROW(
    smoother.smooth(path, start_dir, end_dir, nullptr, params),
    constrained_smoother::InvalidCostmap);
}

TEST(KinematicSmootherTest, ObstacleCostCheckPointsDoNotThrow)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);
  for (unsigned int y = 25; y < 55; ++y) {
    for (unsigned int x = 35; x < 45; ++x) {
      costmap.setCost(x, y, constrained_smoother::Costmap2D::LETHAL_OBSTACLE);
    }
  }

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
    {2.5, 2.0, 1.0},
    {3.0, 2.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(20.0);
  params.model_weight_sqrt = std::sqrt(20.0);
  params.costmap_weight_sqrt = std::sqrt(1.0);
  params.cusp_costmap_weight_sqrt = std::sqrt(1.5);
  params.distance_weight_sqrt = std::sqrt(1.0);
  params.curvature_weight_sqrt = std::sqrt(10.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.curvature_weight_sqrt = std::sqrt(10.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.obstacle_safe_distance = 0.35;
  params.cost_check_points = {
    0.0, 0.0, 0.5,
    0.2, 0.15, 1.0,
    0.2, -0.15, 1.0,
    -0.2, 0.15, 1.0,
    -0.2, -0.15, 1.0,
  };

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  EXPECT_NO_THROW(smoother.smooth(path, start_dir, end_dir, &costmap, params));
  EXPECT_GT(smoother.getLastOptimizedKnotCount(), 0u);
}

TEST(KinematicSmootherTest, GoalOrientationCannotSilentlyFlipIntoReverse)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
    {2.5, 2.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(20.0);
  params.model_weight_sqrt = std::sqrt(20.0);
  params.costmap_weight_sqrt = 1e-4;
  params.cusp_costmap_weight_sqrt = 1e-4;
  params.distance_weight_sqrt = std::sqrt(0.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 40;

  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(-1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {smoother.smooth(path, start_dir, end_dir, &costmap, params);});

  EXPECT_NE(error_message.find("motion_direction_constraint@"), std::string::npos);
}

TEST(SmootherValidatorTest, KinematicGoalOrientationUsesGoalStateHeading)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  const std::vector<double> variables = {
    1.0, 2.0, 0.0, 0.0, 0.5,
    1.5, 2.0, 0.0, 0.0, 0.5,
    2.0, 2.2, M_PI / 4.0, 0.0, 0.0,
  };
  const std::vector<Eigen::Vector2d> reference_points = {
    {1.0, 2.0},
    {1.5, 2.0},
    {2.0, 2.2},
  };
  const std::vector<double> gears = {1.0, 1.0};
  const std::vector<bool> is_cusp_segment = {false, false};

  constrained_smoother::SmootherParams params;
  params.keep_goal_orientation = true;
  params.keep_start_orientation = false;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  constrained_smoother::SmoothingFailureInfo failure;
  constrained_smoother::SmootherValidator validator;

  EXPECT_TRUE(validator.validateKinematicSolution(
      {
        variables,
        reference_points,
        gears,
        is_cusp_segment,
        3,
        0.0,
        M_PI / 4.0,
        &costmap,
        params,
        esdf_values,
      },
      &failure));
  EXPECT_EQ(failure.reason, constrained_smoother::SmoothingFailureReason::Unknown);
  EXPECT_EQ(failure.failed_index, -1);
  EXPECT_TRUE(failure.message.empty());
}

TEST(SmootherValidatorTest, KinematicGoalPositionToleranceAllowsGoalSlack)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  const std::vector<double> variables = {
    2.0, 1.0, M_PI / 2.0, 0.0, 0.5,
    2.0, 1.5, M_PI / 2.0, 0.0, 0.5,
    2.05, 2.15, M_PI / 2.0, 0.0, 0.0,
  };
  const std::vector<Eigen::Vector2d> reference_points = {
    {2.0, 1.0},
    {2.0, 1.5},
    {2.0, 2.0},
  };
  const std::vector<double> gears = {1.0, 1.0};
  const std::vector<bool> is_cusp_segment = {false, false};

  constrained_smoother::SmootherParams params;
  params.keep_goal_orientation = true;
  params.keep_start_orientation = true;
  params.goal_longitudinal_tolerance = 0.2;
  params.goal_lateral_tolerance = 0.1;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  constrained_smoother::SmoothingFailureInfo failure;
  constrained_smoother::SmootherValidator validator;

  EXPECT_TRUE(validator.validateKinematicSolution(
      {
        variables,
        reference_points,
        gears,
        is_cusp_segment,
        3,
        M_PI / 2.0,
        M_PI / 2.0,
        &costmap,
        params,
        esdf_values,
      },
      &failure));
}

TEST(SmootherValidatorTest, KinematicGoalPositionToleranceUsesReferenceGoalFrameWhenOrientationDisabled)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  const std::vector<double> variables = {
    1.0, 2.0, 0.0, 0.0, 0.5,
    1.5, 2.0, 0.0, 0.0, 0.5,
    2.15, 2.0, M_PI / 2.0, 0.0, 0.0,
  };
  const std::vector<Eigen::Vector2d> reference_points = {
    {1.0, 2.0},
    {1.5, 2.0},
    {2.0, 2.0},
  };
  const std::vector<double> gears = {1.0, 1.0};
  const std::vector<bool> is_cusp_segment = {false, false};

  constrained_smoother::SmootherParams params;
  params.keep_goal_orientation = false;
  params.keep_start_orientation = true;
  params.goal_longitudinal_tolerance = 0.2;
  params.goal_lateral_tolerance = 0.0;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  constrained_smoother::SmoothingFailureInfo failure;
  constrained_smoother::SmootherValidator validator;

  EXPECT_TRUE(validator.validateKinematicSolution(
      {
        variables,
        reference_points,
        gears,
        is_cusp_segment,
        3,
        0.0,
        M_PI / 2.0,
        &costmap,
        params,
        esdf_values,
      },
      &failure));
}

TEST(SmootherValidatorTest, KinematicGoalPositionToleranceRejectsOutsideGoalBand)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  const std::vector<double> variables = {
    2.0, 1.0, M_PI / 2.0, 0.0, 0.5,
    2.0, 1.5, M_PI / 2.0, 0.0, 0.5,
    2.05, 2.25, M_PI / 2.0, 0.0, 0.0,
  };
  const std::vector<Eigen::Vector2d> reference_points = {
    {2.0, 1.0},
    {2.0, 1.5},
    {2.0, 2.0},
  };
  const std::vector<double> gears = {1.0, 1.0};
  const std::vector<bool> is_cusp_segment = {false, false};

  constrained_smoother::SmootherParams params;
  params.keep_goal_orientation = true;
  params.keep_start_orientation = true;
  params.goal_longitudinal_tolerance = 0.2;
  params.goal_lateral_tolerance = 0.1;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  constrained_smoother::SmoothingFailureInfo failure;
  constrained_smoother::SmootherValidator validator;

  EXPECT_FALSE(validator.validateKinematicSolution(
      {
        variables,
        reference_points,
        gears,
        is_cusp_segment,
        3,
        M_PI / 2.0,
        M_PI / 2.0,
        &costmap,
        params,
        esdf_values,
      },
      &failure));
  EXPECT_EQ(failure.reason, constrained_smoother::SmoothingFailureReason::GoalPositionConstraint);
  EXPECT_NE(failure.message.find("goal position tolerance box"), std::string::npos);
  EXPECT_NEAR(failure.goal_longitudinal_error, 0.25, 1e-9);
  EXPECT_NEAR(failure.goal_lateral_error, -0.05, 1e-9);
  EXPECT_NEAR(failure.goal_longitudinal_tolerance, 0.2, 1e-9);
  EXPECT_NEAR(failure.goal_lateral_tolerance, 0.1, 1e-9);
}

TEST(KinematicSmootherTest, MotionDirectionViolationStoresFailureInfoWithoutThrowing)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
    {2.5, 2.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(20.0);
  params.model_weight_sqrt = std::sqrt(20.0);
  params.costmap_weight_sqrt = 1e-4;
  params.cusp_costmap_weight_sqrt = 1e-4;
  params.distance_weight_sqrt = std::sqrt(0.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 40;

  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(-1.0, 0.0);
  constrained_smoother::SmoothingFailureInfo failure;

  EXPECT_FALSE(smoother.smooth(path, start_dir, end_dir, &costmap, params, nullptr, &failure));
  EXPECT_EQ(failure.reason, constrained_smoother::SmoothingFailureReason::MotionDirectionConstraint);
  EXPECT_GE(failure.failed_index, 0);
  EXPECT_NE(failure.message.find("motion direction"), std::string::npos);
}

TEST(KinematicSmootherTest, FootprintCollisionFailsPostValidation)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);
  for (unsigned int y = 35; y < 45; ++y) {
    for (unsigned int x = 36; x < 42; ++x) {
      costmap.setCost(x, y, constrained_smoother::Costmap2D::LETHAL_OBSTACLE);
    }
  }

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(20.0);
  params.model_weight_sqrt = std::sqrt(20.0);
  params.costmap_weight_sqrt = 1e-4;
  params.cusp_costmap_weight_sqrt = 1e-4;
  params.distance_weight_sqrt = std::sqrt(0.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.18;
  params.cost_check_points = {0.0, 0.0, 1.0};

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {smoother.smooth(path, start_dir, end_dir, &costmap, params);});

  EXPECT_NE(error_message.find("footprint_collision@"), std::string::npos);
}

TEST(KinematicSmootherTest, FootprintRadiusWithoutCheckpointsFailsPostValidation)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);
  for (unsigned int y = 35; y < 45; ++y) {
    for (unsigned int x = 36; x < 42; ++x) {
      costmap.setCost(x, y, constrained_smoother::Costmap2D::LETHAL_OBSTACLE);
    }
  }

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(20.0);
  params.model_weight_sqrt = std::sqrt(20.0);
  params.costmap_weight_sqrt = 1e-4;
  params.cusp_costmap_weight_sqrt = 1e-4;
  params.distance_weight_sqrt = std::sqrt(0.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.18;

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {smoother.smooth(path, start_dir, end_dir, &costmap, params);});

  EXPECT_NE(error_message.find("footprint_collision@"), std::string::npos);
}

TEST(KinematicSmootherTest, PathOutOfBoundsFailsPostValidation)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(20.0);
  params.model_weight_sqrt = std::sqrt(20.0);
  params.costmap_weight_sqrt = 1e-4;
  params.cusp_costmap_weight_sqrt = 1e-4;
  params.distance_weight_sqrt = std::sqrt(0.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.1;
  params.cost_check_points = {4.0, 0.0, 1.0};

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  constrained_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {smoother.smooth(path, start_dir, end_dir, &costmap, params);});

  EXPECT_NE(error_message.find("path_out_of_bounds@"), std::string::npos);
}

// ---- Basic costmap sanity test ----

TEST(CostmapTest, BasicCostmapOperations)
{
  constrained_smoother::Costmap2D costmap(10, 10, 0.05, 1.0, 2.0);
  EXPECT_EQ(costmap.getSizeInCellsX(), 10u);
  EXPECT_EQ(costmap.getSizeInCellsY(), 10u);
  EXPECT_DOUBLE_EQ(costmap.getResolution(), 0.05);
  EXPECT_DOUBLE_EQ(costmap.getOriginX(), 1.0);
  EXPECT_DOUBLE_EQ(costmap.getOriginY(), 2.0);

  costmap.setCost(3, 4, 128);
  EXPECT_EQ(costmap.getCost(3, 4), 128);

  EXPECT_NE(costmap.getCharMap(), nullptr);
}
