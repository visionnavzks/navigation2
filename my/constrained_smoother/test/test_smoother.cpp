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
#include "constrained_smoother/smoother_path_ops.hpp"
#include "constrained_smoother/smoother_run_base.hpp"
#include "constrained_smoother/smoother_validator.hpp"
#include "gtest/gtest.h"
#include "constrained_smoother/smoother.hpp"
#include "constrained_smoother/smoother_cost_function.hpp"

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

class TestableSmootherCostFunction : public constrained_smoother::SmootherCostFunction
{
public:
  TestableSmootherCostFunction(
    const Eigen::Vector2d & original_pos,
    double last_to_current_length_ratio,
    bool reversing,
    const constrained_smoother::Costmap2D * costmap,
    const std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> &
    esdf_interpolator,
    const constrained_smoother::SmootherParams & params,
    double costmap_weight)
  : SmootherCostFunction(
      original_pos, last_to_current_length_ratio, reversing,
      costmap, esdf_interpolator,
      params, costmap_weight)
  {
  }

  inline double getCurvatureResidual(
    const double & weight,
    const Eigen::Vector2d & pt,
    const Eigen::Vector2d & pt_next,
    const Eigen::Vector2d & pt_prev) const
  {
    double r = 0.0;
    addCurvatureResidual<double>(weight, pt, pt_next, pt_prev, r);
    return r;
  }
};

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

TEST(SolverBackedSmootherBaseTest, ValidateCommonInputsRejectsShortPathAndNullCostmap)
{
  TestSolverBackedSmootherBase base;
  constrained_smoother::Costmap2D costmap(10, 10, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> short_path = {{0.0, 0.0, 1.0}};
  const std::vector<Eigen::Vector3d> valid_path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
  };

  EXPECT_THROW(
    base.validateCommonInputs(short_path, &costmap, "Test smoother"),
    constrained_smoother::InvalidPath);
  EXPECT_THROW(
    base.validateCommonInputs(valid_path, nullptr, "Test smoother"),
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

TEST(CostFunctionTest, CurvatureResidual)
{
  constrained_smoother::Costmap2D costmap(10, 10, 0.05, 0.0, 0.0);
  TestableSmootherCostFunction fn(
    Eigen::Vector2d(1.0, 0.0), 1.0, false,
    &costmap, std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>(),
    constrained_smoother::SmootherParams(), 0.0
  );

  Eigen::Vector2d pt(1.0, 0.0);
  Eigen::Vector2d pt_other(0.0, 0.0);
  EXPECT_EQ(fn.getCurvatureResidual(0.0, pt, pt_other, pt_other), 0.0);

  constrained_smoother::SmootherParams params_no_min;
  params_no_min.max_curvature = 1.0f / 0.0;
  TestableSmootherCostFunction fn2(
    Eigen::Vector2d(1.0, 0.0), 1.0, false,
    &costmap, std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>(),
    params_no_min, 0.0
  );
  EXPECT_EQ(fn2.getCurvatureResidual(1.0, pt, pt_other, pt_other), 0.0);
}

TEST(CostFunctionTest, CurvatureRateResidual)
{
  constrained_smoother::CurvatureRateCostFunction fn(2.0);

  double pt_prev[2] = {0.0, 0.0};
  double pt[2] = {1.0, 0.0};
  double pt_next[2] = {2.0, 0.0};
  double pt_next2[2] = {3.0, 0.0};
  double residual[2] = {0.0, 0.0};

  EXPECT_TRUE(fn(pt_prev, pt, pt_next, pt_next2, residual));
  EXPECT_DOUBLE_EQ(residual[0], 0.0);
  EXPECT_DOUBLE_EQ(residual[1], 0.0);

  pt_next2[1] = 1.0;
  EXPECT_TRUE(fn(pt_prev, pt, pt_next, pt_next2, residual));
  EXPECT_DOUBLE_EQ(residual[0], 0.0);
  EXPECT_DOUBLE_EQ(residual[1], 2.0);
}

TEST(UtilsTest, ArcCenterAndTangent)
{
  Eigen::Vector2d pt(1.0, 0.0);
  Eigen::Vector2d pt_prev(0.0, 0.0);
  Eigen::Vector2d pt_next(0.0, 0.0);

  auto center = constrained_smoother::arcCenter(pt_prev, pt, pt_next, false);
  EXPECT_EQ(center[0], std::numeric_limits<double>::infinity());
  EXPECT_EQ(center[1], std::numeric_limits<double>::infinity());

  auto tangent =
    constrained_smoother::tangentDir(pt_prev, pt, pt_next, false).normalized();
  EXPECT_NEAR(tangent[0], 0, 1e-10);
  EXPECT_NEAR(std::abs(tangent[1]), 1, 1e-10);

  tangent = constrained_smoother::tangentDir(pt_prev, pt, pt_next, true).normalized();
  EXPECT_NEAR(std::abs(tangent[0]), 1, 1e-10);
  EXPECT_NEAR(tangent[1], 0, 1e-10);

  pt_prev[0] = -1.0;
  tangent = constrained_smoother::tangentDir(pt_prev, pt, pt_next, true).normalized();
  EXPECT_NEAR(std::abs(tangent[0]), 1, 1e-10);
  EXPECT_NEAR(tangent[1], 0, 1e-10);

  pt_prev[0] = 0.0;
  pt_next[0] = -1.0;
  tangent = constrained_smoother::tangentDir(pt_prev, pt, pt_next, true).normalized();
  EXPECT_NEAR(std::abs(tangent[0]), 1, 1e-10);
  EXPECT_NEAR(tangent[1], 0, 1e-10);
}

// ---- Extracted helper-layer tests ----

TEST(SmootherPathOpsTest, InitializeOptimizationPathAnchorsEndpoints)
{
  std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
    {2.0, 0.0, 1.0},
    {3.0, 0.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  const Eigen::Vector2d start_dir(0.0, 1.0);
  const Eigen::Vector2d end_dir(0.0, 1.0);
  constrained_smoother::SmootherPathOps path_ops(start_dir, end_dir, params);

  std::vector<Eigen::Vector3d> path_optim;
  std::vector<bool> optimized;
  path_ops.initializeOptimizationPath(path, path_optim, optimized);

  ASSERT_EQ(path_optim.size(), path.size());
  ASSERT_EQ(optimized.size(), path.size());
  EXPECT_TRUE(optimized.front());
  EXPECT_FALSE(optimized[1]);
  EXPECT_NEAR(path_optim[1].x(), 0.0, 1e-9);
  EXPECT_NEAR(path_optim[1].y(), 1.0, 1e-9);
  EXPECT_NEAR(path_optim[2].x(), 3.0, 1e-9);
  EXPECT_NEAR(path_optim[2].y(), -1.0, 1e-9);
}

TEST(SmootherPathOpsTest, PopulateOutputRestoresStraightYaw)
{
  const std::vector<Eigen::Vector3d> path_optim = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
    {2.0, 0.0, 1.0},
    {3.0, 0.0, 1.0},
  };
  const std::vector<bool> optimized = {true, true, true, true};

  constrained_smoother::SmootherParams params;
  params.path_upsampling_factor = 1;
  params.keep_start_orientation = false;
  params.keep_goal_orientation = false;

  constrained_smoother::SmootherPathOps path_ops(
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
    params);

  std::vector<Eigen::Vector3d> output;
  path_ops.populateOutput(path_optim, optimized, output);

  ASSERT_EQ(output.size(), path_optim.size());
  for (const auto & pose : output) {
    EXPECT_NEAR(pose.z(), 0.0, 1e-9);
  }
}

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
  params.costmap_weight_sqrt = 0.0;
  params.cusp_costmap_weight_sqrt = 0.0;
  params.distance_weight_sqrt = 0.0;
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

// ---- Geometric smoother behavior and error-surface tests ----

TEST(SmootherTest, SmoothStraightPath)
{
  // Create a small costmap with all free space
  constrained_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);

  // Create a straight path with slight perturbation
  std::vector<Eigen::Vector3d> path;
  for (int i = 0; i < 10; i++) {
    double x = 0.5 + i * 0.1;
    double y = 2.5 + (i == 5 ? 0.05 : 0.0);  // small bump at midpoint
    path.emplace_back(x, y, 1.0);  // forward direction
  }

  Eigen::Vector2d start_dir(1.0, 0.0);
  Eigen::Vector2d end_dir(1.0, 0.0);

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(2000000.0);
  params.costmap_weight_sqrt = std::sqrt(0.015);
  params.cusp_costmap_weight_sqrt = params.costmap_weight_sqrt * std::sqrt(3.0);
  params.cusp_zone_length = 2.5;
  params.distance_weight_sqrt = std::sqrt(0.0);
  params.curvature_weight_sqrt = std::sqrt(30.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 10.0;

  constrained_smoother::OptimizerParams opt_params;

  constrained_smoother::Smoother smoother;
  smoother.initialize(opt_params);

  EXPECT_NO_THROW(smoother.smooth(path, start_dir, end_dir, &costmap, params));
  EXPECT_GE(path.size(), 2u);
}

TEST(SmootherTest, PathTooShortThrows)
{
  constrained_smoother::Costmap2D costmap(10, 10, 0.05, 0.0, 0.0);
  std::vector<Eigen::Vector3d> path;
  path.emplace_back(0.0, 0.0, 1.0);

  Eigen::Vector2d start_dir(1.0, 0.0);
  Eigen::Vector2d end_dir(1.0, 0.0);

  constrained_smoother::SmootherParams params;
  constrained_smoother::OptimizerParams opt_params;

  constrained_smoother::Smoother smoother;
  smoother.initialize(opt_params);

  EXPECT_THROW(
    smoother.smooth(path, start_dir, end_dir, &costmap, params),
    constrained_smoother::InvalidPath);
}

TEST(SmootherTest, NullCostmapThrowsStructuredError)
{
  std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
  };

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  constrained_smoother::SmootherParams params;
  constrained_smoother::OptimizerParams opt_params;

  constrained_smoother::Smoother smoother;
  smoother.initialize(opt_params);

  EXPECT_THROW(
    smoother.smooth(path, start_dir, end_dir, nullptr, params),
    constrained_smoother::InvalidCostmap);
}

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

TEST(SmootherTest, PrecomputedEsdfSizeMismatchThrowsStructuredError)
{
  constrained_smoother::Costmap2D costmap(10, 10, 0.05, 0.0, 0.0);
  std::vector<Eigen::Vector3d> path = {
    Eigen::Vector3d(0.0, 0.0, 1.0),
    Eigen::Vector3d(0.5, 0.0, 1.0),
  };

  Eigen::Vector2d start_dir(1.0, 0.0);
  Eigen::Vector2d end_dir(1.0, 0.0);

  constrained_smoother::SmootherParams params;
  constrained_smoother::OptimizerParams opt_params;
  constrained_smoother::Smoother smoother;
  smoother.initialize(opt_params);

  const std::vector<double> bad_esdf(8, 0.0);

  EXPECT_THROW(
    smoother.smooth(path, start_dir, end_dir, &costmap, params, &bad_esdf),
    constrained_smoother::PrecomputedEsdfSizeMismatch);
}

TEST(SmootherTest, FootprintCollisionFailsPostValidation)
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
  params.smooth_weight_sqrt = std::sqrt(1000.0);
  params.costmap_weight_sqrt = 0.0;
  params.cusp_costmap_weight_sqrt = 0.0;
  params.distance_weight_sqrt = 0.0;
  params.curvature_weight_sqrt = std::sqrt(1.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.18;
  params.cost_check_points = {0.0, 0.0, 1.0};

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  constrained_smoother::Smoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {smoother.smooth(path, start_dir, end_dir, &costmap, params);});

  EXPECT_NE(error_message.find("footprint_collision@"), std::string::npos);
}

TEST(SmootherTest, FootprintCollisionStoresFailureInfoWithoutThrowing)
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
  params.smooth_weight_sqrt = std::sqrt(1000.0);
  params.costmap_weight_sqrt = 0.0;
  params.cusp_costmap_weight_sqrt = 0.0;
  params.distance_weight_sqrt = 0.0;
  params.curvature_weight_sqrt = std::sqrt(1.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.18;
  params.cost_check_points = {0.0, 0.0, 1.0};

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  constrained_smoother::Smoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);
  constrained_smoother::SmoothingFailureInfo failure;

  EXPECT_FALSE(smoother.smooth(path, start_dir, end_dir, &costmap, params, nullptr, &failure));
  EXPECT_EQ(failure.reason, constrained_smoother::SmoothingFailureReason::FootprintCollision);
  EXPECT_GE(failure.failed_index, 0);
  EXPECT_NE(failure.message.find("collides with obstacles"), std::string::npos);
}

TEST(SmootherTest, PathOutOfBoundsFailsPostValidation)
{
  constrained_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = std::sqrt(1000.0);
  params.costmap_weight_sqrt = 0.0;
  params.cusp_costmap_weight_sqrt = 0.0;
  params.distance_weight_sqrt = 0.0;
  params.curvature_weight_sqrt = std::sqrt(1.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.1;
  params.cost_check_points = {4.0, 0.0, 1.0};

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  constrained_smoother::Smoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {smoother.smooth(path, start_dir, end_dir, &costmap, params);});

  EXPECT_NE(error_message.find("path_out_of_bounds@"), std::string::npos);
}

TEST(SmootherTest, CurvatureConstraintFailsPostValidation)
{
  constrained_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path = {
    {1.0, 1.0, 1.0},
    {1.1, 1.0, 1.0},
    {1.1, 1.1, 1.0},
    {1.2, 1.1, 1.0},
  };

  constrained_smoother::SmootherParams params;
  params.smooth_weight_sqrt = 0.0;
  params.costmap_weight_sqrt = 0.0;
  params.cusp_costmap_weight_sqrt = 0.0;
  params.distance_weight_sqrt = 0.0;
  params.curvature_weight_sqrt = 0.0;
  params.max_curvature = 1.0;
  params.max_time = 1.0;

  constrained_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 5;

  constrained_smoother::Smoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(0.0, 1.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {smoother.smooth(path, start_dir, end_dir, &costmap, params);});

  EXPECT_NE(error_message.find("curvature_constraint@"), std::string::npos);
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
  params.costmap_weight_sqrt = std::sqrt(0.5);
  params.cusp_costmap_weight_sqrt = std::sqrt(0.75);
  params.distance_weight_sqrt = std::sqrt(1.0);
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

TEST(KinematicSmootherTest, NullCostmapThrowsStructuredError)
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
  params.costmap_weight_sqrt = std::sqrt(1.0);
  params.cusp_costmap_weight_sqrt = std::sqrt(1.5);
  params.distance_weight_sqrt = std::sqrt(1.0);
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
  params.costmap_weight_sqrt = std::sqrt(0.0);
  params.cusp_costmap_weight_sqrt = std::sqrt(0.0);
  params.distance_weight_sqrt = std::sqrt(0.0);
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
  params.costmap_weight_sqrt = std::sqrt(0.0);
  params.cusp_costmap_weight_sqrt = std::sqrt(0.0);
  params.distance_weight_sqrt = std::sqrt(0.0);
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
  params.costmap_weight_sqrt = std::sqrt(0.0);
  params.cusp_costmap_weight_sqrt = std::sqrt(0.0);
  params.distance_weight_sqrt = std::sqrt(0.0);
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
  params.costmap_weight_sqrt = std::sqrt(0.0);
  params.cusp_costmap_weight_sqrt = std::sqrt(0.0);
  params.distance_weight_sqrt = std::sqrt(0.0);
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
  params.costmap_weight_sqrt = std::sqrt(0.0);
  params.cusp_costmap_weight_sqrt = std::sqrt(0.0);
  params.distance_weight_sqrt = std::sqrt(0.0);
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
