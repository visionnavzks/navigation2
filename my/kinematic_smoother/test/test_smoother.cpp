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

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>
#include <cmath>
#include <string>

#include "kinematic_smoother/kinematic_smoother_problem_builder.hpp"
#include "kinematic_smoother/kinematic_smoother.hpp"
#include "kinematic_smoother/smoother_validator.hpp"
#include "gtest/gtest.h"

namespace
{

template<typename CallableT>
std::string expectFailedToSmoothPath(CallableT && callable)
{
  try {
    callable();
  } catch (const kinematic_smoother::FailedToSmoothPath & error) {
    return error.what();
  } catch (const std::exception & error) {
    ADD_FAILURE() << "Expected FailedToSmoothPath, got: " << error.what();
    return "";
  }

  ADD_FAILURE() << "Expected FailedToSmoothPath to be thrown";
  return "";
}

void expectPathsNear(
  const std::vector<Eigen::Vector3d> & actual,
  const std::vector<Eigen::Vector3d> & expected,
  double tolerance = 1e-9)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_TRUE(actual[index].isApprox(expected[index], tolerance)) << "path mismatch at " << index;
  }
}

}  // namespace

// NOTE: This file currently keeps helper-layer, backend-behavior, and stable-error
// contract tests together because the standalone project still validates most slices
// through one executable target. If it grows further, the cleanest split boundary is:
//   1. math / cost-function / helper tests,
//   2. kinematic smoother behavior and failure-surface tests,
//   3. shared error-code / costmap sanity tests.

struct QuadraticResidual
{
  template<typename T>
  bool operator()(const T * const x, T * residual) const
  {
    residual[0] = x[0];
    return true;
  }
};

// ---- Low-level shared tests ----

TEST(KinematicSmootherProblemBuilderTest, BuildProcessedPathInsertsCuspState)
{
  kinematic_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, -1.0},
    {0.5, 0.0, -1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
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

TEST(KinematicSmootherProblemBuilderTest, BuildProcessedPathTreatsDuplicateGearChangeAsCusp)
{
  kinematic_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
    {1.0, 0.0, -1.0},
    {0.5, 0.0, -1.0},
  };

  kinematic_smoother::SmootherParams params;
  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
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
}

TEST(KinematicSmootherProblemBuilderTest, BuildProcessedPathHonorsDisabledReversing)
{
  kinematic_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, -1.0},
    {0.5, 0.0, -1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.reversing_enabled = false;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
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
  kinematic_smoother::kinematic_smoother_detail::TransitionCostFunctor curvature_cost(
    1.0, false, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  const double current_state[5] = {0.0, 0.0, 0.0, 2.0, 1.0};
  const double next_state[5] = {0.0, 0.0, 0.0, 2.0, 0.0};
  double curvature_residuals[7] = {};

  EXPECT_TRUE(curvature_cost(current_state, next_state, curvature_residuals));
  EXPECT_DOUBLE_EQ(curvature_residuals[3], 6.0);
  EXPECT_DOUBLE_EQ(curvature_residuals[4], 0.0);
  EXPECT_DOUBLE_EQ(curvature_residuals[6], 0.0);

  kinematic_smoother::kinematic_smoother_detail::TransitionCostFunctor curvature_rate_cost(
    1.0, false, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0);
  const double rate_current_state[5] = {0.0, 0.0, 0.0, 1.0, 4.0};
  const double rate_next_state[5] = {0.0, 0.0, 0.0, 3.0, 0.0};
  double curvature_rate_residuals[7] = {};

  EXPECT_TRUE(curvature_rate_cost(rate_current_state, rate_next_state, curvature_rate_residuals));
  EXPECT_DOUBLE_EQ(curvature_rate_residuals[3], 0.0);
  EXPECT_DOUBLE_EQ(curvature_rate_residuals[4], 4.0);
  EXPECT_DOUBLE_EQ(curvature_rate_residuals[6], 0.0);
}

TEST(KinematicSmootherCostTest, TransitionCostUsesExplicitLengthPenalty)
{
  kinematic_smoother::kinematic_smoother_detail::TransitionCostFunctor length_cost(
    1.0, false, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0);
  const double current_state[5] = {0.0, 0.0, 0.0, 0.0, 1.5};
  const double next_state[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double residuals[7] = {};

  EXPECT_TRUE(length_cost(current_state, next_state, residuals));
  EXPECT_DOUBLE_EQ(residuals[6], 3.0);
}

TEST(KinematicSmootherCostTest, BoundaryCostUsesGoalFrameTolerances)
{
  kinematic_smoother::kinematic_smoother_detail::BoundaryCostFunctor goal_cost(
    Eigen::Vector2d(0.0, 0.0),
    M_PI / 2.0,
    true,
    0.2,
    0.1,
    0.05,
    10.0);

  double state_within_tolerance[5] = {0.05, 0.15, M_PI / 2.0 + 0.03, 0.0, 0.0};
  double residuals_within[3] = {};
  EXPECT_TRUE(goal_cost(state_within_tolerance, residuals_within));
  EXPECT_DOUBLE_EQ(residuals_within[0], 0.0);
  EXPECT_DOUBLE_EQ(residuals_within[1], 0.0);
  EXPECT_DOUBLE_EQ(residuals_within[2], 0.0);

  double state_outside_tolerance[5] = {0.05, 0.25, M_PI / 2.0 + 0.07, 0.0, 0.0};
  double residuals_outside[3] = {};
  EXPECT_TRUE(goal_cost(state_outside_tolerance, residuals_outside));
  EXPECT_NEAR(residuals_outside[0], 0.5, 1e-9);
  EXPECT_DOUBLE_EQ(residuals_outside[1], 0.0);
  EXPECT_NEAR(residuals_outside[2], 0.2, 1e-9);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProblemAddsTransitionAndBoundaryBlocks)
{
  kinematic_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
    {1.0, 0.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.model_weight = 1.0;
  params.obstacle_weight = 0.0;
  params.reference_path_weight = 0.0;
  params.kinematic_curvature_weight = 0.0;
  params.kinematic_curvature_rate_weight = 0.0;
  params.path_length_weight = 0.0;

  std::vector<double> esdf_values;
  kinematic_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
  builder.initializeEsdfValues(&costmap, params, nullptr);
  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
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

  const auto unpacked = kinematic_smoother::KinematicSmootherProblemBuilder::unpackPath(
    variables, processed.state_count);
  ASSERT_EQ(unpacked.size(), processed.state_count);
  EXPECT_NEAR(unpacked.front().x(), path.front().x(), 1e-9);
  EXPECT_NEAR(unpacked.back().x(), path.back().x(), 1e-9);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProblemUsesDedicatedModelWeight)
{
  kinematic_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
    {1.0, 0.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.model_weight = 3.0;
  params.obstacle_weight = 0.0;
  params.reference_path_weight = 0.0;
  params.kinematic_curvature_weight = 0.0;
  params.kinematic_curvature_rate_weight = 0.0;
  params.path_length_weight = 0.0;

  std::vector<double> esdf_values;
  kinematic_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
  builder.initializeEsdfValues(&costmap, params, nullptr);
  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
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
  EXPECT_NEAR(cost, 0.03, 1e-6);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProblemUsesDedicatedKinematicCurvatureWeight)
{
  kinematic_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {std::cos(1.0), std::sin(1.0), 1.0},
  };

  auto evaluate_cost = [&](double kinematic_curvature_weight) {
    kinematic_smoother::SmootherParams params;
    params.model_weight = 0.0;
    params.obstacle_weight = 0.0;
    params.reference_path_weight = 0.0;
    params.kinematic_curvature_weight = kinematic_curvature_weight;
    params.kinematic_curvature_rate_weight = 0.0;
    params.path_length_weight = 0.0;
    params.keep_start_orientation = false;
    params.keep_goal_orientation = false;

    std::vector<double> esdf_values;
    kinematic_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
    builder.initializeEsdfValues(&costmap, params, nullptr);
    const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
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

  const double zero_cost = evaluate_cost(0.0);
  const double kinematic_cost = evaluate_cost(3.0);

  EXPECT_NEAR(zero_cost, 0.0, 1e-9);
  EXPECT_GT(kinematic_cost, 1.0);
}

TEST(KinematicSmootherProblemBuilderTest, BuildProblemUsesDedicatedKinematicSpacingWeight)
{
  kinematic_smoother::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
  };

  auto evaluate_cost = [&](double spacing_weight) {
    kinematic_smoother::SmootherParams params;
    params.model_weight = 0.0;
    params.obstacle_weight = 0.0;
    params.reference_path_weight = 0.0;
    params.kinematic_curvature_weight = 0.0;
    params.kinematic_curvature_rate_weight = 0.0;
    params.kinematic_spacing_weight = spacing_weight;
    params.path_length_weight = 0.0;
    params.keep_start_orientation = false;
    params.keep_goal_orientation = false;
    params.goal_longitudinal_tolerance = 2.0;
    params.goal_lateral_tolerance = 2.0;

    std::vector<double> esdf_values;
    kinematic_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
    builder.initializeEsdfValues(&costmap, params, nullptr);
    const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
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
  EXPECT_NEAR(evaluate_cost(3.0), 1.5, 1e-9);
}

TEST(KinematicSmootherProblemBuilderTest, ApplyBoundsOnlyCapsUsedNonCuspDs)
{
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 0.0;
  params.keep_start_orientation = false;
  params.keep_goal_orientation = false;
  params.goal_longitudinal_tolerance = 2.0;
  params.goal_lateral_tolerance = 2.0;

  std::vector<double> esdf_values;
  kinematic_smoother::KinematicSmootherProblemBuilder builder(esdf_values);
  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d::UnitX(),
    Eigen::Vector2d::UnitX(),
    params,
    nullptr);

  std::vector<double> variables = processed.initial_variables;
  ceres::Problem problem;
  builder.buildProblem(processed, nullptr, params, variables, problem);
  kinematic_smoother::KinematicSmootherProblemBuilder::applyBounds(
    problem,
    variables.data(),
    processed.reference_points,
    processed.is_cusp_segment,
    processed.state_count,
    2.0,
    0.25,
    0.0);

  const double * first_state = kinematic_smoother::KinematicStateLayout::data(variables, 0);
  const double * last_state = kinematic_smoother::KinematicStateLayout::data(variables, 1);
  EXPECT_NEAR(
    problem.GetParameterLowerBound(first_state, kinematic_smoother::KinematicStateLayout::Ds),
    kinematic_smoother::KinematicStateLayout::GeometryEpsilon,
    1e-12);
  EXPECT_NEAR(
    problem.GetParameterUpperBound(first_state, kinematic_smoother::KinematicStateLayout::Ds),
    0.25,
    1e-12);
  EXPECT_NEAR(
    problem.GetParameterLowerBound(last_state, kinematic_smoother::KinematicStateLayout::Ds),
    0.0,
    1e-12);
  EXPECT_GT(
    problem.GetParameterUpperBound(last_state, kinematic_smoother::KinematicStateLayout::Ds),
    1e100);
}

TEST(KinematicSmootherProblemBuilderTest, UpsamplePathKinematicDistributesClosureError)
{
  kinematic_smoother::KinematicProcessedPath processed;
  processed.state_count = 2;
  processed.gears = {1.0};
  processed.is_cusp_segment = {false};

  std::vector<double> variables = {
    0.0, 0.0, 0.0, 0.0, 1.0,
    0.6, 0.4, 0.2, 0.0, 0.0,
  };

  kinematic_smoother::SmootherParams params;
  params.path_upsampling_factor = 4;

  const auto upsampled = kinematic_smoother::KinematicSmootherProblemBuilder::upsamplePathKinematic(
    variables,
    processed,
    params);

  ASSERT_EQ(upsampled.size(), 5u);

  const auto step_length = [&](size_t from, size_t to) {
    return (upsampled[to].head<2>() - upsampled[from].head<2>()).norm();
  };

  const double step_01 = step_length(0, 1);
  const double step_12 = step_length(1, 2);
  const double step_23 = step_length(2, 3);
  const double step_34 = step_length(3, 4);
  const double regular_step = std::max(step_01, std::max(step_12, step_23));

  EXPECT_LE(step_34, regular_step * 1.1);
  EXPECT_NEAR(upsampled[1].x(), 0.15, 1e-9);
  EXPECT_NEAR(upsampled[1].y(), 0.10, 1e-9);
  EXPECT_NEAR(upsampled[2].x(), 0.30, 1e-9);
  EXPECT_NEAR(upsampled[2].y(), 0.20, 1e-9);
  EXPECT_NEAR(upsampled[3].x(), 0.45, 1e-9);
  EXPECT_NEAR(upsampled[3].y(), 0.30, 1e-9);
}

TEST(KinematicSmootherProblemBuilderTest, OutputSpacingUpsamplesByMetricDistance)
{
  kinematic_smoother::KinematicProcessedPath processed;
  processed.state_count = 2;
  processed.gears = {1.0};
  processed.is_cusp_segment = {false};

  std::vector<double> variables = {
    0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0,
  };

  kinematic_smoother::SmootherParams params;
  params.path_output_spacing = 0.25;

  const auto upsampled = kinematic_smoother::KinematicSmootherProblemBuilder::upsamplePathKinematic(
    variables,
    processed,
    params);

  ASSERT_EQ(upsampled.size(), 5u);
  for (size_t index = 0; index + 1 < upsampled.size(); ++index) {
    EXPECT_NEAR((upsampled[index + 1].head<2>() - upsampled[index].head<2>()).norm(), 0.25, 1e-12);
  }
}

TEST(KinematicSmootherProblemBuilderTest, OutputProfileUpsamplesStraightCurvature)
{
  kinematic_smoother::KinematicProcessedPath processed;
  processed.state_count = 2;
  processed.gears = {1.0};
  processed.is_cusp_segment = {false};

  std::vector<double> variables = {
    0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0,
  };

  kinematic_smoother::SmootherParams params;
  params.path_output_spacing = 0.25;

  const auto profile =
    kinematic_smoother::KinematicSmootherProblemBuilder::upsamplePathKinematicProfile(
    variables,
    processed,
    params);

  ASSERT_EQ(profile.path.size(), 5u);
  ASSERT_EQ(profile.curvatures.size(), profile.path.size());
  ASSERT_EQ(profile.curvature_rates.size(), profile.path.size());
  for (size_t index = 0; index < profile.path.size(); ++index) {
    EXPECT_NEAR(profile.curvatures[index], 0.0, 1e-12);
    EXPECT_NEAR(profile.curvature_rates[index], 0.0, 1e-12);
  }
}

TEST(KinematicSmootherProblemBuilderTest, OutputProfileSamplesLinearCurvature)
{
  kinematic_smoother::KinematicProcessedPath processed;
  processed.state_count = 2;
  processed.gears = {1.0};
  processed.is_cusp_segment = {false};

  std::vector<double> variables = {
    0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.1, 0.2, 0.0,
  };

  kinematic_smoother::SmootherParams params;
  params.path_output_spacing = 0.25;

  const auto profile =
    kinematic_smoother::KinematicSmootherProblemBuilder::upsamplePathKinematicProfile(
    variables,
    processed,
    params);

  ASSERT_EQ(profile.path.size(), 5u);
  ASSERT_EQ(profile.curvatures.size(), profile.path.size());
  ASSERT_EQ(profile.curvature_rates.size(), profile.path.size());

  const std::vector<double> expected_curvatures = {0.0, 0.05, 0.10, 0.15, 0.20};
  for (size_t index = 0; index < profile.path.size(); ++index) {
    EXPECT_NEAR(profile.curvatures[index], expected_curvatures[index], 1e-12);
    EXPECT_NEAR(profile.curvature_rates[index], 0.2, 1e-12);
  }
}

TEST(KinematicSmootherProblemBuilderTest, OutputProfileLeavesCuspRateUndefined)
{
  kinematic_smoother::KinematicProcessedPath processed;
  processed.state_count = 2;
  processed.gears = {0.0};
  processed.is_cusp_segment = {true};

  std::vector<double> variables = {
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 2.0, 0.0,
  };

  kinematic_smoother::SmootherParams params;
  params.path_output_spacing = 0.25;

  const auto profile =
    kinematic_smoother::KinematicSmootherProblemBuilder::upsamplePathKinematicProfile(
    variables,
    processed,
    params);

  ASSERT_EQ(profile.path.size(), 2u);
  ASSERT_EQ(profile.curvature_rates.size(), profile.path.size());
  EXPECT_FALSE(std::isfinite(profile.curvature_rates[0]));
  EXPECT_FALSE(std::isfinite(profile.curvature_rates[1]));
}

TEST(KinematicSmootherProblemBuilderTest, PathTargetSpacingResamplesByMetricDistance)
{
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
    {2.0, 0.0, 1.0},
    {3.0, 0.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.path_target_spacing = 0.5;

  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d::UnitX(),
    Eigen::Vector2d::UnitX(),
    params,
    nullptr);

  ASSERT_EQ(processed.reference_points.size(), 7u);
  EXPECT_NEAR(processed.target_spacing, 0.5, 1e-12);
  for (size_t index = 0; index + 1 < processed.reference_points.size(); ++index) {
    EXPECT_NEAR(
      (processed.reference_points[index + 1] - processed.reference_points[index]).norm(),
      0.5,
      1e-12);
  }
}

TEST(KinematicSmootherProblemBuilderTest, PathTargetSpacingPreservesCuspPoint)
{
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
    {2.0, 0.0, -1.0},
    {3.0, 0.0, -1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.path_target_spacing = 0.75;

  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d::UnitX(),
    Eigen::Vector2d::UnitX(),
    params,
    nullptr);

  ASSERT_EQ(processed.gears.size(), processed.is_cusp_segment.size());
  const auto cusp_it = std::find(
    processed.is_cusp_segment.begin(),
    processed.is_cusp_segment.end(),
    true);
  ASSERT_NE(cusp_it, processed.is_cusp_segment.end());

  const size_t cusp_index = static_cast<size_t>(
    std::distance(processed.is_cusp_segment.begin(), cusp_it));
  EXPECT_NEAR(processed.reference_points[cusp_index + 1].x(), 2.0, 1e-12);
  EXPECT_NEAR(processed.reference_points[cusp_index + 1].y(), 0.0, 1e-12);
  EXPECT_EQ(processed.gears[cusp_index], 0.0);
  EXPECT_NEAR(processed.target_spacing, 0.75, 1e-12);
}

TEST(KinematicSmootherProblemBuilderTest, PathTargetSpacingPreservesDuplicateStartGearChange)
{
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.0, 0.0, -1.0},
    {-1.0, 0.0, -1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.path_target_spacing = 0.5;

  const auto processed = kinematic_smoother::KinematicSmootherProblemBuilder::buildProcessedPath(
    path,
    Eigen::Vector2d::UnitX(),
    -Eigen::Vector2d::UnitX(),
    params,
    nullptr);

  ASSERT_EQ(processed.gears.size(), processed.is_cusp_segment.size());
  ASSERT_GE(processed.state_count, 3u);
  EXPECT_DOUBLE_EQ(processed.gears.front(), 0.0);
  EXPECT_TRUE(processed.is_cusp_segment.front());
  EXPECT_NEAR(processed.reference_points[0].x(), 0.0, 1e-12);
  EXPECT_NEAR(processed.reference_points[1].x(), 0.0, 1e-12);
  EXPECT_NEAR(processed.reference_points[0].y(), 0.0, 1e-12);
  EXPECT_NEAR(processed.reference_points[1].y(), 0.0, 1e-12);
  EXPECT_EQ(processed.gears[1], -1.0);
  EXPECT_FALSE(processed.is_cusp_segment[1]);
  EXPECT_NEAR(processed.target_spacing, 0.5, 1e-12);
}

// ---- Stable error-code and failure-message contract tests ----

TEST(ErrorTest, InvalidPathCarriesStableCode)
{
  const kinematic_smoother::InvalidPath error("test invalid path");

  EXPECT_EQ(error.code(), kinematic_smoother::ErrorCode::InvalidPath);
  EXPECT_STREQ(error.codeString(), "CS_INVALID_PATH");
  EXPECT_STREQ(error.what(), std::string("test invalid path").c_str());
}

TEST(ErrorTest, SmoothingFailureMessageCarriesReasonAndIndex)
{
  const std::string message = kinematic_smoother::buildSmoothingFailureMessage(
    kinematic_smoother::SmoothingFailureReason::GoalOrientationConstraint,
    "test smoothing failure",
    7);

  EXPECT_EQ(message, "goal_orientation_constraint@7: test smoothing failure");
}

TEST(KinematicSmootherTest, InvalidOptimizerIterationLimitThrowsInvalidArgument)
{
  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 0;

  kinematic_smoother::KinematicSmoother smoother;

  EXPECT_THROW(smoother.initialize(opt_params), std::invalid_argument);
}

TEST(KinematicSmootherTest, InvalidOptimizerToleranceThrowsInvalidArgument)
{
  kinematic_smoother::OptimizerParams opt_params;
  opt_params.parameter_tolerance = -1.0;

  kinematic_smoother::KinematicSmoother smoother;

  EXPECT_THROW(smoother.initialize(opt_params), std::invalid_argument);

  opt_params.parameter_tolerance = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(smoother.initialize(opt_params), std::invalid_argument);
}

// ---- Kinematic smoother behavior and error-surface tests ----

TEST(KinematicSmootherTest, SmoothStraightPath)
{
  kinematic_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path;
  for (int i = 0; i < 10; ++i) {
    const double x = 0.5 + i * 0.1;
    const double y = 2.5 + (i == 5 ? 0.04 : 0.0);
    path.emplace_back(x, y, 1.0);
  }

  kinematic_smoother::SmootherParams params;
  params.model_weight = 20.0;
  params.obstacle_weight = 0.5;
  params.reference_path_weight = 1.0;
  params.kinematic_curvature_weight = 30.0;
  params.kinematic_curvature_rate_weight = 5.0;
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.obstacle_safe_distance = 0.5;

  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 30;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);
  const auto input_path = path;

  const auto result = smoother.smooth({path, start_dir, end_dir, &costmap, params, nullptr, nullptr});

  EXPECT_TRUE(result.success);
  EXPECT_FALSE(result.candidate_path.empty());
  EXPECT_GE(result.smoothed_path.size(), 2u);
  EXPECT_EQ(result.smoothed_curvatures.size(), result.smoothed_path.size());
  EXPECT_EQ(result.smoothed_curvature_rates.size(), result.smoothed_path.size());
  EXPECT_GT(result.optimized_knot_count, 0u);
  EXPECT_GT(result.target_spacing, 0.0);
  expectPathsNear(path, input_path);
}

TEST(KinematicSmootherTest, ReferencePointMaxDeviationDefaultsOffAndBoundsOptimizedPoint)
{
  kinematic_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);

  const std::vector<Eigen::Vector3d> reference_path = {
    {0.5, 0.5, 1.0},
    {1.0, 1.0, 1.0},
    {1.5, 0.5, 1.0},
    {2.0, 0.5, 1.0},
  };

  auto run_case = [&](double max_deviation) {
      std::vector<Eigen::Vector3d> path = reference_path;

      kinematic_smoother::SmootherParams params;
      params.model_weight = 20.0;
      params.obstacle_weight = 0.0;
      params.reference_path_weight = 0.0;
      params.kinematic_curvature_weight = 0.0;
      params.kinematic_curvature_rate_weight = 0.0;
      params.path_length_weight = 5.0;
      params.max_curvature = 10.0;
      params.max_time = 1.0;
      params.keep_start_orientation = false;
      params.keep_goal_orientation = false;
      params.reference_point_max_deviation_m = max_deviation;

      kinematic_smoother::OptimizerParams opt_params;
      opt_params.max_iterations = 60;

      kinematic_smoother::KinematicSmoother smoother;
      smoother.initialize(opt_params);
      const auto input_path = path;
      const auto result = smoother.smooth(
        {path, Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(1.0, 0.0), &costmap, params, nullptr, nullptr});
      EXPECT_TRUE(result.success);
      expectPathsNear(path, input_path);
      return result.smoothed_path;
    };

  const auto unbounded = run_case(0.0);
  const auto bounded = run_case(0.1);

  EXPECT_GT(std::abs(unbounded[1].y() - reference_path[1].y()), 0.100001);
  EXPECT_LE(std::abs(bounded[1].y() - reference_path[1].y()), 0.100001);
}

TEST(KinematicSmootherTest, SmoothCuspPath)
{
  kinematic_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path;
  constexpr double spacing = 0.2;
  for (double x = 1.0; x <= 6.0 + 1e-9; x += spacing) {
    path.emplace_back(x, 2.0, 1.0);
  }
  path.emplace_back(6.0, 2.0, -1.0);
  for (double x = 6.0 - spacing; x >= 1.4 - 1e-9; x -= spacing) {
    path.emplace_back(x, 2.0, -1.0);
  }
  const auto input_size = path.size();

  kinematic_smoother::SmootherParams params;
  params.model_weight = 20.0;
  params.obstacle_weight = 0.0;
  params.reference_path_weight = 0.0;
  params.kinematic_curvature_weight = 30.0;
  params.kinematic_curvature_rate_weight = 5.0;
  params.path_length_weight = 0.0;
  params.max_curvature = 100.0;
  params.max_time = 1.0;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 40;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);
  const auto input_path = path;

  const auto result = smoother.smooth({path, start_dir, end_dir, &costmap, params, nullptr, nullptr});

  EXPECT_TRUE(result.success);
  EXPECT_GE(result.smoothed_path.size(), 2u);
  expectPathsNear(path, input_path);
  EXPECT_GE(result.optimized_knot_count, input_size);
}

TEST(KinematicSmootherTest, NullCostmapAllowedWhenObstacleTermsDisabled)
{
  std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
  };

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 0.0;
  kinematic_smoother::OptimizerParams opt_params;
  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);
  const auto input_path = path;

  const auto result = smoother.smooth({path, start_dir, end_dir, nullptr, params, nullptr, nullptr});

  EXPECT_TRUE(result.success);
  expectPathsNear(path, input_path);
}

TEST(KinematicSmootherTest, NullCostmapStillRejectedWhenObstacleTermsEnabled)
{
  std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
  };

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 1.0;
  kinematic_smoother::OptimizerParams opt_params;
  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  EXPECT_THROW(
    (void)smoother.smooth({path, start_dir, end_dir, nullptr, params, nullptr, nullptr}),
    kinematic_smoother::InvalidCostmap);
}

TEST(KinematicSmootherTest, NonFinitePathPointThrowsInvalidPath)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {nan, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 0.0;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(kinematic_smoother::OptimizerParams{});

  EXPECT_THROW(
    (void)smoother.smooth(
      {path, Eigen::Vector2d::UnitX(), Eigen::Vector2d::UnitX(), nullptr, params, nullptr, nullptr}),
    kinematic_smoother::InvalidPath);
}

TEST(KinematicSmootherTest, NonFiniteEndpointDirectionThrowsInvalidPath)
{
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 0.0;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(kinematic_smoother::OptimizerParams{});

  EXPECT_THROW(
    (void)smoother.smooth(
      {
        path,
        Eigen::Vector2d(infinity, 0.0),
        Eigen::Vector2d::UnitX(),
        nullptr,
        params,
        nullptr,
        nullptr,
      }),
    kinematic_smoother::InvalidPath);
}

TEST(KinematicSmootherTest, NonFiniteSmootherParamThrowsInvalidArgument)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 0.0;
  params.model_weight = nan;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(kinematic_smoother::OptimizerParams{});

  EXPECT_THROW(
    (void)smoother.smooth(
      {path, Eigen::Vector2d::UnitX(), Eigen::Vector2d::UnitX(), nullptr, params, nullptr, nullptr}),
    std::invalid_argument);
}

TEST(KinematicSmootherTest, InvalidCostCheckPointsShapeThrowsInvalidArgument)
{
  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 0.0;
  params.cost_check_points = {0.0, 0.0};

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(kinematic_smoother::OptimizerParams{});

  EXPECT_THROW(
    (void)smoother.smooth(
      {path, Eigen::Vector2d::UnitX(), Eigen::Vector2d::UnitX(), nullptr, params, nullptr, nullptr}),
    std::invalid_argument);
}

TEST(KinematicSmootherTest, NonFiniteCostCheckPointThrowsInvalidArgument)
{
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 0.0;
  params.cost_check_points = {0.0, infinity, 1.0};

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(kinematic_smoother::OptimizerParams{});

  EXPECT_THROW(
    (void)smoother.smooth(
      {path, Eigen::Vector2d::UnitX(), Eigen::Vector2d::UnitX(), nullptr, params, nullptr, nullptr}),
    std::invalid_argument);
}

TEST(KinematicSmootherTest, ObstacleCostCheckPointsDoNotThrow)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);
  for (unsigned int y = 25; y < 55; ++y) {
    for (unsigned int x = 35; x < 45; ++x) {
      costmap.setCost(x, y, kinematic_smoother::Costmap2D::LETHAL_OBSTACLE);
    }
  }

  std::vector<Eigen::Vector3d> path = {
    {1.0, 3.2, 1.0},
    {1.5, 3.2, 1.0},
    {2.0, 3.2, 1.0},
    {2.5, 3.2, 1.0},
    {3.0, 3.2, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.model_weight = 20.0;
  params.obstacle_weight = 1.0;
  params.reference_path_weight = 1.0;
  params.kinematic_curvature_weight = 10.0;
  params.kinematic_curvature_rate_weight = 5.0;
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

  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);
  const auto input_path = path;

  const auto result = smoother.smooth({path, start_dir, end_dir, &costmap, params, nullptr, nullptr});

  EXPECT_TRUE(result.success);
  expectPathsNear(path, input_path);
  EXPECT_GT(result.optimized_knot_count, 0u);
}

TEST(KinematicSmootherTest, GoalOrientationCannotSilentlyFlipIntoReverse)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
    {2.5, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.model_weight = 20.0;
  params.obstacle_weight = 1e-4;
  params.reference_path_weight = 0.0;
  params.kinematic_curvature_weight = 30.0;
  params.kinematic_curvature_rate_weight = 5.0;
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 40;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(-1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {(void)smoother.smooth({path, start_dir, end_dir, &costmap, params, nullptr, nullptr});});

  EXPECT_NE(error_message.find("goal_orientation_constraint@"), std::string::npos);
}

TEST(SmootherValidatorTest, KinematicGoalOrientationUsesGoalStateHeading)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

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

  kinematic_smoother::SmootherParams params;
  params.keep_goal_orientation = true;
  params.keep_start_orientation = false;
  params.max_curvature = 10.0;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  kinematic_smoother::SmoothingFailureInfo failure;
  kinematic_smoother::SmootherValidator validator;

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
  EXPECT_EQ(failure.reason, kinematic_smoother::SmoothingFailureReason::Unknown);
  EXPECT_EQ(failure.failed_index, -1);
  EXPECT_TRUE(failure.message.empty());
}

TEST(SmootherValidatorTest, KinematicGoalOrientationDefaultToleranceIsStrict)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  const std::vector<double> variables = {
    1.0, 2.0, 0.0, 0.0, 0.5,
    1.5, 2.0, 0.05, 0.0, 0.0,
  };
  const std::vector<Eigen::Vector2d> reference_points = {
    {1.0, 2.0},
    {1.5, 2.0},
  };
  const std::vector<double> gears = {1.0};
  const std::vector<bool> is_cusp_segment = {false};

  kinematic_smoother::SmootherParams params;
  params.keep_goal_orientation = true;
  params.keep_start_orientation = true;
  params.max_curvature = 10.0;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  kinematic_smoother::SmoothingFailureInfo failure;
  kinematic_smoother::SmootherValidator validator;

  EXPECT_FALSE(validator.validateKinematicSolution(
      {
        variables,
        reference_points,
        gears,
        is_cusp_segment,
        2,
        0.0,
        0.0,
        &costmap,
        params,
        esdf_values,
      },
      &failure));
  EXPECT_EQ(failure.reason, kinematic_smoother::SmoothingFailureReason::GoalOrientationConstraint);
  EXPECT_EQ(failure.failed_index, 1);
}

TEST(SmootherValidatorTest, KinematicGoalOrientationHonorsConfiguredTolerance)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  const std::vector<double> variables = {
    1.0, 2.0, 0.0, 0.0, 0.5,
    1.5, 2.0, 0.05, 0.0, 0.0,
  };
  const std::vector<Eigen::Vector2d> reference_points = {
    {1.0, 2.0},
    {1.5, 2.0},
  };
  const std::vector<double> gears = {1.0};
  const std::vector<bool> is_cusp_segment = {false};

  kinematic_smoother::SmootherParams params;
  params.keep_goal_orientation = true;
  params.keep_start_orientation = true;
  params.goal_orientation_tolerance = 0.06;
  params.max_curvature = 10.0;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  kinematic_smoother::SmoothingFailureInfo failure;
  kinematic_smoother::SmootherValidator validator;

  EXPECT_TRUE(validator.validateKinematicSolution(
      {
        variables,
        reference_points,
        gears,
        is_cusp_segment,
        2,
        0.0,
        0.0,
        &costmap,
        params,
        esdf_values,
      },
      &failure));
  EXPECT_EQ(failure.reason, kinematic_smoother::SmoothingFailureReason::Unknown);
}

TEST(SmootherValidatorTest, KinematicGoalPositionToleranceAllowsGoalSlack)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

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

  kinematic_smoother::SmootherParams params;
  params.keep_goal_orientation = true;
  params.keep_start_orientation = true;
  params.goal_longitudinal_tolerance = 0.2;
  params.goal_lateral_tolerance = 0.1;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  kinematic_smoother::SmoothingFailureInfo failure;
  kinematic_smoother::SmootherValidator validator;

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
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

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

  kinematic_smoother::SmootherParams params;
  params.keep_goal_orientation = false;
  params.keep_start_orientation = true;
  params.goal_longitudinal_tolerance = 0.2;
  params.goal_lateral_tolerance = 0.0;
  params.max_curvature = 10.0;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  kinematic_smoother::SmoothingFailureInfo failure;
  kinematic_smoother::SmootherValidator validator;

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
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

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

  kinematic_smoother::SmootherParams params;
  params.keep_goal_orientation = true;
  params.keep_start_orientation = true;
  params.goal_longitudinal_tolerance = 0.2;
  params.goal_lateral_tolerance = 0.1;

  const std::vector<double> esdf_values(costmap.getSizeInCellsX() * costmap.getSizeInCellsY(), 1.0);
  kinematic_smoother::SmoothingFailureInfo failure;
  kinematic_smoother::SmootherValidator validator;

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
  EXPECT_EQ(failure.reason, kinematic_smoother::SmoothingFailureReason::GoalPositionConstraint);
  EXPECT_NE(failure.message.find("goal position tolerance box"), std::string::npos);
  EXPECT_NEAR(failure.goal_longitudinal_error, 0.25, 1e-9);
  EXPECT_NEAR(failure.goal_lateral_error, -0.05, 1e-9);
  EXPECT_NEAR(failure.goal_longitudinal_tolerance, 0.2, 1e-9);
  EXPECT_NEAR(failure.goal_lateral_tolerance, 0.1, 1e-9);
}

TEST(KinematicSmootherTest, MotionDirectionViolationStoresFailureInfoWithoutThrowing)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
    {2.5, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.model_weight = 20.0;
  params.obstacle_weight = 1e-4;
  params.reference_path_weight = 0.0;
  params.kinematic_curvature_weight = 30.0;
  params.kinematic_curvature_rate_weight = 5.0;
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;

  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 40;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(-1.0, 0.0);
  kinematic_smoother::SmoothingFailureInfo failure;
  const auto input_path = path;

  const auto result = smoother.smooth({path, start_dir, end_dir, &costmap, params, nullptr, &failure});

  EXPECT_FALSE(result.success);
  EXPECT_FALSE(result.candidate_path.empty());
  expectPathsNear(path, input_path);
  EXPECT_EQ(failure.reason, kinematic_smoother::SmoothingFailureReason::GoalOrientationConstraint);
  EXPECT_GE(failure.failed_index, 0);
  EXPECT_NE(failure.message.find("goal orientation"), std::string::npos);
}

TEST(KinematicSmootherTest, FootprintCollisionFailsPostValidation)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);
  for (unsigned int y = 35; y < 45; ++y) {
    for (unsigned int x = 36; x < 42; ++x) {
      costmap.setCost(x, y, kinematic_smoother::Costmap2D::LETHAL_OBSTACLE);
    }
  }

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.model_weight = 20.0;
  params.obstacle_weight = 1e-4;
  params.reference_path_weight = 0.0;
  params.kinematic_curvature_weight = 30.0;
  params.kinematic_curvature_rate_weight = 5.0;
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.18;
  params.cost_check_points = {0.0, 0.0, 1.0};

  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {(void)smoother.smooth({path, start_dir, end_dir, &costmap, params, nullptr, nullptr});});

  EXPECT_NE(error_message.find("footprint_collision@"), std::string::npos);
}

TEST(SmootherValidatorTest, ObstacleSafeDistanceIsSoftAndDoesNotFailPostValidation)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  const std::vector<double> variables = {
    1.0, 1.0, 0.0, 0.0, 0.5,
    1.5, 1.0, 0.0, 0.0, 0.0,
  };
  const std::vector<Eigen::Vector2d> reference_points = {
    {1.0, 1.0},
    {1.5, 1.0},
  };
  const std::vector<double> gears = {1.0};
  const std::vector<bool> is_cusp_segment = {false};

  kinematic_smoother::SmootherParams params;
  params.obstacle_weight = 1.0;
  params.cost_check_radius = 0.10;
  params.obstacle_safe_distance = 0.50;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;
  params.max_curvature = 10.0;

  // Clearance is greater than the footprint radius, so there is no collision,
  // but it is less than radius + obstacle_safe_distance. That soft margin should
  // affect optimization cost only, not post-validation acceptance.
  const std::vector<double> esdf_values(
    costmap.getSizeInCellsX() * costmap.getSizeInCellsY(),
    0.25);
  kinematic_smoother::SmoothingFailureInfo failure;
  kinematic_smoother::SmootherValidator validator;

  EXPECT_TRUE(validator.validateKinematicSolution(
      {
        variables,
        reference_points,
        gears,
        is_cusp_segment,
        2,
        0.0,
        0.0,
        &costmap,
        params,
        esdf_values,
      },
      &failure));
  EXPECT_EQ(failure.reason, kinematic_smoother::SmoothingFailureReason::Unknown);
  EXPECT_EQ(failure.failed_index, -1);
  EXPECT_TRUE(failure.message.empty());
}

TEST(KinematicSmootherTest, FootprintRadiusWithoutCheckpointsFailsPostValidation)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);
  for (unsigned int y = 35; y < 45; ++y) {
    for (unsigned int x = 36; x < 42; ++x) {
      costmap.setCost(x, y, kinematic_smoother::Costmap2D::LETHAL_OBSTACLE);
    }
  }

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.model_weight = 20.0;
  params.obstacle_weight = 1e-4;
  params.reference_path_weight = 0.0;
  params.kinematic_curvature_weight = 30.0;
  params.kinematic_curvature_rate_weight = 5.0;
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.18;

  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {(void)smoother.smooth({path, start_dir, end_dir, &costmap, params, nullptr, nullptr});});

  EXPECT_NE(error_message.find("footprint_collision@"), std::string::npos);
}

TEST(KinematicSmootherTest, PathOutOfBoundsFailsPostValidation)
{
  kinematic_smoother::Costmap2D costmap(80, 80, 0.05, 0.0, 0.0);

  std::vector<Eigen::Vector3d> path = {
    {1.0, 2.0, 1.0},
    {1.5, 2.0, 1.0},
    {2.0, 2.0, 1.0},
  };

  kinematic_smoother::SmootherParams params;
  params.model_weight = 20.0;
  params.obstacle_weight = 1e-4;
  params.reference_path_weight = 0.0;
  params.kinematic_curvature_weight = 30.0;
  params.kinematic_curvature_rate_weight = 5.0;
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;
  params.cost_check_radius = 0.1;
  params.cost_check_points = {4.0, 0.0, 1.0};

  kinematic_smoother::OptimizerParams opt_params;
  opt_params.max_iterations = 20;

  kinematic_smoother::KinematicSmoother smoother;
  smoother.initialize(opt_params);

  const Eigen::Vector2d start_dir(1.0, 0.0);
  const Eigen::Vector2d end_dir(1.0, 0.0);

  const std::string error_message = expectFailedToSmoothPath(
    [&]() {(void)smoother.smooth({path, start_dir, end_dir, &costmap, params, nullptr, nullptr});});

  EXPECT_NE(error_message.find("path_out_of_bounds@"), std::string::npos);
}

// ---- Basic costmap sanity test ----

TEST(CostmapTest, BasicCostmapOperations)
{
  kinematic_smoother::Costmap2D costmap(10, 10, 0.05, 1.0, 2.0);
  EXPECT_EQ(costmap.getSizeInCellsX(), 10u);
  EXPECT_EQ(costmap.getSizeInCellsY(), 10u);
  EXPECT_DOUBLE_EQ(costmap.getResolution(), 0.05);
  EXPECT_DOUBLE_EQ(costmap.getOriginX(), 1.0);
  EXPECT_DOUBLE_EQ(costmap.getOriginY(), 2.0);

  costmap.setCost(3, 4, 128);
  EXPECT_EQ(costmap.getCost(3, 4), 128);

  EXPECT_NE(costmap.getCharMap(), nullptr);
}
