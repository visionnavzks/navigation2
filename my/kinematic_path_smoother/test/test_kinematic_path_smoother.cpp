#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "kinematic_path_smoother/kinematic_smoother.hpp"
#include "kinematic_path_smoother/kinematic_smoother_costs.hpp"
#include "kinematic_path_smoother/kinematic_smoother_problem_builder.hpp"

namespace kps = kinematic_path_smoother;

TEST(KinematicPathSmootherBuilder, InsertsExplicitCuspState)
{
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, -1.0},
    {0.5, 0.0, -1.0},
  };

  kps::SmootherParams params;
  const auto processed = kps::KinematicSmootherProblemBuilder::buildProcessedPath(
    path, {1.0, 0.0}, {-1.0, 0.0}, params, nullptr);

  ASSERT_EQ(processed.size, 4u);
  ASSERT_EQ(processed.gears.size(), 3u);
  ASSERT_EQ(processed.cusp_segments.size(), 3u);
  EXPECT_DOUBLE_EQ(processed.gears[0], 1.0);
  EXPECT_DOUBLE_EQ(processed.gears[1], 0.0);
  EXPECT_DOUBLE_EQ(processed.gears[2], -1.0);
  EXPECT_FALSE(processed.cusp_segments[0]);
  EXPECT_TRUE(processed.cusp_segments[1]);
  EXPECT_FALSE(processed.cusp_segments[2]);
  EXPECT_EQ(processed.variables.size(), processed.size * 5);
}

TEST(KinematicPathSmootherBuilder, ReversingCanBeDisabled)
{
  const std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {1.0, 0.0, -1.0},
    {2.0, 0.0, -1.0},
  };

  kps::SmootherParams params;
  params.reversing_enabled = false;
  const auto processed = kps::KinematicSmootherProblemBuilder::buildProcessedPath(
    path, {1.0, 0.0}, {1.0, 0.0}, params, nullptr);

  ASSERT_EQ(processed.size, 3u);
  EXPECT_EQ(processed.cusp_segments, std::vector<bool>({false, false}));
  EXPECT_EQ(processed.gears, std::vector<double>({1.0, 1.0}));
}

TEST(KinematicPathSmootherCosts, MotionCostSeparatesCurvatureTerms)
{
  kps::detail::MotionCost curvature_cost(1.0, false, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  const double current[5] = {0.0, 0.0, 0.0, 2.0, 1.0};
  const double next[5] = {0.0, 0.0, 0.0, 2.0, 0.0};
  double residuals[7] = {};
  ASSERT_TRUE(curvature_cost(current, next, residuals));
  EXPECT_DOUBLE_EQ(residuals[3], 6.0);
  EXPECT_DOUBLE_EQ(residuals[4], 0.0);

  kps::detail::MotionCost rate_cost(1.0, false, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0);
  const double rate_current[5] = {0.0, 0.0, 0.0, 1.0, 4.0};
  const double rate_next[5] = {0.0, 0.0, 0.0, 3.0, 0.0};
  double rate_residuals[7] = {};
  ASSERT_TRUE(rate_cost(rate_current, rate_next, rate_residuals));
  EXPECT_DOUBLE_EQ(rate_residuals[3], 0.0);
  EXPECT_DOUBLE_EQ(rate_residuals[4], 4.0);
}

TEST(KinematicPathSmootherCosts, EndpointCostUsesGoalFrameTolerance)
{
  kps::detail::EndpointCost cost(
    {0.0, 0.0}, M_PI / 2.0, true, 0.2, 0.1, 0.05, 10.0, false);

  double inside[5] = {0.05, 0.15, M_PI / 2.0 + 0.03, 0.0, 0.0};
  double inside_residuals[4] = {};
  ASSERT_TRUE(cost(inside, inside_residuals));
  EXPECT_DOUBLE_EQ(inside_residuals[0], 0.0);
  EXPECT_DOUBLE_EQ(inside_residuals[1], 0.0);
  EXPECT_DOUBLE_EQ(inside_residuals[2], 0.0);

  double outside[5] = {0.05, 0.25, M_PI / 2.0 + 0.07, 0.0, 0.0};
  double outside_residuals[4] = {};
  ASSERT_TRUE(cost(outside, outside_residuals));
  EXPECT_NEAR(outside_residuals[0], 0.5, 1e-9);
  EXPECT_DOUBLE_EQ(outside_residuals[1], 0.0);
  EXPECT_NEAR(outside_residuals[2], 0.2, 1e-9);
}

TEST(KinematicPathSmootherCosts, ObstacleCostReturnsHingeResidual)
{
  std::vector<unsigned char> costs(25, 0);
  kps::Costmap2D costmap(5, 5, 1.0, 0.0, 0.0, costs.data());

  std::vector<double> esdf(25, 0.25);
  auto grid = std::make_shared<ceres::Grid2D<double>>(esdf.data(), 0, 5, 0, 5);
  auto interpolator = std::make_shared<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>(*grid);

  kps::SmootherParams params;
  params.obstacle_weight = 4.0;
  params.cusp_obstacle_weight = 4.0;
  params.obstacle_safe_distance = 1.0;
  params.footprint_radius = 0.0;

  kps::detail::ObstacleCost cost(false, costmap, params, grid, interpolator);
  const double state[5] = {2.0, 2.0, 0.0, 0.0, 0.0};
  const double * blocks[1] = {state};
  double residuals[1] = {};

  ASSERT_TRUE(cost(blocks, residuals));
  EXPECT_NEAR(residuals[0], 1.5, 1e-9);
}

TEST(KinematicPathSmoother, SmoothsStraightPath)
{
  std::vector<Eigen::Vector3d> path = {
    {0.0, 0.0, 1.0},
    {0.5, 0.0, 1.0},
    {1.0, 0.0, 1.0},
  };

  kps::OptimizerParams optimizer;
  optimizer.linear_solver = kps::OptimizerParams::LinearSolver::DenseQr;

  kps::SmootherParams params;
  params.model_weight = 1.0;
  params.spacing_weight = 1.0;
  params.fix_weight = 100.0;
  params.max_curvature = 2.0;
  params.path_upsampling_factor = 3;

  kps::KinematicPathSmoother smoother;
  smoother.initialize(optimizer);
  kps::FailureInfo failure;
  const kps::SmoothingRequest request{
    path,
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
    nullptr,
    params,
    nullptr,
    &failure};

  const auto result = smoother.smooth(request);
  ASSERT_TRUE(result.success) << failure.formattedMessage();
  ASSERT_EQ(result.optimized_path.size(), 3u);
  ASSERT_EQ(result.path.size(), 7u);
  EXPECT_NEAR(result.path.front().x(), 0.0, 1e-6);
  EXPECT_NEAR(result.path.back().x(), 1.0, 1e-6);
  EXPECT_NEAR(result.path.back().z(), 0.0, 1e-6);
}
