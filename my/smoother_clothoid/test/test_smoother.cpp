#include <vector>
#include <cmath>
#include <string>

#include "smoother_clothoid/problem_builder.hpp"
#include "smoother_clothoid/smoother.hpp"
#include "smoother_clothoid/validator.hpp"
#include "gtest/gtest.h"

namespace
{

template<typename CallableT>
std::string expectFailed(CallableT && callable)
{
  try { callable(); }
  catch (const smoother_clothoid::FailedToSmoothPath & e) { return e.what(); }
  catch (const std::exception & e) { ADD_FAILURE() << "Expected FailedToSmoothPath, got: " << e.what(); return ""; }
  ADD_FAILURE() << "Expected FailedToSmoothPath to be thrown";
  return "";
}

void expectNear(
  const std::vector<Eigen::Vector3d> & a, const std::vector<Eigen::Vector3d> & b, double tol = 1e-9)
{
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_TRUE(a[i].isApprox(b[i], tol)) << "mismatch at " << i;
}

}  // namespace

TEST(ProcessedPathTest, InsertsCuspState)
{
  smoother_clothoid::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {{0,0,1},{1,0,-1},{0.5,0,-1}};
  smoother_clothoid::SmootherParams params;
  params.keep_start_orientation = true;
  params.keep_goal_orientation = true;
  const auto p = smoother_clothoid::ProblemBuilder::buildProcessedPath(
    path, {1,0}, {-1,0}, params, &costmap);
  EXPECT_EQ(p.state_count, 4u);
  EXPECT_DOUBLE_EQ(p.gears[0], 1.0);
  EXPECT_DOUBLE_EQ(p.gears[1], 0.0);
  EXPECT_DOUBLE_EQ(p.gears[2], -1.0);
  EXPECT_FALSE(p.is_cusp_segment[0]);
  EXPECT_TRUE(p.is_cusp_segment[1]);
  EXPECT_FALSE(p.is_cusp_segment[2]);
}

TEST(ProcessedPathTest, DisabledReversing)
{
  smoother_clothoid::Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> path = {{0,0,1},{1,0,-1},{0.5,0,-1}};
  smoother_clothoid::SmootherParams params;
  params.reversing_enabled = false;
  const auto p = smoother_clothoid::ProblemBuilder::buildProcessedPath(
    path, {1,0}, {1,0}, params, &costmap);
  EXPECT_EQ(p.state_count, 3u);
  EXPECT_DOUBLE_EQ(p.gears[0], 1.0);
  EXPECT_DOUBLE_EQ(p.gears[1], 1.0);
}

TEST(CostTest, TransitionCostKeepsWeightsIndependent)
{
  smoother_clothoid::detail::TransitionCostFunctor cc(1.0, false, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  const double c[5] = {0,0,0,2,1}, n[5] = {0,0,0,2,0};
  double r[7] = {};
  EXPECT_TRUE(cc(c, n, r));
  EXPECT_DOUBLE_EQ(r[3], 6.0);
  EXPECT_DOUBLE_EQ(r[4], 0.0);
  EXPECT_DOUBLE_EQ(r[6], 0.0);
}

TEST(CostTest, BoundaryCostUsesTolerances)
{
  smoother_clothoid::detail::BoundaryCostFunctor gc({0,0}, M_PI/2, true, 0.2, 0.1, 0.05, 10.0, false);
  double s1[5] = {0.05, 0.15, M_PI/2+0.03, 0, 0};
  double r1[4] = {};
  EXPECT_TRUE(gc(s1, r1));
  EXPECT_DOUBLE_EQ(r1[0], 0.0);
  EXPECT_DOUBLE_EQ(r1[1], 0.0);

  double s2[5] = {0.05, 0.25, M_PI/2+0.07, 0, 0};
  double r2[4] = {};
  EXPECT_TRUE(gc(s2, r2));
  EXPECT_NEAR(r2[0], 0.5, 1e-9);
}

TEST(ErrorTest, InvalidPathCarriesCode)
{
  const smoother_clothoid::InvalidPath e("test");
  EXPECT_EQ(e.code(), smoother_clothoid::ErrorCode::InvalidPath);
  EXPECT_STREQ(e.codeString(), "SC_INVALID_PATH");
}

TEST(ErrorTest, FailureMessageFormat)
{
  const auto m = smoother_clothoid::buildSmoothingFailureMessage(
    smoother_clothoid::SmoothingFailureReason::GoalOrientationConstraint, "test", 7);
  EXPECT_EQ(m, "goal_orientation_constraint@7: test");
}

TEST(SmootherTest, SmoothStraightPath)
{
  smoother_clothoid::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);
  std::vector<Eigen::Vector3d> path;
  for (int i = 0; i < 10; ++i) path.emplace_back(0.5 + i * 0.1, 2.5, 1.0);

  smoother_clothoid::SmootherParams params;
  params.model_weight_sqrt = std::sqrt(20.0);
  params.reference_path_weight_sqrt = std::sqrt(1.0);
  params.kinematic_curvature_weight_sqrt = std::sqrt(30.0);
  params.kinematic_curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 1.0 / 0.4;
  params.max_time = 1.0;

  smoother_clothoid::OptimizerParams opt;
  opt.max_iterations = 30;

  smoother_clothoid::ClothoidSmoother smoother;
  smoother.initialize(opt);

  const auto input = path;
  const auto result = smoother.smooth({path, {1,0}, {1,0}, &costmap, params, nullptr, nullptr});
  EXPECT_TRUE(result.success);
  EXPECT_FALSE(result.candidate_path.empty());
  EXPECT_GE(result.smoothed_path.size(), 2u);
  expectNear(path, input);
}

TEST(SmootherTest, NullCostmapAllowedWhenNoObstacles)
{
  std::vector<Eigen::Vector3d> path = {{0,0,1},{0.5,0,1}};
  smoother_clothoid::SmootherParams params;
  smoother_clothoid::OptimizerParams opt;
  smoother_clothoid::ClothoidSmoother smoother;
  smoother.initialize(opt);
  const auto input = path;
  const auto result = smoother.smooth({path, {1,0}, {1,0}, nullptr, params, nullptr, nullptr});
  EXPECT_TRUE(result.success);
  expectNear(path, input);
}

TEST(SmootherTest, NullCostmapRejectedWhenObstaclesEnabled)
{
  std::vector<Eigen::Vector3d> path = {{0,0,1},{0.5,0,1}};
  smoother_clothoid::SmootherParams params;
  params.costmap_weight_sqrt = 1.0;
  smoother_clothoid::OptimizerParams opt;
  smoother_clothoid::ClothoidSmoother smoother;
  smoother.initialize(opt);
  EXPECT_THROW((void)smoother.smooth({path, {1,0}, {1,0}, nullptr, params, nullptr, nullptr}),
    smoother_clothoid::InvalidCostmap);
}

TEST(SmootherTest, CuspPathSmooths)
{
  smoother_clothoid::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);
  std::vector<Eigen::Vector3d> path;
  constexpr double sp = 0.2;
  for (double x = 1.0; x <= 6.0 + 1e-9; x += sp) path.emplace_back(x, 2.0, 1.0);
  for (double x = 6.0 - sp; x >= 1.4 - 1e-9; x -= sp) path.emplace_back(x, 2.0, -1.0);
  const auto input = path;

  smoother_clothoid::SmootherParams params;
  params.model_weight_sqrt = std::sqrt(20.0);
  params.kinematic_curvature_weight_sqrt = std::sqrt(30.0);
  params.kinematic_curvature_rate_weight_sqrt = std::sqrt(5.0);
  params.max_curvature = 100.0;
  params.max_time = 1.0;

  smoother_clothoid::OptimizerParams opt;
  opt.max_iterations = 40;

  smoother_clothoid::ClothoidSmoother smoother;
  smoother.initialize(opt);
  const auto result = smoother.smooth({path, {1,0}, {1,0}, &costmap, params, nullptr, nullptr});
  EXPECT_TRUE(result.success);
  EXPECT_GE(result.smoothed_path.size(), 2u);
  expectNear(path, input);
}

TEST(SmootherTest, UpsampleDistributesClosureError)
{
  smoother_clothoid::ProcessedPath proc;
  proc.state_count = 2;
  proc.gears = {1.0};
  proc.is_cusp_segment = {false};
  std::vector<double> vars = {0,0,0,0,1, 0.6,0.4,0.2,0,0};
  smoother_clothoid::SmootherParams params;
  params.path_upsampling_factor = 4;
  const auto up = smoother_clothoid::ProblemBuilder::upsamplePath(vars, proc, params);
  ASSERT_EQ(up.size(), 5u);
  EXPECT_NEAR(up[1].x(), 0.15, 1e-9);
  EXPECT_NEAR(up[2].x(), 0.30, 1e-9);
  EXPECT_NEAR(up[3].x(), 0.45, 1e-9);
}
