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

#include "constrained_smoother/kinematic_smoother.hpp"
#include "gtest/gtest.h"
#include "constrained_smoother/smoother.hpp"
#include "constrained_smoother/smoother_cost_function.hpp"

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

// ---- Tests ----

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

TEST(ErrorTest, InvalidPathCarriesStableCode)
{
  const constrained_smoother::InvalidPath error("test invalid path");

  EXPECT_EQ(error.code(), constrained_smoother::ErrorCode::InvalidPath);
  EXPECT_STREQ(error.codeString(), "CS_INVALID_PATH");
  EXPECT_STREQ(error.what(), std::string("test invalid path").c_str());
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
