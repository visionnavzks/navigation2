// Distance Transform and Inflation Tests

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "costmap_2d/distance_transform.hpp"
#include "costmap_2d/inflation_layer_core.hpp"
#include "costmap_2d/cost_values.hpp"
#include "costmap_2d/costmap_2d.hpp"

using namespace costmap_2d;

// ===================== Distance Transform Tests =====================

TEST(DistanceTransform, transform1D)
{
  const int n = 5;
  std::vector<float> f = {4.0f, 3.0f, 2.0f, 1.0f, 0.0f};
  std::vector<float> d(n);
  std::vector<int> v(n);
  std::vector<float> z(n + 1);

  DistanceTransform::distanceTransform1D(f.data(), d.data(), n, v.data(), z.data());

  // distanceTransform1D computes min over v of f[v] + (q-v)^2
  // For f = {4, 3, 2, 1, 0}, closest zero at index 4
  // d[0] = min(4+0, 3+1, 2+4, 1+9, 0+16) = 4
  // d[1] = min(4+1, 3+0, 2+1, 1+4, 0+9) = 3
  // d[2] = min(4+4, 3+1, 2+0, 1+1, 0+4) = 2
  // d[3] = min(4+9, 3+4, 2+1, 1+0, 0+1) = 1
  // d[4] = min(4+16, 3+9, 2+4, 1+1, 0+0) = 0
  EXPECT_NEAR(d[0], 4.0f, 0.1f);
  EXPECT_NEAR(d[1], 3.0f, 0.1f);
  EXPECT_NEAR(d[2], 2.0f, 0.1f);
  EXPECT_NEAR(d[3], 1.0f, 0.1f);
  EXPECT_NEAR(d[4], 0.0f, 0.1f);
}

TEST(DistanceTransform, transform2D)
{
  // Create a 5x5 image with a single obstacle at (2,2)
  MatrixXfRM img = MatrixXfRM::Constant(5, 5, DistanceTransform::DT_INF);
  img(2, 2) = 0.0f;

  DistanceTransform::distanceTransform2D(img, 5, 5);

  // The distance at (2,2) should be 0
  EXPECT_NEAR(img(2, 2), 0.0f, 0.1f);

  // The distance at (2,3) should be 1.0
  EXPECT_NEAR(img(2, 3), 1.0f, 0.1f);
  EXPECT_NEAR(img(3, 2), 1.0f, 0.1f);

  // The distance at (3,3) should be sqrt(2)
  EXPECT_NEAR(img(3, 3), std::sqrt(2.0f), 0.1f);

  // The distance at (0,0) should be sqrt(8) = 2*sqrt(2)
  EXPECT_NEAR(img(0, 0), 2.0f * std::sqrt(2.0f), 0.1f);
}

TEST(DistanceTransform, transform2DMultipleObstacles)
{
  // Create a 5x5 image with obstacles at (1,1) and (3,3)
  MatrixXfRM img = MatrixXfRM::Constant(5, 5, DistanceTransform::DT_INF);
  img(1, 1) = 0.0f;
  img(3, 3) = 0.0f;

  DistanceTransform::distanceTransform2D(img, 5, 5);

  // The distance at (2,2) should be sqrt(2) (closest to either obstacle)
  EXPECT_NEAR(img(2, 2), std::sqrt(2.0f), 0.1f);

  // The distance at (0,0) should be sqrt(2) (closest to (1,1))
  EXPECT_NEAR(img(0, 0), std::sqrt(2.0f), 0.1f);
}

TEST(DistanceTransform, transform2DEmpty)
{
  // Create a 3x3 image with no obstacles
  MatrixXfRM img = MatrixXfRM::Constant(3, 3, DistanceTransform::DT_INF);

  DistanceTransform::distanceTransform2D(img, 3, 3);

  // All distances should be large
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_GT(img(i, j), 100.0f);
    }
  }
}

// ===================== Inflation Cost Tests =====================

TEST(InflationCost, computeInflationCost)
{
  double resolution = 0.05;
  double inscribed_radius = 0.1;
  double cost_scaling_factor = 10.0;

  // At obstacle (distance = 0)
  unsigned char cost = computeInflationCost(0.0, resolution, inscribed_radius, cost_scaling_factor);
  EXPECT_EQ(cost, LETHAL_OBSTACLE);

  // At inscribed radius (distance * resolution = inscribed_radius)
  cost = computeInflationCost(
    inscribed_radius / resolution, resolution, inscribed_radius, cost_scaling_factor);
  EXPECT_EQ(cost, INSCRIBED_INFLATED_OBSTACLE);

  // Beyond inscribed radius
  cost = computeInflationCost(
    (inscribed_radius + resolution) / resolution, resolution, inscribed_radius, cost_scaling_factor);
  EXPECT_LT(cost, INSCRIBED_INFLATED_OBSTACLE);
  EXPECT_GT(cost, FREE_SPACE);

  // Far from obstacle - should be very small cost
  cost = computeInflationCost(100.0, resolution, inscribed_radius, cost_scaling_factor);
  EXPECT_LE(cost, 1u);
}

TEST(InflationCost, computeInflationCostZeroResolution)
{
  double resolution = 0.0;
  double inscribed_radius = 0.1;
  double cost_scaling_factor = 10.0;

  // At obstacle
  unsigned char cost = computeInflationCost(0.0, resolution, inscribed_radius, cost_scaling_factor);
  EXPECT_EQ(cost, LETHAL_OBSTACLE);
}

TEST(InflationCost, applyInflation)
{
  // Create a simple costmap
  Costmap2D costmap(10, 10, 1.0, 0.0, 0.0);

  // Create a distance map with obstacle at (5,5) - distance 0 means the cell IS the obstacle
  // and should already be marked by the obstacle layer. Inflation only processes distance > 0.
  MatrixXfRM distance_map = MatrixXfRM::Zero(10, 10);
  // Set distances: cell (5,5) is the obstacle (distance 0), neighbors have distance 1, etc.
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      double dx = i - 5;
      double dy = j - 5;
      distance_map(i, j) = std::sqrt(dx * dx + dy * dy);
    }
  }

  // Apply inflation
  applyInflation(
    costmap.getCharMap(),
    distance_map,
    0, 0, 10, 10,
    0, 0,
    10,
    1.0, 1.0, 10.0);

  // The obstacle cell (5,5) should remain FREE_SPACE because applyInflation
  // only processes distance > 0. The obstacle layer marks it separately.
  EXPECT_EQ(costmap.getCost(5, 5), FREE_SPACE);

  // Neighbors should have inflated costs
  EXPECT_GT(costmap.getCost(4, 5), FREE_SPACE);
  EXPECT_GT(costmap.getCost(6, 5), FREE_SPACE);
  EXPECT_GT(costmap.getCost(5, 4), FREE_SPACE);
  EXPECT_GT(costmap.getCost(5, 6), FREE_SPACE);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
