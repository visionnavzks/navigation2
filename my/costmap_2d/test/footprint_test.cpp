// Copyright (c) 2013, Willow Garage, Inc.
// All rights reserved.
//
// Ported to ROS-free costmap_2d library

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <limits>
#include <cmath>

#include "costmap_2d/footprint.hpp"
#include "costmap_2d/costmap_math.hpp"

using namespace costmap_2d;

// ===================== Footprint String Parsing Tests =====================

TEST(FootprintTest, makeFootprintFromString)
{
  std::vector<Point> footprint;
  bool result = makeFootprintFromString(
    "[[1, 2.2], [.3, -4e4], [-.3, -4e4], [-1, 2.2]]", footprint);
  EXPECT_TRUE(result);
  EXPECT_EQ(footprint.size(), 4u);
  EXPECT_NEAR(footprint[0].x, 1.0, 1e-5);
  EXPECT_NEAR(footprint[0].y, 2.2, 1e-5);
  EXPECT_NEAR(footprint[1].x, 0.3, 1e-5);
  EXPECT_NEAR(footprint[1].y, -4e4, 1e-5);
  EXPECT_NEAR(footprint[2].x, -0.3, 1e-5);
  EXPECT_NEAR(footprint[2].y, -4e4, 1e-5);
  EXPECT_NEAR(footprint[3].x, -1.0, 1e-5);
  EXPECT_NEAR(footprint[3].y, 2.2, 1e-5);
}

TEST(FootprintTest, makeFootprintFromStringParseError)
{
  std::vector<Point> footprint;
  bool result = makeFootprintFromString("[[bad_string", footprint);
  EXPECT_FALSE(result);
}

TEST(FootprintTest, makeFootprintFromStringTwoPoints)
{
  std::vector<Point> footprint;
  bool result = makeFootprintFromString(
    "[[1, 2.2], [.3, -4e4]]", footprint);
  EXPECT_FALSE(result);
}

TEST(FootprintTest, makeFootprintFromStringThreePoints)
{
  std::vector<Point> footprint;
  bool result = makeFootprintFromString(
    "[[1, 2], [3, 4], [5, 6]]", footprint);
  EXPECT_TRUE(result);
  EXPECT_EQ(footprint.size(), 3u);
}

// ===================== makeFootprintFromRadius Tests =====================

TEST(FootprintTest, makeFootprintFromRadius)
{
  std::vector<Point> footprint = makeFootprintFromRadius(0.2);
  EXPECT_EQ(footprint.size(), 16u);

  for (const auto & pt : footprint) {
    double dist = std::sqrt(pt.x * pt.x + pt.y * pt.y);
    EXPECT_NEAR(dist, 0.2, 0.01);
  }
}

TEST(FootprintTest, makeFootprintFromRadiusZero)
{
  std::vector<Point> footprint = makeFootprintFromRadius(0.0);
  EXPECT_EQ(footprint.size(), 16u);
}

// ===================== calculateMinAndMaxDistances Tests =====================

TEST(FootprintTest, calculateMinAndMaxDistances)
{
  std::vector<Point> footprint;
  footprint.push_back({-1.0, 0.0, 0.0});
  footprint.push_back({0.0, 1.0, 0.0});
  footprint.push_back({1.0, 0.0, 0.0});
  footprint.push_back({0.0, -1.0, 0.0});

  auto [min_dist, max_dist] = calculateMinAndMaxDistances(footprint);
  EXPECT_NEAR(min_dist, 1.0 / std::sqrt(2.0), 1e-5);  // distance to edge
  EXPECT_NEAR(max_dist, 1.0, 1e-5);  // distance to vertex
}

TEST(FootprintTest, calculateMinAndMaxDistancesNotEnoughPoints)
{
  std::vector<Point> footprint;
  footprint.push_back({2.0, 2.0, 0.0});
  footprint.push_back({-2.0, -2.0, 0.0});

  auto [min_dist, max_dist] = calculateMinAndMaxDistances(footprint);
  EXPECT_EQ(min_dist, std::numeric_limits<double>::max());
  EXPECT_EQ(max_dist, 0.0);
}

TEST(FootprintTest, calculateMinAndMaxDistancesSquare)
{
  std::vector<Point> footprint;
  footprint.push_back({-1.0, -1.0, 0.0});
  footprint.push_back({1.0, -1.0, 0.0});
  footprint.push_back({1.0, 1.0, 0.0});
  footprint.push_back({-1.0, 1.0, 0.0});

  auto [min_dist, max_dist] = calculateMinAndMaxDistances(footprint);
  EXPECT_NEAR(min_dist, 1.0, 1e-5);  // distance to edge
  EXPECT_NEAR(max_dist, std::sqrt(2.0), 1e-5);  // distance to corner
}

TEST(FootprintTest, calculateMinAndMaxDistancesRectangle)
{
  std::vector<Point> footprint;
  footprint.push_back({-2.0, -1.0, 0.0});
  footprint.push_back({2.0, -1.0, 0.0});
  footprint.push_back({2.0, 1.0, 0.0});
  footprint.push_back({-2.0, 1.0, 0.0});

  auto [min_dist, max_dist] = calculateMinAndMaxDistances(footprint);
  EXPECT_NEAR(min_dist, 1.0, 1e-5);  // distance to short edge
  EXPECT_NEAR(max_dist, std::sqrt(5.0), 1e-5);  // distance to far corner
}

// ===================== transformFootprint Tests =====================

TEST(FootprintTest, transformFootprint)
{
  std::vector<Point> footprint;
  footprint.push_back({-1.0, 0.0, 0.0});
  footprint.push_back({0.0, 1.0, 0.0});
  footprint.push_back({1.0, 0.0, 0.0});
  footprint.push_back({0.0, -1.0, 0.0});

  std::vector<Point> oriented;
  transformFootprint(5.0, 5.0, 0.0, footprint, oriented);
  EXPECT_EQ(oriented.size(), footprint.size());

  EXPECT_NEAR(oriented[0].x, 4.0, 1e-5);
  EXPECT_NEAR(oriented[0].y, 5.0, 1e-5);
  EXPECT_NEAR(oriented[1].x, 5.0, 1e-5);
  EXPECT_NEAR(oriented[1].y, 6.0, 1e-5);
}

TEST(FootprintTest, transformFootprintRotated)
{
  std::vector<Point> footprint;
  footprint.push_back({1.0, 0.0, 0.0});

  std::vector<Point> oriented;
  transformFootprint(0.0, 0.0, M_PI / 2, footprint, oriented);
  EXPECT_NEAR(oriented[0].x, 0.0, 1e-5);
  EXPECT_NEAR(oriented[0].y, 1.0, 1e-5);
}

// ===================== padFootprint Tests =====================

TEST(FootprintTest, padFootprint)
{
  std::vector<Point> footprint;
  footprint.push_back({1.0, 1.0, 0.0});
  footprint.push_back({-1.0, -1.0, 0.0});

  padFootprint(footprint, 0.5);
  EXPECT_NEAR(footprint[0].x, 1.5, 1e-5);
  EXPECT_NEAR(footprint[0].y, 1.5, 1e-5);
  EXPECT_NEAR(footprint[1].x, -1.5, 1e-5);
  EXPECT_NEAR(footprint[1].y, -1.5, 1e-5);
}

TEST(FootprintTest, padFootprintZero)
{
  std::vector<Point> footprint;
  footprint.push_back({1.0, 1.0, 0.0});
  footprint.push_back({-1.0, -1.0, 0.0});

  padFootprint(footprint, 0.0);
  EXPECT_NEAR(footprint[0].x, 1.0, 1e-5);
  EXPECT_NEAR(footprint[0].y, 1.0, 1e-5);
}

// ===================== CostmapMath Tests =====================

TEST(CostmapMath, sign)
{
  EXPECT_DOUBLE_EQ(sign(-5.0), -1.0);
  EXPECT_DOUBLE_EQ(sign(0.0), 1.0);
  EXPECT_DOUBLE_EQ(sign(5.0), 1.0);
}

TEST(CostmapMath, sign0)
{
  EXPECT_DOUBLE_EQ(sign0(-5.0), -1.0);
  EXPECT_DOUBLE_EQ(sign0(0.0), 0.0);
  EXPECT_DOUBLE_EQ(sign0(5.0), 1.0);
}

TEST(CostmapMath, distance)
{
  EXPECT_DOUBLE_EQ(distance(0.0, 0.0, 3.0, 4.0), 5.0);
  EXPECT_DOUBLE_EQ(distance(0.0, 0.0, 0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(distance(1.0, 1.0, 4.0, 5.0), 5.0);
}

TEST(CostmapMath, distanceToLine)
{
  // Point at (0, 0), line from (0, 1) to (1, 1)
  EXPECT_DOUBLE_EQ(distanceToLine(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 1.0);

  // Point at (0.5, 0), line from (0, 1) to (1, 1)
  EXPECT_DOUBLE_EQ(distanceToLine(0.5, 0.0, 0.0, 1.0, 1.0, 1.0), 1.0);

  // Point at (0.5, 0.5), line from (0, 0) to (1, 0)
  EXPECT_DOUBLE_EQ(distanceToLine(0.5, 0.5, 0.0, 0.0, 1.0, 0.0), 0.5);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
