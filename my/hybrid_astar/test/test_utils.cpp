#include <gtest/gtest.h>
#include "hybrid_astar/costmap_2d.hpp"
#include "hybrid_astar/utils.hpp"

using namespace hybrid_astar;

TEST(UtilsTest, GetWorldCoords) {
  Costmap2D costmap(10, 10, 0.5, -2.5, -2.5);
  Pose p = getWorldCoords(2.0f, 3.0f, &costmap);
  EXPECT_FLOAT_EQ(p.x, -1.5f);
  EXPECT_FLOAT_EQ(p.y, -1.0f);
  EXPECT_FLOAT_EQ(p.theta, 0.0f);
}

TEST(UtilsTest, GetWorldCoordsOrigin) {
  Costmap2D costmap(10, 10, 0.5, -2.5, -2.5);
  Pose p = getWorldCoords(0.0f, 0.0f, &costmap);
  EXPECT_FLOAT_EQ(p.x, -2.5f);
  EXPECT_FLOAT_EQ(p.y, -2.5f);
}

TEST(UtilsTest, FindCircumscribedCost) {
  Costmap2D costmap(10, 10, 0.1, 0.0, 0.0);
  double cost = findCircumscribedCost(&costmap, 0.3, 0.5);
  EXPECT_GT(cost, 0.0);
  EXPECT_LE(cost, INSCRIBED_COST);
}

TEST(UtilsTest, FindCircumscribedCostInflationTooSmall) {
  Costmap2D costmap(10, 10, 0.1, 0.0, 0.0);
  double cost = findCircumscribedCost(&costmap, 0.5, 0.3);
  EXPECT_FLOAT_EQ(cost, 0.0);
}
