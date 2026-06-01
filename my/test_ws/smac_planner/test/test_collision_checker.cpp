#include <gtest/gtest.h>
#include "my/smac_planner/collision_checker.hpp"
#include "my/smac_planner/costmap_2d.hpp"
#include "my/smac_planner/constants.hpp"

using namespace smac_planner;

TEST(CollisionCheckerTest, BasicCollisionCheck) {
  Costmap2D costmap(10, 10, 1.0, 0.0, 0.0);
  costmap.setCost(5, 5, OCCUPIED_COST);

  GridCollisionChecker checker(&costmap, 1);
  Footprint footprint;
  checker.setFootprint(footprint, true, 0.0);

  EXPECT_FALSE(checker.inCollision(2.0f, 2.0f, 0.0f, false));
  EXPECT_TRUE(checker.inCollision(5.0f, 5.0f, 0.0f, false));

  EXPECT_FALSE(checker.inCollision(0, false));
  unsigned int obstacle_idx = 5 * 10 + 5;
  EXPECT_TRUE(checker.inCollision(obstacle_idx, false));
}

TEST(CollisionCheckerTest, GetCost) {
  Costmap2D costmap(10, 10, 1.0, 0.0, 0.0);
  costmap.setCost(3, 3, OCCUPIED_COST);

  GridCollisionChecker checker(&costmap, 1);
  checker.setFootprint(Footprint(), true, 0.0);

  (void)checker.inCollision(3.0f, 3.0f, 0.0f, false);
  EXPECT_FLOAT_EQ(checker.getCost(), OCCUPIED_COST);
}

TEST(CollisionCheckerTest, UnknownTraversal) {
  Costmap2D costmap(10, 10, 1.0, 0.0, 0.0);
  costmap.setCost(3, 3, UNKNOWN_COST);

  GridCollisionChecker checker(&costmap, 1);
  checker.setFootprint(Footprint(), true, 0.0);

  EXPECT_TRUE(checker.inCollision(3.0f, 3.0f, 0.0f, false));
  EXPECT_FALSE(checker.inCollision(3.0f, 3.0f, 0.0f, true));
}
