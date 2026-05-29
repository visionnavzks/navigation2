// Copyright (c) 2019 Intel Corporation
// Ported to ROS-free my_costmap_2d library

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "my_costmap_2d/footprint_collision_checker.hpp"
#include "my_costmap_2d/costmap_2d.hpp"
#include "my_costmap_2d/cost_values.hpp"

using namespace my_costmap_2d;

// ===================== Basic Collision Checker Tests =====================

TEST(CollisionChecker, basicCost)
{
  auto costmap = std::make_shared<Costmap2D>(100, 100, 0.1, 0, 0, 0);

  Point p1{-0.5, 0.0, 0.0};
  Point p2{0.0, 0.5, 0.0};
  Point p3{0.5, 0.0, 0.0};
  Point p4{0.0, -0.5, 0.0};

  Footprint footprint = {p1, p2, p3, p4};

  FootprintCollisionChecker<std::shared_ptr<Costmap2D>> collision_checker(costmap);

  auto value = collision_checker.footprintCostAtPose(5.0, 5.0, 0.0, footprint);
  EXPECT_NEAR(value, 0.0, 0.001);
}

TEST(CollisionChecker, pointCost)
{
  auto costmap = std::make_shared<Costmap2D>(100, 100, 0.1, 0, 0, 0);

  FootprintCollisionChecker<std::shared_ptr<Costmap2D>> collision_checker(costmap);

  auto value = collision_checker.pointCost(50, 50);
  EXPECT_NEAR(value, 0.0, 0.001);
}

TEST(CollisionChecker, worldToMap)
{
  auto costmap = std::make_shared<Costmap2D>(100, 100, 0.1, 0, 0, 0);

  FootprintCollisionChecker<std::shared_ptr<Costmap2D>> collision_checker(costmap);

  unsigned int x, y;
  collision_checker.worldToMap(1.0, 1.0, x, y);

  auto value = collision_checker.pointCost(x, y);
  EXPECT_NEAR(value, 0.0, 0.001);

  costmap->setCost(50, 50, 200);
  collision_checker.worldToMap(5.0, 5.0, x, y);

  EXPECT_NEAR(collision_checker.pointCost(x, y), 200.0, 0.001);
}

TEST(CollisionChecker, footprintAtPoseWithMovement)
{
  auto costmap = std::make_shared<Costmap2D>(100, 100, 0.1, 0, 0, 254);

  for (unsigned int i = 40; i <= 60; ++i) {
    for (unsigned int j = 40; j <= 60; ++j) {
      costmap->setCost(i, j, 0);
    }
  }

  Point p1{-1.0, 1.0, 0.0};
  Point p2{1.0, 1.0, 0.0};
  Point p3{1.0, -1.0, 0.0};
  Point p4{-1.0, -1.0, 0.0};

  Footprint footprint = {p1, p2, p3, p4};

  FootprintCollisionChecker<std::shared_ptr<Costmap2D>> collision_checker(costmap);

  auto value = collision_checker.footprintCostAtPose(5.0, 5.0, 0.0, footprint);
  EXPECT_NEAR(value, 0.0, 0.001);

  auto up_value = collision_checker.footprintCostAtPose(5.0, 4.9, 0.0, footprint);
  EXPECT_NEAR(up_value, 254.0, 0.001);

  auto down_value = collision_checker.footprintCostAtPose(5.0, 5.2, 0.0, footprint);
  EXPECT_NEAR(down_value, 254.0, 0.001);
}

TEST(CollisionChecker, pointAndLineCost)
{
  auto costmap = std::make_shared<Costmap2D>(100, 100, 0.1, 0, 0, 0);

  costmap->setCost(62, 50, 254);
  costmap->setCost(39, 60, 254);

  Point p1{-1.0, 1.0, 0.0};
  Point p2{1.0, 1.0, 0.0};
  Point p3{1.0, -1.0, 0.0};
  Point p4{-1.0, -1.0, 0.0};

  Footprint footprint = {p1, p2, p3, p4};

  FootprintCollisionChecker<std::shared_ptr<Costmap2D>> collision_checker(costmap);

  auto value = collision_checker.footprintCostAtPose(5.0, 5.0, 0.0, footprint);
  EXPECT_NEAR(value, 0.0, 0.001);

  auto left_value = collision_checker.footprintCostAtPose(4.9, 5.0, 0.0, footprint);
  EXPECT_NEAR(left_value, 254.0, 0.001);

  auto right_value = collision_checker.footprintCostAtPose(5.2, 5.0, 0.0, footprint);
  EXPECT_NEAR(right_value, 254.0, 0.001);
}

TEST(CollisionChecker, setCostmap)
{
  auto costmap1 = std::make_shared<Costmap2D>(100, 100, 0.1, 0, 0, 0);
  auto costmap2 = std::make_shared<Costmap2D>(50, 50, 0.2, 0, 0, 100);

  FootprintCollisionChecker<std::shared_ptr<Costmap2D>> collision_checker(costmap1);

  collision_checker.setCostmap(costmap2);
  EXPECT_EQ(collision_checker.getCostmap(), costmap2);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
