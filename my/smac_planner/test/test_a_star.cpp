#include <gtest/gtest.h>
#include "my/smac_planner/a_star.hpp"
#include "my/smac_planner/collision_checker.hpp"
#include "my/smac_planner/costmap_2d.hpp"
#include "my/smac_planner/constants.hpp"

using namespace smac_planner;

TEST(AStarTest, Simple2DPlan) {
  const unsigned int SIZE_X = 32;
  const unsigned int SIZE_Y = 32;
  Costmap2D costmap(SIZE_X, SIZE_Y, 1.0, 0.0, 0.0);

  for (unsigned int y = 0; y < SIZE_Y; y++) {
    for (unsigned int x = 0; x < SIZE_X; x++) {
      if (x < 5 || x > 26) {
        costmap.setCost(x, y, OCCUPIED_COST);
      }
    }
  }

  GridCollisionChecker collision_checker(&costmap, 1);
  collision_checker.setFootprint(Footprint(), true, 0.0);

  SearchInfo search_info;
  search_info.cost_penalty = 2.0;
  MotionModel motion_model = MotionModel::TWOD;

  AStarAlgorithm<Node2D> astar(motion_model, search_info);
  int max_iterations = 1000000;
  astar.initialize(
    true,
    max_iterations,
    1000,
    5000,
    10.0,
    0.0,
    1);

  astar.setCollisionChecker(&collision_checker);
  astar.setStart(7.0f, 7.0f, 0);
  astar.setGoal(24.0f, 24.0f, 0);

  Node2D::CoordinateVector path;
  int num_iterations = 0;
  bool found = astar.createPath(path, num_iterations, 0.0f, []() { return false; });

  EXPECT_TRUE(found);
  EXPECT_GT(path.size(), 0u);
  EXPECT_GT(num_iterations, 0);
}

TEST(AStarTest, PlanWithTolerance) {
  const unsigned int SIZE_X = 16;
  const unsigned int SIZE_Y = 16;
  Costmap2D costmap(SIZE_X, SIZE_Y, 1.0, 0.0, 0.0);

  GridCollisionChecker collision_checker(&costmap, 1);
  collision_checker.setFootprint(Footprint(), true, 0.0);

  SearchInfo search_info;
  AStarAlgorithm<Node2D> astar(MotionModel::TWOD, search_info);
  int max_iterations = 100000;
  astar.initialize(true, max_iterations, 1000, 5000, 10.0, 0.0, 1);
  astar.setCollisionChecker(&collision_checker);
  astar.setStart(1.0f, 1.0f, 0);
  astar.setGoal(10.0f, 10.0f, 0);

  Node2D::CoordinateVector path;
  int num_iterations = 0;
  bool found = astar.createPath(path, num_iterations, 3.0f, []() { return false; });

  EXPECT_TRUE(found);
  EXPECT_GT(path.size(), 0u);
}
