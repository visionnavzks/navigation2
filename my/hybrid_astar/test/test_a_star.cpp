#include <gtest/gtest.h>
#include "hybrid_astar/a_star.hpp"
#include "hybrid_astar/collision_checker.hpp"
#include "hybrid_astar/costmap_2d.hpp"
#include "hybrid_astar/constants.hpp"

using namespace hybrid_astar;

TEST(AStarTest, SimpleHybridPlan) {
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

  GridCollisionChecker collision_checker(&costmap, 72);
  collision_checker.setFootprint(Footprint(), true, 0.0);

  SearchInfo search_info;
  search_info.cost_penalty = 2.0;
  MotionModel motion_model = MotionModel::DUBIN;

  AStarAlgorithm<NodeHybrid> astar(motion_model, search_info);
  int max_iterations = 1000000;
  astar.initialize(
    true,
    max_iterations,
    1000,
    5000,
    10.0,
    20.0,
    72);

  astar.setCollisionChecker(&collision_checker);
  astar.setStart(7.0f, 7.0f, 0);
  astar.setGoal(24.0f, 24.0f, 0);

  NodeHybrid::CoordinateVector path;
  int num_iterations = 0;
  bool found = astar.createPath(path, num_iterations, 0.0f, []() { return false; });

  EXPECT_TRUE(found);
  EXPECT_GT(path.size(), 0u);
  EXPECT_GT(num_iterations, 0);
}
