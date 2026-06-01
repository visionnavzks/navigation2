#include <gtest/gtest.h>
#include "hybrid_astar/obstacle_heuristic.hpp"
#include "hybrid_astar/costmap_2d.hpp"
#include "hybrid_astar/constants.hpp"

using namespace hybrid_astar;

TEST(ObstacleHeuristicTest, ResetAndQuery) {
  Costmap2D costmap(30, 30, 0.05, 0.0, 0.0);
  ObstacleHeuristic oh;

  oh.resetObstacleHeuristic(&costmap, 5, 5, 24, 24, false);
  Coordinates node_coords(5.0f, 5.0f, 0.0f);
  float h = oh.getObstacleHeuristic(node_coords, 2.0f, false, false);
  EXPECT_GT(h, 0.0f);
}

TEST(ObstacleHeuristicTest, GoalHasSmallHeuristic) {
  Costmap2D costmap(30, 30, 0.05, 0.0, 0.0);
  ObstacleHeuristic oh;

  oh.resetObstacleHeuristic(&costmap, 10, 10, 15, 15, false);
  Coordinates goal_coords(15.0f, 15.0f, 0.0f);
  float h = oh.getObstacleHeuristic(goal_coords, 2.0f, false, false);
  EXPECT_LE(h, 0.01f);
}

TEST(ObstacleHeuristicTest, DownsampledMode) {
  Costmap2D costmap(40, 40, 0.05, 0.0, 0.0);
  ObstacleHeuristic oh;

  oh.resetObstacleHeuristic(&costmap, 10, 10, 30, 30, true);
  Coordinates node_coords(10.0f, 10.0f, 0.0f);
  float h = oh.getObstacleHeuristic(node_coords, 2.0f, false, true);
  EXPECT_GT(h, 0.0f);
}

TEST(ObstacleHeuristicTest, ObstacleIncreasesHeuristic) {
  Costmap2D costmap(30, 30, 0.05, 0.0, 0.0);
  for (unsigned int i = 5; i < 25; i++) {
    costmap.setCost(i, 15, OCCUPIED_COST);
  }

  ObstacleHeuristic oh;
  oh.resetObstacleHeuristic(&costmap, 5, 10, 24, 20, false);
  Coordinates node_coords(5.0f, 10.0f, 0.0f);
  float h = oh.getObstacleHeuristic(node_coords, 2.0f, false, false);
  EXPECT_GT(h, 0.0f);
}

TEST(ObstacleHeuristicTest, DistanceHeuristic2D) {
  ObstacleHeuristic oh;
  float d = oh.distanceHeuristic2D(12, 10, 5, 4);
  EXPECT_NEAR(d, std::sqrt(9.0f + 9.0f), 1e-5f);
}
