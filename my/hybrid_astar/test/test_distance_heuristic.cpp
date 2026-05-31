#include <gtest/gtest.h>
#include "my/hybrid_astar/distance_heuristic.hpp"
#include "my/hybrid_astar/node_hybrid.hpp"
#include "my/hybrid_astar/costmap_2d.hpp"

using namespace hybrid_astar;

class DistanceHeuristicTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    search_info.minimum_turning_radius = 2.0;
    search_info.non_straight_penalty = 1.05;
    search_info.change_penalty = 0.0;
    search_info.reverse_penalty = 2.0;
    search_info.cost_penalty = 2.0;
    search_info.retrospective_penalty = 0.015;
    search_info.allow_primitive_interpolation = false;
    search_info.downsample_obstacle_heuristic = true;
    search_info.use_quadratic_cost_penalty = false;

    size_x = 20;
    size_y = 20;
    num_angles = 72;

    ctx.motion_table.initDubin(size_x, size_y, num_angles, search_info);
  }

  NodeHybrid::NodeContext ctx;
  SearchInfo search_info;
  unsigned int size_x;
  unsigned int size_y;
  unsigned int num_angles;
};

TEST_F(DistanceHeuristicTest, PrecomputeAndQuery) {
  float lookup_dim = 20.0f;
  ctx.distance_heuristic->precomputeDistanceHeuristic(
    lookup_dim, MotionModel::DUBIN, num_angles, search_info, ctx.motion_table);

  Coordinates node_coords(5.0f, 5.0f, 0.0f);
  Coordinates goal_coords(10.0f, 10.0f, 36.0f);
  float h = ctx.distance_heuristic->getDistanceHeuristic(
    node_coords, goal_coords, 0.0f, ctx.motion_table);
  EXPECT_GT(h, 0.0f);
}

TEST_F(DistanceHeuristicTest, ReedsSheppModel) {
  ctx.motion_table.initReedsShepp(size_x, size_y, num_angles, search_info);

  float lookup_dim = 20.0f;
  ctx.distance_heuristic->precomputeDistanceHeuristic(
    lookup_dim, MotionModel::REEDS_SHEPP, num_angles, search_info, ctx.motion_table);

  Coordinates node_coords(2.0f, 2.0f, 0.0f);
  Coordinates goal_coords(8.0f, 8.0f, 10.0f);
  float h = ctx.distance_heuristic->getDistanceHeuristic(
    node_coords, goal_coords, 0.0f, ctx.motion_table);
  EXPECT_GT(h, 0.0f);
}

TEST_F(DistanceHeuristicTest, ObstacleHeuristicFallback) {
  float lookup_dim = 5.0f;
  ctx.distance_heuristic->precomputeDistanceHeuristic(
    lookup_dim, MotionModel::DUBIN, num_angles, search_info, ctx.motion_table);

  Coordinates node_coords(15.0f, 15.0f, 0.0f);
  Coordinates goal_coords(0.0f, 0.0f, 0.0f);
  float h = ctx.distance_heuristic->getDistanceHeuristic(
    node_coords, goal_coords, 0.0f, ctx.motion_table);
  EXPECT_GT(h, 0.0f);
}
