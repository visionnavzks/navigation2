#include <cmath>
#include <gtest/gtest.h>
#include "ompl/base/ScopedState.h"
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

// Regression test for Bug #29: the lookup stride must match the precompute
// layout. For even size_lookup_ (e.g., 20), the y range is [0, size/2], so
// each x-row occupies (size/2 + 1) * dim_3 entries. Using ceil(size/2) as
// the stride misaligns every row past the first. This test samples several
// (x, y, theta) cells across odd AND even lookup dims and verifies the
// returned value equals the OMPL state-space distance precomputed at that
// exact (x, y, theta).
TEST_F(DistanceHeuristicTest, LookupMatchesOmplStridedByYRange) {
  const std::vector<float> lookup_dims = {19.0f, 20.0f, 21.0f};

  for (float lookup_dim : lookup_dims) {
    ctx.distance_heuristic->precomputeDistanceHeuristic(
      lookup_dim, MotionModel::DUBIN, num_angles, search_info, ctx.motion_table);

    const int floored_size = static_cast<int>(std::floor(lookup_dim / 2.0));
    const float bin_size =
      2.0f * static_cast<float>(M_PI) / static_cast<float>(num_angles);

    const std::vector<int> x_indices = {0, 1, 5, floored_size, floored_size + 1, 2 * floored_size};
    const std::vector<int> y_indices = {0, 1, 5, floored_size};
    const std::vector<int> theta_indices = {
      0, 1, static_cast<int>(num_angles / 2), static_cast<int>(num_angles - 1)};

    for (int i : x_indices) {
      for (int j : y_indices) {
        for (int k : theta_indices) {
          const float x_node = static_cast<float>(i - floored_size);
          const float y_node = static_cast<float>(j);
          const float theta_bin = static_cast<float>(k);

          Coordinates node_coords(x_node, y_node, theta_bin);
          Coordinates goal_coords(0.0f, 0.0f, 0.0f);

          ompl::base::ScopedState<> from(ctx.motion_table.state_space);
          ompl::base::ScopedState<> to(ctx.motion_table.state_space);
          from[0] = x_node;
          from[1] = y_node;
          from[2] = static_cast<double>(theta_bin) * bin_size;
          to[0] = 0.0;
          to[1] = 0.0;
          to[2] = 0.0;
          const float expected =
            ctx.motion_table.state_space->distance(from(), to());

          const float actual = ctx.distance_heuristic->getDistanceHeuristic(
            node_coords, goal_coords, 0.0f, ctx.motion_table);

          ASSERT_NEAR(actual, expected, 1e-3f)
            << "lookup_dim=" << lookup_dim << " i=" << i << " j=" << j
            << " k=" << k
            << " (expected " << expected << ", got " << actual << ")";
        }
      }
    }
  }
}
