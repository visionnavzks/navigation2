// End-to-end integration test: configure SmacPlannerHybrid with the
// ESDF + capsule footprint path, plan a path around a known obstacle, and
// verify that every pose in the returned path is in free space (per the
// ESDF) and that the path actually clears the obstacle.

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "hybrid_astar/costmap_2d.hpp"
#include "hybrid_astar/esdf_holder.hpp"
#include "hybrid_astar/smac_planner_hybrid.hpp"

using hybrid_astar::Costmap2D;
using hybrid_astar::EsdfHolder;
using hybrid_astar::GridCollisionChecker;
using hybrid_astar::OCCUPIED_COST;
using hybrid_astar::Pose;
using hybrid_astar::Path;
using hybrid_astar::SearchInfo;
using hybrid_astar::SmacPlannerHybrid;
using hybrid_astar::SmacPlannerHybridConfig;

namespace
{
constexpr double kResolution = 0.5;
constexpr unsigned int kSizeX = 20;
constexpr unsigned int kSizeY = 20;

// 3-checkpoint capsule: front, center, rear, all along the x-axis.
// Offsets in the robot's local frame; theta rotates them into the world.
std::vector<double> makeCapsuleCheckPoints()
{
  return std::vector<double>{
     0.30, 0.00, 1.0,
     0.00, 0.00, 1.0,
    -0.30, 0.00, 1.0,
  };
}

// (mx, my) -> world coords (cell center).
double cellCenterWorldX(unsigned int mx)
{
  return (static_cast<double>(mx) + 0.5) * kResolution;
}
double cellCenterWorldY(unsigned int my)
{
  return (static_cast<double>(my) + 0.5) * kResolution;
}

void fillObstacleCostmap(Costmap2D & costmap)
{
  for (unsigned int my = 8; my <= 11; ++my) {
    for (unsigned int mx = 8; mx <= 11; ++mx) {
      costmap.setCost(mx, my, OCCUPIED_COST);
    }
  }
}

SearchInfo makeSearchInfo()
{
  SearchInfo info;
  info.minimum_turning_radius = 4.0f;
  info.non_straight_penalty = 1.05f;
  info.change_penalty = 0.0f;
  info.reverse_penalty = 2.0f;
  info.cost_penalty = 2.0f;
  info.retrospective_penalty = 0.015f;
  info.rotation_penalty = 5.0f;
  info.analytic_expansion_ratio = 3.5f;
  info.analytic_expansion_max_length = 60.0f;
  info.analytic_expansion_max_cost = 200.0f;
  info.analytic_expansion_max_cost_override = false;
  info.cache_obstacle_heuristic = false;
  info.allow_reverse_expansion = false;
  info.allow_primitive_interpolation = false;
  info.downsample_obstacle_heuristic = true;
  info.use_quadratic_cost_penalty = true;
  return info;
}
}  // namespace

TEST(EsdfIntegration, PlansAroundObstacleWithCapsuleFootprint)
{
  // 10 m x 10 m map at 0.5 m / cell. Lethal 2 m x 2 m block in the middle.
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  fillObstacleCostmap(costmap);

  // Configure the planner with the ESDF + capsule path. No soft penalty
  // (safe_distance=0) so the test is straightforwardly about hard rejection.
  SmacPlannerHybridConfig config;
  config.downsample_costmap = false;
  config.downsampling_factor = 1;
  config.angle_quantization_bins = 72;
  config.tolerance = 0.25f;
  config.allow_unknown = true;
  config.max_iterations = 1000000;
  config.max_on_approach_iterations = 1000;
  config.terminal_checking_interval = 5000;
  config.smooth_path = true;
  config.max_planning_time = 5.0;
  config.lookup_table_size = 20.0;
  config.debug_visualizations = false;
  config.motion_model_for_search = "DUBIN";
  config.goal_heading_mode = "DEFAULT";
  config.coarse_search_resolution = 1;
  config.search_info = makeSearchInfo();
  config.use_radius = true;
  config.circumscribed_cost = -1.0;
  config.inflation_radius = 0.5;
  config.circumscribed_radius = 0.5;
  // ESDF + capsule path:
  config.use_esdf_footprint = true;
  config.use_exact_esdf = true;
  config.cost_check_points = makeCapsuleCheckPoints();
  config.robot_radius = 0.05;
  config.safe_distance = 0.0;

  SmacPlannerHybrid planner;
  planner.configure(&costmap, config);

  // Start: cell (1, 5), world (0.75, 2.75). Heading along +x.
  // Goal:  cell (18, 14), world (9.25, 7.25). Heading along +x.
  // The obstacle is in the middle of the map, so the path must go either
  // above (y > 6) or below (y < 4) it.
  const Pose start{cellCenterWorldX(1), cellCenterWorldY(5), 0.0};
  const Pose goal{cellCenterWorldX(18), cellCenterWorldY(14), 0.0};
  const Path plan = planner.createPlan(start, goal, []() { return false; });

  // Basic sanity: a path was found.
  ASSERT_FALSE(plan.empty()) << "planner returned an empty path";

  // Build a fresh EsdfHolder + GridCollisionChecker so we can re-verify
  // each pose's clearance. (The planner already validated via the same
  // data; this is a post-hoc audit that exercises the public API.)
  EsdfHolder holder;
  holder.rebuild(&costmap, true);
  GridCollisionChecker checker(&costmap, 72);
  checker.setEsdfFootprint(
    makeCapsuleCheckPoints(), /*robot_radius=*/0.05,
    /*safe_distance=*/0.0, &holder);

  for (size_t i = 0; i < plan.size(); ++i) {
    const auto & p = plan[i];
    const double min_cl = checker.getMinClearance(p.x, p.y, p.theta);
    EXPECT_GE(min_cl, 0.0)
      << "pose #" << i << " (x=" << p.x << ", y=" << p.y << ", theta=" << p.theta
      << ") has negative clearance (in collision): " << min_cl;
    EXPECT_GE(min_cl, 0.05)
      << "pose #" << i << " violates robot_radius=0.05: min_cl=" << min_cl;
  }

  // The path should not just stop early: it should reach near the goal.
  // Allow a tolerance of one cell (0.5 m) since A* snaps to the goal cell.
  const Pose & last = plan.back();
  EXPECT_LT(std::hypot(last.x - goal.x, last.y - goal.y), 1.0)
    << "final pose is far from goal: (" << last.x << ", " << last.y << ")";
}
