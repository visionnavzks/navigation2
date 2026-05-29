#ifndef SMAC_PLANNER__SMAC_PLANNER_2D_HPP_
#define SMAC_PLANNER__SMAC_PLANNER_2D_HPP_

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <mutex>

#include "my/smac_planner/a_star.hpp"
#include "my/smac_planner/smoother.hpp"
#include "my/smac_planner/utils.hpp"
#include "my/smac_planner/costmap_downsampler.hpp"
#include "my/smac_planner/collision_checker.hpp"
#include "my/smac_planner/types.hpp"

namespace smac_planner
{

struct SmacPlanner2DConfig
{
  bool downsample_costmap{false};
  int downsampling_factor{1};
  float tolerance{0.125f};
  bool allow_unknown{true};
  int max_iterations{1000000};
  int max_on_approach_iterations{1000};
  int terminal_checking_interval{5000};
  double max_planning_time{2.0};
  bool use_final_approach_orientation{false};

  SearchInfo search_info;
  SmootherParams smoother_params;
  Footprint robot_footprint;
  bool use_radius{true};
};

class SmacPlanner2D
{
public:
  SmacPlanner2D();
  ~SmacPlanner2D();

  void configure(
    Costmap2D * costmap,
    const SmacPlanner2DConfig & config);

  Path createPlan(
    const Pose & start,
    const Pose & goal,
    std::function<bool()> cancel_checker = []() { return false; });

  void setFootprint(const Footprint & footprint, bool use_radius);

  AStarAlgorithm<Node2D> * getAStar() { return _a_star.get(); }

protected:
  std::unique_ptr<AStarAlgorithm<Node2D>> _a_star;
  GridCollisionChecker _collision_checker;
  std::unique_ptr<Smoother> _smoother;
  Costmap2D * _costmap;
  std::unique_ptr<CostmapDownsampler> _costmap_downsampler;

  SmacPlanner2DConfig _config;
  std::mutex _mutex;
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__SMAC_PLANNER_2D_HPP_
