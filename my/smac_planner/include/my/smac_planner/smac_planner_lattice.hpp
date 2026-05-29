#ifndef SMAC_PLANNER__SMAC_PLANNER_LATTICE_HPP_
#define SMAC_PLANNER__SMAC_PLANNER_LATTICE_HPP_

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

struct SmacPlannerLatticeConfig
{
  float tolerance{0.25f};
  bool allow_unknown{true};
  int max_iterations{1000000};
  int max_on_approach_iterations{1000};
  int terminal_checking_interval{5000};
  bool smooth_path{true};
  double max_planning_time{5.0};
  double lookup_table_size{20.0};
  bool debug_visualizations{false};
  std::string goal_heading_mode{"DEFAULT"};
  int coarse_search_resolution{1};

  SearchInfo search_info;
  SmootherParams smoother_params;
  Footprint robot_footprint;
  bool use_radius{false};
  double circumscribed_cost{-1.0};
  double inflation_radius{0.5};
  double circumscribed_radius{0.5};
};

class SmacPlannerLattice
{
public:
  SmacPlannerLattice();
  ~SmacPlannerLattice();

  void configure(
    Costmap2D * costmap,
    const SmacPlannerLatticeConfig & config);

  Path createPlan(
    const Pose & start,
    const Pose & goal,
    std::function<bool()> cancel_checker = []() { return false; });

  void setFootprint(const Footprint & footprint, bool use_radius, double circumscribed_cost);

  AStarAlgorithm<NodeLattice> * getAStar() { return _a_star.get(); }

protected:
  std::unique_ptr<AStarAlgorithm<NodeLattice>> _a_star;
  GridCollisionChecker _collision_checker;
  std::unique_ptr<Smoother> _smoother;
  Costmap2D * _costmap;

  SmacPlannerLatticeConfig _config;
  LatticeMetadata _metadata;
  float _lookup_table_dim;
  GoalHeadingMode _goal_heading_mode;
  std::mutex _mutex;
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__SMAC_PLANNER_LATTICE_HPP_
