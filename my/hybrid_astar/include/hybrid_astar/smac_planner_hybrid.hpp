#ifndef HYBRID_ASTAR__SMAC_PLANNER_HYBRID_HPP_
#define HYBRID_ASTAR__SMAC_PLANNER_HYBRID_HPP_

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <mutex>

#include "hybrid_astar/a_star.hpp"
#include "hybrid_astar/smoother.hpp"
#include "hybrid_astar/utils.hpp"
#include "hybrid_astar/costmap_downsampler.hpp"
#include "hybrid_astar/collision_checker.hpp"
#include "hybrid_astar/types.hpp"

namespace hybrid_astar
{

struct SmacPlannerHybridConfig
{
  bool downsample_costmap{false};
  int downsampling_factor{1};
  unsigned int angle_quantization_bins{72};
  float tolerance{0.25f};
  bool allow_unknown{true};
  int max_iterations{1000000};
  int max_on_approach_iterations{1000};
  int terminal_checking_interval{5000};
  bool smooth_path{true};
  double max_planning_time{5.0};
  double lookup_table_size{20.0};
  bool debug_visualizations{false};
  std::string motion_model_for_search{"DUBIN"};
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

class SmacPlannerHybrid
{
public:
  SmacPlannerHybrid();
  ~SmacPlannerHybrid();

  void configure(
    Costmap2D * costmap,
    const SmacPlannerHybridConfig & config);

  Path createPlan(
    const Pose & start,
    const Pose & goal,
    std::function<bool()> cancel_checker = []() { return false; });

  void setFootprint(const Footprint & footprint, bool use_radius, double circumscribed_cost);

  AStarAlgorithm<NodeHybrid> * getAStar() { return _a_star.get(); }

protected:
  std::unique_ptr<AStarAlgorithm<NodeHybrid>> _a_star;
  GridCollisionChecker _collision_checker;
  std::unique_ptr<Smoother> _smoother;
  Costmap2D * _costmap;
  std::unique_ptr<CostmapDownsampler> _costmap_downsampler;

  SmacPlannerHybridConfig _config;
  float _angle_bin_size;
  unsigned int _angle_quantizations;
  float _lookup_table_dim;
  MotionModel _motion_model;
  GoalHeadingMode _goal_heading_mode;
  std::mutex _mutex;
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__SMAC_PLANNER_HYBRID_HPP_
