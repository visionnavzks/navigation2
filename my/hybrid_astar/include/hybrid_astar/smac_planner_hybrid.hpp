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
#include "hybrid_astar/esdf_holder.hpp"
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

  // ---- ESDF + capsule footprint configuration ----
  //
  // Footprint interpretation, in order of precedence:
  //   1. If use_esdf_footprint == true (or cost_check_points is non-empty), the
  //      planner switches to ESDF-based collision checking and uses the multi-
  //      checkpoint "capsule" model: each triple (lx, ly, weight) in
  //      cost_check_points describes one local point on the robot that is
  //      transformed into the world frame at every search node, then checked
  //      against the cached ESDF.
  //   2. Otherwise, the legacy polygon footprint (robot_footprint) or single-
  //      radius path is used. This is the original behavior; do not change the
  //      defaults unless you want to opt in.
  bool use_esdf_footprint{false};
  bool use_exact_esdf{true};
  std::vector<double> cost_check_points{};
  // Per-checkpoint inflation radius (m). The minimum ESDF distance at any
  // checkpoint must be >= this value (typically the inscribed/circumscribed
  // radius). 0 means "treat every checkpoint as a point".
  double robot_radius{0.0};
  // Soft-penalty threshold (m). When (min_clearance - robot_radius) drops below
  // this value, a smooth quadratic penalty is added to the cell cost. Set to 0
  // to disable soft penalties and only do hard rejection.
  double safe_distance{0.0};
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
  EsdfHolder _esdf_holder;

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
