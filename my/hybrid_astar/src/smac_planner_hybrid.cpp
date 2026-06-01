#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include <chrono>

#include "hybrid_astar/smac_planner_hybrid.hpp"

// #define BENCHMARK_TESTING

namespace hybrid_astar
{

using namespace std::chrono;

SmacPlannerHybrid::SmacPlannerHybrid()
: _a_star(nullptr),
  _collision_checker(nullptr, 1),
  _smoother(nullptr),
  _costmap(nullptr),
  _costmap_downsampler(nullptr)
{
}

SmacPlannerHybrid::~SmacPlannerHybrid()
{
}

void SmacPlannerHybrid::configure(
  Costmap2D * costmap,
  const SmacPlannerHybridConfig & config)
{
  _costmap = costmap;
  _config = config;

  _angle_bin_size = 2.0 * M_PI / static_cast<float>(_config.angle_quantization_bins);
  _angle_quantizations = _config.angle_quantization_bins;

  int max_iterations = _config.max_iterations;
  int max_on_approach_iterations = _config.max_on_approach_iterations;

  if (max_on_approach_iterations <= 0) {
    max_on_approach_iterations = std::numeric_limits<int>::max();
  }

  if (max_iterations <= 0) {
    max_iterations = std::numeric_limits<int>::max();
  }

  if (_config.coarse_search_resolution <= 0) {
    _config.coarse_search_resolution = 1;
  }

  if (_angle_quantizations % static_cast<unsigned int>(_config.coarse_search_resolution) != 0) {
    throw std::runtime_error(
      "coarse iteration should be an increment of the number of angular bins configured");
  }

  _motion_model = fromString(_config.motion_model_for_search);
  _goal_heading_mode = toGoalHeadingMode(_config.goal_heading_mode);

  if (_goal_heading_mode == GoalHeadingMode::UNKNOWN) {
    throw std::runtime_error(
      "Unable to get GoalHeader type. Given '" + _config.goal_heading_mode + "' "
      "Valid options are DEFAULT, BIDIRECTIONAL, ALL_DIRECTION.");
  }

  float lookup_table_dim =
    static_cast<float>(_config.lookup_table_size) /
    static_cast<float>(_costmap->getResolution());

  lookup_table_dim = static_cast<float>(static_cast<int>(lookup_table_dim));

  _lookup_table_dim = lookup_table_dim;

  double circumscribed_cost = _config.circumscribed_cost;
  if (circumscribed_cost < 0.0) {
    circumscribed_cost = findCircumscribedCost(
      _costmap, _config.circumscribed_radius, _config.inflation_radius);
  }

  _collision_checker = GridCollisionChecker(_costmap, _angle_quantizations);

  // Decide which footprint backend to use. The ESDF path is enabled by
  // either an explicit flag or by the presence of cost_check_points.
  const bool want_esdf_footprint =
    _config.use_esdf_footprint || !_config.cost_check_points.empty();

  if (want_esdf_footprint) {
    // ESDF needs the costmap to be built before the collision checker
    // queries it. Build with the active (possibly downsampled) costmap.
    _esdf_holder.rebuild(costmap, _config.use_exact_esdf);
    _collision_checker.setEsdfFootprint(
      _config.cost_check_points,
      _config.robot_radius,
      _config.safe_distance,
      &_esdf_holder);
  } else {
    _collision_checker.setFootprint(
      _config.robot_footprint,
      _config.use_radius,
      circumscribed_cost);
  }

  _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _config.search_info);
  _a_star->initialize(
    _config.allow_unknown,
    max_iterations,
    max_on_approach_iterations,
    _config.terminal_checking_interval,
    _config.max_planning_time,
    _lookup_table_dim,
    _angle_quantizations);
  if (want_esdf_footprint) {
    _a_star->setEsdfResources(
      &_esdf_holder,
      _config.cost_check_points,
      _config.robot_radius,
      _config.safe_distance);
  }

  if (_config.smooth_path) {
    _smoother = std::make_unique<Smoother>(_config.smoother_params);
    double min_turning_radius = _config.search_info.minimum_turning_radius *
      _costmap->getResolution();
    _smoother->initialize(min_turning_radius);
  }

  _costmap_downsampler = std::make_unique<CostmapDownsampler>();
  _costmap_downsampler->on_configure(_costmap, _config.downsampling_factor);
}

void SmacPlannerHybrid::setFootprint(
  const Footprint & footprint, bool use_radius, double circumscribed_cost)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _config.robot_footprint = footprint;
  _config.use_radius = use_radius;
  if (circumscribed_cost < 0.0) {
    circumscribed_cost = findCircumscribedCost(
      _costmap, _config.circumscribed_radius, _config.inflation_radius);
  }
  _config.circumscribed_cost = circumscribed_cost;
  _collision_checker.setFootprint(footprint, use_radius, circumscribed_cost);
}

Path SmacPlannerHybrid::createPlan(
  const Pose & start,
  const Pose & goal,
  std::function<bool()> cancel_checker)
{
  std::lock_guard<std::mutex> lock_reinit(_mutex);
  steady_clock::time_point a = steady_clock::now();

  Costmap2D * costmap = _costmap;
  if (_config.downsample_costmap && _config.downsampling_factor > 1) {
    costmap = _costmap_downsampler->downsample(_config.downsampling_factor);
    _collision_checker.setCostmap(costmap);
  }

  // If the ESDF path is active, the cached ESDF must be rebuilt against the
  // (possibly downsampled) costmap before the A* reads it.
  if (_config.use_esdf_footprint || !_config.cost_check_points.empty()) {
    _esdf_holder.rebuild(costmap, _config.use_exact_esdf);
  }

  _a_star->setCollisionChecker(&_collision_checker);

  float mx_start, my_start, mx_goal, my_goal;
  if (!costmap->worldToMapContinuous(start.x, start.y, mx_start, my_start)) {
    throw std::runtime_error(
      "Start Coordinates of(" + std::to_string(start.x) + ", " +
      std::to_string(start.y) + ") was outside bounds");
  }

  unsigned int start_orientation_bin_int =
    wrapBinIndex(static_cast<int>(std::round(start.theta / _angle_bin_size)), _angle_quantizations);
  _a_star->setStart(mx_start, my_start, start_orientation_bin_int);

  if (!costmap->worldToMapContinuous(goal.x, goal.y, mx_goal, my_goal)) {
    throw std::runtime_error(
      "Goal Coordinates of(" + std::to_string(goal.x) + ", " +
      std::to_string(goal.y) + ") was outside bounds");
  }

  unsigned int goal_orientation_bin_int =
    wrapBinIndex(static_cast<int>(std::round(goal.theta / _angle_bin_size)), _angle_quantizations);
  _a_star->setGoal(
    mx_goal, my_goal, goal_orientation_bin_int,
    _goal_heading_mode, _config.coarse_search_resolution);

  if (static_cast<int>(mx_start) == static_cast<int>(mx_goal) &&
    static_cast<int>(my_start) == static_cast<int>(my_goal) &&
    start_orientation_bin_int == goal_orientation_bin_int)
  {
    Path plan;
    plan.push_back(goal);
    return plan;
  }

  NodeHybrid::CoordinateVector path;
  int num_iterations = 0;
  std::unique_ptr<std::vector<std::tuple<float, float, float>>> expansions = nullptr;
  if (_config.debug_visualizations) {
    expansions = std::make_unique<std::vector<std::tuple<float, float, float>>>();
  }

  if (!_a_star->createPath(
      path, num_iterations,
      _config.tolerance / static_cast<float>(costmap->getResolution()), cancel_checker,
      expansions.get()))
  {
    if (num_iterations == 1) {
      throw std::runtime_error("Start occupied");
    }

    if (num_iterations < _a_star->getMaxIterations()) {
      throw std::runtime_error("No valid path found");
    } else {
      throw std::runtime_error("Exceeded maximum iterations");
    }
  }

  Path plan;
  plan.reserve(path.size());
  for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
    Pose pose = getWorldCoords(path[i].x, path[i].y, costmap);
    pose.theta = path[i].theta;
    plan.push_back(pose);
  }

  steady_clock::time_point b = steady_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(b - a);
  double time_remaining = _config.max_planning_time - time_span.count();

#ifdef BENCHMARK_TESTING
  std::cout << "It took " << time_span.count() * 1000 <<
    " milliseconds with " << num_iterations << " iterations." << std::endl;
#endif

  if (_smoother && num_iterations > 1) {
    _smoother->smooth(plan, costmap, time_remaining);
  }

#ifdef BENCHMARK_TESTING
  steady_clock::time_point c = steady_clock::now();
  duration<double> time_span2 = duration_cast<duration<double>>(c - b);
  std::cout << "It took " << time_span2.count() * 1000 <<
    " milliseconds to smooth path." << std::endl;
#endif

  return plan;
}

}  // namespace hybrid_astar
