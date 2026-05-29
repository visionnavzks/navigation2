#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>

#include "my/smac_planner/smac_planner_lattice.hpp"

// #define BENCHMARK_TESTING

namespace smac_planner
{

using namespace std::chrono;

SmacPlannerLattice::SmacPlannerLattice()
: _a_star(nullptr),
  _collision_checker(nullptr, 1),
  _smoother(nullptr),
  _costmap(nullptr)
{
}

SmacPlannerLattice::~SmacPlannerLattice()
{
}

void SmacPlannerLattice::configure(
  Costmap2D * costmap,
  const SmacPlannerLatticeConfig & config)
{
  _costmap = costmap;
  _config = config;

  _metadata = LatticeMotionTable::getLatticeMetadata(_config.search_info.lattice_filepath);
  _config.search_info.minimum_turning_radius =
    _metadata.min_turning_radius / _costmap->getResolution();

  if (_metadata.motion_model == "omni" && _config.search_info.allow_reverse_expansion) {
    _config.search_info.allow_reverse_expansion = false;
  }

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

  if (_metadata.number_of_headings % static_cast<unsigned int>(_config.coarse_search_resolution) != 0) {
    throw std::runtime_error(
      "coarse iteration should be an increment of the number of angular bins configured");
  }

  _goal_heading_mode = fromStringToGH(_config.goal_heading_mode);
  if (_goal_heading_mode == GoalHeadingMode::UNKNOWN) {
    throw std::runtime_error(
      "Unable to get GoalHeader type. Given '" + _config.goal_heading_mode + "' "
      "Valid options are DEFAULT, BIDIRECTIONAL, ALL_DIRECTION.");
  }

  float lookup_table_dim =
    static_cast<float>(_config.lookup_table_size) /
    static_cast<float>(_costmap->getResolution());

  lookup_table_dim = static_cast<float>(static_cast<int>(lookup_table_dim));

  if (static_cast<int>(lookup_table_dim) % 2 == 0) {
    lookup_table_dim += 1.0f;
  }
  _lookup_table_dim = lookup_table_dim;

  double circumscribed_cost = _config.circumscribed_cost;
  if (circumscribed_cost < 0.0) {
    circumscribed_cost = findCircumscribedCost(
      _costmap, _config.circumscribed_radius, _config.inflation_radius);
  }

  _collision_checker = GridCollisionChecker(_costmap, 72u);
  _collision_checker.setFootprint(
    _config.robot_footprint,
    _config.use_radius,
    circumscribed_cost);

  _a_star = std::make_unique<AStarAlgorithm<NodeLattice>>(
    MotionModel::STATE_LATTICE, _config.search_info);
  _a_star->initialize(
    _config.allow_unknown,
    max_iterations,
    max_on_approach_iterations,
    _config.terminal_checking_interval,
    _config.max_planning_time,
    _lookup_table_dim,
    _metadata.number_of_headings);

  if (_config.smooth_path) {
    SmootherParams smoother_params = _config.smoother_params;
    if (_metadata.motion_model == "omni") {
      smoother_params.holonomic_ = true;
    }
    _smoother = std::make_unique<Smoother>(smoother_params);
    _smoother->initialize(_metadata.min_turning_radius);
  }
}

void SmacPlannerLattice::setFootprint(
  const Footprint & footprint, bool use_radius, double circumscribed_cost)
{
  _config.robot_footprint = footprint;
  _config.use_radius = use_radius;
  if (circumscribed_cost < 0.0) {
    circumscribed_cost = findCircumscribedCost(
      _costmap, _config.circumscribed_radius, _config.inflation_radius);
  }
  _config.circumscribed_cost = circumscribed_cost;
  _collision_checker.setFootprint(footprint, use_radius, circumscribed_cost);
}

Path SmacPlannerLattice::createPlan(
  const Pose & start,
  const Pose & goal,
  std::function<bool()> cancel_checker)
{
  std::lock_guard<std::mutex> lock_reinit(_mutex);
  steady_clock::time_point a = steady_clock::now();

  _a_star->setCollisionChecker(&_collision_checker);

  float mx_start, my_start, mx_goal, my_goal;
  if (!_costmap->worldToMapContinuous(start.x, start.y, mx_start, my_start)) {
    throw std::runtime_error(
      "Start Coordinates of(" + std::to_string(start.x) + ", " +
      std::to_string(start.y) + ") was outside bounds");
  }
  unsigned int start_bin =
    _a_star->getContext()->motion_table.getClosestAngularBin(start.theta);
  _a_star->setStart(mx_start, my_start, start_bin);

  if (!_costmap->worldToMapContinuous(goal.x, goal.y, mx_goal, my_goal)) {
    throw std::runtime_error(
      "Goal Coordinates of(" + std::to_string(goal.x) + ", " +
      std::to_string(goal.y) + ") was outside bounds");
  }
  unsigned int goal_bin =
    _a_star->getContext()->motion_table.getClosestAngularBin(goal.theta);
  _a_star->setGoal(
    mx_goal, my_goal, goal_bin,
    _goal_heading_mode, _config.coarse_search_resolution);

  if (static_cast<int>(mx_start) == static_cast<int>(mx_goal) &&
    static_cast<int>(my_start) == static_cast<int>(my_goal) &&
    start_bin == goal_bin)
  {
    Path plan;
    plan.push_back(goal);
    return plan;
  }

  NodeLattice::CoordinateVector path;
  int num_iterations = 0;
  std::unique_ptr<std::vector<std::tuple<float, float, float>>> expansions = nullptr;
  if (_config.debug_visualizations) {
    expansions = std::make_unique<std::vector<std::tuple<float, float, float>>>();
  }

  if (!_a_star->createPath(
      path, num_iterations,
      _config.tolerance / static_cast<float>(_costmap->getResolution()), cancel_checker,
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
  Pose last_pose;
  bool first = true;
  for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
    Pose pose = getWorldCoords(path[i].x, path[i].y, _costmap);
    pose.theta = path[i].theta;
    if (!first && fabs(pose.x - last_pose.x) < 1e-4 &&
      fabs(pose.y - last_pose.y) < 1e-4 &&
      fabs(pose.theta - last_pose.theta) < 1e-4)
    {
      continue;
    }
    first = false;
    last_pose = pose;
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
    _smoother->smooth(plan, _costmap, time_remaining);
  }

#ifdef BENCHMARK_TESTING
  steady_clock::time_point c = steady_clock::now();
  duration<double> time_span2 = duration_cast<duration<double>>(c - b);
  std::cout << "It took " << time_span2.count() * 1000 <<
    " milliseconds to smooth path." << std::endl;
#endif

  return plan;
}

}  // namespace smac_planner
