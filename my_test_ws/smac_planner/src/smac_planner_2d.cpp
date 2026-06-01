#include <string>
#include <memory>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <chrono>

#include "my/smac_planner/smac_planner_2d.hpp"

// #define BENCHMARK_TESTING

namespace smac_planner
{

using namespace std::chrono;

SmacPlanner2D::SmacPlanner2D()
: _a_star(nullptr),
  _collision_checker(nullptr, 1),
  _smoother(nullptr),
  _costmap(nullptr),
  _costmap_downsampler(nullptr)
{
}

SmacPlanner2D::~SmacPlanner2D()
{
}

void SmacPlanner2D::configure(
  Costmap2D * costmap,
  const SmacPlanner2DConfig & config)
{
  _costmap = costmap;
  _config = config;

  int max_iterations = _config.max_iterations;
  int max_on_approach_iterations = _config.max_on_approach_iterations;

  if (max_on_approach_iterations <= 0) {
    max_on_approach_iterations = std::numeric_limits<int>::max();
  }

  if (max_iterations <= 0) {
    max_iterations = std::numeric_limits<int>::max();
  }

  _collision_checker = GridCollisionChecker(_costmap, 1);
  _collision_checker.setFootprint(
    _config.robot_footprint,
    true,
    0.0);

  _a_star = std::make_unique<AStarAlgorithm<Node2D>>(MotionModel::TWOD, _config.search_info);
  _a_star->initialize(
    _config.allow_unknown,
    max_iterations,
    max_on_approach_iterations,
    _config.terminal_checking_interval,
    _config.max_planning_time,
    0.0,
    1.0);

  SmootherParams smoother_params = _config.smoother_params;
  smoother_params.holonomic_ = true;
  _smoother = std::make_unique<Smoother>(smoother_params);
  _smoother->initialize(1e-50);

  _costmap_downsampler = std::make_unique<CostmapDownsampler>();
  _costmap_downsampler->on_configure(_costmap, _config.downsampling_factor);
}

void SmacPlanner2D::setFootprint(
  const Footprint & footprint, bool use_radius)
{
  _config.robot_footprint = footprint;
  _config.use_radius = use_radius;
  _collision_checker.setFootprint(footprint, true, 0.0);
}

Path SmacPlanner2D::createPlan(
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

  _a_star->setCollisionChecker(&_collision_checker);

  float mx_start, my_start, mx_goal, my_goal;
  if (!costmap->worldToMapContinuous(start.x, start.y, mx_start, my_start)) {
    throw std::runtime_error(
      "Start Coordinates of(" + std::to_string(start.x) + ", " +
      std::to_string(start.y) + ") was outside bounds");
  }
  _a_star->setStart(mx_start, my_start, 0);

  if (!costmap->worldToMapContinuous(goal.x, goal.y, mx_goal, my_goal)) {
    throw std::runtime_error(
      "Goal Coordinates of(" + std::to_string(goal.x) + ", " +
      std::to_string(goal.y) + ") was outside bounds");
  }
  _a_star->setGoal(mx_goal, my_goal, 0);

  if (static_cast<int>(mx_start) == static_cast<int>(mx_goal) &&
    static_cast<int>(my_start) == static_cast<int>(my_goal))
  {
    Path plan;
    Pose p = goal;
    if (start.theta != goal.theta && !_config.use_final_approach_orientation) {
      p.theta = goal.theta;
    } else if (_config.use_final_approach_orientation) {
      p.theta = start.theta;
    }
    plan.push_back(p);
    return plan;
  }

  Node2D::CoordinateVector path;
  int num_iterations = 0;

  if (!_a_star->createPath(
      path, num_iterations,
      _config.tolerance / static_cast<float>(costmap->getResolution()), cancel_checker))
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
    plan.push_back(pose);
  }

  steady_clock::time_point b = steady_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(b - a);
  double time_remaining = _config.max_planning_time - time_span.count();

#ifdef BENCHMARK_TESTING
  std::cout << "It took " << time_span.count() * 1000 <<
    " milliseconds with " << num_iterations << " iterations." << std::endl;
#endif

  _smoother->smooth(plan, costmap, time_remaining);

  if (_config.use_final_approach_orientation) {
    if (plan.size() == 1) {
      plan.back().theta = start.theta;
    } else if (plan.size() > 1) {
      double dx = plan.back().x - plan[plan.size() - 2].x;
      double dy = plan.back().y - plan[plan.size() - 2].y;
      plan.back().theta = atan2(dy, dx);
    }
  } else if (!plan.empty()) {
    plan.back().theta = goal.theta;
  }

  return plan;
}

}  // namespace smac_planner
