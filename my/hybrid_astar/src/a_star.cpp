#include <cmath>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

#include "hybrid_astar/a_star.hpp"
using namespace std::chrono;  // NOLINT

namespace hybrid_astar
{

template<typename NodeT>
AStarAlgorithm<NodeT>::AStarAlgorithm(
  const MotionModel & motion_model,
  const SearchInfo & search_info)
: _traverse_unknown(true),
  _is_initialized(false),
  _max_iterations(0),
  _terminal_checking_interval(5000),
  _max_planning_time(0),
  _x_size(0),
  _y_size(0),
  _search_info(search_info),
  _start(nullptr),
  _goal_manager(GoalManagerT()),
  _motion_model(motion_model)
{
  _graph.reserve(100000);
}

template<typename NodeT>
AStarAlgorithm<NodeT>::~AStarAlgorithm()
{
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::initialize(
  const bool & allow_unknown,
  int & max_iterations,
  const int & max_on_approach_iterations,
  const int & terminal_checking_interval,
  const double & max_planning_time,
  const float & lookup_table_size,
  const unsigned int & dim_3_size)
{
  _traverse_unknown = allow_unknown;
  _max_iterations = max_iterations;
  _max_on_approach_iterations = max_on_approach_iterations;
  _terminal_checking_interval = terminal_checking_interval;
  _max_planning_time = max_planning_time;
  if (!_is_initialized) {
    _shared_ctx = std::make_shared<NodeContext>();
    _shared_ctx->distance_heuristic->precomputeDistanceHeuristic(lookup_table_size, _motion_model,
      dim_3_size,
      _search_info, _shared_ctx->motion_table);
  }
  _is_initialized = true;
  _dim3_size = dim_3_size;
  _expander = std::make_unique<AnalyticExpansion<NodeT>>(
    _motion_model, _search_info, _traverse_unknown, _dim3_size);
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::setCollisionChecker(GridCollisionChecker * collision_checker)
{
  _collision_checker = collision_checker;
  _costmap = collision_checker->getCostmap();
  unsigned int x_size = _costmap->getSizeInCellsX();
  unsigned int y_size = _costmap->getSizeInCellsY();

  clearGraph();

  if (getSizeX() != x_size || getSizeY() != y_size) {
    _x_size = x_size;
    _y_size = y_size;
  }

  NodeT::initMotionModel(_shared_ctx.get(), _motion_model, _x_size, _y_size, _dim3_size,
      _search_info);

  _goal_manager.setContext(_shared_ctx.get());
  _expander->setContext(_shared_ctx.get());
  _expander->setCollisionChecker(_collision_checker);
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::setEsdfResources(
  EsdfHolder * holder,
  const std::vector<double> & cost_check_points,
  double robot_radius,
  double safe_distance)
{
  if (_shared_ctx && _shared_ctx->obstacle_heuristic) {
    _shared_ctx->obstacle_heuristic->setEsdfHolder(holder);
    _shared_ctx->obstacle_heuristic->setEsdfFootprintParams(
      cost_check_points, robot_radius, safe_distance);
  }
}

template<typename NodeT>
typename AStarAlgorithm<NodeT>::NodePtr AStarAlgorithm<NodeT>::addToGraph(
  const uint64_t & index)
{
  auto iter = _graph.find(index);
  if (iter != _graph.end()) {
    return &(iter->second);
  }

  return &(_graph.emplace(index, NodeT(index, _shared_ctx.get())).first->second);
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::setStart(
  const float & mx,
  const float & my,
  const unsigned int & dim_3)
{
  _start = addToGraph(
    getIndex(
      static_cast<unsigned int>(mx),
      static_cast<unsigned int>(my),
      dim_3));
  _start->setPose(Coordinates(mx, my, dim_3));
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::populateExpansionsLog(
  const NodePtr & node,
  std::vector<std::tuple<float, float, float>> * expansions_log)
{
  typename NodeT::Coordinates coords = node->pose;
  expansions_log->emplace_back(
    _costmap->getOriginX() + ((coords.x + 0.5) * _costmap->getResolution()),
    _costmap->getOriginY() + ((coords.y + 0.5) * _costmap->getResolution()),
    _shared_ctx->motion_table.getAngleFromBin(coords.theta));
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::setGoal(
  const float & mx,
  const float & my,
  const unsigned int & dim_3,
  const GoalHeadingMode & goal_heading_mode,
  const int & coarse_search_resolution)
{
  _coarse_search_resolution = coarse_search_resolution;

  _goal_manager.clear();
  Coordinates ref_goal_coord(mx, my, static_cast<float>(dim_3));

  if (!_search_info.cache_obstacle_heuristic ||
    _goal_manager.hasGoalChanged(ref_goal_coord))
  {
    if (!_start) {
      throw std::runtime_error("Start must be set before goal.");
    }

    _shared_ctx->obstacle_heuristic->resetObstacleHeuristic(
      _collision_checker->getCostmap(), _start->pose.x, _start->pose.y, mx, my,
        _shared_ctx->motion_table.downsample_obstacle_heuristic);
  }

  _goal_manager.setRefGoalCoordinates(ref_goal_coord);

  unsigned int num_bins = _shared_ctx->motion_table.num_angle_quantization;
  switch (goal_heading_mode) {
    case GoalHeadingMode::DEFAULT: {
        auto goal = addToGraph(
          getIndex(
            static_cast<unsigned int>(mx),
            static_cast<unsigned int>(my),
            dim_3));
        goal->setPose(typename NodeT::Coordinates(mx, my, static_cast<float>(dim_3)));
        _goal_manager.addGoal(goal);
        break;
      }

    case GoalHeadingMode::BIDIRECTIONAL: {
        auto goal = addToGraph(
          getIndex(
            static_cast<unsigned int>(mx),
            static_cast<unsigned int>(my),
            dim_3));
        goal->setPose(typename NodeT::Coordinates(mx, my, static_cast<float>(dim_3)));
        _goal_manager.addGoal(goal);

        unsigned int opposite_heading = (dim_3 + (num_bins / 2)) % num_bins;
        auto opposite_goal = addToGraph(
          getIndex(
            static_cast<unsigned int>(mx),
            static_cast<unsigned int>(my),
            opposite_heading));
        opposite_goal->setPose(
          typename NodeT::Coordinates(mx, my, static_cast<float>(opposite_heading)));
        _goal_manager.addGoal(opposite_goal);
        break;
      }

    case GoalHeadingMode::ALL_DIRECTION: {
        for (unsigned int i = 0; i < num_bins; ++i) {
          auto goal = addToGraph(
            getIndex(
              static_cast<unsigned int>(mx),
              static_cast<unsigned int>(my),
              i));
          goal->setPose(typename NodeT::Coordinates(mx, my, static_cast<float>(i)));
          _goal_manager.addGoal(goal);
        }
        break;
      }
    case GoalHeadingMode::UNKNOWN:
      throw std::runtime_error("Goal heading is UNKNOWN.");
  }
}

template<typename NodeT>
bool AStarAlgorithm<NodeT>::areInputsValid()
{
  if (_graph.empty()) {
    throw std::runtime_error("Failed to compute path, no costmap given.");
  }

  if (!_start || _goal_manager.goalsIsEmpty()) {
    throw std::runtime_error("Failed to compute path, no valid start or goal given.");
  }

  _goal_manager.removeInvalidGoals(getToleranceHeuristic(), _collision_checker, _traverse_unknown);

  if (_goal_manager.getGoalsSet().empty()) {
    throw std::runtime_error("Goal was in lethal cost");
  }

  return true;
}

template<typename NodeT>
bool AStarAlgorithm<NodeT>::getClosestPathWithinTolerance(CoordinateVector & path)
{
  if (_best_heuristic_node.first < getToleranceHeuristic()) {
    _graph.at(_best_heuristic_node.second).backtracePath(path);
    return true;
  }

  return false;
}

template<typename NodeT>
bool AStarAlgorithm<NodeT>::createPath(
  CoordinateVector & path, int & iterations,
  const float & tolerance,
  std::function<bool()> cancel_checker,
  std::vector<std::tuple<float, float, float>> * expansions_log)
{
  steady_clock::time_point start_time = steady_clock::now();
  _tolerance = tolerance;
  _best_heuristic_node = {std::numeric_limits<float>::max(), 0};
  clearQueue();

  if (!areInputsValid()) {
    return false;
  }

  NodeVector coarse_check_goals, fine_check_goals;
  _goal_manager.prepareGoalsForAnalyticExpansion(
    coarse_check_goals, fine_check_goals,
    _coarse_search_resolution);

  addNode(0.0, getStart());
  getStart()->setAccumulatedCost(0.0);

  NodePtr current_node = nullptr;
  NodePtr neighbor = nullptr;
  NodePtr expansion_result = nullptr;
  float g_cost = 0.0;
  NodeVector neighbors;
  int approach_iterations = 0;
  NeighborIterator neighbor_iterator;
  int analytic_iterations = 0;
  int closest_distance = std::numeric_limits<int>::max();

  const uint64_t max_index = static_cast<uint64_t>(getSizeX()) *
    static_cast<uint64_t>(getSizeY()) *
    static_cast<uint64_t>(getSizeDim3());
  NodeGetter neighborGetter =
    [&, this](const uint64_t & index, NodePtr & neighbor_rtn) -> bool
    {
      if (index >= max_index) {
        return false;
      }

      neighbor_rtn = addToGraph(index);
      return true;
    };

  while (iterations < getMaxIterations() && !_queue.empty()) {
    if (iterations % _terminal_checking_interval == 0) {
      if (cancel_checker()) {
        throw std::runtime_error("Planner was cancelled");
      }
      std::chrono::duration<double> planning_duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(steady_clock::now() - start_time);
      if (static_cast<double>(planning_duration.count()) >= _max_planning_time) {
        return getClosestPathWithinTolerance(path);
      }
    }

    current_node = getNextNode();

    if (expansions_log) {
      populateExpansionsLog(current_node, expansions_log);
    }

    if (current_node->wasVisited()) {
      continue;
    }

    iterations++;

    current_node->visited();

    expansion_result = nullptr;
    expansion_result = _expander->tryAnalyticExpansion(
      current_node, coarse_check_goals, fine_check_goals,
      _goal_manager.getGoalsCoordinates(), neighborGetter, analytic_iterations, closest_distance);
    if (expansion_result != nullptr) {
      current_node = expansion_result;
    }

    if (_goal_manager.isGoal(current_node)) {
      return current_node->backtracePath(path);
    } else if (_best_heuristic_node.first < getToleranceHeuristic()) {
      approach_iterations++;
      if (approach_iterations >= getOnApproachMaxIterations()) {
        return _graph.at(_best_heuristic_node.second).backtracePath(path);
      }
    }

    neighbors.clear();
    current_node->getNeighbors(neighborGetter, _collision_checker, _traverse_unknown, neighbors);

    for (neighbor_iterator = neighbors.begin();
      neighbor_iterator != neighbors.end(); ++neighbor_iterator)
    {
      neighbor = *neighbor_iterator;

      g_cost = current_node->getAccumulatedCost() + current_node->getTraversalCost(neighbor);

      if (g_cost < neighbor->getAccumulatedCost()) {
        neighbor->setAccumulatedCost(g_cost);
        neighbor->parent = current_node;

        addNode(g_cost + getHeuristicCost(neighbor), neighbor);
      }
    }
  }

  return getClosestPathWithinTolerance(path);
}

template<typename NodeT>
typename AStarAlgorithm<NodeT>::NodePtr & AStarAlgorithm<NodeT>::getStart()
{
  return _start;
}

template<typename NodeT>
typename AStarAlgorithm<NodeT>::NodePtr AStarAlgorithm<NodeT>::getNextNode()
{
  NodeBasic<NodeT> node = _queue.top().second;
  _queue.pop();
  node.processSearchNode();
  return node.graph_node_ptr;
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::addNode(const float & cost, NodePtr & node)
{
  NodeBasic<NodeT> queued_node(node->getIndex());
  queued_node.populateSearchNode(node);
  _queue.emplace(cost, queued_node);
}

template<typename NodeT>
float AStarAlgorithm<NodeT>::getHeuristicCost(const NodePtr & node)
{
  const Coordinates node_coords =
    NodeT::getCoords(node->getIndex(), getSizeX(), getSizeDim3());
  float heuristic = node->getHeuristicCost(node_coords, _goal_manager.getGoalsCoordinates());
  if (heuristic < _best_heuristic_node.first) {
    _best_heuristic_node = {heuristic, node->getIndex()};
  }

  return heuristic;
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::clearQueue()
{
  NodeQueue q;
  std::swap(_queue, q);
}

template<typename NodeT>
void AStarAlgorithm<NodeT>::clearGraph()
{
  Graph g;
  std::swap(_graph, g);
  _graph.reserve(100000);
}

template<typename NodeT>
uint64_t AStarAlgorithm<NodeT>::getIndex(
  const unsigned int & x, const unsigned int & y,
  const unsigned int & dim_3)
{
  return NodeT::getIndex(x, y, dim_3, _shared_ctx->motion_table.size_x,
      _shared_ctx->motion_table.num_angle_quantization);
}

template<typename NodeT>
int AStarAlgorithm<NodeT>::getMaxIterations() const
{
  return _max_iterations;
}

template<typename NodeT>
int AStarAlgorithm<NodeT>::getOnApproachMaxIterations() const
{
  return _max_on_approach_iterations;
}

template<typename NodeT>
float AStarAlgorithm<NodeT>::getToleranceHeuristic() const
{
  return _tolerance;
}

template<typename NodeT>
unsigned int AStarAlgorithm<NodeT>::getSizeX() const
{
  return _x_size;
}

template<typename NodeT>
unsigned int AStarAlgorithm<NodeT>::getSizeY() const
{
  return _y_size;
}

template<typename NodeT>
unsigned int AStarAlgorithm<NodeT>::getSizeDim3() const
{
  return _dim3_size;
}

template<typename NodeT>
unsigned int AStarAlgorithm<NodeT>::getCoarseSearchResolution() const
{
  return _coarse_search_resolution;
}

template<typename NodeT>
const typename AStarAlgorithm<NodeT>::GoalManagerT &
AStarAlgorithm<NodeT>::getGoalManager() const
{
  return _goal_manager;
}

template<typename NodeT>
typename AStarAlgorithm<NodeT>::NodeContext * AStarAlgorithm<NodeT>::getContext()
{
  return _shared_ctx.get();
}

template class AStarAlgorithm<NodeHybrid>;

}  // namespace hybrid_astar
