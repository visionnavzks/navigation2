#include <math.h>
#include <chrono>
#include <vector>
#include <memory>
#include <algorithm>
#include <queue>
#include <limits>
#include <utility>

#include "my/hybrid_astar/node_hybrid.hpp"
#include "my/hybrid_astar/steering_state_space.hpp"

using namespace std::chrono;  // NOLINT

namespace hybrid_astar
{

void HybridMotionTable::initCommon(
  const unsigned int & size_x_in,
  const unsigned int & size_y_in,
  const unsigned int & num_angle_quantization_in,
  SearchInfo & search_info,
  MotionModel model)
{
  size_x = size_x_in;
  size_y = size_y_in;
  change_penalty = search_info.change_penalty;
  non_straight_penalty = search_info.non_straight_penalty;
  cost_penalty = search_info.cost_penalty;
  reverse_penalty = search_info.reverse_penalty;
  travel_distance_reward = 1.0f - search_info.retrospective_penalty;
  downsample_obstacle_heuristic = search_info.downsample_obstacle_heuristic;
  use_quadratic_cost_penalty = search_info.use_quadratic_cost_penalty;

  if (num_angle_quantization_in == num_angle_quantization &&
    min_turning_radius == search_info.minimum_turning_radius &&
    allow_primitive_interpolation == search_info.allow_primitive_interpolation &&
    motion_model == model)
  {
    return;
  }

  num_angle_quantization = num_angle_quantization_in;
  num_angle_quantization_float = static_cast<float>(num_angle_quantization);
  min_turning_radius = search_info.minimum_turning_radius;
  allow_primitive_interpolation = search_info.allow_primitive_interpolation;
  motion_model = model;

  float asin_arg = std::min(1.0, sqrt(2.0) / (2 * min_turning_radius));
  float angle = 2.0 * asin(asin_arg);
  bin_size =
    2.0f * static_cast<float>(M_PI) / static_cast<float>(num_angle_quantization);
  float increments;
  if (angle < bin_size) {
    increments = 1.0f;
  } else {
    increments = ceil(angle / bin_size);
  }
  angle = increments * bin_size;

  const float delta_x = min_turning_radius * sin(angle);
  const float delta_y = min_turning_radius - (min_turning_radius * cos(angle));
  const float delta_dist = hypotf(delta_x, delta_y);

  projections.clear();

  // Forward + Left + Right (shared by both models)
  projections.emplace_back(delta_dist, 0.0, 0.0, TurnDirection::FORWARD);
  projections.emplace_back(delta_x, delta_y, increments, TurnDirection::LEFT);
  projections.emplace_back(delta_x, -delta_y, -increments, TurnDirection::RIGHT);

  if (model == MotionModel::REEDS_SHEPP) {
    projections.emplace_back(-delta_dist, 0.0, 0.0, TurnDirection::REVERSE);
    projections.emplace_back(-delta_x, delta_y, -increments, TurnDirection::REV_LEFT);
    projections.emplace_back(-delta_x, -delta_y, increments, TurnDirection::REV_RIGHT);
  }

  const unsigned int base_count = (model == MotionModel::REEDS_SHEPP) ? 6u : 3u;

  if (search_info.allow_primitive_interpolation && increments > 1.0f) {
    projections.reserve(base_count + (2 * base_count * (increments - 1)));
    for (unsigned int i = 1; i < static_cast<unsigned int>(increments); i++) {
      const float angle_n = static_cast<float>(i) * bin_size;
      const float turning_rad_n = delta_dist / (2.0f * sin(angle_n / 2.0f));
      const float delta_x_n = turning_rad_n * sin(angle_n);
      const float delta_y_n = turning_rad_n - (turning_rad_n * cos(angle_n));
      projections.emplace_back(
        delta_x_n, delta_y_n, static_cast<float>(i), TurnDirection::LEFT);
      projections.emplace_back(
        delta_x_n, -delta_y_n, -static_cast<float>(i), TurnDirection::RIGHT);
      if (model == MotionModel::REEDS_SHEPP) {
        projections.emplace_back(
          -delta_x_n, delta_y_n, -static_cast<float>(i), TurnDirection::REV_LEFT);
        projections.emplace_back(
          -delta_x_n, -delta_y_n, static_cast<float>(i), TurnDirection::REV_RIGHT);
      }
    }
  }

  state_space = createSteeringStateSpace(model, min_turning_radius);

  delta_xs.resize(projections.size());
  delta_ys.resize(projections.size());
  trig_values.resize(num_angle_quantization);

  for (unsigned int i = 0; i != projections.size(); i++) {
    delta_xs[i].resize(num_angle_quantization);
    delta_ys[i].resize(num_angle_quantization);

    for (unsigned int j = 0; j != num_angle_quantization; j++) {
      double cos_theta = cos(bin_size * j);
      double sin_theta = sin(bin_size * j);
      if (i == 0) {
        trig_values[j] = {cos_theta, sin_theta};
      }
      delta_xs[i][j] = projections[i]._x * cos_theta - projections[i]._y * sin_theta;
      delta_ys[i][j] = projections[i]._x * sin_theta + projections[i]._y * cos_theta;
    }
  }

  travel_costs.resize(projections.size());
  for (unsigned int i = 0; i != projections.size(); i++) {
    const TurnDirection turn_dir = projections[i]._turn_dir;
    if (turn_dir != TurnDirection::FORWARD && turn_dir != TurnDirection::REVERSE) {
      const float arc_angle = projections[i]._theta * bin_size;
      const float turning_rad = delta_dist / (2.0f * sin(arc_angle / 2.0f));
      travel_costs[i] = turning_rad * arc_angle;
    } else {
      travel_costs[i] = delta_dist;
    }
  }
}

void HybridMotionTable::initDubin(
  const unsigned int & size_x_in,
  const unsigned int & size_y_in,
  const unsigned int & num_angle_quantization_in,
  SearchInfo & search_info)
{
  initCommon(size_x_in, size_y_in, num_angle_quantization_in, search_info, MotionModel::DUBIN);
}

void HybridMotionTable::initReedsShepp(
  const unsigned int & size_x_in,
  const unsigned int & size_y_in,
  const unsigned int & num_angle_quantization_in,
  SearchInfo & search_info)
{
  initCommon(size_x_in, size_y_in, num_angle_quantization_in, search_info, MotionModel::REEDS_SHEPP);
}

MotionPoses HybridMotionTable::getProjections(const NodeHybrid * node)
{
  MotionPoses projection_list;
  projection_list.reserve(projections.size());

  for (unsigned int i = 0; i != projections.size(); i++) {
    const MotionPose & proj_motion_model = projections[i];

    const float & node_heading = node->pose.theta;
    float new_heading = node_heading + proj_motion_model._theta;
    new_heading = static_cast<float>(wrapBinIndex(
      static_cast<int>(new_heading), num_angle_quantization));

    projection_list.emplace_back(
      delta_xs[i][node_heading] + node->pose.x,
      delta_ys[i][node_heading] + node->pose.y,
      new_heading, proj_motion_model._turn_dir);
  }

  return projection_list;
}

unsigned int HybridMotionTable::getClosestAngularBin(const double & theta) const
{
  auto bin = static_cast<unsigned int>(round(wrapAngle(theta) / bin_size));
  return bin < num_angle_quantization ? bin : 0u;
}

float HybridMotionTable::getAngleFromBin(const unsigned int & bin_idx) const
{
  return bin_idx * bin_size;
}

double HybridMotionTable::getAngle(const double & theta) const
{
  return theta / bin_size;
}

NodeHybrid::NodeHybrid(const uint64_t index, NodeContext * ctx)
: parent(nullptr),
  pose(0.0f, 0.0f, 0.0f),
  _cell_cost(std::numeric_limits<float>::quiet_NaN()),
  _accumulated_cost(std::numeric_limits<float>::max()),
  _index(index),
  _was_visited(false),
  _motion_primitive_index(std::numeric_limits<unsigned int>::max()),
  _is_node_valid(false),
  _ctx(ctx)
{
}

NodeHybrid::~NodeHybrid()
{
}

void NodeHybrid::reset()
{
  parent = nullptr;
  _cell_cost = std::numeric_limits<float>::quiet_NaN();
  _accumulated_cost = std::numeric_limits<float>::max();
  _was_visited = false;
  _motion_primitive_index = std::numeric_limits<unsigned int>::max();
  pose.x = 0.0f;
  pose.y = 0.0f;
  pose.theta = 0.0f;
  _is_node_valid = false;
}

bool NodeHybrid::isNodeValid(
  const bool & traverse_unknown,
  GridCollisionChecker * collision_checker)
{
  if (!std::isnan(_cell_cost)) {
    return _is_node_valid;
  }

  _is_node_valid = !collision_checker->inCollision(
    this->pose.x, this->pose.y, this->pose.theta, traverse_unknown);
  _cell_cost = collision_checker->getCost();
  return _is_node_valid;
}

float NodeHybrid::getTraversalCost(const NodePtr & child)
{
  const float normalized_cost = child->getCost() / MAX_NON_OBSTACLE_COST;
  if (std::isnan(normalized_cost)) {
    throw std::runtime_error(
            "Node attempted to get traversal "
            "cost without a known SE2 collision cost!");
  }

  const TurnDirection & child_turn_dir = child->getTurnDirection();
  float travel_cost_raw = _ctx->motion_table.travel_costs[child->getMotionPrimitiveIndex()];
  float travel_cost = 0.0;

  if (_ctx->motion_table.use_quadratic_cost_penalty) {
    travel_cost_raw *=
      (_ctx->motion_table.travel_distance_reward +
      (_ctx->motion_table.cost_penalty * normalized_cost * normalized_cost));
  } else {
    travel_cost_raw *=
      (_ctx->motion_table.travel_distance_reward + _ctx->motion_table.cost_penalty *
      normalized_cost);
  }

  if (child_turn_dir == TurnDirection::FORWARD || child_turn_dir == TurnDirection::REVERSE ||
    getMotionPrimitiveIndex() == std::numeric_limits<unsigned int>::max())
  {
    travel_cost = travel_cost_raw;
  } else {
    if (getTurnDirection() == child_turn_dir) {
      travel_cost = travel_cost_raw * _ctx->motion_table.non_straight_penalty;
    } else {
      travel_cost = travel_cost_raw *
        (_ctx->motion_table.non_straight_penalty + _ctx->motion_table.change_penalty);
    }
  }

  if (child_turn_dir == TurnDirection::REV_RIGHT ||
    child_turn_dir == TurnDirection::REV_LEFT ||
    child_turn_dir == TurnDirection::REVERSE)
  {
    travel_cost *= _ctx->motion_table.reverse_penalty;
  }

  return travel_cost;
}

float NodeHybrid::getHeuristicCost(
  const Coordinates & node_coords,
  const CoordinateVector & goals_coords)
{
  const float obstacle_heuristic =
    _ctx->obstacle_heuristic->getObstacleHeuristic(node_coords, _ctx->motion_table.cost_penalty,
      _ctx->motion_table.use_quadratic_cost_penalty,
      _ctx->motion_table.downsample_obstacle_heuristic);
  float distance_heuristic = std::numeric_limits<float>::max();
  for (unsigned int i = 0; i < goals_coords.size(); i++) {
    distance_heuristic = std::min(
      distance_heuristic,
      _ctx->distance_heuristic->getDistanceHeuristic(node_coords, goals_coords[i],
        obstacle_heuristic, _ctx->motion_table));
  }
  return std::max(obstacle_heuristic, distance_heuristic);
}

void NodeHybrid::initMotionModel(
  NodeContext * ctx,
  const MotionModel & motion_model,
  const unsigned int & size_x,
  const unsigned int & size_y,
  const unsigned int & num_angle_quantization,
  SearchInfo & search_info)
{
  switch (motion_model) {
    case MotionModel::DUBIN:
      ctx->motion_table.initDubin(size_x, size_y, num_angle_quantization, search_info);
      break;
    case MotionModel::REEDS_SHEPP:
      ctx->motion_table.initReedsShepp(size_x, size_y, num_angle_quantization, search_info);
      break;
    default:
      throw std::runtime_error(
              "Invalid motion model for Hybrid A*. Please select between"
              " Dubin (Ackermann forward only),"
              " Reeds-Shepp (Ackermann forward and back).");
  }
}

void NodeHybrid::getNeighbors(
  std::function<bool(const uint64_t &,
  hybrid_astar::NodeHybrid * &)> & NeighborGetter,
  GridCollisionChecker * collision_checker,
  const bool & traverse_unknown,
  NodeVector & neighbors)
{
  uint64_t index = 0;
  NodePtr neighbor = nullptr;
  Coordinates initial_node_coords;
  const MotionPoses & motion_projections = _ctx->motion_table.getProjections(this);

  for (unsigned int i = 0; i != motion_projections.size(); i++) {
    const int px_signed = static_cast<int>(motion_projections[i]._x);
    const int py_signed = static_cast<int>(motion_projections[i]._y);
    if (px_signed < 0 || py_signed < 0 ||
      static_cast<unsigned int>(px_signed) >= _ctx->motion_table.size_x ||
      static_cast<unsigned int>(py_signed) >= _ctx->motion_table.size_y)
    {
      continue;
    }
    index = NodeHybrid::getIndex(
      static_cast<unsigned int>(px_signed),
      static_cast<unsigned int>(py_signed),
      static_cast<unsigned int>(motion_projections[i]._theta),
      _ctx->motion_table.size_x, _ctx->motion_table.num_angle_quantization);

    if (NeighborGetter(index, neighbor) && !neighbor->wasVisited()) {
      initial_node_coords = neighbor->pose;
      neighbor->setPose(
        Coordinates(
          motion_projections[i]._x,
          motion_projections[i]._y,
          motion_projections[i]._theta));
      if (neighbor->isNodeValid(traverse_unknown, collision_checker)) {
        neighbor->setMotionPrimitiveIndex(i, motion_projections[i]._turn_dir);
        neighbors.push_back(neighbor);
      } else {
        neighbor->setPose(initial_node_coords);
      }
    }
  }
}

bool NodeHybrid::backtracePath(CoordinateVector & path)
{
  if (!this->parent) {
    return false;
  }

  NodePtr current_node = this;

  while (current_node->parent) {
    path.push_back(current_node->pose);
    path.back().theta = _ctx->motion_table.getAngleFromBin(path.back().theta);
    current_node = current_node->parent;
  }

  path.push_back(current_node->pose);
  path.back().theta = _ctx->motion_table.getAngleFromBin(path.back().theta);

  return true;
}

}  // namespace hybrid_astar
