#include <algorithm>
#include <vector>
#include <memory>

#include "my/hybrid_astar/analytic_expansion.hpp"

namespace hybrid_astar
{

template<typename NodeT>
AnalyticExpansion<NodeT>::AnalyticExpansion(
  const MotionModel & motion_model,
  const SearchInfo & search_info,
  const bool & traverse_unknown,
  const unsigned int & dim_3_size)
: _motion_model(motion_model),
  _search_info(search_info),
  _traverse_unknown(traverse_unknown),
  _dim_3_size(dim_3_size),
  _collision_checker(nullptr)
{
}

template<typename NodeT>
void AnalyticExpansion<NodeT>::setCollisionChecker(
  GridCollisionChecker * collision_checker)
{
  _collision_checker = collision_checker;
}

template<typename NodeT>
void AnalyticExpansion<NodeT>::setContext(NodeContext * ctx)
{
  _ctx = ctx;
}

template<typename NodeT>
typename AnalyticExpansion<NodeT>::NodePtr AnalyticExpansion<NodeT>::tryAnalyticExpansion(
  const NodePtr & current_node,
  const NodeVector & coarse_check_goals,
  const NodeVector & fine_check_goals,
  const CoordinateVector & goals_coords,
  const NodeGetter & getter, int & analytic_iterations,
  int & closest_distance)
{
  if (_motion_model == MotionModel::DUBIN || _motion_model == MotionModel::REEDS_SHEPP) {
    const Coordinates node_coords =
      NodeT::getCoords(
      current_node->getIndex(), _collision_checker->getCostmap()->getSizeInCellsX(), _dim_3_size);

    AnalyticExpansionNodes current_best_analytic_nodes;
    NodePtr current_best_goal = nullptr;
    NodePtr current_best_node = nullptr;
    float current_best_score = std::numeric_limits<float>::max();

    closest_distance = std::min(
      closest_distance,
      static_cast<int>(current_node->getHeuristicCost(node_coords, goals_coords)));
    int desired_iterations = std::max(
      static_cast<int>(closest_distance / _search_info.analytic_expansion_ratio),
      static_cast<int>(std::ceil(_search_info.analytic_expansion_ratio)));

    analytic_iterations =
      std::min(analytic_iterations, desired_iterations);

    if (analytic_iterations <= 0) {
      analytic_iterations = desired_iterations;
      bool found_valid_expansion = false;

      for (auto & current_goal_node : coarse_check_goals) {
        AnalyticExpansionNodes analytic_nodes =
          getAnalyticPath(
          current_node, current_goal_node, getter,
          _ctx->motion_table.state_space);
        if (!analytic_nodes.nodes.empty()) {
          found_valid_expansion = true;
          NodePtr node = current_node;
          float score = refineAnalyticPath(
            node, current_goal_node, getter, analytic_nodes);
          if (score < current_best_score) {
            current_best_analytic_nodes = analytic_nodes;
            current_best_goal = current_goal_node;
            current_best_score = score;
            current_best_node = node;
          }
        }
      }

      if (found_valid_expansion) {
        for (auto & current_goal_node : fine_check_goals) {
          AnalyticExpansionNodes analytic_nodes =
            getAnalyticPath(
            current_node, current_goal_node, getter,
            _ctx->motion_table.state_space);
          if (!analytic_nodes.nodes.empty()) {
            NodePtr node = current_node;
            float score = refineAnalyticPath(
              node, current_goal_node, getter, analytic_nodes);
            if (score < current_best_score) {
              current_best_analytic_nodes = analytic_nodes;
              current_best_goal = current_goal_node;
              current_best_score = score;
              current_best_node = node;
            }
          }
        }
      }
    }

    if (!current_best_analytic_nodes.nodes.empty()) {
      return setAnalyticPath(
        current_best_node, current_best_goal,
        current_best_analytic_nodes);
    }
    analytic_iterations--;
  }

  return NodePtr(nullptr);
}

template<typename NodeT>
int AnalyticExpansion<NodeT>::countDirectionChanges(
  const ompl::base::ReedsSheppStateSpace::ReedsSheppPath & path)
{
  const double * lengths = path.length_;
  int changes = 0;
  int last_dir = 0;
  for (int i = 0; i < 5; ++i) {
    if (lengths[i] == 0.0) {
      continue;
    }

    int currentDirection = (lengths[i] > 0.0) ? 1 : -1;
    if (last_dir != 0 && currentDirection != last_dir) {
      ++changes;
    }
    last_dir = currentDirection;
  }

  return changes;
}

template<typename NodeT>
typename AnalyticExpansion<NodeT>::AnalyticExpansionNodes AnalyticExpansion<NodeT>::getAnalyticPath(
  const NodePtr & node,
  const NodePtr & goal,
  const NodeGetter & node_getter,
  const ompl::base::StateSpacePtr & state_space)
{
  ompl::base::ScopedState<> from(state_space), to(state_space), s(state_space);
  from[0] = node->pose.x;
  from[1] = node->pose.y;
  from[2] = _ctx->motion_table.getAngleFromBin(node->pose.theta);
  to[0] = goal->pose.x;
  to[1] = goal->pose.y;
  to[2] = _ctx->motion_table.getAngleFromBin(goal->pose.theta);

  float d = state_space->distance(from(), to());

  auto rs_state_space = dynamic_cast<ompl::base::ReedsSheppStateSpace *>(state_space.get());
  int direction_changes = 0;
  if (rs_state_space) {
    direction_changes = countDirectionChanges(rs_state_space->reedsShepp(from.get(), to.get()));
  }

  static const float sqrt_2 = sqrtf(2.0f);

  if (d > _search_info.analytic_expansion_max_length || d < sqrt_2) {
    return AnalyticExpansionNodes();
  }

  unsigned int num_intervals = static_cast<unsigned int>(std::floor(d / sqrt_2));

  AnalyticExpansionNodes possible_nodes;
  possible_nodes.nodes.reserve(num_intervals);
  std::vector<double> reals;
  double theta;

  NodePtr prev(node);
  uint64_t index = 0;
  NodePtr next(nullptr);
  float angle = 0.0;
  Coordinates proposed_coordinates;
  bool failure = false;
  std::vector<float> node_costs;
  node_costs.reserve(num_intervals);

  for (float i = 1; i <= num_intervals; i++) {
    state_space->interpolate(from(), to(), i / num_intervals, s());
    reals = s.reals();
    theta = (reals[2] < 0.0) ? (reals[2] + 2.0 * M_PI) : reals[2];
    theta = (theta > 2.0 * M_PI) ? (theta - 2.0 * M_PI) : theta;
    angle = _ctx->motion_table.getAngle(theta);

    index = NodeT::getIndex(
      static_cast<unsigned int>(reals[0]),
      static_cast<unsigned int>(reals[1]),
      static_cast<unsigned int>(angle),
      _ctx->motion_table.size_x,
      _ctx->motion_table.num_angle_quantization);
    if (node_getter(index, next)) {
      Coordinates initial_node_coords = next->pose;
      proposed_coordinates = {static_cast<float>(reals[0]), static_cast<float>(reals[1]), angle};
      next->setPose(proposed_coordinates);
      if (next->isNodeValid(_traverse_unknown, _collision_checker) && next != prev) {
        possible_nodes.add(next, initial_node_coords, proposed_coordinates);
        node_costs.emplace_back(next->getCost());
        prev = next;
      } else {
        next->setPose(initial_node_coords);
        failure = true;
        break;
      }
    } else {
      failure = true;
      break;
    }
  }

  if (!failure) {
    const float max_cost = _search_info.analytic_expansion_max_cost;
    auto max_cost_it = std::max_element(node_costs.begin(), node_costs.end());
    if (max_cost_it != node_costs.end() && *max_cost_it > max_cost) {
      bool cost_exit_high_cost_region = false;
      for (auto iter = node_costs.rbegin(); iter != node_costs.rend(); ++iter) {
        const float & curr_cost = *iter;
        if (curr_cost <= max_cost) {
          cost_exit_high_cost_region = true;
        } else if (curr_cost > max_cost && cost_exit_high_cost_region) {
          failure = true;
          break;
        }
      }

      if (failure) {
        if (d < 2.0f * M_PI * _ctx->motion_table.min_turning_radius &&
          _search_info.analytic_expansion_max_cost_override)
        {
          failure = false;
        }
      }
    }
  }

  for (const auto & node_pose : possible_nodes.nodes) {
    const auto & n = node_pose.node;
    n->setPose(node_pose.initial_coords);
  }

  if (failure) {
    return AnalyticExpansionNodes();
  }

  possible_nodes.setDirectionChanges(direction_changes);
  return possible_nodes;
}

template<typename NodeT>
float AnalyticExpansion<NodeT>::refineAnalyticPath(
  NodePtr & node,
  const NodePtr & goal_node,
  const NodeGetter & getter,
  AnalyticExpansionNodes & analytic_nodes)
{
  NodePtr test_node = node;
  AnalyticExpansionNodes refined_analytic_nodes;
  for (int i = 0; i < 8; i++) {
    if (test_node->parent && test_node->parent->parent &&
      test_node->parent->parent->parent &&
      test_node->parent->parent->parent->parent &&
      test_node->parent->parent->parent->parent->parent)
    {
      test_node = test_node->parent->parent->parent->parent->parent;
      refined_analytic_nodes =
        getAnalyticPath(
        test_node, goal_node, getter,
        _ctx->motion_table.state_space);
      if (refined_analytic_nodes.nodes.empty()) {
        break;
      }
      if (refined_analytic_nodes.direction_changes > analytic_nodes.direction_changes) {
        continue;
      }
      analytic_nodes = refined_analytic_nodes;
      node = test_node;
    } else {
      break;
    }
  }

  auto scoringFn = [&](const AnalyticExpansionNodes & expansion) {
      if (expansion.nodes.size() < 2) {
        return std::numeric_limits<float>::max();
      }

      float score = 0.0;
      float normalized_cost = 0.0;
      const float distance = hypotf(
        expansion.nodes[1].proposed_coords.x - expansion.nodes[0].proposed_coords.x,
        expansion.nodes[1].proposed_coords.y - expansion.nodes[0].proposed_coords.y);
      const float & weight = _ctx->motion_table.cost_penalty;
      for (auto iter = expansion.nodes.begin(); iter != expansion.nodes.end(); ++iter) {
        normalized_cost = iter->node->getCost() / 252.0f;
        score += distance * (1.0 + weight * normalized_cost);
      }
      return score;
    };

  float original_score = scoringFn(analytic_nodes);
  float best_score = original_score;
  float score = std::numeric_limits<float>::max();
  float min_turn_rad = _ctx->motion_table.min_turning_radius;
  const float max_min_turn_rad = 4.0 * min_turn_rad;

  // Motion model is always DUBIN or REEDS_SHEPP for hybrid A*

  while (min_turn_rad < max_min_turn_rad) {
    min_turn_rad += 0.5;
    ompl::base::StateSpacePtr state_space;
    if (_ctx->motion_table.motion_model == MotionModel::DUBIN) {
      state_space = std::make_shared<ompl::base::DubinsStateSpace>(min_turn_rad);
    } else {
      state_space = std::make_shared<ompl::base::ReedsSheppStateSpace>(min_turn_rad);
    }
    refined_analytic_nodes = getAnalyticPath(node, goal_node, getter, state_space);
    score = scoringFn(refined_analytic_nodes);

    if (score <= best_score &&
      refined_analytic_nodes.direction_changes <= analytic_nodes.direction_changes)
    {
      analytic_nodes = refined_analytic_nodes;
      best_score = score;
      continue;
    }

    if (score <= original_score &&
      refined_analytic_nodes.direction_changes < analytic_nodes.direction_changes)
    {
      analytic_nodes = refined_analytic_nodes;
      best_score = score;
    }
  }

  return best_score;
}

template<typename NodeT>
typename AnalyticExpansion<NodeT>::NodePtr AnalyticExpansion<NodeT>::setAnalyticPath(
  const NodePtr & node,
  const NodePtr & goal_node,
  const AnalyticExpansionNodes & expanded_nodes)
{
  _detached_nodes.clear();
  NodePtr prev = node;
  for (const auto & node_pose : expanded_nodes.nodes) {
    auto n = node_pose.node;
    cleanNode(n);
    if (n->getIndex() != goal_node->getIndex()) {
      if (n->wasVisited()) {
        _detached_nodes.push_back(std::make_unique<NodeT>(-1, _ctx));
        n = _detached_nodes.back().get();
      }
      n->parent = prev;
      n->pose = node_pose.proposed_coords;
      n->visited();
      prev = n;
    }
  }
  if (goal_node != prev) {
    goal_node->parent = prev;
    cleanNode(goal_node);
    goal_node->visited();
  }
  return goal_node;
}

template<typename NodeT>
void AnalyticExpansion<NodeT>::cleanNode(const NodePtr & /*expanded_nodes*/)
{
}

template class AnalyticExpansion<NodeHybrid>;

}  // namespace hybrid_astar
