#ifndef HYBRID_ASTAR__ANALYTIC_EXPANSION_HPP_
#define HYBRID_ASTAR__ANALYTIC_EXPANSION_HPP_

#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/DubinsStateSpace.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>

#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "my/hybrid_astar/node_hybrid.hpp"
#include "my/hybrid_astar/types.hpp"
#include "my/hybrid_astar/constants.hpp"

namespace hybrid_astar
{

template<typename NodeT>
class AnalyticExpansion
{
public:
  typedef NodeT * NodePtr;
  typedef std::vector<NodePtr> NodeVector;
  typedef typename NodeT::Coordinates Coordinates;
  typedef std::function<bool (const uint64_t &, NodeT * &)> NodeGetter;
  typedef typename NodeT::CoordinateVector CoordinateVector;
  using NodeContext = typename NodeT::NodeContext;

  struct AnalyticExpansionNode
  {
    AnalyticExpansionNode(
      NodePtr & node_in,
      Coordinates & initial_coords_in,
      Coordinates & proposed_coords_in)
    : node(node_in),
      initial_coords(initial_coords_in),
      proposed_coords(proposed_coords_in)
    {
    }

    NodePtr node;
    Coordinates initial_coords;
    Coordinates proposed_coords;
  };

  struct AnalyticExpansionNodes
  {
    AnalyticExpansionNodes() = default;

    void add(
      NodePtr & node,
      Coordinates & initial_coords,
      Coordinates & proposed_coords)
    {
      nodes.emplace_back(node, initial_coords, proposed_coords);
    }

    void setDirectionChanges(int changes)
    {
      direction_changes = changes;
    }

    std::vector<AnalyticExpansionNode> nodes;
    int direction_changes{0};
  };

  AnalyticExpansion(
    const MotionModel & motion_model,
    const SearchInfo & search_info,
    const bool & traverse_unknown,
    const unsigned int & dim_3_size);

  void setCollisionChecker(GridCollisionChecker * collision_checker);

  void setContext(NodeContext * ctx);

  NodePtr tryAnalyticExpansion(
    const NodePtr & current_node,
    const NodeVector & coarse_check_goals,
    const NodeVector & fine_check_goals,
    const CoordinateVector & goals_coords,
    const NodeGetter & getter, int & iterations,
    int & closest_distance);

  AnalyticExpansionNodes getAnalyticPath(
    const NodePtr & node, const NodePtr & goal,
    const NodeGetter & getter, const ompl::base::StateSpacePtr & state_space);

  float refineAnalyticPath(
    NodePtr & node,
    const NodePtr & goal_node,
    const NodeGetter & getter,
    AnalyticExpansionNodes & analytic_nodes);

  NodePtr setAnalyticPath(
    const NodePtr & node, const NodePtr & goal,
    const AnalyticExpansionNodes & expanded_nodes);

  int countDirectionChanges(const ompl::base::ReedsSheppStateSpace::ReedsSheppPath & path);

protected:
  MotionModel _motion_model;
  SearchInfo _search_info;
  bool _traverse_unknown;
  unsigned int _dim_3_size;
  GridCollisionChecker * _collision_checker;
  std::list<std::unique_ptr<NodeT>> _detached_nodes;
  NodeContext * _ctx = nullptr;
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__ANALYTIC_EXPANSION_HPP_
