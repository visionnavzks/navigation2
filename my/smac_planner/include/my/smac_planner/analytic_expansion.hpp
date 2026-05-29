#ifndef SMAC_PLANNER__ANALYTIC_EXPANSION_HPP_
#define SMAC_PLANNER__ANALYTIC_EXPANSION_HPP_

#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/DubinsStateSpace.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>

#include <functional>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "my/smac_planner/node_2d.hpp"
#include "my/smac_planner/node_hybrid.hpp"
#include "my/smac_planner/node_lattice.hpp"
#include "my/smac_planner/types.hpp"
#include "my/smac_planner/constants.hpp"

namespace smac_planner
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

  void cleanNode(const NodePtr & nodes);

protected:
  MotionModel _motion_model;
  SearchInfo _search_info;
  bool _traverse_unknown;
  unsigned int _dim_3_size;
  GridCollisionChecker * _collision_checker;
  std::list<std::unique_ptr<NodeT>> _detached_nodes;
  NodeContext * _ctx = nullptr;
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__ANALYTIC_EXPANSION_HPP_
