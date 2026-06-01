#ifndef SMAC_PLANNER__NODE_BASIC_HPP_
#define SMAC_PLANNER__NODE_BASIC_HPP_

#include "my/smac_planner/constants.hpp"
#include "my/smac_planner/node_hybrid.hpp"
#include "my/smac_planner/node_lattice.hpp"
#include "my/smac_planner/node_2d.hpp"
#include "my/smac_planner/types.hpp"
#include "my/smac_planner/collision_checker.hpp"

namespace smac_planner
{

template<typename NodeT>
class NodeBasic
{
public:
  explicit NodeBasic(const uint64_t new_index)
  : graph_node_ptr(nullptr),
    index(new_index)
  {
  }

  void populateSearchNode(NodeT * & node);
  void processSearchNode();

  typename NodeT::Coordinates pose;
  NodeT * graph_node_ptr;
  MotionPrimitive * prim_ptr;
  uint64_t index;
  unsigned int motion_index;
  bool backward;
  TurnDirection turn_dir;
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__NODE_BASIC_HPP_
