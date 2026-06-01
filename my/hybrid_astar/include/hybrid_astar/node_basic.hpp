#ifndef HYBRID_ASTAR__NODE_BASIC_HPP_
#define HYBRID_ASTAR__NODE_BASIC_HPP_

#include "hybrid_astar/constants.hpp"
#include "hybrid_astar/node_hybrid.hpp"
#include "hybrid_astar/types.hpp"
#include "hybrid_astar/collision_checker.hpp"

namespace hybrid_astar
{

template<typename NodeT>
class NodeBasic
{
public:
  explicit NodeBasic(const uint64_t new_index)
  : graph_node_ptr(nullptr),
    index(new_index),
    motion_index(0u),
    turn_dir(TurnDirection::UNKNOWN)
  {
  }

  void populateSearchNode(NodeT * & node);
  void processSearchNode();

  typename NodeT::Coordinates pose;
  NodeT * graph_node_ptr;
  uint64_t index;
  unsigned int motion_index;
  TurnDirection turn_dir;
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__NODE_BASIC_HPP_
