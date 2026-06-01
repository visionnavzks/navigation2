#include "hybrid_astar/node_basic.hpp"

namespace hybrid_astar
{

template<>
void NodeBasic<NodeHybrid>::processSearchNode()
{
  if (!this->graph_node_ptr->wasVisited()) {
    this->graph_node_ptr->pose = this->pose;
    this->graph_node_ptr->setMotionPrimitiveIndex(this->motion_index, this->turn_dir);
  }
}

template<>
void NodeBasic<NodeHybrid>::populateSearchNode(NodeHybrid * & node)
{
  this->pose = node->pose;
  this->graph_node_ptr = node;
  this->motion_index = node->getMotionPrimitiveIndex();
  this->turn_dir = node->getTurnDirection();
}

template class NodeBasic<NodeHybrid>;

}  // namespace hybrid_astar
