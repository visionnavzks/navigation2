#ifndef SMAC_PLANNER__NODE_2D_HPP_
#define SMAC_PLANNER__NODE_2D_HPP_

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "my/smac_planner/types.hpp"
#include "my/smac_planner/constants.hpp"

namespace smac_planner
{

class GridCollisionChecker;
class NodeHybrid;

class Node2D
{
public:
  typedef Node2D * NodePtr;
  typedef std::unique_ptr<std::vector<Node2D>> Graph;
  typedef std::vector<NodePtr> NodeVector;
  using Coordinates = smac_planner::Coordinates2D;
  typedef std::vector<Coordinates> CoordinateVector;

  struct NodeContext
  {
    float cost_travel_multiplier;
    std::vector<int> neighbors_grid_offsets;
  };

  explicit Node2D(const uint64_t index, NodeContext * ctx);
  ~Node2D();

  bool operator==(const Node2D & rhs) const
  {
    return this->_index == rhs._index;
  }

  inline void setPose(const Coordinates & pose_in)
  {
    pose = pose_in;
  }

  void reset();

  inline float getAccumulatedCost()
  {
    return _accumulated_cost;
  }

  inline void setAccumulatedCost(const float & cost_in)
  {
    _accumulated_cost = cost_in;
  }

  inline float getCost()
  {
    return _cell_cost;
  }

  inline void setCost(const float & cost)
  {
    _cell_cost = cost;
  }

  inline bool wasVisited()
  {
    return _was_visited;
  }

  inline void visited()
  {
    _was_visited = true;
    _is_queued = false;
  }

  inline bool & isQueued()
  {
    return _is_queued;
  }

  inline void queued()
  {
    _is_queued = true;
  }

  inline uint64_t getIndex()
  {
    return _index;
  }

  bool isNodeValid(const bool & traverse_unknown, GridCollisionChecker * collision_checker);

  float getTraversalCost(const NodePtr & child);

  static inline uint64_t getIndex(
    const unsigned int & x, const unsigned int & y, const unsigned int & width)
  {
    return static_cast<uint64_t>(x) + static_cast<uint64_t>(y) *
           static_cast<uint64_t>(width);
  }

  static inline Coordinates getCoords(
    const uint64_t & index, const unsigned int & width, const unsigned int & angles)
  {
    if (angles != 1) {
      throw std::runtime_error("Node type Node2D does not have a valid angle quantization.");
    }

    return Coordinates(index % width, index / width);
  }

  inline Coordinates getCoords(const uint64_t & index)
  {
    const unsigned int & size_x = _ctx->neighbors_grid_offsets[3];
    return Coordinates(index % size_x, index / size_x);
  }

  float getHeuristicCost(
    const Coordinates & node_coords,
    const CoordinateVector & goals_coords);

  static void initMotionModel(
    NodeContext * ctx,
    const MotionModel & motion_model,
    unsigned int & size_x,
    unsigned int & size_y,
    unsigned int & num_angle_quantization,
    SearchInfo & search_info);

  void getNeighbors(
    std::function<bool(const uint64_t &,
    smac_planner::Node2D * &)> & validity_checker,
    GridCollisionChecker * collision_checker,
    const bool & traverse_unknown,
    NodeVector & neighbors);

  bool backtracePath(CoordinateVector & path);

  Node2D * parent;
  Coordinates pose;

private:
  float _cell_cost;
  float _accumulated_cost;
  uint64_t _index;
  bool _was_visited;
  bool _is_queued;
  bool _in_collision{false};
  NodeContext * _ctx = nullptr;
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__NODE_2D_HPP_
