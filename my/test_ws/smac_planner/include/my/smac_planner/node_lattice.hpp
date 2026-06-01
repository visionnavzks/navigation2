#ifndef SMAC_PLANNER__NODE_LATTICE_HPP_
#define SMAC_PLANNER__NODE_LATTICE_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ompl/base/StateSpace.h"

#include "my/smac_planner/constants.hpp"
#include "my/smac_planner/types.hpp"
#include "my/smac_planner/obstacle_heuristic.hpp"
#include "my/smac_planner/distance_heuristic.hpp"
#include "my/smac_planner/node_hybrid.hpp"
#include "my/smac_planner/utils.hpp"

namespace smac_planner
{

class NodeLattice;
class GridCollisionChecker;

struct LatticeMotionTable
{
  LatticeMotionTable() {}

  void initMotionModel(
    unsigned int & size_x_in,
    SearchInfo & search_info);

  MotionPrimitivePtrs getMotionPrimitives(
    const NodeLattice * node,
    unsigned int & direction_change_index);

  static LatticeMetadata getLatticeMetadata(const std::string & lattice_filepath);

  unsigned int getClosestAngularBin(const double & theta);

  float & getAngleFromBin(const unsigned int & bin_idx);

  double getAngle(const double & theta);

  unsigned int size_x;
  unsigned int num_angle_quantization;
  float change_penalty;
  float non_straight_penalty;
  float cost_penalty;
  float reverse_penalty;
  float travel_distance_reward;
  float rotation_penalty;
  float min_turning_radius;
  bool allow_reverse_expansion;
  bool downsample_obstacle_heuristic;
  bool use_quadratic_cost_penalty;
  std::vector<std::vector<MotionPrimitive>> motion_primitives;
  ompl::base::StateSpacePtr state_space;
  std::vector<TrigValues> trig_values;
  std::string current_lattice_filepath;
  LatticeMetadata lattice_metadata;
  MotionModel motion_model = MotionModel::UNKNOWN;
};

class NodeLattice
{
public:
  typedef NodeLattice * NodePtr;
  typedef std::unique_ptr<std::vector<NodeLattice>> Graph;
  typedef std::vector<NodePtr> NodeVector;
  typedef NodeHybrid::Coordinates Coordinates;
  typedef NodeHybrid::CoordinateVector CoordinateVector;

  struct NodeContext
  {
    NodeContext()
    {
      obstacle_heuristic = std::make_unique<ObstacleHeuristic>();
      distance_heuristic = std::make_unique<DistanceHeuristic<NodeLattice>>();
    }

    LatticeMotionTable motion_table;
    std::unique_ptr<ObstacleHeuristic> obstacle_heuristic;
    std::unique_ptr<DistanceHeuristic<NodeLattice>> distance_heuristic;
  };

  explicit NodeLattice(const uint64_t index, NodeContext * ctx);
  ~NodeLattice();

  bool operator==(const NodeLattice & rhs) const
  {
    return this->_index == rhs._index;
  }

  inline void setPose(const Coordinates & pose_in)
  {
    pose = pose_in;
  }

  void reset();

  inline void setMotionPrimitive(MotionPrimitive * prim)
  {
    _motion_primitive = prim;
  }

  inline MotionPrimitive * & getMotionPrimitive()
  {
    return _motion_primitive;
  }

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

  inline bool wasVisited()
  {
    return _was_visited;
  }

  inline void visited()
  {
    _was_visited = true;
  }

  inline uint64_t getIndex()
  {
    return _index;
  }

  inline void backwards(bool back = true)
  {
    _backwards = back;
  }

  inline bool isBackward()
  {
    return _backwards;
  }

  bool isNodeValid(
    const bool & traverse_unknown,
    GridCollisionChecker * collision_checker,
    MotionPrimitive * primitive = nullptr,
    bool is_backwards = false);

  float getTraversalCost(const NodePtr & child);

  static inline uint64_t getIndex(
    const unsigned int & x, const unsigned int & y, const unsigned int & angle,
    const unsigned int & width, const unsigned int & angle_quantization)
  {
    return NodeHybrid::getIndex(x, y, angle, width, angle_quantization);
  }

  static inline Coordinates getCoords(
    const uint64_t & index,
    const unsigned int & width, const unsigned int & angle_quantization)
  {
    return NodeHybrid::Coordinates(
      (index / angle_quantization) % width,
      index / (angle_quantization * width),
      index % angle_quantization);
  }

  float getHeuristicCost(
    const Coordinates & node_coords,
    const CoordinateVector & goals_coords);

  static void initMotionModel(
    NodeContext * ctx,
    const MotionModel & motion_model,
    unsigned int & size_x,
    unsigned int & size_y,
    unsigned int & angle_quantization,
    SearchInfo & search_info);

  void getNeighbors(
    std::function<bool(const uint64_t &,
    smac_planner::NodeLattice * &)> & validity_checker,
    GridCollisionChecker * collision_checker,
    const bool & traverse_unknown,
    NodeVector & neighbors);

  bool backtracePath(CoordinateVector & path);

  void addNodeToPath(NodePtr current_node, CoordinateVector & path);

  NodeLattice * parent;
  Coordinates pose;

private:
  float _cell_cost;
  float _accumulated_cost;
  uint64_t _index;
  bool _was_visited;
  MotionPrimitive * _motion_primitive;
  bool _backwards;
  bool _is_node_valid{false};
  NodeContext * _ctx = nullptr;
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__NODE_LATTICE_HPP_
