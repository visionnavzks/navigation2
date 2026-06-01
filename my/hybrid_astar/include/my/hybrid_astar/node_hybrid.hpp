#ifndef HYBRID_ASTAR__NODE_HYBRID_HPP_
#define HYBRID_ASTAR__NODE_HYBRID_HPP_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "ompl/base/StateSpace.h"

#include "my/hybrid_astar/constants.hpp"
#include "my/hybrid_astar/types.hpp"
#include "my/hybrid_astar/collision_checker.hpp"
#include "my/hybrid_astar/obstacle_heuristic.hpp"
#include "my/hybrid_astar/distance_heuristic.hpp"

namespace hybrid_astar
{

class NodeHybrid;

struct HybridMotionTable
{
  HybridMotionTable()
  : motion_model(MotionModel::UNKNOWN),
    size_x(0),
    num_angle_quantization(0),
    num_angle_quantization_float(0.0f),
    min_turning_radius(0.0f),
    bin_size(0.0f),
    change_penalty(0.0f),
    non_straight_penalty(0.0f),
    cost_penalty(0.0f),
    reverse_penalty(0.0f),
    travel_distance_reward(0.0f),
    downsample_obstacle_heuristic(false),
    use_quadratic_cost_penalty(false),
    allow_primitive_interpolation(false)
  {}

  void initDubin(
    const unsigned int & size_x_in,
    const unsigned int & size_y_in,
    const unsigned int & angle_quantization_in,
    SearchInfo & search_info);

  void initReedsShepp(
    const unsigned int & size_x_in,
    const unsigned int & size_y_in,
    const unsigned int & angle_quantization_in,
    SearchInfo & search_info);

  MotionPoses getProjections(const NodeHybrid * node);

  unsigned int getClosestAngularBin(const double & theta) const;

  float getAngleFromBin(const unsigned int & bin_idx) const;

  double getAngle(const double & theta) const;

  void initCommon(
    const unsigned int & size_x_in,
    const unsigned int & size_y_in,
    const unsigned int & num_angle_quantization_in,
    SearchInfo & search_info,
    MotionModel model);

  MotionModel motion_model = MotionModel::UNKNOWN;
  MotionPoses projections;
  unsigned int size_x;
  unsigned int size_y;
  unsigned int num_angle_quantization;
  float num_angle_quantization_float;
  float min_turning_radius;
  float bin_size;
  float change_penalty;
  float non_straight_penalty;
  float cost_penalty;
  float reverse_penalty;
  float travel_distance_reward;
  bool downsample_obstacle_heuristic;
  bool use_quadratic_cost_penalty;
  bool allow_primitive_interpolation;
  ompl::base::StateSpacePtr state_space;
  std::vector<std::vector<double>> delta_xs;
  std::vector<std::vector<double>> delta_ys;
  std::vector<TrigValues> trig_values;
  std::vector<float> travel_costs;
};

class NodeHybrid
{
public:
  typedef NodeHybrid * NodePtr;
  typedef std::unique_ptr<std::vector<NodeHybrid>> Graph;
  typedef std::vector<NodePtr> NodeVector;
  using Coordinates = hybrid_astar::Coordinates;
  typedef std::vector<Coordinates> CoordinateVector;

  struct NodeContext
  {
    NodeContext()
    {
      obstacle_heuristic = std::make_unique<ObstacleHeuristic>();
      distance_heuristic = std::make_unique<DistanceHeuristic<NodeHybrid>>();
    }
    HybridMotionTable motion_table;
    std::unique_ptr<ObstacleHeuristic> obstacle_heuristic;
    std::unique_ptr<DistanceHeuristic<NodeHybrid>> distance_heuristic;
  };

  explicit NodeHybrid(const uint64_t index, NodeContext * ctx);
  ~NodeHybrid();

  bool operator==(const NodeHybrid & rhs) const
  {
    return this->_index == rhs._index;
  }

  inline void setPose(const Coordinates & pose_in)
  {
    pose = pose_in;
  }

  void reset();

  inline float getAccumulatedCost() const
  {
    return _accumulated_cost;
  }

  inline void setAccumulatedCost(const float & cost_in)
  {
    _accumulated_cost = cost_in;
  }

  inline void setMotionPrimitiveIndex(const unsigned int & idx, const TurnDirection & turn_dir)
  {
    _motion_primitive_index = idx;
    _turn_dir = turn_dir;
  }

  inline unsigned int getMotionPrimitiveIndex() const
  {
    return _motion_primitive_index;
  }

  inline TurnDirection getTurnDirection() const
  {
    return _turn_dir;
  }

  inline float getCost() const
  {
    return _cell_cost;
  }

  inline bool wasVisited() const
  {
    return _was_visited;
  }

  inline void visited()
  {
    _was_visited = true;
  }

  inline uint64_t getIndex() const
  {
    return _index;
  }

  bool isNodeValid(
    const bool & traverse_unknown,
    GridCollisionChecker * collision_checker);

  float getTraversalCost(const NodePtr & child);

  static inline uint64_t getIndex(
    const unsigned int & x, const unsigned int & y, const unsigned int & angle,
    const unsigned int & width, const unsigned int & angle_quantization)
  {
    return static_cast<uint64_t>(angle) + static_cast<uint64_t>(x) *
           static_cast<uint64_t>(angle_quantization) +
           static_cast<uint64_t>(y) * static_cast<uint64_t>(width) *
           static_cast<uint64_t>(angle_quantization);
  }

  static inline Coordinates getCoords(
    const uint64_t & index,
    const unsigned int & width, const unsigned int & angle_quantization)
  {
    return Coordinates(
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
    const unsigned int & size_x,
    const unsigned int & size_y,
    const unsigned int & angle_quantization,
    SearchInfo & search_info);

  void getNeighbors(
    std::function<bool(const uint64_t &,
    hybrid_astar::NodeHybrid * &)> & validity_checker,
    GridCollisionChecker * collision_checker,
    const bool & traverse_unknown,
    NodeVector & neighbors);

  bool backtracePath(CoordinateVector & path);

  NodeHybrid * parent;
  Coordinates pose;

private:
  float _cell_cost;
  float _accumulated_cost;
  uint64_t _index;
  bool _was_visited;
  unsigned int _motion_primitive_index;
  TurnDirection _turn_dir;
  bool _is_node_valid{false};
  NodeContext * _ctx = nullptr;
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__NODE_HYBRID_HPP_
