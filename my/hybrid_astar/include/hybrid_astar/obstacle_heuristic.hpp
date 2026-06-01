#ifndef HYBRID_ASTAR__OBSTACLE_HEURISTIC_HPP_
#define HYBRID_ASTAR__OBSTACLE_HEURISTIC_HPP_

#include <utility>
#include <vector>
#include <memory>
#include "hybrid_astar/constants.hpp"
#include "hybrid_astar/types.hpp"
#include "hybrid_astar/costmap_2d.hpp"

namespace hybrid_astar
{

class Costmap2D;
class GridCollisionChecker;
class EsdfHolder;

typedef std::vector<NodeHeuristicPair> ObstacleHeuristicQueue;

class ObstacleHeuristic
{
public:
  ObstacleHeuristic() = default;
  ~ObstacleHeuristic() = default;

  void resetObstacleHeuristic(
    Costmap2D * costmap,
    const float & start_x, const float & start_y,
    const float & goal_x, const float & goal_y,
    const bool downsample_obstacle_heuristic);

  /// Provide the cached ESDF holder so that, when the ESDF path is enabled,
  /// the heuristic cost is driven by the ESDF soft penalty rather than the
  /// raw costmap cell cost. May be nullptr (legacy behavior).
  void setEsdfHolder(EsdfHolder * holder) { esdf_holder_ = holder; }
  void setEsdfFootprintParams(
    const std::vector<double> & cost_check_points,
    double robot_radius,
    double safe_distance)
  {
    cost_check_points_ = cost_check_points;
    robot_radius_ = robot_radius;
    safe_distance_ = safe_distance;
  }

  float getObstacleHeuristic(
    const Coordinates & node_coords,
    const float & cost_penalty,
    const bool use_quadratic_cost_penalty,
    const bool downsample_obstacle_heuristic);

  inline float distanceHeuristic2D(
    const uint64_t idx, const unsigned int size_x,
    const unsigned int target_x, const unsigned int target_y) const
  {
    int dx = static_cast<int>(idx % size_x) - static_cast<int>(target_x);
    int dy = static_cast<int>(idx / size_x) - static_cast<int>(target_y);
    return std::sqrt(dx * dx + dy * dy);
  }

protected:
  /// Compute the cost of a single costmap cell for the Dijkstra sweep.
  /// In the legacy path this is just the cell's raw cost. In the ESDF path
  /// it is the soft penalty derived from the per-cell ESDF clearance, with
  /// the downsample_obstacle_heuristic 2x2 reduction applied as before.
  float cellCostForHeuristic(unsigned int mx, unsigned int my);

  LookupTable obstacle_heuristic_lookup_table_;
  ObstacleHeuristicQueue obstacle_heuristic_queue_;
  Costmap2D * costmap{nullptr};
  unsigned int cached_size_x_{0};
  unsigned int cached_size_y_{0};

  // Optional ESDF path state.
  EsdfHolder * esdf_holder_{nullptr};
  std::vector<double> cost_check_points_{};
  double robot_radius_{0.0};
  double safe_distance_{0.0};
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__OBSTACLE_HEURISTIC_HPP_
