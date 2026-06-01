#ifndef HYBRID_ASTAR__OBSTACLE_HEURISTIC_HPP_
#define HYBRID_ASTAR__OBSTACLE_HEURISTIC_HPP_

#include <utility>
#include <vector>
#include <memory>
#include "my/hybrid_astar/constants.hpp"
#include "my/hybrid_astar/types.hpp"

namespace hybrid_astar
{

class Costmap2D;

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
  LookupTable obstacle_heuristic_lookup_table_;
  ObstacleHeuristicQueue obstacle_heuristic_queue_;
  Costmap2D * costmap{nullptr};
  unsigned int cached_size_x_{0};
  unsigned int cached_size_y_{0};
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__OBSTACLE_HEURISTIC_HPP_
