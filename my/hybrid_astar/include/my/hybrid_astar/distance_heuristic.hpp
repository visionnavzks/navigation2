#ifndef HYBRID_ASTAR__DISTANCE_HEURISTIC_HPP_
#define HYBRID_ASTAR__DISTANCE_HEURISTIC_HPP_

#include "my/hybrid_astar/constants.hpp"
#include "my/hybrid_astar/types.hpp"

namespace hybrid_astar
{
struct HybridMotionTable;
class NodeHybrid;

template<typename NodeT>
class DistanceHeuristic
{
public:
  DistanceHeuristic() = default;

  template<typename MotionTableT>
  void precomputeDistanceHeuristic(
    const float & lookup_table_dim,
    const MotionModel & motion_model,
    const unsigned int & dim_3_size,
    const SearchInfo & search_info,
    MotionTableT & motion_table);

  template<typename MotionTableT>
  float getDistanceHeuristic(
    const Coordinates & node_coords,
    const Coordinates & goal_coords,
    const float & obstacle_heuristic,
    MotionTableT & motion_table);

protected:
  LookupTable dist_heuristic_lookup_table_;
  float size_lookup_{0.0f};
};

}  // namespace hybrid_astar
#endif  // HYBRID_ASTAR__DISTANCE_HEURISTIC_HPP_
