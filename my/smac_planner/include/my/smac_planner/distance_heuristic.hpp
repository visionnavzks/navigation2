#ifndef SMAC_PLANNER__DISTANCE_HEURISTIC_HPP_
#define SMAC_PLANNER__DISTANCE_HEURISTIC_HPP_

#include "my/smac_planner/constants.hpp"
#include "my/smac_planner/types.hpp"

namespace smac_planner
{
struct HybridMotionTable;
struct LatticeMotionTable;
class NodeHybrid;
class NodeLattice;

template<typename NodeT>
class DistanceHeuristic
{
public:
  DistanceHeuristic() {}

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
  float size_lookup_;
};

}  // namespace smac_planner
#endif  // SMAC_PLANNER__DISTANCE_HEURISTIC_HPP_
