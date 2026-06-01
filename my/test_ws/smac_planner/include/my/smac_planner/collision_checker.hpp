#ifndef SMAC_PLANNER__COLLISION_CHECKER_HPP_
#define SMAC_PLANNER__COLLISION_CHECKER_HPP_

#include <memory>
#include <vector>
#include <cmath>

#include "my/smac_planner/constants.hpp"
#include "my/smac_planner/costmap_2d.hpp"
#include "my/smac_planner/types.hpp"

namespace smac_planner
{

class GridCollisionChecker
{
public:
  GridCollisionChecker(
    Costmap2D * costmap,
    unsigned int num_quantizations);

  void setFootprint(
    const Footprint & footprint,
    const bool & radius,
    const double & possible_collision_cost);

  bool inCollision(
    const float & x, const float & y,
    const float & theta, const bool & traverse_unknown);

  bool inCollision(
    const unsigned int & i, const bool & traverse_unknown);

  float getCost();

  std::vector<float> & getPrecomputedAngles() { return angles_; }

  Costmap2D * getCostmap() { return costmap_; }
  const Costmap2D * getCostmap() const { return costmap_; }

  void setCostmap(Costmap2D * costmap) { costmap_ = costmap; }

  bool outsideRange(const unsigned int & max, const float & value);

protected:
  Costmap2D * costmap_;
  std::vector<Footprint> oriented_footprints_;
  Footprint unoriented_footprint_;
  float center_cost_;
  bool footprint_is_radius_{false};
  std::vector<float> angles_;
  float possible_collision_cost_{-1};
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__COLLISION_CHECKER_HPP_
