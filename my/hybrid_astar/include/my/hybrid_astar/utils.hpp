#ifndef HYBRID_ASTAR__UTILS_HPP_
#define HYBRID_ASTAR__UTILS_HPP_

#include <vector>
#include <memory>
#include <string>
#include <cmath>

#include "my/hybrid_astar/types.hpp"
#include "my/hybrid_astar/constants.hpp"

namespace hybrid_astar
{

class Costmap2D;

inline Pose getWorldCoords(
  const float & mx, const float & my, const Costmap2D * costmap)
{
  Pose p;
  p.x = costmap->getOriginX() + mx * costmap->getResolution();
  p.y = costmap->getOriginY() + my * costmap->getResolution();
  p.theta = 0.0;
  return p;
}

inline double findCircumscribedCost(
  Costmap2D * costmap,
  double circumscribed_radius,
  double inflation_radius)
{
  if (inflation_radius < circumscribed_radius) {
    return 0.0;
  }
  double resolution = costmap->getResolution();
  double distance_cells = circumscribed_radius / resolution;
  double inflation_cells = inflation_radius / resolution;
  double cost = INSCRIBED_COST * (1.0 - distance_cells / inflation_cells);
  return std::max(0.0, cost);
}

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__UTILS_HPP_
