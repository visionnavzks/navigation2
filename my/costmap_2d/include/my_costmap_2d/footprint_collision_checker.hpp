// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MY_COSTMAP_2D__FOOTPRINT_COLLISION_CHECKER_HPP_
#define MY_COSTMAP_2D__FOOTPRINT_COLLISION_CHECKER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "my_costmap_2d/costmap_2d.hpp"
#include "my_costmap_2d/exceptions.hpp"
#include "my_costmap_2d/costmap_math.hpp"
#include "my_costmap_2d/footprint.hpp"
#include "my_costmap_2d/cost_values.hpp"

namespace my_costmap_2d
{

typedef std::vector<Point> Footprint;

template<typename CostmapT>
class FootprintCollisionChecker
{
public:
  FootprintCollisionChecker();
  explicit FootprintCollisionChecker(CostmapT costmap);
  double footprintCost(const Footprint & footprint);
  double footprintCostAtPose(double x, double y, double theta, const Footprint & footprint);
  double lineCost(int x0, int x1, int y0, int y1) const;
  bool worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my);
  double pointCost(int x, int y) const;
  void setCostmap(CostmapT costmap);
  CostmapT getCostmap()
  {
    return costmap_;
  }

protected:
  CostmapT costmap_;
};

template<typename CostmapT>
FootprintCollisionChecker<CostmapT>::FootprintCollisionChecker()
: costmap_(nullptr)
{
}

template<typename CostmapT>
FootprintCollisionChecker<CostmapT>::FootprintCollisionChecker(CostmapT costmap)
: costmap_(costmap)
{
}

template<typename CostmapT>
double FootprintCollisionChecker<CostmapT>::footprintCost(const Footprint & footprint)
{
  double cost = 0.0;

  for (unsigned int i = 0; i < footprint.size() - 1; ++i) {
    unsigned int x0, y0, x1, y1;
    if (!worldToMap(footprint[i].x, footprint[i].y, x0, y0) ||
      !worldToMap(footprint[i + 1].x, footprint[i + 1].y, x1, y1))
    {
      return -1.0;
    }

    double line_cost = lineCost(x0, x1, y0, y1);
    if (line_cost < 0) {
      return -1.0;
    }

    cost = std::max(cost, line_cost);
  }

  // check the last point and first point
  unsigned int x0, y0, x1, y1;
  if (!worldToMap(footprint.back().x, footprint.back().y, x0, y0) ||
    !worldToMap(footprint.front().x, footprint.front().y, x1, y1))
  {
    return -1.0;
  }

  double line_cost = lineCost(x0, x1, y0, y1);
  if (line_cost < 0) {
    return -1.0;
  }

  cost = std::max(cost, line_cost);

  return cost;
}

template<typename CostmapT>
double FootprintCollisionChecker<CostmapT>::footprintCostAtPose(
  double x, double y, double theta, const Footprint & footprint)
{
  std::vector<Point> oriented_footprint;
  transformFootprint(x, y, theta, footprint, oriented_footprint);
  return footprintCost(oriented_footprint);
}

template<typename CostmapT>
double FootprintCollisionChecker<CostmapT>::lineCost(int x0, int x1, int y0, int y1) const
{
  double cost = 0.0;

  if (x0 == x1 && y0 == y1) {
    return pointCost(x0, y0);
  }

  int dx = x1 - x0;
  int dy = y1 - y0;
  int steps = std::max(abs(dx), abs(dy));
  double x_step = dx / static_cast<double>(steps);
  double y_step = dy / static_cast<double>(steps);

  for (int i = 0; i <= steps; ++i) {
    int x = x0 + static_cast<int>(x_step * i);
    int y = y0 + static_cast<int>(y_step * i);
    double point_cost = pointCost(x, y);
    if (point_cost < 0) {
      return -1.0;
    }
    cost = std::max(cost, point_cost);
  }

  return cost;
}

template<typename CostmapT>
bool FootprintCollisionChecker<CostmapT>::worldToMap(
  double wx, double wy, unsigned int & mx, unsigned int & my)
{
  if (costmap_ == nullptr) {
    return false;
  }
  return costmap_->worldToMap(wx, wy, mx, my);
}

template<typename CostmapT>
double FootprintCollisionChecker<CostmapT>::pointCost(int x, int y) const
{
  if (costmap_ == nullptr) {
    return -1.0;
  }
  return costmap_->getCost(x, y);
}

template<typename CostmapT>
void FootprintCollisionChecker<CostmapT>::setCostmap(CostmapT costmap)
{
  costmap_ = costmap;
}

}  // namespace my_costmap_2d

#endif  // MY_COSTMAP_2D__FOOTPRINT_COLLISION_CHECKER_HPP_
