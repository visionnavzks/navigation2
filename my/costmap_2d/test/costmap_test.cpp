// Simple test for costmap_2d library
#include <iostream>
#include <cassert>
#include <cmath>

#include "costmap_2d/costmap_2d.hpp"
#include "costmap_2d/cost_values.hpp"
#include "costmap_2d/footprint.hpp"
#include "costmap_2d/costmap_layer_algorithms.hpp"
#include "costmap_2d/inflation_layer_core.hpp"
#include "costmap_2d/distance_transform.hpp"

void testCostmap2D()
{
  std::cout << "Testing Costmap2D..." << std::endl;

  // Create a 10x10 costmap with 0.05m resolution
  costmap_2d::Costmap2D costmap(10, 10, 0.05, 0.0, 0.0);

  // Test basic properties
  assert(costmap.getSizeInCellsX() == 10);
  assert(costmap.getSizeInCellsY() == 10);
  assert(std::abs(costmap.getResolution() - 0.05) < 1e-6);
  assert(std::abs(costmap.getOriginX() - 0.0) < 1e-6);
  assert(std::abs(costmap.getOriginY() - 0.0) < 1e-6);

  // Test set/get cost
  costmap.setCost(5, 5, costmap_2d::LETHAL_OBSTACLE);
  assert(costmap.getCost(5, 5) == costmap_2d::LETHAL_OBSTACLE);

  // Test coordinate conversion
  double wx, wy;
  costmap.mapToWorld(5, 5, wx, wy);
  assert(std::abs(wx - 0.275) < 1e-6);  // 0.0 + (5 + 0.5) * 0.05
  assert(std::abs(wy - 0.275) < 1e-6);

  unsigned int mx, my;
  bool in_bounds = costmap.worldToMap(0.275, 0.275, mx, my);
  assert(in_bounds);
  assert(mx == 5);
  assert(my == 5);

  // Test out of bounds
  in_bounds = costmap.worldToMap(-1.0, -1.0, mx, my);
  assert(!in_bounds);

  // Test cellDistance
  unsigned int cell_dist = costmap.cellDistance(0.1);
  assert(cell_dist == 2);  // ceil(0.1 / 0.05)

  // Test copy constructor
  costmap_2d::Costmap2D costmap2(costmap);
  assert(costmap2.getCost(5, 5) == costmap_2d::LETHAL_OBSTACLE);

  std::cout << "Costmap2D tests passed!" << std::endl;
}

void testFootprint()
{
  std::cout << "Testing Footprint..." << std::endl;

  // Create a circular footprint
  std::vector<costmap_2d::Point> footprint = costmap_2d::makeFootprintFromRadius(0.2);
  assert(footprint.size() == 16);

  // Test calculateMinAndMaxDistances
  auto [min_dist, max_dist] = costmap_2d::calculateMinAndMaxDistances(footprint);
  assert(std::abs(min_dist - 0.2) < 0.01);  // radius
  assert(std::abs(max_dist - 0.2) < 0.01);  // radius for circle

  // Test padFootprint
  costmap_2d::padFootprint(footprint, 0.1);
  auto [min_dist2, max_dist2] = costmap_2d::calculateMinAndMaxDistances(footprint);
  assert(min_dist2 > min_dist);

  // Test transformFootprint
  std::vector<costmap_2d::Point> oriented;
  costmap_2d::transformFootprint(1.0, 2.0, 0.0, footprint, oriented);
  assert(oriented.size() == footprint.size());

  std::cout << "Footprint tests passed!" << std::endl;
}

void testInflationCost()
{
  std::cout << "Testing InflationCost..." << std::endl;

  // Test computeInflationCost
  double resolution = 0.05;
  double inscribed_radius = 0.1;
  double cost_scaling_factor = 10.0;

  // At obstacle
  unsigned char cost = costmap_2d::computeInflationCost(
    0.0, resolution, inscribed_radius, cost_scaling_factor);
  assert(cost == costmap_2d::LETHAL_OBSTACLE);

  // At inscribed radius
  cost = costmap_2d::computeInflationCost(
    inscribed_radius / resolution, resolution, inscribed_radius, cost_scaling_factor);
  assert(cost == costmap_2d::INSCRIBED_INFLATED_OBSTACLE);

  // Far from obstacle
  cost = costmap_2d::computeInflationCost(
    10.0, resolution, inscribed_radius, cost_scaling_factor);
  assert(cost < costmap_2d::INSCRIBED_INFLATED_OBSTACLE);

  std::cout << "InflationCost tests passed!" << std::endl;
}

int main()
{
  std::cout << "Running costmap_2d tests..." << std::endl;

  testCostmap2D();
  testFootprint();
  testInflationCost();

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
