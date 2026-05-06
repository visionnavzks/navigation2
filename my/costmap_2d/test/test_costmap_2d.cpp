#include <cassert>
#include <iostream>
#include <set>
#include <utility>

#include "../include/costmap_2d/costmap_2d.hpp"
#include "../include/costmap_2d/costmap_layer.hpp"
#include "../include/costmap_2d/keepout_filter.hpp"

namespace {

void testCopyWindow() {
  costmap_2d::Costmap2D src(10, 10, 0.1, 0.0, 0.0);
  costmap_2d::Costmap2D dst(5, 5, 0.2, 100.0, 100.0);
  src.setCost(2, 2, 100);
  src.setCost(5, 5, 200);

  assert(dst.copyWindow(src, 2, 2, 6, 6, 0, 0));
  assert(dst.getCost(0, 0) == 100);
  assert(dst.getCost(3, 3) == 200);
  assert(!dst.copyWindow(src, 9, 9, 11, 11, 0, 0));
  assert(!dst.copyWindow(src, 0, 0, 1, 1, 5, 5));
  assert(!dst.copyWindow(src, 0, 0, 6, 6, 0, 0));
}

void testWorldToMapAndMaskCost() {
  costmap_2d::OccupancyGrid mask(3, 3, 1.0, 3.0, 3.0,
                                 costmap_2d::OCC_GRID_OCCUPIED);

  unsigned int mx, my;
  assert(costmap_2d::worldToMap(mask, 4.0, 5.0, mx, my));
  assert(mx == 1 && my == 2);
  assert(costmap_2d::worldToMap(mask, 3.0, 3.0, mx, my));
  assert(mx == 0 && my == 0);
  assert(costmap_2d::worldToMap(mask, 5.9, 5.9, mx, my));
  assert(mx == 2 && my == 2);
  assert(!costmap_2d::worldToMap(mask, 2.9, 2.9, mx, my));
  assert(!costmap_2d::worldToMap(mask, 6.0, 6.0, mx, my));

  costmap_2d::OccupancyGrid values(2, 2, 1.0, 0.0, 0.0);
  values.setData(0, 0, costmap_2d::OCC_GRID_UNKNOWN);
  values.setData(1, 0, costmap_2d::OCC_GRID_FREE);
  values.setData(0, 1, costmap_2d::OCC_GRID_OCCUPIED / 2);
  values.setData(1, 1, costmap_2d::OCC_GRID_OCCUPIED);

  costmap_2d::KeepoutFilter filter;
  assert(filter.getMaskCost(values, 0, 0) == costmap_2d::NO_INFORMATION);
  assert(filter.getMaskCost(values, 1, 0) == costmap_2d::FREE_SPACE);
  assert(filter.getMaskCost(values, 0, 1) == costmap_2d::LETHAL_OBSTACLE / 2);
  assert(filter.getMaskCost(values, 1, 1) == costmap_2d::LETHAL_OBSTACLE);
}

void verifyMasterGrid(
    const costmap_2d::Costmap2D &master, unsigned char free_value,
    unsigned char keepout_value,
    const std::set<std::pair<unsigned int, unsigned int>> &keepout_points) {
  for (unsigned int y = 0; y < master.getSizeInCellsY(); ++y) {
    for (unsigned int x = 0; x < master.getSizeInCellsX(); ++x) {
      if (keepout_points.count({x, y}) != 0) {
        assert(master.getCost(x, y) == keepout_value);
      } else {
        assert(master.getCost(x, y) == free_value);
      }
    }
  }
}

void testKeepoutStandardScenario() {
  const unsigned char free_value = costmap_2d::FREE_SPACE;
  const unsigned char keepout_value = costmap_2d::LETHAL_OBSTACLE;
  costmap_2d::Costmap2D master(10, 10, 1.0, 0.0, 0.0, free_value);
  costmap_2d::OccupancyGrid mask(3, 3, 1.0, 3.0, 3.0,
                                 costmap_2d::OCC_GRID_OCCUPIED, "map");
  costmap_2d::KeepoutFilter filter;
  filter.setGlobalFrame("map");
  filter.setMask(mask);

  std::set<std::pair<unsigned int, unsigned int>> keepout_points;

  filter.process(master, 2, 2, 5, 5);
  keepout_points.insert({3, 3});
  keepout_points.insert({3, 4});
  keepout_points.insert({4, 3});
  keepout_points.insert({4, 4});
  verifyMasterGrid(master, free_value, keepout_value, keepout_points);

  filter.process(master, 3, 6, 5, 7);
  filter.process(master, 6, 3, 7, 5);
  verifyMasterGrid(master, free_value, keepout_value, keepout_points);

  filter.process(master, 5, 5, 6, 6);
  keepout_points.insert({5, 5});
  verifyMasterGrid(master, free_value, keepout_value, keepout_points);

  filter.process(master, 0, 0, 2, 2);
  filter.process(master, 0, 7, 2, 9);
  filter.process(master, 7, 0, 9, 2);
  filter.process(master, 7, 7, 9, 9);
  verifyMasterGrid(master, free_value, keepout_value, keepout_points);
}

void testLayerMergePolicies() {
  costmap_2d::Costmap2D master(3, 3, 1.0, 0.0, 0.0, costmap_2d::FREE_SPACE);
  costmap_2d::Costmap2D layer(3, 3, 1.0, 0.0, 0.0, costmap_2d::NO_INFORMATION);
  layer.setCost(1, 1, 50);
  costmap_2d::updateWithOverwrite(layer, master, 0, 0, 3, 3);
  assert(master.getCost(1, 1) == 50);
  assert(master.getCost(0, 0) == costmap_2d::FREE_SPACE);

  layer.setCost(2, 2, 100);
  master.setCost(2, 2, 20);
  costmap_2d::updateWithMax(layer, master, 0, 0, 3, 3);
  assert(master.getCost(2, 2) == 100);

  master.setCost(0, 0, costmap_2d::NO_INFORMATION);
  layer.setCost(0, 0, 100);
  costmap_2d::updateWithMaxWithoutUnknownOverwrite(layer, master, 0, 0, 3, 3);
  assert(master.getCost(0, 0) == costmap_2d::NO_INFORMATION);

  master.setCost(0, 1, 200);
  layer.setCost(0, 1, 100);
  costmap_2d::updateWithAddition(layer, master, 0, 0, 3, 3);
  assert(master.getCost(0, 1) == costmap_2d::INSCRIBED_INFLATED_OBSTACLE - 1);
}

} // namespace

int main() {
  testCopyWindow();
  testWorldToMapAndMaskCost();
  testKeepoutStandardScenario();
  testLayerMergePolicies();
  std::cout << "costmap_2d standalone tests passed\n";
  return 0;
}