// Copyright (c) 2008, Willow Garage, Inc.
// All rights reserved.
//
// Software License Agreement (BSD License 2.0)
//
// Ported to ROS-free my_costmap_2d library

#include <gtest/gtest.h>
#include <set>
#include <vector>
#include <cmath>

#include "my_costmap_2d/costmap_2d.hpp"
#include "my_costmap_2d/cost_values.hpp"

using namespace my_costmap_2d;

const unsigned char MAP_10_BY_10_CHAR[] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 200, 200, 200,
  0, 0, 0, 0, 100, 0, 0, 200, 200, 200,
  0, 0, 0, 0, 100, 0, 0, 200, 200, 200,
  70, 70, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 200, 200, 200, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
  0, 0, 0, 0, 0, 0, 0, 255, 255, 255
};

const unsigned int GRID_WIDTH(10);
const unsigned int GRID_HEIGHT(10);
const double RESOLUTION(1.0);

std::vector<unsigned char> MAP_10_BY_10;
std::vector<unsigned char> EMPTY_10_BY_10;

bool find(const std::vector<unsigned int> & l, unsigned int n)
{
  for (auto it = l.begin(); it != l.end(); ++it) {
    if (*it == n) {return true;}
  }
  return false;
}

// ===================== Costmap2D Core Tests =====================

TEST(Costmap2D, testDefaultConstructor)
{
  Costmap2D map;
  EXPECT_EQ(map.getSizeInCellsX(), 0u);
  EXPECT_EQ(map.getSizeInCellsY(), 0u);
  EXPECT_DOUBLE_EQ(map.getResolution(), 0.0);
  EXPECT_DOUBLE_EQ(map.getOriginX(), 0.0);
  EXPECT_DOUBLE_EQ(map.getOriginY(), 0.0);
}

TEST(Costmap2D, testParameterizedConstructor)
{
  Costmap2D map(10, 10, 0.05, 1.0, 2.0);
  EXPECT_EQ(map.getSizeInCellsX(), 10u);
  EXPECT_EQ(map.getSizeInCellsY(), 10u);
  EXPECT_DOUBLE_EQ(map.getResolution(), 0.05);
  EXPECT_DOUBLE_EQ(map.getOriginX(), 1.0);
  EXPECT_DOUBLE_EQ(map.getOriginY(), 2.0);
  EXPECT_EQ(map.getDefaultValue(), 0u);
}

TEST(Costmap2D, testDefaultValue)
{
  Costmap2D map(5, 5, 1.0, 0.0, 0.0, NO_INFORMATION);
  EXPECT_EQ(map.getDefaultValue(), NO_INFORMATION);
  for (unsigned int i = 0; i < 5; ++i) {
    for (unsigned int j = 0; j < 5; ++j) {
      EXPECT_EQ(map.getCost(i, j), NO_INFORMATION);
    }
  }
}

TEST(Costmap2D, testCopyConstructor)
{
  Costmap2D map(10, 10, 0.1, 0.0, 0.0);
  map.setCost(5, 5, LETHAL_OBSTACLE);

  Costmap2D map2(map);
  EXPECT_EQ(map2.getSizeInCellsX(), 10u);
  EXPECT_EQ(map2.getCost(5, 5), LETHAL_OBSTACLE);
}

TEST(Costmap2D, testAssignmentOperator)
{
  Costmap2D map(10, 10, 0.1, 0.0, 0.0);
  map.setCost(3, 3, 200);

  Costmap2D map2;
  map2 = map;
  EXPECT_EQ(map2.getSizeInCellsX(), 10u);
  EXPECT_EQ(map2.getCost(3, 3), 200u);
}

TEST(Costmap2D, testSetGetCost)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);
  map.setCost(5, 5, LETHAL_OBSTACLE);
  EXPECT_EQ(map.getCost(5, 5), LETHAL_OBSTACLE);

  map.setCost(0, 0, FREE_SPACE);
  EXPECT_EQ(map.getCost(0, 0), FREE_SPACE);

  map.setCost(9, 9, 128);
  EXPECT_EQ(map.getCost(9, 9), 128u);
}

TEST(Costmap2D, testGetCostByIndex)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);
  map.setCost(5, 5, LETHAL_OBSTACLE);
  unsigned int index = map.getIndex(5, 5);
  EXPECT_EQ(map.getCost(index), LETHAL_OBSTACLE);
}

// ===================== Coordinate Transform Tests =====================

TEST(Costmap2D, testMapToWorld)
{
  Costmap2D map(10, 10, 0.1, 0.0, 0.0);
  double wx, wy;
  map.mapToWorld(0, 0, wx, wy);
  EXPECT_DOUBLE_EQ(wx, 0.05);
  EXPECT_DOUBLE_EQ(wy, 0.05);

  map.mapToWorld(5, 5, wx, wy);
  EXPECT_DOUBLE_EQ(wx, 0.55);
  EXPECT_DOUBLE_EQ(wy, 0.55);
}

TEST(Costmap2D, testMapToWorldWithOrigin)
{
  Costmap2D map(10, 10, 1.0, 5.0, 10.0);
  double wx, wy;
  map.mapToWorld(0, 0, wx, wy);
  EXPECT_DOUBLE_EQ(wx, 5.5);
  EXPECT_DOUBLE_EQ(wy, 10.5);
}

TEST(Costmap2D, testWorldToMap)
{
  Costmap2D map(10, 10, 0.1, 0.0, 0.0);
  unsigned int mx, my;

  EXPECT_TRUE(map.worldToMap(0.05, 0.05, mx, my));
  EXPECT_EQ(mx, 0u);
  EXPECT_EQ(my, 0u);

  EXPECT_TRUE(map.worldToMap(0.55, 0.55, mx, my));
  EXPECT_EQ(mx, 5u);
  EXPECT_EQ(my, 5u);
}

TEST(Costmap2D, testWorldToMapOutOfBounds)
{
  Costmap2D map(10, 10, 0.1, 0.0, 0.0);
  unsigned int mx, my;

  EXPECT_FALSE(map.worldToMap(-1.0, 0.5, mx, my));
  EXPECT_FALSE(map.worldToMap(0.5, -1.0, mx, my));
  EXPECT_FALSE(map.worldToMap(2.0, 0.5, mx, my));
  EXPECT_FALSE(map.worldToMap(0.5, 2.0, mx, my));
}

TEST(Costmap2D, testMapToWorldNoBounds)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);
  double wx, wy;

  map.mapToWorldNoBounds(-1, -1, wx, wy);
  EXPECT_DOUBLE_EQ(wx, -0.5);
  EXPECT_DOUBLE_EQ(wy, -0.5);

  Costmap2D map2(10, 10, 1.0, 1.0, 2.0);
  map2.mapToWorldNoBounds(-5, -5, wx, wy);
  EXPECT_DOUBLE_EQ(wx, -3.5);
  EXPECT_DOUBLE_EQ(wy, -2.5);

  Costmap2D map3(10, 10, 2.0, 3.0, 4.0);
  map3.mapToWorldNoBounds(-10, -10, wx, wy);
  EXPECT_DOUBLE_EQ(wx, -16.0);
  EXPECT_DOUBLE_EQ(wy, -15.0);
}

TEST(Costmap2D, testWorldToMapEnforceBounds)
{
  Costmap2D map(10, 10, 0.1, 0.0, 0.0);
  int mx, my;

  map.worldToMapEnforceBounds(-1.0, -1.0, mx, my);
  EXPECT_EQ(mx, 0);
  EXPECT_EQ(my, 0);

  map.worldToMapEnforceBounds(5.0, 5.0, mx, my);
  EXPECT_EQ(mx, 9);
  EXPECT_EQ(my, 9);
}

TEST(Costmap2D, testWorldToMapContinuous)
{
  Costmap2D map(10, 10, 0.1, 0.0, 0.0);
  float mx, my;

  EXPECT_TRUE(map.worldToMapContinuous(0.55, 0.55, mx, my));
  EXPECT_NEAR(mx, 5.5f, 0.01f);
  EXPECT_NEAR(my, 5.5f, 0.01f);
}

TEST(Costmap2D, testIndexToCells)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);
  unsigned int mx, my;

  map.indexToCells(0, mx, my);
  EXPECT_EQ(mx, 0u);
  EXPECT_EQ(my, 0u);

  map.indexToCells(55, mx, my);
  EXPECT_EQ(mx, 5u);
  EXPECT_EQ(my, 5u);

  map.indexToCells(99, mx, my);
  EXPECT_EQ(mx, 9u);
  EXPECT_EQ(my, 9u);
}

TEST(Costmap2D, testGetIndex)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);
  EXPECT_EQ(map.getIndex(0, 0), 0u);
  EXPECT_EQ(map.getIndex(5, 5), 55u);
  EXPECT_EQ(map.getIndex(9, 9), 99u);
}

TEST(Costmap2D, testWorldToIndexRoundTrip)
{
  Costmap2D map(GRID_WIDTH, GRID_HEIGHT, RESOLUTION, 0.0, 0.0);

  auto worldToIndex = [&](double wx, double wy) {
      unsigned int mx, my;
      map.worldToMap(wx, wy, mx, my);
      return map.getIndex(mx, my);
    };

  EXPECT_EQ(worldToIndex(0.0, 0.0), 0u);
  EXPECT_EQ(worldToIndex(0.0, 0.99), 0u);
  EXPECT_EQ(worldToIndex(0.0, 1.0), 10u);
  EXPECT_EQ(worldToIndex(1.0, 0.99), 1u);
  EXPECT_EQ(worldToIndex(9.99, 9.99), 99u);
  EXPECT_EQ(worldToIndex(8.2, 3.4), 38u);
}

// ===================== Window Copy Tests =====================

TEST(Costmap2D, testCopyCostmapWindowTooBig)
{
  Costmap2D map(10, 10, RESOLUTION, 0.0, 0.0);
  Costmap2D windowCopy;

  bool result = windowCopy.copyCostmapWindow(map, 2.0, 2.0, 6.0, 12.0);
  EXPECT_FALSE(result);
  EXPECT_EQ(windowCopy.getSizeInCellsX(), 0u);
  EXPECT_EQ(windowCopy.getSizeInCellsY(), 0u);
}

TEST(Costmap2D, testCopyCostmapWindowSelf)
{
  Costmap2D map(10, 10, RESOLUTION, 0.0, 0.0);
  bool result = map.copyCostmapWindow(map, 2.0, 2.0, 6.0, 6.0);
  EXPECT_FALSE(result);
  EXPECT_EQ(map.getSizeInCellsX(), 10u);
  EXPECT_EQ(map.getSizeInCellsY(), 10u);
}

TEST(Costmap2D, testCopyCostmapWindowValid)
{
  Costmap2D map(10, 10, RESOLUTION, 0.0, 0.0);
  Costmap2D windowCopy;

  bool result = windowCopy.copyCostmapWindow(map, 2.0, 2.0, 6.0, 6.0);
  EXPECT_TRUE(result);
  EXPECT_EQ(windowCopy.getSizeInCellsX(), 6u);
  EXPECT_EQ(windowCopy.getSizeInCellsY(), 6u);

  for (unsigned int i = 0; i < windowCopy.getSizeInCellsX(); ++i) {
    for (unsigned int j = 0; j < windowCopy.getSizeInCellsY(); ++j) {
      EXPECT_EQ(windowCopy.getCost(i, j), map.getCost(i + 2, j + 2));
    }
  }
}

TEST(Costmap2D, testCopyWindow)
{
  Costmap2D src(10, 10, 0.1, 0.0, 0.0);
  Costmap2D dst(5, 5, 0.2, 100.0, 100.0);

  src.setCost(2, 2, 100);
  src.setCost(5, 5, 200);

  EXPECT_TRUE(dst.copyWindow(src, 2, 2, 6, 6, 0, 0));
  EXPECT_EQ(dst.getCost(0, 0), 100u);
  EXPECT_EQ(dst.getCost(3, 3), 200u);
}

TEST(Costmap2D, testCopyWindowInvalidSource)
{
  Costmap2D src(10, 10, 0.1, 0.0, 0.0);
  Costmap2D dst(5, 5, 0.2, 100.0, 100.0);

  EXPECT_FALSE(dst.copyWindow(src, 9, 9, 11, 11, 0, 0));
}

TEST(Costmap2D, testCopyWindowInvalidDest)
{
  Costmap2D src(10, 10, 0.1, 0.0, 0.0);
  Costmap2D dst(5, 5, 0.2, 100.0, 100.0);

  EXPECT_FALSE(dst.copyWindow(src, 0, 0, 1, 1, 5, 5));
}

// ===================== Resize and Reset Tests =====================

TEST(Costmap2D, testResizeMap)
{
  Costmap2D map(5, 5, 0.1, 0.0, 0.0);
  map.setCost(2, 2, 100);

  map.resizeMap(10, 10, 0.05, 1.0, 2.0);
  EXPECT_EQ(map.getSizeInCellsX(), 10u);
  EXPECT_EQ(map.getSizeInCellsY(), 10u);
  EXPECT_DOUBLE_EQ(map.getResolution(), 0.05);
  EXPECT_DOUBLE_EQ(map.getOriginX(), 1.0);
  EXPECT_DOUBLE_EQ(map.getOriginY(), 2.0);
}

TEST(Costmap2D, testResetMap)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);
  map.setCost(5, 5, LETHAL_OBSTACLE);
  EXPECT_EQ(map.getCost(5, 5), LETHAL_OBSTACLE);

  map.resetMap(0, 0, 10, 10);
  EXPECT_EQ(map.getCost(5, 5), FREE_SPACE);
}

TEST(Costmap2D, testResetMapToValue)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0, NO_INFORMATION);
  map.setCost(5, 5, LETHAL_OBSTACLE);

  map.resetMapToValue(0, 0, 10, 10, FREE_SPACE);
  EXPECT_EQ(map.getCost(5, 5), FREE_SPACE);
}

TEST(Costmap2D, testCellDistance)
{
  Costmap2D map(10, 10, 0.05, 0.0, 0.0);
  EXPECT_EQ(map.cellDistance(0.0), 0u);
  EXPECT_EQ(map.cellDistance(0.01), 1u);
  EXPECT_EQ(map.cellDistance(0.05), 1u);
  EXPECT_EQ(map.cellDistance(0.06), 2u);
  EXPECT_EQ(map.cellDistance(1.0), 20u);
}

// ===================== Static Map Tests =====================

TEST(Costmap2D, testStaticMap)
{
  Costmap2D map(GRID_WIDTH, GRID_HEIGHT, RESOLUTION, 0.0, 0.0);
  for (unsigned int i = 0; i < GRID_WIDTH * GRID_HEIGHT; ++i) {
    unsigned int x = i % GRID_WIDTH;
    unsigned int y = i / GRID_WIDTH;
    map.setCost(x, y, MAP_10_BY_10_CHAR[i]);
  }

  EXPECT_EQ(map.getSizeInCellsX(), GRID_WIDTH);
  EXPECT_EQ(map.getSizeInCellsY(), GRID_HEIGHT);

  std::vector<unsigned int> occupiedCells;
  for (unsigned int i = 0; i < 10; ++i) {
    for (unsigned int j = 0; j < 10; ++j) {
      if (map.getCost(i, j) >= 100) {
        occupiedCells.push_back(map.getIndex(i, j));
      }
    }
  }

  EXPECT_GT(occupiedCells.size(), 0u);

  for (auto it = occupiedCells.begin(); it != occupiedCells.end(); ++it) {
    unsigned int ind = *it;
    unsigned int x, y;
    map.indexToCells(ind, x, y);
    EXPECT_TRUE(find(occupiedCells, map.getIndex(x, y)));
  }
}

// ===================== Polygon Fill Tests =====================

TEST(Costmap2D, testConvexFillCells)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);

  std::vector<MapLocation> polygon;
  MapLocation p1{2, 2, 0}; polygon.push_back(p1);
  MapLocation p2{5, 2, 0}; polygon.push_back(p2);
  MapLocation p3{5, 5, 0}; polygon.push_back(p3);
  MapLocation p4{2, 5, 0}; polygon.push_back(p4);

  std::vector<MapLocation> cells;
  map.convexFillCells(polygon, cells);

  EXPECT_GT(cells.size(), 0u);
}

TEST(Costmap2D, testSetConvexPolygonCost)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);

  std::vector<Point> polygon;
  polygon.push_back({2.5, 2.5, 0.0});
  polygon.push_back({5.5, 2.5, 0.0});
  polygon.push_back({5.5, 5.5, 0.0});
  polygon.push_back({2.5, 5.5, 0.0});

  bool result = map.setConvexPolygonCost(polygon, LETHAL_OBSTACLE);
  EXPECT_TRUE(result);

  EXPECT_EQ(map.getCost(3, 3), LETHAL_OBSTACLE);
  EXPECT_EQ(map.getCost(4, 4), LETHAL_OBSTACLE);
}

// ===================== Save Map Test =====================

TEST(Costmap2D, testSaveMap)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);
  map.setCost(5, 5, LETHAL_OBSTACLE);

  bool result = map.saveMap("/tmp/test_costmap.pgm");
  EXPECT_TRUE(result);
}

// ===================== Update Origin Test =====================

TEST(Costmap2D, testUpdateOrigin)
{
  Costmap2D map(10, 10, 1.0, 0.0, 0.0);
  map.setCost(5, 5, LETHAL_OBSTACLE);

  map.updateOrigin(1.0, 1.0);
  EXPECT_DOUBLE_EQ(map.getOriginX(), 1.0);
  EXPECT_DOUBLE_EQ(map.getOriginY(), 1.0);
}

int main(int argc, char ** argv)
{
  for (unsigned int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
    EMPTY_10_BY_10.push_back(0);
    MAP_10_BY_10.push_back(MAP_10_BY_10_CHAR[i]);
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
