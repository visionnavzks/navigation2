/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, 2013, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Eitan Marder-Eppstein
 *         David V. Lu!!
 *********************************************************************/
#include "costmap_2d/costmap_2d.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "costmap_2d/cost_values.hpp"

namespace costmap_2d
{

// Inline raytrace from nav2_util (ROS-free version)
namespace raytrace
{

inline int sign(int x)
{
  return x > 0 ? 1 : -1;
}

template<class ActionType>
inline void bresenham2D(
  ActionType at, unsigned int abs_da, unsigned int abs_db, int error_b,
  int offset_a, int offset_b, unsigned int offset,
  unsigned int max_length)
{
  unsigned int end = std::min(max_length, abs_da);
  for (unsigned int i = 0; i < end; ++i) {
    at(offset);
    offset += offset_a;
    error_b += abs_db;
    if ((unsigned int)error_b >= abs_da) {
      offset += offset_b;
      error_b -= abs_da;
    }
  }
  at(offset);
}

template<class ActionType>
inline void raytraceLine(
  ActionType at, unsigned int x0, unsigned int y0, unsigned int x1,
  unsigned int y1, unsigned int step_x,
  unsigned int max_length = UINT_MAX, unsigned int min_length = 0)
{
  int dx_full = x1 - x0;
  int dy_full = y1 - y0;

  double dist = std::hypot(dx_full, dy_full);
  if (dist < min_length) {
    return;
  }

  unsigned int min_x0, min_y0;
  if (dist > 0.0) {
    min_x0 = (unsigned int)(x0 + dx_full / dist * min_length);
    min_y0 = (unsigned int)(y0 + dy_full / dist * min_length);
  } else {
    min_x0 = x0;
    min_y0 = y0;
  }
  unsigned int offset = min_y0 * step_x + min_x0;

  int dx = x1 - min_x0;
  int dy = y1 - min_y0;

  unsigned int abs_dx = abs(dx);
  unsigned int abs_dy = abs(dy);

  int offset_dx = sign(dx);
  int offset_dy = sign(dy) * step_x;

  double scale = (dist == 0.0) ? 1.0 : std::min(1.0, max_length / dist);
  if (abs_dx >= abs_dy) {
    int error_y = abs_dx / 2;
    bresenham2D(
      at, abs_dx, abs_dy, error_y, offset_dx, offset_dy, offset, (unsigned int)(scale * abs_dx));
    return;
  }

  int error_x = abs_dy / 2;
  bresenham2D(
    at, abs_dy, abs_dx, error_x, offset_dy, offset_dx, offset, (unsigned int)(scale * abs_dy));
}

}  // namespace raytrace

Costmap2D::Costmap2D(
  unsigned int cells_size_x, unsigned int cells_size_y, double resolution,
  double origin_x, double origin_y, unsigned char default_value)
: resolution_(resolution), origin_x_(origin_x),
  origin_y_(origin_y), costmap_(NULL), default_value_(default_value)
{
  access_ = new mutex_t();
  initMaps(cells_size_x, cells_size_y);
  resetMaps();
}

void Costmap2D::deleteMaps()
{
  std::unique_lock<mutex_t> lock(*access_);
  delete[] costmap_;
  costmap_ = NULL;
}

void Costmap2D::initMaps(unsigned int size_x, unsigned int size_y)
{
  std::unique_lock<mutex_t> lock(*access_);
  delete[] costmap_;
  size_x_ = size_x;
  size_y_ = size_y;
  costmap_ = new unsigned char[size_x * size_y];
}

void Costmap2D::resizeMap(
  unsigned int size_x, unsigned int size_y, double resolution,
  double origin_x, double origin_y)
{
  resolution_ = resolution;
  origin_x_ = origin_x;
  origin_y_ = origin_y;
  initMaps(size_x, size_y);
  resetMaps();
}

void Costmap2D::resetMaps()
{
  std::unique_lock<mutex_t> lock(*access_);
  memset(costmap_, default_value_, size_x_ * size_y_ * sizeof(unsigned char));
}

void Costmap2D::resetMap(unsigned int x0, unsigned int y0, unsigned int xn, unsigned int yn)
{
  resetMapToValue(x0, y0, xn, yn, default_value_);
}

void Costmap2D::resetMapToValue(
  unsigned int x0, unsigned int y0, unsigned int xn, unsigned int yn, unsigned char value)
{
  std::unique_lock<mutex_t> lock(*(access_));
  unsigned int len = xn - x0;
  for (unsigned int y = y0 * size_x_ + x0; y < yn * size_x_ + x0; y += size_x_) {
    memset(costmap_ + y, value, len * sizeof(unsigned char));
  }
}

bool Costmap2D::copyCostmapWindow(
  const Costmap2D & map, double win_origin_x, double win_origin_y,
  double win_size_x,
  double win_size_y)
{
  if (this == &map) {
    return false;
  }

  deleteMaps();

  unsigned int lower_left_x, lower_left_y, upper_right_x, upper_right_y;
  if (!map.worldToMap(win_origin_x, win_origin_y, lower_left_x, lower_left_y) ||
    !map.worldToMap(
      win_origin_x + win_size_x, win_origin_y + win_size_y, upper_right_x,
      upper_right_y))
  {
    return false;
  }
  resolution_ = map.resolution_;
  origin_x_ = win_origin_x;
  origin_y_ = win_origin_y;

  initMaps(upper_right_x - lower_left_x, upper_right_y - lower_left_y);

  copyMapRegion(
    map.costmap_, lower_left_x, lower_left_y, map.size_x_, costmap_, 0, 0, size_x_,
    size_x_,
    size_y_);
  return true;
}

bool Costmap2D::copyWindow(
  const Costmap2D & source,
  unsigned int sx0, unsigned int sy0, unsigned int sxn, unsigned int syn,
  unsigned int dx0, unsigned int dy0)
{
  const unsigned int sz_x = sxn - sx0;
  const unsigned int sz_y = syn - sy0;

  if (sxn > source.getSizeInCellsX() || syn > source.getSizeInCellsY()) {
    return false;
  }

  if (dx0 + sz_x > size_x_ || dy0 + sz_y > size_y_) {
    return false;
  }

  copyMapRegion(
    source.costmap_, sx0, sy0, source.size_x_,
    costmap_, dx0, dy0, size_x_,
    sz_x, sz_y);
  return true;
}

Costmap2D & Costmap2D::operator=(const Costmap2D & map)
{
  if (this == &map) {
    return *this;
  }

  deleteMaps();

  size_x_ = map.size_x_;
  size_y_ = map.size_y_;
  resolution_ = map.resolution_;
  origin_x_ = map.origin_x_;
  origin_y_ = map.origin_y_;
  default_value_ = map.default_value_;

  initMaps(size_x_, size_y_);

  memcpy(costmap_, map.costmap_, size_x_ * size_y_ * sizeof(unsigned char));

  return *this;
}

Costmap2D::Costmap2D(const Costmap2D & map)
: costmap_(NULL)
{
  access_ = new mutex_t();
  *this = map;
}

Costmap2D::Costmap2D()
: size_x_(0), size_y_(0), resolution_(0.0), origin_x_(0.0), origin_y_(0.0), costmap_(NULL)
{
  access_ = new mutex_t();
}

Costmap2D::~Costmap2D()
{
  deleteMaps();
  delete access_;
}

unsigned int Costmap2D::cellDistance(double world_dist)
{
  double cells_dist = std::max(0.0, ceil(world_dist / resolution_));
  return (unsigned int)cells_dist;
}

unsigned char * Costmap2D::getCharMap() const
{
  return costmap_;
}

unsigned char Costmap2D::getCost(unsigned int mx, unsigned int my) const
{
  return costmap_[getIndex(mx, my)];
}

unsigned char Costmap2D::getCost(unsigned int index) const
{
  return costmap_[index];
}

void Costmap2D::setCost(unsigned int mx, unsigned int my, unsigned char cost)
{
  costmap_[getIndex(mx, my)] = cost;
}

void Costmap2D::mapToWorld(unsigned int mx, unsigned int my, double & wx, double & wy) const
{
  wx = origin_x_ + (mx + 0.5) * resolution_;
  wy = origin_y_ + (my + 0.5) * resolution_;
}

void Costmap2D::mapToWorldNoBounds(int mx, int my, double & wx, double & wy) const
{
  wx = origin_x_ + (mx + 0.5) * resolution_;
  wy = origin_y_ + (my + 0.5) * resolution_;
}

bool Costmap2D::worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const
{
  if (wx < origin_x_ || wy < origin_y_) {
    return false;
  }

  mx = static_cast<unsigned int>((wx - origin_x_) / resolution_);
  my = static_cast<unsigned int>((wy - origin_y_) / resolution_);

  if (mx < size_x_ && my < size_y_) {
    return true;
  }
  return false;
}

bool Costmap2D::worldToMapContinuous(double wx, double wy, float & mx, float & my) const
{
  if (wx < origin_x_ || wy < origin_y_) {
    return false;
  }

  mx = static_cast<float>((wx - origin_x_) / resolution_);
  my = static_cast<float>((wy - origin_y_) / resolution_);

  if (mx < size_x_ && my < size_y_) {
    return true;
  }
  return false;
}

void Costmap2D::worldToMapNoBounds(double wx, double wy, int & mx, int & my) const
{
  mx = static_cast<int>((wx - origin_x_) / resolution_);
  my = static_cast<int>((wy - origin_y_) / resolution_);
}

void Costmap2D::worldToMapEnforceBounds(double wx, double wy, int & mx, int & my) const
{
  if (wx < origin_x_) {
    mx = 0;
  } else if (wx > resolution_ * size_x_ + origin_x_) {
    mx = size_x_ - 1;
  } else {
    mx = static_cast<int>((wx - origin_x_) / resolution_);
  }

  if (wy < origin_y_) {
    my = 0;
  } else if (wy > resolution_ * size_y_ + origin_y_) {
    my = size_y_ - 1;
  } else {
    my = static_cast<int>((wy - origin_y_) / resolution_);
  }
}

void Costmap2D::updateOrigin(double new_origin_x, double new_origin_y)
{
  int cell_ox, cell_oy;
  cell_ox = static_cast<int>((new_origin_x - origin_x_) / resolution_);
  cell_oy = static_cast<int>((new_origin_y - origin_y_) / resolution_);

  double new_grid_ox, new_grid_oy;
  new_grid_ox = origin_x_ + cell_ox * resolution_;
  new_grid_oy = origin_y_ + cell_oy * resolution_;

  int size_x = size_x_;
  int size_y = size_y_;

  int lower_left_x, lower_left_y, upper_right_x, upper_right_y;
  lower_left_x = std::min(std::max(cell_ox, 0), size_x);
  lower_left_y = std::min(std::max(cell_oy, 0), size_y);
  upper_right_x = std::min(std::max(cell_ox + size_x, 0), size_x);
  upper_right_y = std::min(std::max(cell_oy + size_y, 0), size_y);

  unsigned int cell_size_x = upper_right_x - lower_left_x;
  unsigned int cell_size_y = upper_right_y - lower_left_y;

  unsigned char * local_map = new unsigned char[cell_size_x * cell_size_y];

  copyMapRegion(
    costmap_, lower_left_x, lower_left_y, size_x_, local_map, 0, 0, cell_size_x,
    cell_size_x,
    cell_size_y);

  resetMaps();

  origin_x_ = new_grid_ox;
  origin_y_ = new_grid_oy;

  int start_x = lower_left_x - cell_ox;
  int start_y = lower_left_y - cell_oy;

  copyMapRegion(
    local_map, 0, 0, cell_size_x, costmap_, start_x, start_y, size_x_, cell_size_x,
    cell_size_y);

  delete[] local_map;
}

bool Costmap2D::setConvexPolygonCost(
  const std::vector<Point> & polygon,
  unsigned char cost_value)
{
  std::vector<MapLocation> polygon_map_region;
  polygon_map_region.reserve(100);
  if (!getMapRegionOccupiedByPolygon(polygon, polygon_map_region)) {
    return false;
  }

  setMapRegionOccupiedByPolygon(polygon_map_region, cost_value);
  return true;
}

void Costmap2D::setMapRegionOccupiedByPolygon(
  const std::vector<MapLocation> & polygon_map_region,
  unsigned char new_cost_value)
{
  for (const auto & cell : polygon_map_region) {
    setCost(cell.x, cell.y, new_cost_value);
  }
}

void Costmap2D::restoreMapRegionOccupiedByPolygon(
  const std::vector<MapLocation> & polygon_map_region)
{
  for (const auto & cell : polygon_map_region) {
    setCost(cell.x, cell.y, cell.cost);
  }
}

bool Costmap2D::getMapRegionOccupiedByPolygon(
  const std::vector<Point> & polygon,
  std::vector<MapLocation> & polygon_map_region)
{
  std::vector<MapLocation> map_polygon;
  for (const auto & cell : polygon) {
    MapLocation loc;
    if (!worldToMap(cell.x, cell.y, loc.x, loc.y)) {
      return false;
    }
    map_polygon.push_back(loc);
  }

  convexFillCells(map_polygon, polygon_map_region);
  return true;
}

void Costmap2D::polygonOutlineCells(
  const std::vector<MapLocation> & polygon,
  std::vector<MapLocation> & polygon_cells)
{
  PolygonOutlineCells cell_gatherer(*this, costmap_, polygon_cells);
  for (unsigned int i = 0; i < polygon.size() - 1; ++i) {
    raytrace::raytraceLine(
      cell_gatherer, polygon[i].x, polygon[i].y, polygon[i + 1].x, polygon[i + 1].y, size_x_);
  }
  if (!polygon.empty()) {
    unsigned int last_index = polygon.size() - 1;
    raytrace::raytraceLine(
      cell_gatherer, polygon[last_index].x, polygon[last_index].y, polygon[0].x,
      polygon[0].y, size_x_);
  }
}

void Costmap2D::convexFillCells(
  const std::vector<MapLocation> & polygon,
  std::vector<MapLocation> & polygon_cells)
{
  if (polygon.size() < 3) {
    return;
  }

  polygonOutlineCells(polygon, polygon_cells);

  MapLocation swap;
  unsigned int i = 0;
  while (i < polygon_cells.size() - 1) {
    if (polygon_cells[i].x > polygon_cells[i + 1].x) {
      swap = polygon_cells[i];
      polygon_cells[i] = polygon_cells[i + 1];
      polygon_cells[i + 1] = swap;

      if (i > 0) {
        --i;
      }
    } else {
      ++i;
    }
  }

  i = 0;
  MapLocation min_pt;
  MapLocation max_pt;
  unsigned int min_x = polygon_cells[0].x;
  unsigned int max_x = polygon_cells[polygon_cells.size() - 1].x;

  for (unsigned int x = min_x; x <= max_x; ++x) {
    if (i >= polygon_cells.size() - 1) {
      break;
    }

    if (polygon_cells[i].y < polygon_cells[i + 1].y) {
      min_pt = polygon_cells[i];
      max_pt = polygon_cells[i + 1];
    } else {
      min_pt = polygon_cells[i + 1];
      max_pt = polygon_cells[i];
    }

    i += 2;
    while (i < polygon_cells.size() && polygon_cells[i].x == x) {
      if (polygon_cells[i].y < min_pt.y) {
        min_pt = polygon_cells[i];
      } else if (polygon_cells[i].y > max_pt.y) {
        max_pt = polygon_cells[i];
      }
      ++i;
    }

    MapLocation pt;
    for (unsigned int y = min_pt.y; y <= max_pt.y; ++y) {
      pt.x = x;
      pt.y = y;
      pt.cost = getCost(x, y);
      polygon_cells.push_back(pt);
    }
  }
}

unsigned int Costmap2D::getSizeInCellsX() const {return size_x_;}
unsigned int Costmap2D::getSizeInCellsY() const {return size_y_;}
double Costmap2D::getSizeInMetersX() const {return size_x_ * resolution_;}
double Costmap2D::getSizeInMetersY() const {return size_y_ * resolution_;}
double Costmap2D::getOriginX() const {return origin_x_;}
double Costmap2D::getOriginY() const {return origin_y_;}
double Costmap2D::getResolution() const {return resolution_;}

bool Costmap2D::saveMap(std::string file_name)
{
  FILE * fp = fopen(file_name.c_str(), "w");
  if (!fp) {
    return false;
  }

  fprintf(fp, "P2\n%u\n%u\n%u\n", size_x_, size_y_, 0xff);
  for (unsigned int iy = 0; iy < size_y_; iy++) {
    for (unsigned int ix = 0; ix < size_x_; ix++) {
      unsigned char cost = getCost(ix, iy);
      fprintf(fp, "%d ", cost);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  return true;
}

}  // namespace costmap_2d
