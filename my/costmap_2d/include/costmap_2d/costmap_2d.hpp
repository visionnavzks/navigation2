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
#ifndef COSTMAP_2D__COSTMAP_2D_HPP_
#define COSTMAP_2D__COSTMAP_2D_HPP_

#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <mutex>

#include "costmap_2d/point.hpp"

namespace costmap_2d
{

struct MapLocation
{
  unsigned int x;
  unsigned int y;
  unsigned char cost;
};

/**
 * @class Costmap2D
 * @brief A 2D costmap provides a mapping between points in the world and their associated "costs".
 */
class Costmap2D
{
  friend class CostmapTester;

public:
  Costmap2D(
    unsigned int cells_size_x, unsigned int cells_size_y, double resolution,
    double origin_x, double origin_y, unsigned char default_value = 0);

  Costmap2D(const Costmap2D & map);

  Costmap2D & operator=(const Costmap2D & map);

  bool copyCostmapWindow(
    const Costmap2D & map, double win_origin_x, double win_origin_y,
    double win_size_x,
    double win_size_y);

  bool copyWindow(
    const Costmap2D & source,
    unsigned int sx0, unsigned int sy0, unsigned int sxn, unsigned int syn,
    unsigned int dx0, unsigned int dy0);

  Costmap2D();

  virtual ~Costmap2D();

  unsigned char getCost(unsigned int mx, unsigned int my) const;

  unsigned char getCost(unsigned int index) const;

  void setCost(unsigned int mx, unsigned int my, unsigned char cost);

  void mapToWorld(unsigned int mx, unsigned int my, double & wx, double & wy) const;

  void mapToWorldNoBounds(int mx, int my, double & wx, double & wy) const;

  bool worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const;

  bool worldToMapContinuous(double wx, double wy, float & mx, float & my) const;

  void worldToMapNoBounds(double wx, double wy, int & mx, int & my) const;

  void worldToMapEnforceBounds(double wx, double wy, int & mx, int & my) const;

  inline unsigned int getIndex(unsigned int mx, unsigned int my) const
  {
    return my * size_x_ + mx;
  }

  inline void indexToCells(unsigned int index, unsigned int & mx, unsigned int & my) const
  {
    my = index / size_x_;
    mx = index - (my * size_x_);
  }

  unsigned char * getCharMap() const;

  unsigned int getSizeInCellsX() const;

  unsigned int getSizeInCellsY() const;

  double getSizeInMetersX() const;

  double getSizeInMetersY() const;

  double getOriginX() const;

  double getOriginY() const;

  double getResolution() const;

  void setDefaultValue(unsigned char c)
  {
    default_value_ = c;
  }

  unsigned char getDefaultValue()
  {
    return default_value_;
  }

  bool setConvexPolygonCost(
    const std::vector<Point> & polygon,
    unsigned char cost_value);

  bool getMapRegionOccupiedByPolygon(
    const std::vector<Point> & polygon,
    std::vector<MapLocation> & polygon_map_region);

  void setMapRegionOccupiedByPolygon(
    const std::vector<MapLocation> & polygon_map_region,
    unsigned char new_cost_value);

  void restoreMapRegionOccupiedByPolygon(
    const std::vector<MapLocation> & polygon_map_region);

  void polygonOutlineCells(
    const std::vector<MapLocation> & polygon,
    std::vector<MapLocation> & polygon_cells);

  void convexFillCells(
    const std::vector<MapLocation> & polygon,
    std::vector<MapLocation> & polygon_cells);

  virtual void updateOrigin(double new_origin_x, double new_origin_y);

  bool saveMap(std::string file_name);

  void resizeMap(
    unsigned int size_x, unsigned int size_y, double resolution, double origin_x,
    double origin_y);

  void resetMap(unsigned int x0, unsigned int y0, unsigned int xn, unsigned int yn);

  void resetMapToValue(
    unsigned int x0, unsigned int y0, unsigned int xn, unsigned int yn, unsigned char value);

  unsigned int cellDistance(double world_dist);

  typedef std::recursive_mutex mutex_t;
  mutex_t * getMutex()
  {
    return access_;
  }

protected:
  template<typename data_type>
  void copyMapRegion(
    data_type * source_map, unsigned int sm_lower_left_x,
    unsigned int sm_lower_left_y,
    unsigned int sm_size_x, data_type * dest_map, unsigned int dm_lower_left_x,
    unsigned int dm_lower_left_y, unsigned int dm_size_x, unsigned int region_size_x,
    unsigned int region_size_y)
  {
    data_type * sm_index = source_map + (sm_lower_left_y * sm_size_x + sm_lower_left_x);
    data_type * dm_index = dest_map + (dm_lower_left_y * dm_size_x + dm_lower_left_x);

    for (unsigned int i = 0; i < region_size_y; ++i) {
      memcpy(dm_index, sm_index, region_size_x * sizeof(data_type));
      sm_index += sm_size_x;
      dm_index += dm_size_x;
    }
  }

  virtual void deleteMaps();

  virtual void resetMaps();

  virtual void initMaps(unsigned int size_x, unsigned int size_y);

private:
  mutex_t * access_;

protected:
  unsigned int size_x_;
  unsigned int size_y_;
  double resolution_;
  double origin_x_;
  double origin_y_;
  unsigned char * costmap_;
  unsigned char default_value_;

  class MarkCell
  {
  public:
    MarkCell(unsigned char * costmap, unsigned char value)
    : costmap_(costmap), value_(value)
    {
    }
    inline void operator()(unsigned int offset)
    {
      costmap_[offset] = value_;
    }

  private:
    unsigned char * costmap_;
    unsigned char value_;
  };

  class PolygonOutlineCells
  {
  public:
    PolygonOutlineCells(
      const Costmap2D & costmap, const unsigned char * /*char_map*/,
      std::vector<MapLocation> & cells)
    : costmap_(costmap), cells_(cells)
    {
    }

    inline void operator()(unsigned int offset)
    {
      MapLocation loc;
      costmap_.indexToCells(offset, loc.x, loc.y);
      loc.cost = costmap_.getCost(loc.x, loc.y);
      cells_.push_back(loc);
    }

  private:
    const Costmap2D & costmap_;
    std::vector<MapLocation> & cells_;
  };
};

}  // namespace costmap_2d

#endif  // COSTMAP_2D__COSTMAP_2D_HPP_
