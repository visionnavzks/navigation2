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
#ifndef MY_COSTMAP_2D__LAYERED_COSTMAP_HPP_
#define MY_COSTMAP_2D__LAYERED_COSTMAP_HPP_

#include <memory>
#include <string>
#include <vector>

#include "my_costmap_2d/cost_values.hpp"
#include "my_costmap_2d/layer_interface.hpp"
#include "my_costmap_2d/costmap_2d.hpp"
#include "my_costmap_2d/point.hpp"

namespace my_costmap_2d
{

class LayerInterface;

/**
 * @class LayeredCostmap
 * @brief Instantiates different layer plugins and aggregates them into one score.
 * ROS-free version.
 */
class LayeredCostmap
{
public:
  LayeredCostmap(std::string global_frame, bool rolling_window, bool track_unknown);

  ~LayeredCostmap();

  void updateMap(double robot_x, double robot_y, double robot_yaw);

  std::string getGlobalFrameID() const {return global_frame_;}

  void resizeMap(
    unsigned int size_x, unsigned int size_y, double resolution, double origin_x,
    double origin_y,
    bool size_locked = false);

  void getUpdatedBounds(double & minx, double & miny, double & maxx, double & maxy)
  {
    minx = minx_;
    miny = miny_;
    maxx = maxx_;
    maxy = maxy_;
  }

  bool isCurrent();

  Costmap2D * getCostmap() {return &combined_costmap_;}

  bool isRolling() {return rolling_window_;}

  bool isTrackingUnknown()
  {
    return combined_costmap_.getDefaultValue() == my_costmap_2d::NO_INFORMATION;
  }

  std::vector<std::shared_ptr<LayerInterface>> * getPlugins() {return &plugins_;}

  std::vector<std::shared_ptr<LayerInterface>> * getFilters() {return &filters_;}

  void addPlugin(std::shared_ptr<LayerInterface> plugin);

  void addFilter(std::shared_ptr<LayerInterface> filter);

  bool isSizeLocked() {return size_locked_;}

  void getBounds(unsigned int * x0, unsigned int * xn, unsigned int * y0, unsigned int * yn)
  {
    *x0 = bx0_;
    *xn = bxn_;
    *y0 = by0_;
    *yn = byn_;
  }

  bool isInitialized() {return initialized_;}

  void setFootprint(const std::vector<Point> & footprint_spec);

  const std::vector<Point> & getFootprint() {return *std::atomic_load(&footprint_);}

  double getCircumscribedRadius() {return circumscribed_radius_.load();}

  double getInscribedRadius() {return inscribed_radius_.load();}

  bool isOutofBounds(double robot_x, double robot_y);

private:
  Costmap2D primary_costmap_, combined_costmap_;
  std::string global_frame_;
  bool rolling_window_;

  double minx_, miny_, maxx_, maxy_;
  unsigned int bx0_, bxn_, by0_, byn_;

  std::vector<std::shared_ptr<LayerInterface>> plugins_;
  std::vector<std::shared_ptr<LayerInterface>> filters_;

  bool initialized_;
  bool size_locked_;
  std::atomic<double> circumscribed_radius_, inscribed_radius_;
  std::shared_ptr<std::vector<Point>> footprint_;
};

}  // namespace my_costmap_2d

#endif  // MY_COSTMAP_2D__LAYERED_COSTMAP_HPP_
