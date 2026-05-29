// Copyright (c) 2008, 2013, Willow Garage, Inc.
// All rights reserved.
//
// Software License Agreement (BSD License 2.0)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Willow Garage, Inc. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef MY_COSTMAP_2D__LAYER_INTERFACE_HPP_
#define MY_COSTMAP_2D__LAYER_INTERFACE_HPP_

#include <atomic>
#include <string>
#include <vector>
#include <memory>

#include "my_costmap_2d/costmap_2d.hpp"
#include "my_costmap_2d/point.hpp"

namespace my_costmap_2d
{

class LayeredCostmap;

/**
 * @class LayerInterface
 * @brief Pure virtual interface for costmap layers, no ROS dependencies.
 */
class LayerInterface
{
public:
  LayerInterface()
  : layered_costmap_(nullptr), current_(false), enabled_(true) {}

  virtual ~LayerInterface() {}

  virtual void reset() = 0;

  virtual bool isClearable() = 0;

  virtual void updateBounds(
    double robot_x, double robot_y, double robot_yaw, double * min_x,
    double * min_y,
    double * max_x,
    double * max_y) = 0;

  virtual void updateCosts(
    Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) = 0;

  virtual void matchSize() {}

  virtual void onFootprintChanged() {}

  std::string getName() const {return name_;}

  bool isCurrent() const {return current_;}
  void setCurrent(bool current) {current_ = current;}

  bool isEnabled() const {return enabled_;}
  void setEnabled(bool enabled) {enabled_ = enabled;}

  const std::vector<Point> & getFootprint() const;

  void setParent(LayeredCostmap * parent) {layered_costmap_ = parent;}
  void setName(const std::string & name) {name_ = name;}

protected:
  virtual void onInitialize() {}

  LayeredCostmap * layered_costmap_;
  std::string name_;
  std::atomic_bool current_;
  bool enabled_;
};

}  // namespace my_costmap_2d

#endif  // MY_COSTMAP_2D__LAYER_INTERFACE_HPP_
