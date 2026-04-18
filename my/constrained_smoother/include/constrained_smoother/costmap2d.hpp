// Copyright (c) 2021 RoboTech Vision
// Copyright (c) 2020, Samsung Research America
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

#ifndef CONSTRAINED_SMOOTHER__COSTMAP2D_HPP_
#define CONSTRAINED_SMOOTHER__COSTMAP2D_HPP_

#include <cstring>
#include <vector>

namespace constrained_smoother
{

/**
 * @class constrained_smoother::Costmap2D
 * @brief A lightweight 2D costmap representation, independent of ROS.
 *
 * Stores an occupancy grid as a flat unsigned char array in row-major order.
 * Provides the minimal interface required by the constrained smoother.
 */
class Costmap2D
{
public:
  Costmap2D()
  : size_x_(0), size_y_(0), resolution_(1.0), origin_x_(0.0), origin_y_(0.0), data_(nullptr)
  {
  }

  /**
   * @brief Construct a costmap with given dimensions and resolution.
   * @param size_x Number of cells in x
   * @param size_y Number of cells in y
   * @param resolution Meters per cell
   * @param origin_x World x coordinate of the lower-left corner
   * @param origin_y World y coordinate of the lower-left corner
   */
  Costmap2D(
    unsigned int size_x, unsigned int size_y, double resolution,
    double origin_x, double origin_y)
  : size_x_(size_x), size_y_(size_y), resolution_(resolution),
    origin_x_(origin_x), origin_y_(origin_y)
  {
    data_storage_.resize(size_x_ * size_y_, 0);
    data_ = data_storage_.data();
  }

  /**
   * @brief Construct a costmap wrapping an external data buffer (non-owning).
   * @param size_x Number of cells in x
   * @param size_y Number of cells in y
   * @param resolution Meters per cell
   * @param origin_x World x coordinate of the lower-left corner
   * @param origin_y World y coordinate of the lower-left corner
   * @param data Pointer to externally managed cost data (row-major, size_x * size_y)
   */
  Costmap2D(
    unsigned int size_x, unsigned int size_y, double resolution,
    double origin_x, double origin_y,
    unsigned char * data)
  : size_x_(size_x), size_y_(size_y), resolution_(resolution),
    origin_x_(origin_x), origin_y_(origin_y), data_(data)
  {
  }

  unsigned int getSizeInCellsX() const {return size_x_;}
  unsigned int getSizeInCellsY() const {return size_y_;}
  double getResolution() const {return resolution_;}
  double getOriginX() const {return origin_x_;}
  double getOriginY() const {return origin_y_;}
  unsigned char * getCharMap() const {return data_;}

  unsigned char getCost(unsigned int mx, unsigned int my) const
  {
    return data_[my * size_x_ + mx];
  }

  void setCost(unsigned int mx, unsigned int my, unsigned char cost)
  {
    data_[my * size_x_ + mx] = cost;
  }

  static constexpr unsigned char NO_INFORMATION = 255;
  static constexpr unsigned char LETHAL_OBSTACLE = 254;
  static constexpr unsigned char INSCRIBED_INFLATED_OBSTACLE = 253;
  static constexpr unsigned char FREE_SPACE = 0;

private:
  unsigned int size_x_;
  unsigned int size_y_;
  double resolution_;
  double origin_x_;
  double origin_y_;
  std::vector<unsigned char> data_storage_;  // owned storage (when applicable)
  unsigned char * data_;
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__COSTMAP2D_HPP_
