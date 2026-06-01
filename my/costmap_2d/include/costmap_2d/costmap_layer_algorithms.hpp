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
#ifndef COSTMAP_2D__COSTMAP_LAYER_ALGORITHMS_HPP_
#define COSTMAP_2D__COSTMAP_LAYER_ALGORITHMS_HPP_

#include "costmap_2d/costmap_2d.hpp"
#include "costmap_2d/cost_values.hpp"

namespace costmap_2d
{

/**
 * @brief Updates master_grid with layer's values using true overwrite.
 * Every value from this layer is written into the master grid.
 */
inline void updateWithTrueOverwrite(
  unsigned char * layer_data,
  Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  unsigned char * master = master_grid.getCharMap();
  unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; j++) {
    unsigned int it = span * j + min_i;
    for (int i = min_i; i < max_i; i++) {
      master[it] = layer_data[it];
      it++;
    }
  }
}

/**
 * @brief Updates master_grid with layer's values.
 * Every valid value (not NO_INFORMATION) from this layer is written into the master grid.
 */
inline void updateWithOverwrite(
  unsigned char * layer_data,
  Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  unsigned char * master = master_grid.getCharMap();
  unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; j++) {
    unsigned int it = span * j + min_i;
    for (int i = min_i; i < max_i; i++) {
      if (layer_data[it] != NO_INFORMATION) {
        master[it] = layer_data[it];
      }
      it++;
    }
  }
}

/**
 * @brief Updates master_grid with layer's values using max combination.
 * Sets the new value to the maximum of master_grid's value and this layer's value.
 * If the master value is NO_INFORMATION, it is overwritten.
 * If the layer's value is NO_INFORMATION, the master value does not change.
 */
inline void updateWithMax(
  unsigned char * layer_data,
  Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  unsigned char * master_array = master_grid.getCharMap();
  unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; j++) {
    unsigned int it = j * span + min_i;
    for (int i = min_i; i < max_i; i++) {
      if (layer_data[it] == NO_INFORMATION) {
        it++;
        continue;
      }

      unsigned char old_cost = master_array[it];
      if (old_cost == NO_INFORMATION || old_cost < layer_data[it]) {
        master_array[it] = layer_data[it];
      }
      it++;
    }
  }
}

/**
 * @brief Updates master_grid with layer's values using max without unknown overwrite.
 * If the master value is NO_INFORMATION, it is NOT overwritten.
 * If the layer's value is NO_INFORMATION, the master value does not change.
 */
inline void updateWithMaxWithoutUnknownOverwrite(
  unsigned char * layer_data,
  Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  unsigned char * master_array = master_grid.getCharMap();
  unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; j++) {
    unsigned int it = j * span + min_i;
    for (int i = min_i; i < max_i; i++) {
      if (layer_data[it] == NO_INFORMATION) {
        it++;
        continue;
      }

      unsigned char old_cost = master_array[it];
      if (old_cost != NO_INFORMATION && old_cost < layer_data[it]) {
        master_array[it] = layer_data[it];
      }
      it++;
    }
  }
}

/**
 * @brief Updates master_grid with layer's values using addition combination.
 * Sets the new value to the sum of master grid's value and this layer's value.
 */
inline void updateWithAddition(
  unsigned char * layer_data,
  Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  unsigned char * master_array = master_grid.getCharMap();
  unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; j++) {
    unsigned int it = j * span + min_i;
    for (int i = min_i; i < max_i; i++) {
      if (layer_data[it] == NO_INFORMATION) {
        it++;
        continue;
      }

      unsigned char old_cost = master_array[it];
      if (old_cost == NO_INFORMATION) {
        master_array[it] = layer_data[it];
      } else {
        int sum = old_cost + layer_data[it];
        if (sum >= INSCRIBED_INFLATED_OBSTACLE) {
          master_array[it] = INSCRIBED_INFLATED_OBSTACLE - 1;
        } else {
          master_array[it] = sum;
        }
      }
      it++;
    }
  }
}

}  // namespace costmap_2d

#endif  // COSTMAP_2D__COSTMAP_LAYER_ALGORITHMS_HPP_
