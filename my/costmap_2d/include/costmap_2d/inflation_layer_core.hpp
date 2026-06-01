// Copyright (c) 2026, Dexory (Tony Najjar)
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

#ifndef COSTMAP_2D__INFLATION_LAYER_CORE_HPP_
#define COSTMAP_2D__INFLATION_LAYER_CORE_HPP_

#include <cmath>
#include <vector>

#include "costmap_2d/cost_values.hpp"
#include "costmap_2d/distance_transform.hpp"

namespace costmap_2d
{

/**
 * @brief Compute cost from distance using exponential decay.
 * @param distance Distance from obstacle in cells
 * @param resolution Costmap resolution in meters/cell
 * @param inscribed_radius Inscribed radius in meters
 * @param cost_scaling_factor Cost scaling factor
 * @return Cost value
 */
inline unsigned char computeInflationCost(
  double distance, double resolution,
  double inscribed_radius, double cost_scaling_factor)
{
  unsigned char cost = 0;
  if (distance == 0) {
    cost = LETHAL_OBSTACLE;
  } else if (distance * resolution <= inscribed_radius) {
    cost = INSCRIBED_INFLATED_OBSTACLE;
  } else {
    double factor =
      exp(-1.0 * cost_scaling_factor * (distance * resolution - inscribed_radius));
    cost = static_cast<unsigned char>((INSCRIBED_INFLATED_OBSTACLE - 1) * factor);
  }
  return cost;
}

/**
 * @brief Apply inflation costs from distance map to costmap.
 * @param master_array Pointer to the costmap data
 * @param distance_map Distance transform result
 * @param min_i Minimum x index of update region
 * @param min_j Minimum y index of update region
 * @param max_i Maximum x index of update region
 * @param max_j Maximum y index of update region
 * @param roi_min_i ROI minimum x offset
 * @param roi_min_j ROI minimum y offset
 * @param size_x Width of the costmap
 * @param resolution Costmap resolution
 * @param inscribed_radius Inscribed radius
 * @param cost_scaling_factor Cost scaling factor
 */
inline void applyInflation(
  unsigned char * master_array,
  const MatrixXfRM & distance_map,
  int min_i, int min_j, int max_i, int max_j,
  int roi_min_i, int roi_min_j,
  unsigned int size_x,
  double resolution, double inscribed_radius, double cost_scaling_factor)
{
  for (int j = min_j; j < max_j; ++j) {
    for (int i = min_i; i < max_i; ++i) {
      unsigned int index = j * size_x + i;
      float distance = distance_map(j - min_j + roi_min_j, i - min_i + roi_min_i);

      if (distance > 0) {
        unsigned char cost = computeInflationCost(
          distance, resolution, inscribed_radius, cost_scaling_factor);
        if (cost > master_array[index]) {
          master_array[index] = cost;
        }
      }
    }
  }
}

}  // namespace costmap_2d

#endif  // COSTMAP_2D__INFLATION_LAYER_CORE_HPP_
