// Copyright (c) 2021 RoboTech Vision
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

#ifndef ESDF_CORE__ESDF_HPP_
#define ESDF_CORE__ESDF_HPP_

#include <array>
#include <limits>
#include <queue>
#include <thread>
#include <vector>

#include <distance_transform/distance_transform.hpp>
#include "esdf_core/costmap2d.hpp"
#include "esdf_core/exceptions.hpp"

namespace esdf_core
{

enum class ESDFAlgorithm
{
  Exact,
  Approximate,
};

/**
 * @class esdf_core::ESDF
 * @brief 2D Euclidean Signed Distance Field computation.
 *
 * Given a Costmap2D, produces a flat double array (row-major, length size_x * size_y)
 * where each entry is the signed distance to the nearest obstacle, in meters:
 *   - positive values: outside obstacles, distance to nearest obstacle surface
 *   - negative values: inside obstacles, distance to the nearest free cell
 *   - +infinity: very far from any obstacle
 *   - 0.0: on the obstacle boundary
 *
 * Two algorithms are provided:
 *   - Exact: uses Felipe Barriga's 2D L2 distance transform (multi-threaded)
 *   - Approximate: Dijkstra-style 8-neighborhood sweep (much slower, useful for
 *                  debugging or for maps where external DT is unavailable)
 */
class ESDF
{
public:
  /**
   * @brief Compute the ESDF for the given costmap, dispatching on `algorithm`.
   */
  static std::vector<double> ComputeESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost,
    ESDFAlgorithm algorithm = ESDFAlgorithm::Exact)
  {
    if (costmap == nullptr) {
      throw InvalidCostmap("ESDF::ComputeESDF received a null costmap");
    }
    if (algorithm == ESDFAlgorithm::Approximate) {
      return ComputeApproximateESDF(costmap, lethal_cost);
    }

    return ComputeExactESDF(costmap, lethal_cost);
  }

  /**
   * @brief Compute the exact ESDF using the L2 distance transform.
   */
  static std::vector<double> ComputeExactESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost)
  {
    const size_t size_x = static_cast<size_t>(costmap->getSizeInCellsX());
    const size_t size_y = static_cast<size_t>(costmap->getSizeInCellsY());
    std::vector<double> outside_esdf = ComputeExactUnsignedESDF(costmap, lethal_cost, true);
    std::vector<double> inside_esdf = ComputeExactUnsignedESDF(costmap, lethal_cost, false);
    std::vector<double> signed_esdf = outside_esdf;

    for (size_t my = 0; my < size_y; ++my) {
      for (size_t mx = 0; mx < size_x; ++mx) {
        const size_t index = toIndex(mx, my, size_x);
        if (costmap->getCost(static_cast<unsigned int>(mx), static_cast<unsigned int>(my)) >= lethal_cost) {
          signed_esdf[index] = -inside_esdf[index];
        }
      }
    }

    return signed_esdf;
  }

  /**
   * @brief Compute an approximate ESDF using Dijkstra on the 8-neighborhood.
   */
  static std::vector<double> ComputeApproximateESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost)
  {
    const size_t size_x = static_cast<size_t>(costmap->getSizeInCellsX());
    const size_t size_y = static_cast<size_t>(costmap->getSizeInCellsY());
    std::vector<double> outside_esdf = ComputeApproximateUnsignedESDF(costmap, lethal_cost, true);
    std::vector<double> inside_esdf = ComputeApproximateUnsignedESDF(costmap, lethal_cost, false);
    std::vector<double> signed_esdf = outside_esdf;

    for (size_t my = 0; my < size_y; ++my) {
      for (size_t mx = 0; mx < size_x; ++mx) {
        const size_t index = toIndex(mx, my, size_x);
        if (costmap->getCost(static_cast<unsigned int>(mx), static_cast<unsigned int>(my)) >= lethal_cost) {
          signed_esdf[index] = -inside_esdf[index];
        }
      }
    }

    return signed_esdf;
  }

private:
  static std::vector<double> ComputeExactUnsignedESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost,
    bool treat_obstacles_as_zero)
  {
    const size_t size_x = static_cast<size_t>(costmap->getSizeInCellsX());
    const size_t size_y = static_cast<size_t>(costmap->getSizeInCellsY());
    const size_t cell_count = size_x * size_y;

    dope::Index2 size({static_cast<dope::SizeType>(size_y),
                       static_cast<dope::SizeType>(size_x)});
    dope::Grid<float, 2> f(size);

    for (dope::SizeType my = 0; my < size[0]; ++my) {
      for (dope::SizeType mx = 0; mx < size[1]; ++mx) {
        const bool is_obstacle =
          costmap->getCost(static_cast<int>(mx), static_cast<int>(my)) >= lethal_cost;
        const bool is_zero_seed = treat_obstacles_as_zero ? is_obstacle : !is_obstacle;
        f[my][mx] = is_zero_seed ? 0.0f : std::numeric_limits<float>::max();
      }
    }

    dt::DistanceTransform::distanceTransformL2(f, f, false, std::thread::hardware_concurrency());

    std::vector<double> esdf(cell_count);
    const double resolution = costmap->getResolution();

    for (dope::SizeType my = 0; my < size[0]; ++my) {
      for (dope::SizeType mx = 0; mx < size[1]; ++mx) {
        const size_t index = static_cast<size_t>(my) * size_x + static_cast<size_t>(mx);
        esdf[index] = f[my][mx] * resolution;
      }
    }

    return esdf;
  }

  static std::vector<double> ComputeApproximateUnsignedESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost,
    bool treat_obstacles_as_zero)
  {
    const size_t size_x = static_cast<size_t>(costmap->getSizeInCellsX());
    const size_t size_y = static_cast<size_t>(costmap->getSizeInCellsY());
    const size_t cell_count = size_x * size_y;
    std::vector<double> esdf(cell_count, std::numeric_limits<double>::infinity());

    struct DistanceItem
    {
      double distance;
      int index;

      bool operator>(const DistanceItem & other) const
      {
        return distance > other.distance;
      }
    };

    std::priority_queue<DistanceItem, std::vector<DistanceItem>, std::greater<DistanceItem>> queue;
    for (size_t my = 0; my < size_y; ++my) {
      for (size_t mx = 0; mx < size_x; ++mx) {
        const size_t index = toIndex(mx, my, size_x);
        const bool is_obstacle = costmap->getCost(static_cast<unsigned int>(mx), static_cast<unsigned int>(my)) >= lethal_cost;
        const bool is_zero_seed = treat_obstacles_as_zero ? is_obstacle : !is_obstacle;
        if (is_zero_seed) {
          esdf[index] = 0.0;
          queue.push({0.0, static_cast<int>(index)});
        }
      }
    }

    while (!queue.empty()) {
      const auto current = queue.top();
      queue.pop();
      if (current.distance > esdf[current.index]) {
        continue;
      }

      const int cx = current.index % static_cast<int>(size_x);
      const int cy = current.index / static_cast<int>(size_x);
      for (const auto & neighbor : kNeighbors) {
        const int nx = cx + neighbor.dx;
        const int ny = cy + neighbor.dy;
        if (!inBounds(nx, ny, static_cast<int>(size_x), static_cast<int>(size_y))) {
          continue;
        }

        const size_t next_index = toIndex(static_cast<size_t>(nx), static_cast<size_t>(ny), size_x);
        const double candidate = current.distance + neighbor.distance * costmap->getResolution();
        if (candidate < esdf[next_index]) {
          esdf[next_index] = candidate;
          queue.push({candidate, static_cast<int>(next_index)});
        }
      }
    }

    return esdf;
  }

  struct NeighborOffset
  {
    int dx;
    int dy;
    double distance;
  };

  static constexpr std::array<NeighborOffset, 8> kNeighbors = {{
    {1, 0, 1.0},
    {-1, 0, 1.0},
    {0, 1, 1.0},
    {0, -1, 1.0},
    {1, 1, 1.4142135623730951},
    {1, -1, 1.4142135623730951},
    {-1, 1, 1.4142135623730951},
    {-1, -1, 1.4142135623730951},
  }};

  static bool inBounds(int mx, int my, int size_x, int size_y)
  {
    return mx >= 0 && my >= 0 && mx < size_x && my < size_y;
  }

  static size_t toIndex(size_t mx, size_t my, size_t size_x)
  {
    return my * size_x + mx;
  }
};

}  // namespace esdf_core

#endif  // ESDF_CORE__ESDF_HPP_
