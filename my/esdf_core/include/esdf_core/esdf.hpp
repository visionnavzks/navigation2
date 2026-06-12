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

#include <algorithm>
#include <array>
#include <limits>
#include <queue>
#include <thread>
#include <utility>
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
 * where each entry is the signed distance to the nearest obstacle boundary, in meters:
 *   - positive values: outside obstacles, distance to the nearest obstacle cell
 *   - negative values: inside obstacles, distance to the nearest free cell
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
    return algorithm == ESDFAlgorithm::Approximate ?
           ComputeApproximateESDF(costmap, lethal_cost) :
           ComputeExactESDF(costmap, lethal_cost);
  }

  /**
   * @brief Compute the exact ESDF using the L2 distance transform.
   */
  static std::vector<double> ComputeExactESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost)
  {
    return CombineSigned(
      costmap, lethal_cost,
      ComputeExactUnsignedESDF(costmap, lethal_cost, true),
      ComputeExactUnsignedESDF(costmap, lethal_cost, false));
  }

  /**
   * @brief Compute an approximate ESDF using Dijkstra on the 8-neighborhood.
   */
  static std::vector<double> ComputeApproximateESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost)
  {
    return CombineSigned(
      costmap, lethal_cost,
      ComputeApproximateUnsignedESDF(costmap, lethal_cost, true),
      ComputeApproximateUnsignedESDF(costmap, lethal_cost, false));
  }

private:
  static bool isObstacle(
    const Costmap2D * costmap, size_t mx, size_t my, unsigned char lethal_cost)
  {
    return costmap->getCost(
      static_cast<unsigned int>(mx), static_cast<unsigned int>(my)) >= lethal_cost;
  }

  /**
   * @brief Merge the outside/inside unsigned fields into one signed field:
   *        distances inside obstacles are negated.
   */
  static std::vector<double> CombineSigned(
    const Costmap2D * costmap,
    unsigned char lethal_cost,
    std::vector<double> outside,
    const std::vector<double> & inside)
  {
    const size_t size_x = costmap->getSizeInCellsX();
    const size_t size_y = costmap->getSizeInCellsY();

    for (size_t my = 0; my < size_y; ++my) {
      for (size_t mx = 0; mx < size_x; ++mx) {
        if (isObstacle(costmap, mx, my, lethal_cost)) {
          const size_t index = toIndex(mx, my, size_x);
          outside[index] = -inside[index];
        }
      }
    }

    return outside;
  }

  /**
   * @brief Unsigned distance field via the L2 distance transform.
   * @param seed_obstacles true: distance-to-obstacle; false: distance-to-free-space.
   */
  static std::vector<double> ComputeExactUnsignedESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost,
    bool seed_obstacles)
  {
    const size_t size_x = costmap->getSizeInCellsX();
    const size_t size_y = costmap->getSizeInCellsY();

    dope::Index2 size({static_cast<dope::SizeType>(size_y),
                       static_cast<dope::SizeType>(size_x)});
    dope::Grid<float, 2> f(size);

    for (size_t my = 0; my < size_y; ++my) {
      for (size_t mx = 0; mx < size_x; ++mx) {
        const bool is_seed = isObstacle(costmap, mx, my, lethal_cost) == seed_obstacles;
        f[my][mx] = is_seed ? 0.0f : std::numeric_limits<float>::max();
      }
    }

    // hardware_concurrency() may return 0; the DT library does `range % nThreads`,
    // so passing 0 would be a division by zero.
    const std::size_t num_threads =
      std::max<std::size_t>(1, std::thread::hardware_concurrency());
    dt::DistanceTransform::distanceTransformL2(f, f, false, num_threads);

    std::vector<double> esdf(size_x * size_y);
    const double resolution = costmap->getResolution();

    for (size_t my = 0; my < size_y; ++my) {
      for (size_t mx = 0; mx < size_x; ++mx) {
        esdf[toIndex(mx, my, size_x)] = f[my][mx] * resolution;
      }
    }

    return esdf;
  }

  /**
   * @brief Unsigned distance field via Dijkstra on the 8-neighborhood.
   * @param seed_obstacles true: distance-to-obstacle; false: distance-to-free-space.
   */
  static std::vector<double> ComputeApproximateUnsignedESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost,
    bool seed_obstacles)
  {
    const size_t size_x = costmap->getSizeInCellsX();
    const size_t size_y = costmap->getSizeInCellsY();
    const double resolution = costmap->getResolution();
    std::vector<double> esdf(size_x * size_y, std::numeric_limits<double>::infinity());

    using QueueItem = std::pair<double, size_t>;  // (distance, cell index)
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;

    for (size_t my = 0; my < size_y; ++my) {
      for (size_t mx = 0; mx < size_x; ++mx) {
        if (isObstacle(costmap, mx, my, lethal_cost) == seed_obstacles) {
          const size_t index = toIndex(mx, my, size_x);
          esdf[index] = 0.0;
          queue.emplace(0.0, index);
        }
      }
    }

    while (!queue.empty()) {
      const auto [distance, index] = queue.top();
      queue.pop();
      if (distance > esdf[index]) {
        continue;  // stale entry
      }

      const int cx = static_cast<int>(index % size_x);
      const int cy = static_cast<int>(index / size_x);
      for (const auto & neighbor : kNeighbors) {
        const int nx = cx + neighbor.dx;
        const int ny = cy + neighbor.dy;
        if (nx < 0 || ny < 0 ||
          nx >= static_cast<int>(size_x) || ny >= static_cast<int>(size_y))
        {
          continue;
        }

        const size_t next = toIndex(static_cast<size_t>(nx), static_cast<size_t>(ny), size_x);
        const double candidate = distance + neighbor.distance * resolution;
        if (candidate < esdf[next]) {
          esdf[next] = candidate;
          queue.emplace(candidate, next);
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

  static constexpr double kSqrt2 = 1.4142135623730951;
  static constexpr std::array<NeighborOffset, 8> kNeighbors = {{
    {1, 0, 1.0}, {-1, 0, 1.0}, {0, 1, 1.0}, {0, -1, 1.0},
    {1, 1, kSqrt2}, {1, -1, kSqrt2}, {-1, 1, kSqrt2}, {-1, -1, kSqrt2},
  }};

  static size_t toIndex(size_t mx, size_t my, size_t size_x)
  {
    return my * size_x + mx;
  }
};

}  // namespace esdf_core

#endif  // ESDF_CORE__ESDF_HPP_
