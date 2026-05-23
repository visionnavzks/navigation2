#ifndef CONSTRAINED_SMOOTHER__ESDF_HPP_
#define CONSTRAINED_SMOOTHER__ESDF_HPP_

#include <array>
#include <limits>
#include <queue>
#include <thread>
#include <vector>

#include <distance_transform/distance_transform.hpp>
#include "constrained_smoother/costmap2d.hpp"

namespace constrained_smoother
{

enum class ESDFAlgorithm
{
  Exact,
  Approximate,
};

class ESDF
{
public:
  static std::vector<double> ComputeESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost,
    ESDFAlgorithm algorithm = ESDFAlgorithm::Exact)
  {
    if (algorithm == ESDFAlgorithm::Approximate) {
      return ComputeApproximateESDF(costmap, lethal_cost);
    }

    return ComputeExactESDF(costmap, lethal_cost);
  }

  static std::vector<double> ComputeExactESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost)
  {
    const int size_x = static_cast<int>(costmap->getSizeInCellsX());
    const int size_y = static_cast<int>(costmap->getSizeInCellsY());
    std::vector<double> outside_esdf = ComputeExactUnsignedESDF(costmap, lethal_cost, true);
    std::vector<double> inside_esdf = ComputeExactUnsignedESDF(costmap, lethal_cost, false);
    std::vector<double> signed_esdf = outside_esdf;

    for (int my = 0; my < size_y; ++my) {
      for (int mx = 0; mx < size_x; ++mx) {
        const int index = toIndex(mx, my, size_x);
        if (costmap->getCost(mx, my) >= lethal_cost) {
          signed_esdf[index] = -inside_esdf[index];
        }
      }
    }

    return signed_esdf;
  }

  static std::vector<double> ComputeApproximateESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost)
  {
    const int size_x = static_cast<int>(costmap->getSizeInCellsX());
    const int size_y = static_cast<int>(costmap->getSizeInCellsY());
    std::vector<double> outside_esdf = ComputeApproximateUnsignedESDF(costmap, lethal_cost, true);
    std::vector<double> inside_esdf = ComputeApproximateUnsignedESDF(costmap, lethal_cost, false);
    std::vector<double> signed_esdf = outside_esdf;

    for (int my = 0; my < size_y; ++my) {
      for (int mx = 0; mx < size_x; ++mx) {
        const int index = toIndex(mx, my, size_x);
        if (costmap->getCost(mx, my) >= lethal_cost) {
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
    const int size_x = static_cast<int>(costmap->getSizeInCellsX());
    const int size_y = static_cast<int>(costmap->getSizeInCellsY());
    const int cell_count = size_x * size_y;

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
        const int index = static_cast<int>(my) * size_x + static_cast<int>(mx);
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
    const int size_x = static_cast<int>(costmap->getSizeInCellsX());
    const int size_y = static_cast<int>(costmap->getSizeInCellsY());
    const int cell_count = size_x * size_y;
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
    for (int my = 0; my < size_y; ++my) {
      for (int mx = 0; mx < size_x; ++mx) {
        const int index = toIndex(mx, my, size_x);
        const bool is_obstacle = costmap->getCost(mx, my) >= lethal_cost;
        const bool is_zero_seed = treat_obstacles_as_zero ? is_obstacle : !is_obstacle;
        if (is_zero_seed) {
          esdf[index] = 0.0;
          queue.push({0.0, index});
        }
      }
    }

    while (!queue.empty()) {
      const auto current = queue.top();
      queue.pop();
      if (current.distance > esdf[current.index]) {
        continue;
      }

      const int cx = current.index % size_x;
      const int cy = current.index / size_x;
      for (const auto & neighbor : kNeighbors) {
        const int nx = cx + neighbor.dx;
        const int ny = cy + neighbor.dy;
        if (!inBounds(nx, ny, size_x, size_y)) {
          continue;
        }

        const int next_index = toIndex(nx, ny, size_x);
        const double candidate = current.distance + neighbor.distance * costmap->getResolution();
        if (candidate < esdf[next_index]) {
          esdf[next_index] = candidate;
          queue.push({candidate, next_index});
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

  static int toIndex(int mx, int my, int size_x)
  {
    return my * size_x + mx;
  }
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__ESDF_HPP_