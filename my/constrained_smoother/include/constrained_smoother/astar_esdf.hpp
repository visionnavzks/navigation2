#ifndef CONSTRAINED_SMOOTHER__ASTAR_ESDF_HPP_
#define CONSTRAINED_SMOOTHER__ASTAR_ESDF_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "constrained_smoother/costmap2d.hpp"

namespace constrained_smoother
{

struct AStarPlannerParams
{
  unsigned char lethal_cost{Costmap2D::LETHAL_OBSTACLE};
  double safe_distance{0.5};
  double cost_penalty_weight{1.0};
};

class AStarPlanner
{
public:
  static std::vector<double> ComputeESDF(
    const Costmap2D * costmap,
    unsigned char lethal_cost)
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
        if (costmap->getCost(mx, my) >= lethal_cost) {
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

  static double EvaluatePenalty(
    double distance,
    double safe_distance)
  {
    if (!std::isfinite(distance) || distance >= safe_distance) {
      return 0.0;
    }

    const double clamped_safe_distance = std::max(safe_distance, 1e-6);
    const double normalized_gap = (clamped_safe_distance - distance) / clamped_safe_distance;
    return normalized_gap * normalized_gap;
  }

  std::vector<Eigen::Vector2d> plan(
    const Costmap2D * costmap,
    double start_wx, double start_wy,
    double goal_wx, double goal_wy,
    const AStarPlannerParams & params)
  {
    if (costmap == nullptr) {
      throw std::runtime_error("AStarPlanner requires a valid costmap");
    }

    const int size_x = static_cast<int>(costmap->getSizeInCellsX());
    const int size_y = static_cast<int>(costmap->getSizeInCellsY());
    if (size_x <= 0 || size_y <= 0) {
      return {};
    }

    esdf_ = ComputeESDF(costmap, params.lethal_cost);

    const auto start = worldToGrid(costmap, start_wx, start_wy);
    const auto goal = worldToGrid(costmap, goal_wx, goal_wy);
    if (!inBounds(start.first, start.second, size_x, size_y) ||
      !inBounds(goal.first, goal.second, size_x, size_y))
    {
      return {};
    }

    if (!isTraversable(costmap, start.first, start.second, params) ||
      !isTraversable(costmap, goal.first, goal.second, params))
    {
      return {};
    }

    const int cell_count = size_x * size_y;
    const int start_index = toIndex(start.first, start.second, size_x);
    const int goal_index = toIndex(goal.first, goal.second, size_x);

    std::vector<double> g_score(cell_count, std::numeric_limits<double>::infinity());
    std::vector<int> came_from(cell_count, -1);

    struct QueueItem
    {
      double f_score;
      double g_score;
      int index;

      bool operator>(const QueueItem & other) const
      {
        return f_score > other.f_score;
      }
    };

    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> open_set;
    g_score[start_index] = 0.0;
    open_set.push({heuristic(start.first, start.second, goal.first, goal.second) *
      costmap->getResolution(), 0.0, start_index});

    while (!open_set.empty()) {
      const auto current = open_set.top();
      open_set.pop();

      if (current.g_score > g_score[current.index]) {
        continue;
      }

      if (current.index == goal_index) {
        return reconstructPath(costmap, came_from, goal_index, start_index);
      }

      const int cx = current.index % size_x;
      const int cy = current.index / size_x;
      for (const auto & neighbor : kNeighbors) {
        const int nx = cx + neighbor.dx;
        const int ny = cy + neighbor.dy;
        if (!inBounds(nx, ny, size_x, size_y) || !isTraversable(costmap, nx, ny, params)) {
          continue;
        }

        const int next_index = toIndex(nx, ny, size_x);
        const double step_cost = neighbor.distance * costmap->getResolution();
        const double penalty = params.cost_penalty_weight *
          EvaluatePenalty(
          esdf_[next_index],
          params.safe_distance);
        const double tentative_g = current.g_score + step_cost + penalty * costmap->getResolution();

        if (tentative_g < g_score[next_index]) {
          g_score[next_index] = tentative_g;
          came_from[next_index] = current.index;
          const double f_score = tentative_g +
            heuristic(nx, ny, goal.first, goal.second) * costmap->getResolution();
          open_set.push({f_score, tentative_g, next_index});
        }
      }
    }

    return {};
  }

  const std::vector<double> & getESDF() const
  {
    return esdf_;
  }

private:
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

  static std::pair<int, int> worldToGrid(
    const Costmap2D * costmap,
    double wx, double wy)
  {
    const int mx = static_cast<int>((wx - costmap->getOriginX()) / costmap->getResolution());
    const int my = static_cast<int>((wy - costmap->getOriginY()) / costmap->getResolution());
    return {mx, my};
  }

  static Eigen::Vector2d gridToWorld(const Costmap2D * costmap, int mx, int my)
  {
    return {
      costmap->getOriginX() + (static_cast<double>(mx) + 0.5) * costmap->getResolution(),
      costmap->getOriginY() + (static_cast<double>(my) + 0.5) * costmap->getResolution()};
  }

  static bool isTraversable(
    const Costmap2D * costmap,
    int mx, int my,
    const AStarPlannerParams & params)
  {
    return costmap->getCost(mx, my) < params.lethal_cost;
  }

  static double heuristic(int ax, int ay, int bx, int by)
  {
    const double dx = std::abs(ax - bx);
    const double dy = std::abs(ay - by);
    return std::max(dx, dy) + (1.4142135623730951 - 1.0) * std::min(dx, dy);
  }

  std::vector<Eigen::Vector2d> reconstructPath(
    const Costmap2D * costmap,
    const std::vector<int> & came_from,
    int goal_index,
    int start_index) const
  {
    std::vector<Eigen::Vector2d> path;
    const int size_x = static_cast<int>(costmap->getSizeInCellsX());
    int current = goal_index;
    while (current >= 0) {
      const int mx = current % size_x;
      const int my = current / size_x;
      path.push_back(gridToWorld(costmap, mx, my));
      if (current == start_index) {
        break;
      }
      current = came_from[current];
    }

    if (path.empty() || current < 0) {
      return {};
    }

    std::reverse(path.begin(), path.end());
    return path;
  }

  std::vector<double> esdf_;
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__ASTAR_ESDF_HPP_