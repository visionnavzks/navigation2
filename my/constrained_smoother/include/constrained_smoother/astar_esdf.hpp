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
  double point_radius{0.0};
  bool use_rectangular_footprint{false};
  double rectangular_length{0.0};
  double rectangular_width{0.0};
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

  bool isTraversable(
    const Costmap2D * costmap,
    int mx, int my,
    const AStarPlannerParams & params) const
  {
    if (costmap->getCost(mx, my) >= params.lethal_cost) {
      return false;
    }

    if (params.use_rectangular_footprint) {
      return isAxisAlignedRectangleTraversable(costmap, mx, my, params);
    }

    return isPointRobotTraversable(costmap, mx, my, params);
  }

  bool isPointRobotTraversable(
    const Costmap2D * costmap,
    int mx, int my,
    const AStarPlannerParams & params) const
  {
    const double point_radius = std::max(params.point_radius, 0.0);
    if (point_radius <= 1e-9) {
      return true;
    }

    const auto center = gridToWorld(costmap, mx, my);
    if (!isFootprintInsideMapBounds(costmap, center.x(), center.y(), point_radius, point_radius)) {
      return false;
    }

    const int index = toIndex(mx, my, static_cast<int>(costmap->getSizeInCellsX()));
    return index >= 0 && index < static_cast<int>(esdf_.size()) && esdf_[index] >= point_radius;
  }

  bool isAxisAlignedRectangleTraversable(
    const Costmap2D * costmap,
    int mx, int my,
    const AStarPlannerParams & params) const
  {
    const double half_length = std::max(params.rectangular_length, 0.0) * 0.5;
    const double half_width = std::max(params.rectangular_width, 0.0) * 0.5;
    if (half_length <= 1e-9 && half_width <= 1e-9) {
      return true;
    }

    const auto center = gridToWorld(costmap, mx, my);
    if (!isFootprintInsideMapBounds(costmap, center.x(), center.y(), half_length, half_width)) {
      return false;
    }

    const double resolution = costmap->getResolution();
    const double origin_x = costmap->getOriginX();
    const double origin_y = costmap->getOriginY();
    const int min_mx = static_cast<int>(std::floor((center.x() - half_length - origin_x) / resolution));
    const int max_mx = static_cast<int>(std::ceil((center.x() + half_length - origin_x) / resolution)) - 1;
    const int min_my = static_cast<int>(std::floor((center.y() - half_width - origin_y) / resolution));
    const int max_my = static_cast<int>(std::ceil((center.y() + half_width - origin_y) / resolution)) - 1;

    const int size_x = static_cast<int>(costmap->getSizeInCellsX());
    const int size_y = static_cast<int>(costmap->getSizeInCellsY());
    if (min_mx < 0 || min_my < 0 || max_mx >= size_x || max_my >= size_y) {
      return false;
    }

    for (int check_my = min_my; check_my <= max_my; ++check_my) {
      for (int check_mx = min_mx; check_mx <= max_mx; ++check_mx) {
        if (costmap->getCost(check_mx, check_my) >= params.lethal_cost) {
          return false;
        }
      }
    }

    return true;
  }

  static bool isFootprintInsideMapBounds(
    const Costmap2D * costmap,
    double center_wx,
    double center_wy,
    double half_extent_x,
    double half_extent_y)
  {
    const double min_wx = center_wx - half_extent_x;
    const double max_wx = center_wx + half_extent_x;
    const double min_wy = center_wy - half_extent_y;
    const double max_wy = center_wy + half_extent_y;
    const double map_min_x = costmap->getOriginX();
    const double map_min_y = costmap->getOriginY();
    const double map_max_x = map_min_x + costmap->getSizeInCellsX() * costmap->getResolution();
    const double map_max_y = map_min_y + costmap->getSizeInCellsY() * costmap->getResolution();

    return min_wx >= map_min_x && min_wy >= map_min_y &&
      max_wx <= map_max_x && max_wy <= map_max_y;
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