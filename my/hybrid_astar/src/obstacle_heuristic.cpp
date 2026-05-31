#include "my/hybrid_astar/obstacle_heuristic.hpp"
#include "my/hybrid_astar/costmap_2d.hpp"

namespace hybrid_astar
{

void ObstacleHeuristic::resetObstacleHeuristic(
  Costmap2D * costmap,
  const unsigned int & start_x, const unsigned int & start_y,
  const unsigned int & goal_x, const unsigned int & goal_y,
  const bool downsample_obstacle_heuristic)
{
  this->costmap = costmap;

  unsigned int size = 0u;
  unsigned int size_x = 0u;
  if (downsample_obstacle_heuristic) {
    size_x = ceil(static_cast<float>(costmap->getSizeInCellsX()) / 2.0f);
    size = size_x *
      ceil(static_cast<float>(costmap->getSizeInCellsY()) / 2.0f);
  } else {
    size_x = costmap->getSizeInCellsX();
    size = size_x * costmap->getSizeInCellsY();
  }

  if (obstacle_heuristic_lookup_table_.size() == size) {
    std::fill(
      obstacle_heuristic_lookup_table_.begin(),
      obstacle_heuristic_lookup_table_.end(), 0.0f);
  } else {
    unsigned int obstacle_size = obstacle_heuristic_lookup_table_.size();
    obstacle_heuristic_lookup_table_.resize(size, 0.0f);
    std::fill_n(
      obstacle_heuristic_lookup_table_.begin(), obstacle_size, 0.0f);
  }

  obstacle_heuristic_queue_.clear();
  obstacle_heuristic_queue_.reserve(size);

  unsigned int goal_index;
  if (downsample_obstacle_heuristic) {
    goal_index = floor(goal_y / 2.0f) * size_x + floor(goal_x / 2.0f);
  } else {
    goal_index = floor(goal_y) * size_x + floor(goal_x);
  }

  obstacle_heuristic_queue_.emplace_back(
    distanceHeuristic2D(goal_index, size_x, start_x, start_y), goal_index);

  obstacle_heuristic_lookup_table_[goal_index] = -0.00001f;
}

float ObstacleHeuristic::getObstacleHeuristic(
  const Coordinates & node_coords,
  const float & cost_penalty,
  const bool use_quadratic_cost_penalty,
  const bool downsample_obstacle_heuristic)
{
  unsigned int size_x = 0u;
  unsigned int size_y = 0u;
  if (downsample_obstacle_heuristic) {
    size_x = ceil(static_cast<float>(costmap->getSizeInCellsX()) / 2.0f);
    size_y = ceil(static_cast<float>(costmap->getSizeInCellsY()) / 2.0f);
  } else {
    size_x = costmap->getSizeInCellsX();
    size_y = costmap->getSizeInCellsY();
  }

  unsigned int start_y, start_x;
  if (downsample_obstacle_heuristic) {
    start_y = floor(node_coords.y / 2.0f);
    start_x = floor(node_coords.x / 2.0f);
  } else {
    start_y = floor(node_coords.y);
    start_x = floor(node_coords.x);
  }

  const unsigned int start_index = start_y * size_x + start_x;
  const float & requested_node_cost = obstacle_heuristic_lookup_table_[start_index];
  if (requested_node_cost > 0.0f) {
    return downsample_obstacle_heuristic ? 2.0f * requested_node_cost : requested_node_cost;
  }

  for (auto & n : obstacle_heuristic_queue_) {
    n.first = -obstacle_heuristic_lookup_table_[n.second] +
      distanceHeuristic2D(n.second, size_x, start_x, start_y);
  }
  std::make_heap(
    obstacle_heuristic_queue_.begin(), obstacle_heuristic_queue_.end(),
    ObstacleHeuristicComparator{});

  const int size_x_int = static_cast<int>(size_x);
  const float sqrt2 = sqrtf(2.0f);
  float c_cost, cost, travel_cost, new_cost, existing_cost;
  unsigned int mx, my;
  unsigned int idx, new_idx = 0;

  const std::vector<int> neighborhood = {1, -1,
    size_x_int, -size_x_int,
    size_x_int + 1, size_x_int - 1,
    -size_x_int + 1, -size_x_int - 1};

  while (!obstacle_heuristic_queue_.empty()) {
    idx = obstacle_heuristic_queue_.front().second;
    std::pop_heap(
      obstacle_heuristic_queue_.begin(), obstacle_heuristic_queue_.end(),
      ObstacleHeuristicComparator{});
    obstacle_heuristic_queue_.pop_back();
    c_cost = obstacle_heuristic_lookup_table_[idx];
    if (c_cost > 0.0f) {
      continue;
    }
    c_cost = -c_cost;
    obstacle_heuristic_lookup_table_[idx] = c_cost;

    for (unsigned int i = 0; i != neighborhood.size(); i++) {
      new_idx = static_cast<unsigned int>(static_cast<int>(idx) + neighborhood[i]);

      if (new_idx < size_x * size_y) {
        if (downsample_obstacle_heuristic) {
          unsigned int y_offset = (new_idx / size_x) * 2;
          unsigned int x_offset = (new_idx - ((new_idx / size_x) * size_x)) * 2;
          cost = costmap->getCost(x_offset, y_offset);
          for (unsigned int k = 0; k < 2u; ++k) {
            unsigned int mxd = x_offset + k;
            if (mxd >= costmap->getSizeInCellsX()) {
              continue;
            }
            for (unsigned int j = 0; j < 2u; ++j) {
              unsigned int myd = y_offset + j;
              if (myd >= costmap->getSizeInCellsY()) {
                continue;
              }
              if (k == 0 && j == 0) {
                continue;
              }
              cost = std::min(cost, static_cast<float>(costmap->getCost(mxd, myd)));
            }
          }
        } else {
          cost = static_cast<float>(costmap->getCost(new_idx));
        }

        if (cost >= INSCRIBED_COST) {
          continue;
        }

        my = new_idx / size_x;
        mx = new_idx - (my * size_x);

        if (mx >= size_x - 3 || mx <= 3) {
          continue;
        }
        if (my >= size_y - 3 || my <= 3) {
          continue;
        }

        existing_cost = obstacle_heuristic_lookup_table_[new_idx];
        if (existing_cost <= 0.0f) {
          if (use_quadratic_cost_penalty) {
            travel_cost =
              (i <= 3 ? 1.0f : sqrt2) * (1.0f + (cost_penalty * cost * cost / 63504.0f));
          } else {
            travel_cost =
              ((i <= 3) ? 1.0f : sqrt2) * (1.0f + (cost_penalty * cost / 252.0f));
          }

          new_cost = c_cost + travel_cost;
          if (existing_cost == 0.0f || -existing_cost > new_cost) {
            obstacle_heuristic_lookup_table_[new_idx] = -new_cost;
            obstacle_heuristic_queue_.emplace_back(
              new_cost + distanceHeuristic2D(new_idx, size_x, start_x, start_y), new_idx);
            std::push_heap(
              obstacle_heuristic_queue_.begin(), obstacle_heuristic_queue_.end(),
              ObstacleHeuristicComparator{});
          }
        }
      }
    }

    if (idx == start_index) {
      break;
    }
  }
  return downsample_obstacle_heuristic ? 2.0f * requested_node_cost : requested_node_cost;
}

}  // namespace hybrid_astar
