#include "my/hybrid_astar/obstacle_heuristic.hpp"
#include "my/hybrid_astar/costmap_2d.hpp"

namespace hybrid_astar
{

void ObstacleHeuristic::resetObstacleHeuristic(
  Costmap2D * costmap,
  const float & start_x, const float & start_y,
  const float & goal_x, const float & goal_y,
  const bool downsample_obstacle_heuristic)
{
  this->costmap = costmap;

  unsigned int size = 0u;
  if (downsample_obstacle_heuristic) {
    cached_size_x_ = ceil(static_cast<float>(costmap->getSizeInCellsX()) / 2.0f);
    cached_size_y_ = ceil(static_cast<float>(costmap->getSizeInCellsY()) / 2.0f);
  } else {
    cached_size_x_ = costmap->getSizeInCellsX();
    cached_size_y_ = costmap->getSizeInCellsY();
  }
  size = cached_size_x_ * cached_size_y_;

  if (obstacle_heuristic_lookup_table_.size() == size) {
    std::fill(
      obstacle_heuristic_lookup_table_.begin(),
      obstacle_heuristic_lookup_table_.end(), 0.0f);
  } else {
    obstacle_heuristic_lookup_table_.resize(size, 0.0f);
    std::fill(
      obstacle_heuristic_lookup_table_.begin(),
      obstacle_heuristic_lookup_table_.end(), 0.0f);
  }

  obstacle_heuristic_queue_.clear();
  obstacle_heuristic_queue_.reserve(size);

  const float goal_x_floor = std::max(0.0f, std::floor(goal_x));
  const float goal_y_floor = std::max(0.0f, std::floor(goal_y));
  unsigned int gx = static_cast<unsigned int>(goal_x_floor);
  unsigned int gy = static_cast<unsigned int>(goal_y_floor);
  if (gx >= cached_size_x_) {gx = cached_size_x_ - 1u;}
  if (gy >= cached_size_y_) {gy = cached_size_y_ - 1u;}
  unsigned int goal_index;
  if (downsample_obstacle_heuristic) {
    goal_index = (gy / 2u) * cached_size_x_ + (gx / 2u);
  } else {
    goal_index = gy * cached_size_x_ + gx;
  }

  const float inv = downsample_obstacle_heuristic ? 2.0f : 1.0f;
  const float start_x_floor = std::floor(start_x / inv);
  const float start_y_floor = std::floor(start_y / inv);
  obstacle_heuristic_queue_.emplace_back(
    distanceHeuristic2D(
      goal_index, cached_size_x_,
      static_cast<unsigned int>(start_x_floor),
      static_cast<unsigned int>(start_y_floor)),
    goal_index);

  obstacle_heuristic_lookup_table_[goal_index] = -0.00001f;
}

float ObstacleHeuristic::getObstacleHeuristic(
  const Coordinates & node_coords,
  const float & cost_penalty,
  const bool use_quadratic_cost_penalty,
  const bool downsample_obstacle_heuristic)
{
  const unsigned int size_x = cached_size_x_;
  const unsigned int size_y = cached_size_y_;

  float start_x_f, start_y_f;
  if (downsample_obstacle_heuristic) {
    start_y_f = std::floor(node_coords.y / 2.0f);
    start_x_f = std::floor(node_coords.x / 2.0f);
  } else {
    start_y_f = std::floor(node_coords.y);
    start_x_f = std::floor(node_coords.x);
  }
  if (start_x_f < 0.0f) {start_x_f = 0.0f;}
  if (start_y_f < 0.0f) {start_y_f = 0.0f;}
  const unsigned int start_x = static_cast<unsigned int>(start_x_f);
  const unsigned int start_y = static_cast<unsigned int>(start_y_f);

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
    NodeHeuristicComparator{});

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
      NodeHeuristicComparator{});
    obstacle_heuristic_queue_.pop_back();
    c_cost = obstacle_heuristic_lookup_table_[idx];
    if (c_cost > 0.0f) {
      continue;
    }
    c_cost = -c_cost;
    obstacle_heuristic_lookup_table_[idx] = c_cost;

    for (unsigned int i = 0; i != neighborhood.size(); i++) {
      int new_idx_int = static_cast<int>(idx) + neighborhood[i];
      if (new_idx_int < 0 || new_idx_int >= static_cast<int>(size_x * size_y)) {
        continue;
      }
      new_idx = static_cast<unsigned int>(new_idx_int);

      const int new_my = static_cast<int>(new_idx / size_x);
      const int new_mx_signed = static_cast<int>(new_idx - static_cast<unsigned int>(new_my) * size_x);
      if (new_mx_signed < 0 || static_cast<unsigned int>(new_mx_signed) >= size_x ||
        new_my < 0 || static_cast<unsigned int>(new_my) >= size_y)
      {
        continue;
      }

      {
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

        my = static_cast<unsigned int>(new_my);
        mx = static_cast<unsigned int>(new_mx_signed);

        if (size_x <= 3u || size_y <= 3u) {
          continue;
        }
        if (mx <= 3u || mx >= size_x - 3u) {
          continue;
        }
        if (my <= 3u || my >= size_y - 3u) {
          continue;
        }

        existing_cost = obstacle_heuristic_lookup_table_[new_idx];
        if (existing_cost <= 0.0f) {
          if (use_quadratic_cost_penalty) {
            travel_cost =
              (i <= 3 ? 1.0f : sqrt2) * (1.0f + (cost_penalty * cost * cost / MAX_NON_OBSTACLE_COST_SQ));
          } else {
            travel_cost =
              ((i <= 3) ? 1.0f : sqrt2) * (1.0f + (cost_penalty * cost / MAX_NON_OBSTACLE_COST));
          }

          new_cost = c_cost + travel_cost;
          if (existing_cost == 0.0f || -existing_cost > new_cost) {
            obstacle_heuristic_lookup_table_[new_idx] = -new_cost;
            obstacle_heuristic_queue_.emplace_back(
              new_cost + distanceHeuristic2D(new_idx, size_x, start_x, start_y), new_idx);
            std::push_heap(
              obstacle_heuristic_queue_.begin(), obstacle_heuristic_queue_.end(),
              NodeHeuristicComparator{});
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
