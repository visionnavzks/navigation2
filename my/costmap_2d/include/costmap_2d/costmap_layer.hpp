#ifndef COSTMAP_2D__COSTMAP_LAYER_HPP_
#define COSTMAP_2D__COSTMAP_LAYER_HPP_

#include <algorithm>

#include "costmap_2d.hpp"

namespace costmap_2d {

inline void updateWithTrueOverwrite(const Costmap2D &layer,
                                    Costmap2D &master_grid, int min_i,
                                    int min_j, int max_i, int max_j) {
  const unsigned char *layer_array = layer.getCharMap();
  unsigned char *master = master_grid.getCharMap();
  const unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; ++j) {
    unsigned int index = span * j + min_i;
    for (int i = min_i; i < max_i; ++i) {
      master[index] = layer_array[index];
      ++index;
    }
  }
}

inline void updateWithOverwrite(const Costmap2D &layer, Costmap2D &master_grid,
                                int min_i, int min_j, int max_i, int max_j) {
  const unsigned char *layer_array = layer.getCharMap();
  unsigned char *master = master_grid.getCharMap();
  const unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; ++j) {
    unsigned int index = span * j + min_i;
    for (int i = min_i; i < max_i; ++i) {
      if (layer_array[index] != NO_INFORMATION) {
        master[index] = layer_array[index];
      }
      ++index;
    }
  }
}

inline void updateWithMax(const Costmap2D &layer, Costmap2D &master_grid,
                          int min_i, int min_j, int max_i, int max_j) {
  const unsigned char *layer_array = layer.getCharMap();
  unsigned char *master = master_grid.getCharMap();
  const unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; ++j) {
    unsigned int index = span * j + min_i;
    for (int i = min_i; i < max_i; ++i) {
      if (layer_array[index] != NO_INFORMATION) {
        const unsigned char old_cost = master[index];
        if (old_cost == NO_INFORMATION || old_cost < layer_array[index]) {
          master[index] = layer_array[index];
        }
      }
      ++index;
    }
  }
}

inline void updateWithMaxWithoutUnknownOverwrite(const Costmap2D &layer,
                                                 Costmap2D &master_grid,
                                                 int min_i, int min_j,
                                                 int max_i, int max_j) {
  const unsigned char *layer_array = layer.getCharMap();
  unsigned char *master = master_grid.getCharMap();
  const unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; ++j) {
    unsigned int index = span * j + min_i;
    for (int i = min_i; i < max_i; ++i) {
      if (layer_array[index] != NO_INFORMATION) {
        const unsigned char old_cost = master[index];
        if (old_cost != NO_INFORMATION && old_cost < layer_array[index]) {
          master[index] = layer_array[index];
        }
      }
      ++index;
    }
  }
}

inline void updateWithAddition(const Costmap2D &layer, Costmap2D &master_grid,
                               int min_i, int min_j, int max_i, int max_j) {
  const unsigned char *layer_array = layer.getCharMap();
  unsigned char *master = master_grid.getCharMap();
  const unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; ++j) {
    unsigned int index = span * j + min_i;
    for (int i = min_i; i < max_i; ++i) {
      if (layer_array[index] != NO_INFORMATION) {
        const unsigned char old_cost = master[index];
        if (old_cost == NO_INFORMATION) {
          master[index] = layer_array[index];
        } else {
          const int sum = old_cost + layer_array[index];
          master[index] = sum >= INSCRIBED_INFLATED_OBSTACLE
                              ? INSCRIBED_INFLATED_OBSTACLE - 1
                              : static_cast<unsigned char>(sum);
        }
      }
      ++index;
    }
  }
}

inline CombinationMethod combinationMethodFromInt(int value) {
  switch (value) {
  case 0:
    return CombinationMethod::Overwrite;
  case 1:
    return CombinationMethod::Max;
  case 2:
    return CombinationMethod::MaxWithoutUnknownOverwrite;
  case 3:
    return CombinationMethod::Addition;
  default:
    return CombinationMethod::Max;
  }
}

inline void clearArea(Costmap2D &map, int start_x, int start_y, int end_x,
                      int end_y, bool invert) {
  unsigned char *grid = map.getCharMap();
  const int size_x = static_cast<int>(map.getSizeInCellsX());
  const int size_y = static_cast<int>(map.getSizeInCellsY());

  start_x = std::min(std::max(start_x, 0), size_x);
  start_y = std::min(std::max(start_y, 0), size_y);
  end_x = std::min(std::max(end_x, 0), size_x);
  end_y = std::min(std::max(end_y, 0), size_y);

  for (int x = 0; x < size_x; ++x) {
    const bool xrange = x > start_x && x < end_x;
    for (int y = 0; y < size_y; ++y) {
      if ((xrange && y > start_y && y < end_y) == invert) {
        continue;
      }
      const int index = static_cast<int>(map.getIndex(x, y));
      if (grid[index] != NO_INFORMATION) {
        grid[index] = NO_INFORMATION;
      }
    }
  }
}

} // namespace costmap_2d

#endif // COSTMAP_2D__COSTMAP_LAYER_HPP_