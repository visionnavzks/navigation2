#ifndef COSTMAP_2D__OCCUPANCY_GRID_HPP_
#define COSTMAP_2D__OCCUPANCY_GRID_HPP_

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cost_values.hpp"

namespace costmap_2d {

struct OccupancyGrid {
  unsigned int width{0};
  unsigned int height{0};
  double resolution{1.0};
  double origin_x{0.0};
  double origin_y{0.0};
  std::string frame_id{"map"};
  std::vector<int8_t> data;

  OccupancyGrid() = default;

  OccupancyGrid(unsigned int width_in, unsigned int height_in,
                double resolution_in, double origin_x_in, double origin_y_in,
                int8_t default_value = OCC_GRID_FREE,
                std::string frame_id_in = "map")
      : width(width_in), height(height_in), resolution(resolution_in),
        origin_x(origin_x_in), origin_y(origin_y_in),
        frame_id(std::move(frame_id_in)), data(width * height, default_value) {}

  bool valid() const {
    return width > 0 && height > 0 && resolution > 0.0 &&
           data.size() == width * height;
  }

  unsigned int getIndex(unsigned int mx, unsigned int my) const {
    return my * width + mx;
  }

  int8_t getData(unsigned int mx, unsigned int my) const {
    return data[getIndex(mx, my)];
  }

  void setData(unsigned int mx, unsigned int my, int8_t value) {
    data[getIndex(mx, my)] = value;
  }
};

inline bool worldToMap(const OccupancyGrid &grid, double wx, double wy,
                       unsigned int &mx, unsigned int &my) {
  if (wx < grid.origin_x || wy < grid.origin_y || grid.resolution <= 0.0) {
    return false;
  }

  mx = static_cast<unsigned int>((wx - grid.origin_x) / grid.resolution);
  my = static_cast<unsigned int>((wy - grid.origin_y) / grid.resolution);
  return mx < grid.width && my < grid.height;
}

inline unsigned char occupancyToCost(int8_t data) {
  if (data == OCC_GRID_UNKNOWN) {
    return NO_INFORMATION;
  }

  const double scaled = static_cast<double>(data) *
                        (LETHAL_OBSTACLE - FREE_SPACE) /
                        (OCC_GRID_OCCUPIED - OCC_GRID_FREE);
  return static_cast<unsigned char>(std::round(scaled));
}

} // namespace costmap_2d

#endif // COSTMAP_2D__OCCUPANCY_GRID_HPP_