#ifndef COSTMAP_2D__COSTMAP_2D_HPP_
#define COSTMAP_2D__COSTMAP_2D_HPP_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cost_values.hpp"
#include "geometry.hpp"
#include "occupancy_grid.hpp"

namespace costmap_2d {

struct MapLocation {
  unsigned int x{0};
  unsigned int y{0};
  unsigned char cost{FREE_SPACE};
};

template <typename Action>
void raytraceLine(Action action, unsigned int x0, unsigned int y0,
                  unsigned int x1, unsigned int y1, unsigned int size_x) {
  int dx = static_cast<int>(x1) - static_cast<int>(x0);
  int dy = static_cast<int>(y1) - static_cast<int>(y0);
  const int abs_dx = std::abs(dx);
  const int abs_dy = std::abs(dy);
  const int offset_dx = dx > 0 ? 1 : -1;
  const int offset_dy =
      dy > 0 ? static_cast<int>(size_x) : -static_cast<int>(size_x);
  unsigned int offset = y0 * size_x + x0;

  action(offset);
  if (abs_dx >= abs_dy) {
    int error_y = abs_dx / 2;
    for (int i = 0; i < abs_dx; ++i) {
      offset += offset_dx;
      error_y += abs_dy;
      if (error_y >= abs_dx) {
        offset += offset_dy;
        error_y -= abs_dx;
      }
      action(offset);
    }
  } else {
    int error_x = abs_dy / 2;
    for (int i = 0; i < abs_dy; ++i) {
      offset += offset_dy;
      error_x += abs_dx;
      if (error_x >= abs_dy) {
        offset += offset_dx;
        error_x -= abs_dy;
      }
      action(offset);
    }
  }
}

class Costmap2D {
public:
  using mutex_t = std::recursive_mutex;

  Costmap2D() = default;

  Costmap2D(unsigned int cells_size_x, unsigned int cells_size_y,
            double resolution, double origin_x, double origin_y,
            unsigned char default_value = FREE_SPACE)
      : size_x_(cells_size_x), size_y_(cells_size_y), resolution_(resolution),
        origin_x_(origin_x), origin_y_(origin_y), default_value_(default_value),
        costmap_(size_x_ * size_y_, default_value_) {}

  explicit Costmap2D(const OccupancyGrid &map)
      : size_x_(map.width), size_y_(map.height), resolution_(map.resolution),
        origin_x_(map.origin_x), origin_y_(map.origin_y),
        default_value_(FREE_SPACE), costmap_(size_x_ * size_y_, FREE_SPACE) {
    if (!map.valid()) {
      throw std::invalid_argument("OccupancyGrid is empty or malformed");
    }
    for (unsigned int index = 0; index < costmap_.size(); ++index) {
      costmap_[index] = occupancyToCost(map.data[index]);
    }
  }

  unsigned char getCost(unsigned int mx, unsigned int my) const {
    return costmap_[getIndex(mx, my)];
  }

  unsigned char getCost(unsigned int index) const { return costmap_[index]; }

  void setCost(unsigned int mx, unsigned int my, unsigned char cost) {
    costmap_[getIndex(mx, my)] = cost;
  }

  void mapToWorld(unsigned int mx, unsigned int my, double &wx,
                  double &wy) const {
    wx = origin_x_ + (mx + 0.5) * resolution_;
    wy = origin_y_ + (my + 0.5) * resolution_;
  }

  void mapToWorldNoBounds(int mx, int my, double &wx, double &wy) const {
    wx = origin_x_ + (mx + 0.5) * resolution_;
    wy = origin_y_ + (my + 0.5) * resolution_;
  }

  bool worldToMap(double wx, double wy, unsigned int &mx,
                  unsigned int &my) const {
    if (wx < origin_x_ || wy < origin_y_ || resolution_ <= 0.0) {
      return false;
    }

    mx = static_cast<unsigned int>((wx - origin_x_) / resolution_);
    my = static_cast<unsigned int>((wy - origin_y_) / resolution_);
    return mx < size_x_ && my < size_y_;
  }

  bool worldToMapContinuous(double wx, double wy, float &mx, float &my) const {
    if (wx < origin_x_ || wy < origin_y_ || resolution_ <= 0.0) {
      return false;
    }

    mx = static_cast<float>((wx - origin_x_) / resolution_);
    my = static_cast<float>((wy - origin_y_) / resolution_);
    return mx < size_x_ && my < size_y_;
  }

  void worldToMapNoBounds(double wx, double wy, int &mx, int &my) const {
    mx = static_cast<int>((wx - origin_x_) / resolution_);
    my = static_cast<int>((wy - origin_y_) / resolution_);
  }

  void worldToMapEnforceBounds(double wx, double wy, int &mx, int &my) const {
    if (wx < origin_x_) {
      mx = 0;
    } else if (wx > resolution_ * size_x_ + origin_x_) {
      mx = static_cast<int>(size_x_) - 1;
    } else {
      mx = static_cast<int>((wx - origin_x_) / resolution_);
    }

    if (wy < origin_y_) {
      my = 0;
    } else if (wy > resolution_ * size_y_ + origin_y_) {
      my = static_cast<int>(size_y_) - 1;
    } else {
      my = static_cast<int>((wy - origin_y_) / resolution_);
    }
  }

  unsigned int getIndex(unsigned int mx, unsigned int my) const {
    return my * size_x_ + mx;
  }

  void indexToCells(unsigned int index, unsigned int &mx,
                    unsigned int &my) const {
    my = index / size_x_;
    mx = index - my * size_x_;
  }

  unsigned char *getCharMap() { return costmap_.data(); }

  const unsigned char *getCharMap() const { return costmap_.data(); }

  unsigned int getSizeInCellsX() const { return size_x_; }
  unsigned int getSizeInCellsY() const { return size_y_; }
  double getSizeInMetersX() const { return size_x_ * resolution_; }
  double getSizeInMetersY() const { return size_y_ * resolution_; }
  double getOriginX() const { return origin_x_; }
  double getOriginY() const { return origin_y_; }
  double getResolution() const { return resolution_; }

  void setDefaultValue(unsigned char value) { default_value_ = value; }
  unsigned char getDefaultValue() const { return default_value_; }

  mutex_t *getMutex() { return &access_; }

  void resizeMap(unsigned int size_x, unsigned int size_y, double resolution,
                 double origin_x, double origin_y) {
    resolution_ = resolution;
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    initMaps(size_x, size_y);
    resetMaps();
  }

  void resetMaps() {
    std::lock_guard<mutex_t> lock(access_);
    std::fill(costmap_.begin(), costmap_.end(), default_value_);
  }

  void resetMap(unsigned int x0, unsigned int y0, unsigned int xn,
                unsigned int yn) {
    resetMapToValue(x0, y0, xn, yn, default_value_);
  }

  void resetMapToValue(unsigned int x0, unsigned int y0, unsigned int xn,
                       unsigned int yn, unsigned char value) {
    std::lock_guard<mutex_t> lock(access_);
    if (xn <= x0 || yn <= y0) {
      return;
    }

    const unsigned int row_length = xn - x0;
    for (unsigned int y = y0; y < yn; ++y) {
      std::fill_n(costmap_.begin() + getIndex(x0, y), row_length, value);
    }
  }

  bool copyCostmapWindow(const Costmap2D &map, double win_origin_x,
                         double win_origin_y, double win_size_x,
                         double win_size_y) {
    if (this == &map) {
      return false;
    }

    unsigned int lower_left_x, lower_left_y, upper_right_x, upper_right_y;
    if (!map.worldToMap(win_origin_x, win_origin_y, lower_left_x,
                        lower_left_y) ||
        !map.worldToMap(win_origin_x + win_size_x, win_origin_y + win_size_y,
                        upper_right_x, upper_right_y)) {
      return false;
    }

    resolution_ = map.resolution_;
    origin_x_ = win_origin_x;
    origin_y_ = win_origin_y;
    initMaps(upper_right_x - lower_left_x, upper_right_y - lower_left_y);
    copyMapRegion(map.costmap_.data(), lower_left_x, lower_left_y, map.size_x_,
                  costmap_.data(), 0, 0, size_x_, size_x_, size_y_);
    return true;
  }

  bool copyWindow(const Costmap2D &source, unsigned int sx0, unsigned int sy0,
                  unsigned int sxn, unsigned int syn, unsigned int dx0,
                  unsigned int dy0) {
    if (sxn <= sx0 || syn <= sy0) {
      return false;
    }

    const unsigned int sz_x = sxn - sx0;
    const unsigned int sz_y = syn - sy0;

    if (sxn > source.getSizeInCellsX() || syn > source.getSizeInCellsY()) {
      return false;
    }
    if (dx0 + sz_x > size_x_ || dy0 + sz_y > size_y_) {
      return false;
    }

    copyMapRegion(source.costmap_.data(), sx0, sy0, source.size_x_,
                  costmap_.data(), dx0, dy0, size_x_, sz_x, sz_y);
    return true;
  }

  void updateOrigin(double new_origin_x, double new_origin_y) {
    const int cell_ox =
        static_cast<int>((new_origin_x - origin_x_) / resolution_);
    const int cell_oy =
        static_cast<int>((new_origin_y - origin_y_) / resolution_);
    const double new_grid_ox = origin_x_ + cell_ox * resolution_;
    const double new_grid_oy = origin_y_ + cell_oy * resolution_;
    const int size_x = static_cast<int>(size_x_);
    const int size_y = static_cast<int>(size_y_);
    const int lower_left_x = std::min(std::max(cell_ox, 0), size_x);
    const int lower_left_y = std::min(std::max(cell_oy, 0), size_y);
    const int upper_right_x = std::min(std::max(cell_ox + size_x, 0), size_x);
    const int upper_right_y = std::min(std::max(cell_oy + size_y, 0), size_y);
    const unsigned int cell_size_x = upper_right_x - lower_left_x;
    const unsigned int cell_size_y = upper_right_y - lower_left_y;
    std::vector<unsigned char> local_map(cell_size_x * cell_size_y);

    copyMapRegion(costmap_.data(), lower_left_x, lower_left_y, size_x_,
                  local_map.data(), 0, 0, cell_size_x, cell_size_x,
                  cell_size_y);
    resetMaps();
    origin_x_ = new_grid_ox;
    origin_y_ = new_grid_oy;

    const int start_x = lower_left_x - cell_ox;
    const int start_y = lower_left_y - cell_oy;
    copyMapRegion(local_map.data(), 0, 0, cell_size_x, costmap_.data(), start_x,
                  start_y, size_x_, cell_size_x, cell_size_y);
  }

  unsigned int cellDistance(double world_dist) const {
    return static_cast<unsigned int>(
        std::max(0.0, std::ceil(world_dist / resolution_)));
  }

  bool setConvexPolygonCost(const std::vector<Point> &polygon,
                            unsigned char cost_value) {
    std::vector<MapLocation> polygon_map_region;
    polygon_map_region.reserve(100);
    if (!getMapRegionOccupiedByPolygon(polygon, polygon_map_region)) {
      return false;
    }
    setMapRegionOccupiedByPolygon(polygon_map_region, cost_value);
    return true;
  }

  bool getMapRegionOccupiedByPolygon(
      const std::vector<Point> &polygon,
      std::vector<MapLocation> &polygon_map_region) const {
    std::vector<MapLocation> map_polygon;
    for (const auto &point : polygon) {
      MapLocation loc;
      if (!worldToMap(point.x, point.y, loc.x, loc.y)) {
        return false;
      }
      map_polygon.push_back(loc);
    }

    convexFillCells(map_polygon, polygon_map_region);
    return true;
  }

  void setMapRegionOccupiedByPolygon(
      const std::vector<MapLocation> &polygon_map_region,
      unsigned char new_cost_value) {
    for (const auto &cell : polygon_map_region) {
      setCost(cell.x, cell.y, new_cost_value);
    }
  }

  void restoreMapRegionOccupiedByPolygon(
      const std::vector<MapLocation> &polygon_map_region) {
    for (const auto &cell : polygon_map_region) {
      setCost(cell.x, cell.y, cell.cost);
    }
  }

  void polygonOutlineCells(const std::vector<MapLocation> &polygon,
                           std::vector<MapLocation> &polygon_cells) const {
    auto gather_cell = [this, &polygon_cells](unsigned int offset) {
      MapLocation loc;
      indexToCells(offset, loc.x, loc.y);
      loc.cost = getCost(loc.x, loc.y);
      polygon_cells.push_back(loc);
    };

    for (unsigned int i = 0; i + 1 < polygon.size(); ++i) {
      raytraceLine(gather_cell, polygon[i].x, polygon[i].y, polygon[i + 1].x,
                   polygon[i + 1].y, size_x_);
    }
    if (!polygon.empty()) {
      const unsigned int last_index = polygon.size() - 1;
      raytraceLine(gather_cell, polygon[last_index].x, polygon[last_index].y,
                   polygon[0].x, polygon[0].y, size_x_);
    }
  }

  void convexFillCells(const std::vector<MapLocation> &polygon,
                       std::vector<MapLocation> &polygon_cells) const {
    if (polygon.size() < 3) {
      return;
    }

    polygonOutlineCells(polygon, polygon_cells);
    if (polygon_cells.empty()) {
      return;
    }

    std::stable_sort(polygon_cells.begin(), polygon_cells.end(),
                     [](const MapLocation &lhs, const MapLocation &rhs) {
                       if (lhs.x == rhs.x) {
                         return lhs.y < rhs.y;
                       }
                       return lhs.x < rhs.x;
                     });

    unsigned int i = 0;
    const unsigned int min_x = polygon_cells.front().x;
    const unsigned int max_x = polygon_cells.back().x;
    for (unsigned int x = min_x; x <= max_x; ++x) {
      if (i >= polygon_cells.size() - 1) {
        break;
      }

      MapLocation min_pt = polygon_cells[i];
      MapLocation max_pt = polygon_cells[i + 1];
      if (min_pt.y > max_pt.y) {
        std::swap(min_pt, max_pt);
      }

      i += 2;
      while (i < polygon_cells.size() && polygon_cells[i].x == x) {
        if (polygon_cells[i].y < min_pt.y) {
          min_pt = polygon_cells[i];
        } else if (polygon_cells[i].y > max_pt.y) {
          max_pt = polygon_cells[i];
        }
        ++i;
      }

      for (unsigned int y = min_pt.y; y <= max_pt.y; ++y) {
        polygon_cells.push_back(MapLocation{x, y, getCost(x, y)});
      }
    }
  }

  bool saveMap(const std::string &file_name) const {
    FILE *file = std::fopen(file_name.c_str(), "w");
    if (!file) {
      return false;
    }

    std::fprintf(file, "P2\n%u\n%u\n%u\n", size_x_, size_y_, 0xff);
    for (unsigned int iy = 0; iy < size_y_; ++iy) {
      for (unsigned int ix = 0; ix < size_x_; ++ix) {
        std::fprintf(file, "%d ", getCost(ix, iy));
      }
      std::fprintf(file, "\n");
    }
    std::fclose(file);
    return true;
  }

  template <typename DataType>
  static void
  copyMapRegion(const DataType *source_map, unsigned int sm_lower_left_x,
                unsigned int sm_lower_left_y, unsigned int sm_size_x,
                DataType *dest_map, unsigned int dm_lower_left_x,
                unsigned int dm_lower_left_y, unsigned int dm_size_x,
                unsigned int region_size_x, unsigned int region_size_y) {
    const DataType *sm_index =
        source_map + sm_lower_left_y * sm_size_x + sm_lower_left_x;
    DataType *dm_index =
        dest_map + dm_lower_left_y * dm_size_x + dm_lower_left_x;
    for (unsigned int i = 0; i < region_size_y; ++i) {
      std::memcpy(dm_index, sm_index, region_size_x * sizeof(DataType));
      sm_index += sm_size_x;
      dm_index += dm_size_x;
    }
  }

private:
  void initMaps(unsigned int size_x, unsigned int size_y) {
    std::lock_guard<mutex_t> lock(access_);
    size_x_ = size_x;
    size_y_ = size_y;
    costmap_.assign(size_x_ * size_y_, default_value_);
  }

  mutable mutex_t access_;
  unsigned int size_x_{0};
  unsigned int size_y_{0};
  double resolution_{0.0};
  double origin_x_{0.0};
  double origin_y_{0.0};
  unsigned char default_value_{FREE_SPACE};
  std::vector<unsigned char> costmap_;
};

} // namespace costmap_2d

#endif // COSTMAP_2D__COSTMAP_2D_HPP_