#ifndef HYBRID_ASTAR__COSTMAP_2D_HPP_
#define HYBRID_ASTAR__COSTMAP_2D_HPP_

#include <vector>
#include <cmath>
#include <mutex>
#include <cstring>

namespace hybrid_astar
{

class Costmap2D
{
public:
  using mutex_t = std::recursive_mutex;

  Costmap2D()
  : size_x_(0), size_y_(0), resolution_(0.0), origin_x_(0.0), origin_y_(0.0)
  {}

  Costmap2D(unsigned int size_x, unsigned int size_y, double resolution,
            double origin_x, double origin_y, unsigned char default_cost = 0)
  : size_x_(size_x), size_y_(size_y), resolution_(resolution),
    origin_x_(origin_x), origin_y_(origin_y),
    cost_map_(size_x * size_y, default_cost)
  {}

  unsigned char getCost(unsigned int mx, unsigned int my) const
  {
    return cost_map_[my * size_x_ + mx];
  }

  unsigned char getCost(unsigned int index) const
  {
    return cost_map_[index];
  }

  // Safe float-coordinate lookup: rounds to nearest cell with bounds clamping.
  // Resolves Bug #12 (half-cell offset), Bug #20 (OOB), and Bug #5/#23 (negative coords).
  unsigned char getCost(float fx, float fy) const
  {
    int ix = static_cast<int>(std::floor(fx + 0.5f));
    int iy = static_cast<int>(std::floor(fy + 0.5f));
    if (ix < 0) {ix = 0;} else if (ix >= static_cast<int>(size_x_)) {ix = size_x_ - 1;}
    if (iy < 0) {iy = 0;} else if (iy >= static_cast<int>(size_y_)) {iy = size_y_ - 1;}
    return cost_map_[static_cast<unsigned int>(iy) * size_x_ + static_cast<unsigned int>(ix)];
  }

  // Convert world coords directly to map coords (float version).
  // Returns false if the point is outside the map.
  bool worldToMapContinuous(float wx, float wy, float & mx, float & my) const
  {
    mx = static_cast<float>((wx - origin_x_) / resolution_);
    my = static_cast<float>((wy - origin_y_) / resolution_);
    return mx >= 0.0f && mx < static_cast<float>(size_x_) &&
           my >= 0.0f && my < static_cast<float>(size_y_);
  }

  // Convert continuous cell coordinate to world coords (no +0.5 offset).
  // Use mapToWorld() for integer cell index to cell-center conversion.
  void mapCellToWorld(float mx, float my, float & wx, float & wy) const
  {
    wx = static_cast<float>(origin_x_) + mx * static_cast<float>(resolution_);
    wy = static_cast<float>(origin_y_) + my * static_cast<float>(resolution_);
  }

  void mapCellToWorld(float mx, float my, double & wx, double & wy) const
  {
    wx = origin_x_ + static_cast<double>(mx) * resolution_;
    wy = origin_y_ + static_cast<double>(my) * resolution_;
  }

  void setCost(unsigned int mx, unsigned int my, unsigned char cost)
  {
    cost_map_[my * size_x_ + mx] = cost;
  }

  unsigned int getSizeInCellsX() const { return size_x_; }
  unsigned int getSizeInCellsY() const { return size_y_; }
  double getResolution() const { return resolution_; }
  double getOriginX() const { return origin_x_; }
  double getOriginY() const { return origin_y_; }

  bool worldToMap(double wx, double wy, unsigned int& mx, unsigned int& my) const
  {
    double map_x = (wx - origin_x_) / resolution_;
    double map_y = (wy - origin_y_) / resolution_;
    if (map_x < 0.0 || map_y < 0.0) {
      return false;
    }
    mx = static_cast<unsigned int>(map_x);
    my = static_cast<unsigned int>(map_y);
    return mx < size_x_ && my < size_y_;
  }

  bool worldToMapContinuous(double wx, double wy, float& mx, float& my) const
  {
    mx = static_cast<float>((wx - origin_x_) / resolution_);
    my = static_cast<float>((wy - origin_y_) / resolution_);
    return mx >= 0.0f && mx < static_cast<float>(size_x_) &&
           my >= 0.0f && my < static_cast<float>(size_y_);
  }

  void mapToWorld(unsigned int mx, unsigned int my, double& wx, double& wy) const
  {
    wx = origin_x_ + (mx + 0.5) * resolution_;
    wy = origin_y_ + (my + 0.5) * resolution_;
  }

  void resizeMap(unsigned int size_x, unsigned int size_y, double resolution,
                 double origin_x, double origin_y)
  {
    size_x_ = size_x;
    size_y_ = size_y;
    resolution_ = resolution;
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    cost_map_.assign(size_x * size_y, 0);
  }

  void reset() { std::fill(cost_map_.begin(), cost_map_.end(), 0); }

  mutex_t* getMutex() { return &mutex_; }

  unsigned char* getCharMap() { return cost_map_.data(); }
  const unsigned char* getCharMap() const { return cost_map_.data(); }

private:
  unsigned int size_x_, size_y_;
  double resolution_, origin_x_, origin_y_;
  std::vector<unsigned char> cost_map_;
  mutable mutex_t mutex_;
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__COSTMAP_2D_HPP_
