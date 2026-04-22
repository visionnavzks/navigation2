// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0
//
// Euclidean Signed Distance Field (ESDF) grid.
// Provides smooth, differentiable obstacle distance queries for Ceres.

#ifndef KINEMATIC_SMOOTHER__ESDF_HPP_
#define KINEMATIC_SMOOTHER__ESDF_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <vector>
#include "ceres/cubic_interpolation.h"

namespace kinematic_smoother
{

/// A simple axis-aligned rectangle obstacle.
struct RectObstacle
{
  double x_min, y_min, x_max, y_max;
};

/// 2-D Euclidean Signed Distance Field stored on a regular grid.
class ESDF
{
public:
  using Grid = ceres::Grid2D<double>;
  using Interpolator = ceres::BiCubicInterpolator<Grid>;

  ESDF() = default;

  /// Build an ESDF from a set of rectangular obstacles.
  void build(double resolution, double origin_x, double origin_y,
             int width, int height,
             const std::vector<RectObstacle> & obstacles)
  {
    resolution_ = resolution;
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    width_ = width;
    height_ = height;
    data_.assign(static_cast<size_t>(width) * height,
                 std::numeric_limits<double>::max());

    // Mark obstacle cells (distance = 0)
    for (auto & obs : obstacles) {
      int gx_min = worldToGridX(obs.x_min);
      int gy_min = worldToGridY(obs.y_min);
      int gx_max = worldToGridX(obs.x_max);
      int gy_max = worldToGridY(obs.y_max);
      gx_min = std::clamp(gx_min, 0, width_ - 1);
      gx_max = std::clamp(gx_max, 0, width_ - 1);
      gy_min = std::clamp(gy_min, 0, height_ - 1);
      gy_max = std::clamp(gy_max, 0, height_ - 1);
      for (int gy = gy_min; gy <= gy_max; ++gy) {
        for (int gx = gx_min; gx <= gx_max; ++gx) {
          data_[index(gx, gy)] = 0.0;
        }
      }
    }

    // Compute the distance transform using BFS
    computeDistanceTransform();

    // Convert grid-cell distances to metric distances
    for (auto & d : data_) {
      d *= resolution_;
    }

    // Build the Ceres interpolator
    grid_ = std::make_unique<Grid>(data_.data(), 0, height_, 0, width_);
    interpolator_ = std::make_unique<Interpolator>(*grid_);
  }

  /// Build from raw distance data (e.g. received from external source).
  void buildFromData(double resolution, double origin_x, double origin_y,
                     int width, int height,
                     std::vector<double> data)
  {
    resolution_ = resolution;
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    width_ = width;
    height_ = height;
    data_ = std::move(data);
    grid_ = std::make_unique<Grid>(data_.data(), 0, height_, 0, width_);
    interpolator_ = std::make_unique<Interpolator>(*grid_);
  }

  bool valid() const { return interpolator_ != nullptr; }

  const Interpolator * interpolator() const { return interpolator_.get(); }
  double resolution() const { return resolution_; }
  double originX() const { return origin_x_; }
  double originY() const { return origin_y_; }
  int width() const { return width_; }
  int height() const { return height_; }
  const std::vector<double> & data() const { return data_; }

private:
  int worldToGridX(double wx) const
  {
    return static_cast<int>(std::round((wx - origin_x_) / resolution_));
  }

  int worldToGridY(double wy) const
  {
    return static_cast<int>(std::round((wy - origin_y_) / resolution_));
  }

  size_t index(int gx, int gy) const
  {
    return static_cast<size_t>(gy) * width_ + gx;
  }

  /// Multi-source BFS distance transform (Chebyshev → Euclidean approx).
  void computeDistanceTransform()
  {
    struct Cell { int x, y; double dist; };
    auto cmp = [](const Cell & a, const Cell & b) { return a.dist > b.dist; };
    std::priority_queue<Cell, std::vector<Cell>, decltype(cmp)> pq(cmp);

    // Seed with obstacle cells
    for (int gy = 0; gy < height_; ++gy) {
      for (int gx = 0; gx < width_; ++gx) {
        if (data_[index(gx, gy)] == 0.0) {
          pq.push({gx, gy, 0.0});
        }
      }
    }

    const int dx8[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy8[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const double dd8[] = {M_SQRT2, 1.0, M_SQRT2, 1.0, 1.0, M_SQRT2, 1.0, M_SQRT2};

    while (!pq.empty()) {
      auto [cx, cy, cd] = pq.top();
      pq.pop();
      if (cd > data_[index(cx, cy)]) {continue;}
      for (int k = 0; k < 8; ++k) {
        int nx = cx + dx8[k], ny = cy + dy8[k];
        if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {continue;}
        double nd = cd + dd8[k];
        if (nd < data_[index(nx, ny)]) {
          data_[index(nx, ny)] = nd;
          pq.push({nx, ny, nd});
        }
      }
    }
  }

  double resolution_{0.1};
  double origin_x_{0.0}, origin_y_{0.0};
  int width_{0}, height_{0};
  std::vector<double> data_;
  std::unique_ptr<Grid> grid_;
  std::unique_ptr<Interpolator> interpolator_;
};

}  // namespace kinematic_smoother

#endif  // KINEMATIC_SMOOTHER__ESDF_HPP_
