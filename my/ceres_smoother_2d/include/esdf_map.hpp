#pragma once

/**
 * ESDF (Euclidean Signed Distance Field) Map for 2D path smoothing.
 *
 * Loads an occupancy grid and computes:
 *   - Signed distance: positive outside obstacles (free), negative inside.
 *
 * Algorithm: Felzenszwalb & Huttenlocher "Distance Transforms of Sampled Functions"
 * (2012), O(n) per row/column for exact Euclidean distance.
 *
 * No ROS dependency. Uses only stb_image and the C++ standard library.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Bilinear (not BiCubic) interpolation throughout: the ESDF has sharp
// discontinuities at obstacle boundaries and BiCubic overshoots across
// them. Bilinear is bounded by the min/max of the 4 nearest cells and
// Jet-differentiable, so it is correct and Ceres-compatible.

// stb_image implementation is in src/stb_image_impl.cpp
#include "stb_image.h"

namespace ceres_smoother_2d
{

class ESDFMap
{
public:
  /**
   * @param occupancy_path  Path to a grayscale PNG (0=obstacle, 255=free).
   * @param resolution      Meters per pixel.
   * @param origin_x        World x of grid pixel (0,0).
   * @param origin_y        World y of grid pixel (0,0).
   * @param obstacle_thresh Threshold: pixel <= this is treated as obstacle.
   */
  ESDFMap(
    const std::string & occupancy_path,
    double resolution,
    double origin_x = 0.0,
    double origin_y = 0.0,
    int obstacle_thresh = 127)
  : resolution_(resolution), origin_x_(origin_x), origin_y_(origin_y)
  {
    // Load grayscale image
    int w, h, channels;
    unsigned char * img = stbi_load(occupancy_path.c_str(), &w, &h, &channels, 1);
    if (!img) {
      throw std::runtime_error("Failed to load image: " + occupancy_path);
    }

    width_ = w;
    height_ = h;
    grid_.resize(height_ * width_);
    // Flip vertically so that PNG row 0 (the visual top) maps to world y = y_max,
    // matching the ROS map_server convention and Plotly's `yanchor: 'top'` layout.
    // After this flip: grid row r corresponds to PNG row (height-1-r), so:
    //   world y = r * resolution_  ⟺  PNG row = height-1-r
    for (int r = 0; r < height_; ++r) {
      const int src_off = r * width_;
      const int dst_off = (height_ - 1 - r) * width_;
      for (int c = 0; c < width_; ++c) {
        // In typical occupancy maps: 0 (black) = obstacle, 255 (white) = free
        // obstacle_thresh is the threshold BELOW which pixels are obstacles.
        grid_[dst_off + c] = (img[src_off + c] <= obstacle_thresh) ? 1 : 0;
      }
    }
    stbi_image_free(img);

    computeESDF();
  }

  /**
   * Construct from raw occupancy data (0=free, 1=obstacle).
   */
  ESDFMap(
    const std::vector<uint8_t> & occupancy,
    int width,
    int height,
    double resolution,
    double origin_x = 0.0,
    double origin_y = 0.0)
  : width_(width), height_(height),
    resolution_(resolution), origin_x_(origin_x), origin_y_(origin_y)
  {
    if (static_cast<int>(occupancy.size()) != width * height) {
      throw std::invalid_argument("Occupancy size does not match width*height");
    }
    grid_.assign(occupancy.begin(), occupancy.end());
    computeESDF();
  }

  // --- Accessors ---

  int width() const {return width_;}
  int height() const {return height_;}
  double resolution() const {return resolution_;}
  double originX() const {return origin_x_;}
  double originY() const {return origin_y_;}

  double worldWidth() const {return width_ * resolution_;}
  double worldHeight() const {return height_ * resolution_;}

  /** Get signed distance at a world coordinate (bilinear interpolation).
   *  Bilinear (not BiCubic) because the ESDF has a sharp discontinuity at
   *  obstacle boundaries — BiCubic overshoots across it and returns wrong
   *  values. Bilinear is bounded by the min/max of the 4 nearest cells. */
  double getDistance(double wx, double wy) const
  {
    return bilinearJet<double>(wx, wy);
  }

  /**
   * Get the signed distance field value at fractional grid coordinates.
   * Uses bilinear interpolation matching getDistance().
   */
  double esdfAtGrid(double col, double row) const
  {
    // grid-space (col, row) -> world coords
    return bilinearJet<double>(col * resolution_ + origin_x_,
                                row * resolution_ + origin_y_);
  }

  /** Jet-aware bilinear ESDF lookup. Returns the distance at the given
   *  world coordinate with partial derivatives wrt `wx` and `wy`.
   *  Used by Ceres cost functions for AutoDiff. Bilinear is C^0 (only
   *  continuous, not differentiable at cell boundaries) but always
   *  bounded by neighbor min/max — unlike BiCubic which overshoots.
   *  Note: PNG input rows are flipped once at load time so internal grid row
   *  indices match world-y grid row indices. Raw occupancy input is already
   *  expected in that same internal convention. */
  template<typename T>
  T bilinearJet(T wx, T wy) const
  {
    if (width_ <= 0 || height_ <= 0) {return T(0.0);}
    const T col = (wx - T(origin_x_)) / T(resolution_);
    const T row = (wy - T(origin_y_)) / T(resolution_);
    const T r_arr = ceres::fmax(T(0.0), ceres::fmin(T(height_ - 1), row));
    const T c = ceres::fmax(T(0.0), ceres::fmin(T(width_ - 1), col));
    // Integer indices: extract the SCALAR part of the Jet (avoids ADL
    // confusion with ceres::Jet members; uses `r.a` for Jet, `r` for double).
    double r_scalar, c_scalar;
    if constexpr (std::is_same<T, double>::value) {
      r_scalar = static_cast<double>(r_arr);
      c_scalar = static_cast<double>(c);
    } else {
      r_scalar = r_arr.a;
      c_scalar = c.a;
    }
    const int r0 = height_ > 1 ?
      std::min(static_cast<int>(std::floor(r_scalar)), height_ - 2) : 0;
    const int c0 = width_ > 1 ?
      std::min(static_cast<int>(std::floor(c_scalar)), width_ - 2) : 0;
    const int r1 = std::min(r0 + 1, height_ - 1);
    const int c1 = std::min(c0 + 1, width_ - 1);
    const T fr = r_arr - T(r0);
    const T fc = c - T(c0);
    const T v00 = T(esdf_[r0 * width_ + c0]);
    const T v10 = T(esdf_[r0 * width_ + c1]);
    const T v01 = T(esdf_[r1 * width_ + c0]);
    const T v11 = T(esdf_[r1 * width_ + c1]);
    return v00 * (T(1.0) - fr) * (T(1.0) - fc)
         + v10 * fc * (T(1.0) - fr)
         + v01 * (T(1.0) - fc) * fr
         + v11 * fc * fr;
  }

  /**
   * Convert world coordinates to fractional grid coordinates.
   */
  void worldToGrid(double wx, double wy, double & col, double & row) const
  {
    col = (wx - origin_x_) / resolution_;
    row = (wy - origin_y_) / resolution_;
  }

  /** Check if a world point is inside the map bounds. */
  bool inBounds(double wx, double wy, double margin = 0.0) const
  {
    // Compare in world coordinates to avoid floating-point precision issues
    // at exact grid edges (e.g. world_width / resolution may not be exact).
    const double min_x = origin_x_ + margin * resolution_;
    const double min_y = origin_y_ + margin * resolution_;
    const double max_x = origin_x_ + (width_  - margin) * resolution_;
    const double max_y = origin_y_ + (height_ - margin) * resolution_;
    return wx >= min_x && wx < max_x &&
           wy >= min_y && wy < max_y;
  }

  /** Get the underlying ESDF grid in row-major internal/world-y order. */
  const std::vector<double> & esdfGrid() const {return esdf_;}

  const std::vector<uint8_t> & occupancyGrid() const {return grid_;}

private:
  int width_, height_;
  double resolution_, origin_x_, origin_y_;
  std::vector<uint8_t> grid_;
  std::vector<double> esdf_;

  // ====================================================================
  // 1D Squared Euclidean Distance Transform
  // Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled Functions"
  // Input: f[q] (cost function), Output: d[q] (squared distance to minimum)
  // ====================================================================
  static void distTransform1D(const double * f, double * d, int n)
  {
    if (n == 0) {return;}

    std::vector<int> v(n);   // locations of parabolas
    std::vector<double> z(n + 1);  // boundaries between parabolas
    int k = 0;
    v[0] = 0;
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] = std::numeric_limits<double>::infinity();

    auto sq = [](double x) {return x * x;};

    for (int q = 1; q < n; ++q) {
      // Intersection of parabola from q with parabola from v[k]
      double s = ((sq(q) - sq(v[k])) + f[q] - f[v[k]]) / (2.0 * (q - v[k]));
      while (k > 0 && s <= z[k]) {
        --k;
        s = ((sq(q) - sq(v[k])) + f[q] - f[v[k]]) / (2.0 * (q - v[k]));
      }
      ++k;
      v[k] = q;
      z[k] = s;
      z[k + 1] = std::numeric_limits<double>::infinity();
    }

    k = 0;
    for (int q = 0; q < n; ++q) {
      while (z[k + 1] < q) {++k;}
      d[q] = sq(q - v[k]) + f[v[k]];
    }
  }

  // ====================================================================
  // 2D Squared Euclidean Distance Transform
  // ====================================================================
  static std::vector<double> edt2d(const std::vector<uint8_t> & binary, int w, int h)
  {
    const double INF = 1e20;
    std::vector<double> d(w * h);

    // Initialize: obstacle=0, free=INF
    for (int i = 0; i < w * h; ++i) {
      d[i] = binary[i] ? 0.0 : INF;
    }

    // Transform columns (each column has h elements)
    std::vector<double> col(h);
    std::vector<double> col_out(h);
    for (int x = 0; x < w; ++x) {
      for (int y = 0; y < h; ++y) {
        col[y] = d[y * w + x];
      }
      distTransform1D(col.data(), col_out.data(), h);
      for (int y = 0; y < h; ++y) {
        d[y * w + x] = col_out[y];
      }
    }

    // Transform rows (each row has w elements)
    std::vector<double> row(w);
    std::vector<double> row_out(w);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        row[x] = d[y * w + x];
      }
      distTransform1D(row.data(), row_out.data(), w);
      for (int x = 0; x < w; ++x) {
        d[y * w + x] = row_out[x];
      }
    }

    return d;
  }

  // ====================================================================
  // Compute Signed Euclidean Distance Field
  // Positive = free space (outside obstacle), Negative = inside obstacle
  // ====================================================================
  void computeESDF()
  {
    esdf_.resize(width_ * height_);

    // Compute distance from free to nearest obstacle: d_free[y*w+x]
    // For free pixels: squared distance to nearest obstacle
    // For obstacle pixels: 0
    auto d_free = edt2d(grid_, width_, height_);

    // Compute distance from obstacle to nearest free: d_obst[y*w+x]
    // For obstacle pixels: squared distance to nearest free pixel
    // For free pixels: 0
    std::vector<uint8_t> inv_grid(width_ * height_);
    for (int i = 0; i < width_ * height_; ++i) {
      inv_grid[i] = grid_[i] ? 0 : 1;
    }
    auto d_obst = edt2d(inv_grid, width_, height_);

    // Signed distance: free -> positive, obstacle -> negative
    for (int i = 0; i < width_ * height_; ++i) {
      if (grid_[i]) {
        // Obstacle pixel: negative distance to nearest free
        esdf_[i] = -std::sqrt(d_obst[i]) * resolution_;
      } else {
        // Free pixel: positive distance to nearest obstacle
        esdf_[i] = std::sqrt(d_free[i]) * resolution_;
      }
    }
  }

};

}  // namespace ceres_smoother_2d
