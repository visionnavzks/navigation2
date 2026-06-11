/**
 * Demo: 2D Path Smoothing with Ceres + ESDF
 *
 * Loads an occupancy map, computes ESDF, creates a reference path,
 * smooths it using Ceres optimization, and saves a visualization.
 *
 * Usage: ./ceres_smoother_2d_demo [path_to_occupancy_map.png]
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "ceres_smoother_2d.hpp"

// stb_image_write implementation is in stb_image_impl.cpp
#include "stb_image_write.h"

using namespace ceres_smoother_2d;

// ========================================================================
// Visualization: draw path on occupancy map and save as PNG
// ========================================================================
static void saveVisualization(
  const std::string & filename,
  const std::vector<uint8_t> & occupancy,
  int map_w, int map_h,
  const std::vector<double> & ref_x, const std::vector<double> & ref_y,
  const std::vector<double> & smooth_x, const std::vector<double> & smooth_y,
  double res, double ox, double oy)
{
  // Create RGB image (3 channels)
  std::vector<uint8_t> img(map_w * map_h * 3);

  // Draw occupancy map (grayscale -> RGB)
  for (int i = 0; i < map_w * map_h; ++i) {
    uint8_t v = occupancy[i];
    img[i * 3 + 0] = v;
    img[i * 3 + 1] = v;
    img[i * 3 + 2] = v;
  }

  // Helper: draw a pixel on the image
  auto drawPixel = [&](double wx, double wy, uint8_t r, uint8_t g, uint8_t b, int radius = 1) {
    int cx = static_cast<int>((wx - ox) / res);
    int cy = static_cast<int>((wy - oy) / res);
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        int px = cx + dx;
        int py = cy + dy;
        if (px >= 0 && px < map_w && py >= 0 && py < map_h) {
          int idx = (py * map_w + px) * 3;
          img[idx + 0] = r;
          img[idx + 1] = g;
          img[idx + 2] = b;
        }
      }
    }
  };

  // Draw line between consecutive points
  auto drawLine = [&](const std::vector<double> & xs, const std::vector<double> & ys,
    uint8_t r, uint8_t g, uint8_t b, int radius = 0) {
    for (size_t i = 0; i + 1 < xs.size(); ++i) {
      double x0 = xs[i], y0 = ys[i];
      double x1 = xs[i + 1], y1 = ys[i + 1];
      double dx = x1 - x0, dy = y1 - y0;
      double dist = std::sqrt(dx * dx + dy * dy);
      int steps = std::max(1, static_cast<int>(dist / res));
      for (int s = 0; s <= steps; ++s) {
        double t = static_cast<double>(s) / steps;
        drawPixel(x0 + t * dx, y0 + t * dy, r, g, b, radius);
      }
    }
  };

  // Draw reference path (blue)
  drawLine(ref_x, ref_y, 0, 100, 255, 0);

  // Draw smoothed path (green)
  drawLine(smooth_x, smooth_y, 0, 255, 0, 1);

  // Draw start (green circle) and end (red circle) of smoothed path
  if (!smooth_x.empty()) {
    drawPixel(smooth_x.front(), smooth_y.front(), 0, 255, 0, 3);
    drawPixel(smooth_x.back(), smooth_y.back(), 255, 0, 0, 3);
  }

  // Save PNG
  if (!stbi_write_png(filename.c_str(), map_w, map_h, 3, img.data(), map_w * 3)) {
    std::cerr << "Failed to write " << filename << std::endl;
  } else {
    std::cout << "Saved visualization: " << filename << std::endl;
  }
}

// ========================================================================
// Generate a test reference path: straight line from left to right
// at the bottom of the free space
// ========================================================================
static void generateTestPath(
  const ESDFMap & map,
  std::vector<double> & ref_x,
  std::vector<double> & ref_y)
{
  double wx = map.worldWidth();
  double wy = map.worldHeight();

  // Find a good Y level in free space by scanning from bottom
  // Look for a row that's mostly free (high ESDF values)
  double best_y = wy * 0.9;  // default: near bottom
  double best_score = -1;

  for (int row = map.height() - 1; row >= 0; row -= 5) {
    double y = map.originY() + (row + 0.5) * map.resolution();
    double score = 0;
    int count = 0;
    for (int col = 0; col < map.width(); col += 10) {
      double x = map.originX() + (col + 0.5) * map.resolution();
      double d = map.getDistance(x, y);
      if (d > 0) {
        score += d;
        count++;
      }
    }
    if (count > 0) {
      double avg_score = score / count;
      if (avg_score > best_score) {
        best_score = avg_score;
        best_y = y;
      }
    }
  }

  std::cout << "Selected path Y level: " << best_y << " m (score: " << best_score << ")" << std::endl;

  // Generate a straight-ish path from left to right at this Y level
  int n_points = 40;
  double start_x = 0.15 * wx;
  double end_x = 0.85 * wx;

  for (int i = 0; i < n_points; ++i) {
    double t = static_cast<double>(i) / (n_points - 1);
    double x = start_x + t * (end_x - start_x);

    // Slight sinusoidal deviation to test smoothing
    double y = best_y + 1.5 * std::sin(2.0 * M_PI * t);

    // Only add points that are in free space
    double d = map.getDistance(x, y);
    if (d > 0.1) {
      ref_x.push_back(x);
      ref_y.push_back(y);
    }
  }

  if (ref_x.size() < 2) {
    // Fallback: simple straight line
    ref_x.clear();
    ref_y.clear();
    for (int i = 0; i < 30; ++i) {
      double t = static_cast<double>(i) / 29.0;
      ref_x.push_back(0.15 * wx + t * 0.7 * wx);
      ref_y.push_back(best_y);
    }
  }

  std::cout << "Generated reference path with " << ref_x.size() << " points" << std::endl;
}

// ========================================================================
// Main
// ========================================================================
int main(int argc, char ** argv)
{
  std::string map_path = "/home/zks/ws/gits/navigation2/my/maps/occupancy_map.png";
  if (argc > 1) {
    map_path = argv[1];
  }

  std::cout << "=== Ceres 2D Path Smoother with ESDF ===" << std::endl;
  std::cout << "Map: " << map_path << std::endl;

  // --- Load map and compute ESDF ---
  double resolution = 0.05;  // 5 cm/pixel (adjust to match your map)
  std::cout << "Resolution: " << resolution << " m/pixel" << std::endl;

  ESDFMap map(map_path, resolution, 0.0, 0.0, 127);
  std::cout << "Map size: " << map.width() << "x" << map.height()
            << " (" << map.worldWidth() << "x" << map.worldHeight() << " m)" << std::endl;

  // --- Generate test reference path ---
  std::vector<double> ref_x, ref_y;
  generateTestPath(map, ref_x, ref_y);

  // --- Configure smoother ---
  SmootherParams params;
  params.max_iterations = 200;
  params.w_smooth = 100.0;
  params.w_max_curvature = 50.0;
  params.min_turning_radius = 0.5;
  params.w_reference = 10.0;
  params.w_obstacle = 200.0;
  params.safety_margin = 1.0;
  params.verbose = true;

  // --- Smooth ---
  std::cout << "\nSmoothing path..." << std::endl;
  PathSmoother2D smoother(params);
  SmootherResult result = smoother.smooth(ref_x, ref_y, map);

  std::cout << "\n" << result.report << std::endl;
  std::cout << "Solve time: " << result.solve_time_ms << " ms" << std::endl;
  std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;

  // --- Print result ---
  std::cout << "\nSmoothed path (" << result.x.size() << " points):" << std::endl;
  for (size_t i = 0; i < result.x.size(); ++i) {
    printf("  [%3zu] (%.4f, %.4f)  ref=(%.4f, %.4f)  dist=%.4f\n",
      i, result.x[i], result.y[i], ref_x[i], ref_y[i],
      std::sqrt(
        (result.x[i] - ref_x[i]) * (result.x[i] - ref_x[i]) +
        (result.y[i] - ref_y[i]) * (result.y[i] - ref_y[i])));
  }

  // --- Verify obstacle clearance ---
  std::cout << "\nObstacle clearance check:" << std::endl;
  double min_dist = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < result.x.size(); ++i) {
    double d = map.getDistance(result.x[i], result.y[i]);
    min_dist = std::min(min_dist, d);
  }
  std::cout << "  Min clearance: " << min_dist << " m"
            << " (safety_margin=" << params.safety_margin << " m)" << std::endl;

  // --- Save visualization ---
  saveVisualization(
    "smoothed_result.png",
    map.occupancyGrid(), map.width(), map.height(),
    ref_x, ref_y,
    result.x, result.y,
    resolution, 0.0, 0.0);

  return result.success ? 0 : 1;
}
