/**
 * Demo：使用 Ceres + ESDF 进行二维路径平滑。
 *
 * 加载占据地图、计算 ESDF、创建参考路径，随后用 Ceres 优化进行平滑，
 * 并保存可视化结果。
 *
 * 用法：./ceres_smoother_2d_demo [path_to_occupancy_map.png]
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "ceres_smoother_2d.hpp"

// stb_image_write 的实现位于 stb_image_impl.cpp。
#include "stb_image_write.h"

using namespace ceres_smoother_2d;

// ========================================================================
// 可视化：在占据地图上绘制路径并保存为 PNG。
// ========================================================================
static void saveVisualization(
  const std::string & filename,
  const std::vector<uint8_t> & occupancy,
  int map_w, int map_h,
  const std::vector<double> & ref_x, const std::vector<double> & ref_y,
  const std::vector<double> & smooth_x, const std::vector<double> & smooth_y,
  double res, double ox, double oy)
{
  // 创建 RGB 图像（3 通道）。
  std::vector<uint8_t> img(map_w * map_h * 3);

  // 绘制占据地图（灰度 -> RGB）。
  for (int i = 0; i < map_w * map_h; ++i) {
    uint8_t v = occupancy[i];
    img[i * 3 + 0] = v;
    img[i * 3 + 1] = v;
    img[i * 3 + 2] = v;
  }

  // 辅助函数：在图像上绘制像素。
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

  // 绘制相邻路径点之间的线段。
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

  // 绘制参考路径（蓝色）。
  drawLine(ref_x, ref_y, 0, 100, 255, 0);

  // 绘制平滑路径（绿色）。
  drawLine(smooth_x, smooth_y, 0, 255, 0, 1);

  // 绘制平滑路径的起点（绿色圆）和终点（红色圆）。
  if (!smooth_x.empty()) {
    drawPixel(smooth_x.front(), smooth_y.front(), 0, 255, 0, 3);
    drawPixel(smooth_x.back(), smooth_y.back(), 255, 0, 0, 3);
  }

  // 保存 PNG。
  if (!stbi_write_png(filename.c_str(), map_w, map_h, 3, img.data(), map_w * 3)) {
    std::cerr << "Failed to write " << filename << std::endl;
  } else {
    std::cout << "Saved visualization: " << filename << std::endl;
  }
}

// ========================================================================
// 生成测试参考路径：在自由空间底部附近从左到右的近似直线。
// ========================================================================
static void generateTestPath(
  const ESDFMap & map,
  std::vector<double> & ref_x,
  std::vector<double> & ref_y)
{
  double wx = map.worldWidth();
  double wy = map.worldHeight();

  // 从底部开始扫描，在自由空间中寻找合适的 Y 层。
  // 目标是找到大部分为自由空间的行（ESDF 值较高）。
  double best_y = wy * 0.9;  // 默认：靠近底部
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

  // 在该 Y 层生成一条从左到右的近似直线路径。
  int n_points = 40;
  double start_x = 0.15 * wx;
  double end_x = 0.85 * wx;

  for (int i = 0; i < n_points; ++i) {
    double t = static_cast<double>(i) / (n_points - 1);
    double x = start_x + t * (end_x - start_x);

    // 加入轻微正弦扰动，用于测试平滑效果。
    double y = best_y + 1.5 * std::sin(2.0 * M_PI * t);

    // 只添加位于自由空间的点。
    double d = map.getDistance(x, y);
    if (d > 0.1) {
      ref_x.push_back(x);
      ref_y.push_back(y);
    }
  }

  if (ref_x.size() < 2) {
    // 回退方案：简单直线。
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
// 主函数
// ========================================================================
int main(int argc, char ** argv)
{
  std::string map_path = "/home/zks/ws/gits/navigation2/my/maps/occupancy_map.png";
  if (argc > 1) {
    map_path = argv[1];
  }

  std::cout << "=== Ceres 2D Path Smoother with ESDF ===" << std::endl;
  std::cout << "Map: " << map_path << std::endl;

  // --- 加载地图并计算 ESDF ---
  double resolution = 0.05;  // 5 cm/像素（按实际地图调整）
  std::cout << "Resolution: " << resolution << " m/pixel" << std::endl;

  ESDFMap map(map_path, resolution, 0.0, 0.0, 127);
  std::cout << "Map size: " << map.width() << "x" << map.height()
            << " (" << map.worldWidth() << "x" << map.worldHeight() << " m)" << std::endl;

  // --- 生成测试参考路径 ---
  std::vector<double> ref_x, ref_y;
  generateTestPath(map, ref_x, ref_y);

  // --- 配置平滑器 ---
  SmootherParams params;
  params.max_iterations = 200;
  params.w_smooth = 100.0;
  params.w_max_curvature = 50.0;
  params.min_turning_radius = 0.5;
  params.w_reference = 10.0;
  params.w_obstacle = 200.0;
  params.safety_margin = 1.0;
  params.verbose = true;

  // --- 执行平滑 ---
  std::cout << "\nSmoothing path..." << std::endl;
  PathSmoother2D smoother(params);
  SmootherResult result = smoother.smooth(ref_x, ref_y, map);

  std::cout << "\n" << result.report << std::endl;
  std::cout << "Solve time: " << result.solve_time_ms << " ms" << std::endl;
  std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;

  // --- 打印结果 ---
  std::cout << "\nSmoothed path (" << result.x.size() << " points):" << std::endl;
  for (size_t i = 0; i < result.x.size(); ++i) {
    printf("  [%3zu] (%.4f, %.4f)  ref=(%.4f, %.4f)  dist=%.4f\n",
      i, result.x[i], result.y[i], ref_x[i], ref_y[i],
      std::sqrt(
        (result.x[i] - ref_x[i]) * (result.x[i] - ref_x[i]) +
        (result.y[i] - ref_y[i]) * (result.y[i] - ref_y[i])));
  }

  // --- 验证障碍物间隙 ---
  std::cout << "\nObstacle clearance check:" << std::endl;
  double min_dist = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < result.x.size(); ++i) {
    double d = map.getDistance(result.x[i], result.y[i]);
    min_dist = std::min(min_dist, d);
  }
  std::cout << "  Min clearance: " << min_dist << " m"
            << " (safety_margin=" << params.safety_margin << " m)" << std::endl;

  // --- 保存可视化结果 ---
  saveVisualization(
    "smoothed_result.png",
    map.occupancyGrid(), map.width(), map.height(),
    ref_x, ref_y,
    result.x, result.y,
    resolution, 0.0, 0.0);

  return result.success ? 0 : 1;
}
