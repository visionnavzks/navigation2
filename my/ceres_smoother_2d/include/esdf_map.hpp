#pragma once

/**
 * 用于二维路径平滑的 ESDF（欧氏有符号距离场）地图。
 *
 * 加载占据栅格并计算：
 *   - 有符号距离：障碍外（自由空间）为正，障碍内为负。
 *
 * 算法：Felzenszwalb & Huttenlocher 的
 * "Distance Transforms of Sampled Functions"（2012），对每行/列用 O(n)
 * 时间计算精确欧氏距离。
 *
 * 无 ROS 依赖。只使用 stb_image 和 C++ 标准库。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// 全程使用双线性插值（不是双三次）：ESDF 在障碍边界处有尖锐不连续，
// 双三次会跨边界过冲。双线性始终受最近 4 个单元的 min/max 约束，
// 且可用于 Jet 自动微分，因此结果正确并兼容 Ceres。

// stb_image 的实现位于 src/stb_image_impl.cpp。
#include "stb_image.h"

namespace ceres_smoother_2d
{

class ESDFMap
{
public:
  /**
   * @param occupancy_path  灰度 PNG 路径（0=障碍，255=自由）。
   * @param resolution      每像素对应的米数。
   * @param origin_x        栅格像素 (0,0) 的世界 x 坐标。
   * @param origin_y        栅格像素 (0,0) 的世界 y 坐标。
   * @param obstacle_thresh 阈值：像素值 <= 该值视为障碍。
   */
  ESDFMap(
    const std::string & occupancy_path,
    double resolution,
    double origin_x = 0.0,
    double origin_y = 0.0,
    int obstacle_thresh = 127)
  : resolution_(resolution), origin_x_(origin_x), origin_y_(origin_y)
  {
    // 加载灰度图。
    int w, h, channels;
    unsigned char * img = stbi_load(occupancy_path.c_str(), &w, &h, &channels, 1);
    if (!img) {
      throw std::runtime_error("Failed to load image: " + occupancy_path);
    }

    width_ = w;
    height_ = h;
    grid_.resize(height_ * width_);
    // 垂直翻转，使 PNG 第 0 行（视觉顶部）映射到世界 y = y_max，
    // 与 ROS map_server 约定和 Plotly 的 `yanchor: 'top'` 布局一致。
    // 翻转后：栅格第 r 行对应 PNG 第 (height-1-r) 行，因此：
    //   world y = r * resolution_  等价于  PNG row = height-1-r
    for (int r = 0; r < height_; ++r) {
      const int src_off = r * width_;
      const int dst_off = (height_ - 1 - r) * width_;
      for (int c = 0; c < width_; ++c) {
        // 典型占据图中：0（黑）= 障碍，255（白）= 自由。
        // 像素值低于 obstacle_thresh 时视为障碍。
        grid_[dst_off + c] = (img[src_off + c] <= obstacle_thresh) ? 1 : 0;
      }
    }
    stbi_image_free(img);

    computeESDF();
  }

  /**
   * 从原始占据数据构造（0=自由，1=障碍）。
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

  // --- 访问器 ---

  int width() const {return width_;}
  int height() const {return height_;}
  double resolution() const {return resolution_;}
  double originX() const {return origin_x_;}
  double originY() const {return origin_y_;}

  double worldWidth() const {return width_ * resolution_;}
  double worldHeight() const {return height_ * resolution_;}

  /** 获取世界坐标处的有符号距离（双线性插值）。
   *  使用双线性（而非双三次）是因为 ESDF 在障碍边界处有尖锐不连续；
   *  双三次会跨边界过冲并返回错误值。双线性受最近 4 个单元的 min/max 约束。 */
  double getDistance(double wx, double wy) const
  {
    return bilinearJet<double>(wx, wy);
  }

  /**
   * 获取小数栅格坐标处的有符号距离场值。
   * 使用与 getDistance() 一致的双线性插值。
   */
  double esdfAtGrid(double col, double row) const
  {
    // 栅格空间 (col, row) -> 世界坐标。
    return bilinearJet<double>(col * resolution_ + origin_x_,
                                row * resolution_ + origin_y_);
  }

  /** 支持 Jet 的双线性 ESDF 查询。返回给定世界坐标处的距离，
   *  同时保留关于 `wx` 和 `wy` 的偏导。
   *  Ceres 代价函数使用它进行 AutoDiff。双线性是 C^0（仅连续，
   *  在单元边界不可微），但始终受邻居 min/max 约束，不会像双三次那样过冲。
   *  注意：PNG 输入行在加载时会翻转一次，使内部栅格行索引与世界 y 栅格
   *  行索引一致。原始 occupancy 输入应已使用相同的内部约定。 */
  template<typename T>
  T bilinearJet(T wx, T wy) const
  {
    if (width_ <= 0 || height_ <= 0) {return T(0.0);}
    const T col = (wx - T(origin_x_)) / T(resolution_);
    const T row = (wy - T(origin_y_)) / T(resolution_);
    const T r_arr = ceres::fmax(T(0.0), ceres::fmin(T(height_ - 1), row));
    const T c = ceres::fmax(T(0.0), ceres::fmin(T(width_ - 1), col));
    // 整数索引：提取 Jet 的标量部分，避免与 ceres::Jet 成员发生 ADL 混淆。
    // Jet 使用 `r.a`，double 直接使用 `r`。
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
   * 将世界坐标转换为小数栅格坐标。
   */
  void worldToGrid(double wx, double wy, double & col, double & row) const
  {
    col = (wx - origin_x_) / resolution_;
    row = (wy - origin_y_) / resolution_;
  }

  /** 检查世界坐标点是否位于地图边界内。 */
  bool inBounds(double wx, double wy, double margin = 0.0) const
  {
    // 在世界坐标中比较，避免精确栅格边界处的浮点精度问题
    // （例如 world_width / resolution 可能不是精确值）。
    const double min_x = origin_x_ + margin * resolution_;
    const double min_y = origin_y_ + margin * resolution_;
    const double max_x = origin_x_ + (width_  - margin) * resolution_;
    const double max_y = origin_y_ + (height_ - margin) * resolution_;
    return wx >= min_x && wx < max_x &&
           wy >= min_y && wy < max_y;
  }

  /** 获取底层 ESDF 栅格，按内部/世界 y 行主序排列。 */
  const std::vector<double> & esdfGrid() const {return esdf_;}

  const std::vector<uint8_t> & occupancyGrid() const {return grid_;}

private:
  int width_, height_;
  double resolution_, origin_x_, origin_y_;
  std::vector<uint8_t> grid_;
  std::vector<double> esdf_;

  // ====================================================================
  // 一维平方欧氏距离变换
  // Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled Functions"
  // 输入：f[q]（代价函数），输出：d[q]（到最小值的平方距离）
  // ====================================================================
  static void distTransform1D(const double * f, double * d, int n)
  {
    if (n == 0) {return;}

    std::vector<int> v(n);   // 抛物线位置
    std::vector<double> z(n + 1);  // 抛物线之间的边界
    int k = 0;
    v[0] = 0;
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] = std::numeric_limits<double>::infinity();

    auto sq = [](double x) {return x * x;};

    for (int q = 1; q < n; ++q) {
      // q 对应抛物线与 v[k] 对应抛物线的交点。
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
  // 二维平方欧氏距离变换
  // ====================================================================
  static std::vector<double> edt2d(const std::vector<uint8_t> & binary, int w, int h)
  {
    const double INF = 1e20;
    std::vector<double> d(w * h);

    // 初始化：障碍=0，自由=INF。
    for (int i = 0; i < w * h; ++i) {
      d[i] = binary[i] ? 0.0 : INF;
    }

    // 列方向变换（每列有 h 个元素）。
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

    // 行方向变换（每行有 w 个元素）。
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
  // 计算有符号欧氏距离场
  // 正值 = 自由空间（障碍外），负值 = 障碍内
  // ====================================================================
  void computeESDF()
  {
    esdf_.resize(width_ * height_);

    // 计算自由空间到最近障碍的距离：d_free[y*w+x]。
    // 自由像素：到最近障碍的平方距离。
    // 障碍像素：0。
    auto d_free = edt2d(grid_, width_, height_);

    // 计算障碍到最近自由空间的距离：d_obst[y*w+x]。
    // 障碍像素：到最近自由像素的平方距离。
    // 自由像素：0。
    std::vector<uint8_t> inv_grid(width_ * height_);
    for (int i = 0; i < width_ * height_; ++i) {
      inv_grid[i] = grid_[i] ? 0 : 1;
    }
    auto d_obst = edt2d(inv_grid, width_, height_);

    // 有符号距离：自由为正，障碍为负。
    for (int i = 0; i < width_ * height_; ++i) {
      if (grid_[i]) {
        // 障碍像素：到最近自由空间的负距离。
        esdf_[i] = -std::sqrt(d_obst[i]) * resolution_;
      } else {
        // 自由像素：到最近障碍的正距离。
        esdf_[i] = std::sqrt(d_free[i]) * resolution_;
      }
    }
  }

};

}  // namespace ceres_smoother_2d
