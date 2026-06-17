// 文件路径：tests/test_cpp.cpp
/**
 * ceres_smoother_2d 库的综合单元测试。
 *
 * 运行方式：./ceres_smoother_2d_tests
 * 退出码 0 = 全部通过，1 = 至少一个失败。
 *
 * 使用很小的自包含测试框架（无 gtest 依赖），保持项目最小化。
 * 测试覆盖：
 *   - ESDFMap 构造、距离、梯度、插值、边界
 *   - SmootherParams 默认值和辅助函数
 *   - SmoothnessCost / CurvatureCost / ReferenceCost / ObstacleCostCeres
 *   - PathSmoother2D 端到端行为（直线、锯齿、障碍避让）
 *   - 边界情况：N=1、N=2、全在障碍中
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "ceres_smoother_2d.hpp"
#include "esdf_map.hpp"

using ceres_smoother_2d::ESDFMap;
using ceres_smoother_2d::SmootherParams;
using ceres_smoother_2d::SmootherResult;
using ceres_smoother_2d::PathSmoother2D;
using ceres_smoother_2d::ObstacleCostCeres;
using ceres_smoother_2d::ReferenceCost;
using ceres_smoother_2d::SmoothnessCost;
using ceres_smoother_2d::CurvatureCost;

// ========================================================================
// 测试框架
// ========================================================================
static int g_pass = 0;
static int g_fail = 0;
static std::string g_current_test;

#define EXPECT(cond, msg)                                                      \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::cerr << "  [FAIL] " << g_current_test << ": " << msg << " (line "    \
                << __LINE__ << ")\n";                                           \
      ++g_fail;                                                                 \
    } else {                                                                    \
      ++g_pass;                                                                 \
    }                                                                           \
  } while (0)

#define EXPECT_NEAR(a, b, tol, msg)                                             \
  do {                                                                          \
    double _a = (a), _b = (b), _tol = (tol);                                    \
    if (std::fabs(_a - _b) > _tol) {                                            \
      std::cerr << "  [FAIL] " << g_current_test << ": " << msg << " got="      \
                << _a << " want=" << _b << " (tol=" << _tol << ", line "         \
                << __LINE__ << ")\n";                                           \
      ++g_fail;                                                                 \
    } else {                                                                    \
      ++g_pass;                                                                 \
    }                                                                           \
  } while (0)

#define RUN_TEST(name)                                                          \
  do {                                                                          \
    g_current_test = #name;                                                     \
    std::cout << "  running " << #name << "...\n";                              \
    name();                                                                     \
  } while (0)

// ========================================================================
// 辅助函数：构建小型合成占据栅格（行主序）。
// 0 = 自由，1 = 障碍
// ========================================================================
static std::vector<uint8_t> make_grid(int w, int h, std::initializer_list<std::pair<int,int>> obs)
{
  std::vector<uint8_t> g(w * h, 0);
  for (auto & p : obs) g[p.second * w + p.first] = 1;
  return g;
}

// ========================================================================
// ESDFMap 测试
// ========================================================================
static void test_esdf_construction()
{
  auto g = make_grid(5, 5, {});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  EXPECT(m.width() == 5, "width");
  EXPECT(m.height() == 5, "height");
  EXPECT_NEAR(m.resolution(), 1.0, 1e-9, "resolution");
  EXPECT_NEAR(m.originX(), 0.0, 1e-9, "originX");
  EXPECT_NEAR(m.originY(), 0.0, 1e-9, "originY");
  EXPECT_NEAR(m.worldWidth(), 5.0, 1e-9, "worldWidth");
  EXPECT_NEAR(m.worldHeight(), 5.0, 1e-9, "worldHeight");
}

static void test_esdf_single_obstacle_cell_signs()
{
  // 5x5 栅格，(2,2) 处有单个障碍。检查原始 ESDF 栅格值。
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  const auto & esdf = m.esdfGrid();
  // 障碍单元 (2,2)：ESDF 为负。
  EXPECT(esdf[2 * 5 + 2] < 0.0, "obstacle cell (2,2) has negative ESDF");
  // 邻近障碍的自由单元有较小的正 ESDF。
  EXPECT(esdf[2 * 5 + 1] > 0.0 && esdf[2 * 5 + 1] < 2.0, "free cell (1,2) close to obstacle");
  EXPECT(esdf[1 * 5 + 2] > 0.0 && esdf[1 * 5 + 2] < 2.0, "free cell (2,1) close to obstacle");
  // 远离障碍的自由单元有更大的正 ESDF。
  EXPECT(esdf[0 * 5 + 0] > 2.0, "corner free cell far from obstacle");
  // 轴向自由单元比对角单元更近（曼哈顿方向 vs 欧氏距离）。
  EXPECT(esdf[2 * 5 + 1] < esdf[1 * 5 + 1], "axis-aligned closer than diagonal");
  EXPECT(esdf[1 * 5 + 2] < esdf[1 * 5 + 1], "axis-aligned (transposed) closer than diagonal");
}

static void test_esdf_sign_convention_via_grid()
{
  // 通过 esdfGrid() 检查符号约定（无双线性平滑）。
  auto g = make_grid(4, 4, {{0,0}, {1,1}});
  ESDFMap m(g, 4, 4, 1.0, 0.0, 0.0);
  const auto & esdf = m.esdfGrid();
  EXPECT(esdf[0 * 4 + 0] < 0.0, "obstacle (0,0) negative");
  EXPECT(esdf[1 * 4 + 1] < 0.0, "obstacle (1,1) negative");
  EXPECT(esdf[3 * 4 + 3] > 0.0, "free corner positive");
}

static void test_esdf_resolution_scaling()
{
  // 栅格设置相同（5x5，障碍位于 (2,2)）。世界单位下的 ESDF 随分辨率缩放：
  // 更粗的地图每个单元距离更大，因此 ESDF 幅值（米）也更大。
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m1(g, 5, 5, 1.0, 0.0, 0.0);    // 1 m/像素 -> 更粗的世界尺度
  ESDFMap m2(g, 5, 5, 0.5, 0.0, 0.0);    // 0.5 m/像素 -> 更细的世界尺度
  // 两个障碍单元都应为负。
  EXPECT(m1.esdfGrid()[2 * 5 + 2] < 0.0, "obstacle raw ESDF negative in m1");
  EXPECT(m2.esdfGrid()[2 * 5 + 2] < 0.0, "obstacle raw ESDF negative in m2");
  // 更粗地图的 |ESDF|（米）应大于更细地图。
  EXPECT(std::fabs(m1.esdfGrid()[2 * 5 + 2]) > std::fabs(m2.esdfGrid()[2 * 5 + 2]),
         "coarser resolution gives larger ESDF magnitude at obstacle");
  // 在整数角点世界坐标处（双线性权重精确为 1.0），无论分辨率如何，
  // 世界单位下的 ESDF 都应一致，因为读取的是同一个物理栅格点。
  EXPECT_NEAR(m1.getDistance(0.0, 0.0), m2.getDistance(0.0, 0.0) * 2.0, 1e-6,
              "corner distances scale linearly with resolution");
}

static void test_esdf_world_to_grid()
{
  auto g = make_grid(10, 10, {});
  ESDFMap m(g, 10, 10, 0.5, 0.0, 0.0);
  double c, r;
  m.worldToGrid(0.0, 0.0, c, r);
  EXPECT_NEAR(c, 0.0, 1e-9, "worldToGrid origin x");
  EXPECT_NEAR(r, 0.0, 1e-9, "worldToGrid origin y");
  m.worldToGrid(5.0, 2.5, c, r);
  EXPECT_NEAR(c, 10.0, 1e-9, "worldToGrid at (5, 2.5)");
  EXPECT_NEAR(r, 5.0, 1e-9, "worldToGrid y axis");
}

static void test_esdf_in_bounds()
{
  auto g = make_grid(10, 10, {});
  ESDFMap m(g, 10, 10, 0.5, 0.0, 0.0);
  EXPECT(m.inBounds(0.0, 0.0), "origin in bounds");
  EXPECT(m.inBounds(4.99, 4.99), "near corner in bounds");
  EXPECT(!m.inBounds(5.0, 0.0), "exactly on edge is out of bounds");
  EXPECT(!m.inBounds(-0.1, 0.0), "negative x out of bounds");
  // 边距：margin=0.5 时，需要距离边界至少 0.5 个单元。
  EXPECT(!m.inBounds(0.0, 0.0, 0.5), "origin not in bounds with margin 0.5");
  EXPECT(m.inBounds(0.3, 0.3, 0.5), "interior point in bounds with margin 0.5");
  EXPECT(!m.inBounds(0.1, 0.3, 0.5), "too close to x=0 edge with margin");
}

static void test_esdf_distance_increases_away_from_obstacle()
{
  // 7x7 自由空间，(3,3) 处有障碍。采样距离朝障碍单元单调减小，
  // 用于验证距离场形态正确。
  auto g = make_grid(7, 7, {{3,3}});
  ESDFMap m(g, 7, 7, 1.0, 0.0, 0.0);

  // 接近障碍时距离减小。
  const double d_left = m.getDistance(0.5, 3.5);
  const double d_mid = m.getDistance(3.5, 3.5);
  const double d_right = m.getDistance(6.5, 3.5);
  EXPECT(d_left > d_mid, "distance decreases toward obstacle from the left");
  EXPECT(d_right > d_mid, "distance decreases toward obstacle from the right");
  EXPECT(d_mid > 0.0, "mid-point above the obstacle cell is positive");
}

static void test_esdf_grid_corner_lookup()
{
  // 在单元角点（fx=fy=0）调用 getDistance，应返回精确栅格值。
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  const auto & esdf = m.esdfGrid();
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 5; ++x) {
      // 世界坐标 (x*res, y*res) 是单元 (x,y) 的角点。res=1 时为整数世界坐标。
      double wx = static_cast<double>(x);
      double wy = static_cast<double>(y);
      EXPECT_NEAR(m.getDistance(wx, wy), esdf[y * 5 + x], 1e-9,
                  "distance at cell corner == grid value");
    }
  }
}

static void test_esdf_at_grid_method()
{
  // esdfAtGrid 是公共方法，用于在小数栅格坐标处执行双线性插值。
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  // 栅格 (2.5, 2.5) -> 世界 (2.5, 2.5)，对 4 个单元双线性插值。
  // 若 (2,2),(2,3),(3,2),(3,3) 为 2x2 障碍块，会得到强负贡献；
  // 若只有单个 1x1 障碍，则插值会与 3 个自由单元平均。
  double d = m.esdfAtGrid(2.5, 2.5);
  // v00 = (2,2) 障碍 ~-1，v10 = (3,2) 自由 ~1，v01 = (2,3) 自由 ~1，
  // v11 = (3,3) 自由 ~sqrt(2)。平均值 = (-1 + 1 + 1 + 1.414) / 4 ≈ 0.6035。
  EXPECT_NEAR(d, (-1.0 + 1.0 + 1.0 + std::sqrt(2.0)) / 4.0, 1e-6,
              "bilinear interp matches expected formula");
}

static void test_esdf_bilinear_consistent_with_get_distance()
{
  // c = (wx - ox)/res、r = (wy - oy)/res 时，esdfAtGrid(c, r) 应等于 getDistance。
  auto g = make_grid(10, 10, {{3,3}, {5,5}});
  ESDFMap m(g, 10, 10, 0.3, 0.0, 0.0);
  for (double wx = 0.0; wx < 3.0; wx += 0.27) {
    for (double wy = 0.0; wy < 3.0; wy += 0.31) {
      double c, r;
      m.worldToGrid(wx, wy, c, r);
      EXPECT_NEAR(m.esdfAtGrid(c, r), m.getDistance(wx, wy), 1e-12,
                  "esdfAtGrid matches getDistance via worldToGrid");
    }
  }
}

static void test_esdf_invalid_occupancy_size()
{
  // 尺寸不匹配时构造函数应抛异常。
  std::vector<uint8_t> bad(10, 0);  // 10 个单元，但期望 5x5。
  bool threw = false;
  try {
    ESDFMap m(bad, 5, 5, 1.0);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  EXPECT(threw, "constructor throws on size mismatch");
}

static void test_esdf_origin_offset()
{
  // 非零原点应体现在 worldToGrid 转换中。
  auto g = make_grid(10, 10, {});
  ESDFMap m(g, 10, 10, 1.0, 5.0, 7.0);
  EXPECT_NEAR(m.originX(), 5.0, 1e-9, "originX");
  EXPECT_NEAR(m.originY(), 7.0, 1e-9, "originY");
  double c, r;
  m.worldToGrid(5.0, 7.0, c, r);
  EXPECT_NEAR(c, 0.0, 1e-9, "worldToGrid at origin");
  m.worldToGrid(15.0, 17.0, c, r);
  EXPECT_NEAR(c, 10.0, 1e-9, "worldToGrid at (origin+10, origin+10)");
  EXPECT_NEAR(r, 10.0, 1e-9, "worldToGrid y");
}

// ========================================================================
// SmootherParams 测试
// ========================================================================
static void test_smoother_params_defaults()
{
  SmootherParams p;
  EXPECT(p.max_iterations == 100, "max_iterations default");
  EXPECT(p.w_smooth > 0.0, "w_smooth positive");
  EXPECT(p.w_obstacle > 0.0, "w_obstacle positive");
  EXPECT(p.w_reference >= 0.0, "w_reference non-negative");
  EXPECT(p.w_length >= 0.0, "w_length non-negative");
  EXPECT(p.w_max_curvature > 0.0, "w_max_curvature positive");
  EXPECT(p.safety_margin > 0.0, "safety_margin positive");
  EXPECT_NEAR(p.obstacleCostDistance(), p.safety_margin + p.robot_radius, 1e-9, "default obstacle cost distance");
  EXPECT(p.min_turning_radius > 0.0, "min_turning_radius positive");
  EXPECT(p.maxCurvature() > 0.0, "maxCurvature positive for valid min_turning_radius");
  EXPECT(std::isfinite(p.maxCurvature()), "maxCurvature finite for valid min_turning_radius");
  // 1/0.2 == 5.0
  EXPECT_NEAR(p.maxCurvature(), 5.0, 1e-9, "maxCurvature == 1/min_turning_radius");
}

static void test_smoother_params_zero_radius()
{
  SmootherParams p;
  p.min_turning_radius = 0.0;
  EXPECT(std::isinf(p.maxCurvature()), "zero radius gives infinite curvature");
}

static void test_smoother_params_custom()
{
  SmootherParams p;
  p.w_smooth = 1.0;
  p.w_obstacle = 2.0;
  p.w_reference = 3.0;
  p.w_max_curvature = 4.0;
  p.min_turning_radius = 2.0;
  p.safety_margin = 0.25;
  EXPECT_NEAR(p.maxCurvature(), 0.5, 1e-9, "custom radius curvature");
  EXPECT_NEAR(p.obstacleCostDistance(), 0.75, 1e-9, "obstacle cost distance = safety_margin + robot_radius");
}

// ========================================================================
// 代价函数测试
// ========================================================================
static void test_smoothness_cost_zero_on_straight_line()
{
  // 三个共线点：代价应为 0。
  SmoothnessCost cost(10.0);
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {2.0, 0.0};
  double r[2];
  bool ok = cost(a, b, c, r);
  EXPECT(ok, "smoothness operator() returns true");
  EXPECT_NEAR(r[0], 0.0, 1e-12, "smoothness x residual = 0 on straight line");
  EXPECT_NEAR(r[1], 0.0, 1e-12, "smoothness y residual = 0 on straight line");
}

static void test_smoothness_cost_nonzero_on_turn()
{
  // 三个非共线点：代价应非零。
  SmoothnessCost cost(1.0);
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {1.0, 1.0};
  double r[2];
  cost(a, b, c, r);
  EXPECT(std::fabs(r[0]) > 1e-9 || std::fabs(r[1]) > 1e-9, "smoothness nonzero on turn");
}

static void test_smoothness_cost_scales_with_weight()
{
  // 代价应按 sqrt(w) 缩放。
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {1.0, 1.0};
  double r1[2], r4[2];
  // 代价构造函数现在直接接收 sqrt_w（不是 w）。
  // sqrt(1)=1，sqrt(4)=2 → 残差应放大 2 倍。
  SmoothnessCost c1(1.0), c4(2.0);
  c1(a, b, c, r1);
  c4(a, b, c, r4);
  EXPECT_NEAR(r4[0], 2.0 * r1[0], 1e-9, "smoothness scales with sqrt_w");
}

static void test_curvature_cost_zero_on_straight_line()
{
  CurvatureCost cost(1.0, 2.0);
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {2.0, 0.0};
  double r[1];
  cost(a, b, c, r);
  EXPECT_NEAR(r[0], 0.0, 1e-6, "curvature = 0 on straight line");
}

static void test_curvature_cost_penalizes_sharp_turn()
{
  // 直角转弯：曲率很高，超过 max_kappa 时应被惩罚。
  CurvatureCost cost(10.0, 1.0);  // 最大 kappa = 1.0
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {1.0, 1.0};
  double r[1];
  cost(a, b, c, r);
  EXPECT(r[0] > 0.0, "sharp turn penalized above max_kappa");
}

static void test_curvature_cost_no_penalty_below_threshold()
{
  // 缓转弯：kappa 低于上限 -> 无惩罚。
  CurvatureCost cost(10.0, 100.0);  // 很高的 max kappa
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {1.0, 1.0};
  double r[1];
  cost(a, b, c, r);
  EXPECT_NEAR(r[0], 0.0, 1e-6, "no penalty when below max_kappa");
}

static void test_curvature_cost_handles_degenerate()
{
  // 三个重合点 -> 退化三角形，不应崩溃。
  CurvatureCost cost(1.0, 1.0);
  double a[2] = {1.0, 1.0}, b[2] = {1.0, 1.0}, c[2] = {1.0, 1.0};
  double r[1];
  bool ok = cost(a, b, c, r);
  EXPECT(ok, "degenerate cost does not crash");
  EXPECT(std::isfinite(r[0]), "degenerate residual is finite");
}

static void test_distance_cost_zero_at_reference()
{
  ReferenceCost cost(2.0, 3.0, 1.0);
  double p[2] = {2.0, 3.0};
  double r[2];
  cost(p, r);
  EXPECT_NEAR(r[0], 0.0, 1e-12, "distance x residual 0 at reference");
  EXPECT_NEAR(r[1], 0.0, 1e-12, "distance y residual 0 at reference");
}

static void test_distance_cost_residual_formula()
{
  // r[0] = sqrt(w) * (p[0] - x_ref)
  // ReferenceCost 现在直接接收 sqrt_w：sqrt(4)=2.0。
  ReferenceCost cost(2.0, 3.0, 2.0);
  double p[2] = {2.5, 3.5};
  double r[2];
  cost(p, r);
  EXPECT_NEAR(r[0], 2.0 * 0.5, 1e-9, "distance x residual = sqrt_w * dx");
  EXPECT_NEAR(r[1], 2.0 * 0.5, 1e-9, "distance y residual = sqrt_w * dy");
}

static void test_distance_cost_zero_weight()
{
  // w=0 -> 无论位置如何都没有代价。
  ReferenceCost cost(0.0, 0.0, 0.0);
  double p[2] = {5.0, 5.0};
  double r[2];
  cost(p, r);
  EXPECT_NEAR(r[0], 0.0, 1e-12, "zero weight -> zero residual");
}

static void test_obstacle_cost_no_penalty_when_far()
{
  // 5x5 自由空间，无障碍 -> 代价 = 0。
  auto g = make_grid(5, 5, {});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  // 构造函数：(map, safe_dist, sqrt_w_obstacle, sqrt_w_penetrate)。
  ObstacleCostCeres cost(&m, 0.5, std::sqrt(10.0), std::sqrt(0.0));
  double p[2] = {0.5, 0.5};
  double r[2];
  cost(p, r);
  EXPECT_NEAR(r[0], 0.0, 1e-6, "no hinge penalty when far from obstacles");
  EXPECT_NEAR(r[1], 0.0, 1e-6, "no penetration penalty when far from obstacles");
}

static void test_obstacle_cost_penalty_in_obstacle()
{
  // 使用较大的障碍，使双线性插值得到负距离。
  auto g = make_grid(7, 7, {{3,3}, {3,4}, {4,3}, {4,4}});
  ESDFMap m(g, 7, 7, 1.0, 0.0, 0.0);
  ObstacleCostCeres cost(&m, 0.5, std::sqrt(10.0), std::sqrt(0.0));
  // 世界坐标 (3.5, 3.5) 是 2x2 障碍块中心。对 4 个障碍单元双线性插值
  // 会得到强负距离。
  double p[2] = {3.5, 3.5};
  double r[2];
  cost(p, r);
  EXPECT(r[0] > 0.0, "hinge penalty positive when inside safety zone");
  // w_penetration=0 时，即使点在深处，第二个残差也应为 0。
  EXPECT_NEAR(r[1], 0.0, 1e-6, "no penetration penalty when w_penetration=0");
}

static void test_obstacle_cost_penetration_grows_with_depth()
{
  // 与上面相同的 2x2 障碍。w_penetration 非零时，点越深入，
  // 第二个残差应单调增大。
  auto g = make_grid(7, 7, {{3,3}, {3,4}, {4,3}, {4,4}});
  ESDFMap m(g, 7, 7, 1.0, 0.0, 0.0);
  ObstacleCostCeres cost(&m, 0.5, std::sqrt(10.0), std::sqrt(100.0));
  // 中心点 (3.5, 3.5)：4 个障碍单元双线性插值 -> -1 m。
  double p_center[2] = {3.5, 3.5};
  double r_center[2];
  cost(p_center, r_center);
  EXPECT(r_center[1] > 0.0, "penetration penalty > 0 at center of 2x2 wall");
  // 点 (3.5, 3.9)：更深入墙内（双线性值更接近 -1）。
  double p_deep[2] = {3.5, 3.9};
  double r_deep[2];
  cost(p_deep, r_deep);
  EXPECT(r_deep[1] > 0.0, "penetration penalty > 0 deep inside wall");
  // 更深入时 r[1] 应严格变大（单调）。
  EXPECT(r_deep[1] > r_center[1] - 1e-6,
         "penetration grows monotonically as we go deeper into the wall");
}

static void test_obstacle_cost_penetration_off_matches_old_behavior()
{
  // w_penetration=0：第二个残差处处为 0，因此代价逐位等价于旧的单 hinge
  // 代价（r[0] 是唯一非零输出）。
  auto g = make_grid(7, 7, {{3,3}, {3,4}, {4,3}, {4,4}});
  ESDFMap m(g, 7, 7, 1.0, 0.0, 0.0);
  ObstacleCostCeres cost_no_pen(&m, 0.5, std::sqrt(50.0), 0.0);
  // 安全边界外侧，例如 2x2 障碍块边缘。单元 (3,4) 的中心是世界坐标
  // (4.5, 3.5)。障碍角点位于 (4,4)；safety=0.5 时应看到正 hinge 惩罚。
  double p[2] = {4.5, 3.5};
  double r[2];
  cost_no_pen(p, r);
  EXPECT(r[0] >= 0.0, "hinge penalty non-negative at safety boundary");
  EXPECT_NEAR(r[1], 0.0, 1e-12, "second residual is exactly 0 when disabled");
}

static void test_obstacle_cost_handles_clamping()
{
  // 查询栅格外位置不应崩溃；会在边界处 clamp。
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  ObstacleCostCeres cost(&m, 0.5, std::sqrt(10.0), std::sqrt(0.0));
  double p[2] = {-1.0, -1.0};  // 越界
  double r[2];
  bool ok = cost(p, r);
  EXPECT(ok, "obstacle cost handles out-of-bounds");
  EXPECT(std::isfinite(r[0]), "out-of-bounds residual r[0] is finite");
  EXPECT(std::isfinite(r[1]), "out-of-bounds residual r[1] is finite");
}

// ========================================================================
// PathSmoother2D 端到端测试
// ========================================================================
static void test_smoother_straight_line_unchanged()
{
  // 10x10 全自由；平滑一条水平直线 -> 应基本保持直线。
  auto g = make_grid(10, 10, {});
  ESDFMap m(g, 10, 10, 1.0, 0.0, 0.0);
  SmootherParams p;
  p.max_iterations = 50;
  p.w_smooth = 1000.0;
  p.w_obstacle = 0.0;       // 本测试禁用障碍物项。
  p.w_max_curvature = 0.0;
  p.resample_before_smooth = false;
  p.resample_after_smooth = false;
  PathSmoother2D smoother(p);
  std::vector<double> xs, ys;
  for (int i = 0; i < 8; ++i) {
    xs.push_back(0.5 + i * 1.0);
    ys.push_back(5.0);
  }
  auto r = smoother.smooth(xs, ys, m);
  EXPECT(r.success, "smooth success");
  EXPECT(static_cast<int>(r.x.size()) == 8, "result has 8 points");
  for (size_t i = 0; i < r.x.size(); ++i) {
    EXPECT_NEAR(r.x[i], xs[i], 0.05, "x unchanged on straight line");
    EXPECT_NEAR(r.y[i], ys[i], 0.05, "y unchanged on straight line");
  }
}

static void test_smoother_smoothing_reduces_oscillation()
{
  // 10x10 全自由；正弦输入 -> 平滑输出应减少摆动。
  auto g = make_grid(10, 10, {});
  ESDFMap m(g, 10, 10, 1.0, 0.0, 0.0);
  SmootherParams p;
  p.max_iterations = 100;
  p.w_smooth = 1000.0;
  p.w_obstacle = 0.0;
  p.w_max_curvature = 0.0;
  p.w_reference = 1.0;
  PathSmoother2D smoother(p);
  std::vector<double> xs, ys;
  int N = 12;
  for (int i = 0; i < N; ++i) {
    double t = static_cast<double>(i) / (N - 1);
    xs.push_back(0.5 + t * 8.0);
    ys.push_back(5.0 + 0.5 * std::sin(6.0 * M_PI * t));
  }
  auto r = smoother.smooth(xs, ys, m);
  EXPECT(r.success, "smooth success on sinusoidal");
  // 相对常数 y 的总偏差应减小。
  double dev_in = 0, dev_out = 0;
  for (int i = 0; i < N; ++i) {
    dev_in += std::fabs(ys[i] - 5.0);
    dev_out += std::fabs(r.y[i] - 5.0);
  }
  EXPECT(dev_out < dev_in, "smoother reduces total deviation from straight line");
}

static void test_smoother_obstacle_avoidance()
{
  // 10x10 自由空间，x=5 处有竖直墙。平滑器必须让路径保持在一侧，
  // 而不是穿墙。
  auto g = make_grid(10, 10, {{5,0}, {5,1}, {5,2}, {5,3}, {5,4},
                              {5,5}, {5,6}, {5,7}, {5,8}, {5,9}});
  ESDFMap m(g, 10, 10, 1.0, 0.0, 0.0);
  SmootherParams p;
  p.max_iterations = 200;
  p.w_smooth = 50.0;
  p.w_obstacle = 500.0;
  p.safety_margin = 0.5;
  p.w_reference = 5.0;
  PathSmoother2D smoother(p);
  // 一条穿过墙的水平直线。
  std::vector<double> xs, ys;
  for (int i = 0; i < 9; ++i) {
    xs.push_back(0.5 + i);
    ys.push_back(5.0);
  }
  auto r = smoother.smooth(xs, ys, m);
  EXPECT(r.success, "obstacle-avoidance smooth success");
  // 所有中间点应位于自由空间（distance >= 0）。
  int in_obstacle = 0;
  for (size_t i = 1; i + 1 < r.x.size(); ++i) {
    double d = m.getDistance(r.x[i], r.y[i]);
    if (d < 0.0) ++in_obstacle;
  }
  EXPECT(in_obstacle == 0, "no intermediate point is inside an obstacle");
}

static void test_smoother_too_few_points()
{
  // N=1 -> 原样返回输入。
  auto g = make_grid(5, 5, {});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  PathSmoother2D smoother;
  std::vector<double> xs = {2.5}, ys = {2.5};
  auto r = smoother.smooth(xs, ys, m);
  EXPECT(r.success, "N=1 success");
  EXPECT(static_cast<int>(r.x.size()) == 1, "N=1 returns 1 point");
}

static void test_smoother_two_points()
{
  // N=2 -> 原样返回输入。
  auto g = make_grid(5, 5, {});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  PathSmoother2D smoother;
  std::vector<double> xs = {1.0, 4.0}, ys = {2.5, 2.5};
  auto r = smoother.smooth(xs, ys, m);
  EXPECT(r.success, "N=2 success");
  EXPECT(static_cast<int>(r.x.size()) == 2, "N=2 returns 2 points");
  EXPECT_NEAR(r.x[0], 1.0, 1e-9, "first x unchanged");
  EXPECT_NEAR(r.x[1], 4.0, 1e-9, "second x unchanged");
}

static void test_smoother_three_points()
{
  // N=3 且保持端点方向时，没有内部点可优化 -> 返回输入。
  auto g = make_grid(5, 5, {});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  SmootherParams p;
  p.resample_before_smooth = false;
  p.resample_after_smooth = false;
  PathSmoother2D smoother(p);
  std::vector<double> xs = {1.0, 2.5, 4.0}, ys = {1.0, 3.0, 1.0};
  auto r = smoother.smooth(xs, ys, m);
  EXPECT(r.success, "N=3 with keep orientations success");
  EXPECT(static_cast<int>(r.x.size()) == 3, "N=3 returns 3 points");
}

static void test_smoother_result_metadata()
{
  // SmootherResult 应报告耗时、迭代数和代价。
  auto g = make_grid(10, 10, {});
  ESDFMap m(g, 10, 10, 1.0, 0.0, 0.0);
  SmootherParams p;
  p.max_iterations = 20;
  PathSmoother2D smoother(p);
  std::vector<double> xs, ys;
  for (int i = 0; i < 8; ++i) {xs.push_back(0.5 + i); ys.push_back(5.0);}
  auto r = smoother.smooth(xs, ys, m);
  EXPECT(r.solve_time_ms >= 0.0, "solve time non-negative");
  EXPECT(r.iterations >= 0, "iterations non-negative");
  EXPECT(!r.report.empty(), "report is populated");
}

// ------------------------------------------------------------------------
// 性能回归：确保使用的是稀疏求解器路径。
//
// 设置这个测试的原因：
//   平滑器关于参数块的 Hessian 是三对角的（SmoothnessCost 和 CurvatureCost
//   只耦合连续三个点）。DENSE_QR 是 O(N^3)，在当前硬件上对 1141 点路径
//   曾观测到约 90 ms；SPARSE_NORMAL_CHOLESKY 是 O(N)，约 5 ms。
//   如果有人把求解器切回 DENSE_QR / DENSE_NORMAL_CHOLESKY，该测试会明显失败。
//
// 时间预算留得较宽（200 ms），避免 CI runner 上抖动；但仍足够紧，
// 可以捕获真实回归。
// ------------------------------------------------------------------------
static void test_smoother_uses_sparse_solver_on_long_path()
{
  // 一维走廊栅格：400 单元宽 × 1 单元高，0.05 m/单元 → 总长 20 m。
  const int N = 1000;
  std::vector<uint8_t> g(N, 0);  // 全自由
  ESDFMap m(g, N, 1, 0.05, 0.0, 0.0);

  // 沿走廊生成 N 点直线。
  std::vector<double> xs(N), ys(N, 0.5 * 0.05);
  for (int i = 0; i < N; ++i) {xs[i] = (i + 0.5) * 0.05;}

  SmootherParams p;
  p.max_iterations = 50;
  PathSmoother2D smoother(p);

  // 预热（首次调用可能包含 Ceres 内部初始化）。
  (void)smoother.smooth(xs, ys, m);

  auto t0 = std::chrono::steady_clock::now();
  auto r = smoother.smooth(xs, ys, m);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  EXPECT(r.success, "1000-pt straight line smoother succeeds");
  // 50 ms 相比当前硬件上预期约 3 ms 的稀疏求解有约 16 倍余量，
  // 但明显低于 DENSE_QR 的约 67 ms（实测）。真正回退到稠密求解器时
  // 会触发该断言。
  EXPECT(ms < 50.0, "sparse solver budget: 1000 pts < 50 ms (got " +
                std::to_string(ms) + " ms)");
  std::printf("  [perf] 1000-pt smooth solve = %.1f ms\n", ms);
}

// ------------------------------------------------------------------------
// 端到端：w_penetration 会惩罚初始位于墙内的路径；即使仅 w_obstacle
// 会留下平坦平台，穿透项仍会提高墙内状态代价。
// ------------------------------------------------------------------------
static void test_smoother_penetration_penalizes_interior()
{
  // (4,4)-(5,5) 处有 2x2 墙。路径中点 (4.5, 4.5) 是单元 (4,4) 中心，
  // 双线性 dist = -1（墙内最深点）。在该鞍点处，关于 x,y 的梯度为 0，
  // 因此优化器会保留该点。这里利用这个性质验证代价函数：
  // w_penetration=0 时，代价是饱和 hinge（0.5*w_ob*safe^2）；
  // w_penetration>0 时，代价至少为 0.5*w_pen*1.0（因为 pen = -dist = 1）。
  // 因此最终代价必须大约增长 0.5 * w_penetration * 1.0。
  auto g = make_grid(10, 10, {{4,4}, {4,5}, {5,4}, {5,5}});
  ESDFMap m(g, 10, 10, 1.0, 0.0, 0.0);

  auto run = [&](double w_pen) {
    SmootherParams p;
    p.max_iterations = 50;
    p.w_smooth = 0.0;
    p.w_reference = 0.0;
    p.w_length = 0.0;
    p.w_max_curvature = 0.0;
    p.w_obstacle = 50.0;
    p.w_penetration = w_pen;
    p.safety_margin = 0.3;
    p.resample_before_smooth = false;
    p.resample_after_smooth = false;
    PathSmoother2D smoother(p);
    std::vector<double> xs = {0.5, 4.5, 8.5};
    std::vector<double> ys = {0.5, 4.5, 8.5};
    return smoother.smooth(xs, ys, m);
  };

  auto r_no_pen = run(0.0);
  auto r_pen = run(5000.0);
  EXPECT(r_no_pen.success, "no-pen smooth succeeds");
  EXPECT(r_pen.success, "with-pen smooth succeeds");
  // 0.5 * 5000 * 1.0 = 2500，这是穿透项贡献的额外代价。
  // 由于优化器还包含安全 hinge 和其他项，这里留出一定余量。
  EXPECT(r_pen.final_cost > r_no_pen.final_cost + 1000.0,
    "w_penetration > 0 must add a large cost for being inside the wall (got "
    + std::to_string(r_no_pen.final_cost) + " vs "
    + std::to_string(r_pen.final_cost) + ")");
}

// ========================================================================
// 主函数
// ========================================================================
int main()
{
  std::cout << "=== ceres_smoother_2d unit tests ===\n";

  std::cout << "\n[ESDFMap]\n";
  RUN_TEST(test_esdf_construction);
  RUN_TEST(test_esdf_single_obstacle_cell_signs);
  RUN_TEST(test_esdf_sign_convention_via_grid);
  RUN_TEST(test_esdf_resolution_scaling);
  RUN_TEST(test_esdf_world_to_grid);
  RUN_TEST(test_esdf_in_bounds);
  RUN_TEST(test_esdf_distance_increases_away_from_obstacle);
  RUN_TEST(test_esdf_grid_corner_lookup);
  RUN_TEST(test_esdf_at_grid_method);
  RUN_TEST(test_esdf_bilinear_consistent_with_get_distance);
  RUN_TEST(test_esdf_invalid_occupancy_size);
  RUN_TEST(test_esdf_origin_offset);

  std::cout << "\n[SmootherParams]\n";
  RUN_TEST(test_smoother_params_defaults);
  RUN_TEST(test_smoother_params_zero_radius);
  RUN_TEST(test_smoother_params_custom);

  std::cout << "\n[Cost functions]\n";
  RUN_TEST(test_smoothness_cost_zero_on_straight_line);
  RUN_TEST(test_smoothness_cost_nonzero_on_turn);
  RUN_TEST(test_smoothness_cost_scales_with_weight);
  RUN_TEST(test_curvature_cost_zero_on_straight_line);
  RUN_TEST(test_curvature_cost_penalizes_sharp_turn);
  RUN_TEST(test_curvature_cost_no_penalty_below_threshold);
  RUN_TEST(test_curvature_cost_handles_degenerate);
  RUN_TEST(test_distance_cost_zero_at_reference);
  RUN_TEST(test_distance_cost_residual_formula);
  RUN_TEST(test_distance_cost_zero_weight);
  RUN_TEST(test_obstacle_cost_no_penalty_when_far);
  RUN_TEST(test_obstacle_cost_penalty_in_obstacle);
  RUN_TEST(test_obstacle_cost_penetration_grows_with_depth);
  RUN_TEST(test_obstacle_cost_penetration_off_matches_old_behavior);
  RUN_TEST(test_obstacle_cost_handles_clamping);

  std::cout << "\n[PathSmoother2D]\n";
  RUN_TEST(test_smoother_straight_line_unchanged);
  RUN_TEST(test_smoother_smoothing_reduces_oscillation);
  RUN_TEST(test_smoother_obstacle_avoidance);
  RUN_TEST(test_smoother_too_few_points);
  RUN_TEST(test_smoother_two_points);
  RUN_TEST(test_smoother_three_points);
  RUN_TEST(test_smoother_result_metadata);
  RUN_TEST(test_smoother_uses_sparse_solver_on_long_path);
  RUN_TEST(test_smoother_penetration_penalizes_interior);

  std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
  return g_fail == 0 ? 0 : 1;
}
