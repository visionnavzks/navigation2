// filepath: tests/test_cpp.cpp
/**
 * Comprehensive unit tests for ceres_smoother_2d library.
 *
 * Run via: ./ceres_smoother_2d_tests
 * Exit code 0 = all pass, 1 = at least one failure.
 *
 * Uses a tiny self-contained test harness (no gtest dependency) to keep
 * the project minimal. The tests cover:
 *   - ESDFMap construction, distance, gradient, interpolation, bounds
 *   - SmootherParams defaults and helpers
 *   - SmoothnessCost / CurvatureCost / ReferenceCost / ObstacleCostCeres
 *   - PathSmoother2D end-to-end (straight line, zigzag, obstacle avoidance)
 *   - Edge cases: N=1, N=2, all-in-obstacle
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
// Test harness
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
// Helper: build a small synthetic occupancy grid (row-major).
// 0 = free, 1 = obstacle
// ========================================================================
static std::vector<uint8_t> make_grid(int w, int h, std::initializer_list<std::pair<int,int>> obs)
{
  std::vector<uint8_t> g(w * h, 0);
  for (auto & p : obs) g[p.second * w + p.first] = 1;
  return g;
}

// ========================================================================
// ESDFMap tests
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
  // 5x5 with single obstacle at (2,2). Inspect raw ESDF grid values.
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  const auto & esdf = m.esdfGrid();
  // Obstacle cell (2,2): ESDF is negative
  EXPECT(esdf[2 * 5 + 2] < 0.0, "obstacle cell (2,2) has negative ESDF");
  // Free cells adjacent to obstacle have small positive ESDF
  EXPECT(esdf[2 * 5 + 1] > 0.0 && esdf[2 * 5 + 1] < 2.0, "free cell (1,2) close to obstacle");
  EXPECT(esdf[1 * 5 + 2] > 0.0 && esdf[1 * 5 + 2] < 2.0, "free cell (2,1) close to obstacle");
  // Free cells far from obstacle have larger positive ESDF
  EXPECT(esdf[0 * 5 + 0] > 2.0, "corner free cell far from obstacle");
  // Axis-aligned free cell is closer than diagonal (Manhattan vs Euclidean)
  EXPECT(esdf[2 * 5 + 1] < esdf[1 * 5 + 1], "axis-aligned closer than diagonal");
  EXPECT(esdf[1 * 5 + 2] < esdf[1 * 5 + 1], "axis-aligned (transposed) closer than diagonal");
}

static void test_esdf_sign_convention_via_grid()
{
  // Sign convention check via esdfGrid() (no bilinear smoothing)
  auto g = make_grid(4, 4, {{0,0}, {1,1}});
  ESDFMap m(g, 4, 4, 1.0, 0.0, 0.0);
  const auto & esdf = m.esdfGrid();
  EXPECT(esdf[0 * 4 + 0] < 0.0, "obstacle (0,0) negative");
  EXPECT(esdf[1 * 4 + 1] < 0.0, "obstacle (1,1) negative");
  EXPECT(esdf[3 * 4 + 3] > 0.0, "free corner positive");
}

static void test_esdf_resolution_scaling()
{
  // The grid setup is the same (5x5 with obstacle at (2,2)). The ESDF in world
  // units scales with resolution: a coarser map has larger per-cell distances
  // and so its ESDF magnitudes (in metres) are larger.
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m1(g, 5, 5, 1.0, 0.0, 0.0);    // 1 m/pixel -> coarser world
  ESDFMap m2(g, 5, 5, 0.5, 0.0, 0.0);    // 0.5 m/pixel -> finer world
  // Both obstacle cells should be negative
  EXPECT(m1.esdfGrid()[2 * 5 + 2] < 0.0, "obstacle raw ESDF negative in m1");
  EXPECT(m2.esdfGrid()[2 * 5 + 2] < 0.0, "obstacle raw ESDF negative in m2");
  // Coarser map has larger |ESDF| (in metres) than finer map
  EXPECT(std::fabs(m1.esdfGrid()[2 * 5 + 2]) > std::fabs(m2.esdfGrid()[2 * 5 + 2]),
         "coarser resolution gives larger ESDF magnitude at obstacle");
  // At an integer-corner world point (where bilinear weights are exactly 1.0)
  // the ESDF in world units must be identical regardless of resolution,
  // because we're reading the same physical grid.
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
  // Margin: with margin=0.5, need to be at least 0.5 cells from edge
  EXPECT(!m.inBounds(0.0, 0.0, 0.5), "origin not in bounds with margin 0.5");
  EXPECT(m.inBounds(0.3, 0.3, 0.5), "interior point in bounds with margin 0.5");
  EXPECT(!m.inBounds(0.1, 0.3, 0.5), "too close to x=0 edge with margin");
}

static void test_esdf_distance_increases_away_from_obstacle()
{
  // 7x7 free, obstacle at (3,3). Sampled distance decreases monotonically
  // toward the obstacle cell — verifies the field is well-formed.
  auto g = make_grid(7, 7, {{3,3}});
  ESDFMap m(g, 7, 7, 1.0, 0.0, 0.0);

  // Distance decreases as we approach the obstacle.
  const double d_left = m.getDistance(0.5, 3.5);
  const double d_mid = m.getDistance(3.5, 3.5);
  const double d_right = m.getDistance(6.5, 3.5);
  EXPECT(d_left > d_mid, "distance decreases toward obstacle from the left");
  EXPECT(d_right > d_mid, "distance decreases toward obstacle from the right");
  EXPECT(d_mid > 0.0, "mid-point above the obstacle cell is positive");
}

static void test_esdf_grid_corner_lookup()
{
  // getDistance at the corner of a cell (where fx=fy=0) returns the exact grid value.
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  const auto & esdf = m.esdfGrid();
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 5; ++x) {
      // World (x*res, y*res) is the corner of cell (x,y). With res=1, integer world coords.
      double wx = static_cast<double>(x);
      double wy = static_cast<double>(y);
      EXPECT_NEAR(m.getDistance(wx, wy), esdf[y * 5 + x], 1e-9,
                  "distance at cell corner == grid value");
    }
  }
}

static void test_esdf_at_grid_method()
{
  // esdfAtGrid is a public method that does bilinear interp at fractional grid coords.
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  // At grid (2.5, 2.5) -> world (2.5, 2.5), bilinear interp of 4 cells.
  // With a 2x2 obstacle block at (2,2),(2,3),(3,2),(3,3) we get a strong negative
  // contribution. With a single 1x1 obstacle, the interp averages with 3 free cells.
  double d = m.esdfAtGrid(2.5, 2.5);
  // v00 = (2,2) obstacle ~-1, v10 = (3,2) free ~1, v01 = (2,3) free ~1,
  // v11 = (3,3) free ~sqrt(2). Average = (-1 + 1 + 1 + 1.414) / 4 ≈ 0.6035
  EXPECT_NEAR(d, (-1.0 + 1.0 + 1.0 + std::sqrt(2.0)) / 4.0, 1e-6,
              "bilinear interp matches expected formula");
}

static void test_esdf_bilinear_consistent_with_get_distance()
{
  // esdfAtGrid(c, r) with c = (wx - ox)/res, r = (wy - oy)/res should equal getDistance.
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
  // Constructor should throw on size mismatch
  std::vector<uint8_t> bad(10, 0);  // 10 cells but 5x5 expected
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
  // Non-zero origin should be reflected in worldToGrid
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
// SmootherParams tests
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
// Cost function tests
// ========================================================================
static void test_smoothness_cost_zero_on_straight_line()
{
  // Three collinear points: cost should be 0
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
  // Three non-collinear points: cost should be nonzero
  SmoothnessCost cost(1.0);
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {1.0, 1.0};
  double r[2];
  cost(a, b, c, r);
  EXPECT(std::fabs(r[0]) > 1e-9 || std::fabs(r[1]) > 1e-9, "smoothness nonzero on turn");
}

static void test_smoothness_cost_scales_with_weight()
{
  // Cost should scale as sqrt(w)
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {1.0, 1.0};
  double r1[2], r4[2];
  // Cost constructors now accept sqrt_w directly (not w).
  // sqrt(1)=1, sqrt(4)=2  → residual should scale by 2.
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
  // Right-angle turn: very high curvature, should be penalized when above max_kappa
  CurvatureCost cost(10.0, 1.0);  // max kappa = 1.0
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {1.0, 1.0};
  double r[1];
  cost(a, b, c, r);
  EXPECT(r[0] > 0.0, "sharp turn penalized above max_kappa");
}

static void test_curvature_cost_no_penalty_below_threshold()
{
  // Gentle turn: kappa below max -> no penalty
  CurvatureCost cost(10.0, 100.0);  // very high max kappa
  double a[2] = {0.0, 0.0}, b[2] = {1.0, 0.0}, c[2] = {1.0, 1.0};
  double r[1];
  cost(a, b, c, r);
  EXPECT_NEAR(r[0], 0.0, 1e-6, "no penalty when below max_kappa");
}

static void test_curvature_cost_handles_degenerate()
{
  // Three coincident points -> degenerate triangle, should not crash
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
  // ReferenceCost now takes sqrt_w directly: sqrt(4)=2.0
  ReferenceCost cost(2.0, 3.0, 2.0);
  double p[2] = {2.5, 3.5};
  double r[2];
  cost(p, r);
  EXPECT_NEAR(r[0], 2.0 * 0.5, 1e-9, "distance x residual = sqrt_w * dx");
  EXPECT_NEAR(r[1], 2.0 * 0.5, 1e-9, "distance y residual = sqrt_w * dy");
}

static void test_distance_cost_zero_weight()
{
  // w=0 -> no cost regardless of position
  ReferenceCost cost(0.0, 0.0, 0.0);
  double p[2] = {5.0, 5.0};
  double r[2];
  cost(p, r);
  EXPECT_NEAR(r[0], 0.0, 1e-12, "zero weight -> zero residual");
}

static void test_obstacle_cost_no_penalty_when_far()
{
  // 5x5 free, no obstacles -> cost = 0
  auto g = make_grid(5, 5, {});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  // Constructor: (map, safe_dist, sqrt_w_obstacle, sqrt_w_penetrate)
  ObstacleCostCeres cost(&m, 0.5, std::sqrt(10.0), std::sqrt(0.0));
  double p[2] = {0.5, 0.5};
  double r[2];
  cost(p, r);
  EXPECT_NEAR(r[0], 0.0, 1e-6, "no hinge penalty when far from obstacles");
  EXPECT_NEAR(r[1], 0.0, 1e-6, "no penetration penalty when far from obstacles");
}

static void test_obstacle_cost_penalty_in_obstacle()
{
  // Larger obstacle so bilinear interp yields negative distance
  auto g = make_grid(7, 7, {{3,3}, {3,4}, {4,3}, {4,4}});
  ESDFMap m(g, 7, 7, 1.0, 0.0, 0.0);
  ObstacleCostCeres cost(&m, 0.5, std::sqrt(10.0), std::sqrt(0.0));
  // At world (3.5, 3.5), center of 2x2 obstacle block. Bilinear interp of
  // 4 obstacle cells -> strong negative distance.
  double p[2] = {3.5, 3.5};
  double r[2];
  cost(p, r);
  EXPECT(r[0] > 0.0, "hinge penalty positive when inside safety zone");
  // With w_penetration=0, the second residual should be 0 even deep inside.
  EXPECT_NEAR(r[1], 0.0, 1e-6, "no penetration penalty when w_penetration=0");
}

static void test_obstacle_cost_penetration_grows_with_depth()
{
  // Same 2x2 obstacle as above. With nonzero w_penetration, the second
  // residual should grow monotonically as the point goes deeper.
  auto g = make_grid(7, 7, {{3,3}, {3,4}, {4,3}, {4,4}});
  ESDFMap m(g, 7, 7, 1.0, 0.0, 0.0);
  ObstacleCostCeres cost(&m, 0.5, std::sqrt(10.0), std::sqrt(100.0));
  // At center (3.5, 3.5): bilinear over 4 obstacle cells -> -1 m
  double p_center[2] = {3.5, 3.5};
  double r_center[2];
  cost(p_center, r_center);
  EXPECT(r_center[1] > 0.0, "penetration penalty > 0 at center of 2x2 wall");
  // At (3.5, 3.9): more inside the wall (bilinear closer to -1)
  double p_deep[2] = {3.5, 3.9};
  double r_deep[2];
  cost(p_deep, r_deep);
  EXPECT(r_deep[1] > 0.0, "penetration penalty > 0 deep inside wall");
  // Going deeper should make r[1] strictly larger (monotone).
  EXPECT(r_deep[1] > r_center[1] - 1e-6,
         "penetration grows monotonically as we go deeper into the wall");
}

static void test_obstacle_cost_penetration_off_matches_old_behavior()
{
  // w_penetration=0: the second residual is identically 0 everywhere, so
  // the cost is bit-for-bit equivalent to the old single-hinge cost
  // (r[0] is the only nonzero output).
  auto g = make_grid(7, 7, {{3,3}, {3,4}, {4,3}, {4,4}});
  ESDFMap m(g, 7, 7, 1.0, 0.0, 0.0);
  ObstacleCostCeres cost_no_pen(&m, 0.5, std::sqrt(50.0), 0.0);
  // Just outside the safety boundary, e.g. at the edge of the 2x2 block.
  // Cell (3,4) center is world (4.5, 3.5). The corner of the obstacle is
  // at (4, 4); with safety=0.5 we should see a positive hinge penalty.
  double p[2] = {4.5, 3.5};
  double r[2];
  cost_no_pen(p, r);
  EXPECT(r[0] >= 0.0, "hinge penalty non-negative at safety boundary");
  EXPECT_NEAR(r[1], 0.0, 1e-12, "second residual is exactly 0 when disabled");
}

static void test_obstacle_cost_handles_clamping()
{
  // Querying outside grid should not crash; clamping at boundaries
  auto g = make_grid(5, 5, {{2,2}});
  ESDFMap m(g, 5, 5, 1.0, 0.0, 0.0);
  ObstacleCostCeres cost(&m, 0.5, std::sqrt(10.0), std::sqrt(0.0));
  double p[2] = {-1.0, -1.0};  // out of bounds
  double r[2];
  bool ok = cost(p, r);
  EXPECT(ok, "obstacle cost handles out-of-bounds");
  EXPECT(std::isfinite(r[0]), "out-of-bounds residual r[0] is finite");
  EXPECT(std::isfinite(r[1]), "out-of-bounds residual r[1] is finite");
}

// ========================================================================
// PathSmoother2D end-to-end tests
// ========================================================================
static void test_smoother_straight_line_unchanged()
{
  // 10x10 all free; smooth a straight horizontal line -> should remain nearly straight
  auto g = make_grid(10, 10, {});
  ESDFMap m(g, 10, 10, 1.0, 0.0, 0.0);
  SmootherParams p;
  p.max_iterations = 50;
  p.w_smooth = 1000.0;
  p.w_obstacle = 0.0;       // disable obstacle for this test
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
  // 10x10 all free; sinusoidal input -> smoothed output should be less wiggly
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
  // Total y deviation from constant should be reduced
  double dev_in = 0, dev_out = 0;
  for (int i = 0; i < N; ++i) {
    dev_in += std::fabs(ys[i] - 5.0);
    dev_out += std::fabs(r.y[i] - 5.0);
  }
  EXPECT(dev_out < dev_in, "smoother reduces total deviation from straight line");
}

static void test_smoother_obstacle_avoidance()
{
  // 10x10 free with a vertical wall at x=5. The smoother must keep the
  // path on one side rather than crossing through.
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
  // Straight horizontal line that crosses the wall
  std::vector<double> xs, ys;
  for (int i = 0; i < 9; ++i) {
    xs.push_back(0.5 + i);
    ys.push_back(5.0);
  }
  auto r = smoother.smooth(xs, ys, m);
  EXPECT(r.success, "obstacle-avoidance smooth success");
  // All intermediate points should be in free space (distance >= 0)
  int in_obstacle = 0;
  for (size_t i = 1; i + 1 < r.x.size(); ++i) {
    double d = m.getDistance(r.x[i], r.y[i]);
    if (d < 0.0) ++in_obstacle;
  }
  EXPECT(in_obstacle == 0, "no intermediate point is inside an obstacle");
}

static void test_smoother_too_few_points()
{
  // N=1 -> returns input unchanged
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
  // N=2 -> returns input unchanged
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
  // N=3 with keep_*_orientation -> no interior points to optimize -> returns input
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
  // SmootherResult should report timing, iteration count, cost
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
// Performance regression: ensure the sparse solver path is being used.
//
// Why this exists:
//   The smoother's Hessian is *tridiagonal* in the parameter blocks
//   (SmoothnessCost and CurvatureCost only couple three consecutive
//   points). DENSE_QR is O(N^3) and was observed to take ~90 ms on a
//   1141-point path on this hardware; SPARSE_NORMAL_CHOLESKY is O(N)
//   and takes ~5 ms. This test fails loudly if anyone flips the solver
//   back to DENSE_QR / DENSE_NORMAL_CHOLESKY.
//
// We keep the budget generous (200 ms) so the test isn't flaky on CI
// runners, but tight enough that a real regression is caught.
// ------------------------------------------------------------------------
static void test_smoother_uses_sparse_solver_on_long_path()
{
  // 1D corridor grid: 400 cells wide × 1 cell tall, 0.05 m/cell → 20 m long
  const int N = 1000;
  std::vector<uint8_t> g(N, 0);  // all free
  ESDFMap m(g, N, 1, 0.05, 0.0, 0.0);

  // Straight line of N points along the corridor
  std::vector<double> xs(N), ys(N, 0.5 * 0.05);
  for (int i = 0; i < N; ++i) {xs[i] = (i + 0.5) * 0.05;}

  SmootherParams p;
  p.max_iterations = 50;
  PathSmoother2D smoother(p);

  // Warm-up (first call may include Ceres internal init)
  (void)smoother.smooth(xs, ys, m);

  auto t0 = std::chrono::steady_clock::now();
  auto r = smoother.smooth(xs, ys, m);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  EXPECT(r.success, "1000-pt straight line smoother succeeds");
  // 50 ms gives ~16x headroom over the expected ~3 ms sparse solve on
  // this hardware, but is well below the ~67 ms that DENSE_QR takes
  // (verified empirically). A genuine regression to a dense solver
  // will trip this assertion.
  EXPECT(ms < 50.0, "sparse solver budget: 1000 pts < 50 ms (got " +
                std::to_string(ms) + " ms)");
  std::printf("  [perf] 1000-pt smooth solve = %.1f ms\n", ms);
}

// ------------------------------------------------------------------------
// End-to-end: w_penetration forces a path that's initially inside a wall
// to be pushed out, even when w_obstacle alone would leave a flat plateau.
// ------------------------------------------------------------------------
static void test_smoother_penetration_penalizes_interior()
{
  // 2x2 wall at (4,4)-(5,5). Path's middle point at (4.5, 4.5) = cell
  // center of (4,4) → bilinear dist = -1 (deepest point inside wall).
  // At the saddle, gradient wrt x,y is 0, so the optimizer leaves the
  // point put. We use this property to verify the cost function: with
  // w_penetration=0 the cost is the saturated hinge (0.5*w_ob*safe^2);
  // with w_penetration>0 the cost is at least 0.5*w_pen*1.0 (because
  // pen = -dist = 1). So the final cost must grow by roughly
  // 0.5 * w_penetration * 1.0.
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
  // 0.5 * 5000 * 1.0 = 2500 — that's how much extra cost the
  // penetration term contributes. Allow some slack because the
  // optimizer also has the safety hinge and other terms.
  EXPECT(r_pen.final_cost > r_no_pen.final_cost + 1000.0,
    "w_penetration > 0 must add a large cost for being inside the wall (got "
    + std::to_string(r_no_pen.final_cost) + " vs "
    + std::to_string(r_pen.final_cost) + ")");
}

// ========================================================================
// Main
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
