// Tests for the esdf_core ESDF computation.
//
// We construct a small synthetic costmap with a single rectangular lethal region
// in the middle, then sample clearance at known points and compare to the
// expected Euclidean distance.

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "esdf_core/costmap2d.hpp"
#include "esdf_core/esdf.hpp"

using esdf_core::Costmap2D;
using esdf_core::ESDF;
using esdf_core::ESDFAlgorithm;

namespace
{
constexpr double kResolution = 0.5;  // meters / cell
constexpr unsigned int kSizeX = 40;
constexpr unsigned int kSizeY = 40;

// Build a costmap with a single rectangular lethal obstacle in the middle.
Costmap2D makeRectangularObstacleCostmap()
{
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  // Lethal rectangle from cell (14, 14) to (25, 25) inclusive.
  for (unsigned int my = 14; my <= 25; ++my) {
    for (unsigned int mx = 14; mx <= 25; ++mx) {
      costmap.setCost(mx, my, Costmap2D::LETHAL_OBSTACLE);
    }
  }
  return costmap;
}

double esdfAt(const std::vector<double> & esdf, unsigned int mx, unsigned int my)
{
  return esdf[my * kSizeX + mx];
}
}  // namespace

TEST(ESDFTest, ExactAndApproximateAgreeNearObstacle)
{
  Costmap2D costmap = makeRectangularObstacleCostmap();
  auto exact = ESDF::ComputeExactESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);
  auto approx = ESDF::ComputeApproximateESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);
  ASSERT_EQ(exact.size(), kSizeX * kSizeY);
  ASSERT_EQ(approx.size(), kSizeX * kSizeY);

  // Only compare in the "near obstacle" band: cells within 5 cells of the obstacle
  // rectangle (mx in [9, 30], my in [9, 30]). Far away the discrete 8-neighbor
  // Dijkstra accumulates a small but tolerable drift (a few cell widths).
  for (unsigned int my = 9; my <= 30; ++my) {
    for (unsigned int mx = 9; mx <= 30; ++mx) {
      const size_t i = my * kSizeX + mx;
      EXPECT_NEAR(exact[i], approx[i], kResolution * 1.05)
        << "mismatch at (mx, my) = (" << mx << ", " << my << ")";
    }
  }
}

TEST(ESDFTest, FreeRegionDistancesAreFinite)
{
  Costmap2D costmap = makeRectangularObstacleCostmap();
  auto esdf = ESDF::ComputeExactESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);

  // (0, 0) is far from the obstacle: should be finite and > 0.
  double d00 = esdfAt(esdf, 0, 0);
  EXPECT_TRUE(std::isfinite(d00));
  EXPECT_GT(d00, 0.0);
}

TEST(ESDFTest, ObstacleInteriorHasNegativeDistance)
{
  Costmap2D costmap = makeRectangularObstacleCostmap();
  auto esdf = ESDF::ComputeExactESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);

  // Center of the obstacle: distance should be negative.
  double d_center = esdfAt(esdf, 19, 19);
  EXPECT_LT(d_center, 0.0);
}

TEST(ESDFTest, DistanceMatchesEuclideanFromObstacleEdge)
{
  Costmap2D costmap = makeRectangularObstacleCostmap();
  auto esdf = ESDF::ComputeExactESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);

  // Cell (10, 20) is 4 cells to the left of the obstacle edge at x=14.
  // Expected distance: 4 * 0.5 = 2.0 m.
  double d = esdfAt(esdf, 10, 20);
  EXPECT_NEAR(d, 2.0, kResolution * 1.05);

  // Cell (30, 20) is 5 cells to the right of the obstacle edge at x=25.
  // Expected distance: 5 * 0.5 = 2.5 m.
  d = esdfAt(esdf, 30, 20);
  EXPECT_NEAR(d, 2.5, kResolution * 1.05);

  // Cell (20, 30) is 5 cells above the obstacle edge at y=25.
  d = esdfAt(esdf, 20, 30);
  EXPECT_NEAR(d, 2.5, kResolution * 1.05);

  // Cell (20, 5) is 9 cells below the obstacle edge at y=14.
  d = esdfAt(esdf, 20, 5);
  EXPECT_NEAR(d, 4.5, kResolution * 1.05);
}

TEST(ESDFTest, CornerOfObstacleHasEuclideanCornerDistance)
{
  Costmap2D costmap = makeRectangularObstacleCostmap();
  auto esdf = ESDF::ComputeExactESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);

  // The obstacle has a corner at cell (14, 14) (just outside).
  // Cell (10, 10) is at the diagonal: dx=4, dy=4 in cells = 2.0 m, 2.0 m.
  // Euclidean distance to the corner: sqrt(2^2 + 2^2) = sqrt(8) ≈ 2.828 m.
  double d = esdfAt(esdf, 10, 10);
  EXPECT_NEAR(d, std::sqrt(2.0 * 2.0 + 2.0 * 2.0), kResolution * 1.05);
}

TEST(ESDFTest, NullCostmapThrows)
{
  EXPECT_THROW(
    ESDF::ComputeESDF(nullptr, Costmap2D::LETHAL_OBSTACLE, ESDFAlgorithm::Exact),
    esdf_core::InvalidCostmap);
}
