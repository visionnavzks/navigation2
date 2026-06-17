// Tests for the esdf_core ESDF computation.
//
// We construct a small synthetic costmap with a single rectangular lethal region
// in the middle, then sample clearance at known points and compare to the
// expected Euclidean distance.

#include <cmath>
#include <limits>
#include <utility>
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

TEST(ESDFTest, ApproximateInteriorIsNegative)
{
  Costmap2D costmap = makeRectangularObstacleCostmap();
  auto approx = ESDF::ComputeApproximateESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);
  ASSERT_EQ(approx.size(), kSizeX * kSizeY);
  EXPECT_LT(esdfAt(approx, 19, 19), 0.0);  // obstacle interior
  EXPECT_GT(esdfAt(approx, 0, 0), 0.0);    // free space
  EXPECT_TRUE(std::isfinite(esdfAt(approx, 0, 0)));
}

TEST(ESDFTest, ObstacleFreeMapClampsToGridDiagonal)
{
  // A costmap with no lethal cells at all: every cell's distance-to-obstacle is
  // undefined and must be clamped to the grid diagonal, not an unbounded
  // sentinel (~1e19 for the exact path, +inf for the approximate path).
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  const double diagonal =
    std::hypot(static_cast<double>(kSizeX), static_cast<double>(kSizeY)) * kResolution;

  auto exact = ESDF::ComputeExactESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);
  auto approx = ESDF::ComputeApproximateESDF(&costmap, Costmap2D::LETHAL_OBSTACLE);
  ASSERT_EQ(exact.size(), kSizeX * kSizeY);
  ASSERT_EQ(approx.size(), kSizeX * kSizeY);

  for (size_t i = 0; i < exact.size(); ++i) {
    ASSERT_TRUE(std::isfinite(exact[i]));
    ASSERT_TRUE(std::isfinite(approx[i]));
    EXPECT_NEAR(exact[i], diagonal, 1e-6);
    EXPECT_NEAR(approx[i], diagonal, 1e-6);
  }
}

TEST(Costmap2DTest, CopyIsDeepForOwningMap)
{
  Costmap2D original(4, 3, 0.1, 0.0, 0.0);
  original.setCost(1, 1, Costmap2D::LETHAL_OBSTACLE);

  Costmap2D copy = original;
  EXPECT_NE(copy.getCharMap(), original.getCharMap());  // independent storage
  EXPECT_EQ(copy.getCost(1, 1), Costmap2D::LETHAL_OBSTACLE);

  // Mutating one map must not affect the other.
  copy.setCost(0, 0, 7);
  EXPECT_EQ(copy.getCost(0, 0), 7);
  EXPECT_EQ(original.getCost(0, 0), Costmap2D::FREE_SPACE);

  original.setCost(2, 2, 9);
  EXPECT_EQ(original.getCost(2, 2), 9);
  EXPECT_EQ(copy.getCost(2, 2), Costmap2D::FREE_SPACE);
}

TEST(Costmap2DTest, MovePreservesOwnedData)
{
  Costmap2D original(4, 3, 0.1, 0.0, 0.0);
  original.setCost(2, 1, 42);

  Costmap2D moved = std::move(original);
  EXPECT_EQ(moved.getSizeInCellsX(), 4u);
  EXPECT_EQ(moved.getSizeInCellsY(), 3u);
  EXPECT_NE(moved.getCharMap(), nullptr);
  EXPECT_EQ(moved.getCost(2, 1), 42);
}

TEST(Costmap2DTest, NonOwningConstructorSharesExternalBuffer)
{
  std::vector<unsigned char> buffer(4 * 3, Costmap2D::FREE_SPACE);
  Costmap2D view(4, 3, 0.1, 0.0, 0.0, buffer.data());
  EXPECT_EQ(view.getCharMap(), buffer.data());  // shares, does not copy

  // Writes through the view land in the external buffer...
  view.setCost(1, 2, 5);
  EXPECT_EQ(buffer[2 * 4 + 1], 5);
  // ...and external writes are visible through the view.
  buffer[0] = 8;
  EXPECT_EQ(view.getCost(0, 0), 8);

  // A copy of a non-owning map keeps pointing at the same external buffer.
  Costmap2D copy = view;
  EXPECT_EQ(copy.getCharMap(), buffer.data());
  buffer[0] = 3;
  EXPECT_EQ(copy.getCost(0, 0), 3);
}
