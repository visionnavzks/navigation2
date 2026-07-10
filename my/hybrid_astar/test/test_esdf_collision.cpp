// Tests for the ESDF + capsule footprint path of the hybrid A* planner.
//
// The test builds a synthetic costmap (a single rectangular obstacle plus
// surrounding free space), constructs an EsdfHolder + GridCollisionChecker
// configured with a 3-checkpoint capsule, and verifies:
//   - inCollisionEsdf rejects poses whose footprint overlaps the obstacle
//   - inCollisionEsdf accepts poses in the clear
//   - getMinClearance returns values close to the expected Euclidean distance
//   - getSoftPenalty returns 0 outside the safe band and a smooth quadratic
//     inside it (0 at the boundary, ~1 at the obstacle surface)

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "hybrid_astar/collision_checker.hpp"
#include "hybrid_astar/costmap_2d.hpp"
#include "hybrid_astar/esdf_holder.hpp"

using hybrid_astar::Costmap2D;
using hybrid_astar::EsdfHolder;
using hybrid_astar::GridCollisionChecker;

namespace
{
// Resolution: 0.5 m / cell. Map: 20 x 20 cells = 10 m x 10 m.
// Lethal rectangle: cells (8, 8) to (12, 12) inclusive, so 2.5 m x 2.5 m.
constexpr double kResolution = 0.5;
constexpr unsigned int kSizeX = 20;
constexpr unsigned int kSizeY = 20;

void fillObstacleCostmap(Costmap2D & costmap)
{
  for (unsigned int my = 8; my <= 11; ++my) {
    for (unsigned int mx = 8; mx <= 11; ++mx) {
      costmap.setCost(mx, my, hybrid_astar::OCCUPIED_COST);
    }
  }
}

// 3-checkpoint capsule: front, center, rear, all along the x-axis.
std::vector<double> makeCapsuleCheckPoints()
{
  return std::vector<double>{
    0.30, 0.00, 1.0,
    0.00, 0.00, 1.0,
   -0.30, 0.00, 1.0,
  };
}

// (mx, my) → world coords (cell center).
double cellCenterWorldX(unsigned int mx)
{
  return (static_cast<double>(mx) + 0.5) * kResolution;
}
double cellCenterWorldY(unsigned int my)
{
  return (static_cast<double>(my) + 0.5) * kResolution;
}
}  // namespace

TEST(EsdfCollisionTest, EsdfHolderBuildsAgainstCostmap)
{
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  fillObstacleCostmap(costmap);
  EsdfHolder holder;
  holder.rebuild(&costmap, true);
  EXPECT_TRUE(holder.valid());
  EXPECT_EQ(holder.values().size(), kSizeX * kSizeY);
}

TEST(EsdfCollisionTest, CapsuleHardRejectionNearObstacle)
{
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  fillObstacleCostmap(costmap);
  EsdfHolder holder;
  holder.rebuild(&costmap, true);

  GridCollisionChecker checker(&costmap, 72);
  checker.setEsdfFootprint(makeCapsuleCheckPoints(), /*robot_radius=*/0.05,
    /*safe_distance=*/0.0, &holder);
  EXPECT_TRUE(checker.usesEsdfFootprint());

  // Cell (10, 10) is the center of the obstacle. World: (5.25, 5.25).
  // With theta=0 the front checkpoint is at (5.55, 5.25), which is inside
  // the obstacle rectangle. Should be in collision.
  const double obs_cx = cellCenterWorldX(10);
  const double obs_cy = cellCenterWorldY(10);
  EXPECT_TRUE(checker.inCollisionEsdf(obs_cx, obs_cy, 0.0, true));

  // Pose well clear of the obstacle (3, 3 cell ~ 1.75, 1.75 m world).
  const double free_cx = cellCenterWorldX(3);
  const double free_cy = cellCenterWorldY(3);
  EXPECT_FALSE(checker.inCollisionEsdf(free_cx, free_cy, 0.0, true));
}

TEST(EsdfCollisionTest, MinClearanceMatchesExpectedDistance)
{
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  fillObstacleCostmap(costmap);
  EsdfHolder holder;
  holder.rebuild(&costmap, true);

  GridCollisionChecker checker(&costmap, 72);
  checker.setEsdfFootprint(makeCapsuleCheckPoints(), 0.0, 0.0, &holder);

  // The obstacle's leftmost *cell* spans world x in [4.0, 4.5]; its center
  // is at 4.25 m. The pose at cell (5, 10) has world (2.75, 5.25). With
  // capsule offsets of +/- 0.3 m along x, the front checkpoint sits in
  // cell 6 (range x in [3.0, 3.5]).
  //
  // The ESDF stores a conservative boundary clearance: esdf_core subtracts
  // half a cell diagonal from the center-to-center transform. Cell 6 center
  // is 1.0 m from cell 8 center, so the expected front-checkpoint clearance
  // is 1.0 - sqrt(0.5) * 0.5 ≈ 0.646 m.
  // (The continuous model would predict 0.95 m, but that would require
  //  a sub-cell distance transform, which we do not implement.)
  const double wx = cellCenterWorldX(5);
  const double wy = cellCenterWorldY(10);
  const double min_cl = checker.getMinClearance(wx, wy, 0.0);
  EXPECT_NEAR(min_cl, 1.0 - std::sqrt(0.5) * kResolution, 1e-9);
}

TEST(EsdfCollisionTest, SoftPenaltyIsZeroOutsideSafeBand)
{
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  fillObstacleCostmap(costmap);
  EsdfHolder holder;
  holder.rebuild(&costmap, true);

  GridCollisionChecker checker(&costmap, 72);
  checker.setEsdfFootprint(makeCapsuleCheckPoints(), 0.0, /*safe_distance=*/0.5, &holder);

  // Way out in the corner: well outside the safe band.
  const double wx = cellCenterWorldX(1);
  const double wy = cellCenterWorldY(1);
  EXPECT_NEAR(checker.getSoftPenalty(wx, wy, 0.0), 0.0, 1e-9);
}

TEST(EsdfCollisionTest, SoftPenaltyIncreasesAsObstacleIsApproached)
{
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  fillObstacleCostmap(costmap);
  EsdfHolder holder;
  holder.rebuild(&costmap, true);

  // 0.2 m safe_distance + 0.0 robot_radius.
  GridCollisionChecker checker(&costmap, 72);
  checker.setEsdfFootprint(makeCapsuleCheckPoints(), 0.0, /*safe_distance=*/0.2, &holder);

  // Three poses at decreasing clearance to the obstacle. With capsule
  // offsets of +/- 0.3 m and 0.5 m cells, the front checkpoint of any
  // pose at cell >= 7 falls inside the obstacle. So pose B and pose C
  // have min_clearance < 0 (in collision) and incur the maximum soft
  // penalty (1.0). Pose A sits one cell further out; its front checkpoint
  // clearance falls inside the safe band, giving a partial penalty.
  //   Pose A: cell (6, 10) world (3.25, 5.25).
  //     front: cell 7 is one cell (0.5 m center-to-center) from the
  //            obstacle; the conservative ESDF stores
  //            0.5 - sqrt(0.5)*0.5 ≈ 0.146 m. surface = 0.146 < safe
  //            (0.2): normalized gap ≈ 0.268, squared penalty ≈ 0.072.
  //   Pose B: cell (7, 10) world (3.75, 5.25).
  //     front: in obstacle → min = -resolution. surface < 0. clamped
  //            gap = safe = 0.2. normalized = 1.0. penalty = 1.0.
  //   Pose C: cell (8, 10) world (4.25, 5.25).
  //     all checkpoints in obstacle → min < 0. penalty = 1.0.
  const double wy = cellCenterWorldY(10);
  const double penA = checker.getSoftPenalty(cellCenterWorldX(6), wy, 0.0);
  const double penB = checker.getSoftPenalty(cellCenterWorldX(7), wy, 0.0);
  const double penC = checker.getSoftPenalty(cellCenterWorldX(8), wy, 0.0);
  const double front_clearance = kResolution - std::sqrt(0.5) * kResolution;
  const double normalized_gap_a = (0.2 - front_clearance) / 0.2;
  EXPECT_NEAR(penA, normalized_gap_a * normalized_gap_a, 1e-6);
  EXPECT_NEAR(penB, 1.0, 1e-6);
  EXPECT_NEAR(penC, 1.0, 1e-6);
  EXPECT_LT(penA, penB);
  EXPECT_LE(penB, penC + 1e-9);
}

TEST(EsdfCollisionTest, NoEsdfHolderMeansLegacyPathActive)
{
  Costmap2D costmap(kSizeX, kSizeY, kResolution, 0.0, 0.0);
  GridCollisionChecker checker(&costmap, 72);
  checker.setFootprint(hybrid_astar::Footprint(), /*radius=*/true, 0.0);
  EXPECT_FALSE(checker.usesEsdfFootprint());
  // Without an ESDF holder the inCollisionEsdf path returns false (no
  // rejection). The legacy inCollision path is still available.
  EXPECT_FALSE(checker.inCollisionEsdf(0.0, 0.0, 0.0, true));
}
