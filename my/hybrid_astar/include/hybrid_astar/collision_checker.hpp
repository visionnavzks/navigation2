#ifndef HYBRID_ASTAR__COLLISION_CHECKER_HPP_
#define HYBRID_ASTAR__COLLISION_CHECKER_HPP_

#include <memory>
#include <vector>
#include <cmath>

#include "hybrid_astar/constants.hpp"
#include "hybrid_astar/costmap_2d.hpp"
#include "hybrid_astar/types.hpp"
#include "hybrid_astar/esdf_holder.hpp"

namespace hybrid_astar
{

/**
 * @class GridCollisionChecker
 * @brief Hybrid A* collision checker.
 *
 * Supports two backends, selected at configuration time:
 *
 *   1. Legacy polygon / single-radius path (setFootprint). This is the
 *      original implementation: pre-rotate every footprint vertex and check
 *      raw cell costs against OCCUPIED/INSCRIBED thresholds. Cheap, but
 *      does not use distance information.
 *
 *   2. ESDF + capsule path (setEsdfFootprint). Each triple (lx, ly, w) in
 *      cost_check_points defines a local checkpoint on the robot body. At
 *      every search pose, the checkpoints are rotated into the world frame
 *      and the cached ESDF is queried. Hard rejection happens when the
 *      minimum ESDF clearance drops below `robot_radius`; a smooth quadratic
 *      soft penalty is added when the clearance is inside `safe_distance`.
 *
 * `inCollisionEsdf()` and `getMinClearance()`/`getSoftPenalty()` are the
 * capsule-path entry points used by NodeHybrid, ObstacleHeuristic, and the
 * analytic expansion.
 */
class GridCollisionChecker
{
public:
  GridCollisionChecker(
    Costmap2D * costmap,
    unsigned int num_quantizations);

  /// Configure the legacy polygon-vertex / single-radius path.
  void setFootprint(
    const Footprint & footprint,
    const bool & radius,
    const double & possible_collision_cost);

  /// Configure the ESDF + capsule path.
  void setEsdfFootprint(
    const std::vector<double> & cost_check_points,
    double robot_radius,
    double safe_distance,
    EsdfHolder * esdf_holder);

  /// True if the checker is currently using the ESDF/capsule path.
  bool usesEsdfFootprint() const { return use_esdf_footprint_; }

  /// Returns the ESDF holder pointer (may be nullptr if not configured).
  EsdfHolder * getEsdfHolder() const { return esdf_holder_; }

  /// Legacy in-collision check (polygon / single radius).
  bool inCollision(
    const float & x, const float & y,
    const float & theta, const bool & traverse_unknown);

  /// Legacy index-based check (kept for backward compatibility).
  bool inCollision(
    const unsigned int & i, const bool & traverse_unknown);

  /// ESDF/capsule in-collision check at a continuous world pose (radians).
  /// Returns true if the robot footprint (as defined by cost_check_points)
  /// cannot fit at (wx, wy, theta) given the cached ESDF.
  bool inCollisionEsdf(
    double wx, double wy, double theta, bool traverse_unknown) const;

  /// Minimum ESDF clearance over the footprint, in meters. -inf if any
  /// checkpoint is outside the map or inside an obstacle.
  double getMinClearance(
    double wx, double wy, double theta) const;

  /// Quadratic soft penalty in [0, 1] driven by `safe_distance`. 0 if the
  /// robot is comfortably outside the safe band; 1 if it is touching an
  /// obstacle (clearance == robot_radius). The maximum cost registered in
  /// the search is `penalty * MAX_NON_OBSTACLE_COST`, which then drives
  /// `getTraversalCost` via the existing `cost_penalty * normalized_cost`
  /// pipeline.
  double getSoftPenalty(
    double wx, double wy, double theta) const;

  /// Returns the most recently observed center cell cost (legacy path).
  float getCost() const;

  std::vector<float> & getPrecomputedAngles() { return angles_; }

  Costmap2D * getCostmap() { return costmap_; }
  const Costmap2D * getCostmap() const { return costmap_; }

  void setCostmap(Costmap2D * costmap) { costmap_ = costmap; }

  bool outsideRange(const unsigned int & max, const float & value) const;

protected:
  Costmap2D * costmap_;
  std::vector<Footprint> oriented_footprints_;
  Footprint unoriented_footprint_;
  float center_cost_;
  bool footprint_is_radius_{false};
  std::vector<float> angles_;
  float possible_collision_cost_{-1};

  // ESDF + capsule path state. `use_esdf_footprint_` is set by
  // setEsdfFootprint() and cleared by setFootprint() (mutually exclusive
  // configurations, just like the legacy path).
  bool use_esdf_footprint_{false};
  std::vector<double> cost_check_points_{};
  double robot_radius_{0.0};
  double safe_distance_{0.0};
  EsdfHolder * esdf_holder_{nullptr};
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__COLLISION_CHECKER_HPP_
