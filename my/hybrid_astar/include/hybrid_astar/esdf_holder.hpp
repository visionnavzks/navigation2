#ifndef HYBRID_ASTAR__ESDF_HOLDER_HPP_
#define HYBRID_ASTAR__ESDF_HOLDER_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "hybrid_astar/costmap_2d.hpp"
#include "esdf_core/costmap2d.hpp"
#include "esdf_core/esdf.hpp"

namespace hybrid_astar
{

/**
 * @class EsdfHolder
 * @brief Cached Euclidean Signed Distance Field for the active Costmap2D.
 *
 * The ESDF is built once per (re)configuration and reused across every
 * collision check, every soft-penalty cost evaluation, and the obstacle
 * heuristic. Costmap changes (e.g. downsampler producing a new grid) are
 * detected by pointer comparison in rebuildIfNeeded().
 *
 * World coordinates are converted to grid coordinates and the ESDF value is
 * fetched by bilinear interpolation between the four nearest cell centers
 * (clamped at the boundary). This makes the cost surface C0-continuous, which
 * matters because both the A* traversal cost and the obstacle heuristic read
 * from it.
 *
 * The holder accepts a `hybrid_astar::Costmap2D` (the planner's local
 * costmap type, which has its own extra helpers) and constructs a
 * non-owning `esdf_core::Costmap2D` view over its raw data buffer before
 * handing it to `esdf_core::ESDF`. The source costmap must outlive the
 * holder's cached ESDF, which is naturally true because both live as
 * members of the same SmacPlannerHybrid.
 */
class EsdfHolder
{
public:
  EsdfHolder() = default;

  /**
   * @brief Build (or rebuild) the ESDF for the given costmap.
   *
   * Does nothing if `costmap` is the same pointer as the last call and the
   * `use_exact` flag is unchanged; in that case the cached ESDF is preserved
   * (cheaper when the planner is queried many times with the same map).
   */
  void rebuild(Costmap2D * costmap, bool use_exact)
  {
    if (costmap == nullptr) {
      valid_ = false;
      return;
    }
    if (costmap == costmap_ && use_exact == use_exact_ && valid_) {
      return;
    }
    costmap_ = costmap;
    use_exact_ = use_exact;
    origin_x_ = costmap->getOriginX();
    origin_y_ = costmap->getOriginY();
    resolution_ = costmap->getResolution();
    inv_resolution_ = 1.0 / resolution_;
    size_x_ = costmap->getSizeInCellsX();
    size_y_ = costmap->getSizeInCellsY();
    // Build a non-owning esdf_core::Costmap2D view over the planner's
    // costmap data. The planner owns the buffer; the view just borrows it.
    esdf_core::Costmap2D esdf_view(
      size_x_, size_y_, resolution_, origin_x_, origin_y_,
      const_cast<unsigned char *>(costmap->getCharMap()));
    const auto algo = use_exact ? esdf_core::ESDFAlgorithm::Exact
                                : esdf_core::ESDFAlgorithm::Approximate;
    esdf_ = esdf_core::ESDF::ComputeESDF(
      &esdf_view, esdf_core::Costmap2D::LETHAL_OBSTACLE, algo);
    valid_ = true;
  }

  /// Return the raw flat ESDF (length size_x*size_y, meters). Empty if invalid.
  const std::vector<double> & values() const { return esdf_; }

  /// Pointer to the costmap the ESDF was built from (or nullptr if invalid).
  Costmap2D * costmap() const { return costmap_; }

  /// True after a successful rebuild.
  bool valid() const { return valid_; }

  /// Resolution in meters per cell.
  double resolution() const { return resolution_; }

  /// Returns true if the (mx, my) cell is within the costmap bounds.
  bool inBounds(int mx, int my) const
  {
    return mx >= 0 && my >= 0 &&
           static_cast<unsigned int>(mx) < size_x_ &&
           static_cast<unsigned int>(my) < size_y_;
  }

  /// Raw ESDF value at a cell (no interpolation). Returns -inf if out of bounds.
  double clearanceAtCell(int mx, int my) const
  {
    if (!inBounds(mx, my)) {
      return -std::numeric_limits<double>::infinity();
    }
    return esdf_[static_cast<size_t>(my) * size_x_ + static_cast<size_t>(mx)];
  }

  /**
   * @brief Signed clearance to the nearest obstacle at a continuous world point.
   *
   * Positive when the point is in free space (clearance is the distance, in
   * meters, from the point to the nearest lethal cell center, minus a
   * conservative half-cell offset to approximate the distance to the cell
   * boundary instead of the cell center). Zero on the obstacle boundary.
   * Negative when the point is inside a lethal cell (with magnitude
   * `resolution`).
   *
   * Returns -infinity if the point is outside the costmap.
   *
   * Implementation note: the cached ESDF stores per-cell-center distances
   * (the distance transform gives the distance from each cell's center to
   * the nearest lethal cell's center). Bilinear interpolation of those
   * values gives a smooth field but is not a correct signed-distance field
   * for points that straddle a lethal / free boundary. We instead do a
   * nearest-cell lookup of the costmap to decide inside/outside and a
   * nearest-cell lookup of the ESDF for the distance, with a half-cell
   * correction to convert cell-center distance to boundary distance.
   */
  double clearanceAtWorld(double wx, double wy) const
  {
    if (!valid_) {
      return -std::numeric_limits<double>::infinity();
    }
    // Cell index in the costmap. Cell `mx` spans world x in
    // [mx*res + origin_x, (mx+1)*res + origin_x), so the point
    // `(wx, wy)` belongs to cell `floor((wx-origin_x)/res)`.
    const int mx = static_cast<int>(std::floor((wx - origin_x_) * inv_resolution_));
    const int my = static_cast<int>(std::floor((wy - origin_y_) * inv_resolution_));
    if (mx < 0 || my < 0 ||
        static_cast<unsigned int>(mx) >= size_x_ ||
        static_cast<unsigned int>(my) >= size_y_)
    {
      return -std::numeric_limits<double>::infinity();
    }

    // Decide inside/outside by the costmap at the cell the point is in.
    // (The ESDF treats lethal cells as having a negative "inside" depth,
    //  which is a discretized quantity and not meaningful for continuous
    //  points that fall on a boundary.)
    const auto idx = static_cast<size_t>(my) * size_x_ + static_cast<size_t>(mx);
    const unsigned char cost = costmap_->getCharMap()[idx];
    if (cost >= esdf_core::Costmap2D::LETHAL_OBSTACLE) {
      return -resolution_;
    }

    // Outside: nearest-cell-center ESDF distance, then subtract half a
    // cell to convert from "center-to-center" to "boundary-to-point".
    // This is an approximation; the true error is up to `resolution/2`
    // for axial neighbors and down to 0 for diagonal neighbors.
    const double d_center = esdf_[idx];
    const double d_boundary = d_center - 0.5 * resolution_;
    return d_boundary > 0.0 ? d_boundary : 0.0;
  }

private:
  Costmap2D * costmap_{nullptr};
  std::vector<double> esdf_{};
  bool use_exact_{true};
  bool valid_{false};
  double origin_x_{0.0};
  double origin_y_{0.0};
  double resolution_{1.0};
  double inv_resolution_{1.0};
  unsigned int size_x_{0};
  unsigned int size_y_{0};
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__ESDF_HOLDER_HPP_
