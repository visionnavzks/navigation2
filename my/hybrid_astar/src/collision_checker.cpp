#include "hybrid_astar/collision_checker.hpp"

namespace hybrid_astar
{

GridCollisionChecker::GridCollisionChecker(
  Costmap2D * costmap,
  unsigned int num_quantizations)
{
  costmap_ = costmap;

  float bin_size = 2.0f * static_cast<float>(M_PI) / static_cast<float>(num_quantizations);
  angles_.reserve(num_quantizations);
  for (unsigned int i = 0; i != num_quantizations; i++) {
    angles_.push_back(bin_size * static_cast<float>(i));
  }
}

void GridCollisionChecker::setFootprint(
  const Footprint & footprint,
  const bool & radius,
  const double & possible_collision_cost)
{
  possible_collision_cost_ = static_cast<float>(possible_collision_cost);
  footprint_is_radius_ = radius;

  // Legacy and ESDF paths are mutually exclusive. Switching to the legacy
  // path must clear the ESDF configuration so the planner doesn't
  // accidentally query an unconfigured holder.
  use_esdf_footprint_ = false;
  cost_check_points_.clear();
  robot_radius_ = 0.0;
  safe_distance_ = 0.0;
  esdf_holder_ = nullptr;

  if (radius) {
    return;
  }

  if (footprint == unoriented_footprint_) {
    return;
  }

  oriented_footprints_.clear();
  oriented_footprints_.reserve(angles_.size());
  double sin_th, cos_th;
  const unsigned int footprint_size = footprint.size();

  for (unsigned int i = 0; i != angles_.size(); i++) {
    sin_th = sin(static_cast<double>(angles_[i]));
    cos_th = cos(static_cast<double>(angles_[i]));
    Footprint oriented_footprint;
    oriented_footprint.reserve(footprint_size);

    for (unsigned int j = 0; j < footprint_size; j++) {
      Footprint::value_type new_pt;
      new_pt.x = footprint[j].x * cos_th - footprint[j].y * sin_th;
      new_pt.y = footprint[j].x * sin_th + footprint[j].y * cos_th;
      oriented_footprint.push_back(new_pt);
    }

    oriented_footprints_.push_back(oriented_footprint);
  }

  unoriented_footprint_ = footprint;
}

void GridCollisionChecker::setEsdfFootprint(
  const std::vector<double> & cost_check_points,
  double robot_radius,
  double safe_distance,
  EsdfHolder * esdf_holder)
{
  // Switching to the ESDF path must invalidate the legacy pre-rotated
  // footprint, so callers can't accidentally read stale data through the
  // legacy API.
  oriented_footprints_.clear();
  unoriented_footprint_.clear();
  footprint_is_radius_ = false;
  possible_collision_cost_ = -1.0f;

  use_esdf_footprint_ = true;
  cost_check_points_ = cost_check_points;
  robot_radius_ = std::max(robot_radius, 0.0);
  safe_distance_ = std::max(safe_distance, 0.0);
  esdf_holder_ = esdf_holder;
}

bool GridCollisionChecker::inCollision(
  const float & x,
  const float & y,
  const float & angle_bin,
  const bool & traverse_unknown)
{
  if (outsideRange(costmap_->getSizeInCellsX(), x) ||
    outsideRange(costmap_->getSizeInCellsY(), y))
  {
    return true;
  }

  center_cost_ = static_cast<float>(costmap_->getCost(x, y));

  if (!footprint_is_radius_) {
    if (center_cost_ < possible_collision_cost_ && possible_collision_cost_ > 0.0f) {
      return false;
    }

    if (center_cost_ == UNKNOWN_COST && !traverse_unknown) {
      return true;
    }

    if (center_cost_ == INSCRIBED_COST || center_cost_ == OCCUPIED_COST) {
      return true;
    }

    float wx, wy;
    costmap_->mapCellToWorld(x, y, wx, wy);
    const Footprint & oriented_footprint =
      oriented_footprints_[static_cast<unsigned int>(angle_bin)];

    for (unsigned int i = 0; i < oriented_footprint.size(); ++i) {
      double px = wx + oriented_footprint[i].x;
      double py = wy + oriented_footprint[i].y;

      unsigned int mx, my;
      if (!costmap_->worldToMap(px, py, mx, my)) {
        return true;
      }

      unsigned char cell_cost = costmap_->getCost(mx, my);
      if (cell_cost >= static_cast<unsigned char>(OCCUPIED_COST)) {
        return true;
      }
      if (cell_cost == static_cast<unsigned char>(UNKNOWN_COST) && !traverse_unknown) {
        return true;
      }
    }

    return false;
  } else {
    if (center_cost_ == UNKNOWN_COST && traverse_unknown) {
      return false;
    }

    return center_cost_ >= INSCRIBED_COST;
  }
}

bool GridCollisionChecker::inCollision(
  const unsigned int & i,
  const bool & traverse_unknown)
{
  center_cost_ = static_cast<float>(costmap_->getCost(i));
  if (center_cost_ == UNKNOWN_COST && traverse_unknown) {
    return false;
  }

  return center_cost_ >= INSCRIBED_COST;
}

float GridCollisionChecker::getCost() const
{
  return center_cost_;
}

bool GridCollisionChecker::outsideRange(const unsigned int & max, const float & value) const
{
  return value < 0.0f || value >= static_cast<float>(max);
}

bool GridCollisionChecker::inCollisionEsdf(
  double wx, double wy, double theta, bool traverse_unknown) const
{
  (void)traverse_unknown;  // ESDF is fully signed; the unknown-as-obstacle
                            // behavior is handled at the costmap level by
                            // pre-letting the costmap_2d layer mark unknowns
                            // as lethal before ESDF construction.
  if (!use_esdf_footprint_ || esdf_holder_ == nullptr || !esdf_holder_->valid()) {
    return false;
  }
  const double min_clearance = getMinClearance(wx, wy, theta);
  if (!std::isfinite(min_clearance)) {
    return true;
  }
  return min_clearance < robot_radius_;
}

double GridCollisionChecker::getMinClearance(
  double wx, double wy, double theta) const
{
  if (!use_esdf_footprint_ || esdf_holder_ == nullptr || !esdf_holder_->valid()) {
    return std::numeric_limits<double>::infinity();
  }
  if (cost_check_points_.empty()) {
    // No checkpoints: a single point at the robot center. If robot_radius is
    // also zero this matches the original point-robot behavior.
    return esdf_holder_->clearanceAtWorld(wx, wy);
  }

  const double cos_t = std::cos(theta);
  const double sin_t = std::sin(theta);
  double min_clearance = std::numeric_limits<double>::infinity();
  for (size_t offset = 0; offset + 2 < cost_check_points_.size(); offset += 3) {
    const double lx = cost_check_points_[offset + 0];
    const double ly = cost_check_points_[offset + 1];
    const double world_x = wx + cos_t * lx - sin_t * ly;
    const double world_y = wy + sin_t * lx + cos_t * ly;
    const double d = esdf_holder_->clearanceAtWorld(world_x, world_y);
    if (d < min_clearance) {
      min_clearance = d;
    }
  }
  return min_clearance;
}

double GridCollisionChecker::getSoftPenalty(
  double wx, double wy, double theta) const
{
  if (!use_esdf_footprint_ || safe_distance_ <= 1e-9) {
    return 0.0;
  }
  const double min_clearance = getMinClearance(wx, wy, theta);
  if (!std::isfinite(min_clearance)) {
    // Off-map or in-obstacle: treat as maximum penalty so the A* cost
    // reflects that this node is unusable. (Hard rejection is the job of
    // inCollisionEsdf; the penalty is just for ordering.)
    return 1.0;
  }
  const double surface_distance = min_clearance - robot_radius_;
  if (surface_distance >= safe_distance_) {
    return 0.0;
  }
  // Map the surface distance to a normalized gap in [0, 1]:
  //   surface <= 0  -> gap = 1 (point is at or inside the obstacle)
  //   0 < surface < safe -> gap = (safe - surface) / safe in (0, 1)
  //   surface >= safe -> gap = 0 (returned above as a fast path)
  // We use the standard clamp-then-subtract form so the result is
  // guaranteed to be non-negative and finite.
  const double clamped_surface = std::min(std::max(surface_distance, 0.0), safe_distance_);
  const double normalized_gap = (safe_distance_ - clamped_surface) / safe_distance_;
  return normalized_gap * normalized_gap;
}

}  // namespace hybrid_astar
