#include "my/hybrid_astar/collision_checker.hpp"

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

}  // namespace hybrid_astar
