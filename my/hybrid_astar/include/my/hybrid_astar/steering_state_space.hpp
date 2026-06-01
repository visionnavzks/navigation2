#ifndef HYBRID_ASTAR__STEERING_STATE_SPACE_HPP_
#define HYBRID_ASTAR__STEERING_STATE_SPACE_HPP_

#include <cmath>
#include <memory>
#include <vector>

#include "steering_functions_lite/dubins_state_space.h"
#include "steering_functions_lite/reeds_shepp_state_space.h"
#include "steering_functions_lite/state.h"

#include "my/hybrid_astar/constants.hpp"

namespace hybrid_astar
{

/// Lightweight 3-DoF state (x, y, theta) that mimics the ompl::base::ScopedState
/// interface so existing code using from[0], from[1], from[2] and .reals()
/// works with minimal changes.
struct SteeringState
{
  double state[3] = {0.0, 0.0, 0.0};  // x, y, theta

  SteeringState() = default;
  SteeringState(double x, double y, double theta) : state{x, y, theta} {}

  double & operator[](int i) { return state[i]; }
  const double & operator[](int i) const { return state[i]; }

  /// Return a copy of the internal state as a vector (mirrors s.reals()).
  std::vector<double> reals() const
  {
    return {state[0], state[1], state[2]};
  }

  /// Convert to the steering_lite::State used by the library.
  steering_lite::State toSteeringLite() const
  {
    return steering_lite::State(state[0], state[1], state[2], 0.0, 0.0);
  }
};

/// Wrapper around steering_lite (Dubins / Reeds-Shepp) that exposes
/// distance() and interpolate() with the same signatures OMPL used,
/// so the rest of hybrid_astar needs only minimal adjustments.
class SteeringStateSpace
{
public:
  SteeringStateSpace(MotionModel model, double turning_radius)
  : model_(model), turning_radius_(turning_radius)
  {
    const double kappa = 1.0 / turning_radius;
    const double disc = 0.05;  // fine enough for interpolation
    if (model == MotionModel::DUBIN) {
      space_ = std::make_unique<steering_lite::DubinsStateSpace>(
        kappa, disc, steering_lite::DubinsDirectionMode::ForwardOrReverse);
    } else if (model == MotionModel::REEDS_SHEPP) {
      space_ = std::make_unique<steering_lite::ReedsSheppStateSpace>(kappa, disc);
    }
  }

  /// Distance between two states (shortest-path length).
  double distance(const SteeringState & s1, const SteeringState & s2) const
  {
    const auto sl_s1 = s1.toSteeringLite();
    const auto sl_s2 = s2.toSteeringLite();

    // Reeds-Shepp has a dedicated fast distance method
    if (model_ == MotionModel::REEDS_SHEPP) {
      const auto * rs =
        dynamic_cast<const steering_lite::ReedsSheppStateSpace *>(space_.get());
      if (rs) {
        return rs->get_distance(sl_s1, sl_s2);
      }
    }

    // Dubins / fallback: sum absolute arc-lengths of the control sequence
    const auto controls = space_->get_controls(sl_s1, sl_s2);
    double dist = 0.0;
    for (const auto & c : controls) {
      dist += std::abs(c.delta_s);
    }
    return dist;
  }

  /// Interpolate along the shortest path from \p from to \p to at fraction \p t.
  /// \p t is clamped to [0, 1].  Result is written into \p result.
  void interpolate(
    const SteeringState & from, const SteeringState & to,
    double t, SteeringState & result) const
  {
    const auto sl_from = from.toSteeringLite();
    const auto sl_to = to.toSteeringLite();
    const auto controls = space_->get_controls(sl_from, sl_to);
    const auto interp = space_->interpolate(sl_from, controls, t);
    result[0] = interp.x;
    result[1] = interp.y;
    result[2] = interp.theta;
  }

  /// Return the shortest-path control sequence (each control has a signed
  /// delta_s whose sign indicates forward/backward motion).  Useful for
  /// analysing direction changes.
  std::vector<steering_lite::Control> getControls(
    const SteeringState & s1, const SteeringState & s2) const
  {
    return space_->get_controls(s1.toSteeringLite(), s2.toSteeringLite());
  }

  MotionModel model() const { return model_; }
  double turningRadius() const { return turning_radius_; }

private:
  MotionModel model_;
  double turning_radius_;
  std::unique_ptr<steering_lite::StateSpace> space_;
};

using SteeringStateSpacePtr = std::shared_ptr<SteeringStateSpace>;

/// Factory: replaces the old createStateSpace that returned ompl::base::StateSpacePtr.
inline SteeringStateSpacePtr createSteeringStateSpace(
  MotionModel model, double turning_radius)
{
  return std::make_shared<SteeringStateSpace>(model, turning_radius);
}

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__STEERING_STATE_SPACE_HPP_
