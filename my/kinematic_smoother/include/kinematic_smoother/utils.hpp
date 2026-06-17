// Copyright (c) 2021 RoboTech Vision
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONSTRAINED_SMOOTHER__UTILS_HPP_
#define CONSTRAINED_SMOOTHER__UTILS_HPP_

#include <cmath>
#include <vector>
#include "Eigen/Core"

namespace kinematic_smoother
{

constexpr double EPSILON = 0.0001;
constexpr double PI = 3.14159265358979323846;

inline double goalPositionFrameHeading(
  const std::vector<Eigen::Vector2d> & reference_points,
  double end_theta,
  bool keep_goal_orientation)
{
  if (keep_goal_orientation || reference_points.size() < 2) {
    return end_theta;
  }

  const Eigen::Vector2d goal_delta = reference_points.back() - reference_points[reference_points.size() - 2];
  if (goal_delta.norm() <= EPSILON) {
    return end_theta;
  }

  return std::atan2(goal_delta.y(), goal_delta.x());
}

}  // namespace kinematic_smoother

#endif  // CONSTRAINED_SMOOTHER__UTILS_HPP_
