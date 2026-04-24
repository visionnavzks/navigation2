// Copyright (c) 2021 RoboTech Vision
// Copyright (c) 2020, Samsung Research America
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

#ifndef CONSTRAINED_SMOOTHER__EXCEPTIONS_HPP_
#define CONSTRAINED_SMOOTHER__EXCEPTIONS_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>

namespace constrained_smoother
{

enum class ErrorCode : uint16_t
{
  InvalidPath = 1001,
  FailedToSmoothPath = 2001,
  InvalidCostmap = 3001,
  PrecomputedEsdfSizeMismatch = 3002,
};

enum class SmoothingFailureReason : uint16_t
{
  Unknown = 0,
  SolverRejectedSolution = 1,
  NoCostImprovement = 2,
  InvalidStateVector = 3,
  NonFiniteState = 4,
  StartPositionConstraint = 5,
  StartOrientationConstraint = 6,
  GoalPositionConstraint = 7,
  GoalOrientationConstraint = 8,
  CuspHoldConstraint = 9,
  CollapsedSegment = 10,
  MotionDirectionConstraint = 11,
  PathOutOfBounds = 12,
  FootprintCollision = 13,
};

inline const char * toErrorCodeString(ErrorCode code)
{
  switch (code) {
    case ErrorCode::InvalidPath:
      return "CS_INVALID_PATH";
    case ErrorCode::FailedToSmoothPath:
      return "CS_SMOOTHING_FAILED";
    case ErrorCode::InvalidCostmap:
      return "CS_INVALID_COSTMAP";
    case ErrorCode::PrecomputedEsdfSizeMismatch:
      return "CS_PRECOMPUTED_ESDF_SIZE_MISMATCH";
    default:
      return "CS_UNKNOWN_ERROR";
  }
}

inline const char * toSmoothingFailureReasonString(SmoothingFailureReason reason)
{
  switch (reason) {
    case SmoothingFailureReason::SolverRejectedSolution:
      return "solver_rejected_solution";
    case SmoothingFailureReason::NoCostImprovement:
      return "no_cost_improvement";
    case SmoothingFailureReason::InvalidStateVector:
      return "invalid_state_vector";
    case SmoothingFailureReason::NonFiniteState:
      return "nonfinite_state";
    case SmoothingFailureReason::StartPositionConstraint:
      return "start_position_constraint";
    case SmoothingFailureReason::StartOrientationConstraint:
      return "start_orientation_constraint";
    case SmoothingFailureReason::GoalPositionConstraint:
      return "goal_position_constraint";
    case SmoothingFailureReason::GoalOrientationConstraint:
      return "goal_orientation_constraint";
    case SmoothingFailureReason::CuspHoldConstraint:
      return "cusp_hold_constraint";
    case SmoothingFailureReason::CollapsedSegment:
      return "collapsed_segment";
    case SmoothingFailureReason::MotionDirectionConstraint:
      return "motion_direction_constraint";
    case SmoothingFailureReason::PathOutOfBounds:
      return "path_out_of_bounds";
    case SmoothingFailureReason::FootprintCollision:
      return "footprint_collision";
    case SmoothingFailureReason::Unknown:
    default:
      return "unknown";
  }
}

inline std::string buildSmoothingFailureMessage(
  SmoothingFailureReason reason,
  const std::string & message,
  int failed_index = -1)
{
  std::string formatted = toSmoothingFailureReasonString(reason);
  if (failed_index >= 0) {
    formatted += "@" + std::to_string(failed_index);
  }
  formatted += ": " + message;
  return formatted;
}

struct SmoothingFailureInfo
{
  SmoothingFailureReason reason{SmoothingFailureReason::Unknown};
  std::string message{};
  int failed_index{-1};

  std::string formattedMessage() const
  {
    return buildSmoothingFailureMessage(reason, message, failed_index);
  }
};

/**
 * @class InvalidPath
 * @brief Thrown when the input path is invalid (e.g. too short).
 */
class InvalidPath : public std::runtime_error
{
public:
  explicit InvalidPath(const std::string & msg)
  : std::runtime_error(msg) {}

  ErrorCode code() const noexcept
  {
    return ErrorCode::InvalidPath;
  }

  const char * codeString() const noexcept
  {
    return toErrorCodeString(code());
  }
};

/**
 * @class FailedToSmoothPath
 * @brief Thrown when the optimizer fails to produce a usable solution.
 */
class FailedToSmoothPath : public std::runtime_error
{
public:
  explicit FailedToSmoothPath(const std::string & msg)
  : std::runtime_error(msg) {}

  ErrorCode code() const noexcept
  {
    return ErrorCode::FailedToSmoothPath;
  }

  const char * codeString() const noexcept
  {
    return toErrorCodeString(code());
  }
};

inline bool throwOrStoreSmoothingFailure(
  SmoothingFailureInfo * failure,
  SmoothingFailureReason reason,
  const std::string & message,
  int failed_index = -1)
{
  if (failure != nullptr) {
    failure->reason = reason;
    failure->message = message;
    failure->failed_index = failed_index;
    return false;
  }

  throw FailedToSmoothPath(buildSmoothingFailureMessage(reason, message, failed_index));
}

class InvalidCostmap : public std::runtime_error
{
public:
  explicit InvalidCostmap(const std::string & msg)
  : std::runtime_error(msg) {}

  ErrorCode code() const noexcept
  {
    return ErrorCode::InvalidCostmap;
  }

  const char * codeString() const noexcept
  {
    return toErrorCodeString(code());
  }
};

class PrecomputedEsdfSizeMismatch : public std::runtime_error
{
public:
  explicit PrecomputedEsdfSizeMismatch(const std::string & msg)
  : std::runtime_error(msg) {}

  ErrorCode code() const noexcept
  {
    return ErrorCode::PrecomputedEsdfSizeMismatch;
  }

  const char * codeString() const noexcept
  {
    return toErrorCodeString(code());
  }
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__EXCEPTIONS_HPP_
