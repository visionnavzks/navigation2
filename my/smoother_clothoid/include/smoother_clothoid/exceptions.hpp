#ifndef SMOOTHER_CLOTHOID__EXCEPTIONS_HPP_
#define SMOOTHER_CLOTHOID__EXCEPTIONS_HPP_

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "esdf_core/exceptions.hpp"

namespace smoother_clothoid
{

using InvalidCostmap = esdf_core::InvalidCostmap;
using PrecomputedEsdfSizeMismatch = esdf_core::PrecomputedEsdfSizeMismatch;

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
  CurvatureConstraint = 14,
};

inline const char * toErrorCodeString(ErrorCode code)
{
  switch (code) {
    case ErrorCode::InvalidPath: return "SC_INVALID_PATH";
    case ErrorCode::FailedToSmoothPath: return "SC_SMOOTHING_FAILED";
    case ErrorCode::InvalidCostmap: return "SC_INVALID_COSTMAP";
    case ErrorCode::PrecomputedEsdfSizeMismatch: return "SC_PRECOMPUTED_ESDF_SIZE_MISMATCH";
    default: return "SC_UNKNOWN_ERROR";
  }
}

inline const char * toSmoothingFailureReasonString(SmoothingFailureReason reason)
{
  switch (reason) {
    case SmoothingFailureReason::SolverRejectedSolution: return "solver_rejected_solution";
    case SmoothingFailureReason::NoCostImprovement: return "no_cost_improvement";
    case SmoothingFailureReason::InvalidStateVector: return "invalid_state_vector";
    case SmoothingFailureReason::NonFiniteState: return "nonfinite_state";
    case SmoothingFailureReason::StartPositionConstraint: return "start_position_constraint";
    case SmoothingFailureReason::StartOrientationConstraint: return "start_orientation_constraint";
    case SmoothingFailureReason::GoalPositionConstraint: return "goal_position_constraint";
    case SmoothingFailureReason::GoalOrientationConstraint: return "goal_orientation_constraint";
    case SmoothingFailureReason::CuspHoldConstraint: return "cusp_hold_constraint";
    case SmoothingFailureReason::CollapsedSegment: return "collapsed_segment";
    case SmoothingFailureReason::MotionDirectionConstraint: return "motion_direction_constraint";
    case SmoothingFailureReason::PathOutOfBounds: return "path_out_of_bounds";
    case SmoothingFailureReason::FootprintCollision: return "footprint_collision";
    case SmoothingFailureReason::CurvatureConstraint: return "curvature_constraint";
    case SmoothingFailureReason::Unknown:
    default: return "unknown";
  }
}

inline std::string buildSmoothingFailureMessage(
  SmoothingFailureReason reason, const std::string & message, int failed_index = -1)
{
  std::string formatted = toSmoothingFailureReasonString(reason);
  if (failed_index >= 0) formatted += "@" + std::to_string(failed_index);
  formatted += ": " + message;
  return formatted;
}

struct SmoothingFailureInfo
{
  SmoothingFailureReason reason{SmoothingFailureReason::Unknown};
  std::string message{};
  int failed_index{-1};
  double actual_curvature{std::numeric_limits<double>::quiet_NaN()};
  double max_curvature{std::numeric_limits<double>::quiet_NaN()};
  double turning_radius{std::numeric_limits<double>::quiet_NaN()};
  double goal_longitudinal_error{std::numeric_limits<double>::quiet_NaN()};
  double goal_lateral_error{std::numeric_limits<double>::quiet_NaN()};
  double goal_longitudinal_tolerance{std::numeric_limits<double>::quiet_NaN()};
  double goal_lateral_tolerance{std::numeric_limits<double>::quiet_NaN()};

  std::string formattedMessage() const
  {
    return buildSmoothingFailureMessage(reason, message, failed_index);
  }
};

class InvalidPath : public std::runtime_error
{
public:
  explicit InvalidPath(const std::string & msg) : std::runtime_error(msg) {}
  ErrorCode code() const noexcept { return ErrorCode::InvalidPath; }
  const char * codeString() const noexcept { return toErrorCodeString(code()); }
};

class FailedToSmoothPath : public std::runtime_error
{
public:
  explicit FailedToSmoothPath(const std::string & msg) : std::runtime_error(msg) {}
  ErrorCode code() const noexcept { return ErrorCode::FailedToSmoothPath; }
  const char * codeString() const noexcept { return toErrorCodeString(code()); }
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

}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__EXCEPTIONS_HPP_
