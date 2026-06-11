#ifndef KINEMATIC_PATH_SMOOTHER__EXCEPTIONS_HPP_
#define KINEMATIC_PATH_SMOOTHER__EXCEPTIONS_HPP_

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "esdf_core/exceptions.hpp"

namespace kinematic_path_smoother
{

/// 复用 esdf_core 的地图异常类型，避免在 smoother 包里重复定义地图错误。
using InvalidCostmap = esdf_core::InvalidCostmap;
using PrecomputedEsdfSizeMismatch = esdf_core::PrecomputedEsdfSizeMismatch;

/// 稳定错误码，用于 API 边界或绑定层做粗粒度分类。
enum class ErrorCode : uint16_t
{
  InvalidPath = 1001,
  SmoothingFailed = 2001,
  InvalidCostmap = 3001,
  PrecomputedEsdfSizeMismatch = 3002,
};

/// 平滑失败的细分原因，主要用于后验校验和求解失败诊断。
enum class FailureReason : uint16_t
{
  Unknown = 0,
  SolverFailure = 1,
  InvalidStateVector = 2,
  NonFiniteState = 3,
  StartConstraint = 4,
  GoalConstraint = 5,
  CuspConstraint = 6,
  CollapsedSegment = 7,
  MotionDirection = 8,
  CurvatureLimit = 9,
  OutOfBounds = 10,
  Collision = 11,
};

/// 将失败原因转为稳定字符串，方便日志、UI 和 Python 绑定复用。
inline const char * toString(FailureReason reason)
{
  switch (reason) {
    case FailureReason::SolverFailure:
      return "solver_failure";
    case FailureReason::InvalidStateVector:
      return "invalid_state_vector";
    case FailureReason::NonFiniteState:
      return "non_finite_state";
    case FailureReason::StartConstraint:
      return "start_constraint";
    case FailureReason::GoalConstraint:
      return "goal_constraint";
    case FailureReason::CuspConstraint:
      return "cusp_constraint";
    case FailureReason::CollapsedSegment:
      return "collapsed_segment";
    case FailureReason::MotionDirection:
      return "motion_direction";
    case FailureReason::CurvatureLimit:
      return "curvature_limit";
    case FailureReason::OutOfBounds:
      return "out_of_bounds";
    case FailureReason::Collision:
      return "collision";
    case FailureReason::Unknown:
    default:
      return "unknown";
  }
}

/// 非异常路径的结构化失败信息。
///
/// 调用方传入 SmoothingRequest::failure 时，smooth() 会填充该结构并返回
/// success=false；不传 failure 时相同错误会抛 FailedToSmoothPath。
struct FailureInfo
{
  /// 失败分类。
  FailureReason reason{FailureReason::Unknown};
  /// 面向开发者的失败说明。
  std::string message{};
  /// 失败关联的 knot/segment 索引；无索引时为 -1。
  int index{-1};
  /// 曲率失败时记录的实际曲率，单位 1/m。
  double actual_curvature{std::numeric_limits<double>::quiet_NaN()};
  /// 曲率失败时记录的曲率上限，单位 1/m。
  double max_curvature{std::numeric_limits<double>::quiet_NaN()};

  /// 拼出可直接打印的失败字符串。
  std::string formattedMessage() const
  {
    std::string text = toString(reason);
    if (index >= 0) {
      text += "@" + std::to_string(index);
    }
    if (!message.empty()) {
      text += ": " + message;
    }
    return text;
  }
};

/// 输入路径非法，例如点数不足。
class InvalidPath : public std::runtime_error
{
public:
  explicit InvalidPath(const std::string & message)
  : std::runtime_error(message) {}

  ErrorCode code() const noexcept {return ErrorCode::InvalidPath;}
};

/// 求解器或后验校验未能产出可交付路径。
class FailedToSmoothPath : public std::runtime_error
{
public:
  explicit FailedToSmoothPath(const std::string & message)
  : std::runtime_error(message) {}

  ErrorCode code() const noexcept {return ErrorCode::SmoothingFailed;}
};

/// 统一处理“写 failure 返回 false”和“抛异常”两种失败语义。
inline bool failOrThrow(
  FailureInfo * failure,
  FailureReason reason,
  const std::string & message,
  int index = -1)
{
  if (failure != nullptr) {
    failure->reason = reason;
    failure->message = message;
    failure->index = index;
    return false;
  }

  FailureInfo local;
  local.reason = reason;
  local.message = message;
  local.index = index;
  throw FailedToSmoothPath(local.formattedMessage());
}

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__EXCEPTIONS_HPP_
