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
