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

#include <stdexcept>
#include <string>

namespace constrained_smoother
{

/**
 * @class InvalidPath
 * @brief Thrown when the input path is invalid (e.g. too short).
 */
class InvalidPath : public std::runtime_error
{
public:
  explicit InvalidPath(const std::string & msg)
  : std::runtime_error(msg) {}
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
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__EXCEPTIONS_HPP_
