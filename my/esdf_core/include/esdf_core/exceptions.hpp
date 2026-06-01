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

#ifndef ESDF_CORE__EXCEPTIONS_HPP_
#define ESDF_CORE__EXCEPTIONS_HPP_

#include <stdexcept>
#include <string>

namespace esdf_core
{

/**
 * @brief Thrown when a null or otherwise invalid costmap is passed to ESDF or footprint
 *        queries.
 */
class InvalidCostmap : public std::runtime_error
{
public:
  explicit InvalidCostmap(const std::string & msg)
  : std::runtime_error(msg) {}
};

/**
 * @brief Thrown when a precomputed ESDF is passed in but its size does not match the
 *        costmap dimensions (size_x * size_y).
 */
class PrecomputedEsdfSizeMismatch : public std::runtime_error
{
public:
  explicit PrecomputedEsdfSizeMismatch(const std::string & msg)
  : std::runtime_error(msg) {}
};

}  // namespace esdf_core

#endif  // ESDF_CORE__EXCEPTIONS_HPP_
