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

#ifndef CONSTRAINED_SMOOTHER__STATE_LAYOUT_HPP_
#define CONSTRAINED_SMOOTHER__STATE_LAYOUT_HPP_

#include <cstddef>
#include <vector>

namespace constrained_smoother
{

struct KinematicStateLayout
{
  static constexpr size_t Size{5};
  static constexpr size_t X{0};
  static constexpr size_t Y{1};
  static constexpr size_t Theta{2};
  static constexpr size_t Kappa{3};
  static constexpr size_t Ds{4};

  static constexpr double EnabledEpsilon{1e-9};
  static constexpr double GeometryEpsilon{1e-6};
  static constexpr double PointEpsilon{1e-9};

  static constexpr size_t offset(size_t index)
  {
    return Size * index;
  }

  static double * data(std::vector<double> & variables, size_t index)
  {
    return variables.data() + offset(index);
  }

  static const double * data(const std::vector<double> & variables, size_t index)
  {
    return variables.data() + offset(index);
  }

  static double * data(double * variables, size_t index)
  {
    return variables + offset(index);
  }

  static const double * data(const double * variables, size_t index)
  {
    return variables + offset(index);
  }
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__STATE_LAYOUT_HPP_
