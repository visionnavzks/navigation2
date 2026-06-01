// Copyright (c) 2026, Dexory (Tony Najjar)
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

#ifndef COSTMAP_2D__DISTANCE_TRANSFORM_HPP_
#define COSTMAP_2D__DISTANCE_TRANSFORM_HPP_

#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <Eigen/Core>

namespace costmap_2d
{

using MatrixXfRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class DistanceTransform
{
public:
  static constexpr float DT_INF = std::numeric_limits<float>::max();

  static void distanceTransform1D(
    const float * f, float * d, int n,
    int * v, float * z)
  {
    if (!f || !d || !v || !z || n <= 0) {
      return;
    }

    int k = 0;
    v[0] = 0;
    z[0] = -DT_INF;
    z[1] = DT_INF;

    for (int q = 1; q < n; q++) {
      float s = (f[q] - f[v[k]] + static_cast<float>(q * q - v[k] * v[k])) /
        (2.0f * static_cast<float>(q - v[k]));
      while (s <= z[k]) {
        k--;
        s = (f[q] - f[v[k]] + static_cast<float>(q * q - v[k] * v[k])) /
          (2.0f * static_cast<float>(q - v[k]));
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k + 1] = DT_INF;
    }

    k = 0;
    for (int q = 0; q < n; q++) {
      while (z[k + 1] < static_cast<float>(q)) {
        k++;
      }
      const int diff = q - v[k];
      d[q] = static_cast<float>(diff * diff) + f[v[k]];
    }
  }

  static void distanceTransform2D(MatrixXfRM & img, int height, int width)
  {
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int x = 0; x < width; x++) {
      std::vector<float> f(height);
      std::vector<float> d(height);
      std::vector<int> v(height);
      std::vector<float> z(height + 1);

      for (int y = 0; y < height; y++) {
        f[y] = img(y, x);
      }

      distanceTransform1D(f.data(), d.data(), height, v.data(), z.data());

      for (int y = 0; y < height; y++) {
        img(y, x) = d[y];
      }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int y = 0; y < height; y++) {
      std::vector<float> f(width);
      std::vector<float> d(width);
      std::vector<int> v(width);
      std::vector<float> z(width + 1);

      for (int x = 0; x < width; x++) {
        f[x] = img(y, x);
      }

      distanceTransform1D(f.data(), d.data(), width, v.data(), z.data());

      for (int x = 0; x < width; x++) {
        img(y, x) = d[x];
      }
    }

    img = img.cwiseSqrt();
  }
};

}  // namespace costmap_2d

#endif  // COSTMAP_2D__DISTANCE_TRANSFORM_HPP_
