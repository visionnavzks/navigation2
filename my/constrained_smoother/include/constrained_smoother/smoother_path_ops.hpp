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
// limitations under the License. Reserved.

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_PATH_OPS_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_PATH_OPS_HPP_

#include <cmath>
#include <vector>

#include "Eigen/Core"

#include "constrained_smoother/options.hpp"
#include "constrained_smoother/smoother_validator.hpp"
#include "constrained_smoother/utils.hpp"

namespace constrained_smoother
{

/// 几何版 smoother 的路径侧 helper。
///
/// 它只处理“路径副本如何锚定”和“优化后结果如何重建”，不参与残差连接或
/// 求解器配置，因此和问题构建逻辑分离。
class SmootherPathOps
{
public:
  SmootherPathOps(
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params)
  : start_dir_(start_dir), end_dir_(end_dir), params_(params)
  {
  }

  void initializeOptimizationPath(
    const std::vector<Eigen::Vector3d> & path,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized) const
  {
    // 复制输入并尽早施加端点朝向锚定，后续问题构建直接围绕这份工作副本展开。
    path_optim = path;
    applyEndpointOrientationAnchors(path_optim);
    optimized = std::vector<bool>(path.size(), false);
    optimized[0] = true;
  }

  void populateOutput(
    const std::vector<Eigen::Vector3d> & path_optim,
    const std::vector<bool> & optimized,
    std::vector<Eigen::Vector3d> & path) const
  {
    // 只遍历保留下来的关键控制点；若中间被下采样跳过，则在这里补点并恢复 yaw。
    path.clear();
    if (params_.path_upsampling_factor > 1) {
      path.reserve(params_.path_upsampling_factor * (path_optim.size() - 1) + 1);
    }
    int last_i = 0;
    int prelast_i = -1;
    Eigen::Vector2d prelast_dir = {0, 0};
    for (int i = 1; i <= static_cast<int>(path_optim.size()); i++) {
      if (i == static_cast<int>(path_optim.size()) || optimized[i]) {
        if (prelast_i != -1) {
          Eigen::Vector2d last_dir;
          auto & prelast = path_optim[prelast_i];
          auto & last = path_optim[last_i];

          if (i < static_cast<int>(path_optim.size())) {
            auto & current = path_optim[i];
            Eigen::Vector2d tangent_dir_val = tangentDir<double>(
              prelast.block<2, 1>(0, 0),
              last.block<2, 1>(0, 0),
              current.block<2, 1>(0, 0),
              prelast[2] * last[2] < 0);

            last_dir =
              tangent_dir_val.dot((current - last).block<2, 1>(0, 0) * last[2]) >= 0 ?
              tangent_dir_val :
              -tangent_dir_val;
            last_dir.normalize();
          } else if (params_.keep_goal_orientation) {
            last_dir = end_dir_;
          } else {
            last_dir = (last - prelast).block<2, 1>(0, 0) * last[2];
            last_dir.normalize();
          }
          double last_angle = atan2(last_dir[1], last_dir[0]);

          int interp_cnt = (last_i - prelast_i) * params_.path_upsampling_factor - 1;
          if (interp_cnt > 0) {
            Eigen::Vector2d last_pt = last.block<2, 1>(0, 0);
            Eigen::Vector2d prelast_pt = prelast.block<2, 1>(0, 0);
            double dist = (last_pt - prelast_pt).norm();
            Eigen::Vector2d pt1 = prelast_pt + prelast_dir * dist * 0.4 * prelast[2];
            Eigen::Vector2d pt2 = last_pt - last_dir * dist * 0.4 * prelast[2];
            for (int j = 1; j <= interp_cnt; j++) {
              double interp = j / static_cast<double>(interp_cnt + 1);
              Eigen::Vector2d pt = cubicBezier(prelast_pt, pt1, pt2, last_pt, interp);
              path.emplace_back(pt[0], pt[1], 0.0);
            }
          }
          path.emplace_back(last[0], last[1], last_angle);

          for (size_t j = path.size() - 1 - interp_cnt; j < path.size() - 1; j++) {
            Eigen::Vector2d tangent_dir_val = tangentDir<double>(
              path[j - 1].block<2, 1>(0, 0),
              path[j].block<2, 1>(0, 0),
              path[j + 1].block<2, 1>(0, 0),
              false);
            tangent_dir_val =
              tangent_dir_val.dot(
              (path[j + 1] - path[j]).block<2, 1>(0, 0) * prelast[2]) >= 0 ?
              tangent_dir_val :
              -tangent_dir_val;
            path[j][2] = atan2(tangent_dir_val[1], tangent_dir_val[0]);
          }

          prelast_dir = last_dir;
        } else {
          auto & start = path_optim[0];
          Eigen::Vector2d dir = params_.keep_start_orientation ?
            start_dir_ :
            ((path_optim[i] - start).block<2, 1>(0, 0) * start[2]).normalized();
          path.emplace_back(start[0], start[1], atan2(dir[1], dir[0]));
          prelast_dir = dir;
        }
        prelast_i = last_i;
        last_i = i;
      }
    }
  }

private:
  void applyEndpointOrientationAnchors(std::vector<Eigen::Vector3d> & path_optim) const
  {
    if (path_optim.size() < 3) {
      return;
    }

    if (params_.keep_start_orientation) {
      const double start_segment_len =
        (path_optim[1] - path_optim[0]).template block<2, 1>(0, 0).norm();
      path_optim[1].template block<2, 1>(0, 0) =
        path_optim[0].template block<2, 1>(0, 0) +
        SmootherValidator::normalizedDirection(start_dir_) * start_segment_len;
    }

    if (params_.keep_goal_orientation) {
      const size_t goal_index = path_optim.size() - 1;
      const size_t pregoal_index = goal_index - 1;
      const double goal_segment_len =
        (path_optim[goal_index] - path_optim[pregoal_index]).template block<2, 1>(0, 0).norm();
      Eigen::Vector2d anchored_pregoal =
        path_optim[goal_index].template block<2, 1>(0, 0) -
        SmootherValidator::normalizedDirection(end_dir_) * goal_segment_len;

      if (params_.keep_start_orientation && pregoal_index == 1) {
        path_optim[pregoal_index].template block<2, 1>(0, 0) =
          0.5 * (path_optim[pregoal_index].template block<2, 1>(0, 0) + anchored_pregoal);
      } else {
        path_optim[pregoal_index].template block<2, 1>(0, 0) = anchored_pregoal;
      }
    }
  }

  static Eigen::Vector2d cubicBezier(
    Eigen::Vector2d & pt0, Eigen::Vector2d & pt1,
    Eigen::Vector2d & pt2, Eigen::Vector2d & pt3, double mu)
  {
    Eigen::Vector2d a, b, c, pt;

    c[0] = 3 * (pt1[0] - pt0[0]);
    c[1] = 3 * (pt1[1] - pt0[1]);
    b[0] = 3 * (pt2[0] - pt1[0]) - c[0];
    b[1] = 3 * (pt2[1] - pt1[1]) - c[1];
    a[0] = pt3[0] - pt0[0] - c[0] - b[0];
    a[1] = pt3[1] - pt0[1] - c[1] - b[1];

    pt[0] = a[0] * mu * mu * mu + b[0] * mu * mu + c[0] * mu + pt0[0];
    pt[1] = a[1] * mu * mu * mu + b[1] * mu * mu + c[1] * mu + pt0[1];

    return pt;
  }

  const Eigen::Vector2d & start_dir_;
  const Eigen::Vector2d & end_dir_;
  const SmootherParams & params_;
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_PATH_OPS_HPP_