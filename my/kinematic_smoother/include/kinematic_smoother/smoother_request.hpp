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

#ifndef KINEMATIC_SMOOTHER__SMOOTHER_REQUEST_HPP_
#define KINEMATIC_SMOOTHER__SMOOTHER_REQUEST_HPP_

#include <vector>

#include "Eigen/Core"

#include "kinematic_smoother/costmap2d.hpp"
#include "kinematic_smoother/exceptions.hpp"
#include "kinematic_smoother/options.hpp"

namespace kinematic_smoother
{

/// 单次 smooth() 调用共享的不可拥有请求视图。
///
/// 顶层 smoother 会在栈上构造它，再把它传给内部 Run 对象；这样多层 helper
/// 不需要继续传递同一串参数，也不会误以为自己拥有 path / costmap / params。
struct SmootherRequest
{
  /// 原地修改的路径缓冲区。
  /// 输入时第三个分量是 direction_sign；成功输出后会被改写成 yaw。
  std::vector<Eigen::Vector3d> & path;
  /// 起点切向方向，始终按向量语义解释，而不是 yaw 标量。
  const Eigen::Vector2d & start_dir;
  /// 终点切向方向，始终按向量语义解释，而不是 yaw 标量。
  const Eigen::Vector2d & end_dir;
  /// 本次优化使用的 costmap，上层需保证生命周期覆盖整个调用过程。
  const Costmap2D * costmap;
  /// 本次平滑的残差权重、边界约束和运行参数。
  const SmootherParams & params;
  /// 可选的预计算 ESDF；若为空则由构建器根据 costmap 现场生成。
  const std::vector<double> * precomputed_esdf;
  /// 可选失败回传槽；为空时失败通常通过异常传播。
  SmoothingFailureInfo * failure;
};

}  // namespace kinematic_smoother

#endif  // KINEMATIC_SMOOTHER__SMOOTHER_REQUEST_HPP_