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

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_REQUEST_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_REQUEST_HPP_

#include <cstddef>
#include <vector>

#include "Eigen/Core"

#include "kinematic_smoother/costmap2d.hpp"
#include "kinematic_smoother/exceptions.hpp"
#include "kinematic_smoother/options.hpp"

namespace kinematic_smoother
{

/// 一次平滑调用产出的显式结果对象。
///
/// 与旧接口“原地改写输入 path”不同，这里把候选路径、最终输出路径和诊断元数据
/// 一并显式返回，避免调用方再依赖隐藏副作用或实例级 getter。
struct SmootherResult
{
  /// 求解后直接从状态向量解包得到的候选路径。
  /// 若后验校验失败，这个字段仍可保留用于诊断或可视化。
  std::vector<Eigen::Vector3d> candidate_path;
  /// 通过后验校验后按运动学模型上采样的最终输出路径。
  std::vector<Eigen::Vector3d> smoothed_path;
  /// 与 smoothed_path 等长的输出采样点曲率 kappa。
  std::vector<double> smoothed_curvatures;
  /// 与 smoothed_path 等长的输出采样点曲率变化率 dk/ds。
  std::vector<double> smoothed_curvature_rates;
  /// 本次参与优化的状态点数量。
  std::size_t optimized_knot_count{0};
  /// 本次优化使用的目标 knot 间距（米）。
  double target_spacing{0.0};
  /// 是否得到了可交付的最终平滑路径。
  bool success{false};
};

/// 单次 smooth() 调用共享的不可拥有请求视图。
///
/// 顶层 smoother 会在栈上构造它，再把它传给内部 Run 对象；这样多层 helper
/// 不需要继续传递同一串参数，也不会误以为自己拥有 path / costmap / params。
struct SmootherRequest
{
  /// 只读输入路径；第三个分量始终表示 direction_sign（+1/-1）。
  /// smooth() 不再原地改写它，而是通过 SmootherResult 返回候选 / 最终路径。
  const std::vector<Eigen::Vector3d> & path;
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

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_REQUEST_HPP_
