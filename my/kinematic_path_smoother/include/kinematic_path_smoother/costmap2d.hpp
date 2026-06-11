#ifndef KINEMATIC_PATH_SMOOTHER__COSTMAP2D_HPP_
#define KINEMATIC_PATH_SMOOTHER__COSTMAP2D_HPP_

#include "esdf_core/costmap2d.hpp"

namespace kinematic_path_smoother
{

/// 平滑器使用的轻量 costmap 类型；实际实现由 esdf_core 提供。
using Costmap2D = esdf_core::Costmap2D;

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__COSTMAP2D_HPP_
