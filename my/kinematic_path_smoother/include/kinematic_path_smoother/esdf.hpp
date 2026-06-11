#ifndef KINEMATIC_PATH_SMOOTHER__ESDF_HPP_
#define KINEMATIC_PATH_SMOOTHER__ESDF_HPP_

#include "esdf_core/esdf.hpp"
#include "esdf_core/exceptions.hpp"

namespace kinematic_path_smoother
{

/// ESDF 计算和算法枚举复用 esdf_core，实现和 smoother 解耦。
using ESDF = esdf_core::ESDF;
using ESDFAlgorithm = esdf_core::ESDFAlgorithm;
using InvalidCostmap = esdf_core::InvalidCostmap;
using PrecomputedEsdfSizeMismatch = esdf_core::PrecomputedEsdfSizeMismatch;

}  // namespace kinematic_path_smoother

#endif  // KINEMATIC_PATH_SMOOTHER__ESDF_HPP_
