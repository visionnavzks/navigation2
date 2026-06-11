#ifndef SMOOTHER_CLOTHOID__SMOOTHER_REQUEST_HPP_
#define SMOOTHER_CLOTHOID__SMOOTHER_REQUEST_HPP_

#include <cstddef>
#include <vector>

#include "Eigen/Core"

#include "smoother_clothoid/costmap2d.hpp"
#include "smoother_clothoid/exceptions.hpp"
#include "smoother_clothoid/options.hpp"

namespace smoother_clothoid
{

struct SmootherResult
{
  std::vector<Eigen::Vector3d> candidate_path;
  std::vector<Eigen::Vector3d> smoothed_path;
  std::size_t optimized_knot_count{0};
  double target_spacing{0.0};
  bool success{false};
};

struct SmootherRequest
{
  const std::vector<Eigen::Vector3d> & path;
  const Eigen::Vector2d & start_dir;
  const Eigen::Vector2d & end_dir;
  const Costmap2D * costmap;
  const SmootherParams & params;
  const std::vector<double> * precomputed_esdf;
  SmoothingFailureInfo * failure;
};

}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__SMOOTHER_REQUEST_HPP_
