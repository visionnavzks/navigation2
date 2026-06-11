#include <cmath>
#include <exception>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include "kinematic_path_smoother/costmap2d.hpp"
#include "kinematic_path_smoother/kinematic_smoother.hpp"
#include "kinematic_path_smoother/options.hpp"

namespace nb = nanobind;
namespace kps = kinematic_path_smoother;

namespace
{

nb::sequence requireSequence(const nb::handle & handle, const char * name)
{
  if (!nb::isinstance<nb::sequence>(handle) || nb::isinstance<nb::str>(handle)) {
    throw nb::value_error((std::string(name) + " must be a numeric sequence").c_str());
  }
  return nb::cast<nb::sequence>(handle);
}

std::vector<Eigen::Vector3d> copyPath(const nb::handle & handle)
{
  nb::sequence sequence = requireSequence(handle, "path");
  std::vector<Eigen::Vector3d> path;
  path.reserve(nb::len(sequence));
  for (std::size_t i = 0; i < static_cast<std::size_t>(nb::len(sequence)); ++i) {
    nb::sequence point = requireSequence(sequence[i], "path entry");
    if (nb::len(point) != 3) {
      throw nb::value_error("path entries must be [x, y, direction_sign]");
    }
    path.emplace_back(
      nb::cast<double>(point[0]),
      nb::cast<double>(point[1]),
      nb::cast<double>(point[2]));
  }
  return path;
}

Eigen::Vector2d copyVector2(const nb::handle & handle, const char * name)
{
  nb::sequence sequence = requireSequence(handle, name);
  if (nb::len(sequence) != 2) {
    throw nb::value_error((std::string(name) + " must contain exactly two values").c_str());
  }
  return {nb::cast<double>(sequence[0]), nb::cast<double>(sequence[1])};
}

template<typename T>
void assignIfPresent(const nb::dict & dict, const char * key, T & out)
{
  if (dict.contains(key)) {
    out = nb::cast<T>(dict[key]);
  }
}

kps::SmootherParams parseParams(const nb::dict & dict)
{
  kps::SmootherParams params;
  assignIfPresent(dict, "model_weight", params.model_weight);
  assignIfPresent(dict, "reference_weight", params.reference_weight);
  assignIfPresent(dict, "obstacle_weight", params.obstacle_weight);
  assignIfPresent(dict, "cusp_obstacle_weight", params.cusp_obstacle_weight);
  assignIfPresent(dict, "curvature_weight", params.curvature_weight);
  assignIfPresent(dict, "curvature_rate_weight", params.curvature_rate_weight);
  assignIfPresent(dict, "spacing_weight", params.spacing_weight);
  assignIfPresent(dict, "length_weight", params.length_weight);
  assignIfPresent(dict, "fix_weight", params.fix_weight);
  assignIfPresent(dict, "max_curvature", params.max_curvature);
  assignIfPresent(dict, "max_segment_length", params.max_segment_length);
  assignIfPresent(dict, "max_reference_deviation", params.max_reference_deviation);
  assignIfPresent(dict, "max_time", params.max_time);
  assignIfPresent(dict, "use_exact_esdf", params.use_exact_esdf);
  assignIfPresent(dict, "obstacle_safe_distance", params.obstacle_safe_distance);
  assignIfPresent(dict, "footprint_radius", params.footprint_radius);
  assignIfPresent(dict, "footprint_points", params.footprint_points);
  assignIfPresent(dict, "path_downsampling_factor", params.path_downsampling_factor);
  assignIfPresent(dict, "path_upsampling_factor", params.path_upsampling_factor);
  assignIfPresent(dict, "reversing_enabled", params.reversing_enabled);
  assignIfPresent(dict, "keep_start_orientation", params.keep_start_orientation);
  assignIfPresent(dict, "keep_goal_orientation", params.keep_goal_orientation);
  assignIfPresent(dict, "goal_longitudinal_tolerance", params.goal_longitudinal_tolerance);
  assignIfPresent(dict, "goal_lateral_tolerance", params.goal_lateral_tolerance);
  assignIfPresent(dict, "goal_orientation_tolerance", params.goal_orientation_tolerance);
  return params;
}

kps::OptimizerParams parseOptimizer(const nb::dict & dict)
{
  kps::OptimizerParams params;
  assignIfPresent(dict, "debug", params.debug);
  assignIfPresent(dict, "max_iterations", params.max_iterations);
  assignIfPresent(dict, "function_tolerance", params.function_tolerance);
  assignIfPresent(dict, "gradient_tolerance", params.gradient_tolerance);
  assignIfPresent(dict, "parameter_tolerance", params.parameter_tolerance);
  if (dict.contains("linear_solver")) {
    params.linear_solver = kps::OptimizerParams::fromString(nb::cast<std::string>(dict["linear_solver"]));
  }
  return params;
}

nb::list pathToPython(const std::vector<Eigen::Vector3d> & path)
{
  nb::list out;
  for (const auto & point : path) {
    nb::list item;
    item.append(point.x());
    item.append(point.y());
    item.append(point.z());
    out.append(item);
  }
  return out;
}

nb::dict failureToDict(const kps::FailureInfo & failure)
{
  nb::dict out;
  out["reason"] = kps::toString(failure.reason);
  out["message"] = failure.message;
  out["index"] = failure.index;
  out["actual_curvature"] = failure.actual_curvature;
  out["max_curvature"] = failure.max_curvature;
  return out;
}

nb::dict smoothPath(
  const nb::handle & path_handle,
  const nb::handle & start_direction_handle,
  const nb::handle & goal_direction_handle,
  unsigned int size_x,
  unsigned int size_y,
  double resolution,
  double origin_x,
  double origin_y,
  const std::vector<unsigned char> & costs,
  const nb::dict & smoother_params,
  const nb::dict & optimizer_params)
{
  if (costs.size() != static_cast<std::size_t>(size_x) * size_y) {
    throw nb::value_error("costmap size does not match size_x * size_y");
  }

  std::vector<unsigned char> mutable_costs = costs;
  kps::Costmap2D costmap(size_x, size_y, resolution, origin_x, origin_y, mutable_costs.data());

  kps::KinematicPathSmoother smoother;
  smoother.initialize(parseOptimizer(optimizer_params));

  const std::vector<Eigen::Vector3d> path = copyPath(path_handle);
  const Eigen::Vector2d start_direction = copyVector2(start_direction_handle, "start_direction");
  const Eigen::Vector2d goal_direction = copyVector2(goal_direction_handle, "goal_direction");
  const kps::SmootherParams params = parseParams(smoother_params);
  kps::FailureInfo failure;

  const kps::SmoothingRequest request{
    path,
    start_direction,
    goal_direction,
    &costmap,
    params,
    nullptr,
    &failure};
  const kps::SmoothingResult result = smoother.smooth(request);

  nb::dict out;
  out["success"] = result.success;
  out["path"] = pathToPython(result.path);
  out["optimized_path"] = pathToPython(result.optimized_path);
  out["optimized_knot_count"] = result.optimized_knot_count;
  out["target_spacing"] = result.target_spacing;
  out["failure"] = result.success ? nb::object(nb::none()) : nb::object(failureToDict(failure));
  return out;
}

}  // namespace

NB_MODULE(py_kinematic_path_smoother, m)
{
  m.doc() = "nanobind bindings for the kinematic path smoother web demo";
  m.def(
    "smooth_path",
    &smoothPath,
    nb::arg("path"),
    nb::arg("start_direction"),
    nb::arg("goal_direction"),
    nb::arg("size_x"),
    nb::arg("size_y"),
    nb::arg("resolution"),
    nb::arg("origin_x"),
    nb::arg("origin_y"),
    nb::arg("costs"),
    nb::arg("smoother_params"),
    nb::arg("optimizer_params"));
}
