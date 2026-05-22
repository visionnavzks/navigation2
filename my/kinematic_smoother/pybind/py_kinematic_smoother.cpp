// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "kinematic_smoother/astar_esdf.hpp"
#include "kinematic_smoother/costmap2d.hpp"
#include "kinematic_smoother/kinematic_smoother.hpp"
#include "kinematic_smoother/options.hpp"

#include "kinematic_smoother/exceptions.hpp"
#include "kinematic_smoother/esdf.hpp"

#include <vector>
#include <cmath>
#include <typeinfo>

namespace py = pybind11;

namespace
{

// ---- Structured error parsing helpers ----

struct ParsedSmoothingFailure
{
  py::object reason{py::none()};
  py::object details{py::none()};
  std::string message;
};

// ---- Python input normalization helpers ----

py::sequence require_sequence(const py::handle & handle, const char * argument_name)
{
  if (!py::isinstance<py::sequence>(handle) || py::isinstance<py::str>(handle)) {
    throw py::value_error(std::string(argument_name) + " must be a numeric sequence");
  }
  return py::reinterpret_borrow<py::sequence>(handle);
}

Eigen::Vector2d copy_vector2d(const py::handle & handle, const char * argument_name)
{
  py::sequence sequence = require_sequence(handle, argument_name);
  if (py::len(sequence) != 2) {
    throw py::value_error(std::string(argument_name) + " must contain exactly 2 values");
  }

  return Eigen::Vector2d(
    py::cast<double>(sequence[0]),
    py::cast<double>(sequence[1]));
}

std::vector<Eigen::Vector3d> copy_path3d(const py::handle & handle, const char * argument_name)
{
  py::sequence outer = require_sequence(handle, argument_name);
  std::vector<Eigen::Vector3d> path;
  path.reserve(py::len(outer));

  for (size_t index = 0; index < static_cast<size_t>(py::len(outer)); ++index) {
    py::handle item = outer[index];
    py::sequence point = require_sequence(item, argument_name);
    if (py::len(point) != 3) {
      throw py::value_error(
              std::string(argument_name) + " entries must contain exactly 3 values");
    }

    path.emplace_back(
      py::cast<double>(point[0]),
      py::cast<double>(point[1]),
      py::cast<double>(point[2]));
  }

  return path;
}

const kinematic_smoother::Costmap2D * copy_optional_costmap(const py::handle & handle)
{
  if (handle.is_none()) {
    return nullptr;
  }

  return &py::cast<const kinematic_smoother::Costmap2D &>(handle);
}

// ---- Failure parsing / folding helpers ----

// Structured result schema used by all try_* bindings:
//   {
//     "ok": bool,
//     "path": list | None,
//     "error_code": str | None,
//     "error_message": str | None,
//     "error_reason": str | None,
//     "error_details": {"failed_index": int} | None,
//   }
// The goal is to keep Python and web callers on one stable error surface even though
// native failures may originate as C++ exceptions or SmoothingFailureInfo payloads.

bool is_known_smoothing_reason(const std::string & reason)
{
  return reason == "unknown" ||
    reason == "solver_rejected_solution" ||
    reason == "no_cost_improvement" ||
    reason == "invalid_state_vector" ||
    reason == "nonfinite_state" ||
    reason == "start_position_constraint" ||
    reason == "start_orientation_constraint" ||
    reason == "goal_position_constraint" ||
    reason == "goal_orientation_constraint" ||
    reason == "cusp_hold_constraint" ||
    reason == "collapsed_segment" ||
    reason == "motion_direction_constraint" ||
    reason == "path_out_of_bounds" ||
    reason == "footprint_collision" ||
    reason == "curvature_constraint";
}

ParsedSmoothingFailure parse_smoothing_failure_message(const std::string & raw_message)
{
  ParsedSmoothingFailure parsed;
  parsed.message = raw_message;

  const size_t separator = raw_message.find(": ");
  if (separator == std::string::npos) {
    return parsed;
  }

  const std::string prefix = raw_message.substr(0, separator);
  const size_t at = prefix.find('@');
  const std::string reason = prefix.substr(0, at);
  if (!is_known_smoothing_reason(reason)) {
    return parsed;
  }

  parsed.reason = py::str(reason);
  parsed.message = raw_message.substr(separator + 2);

  if (at != std::string::npos && at + 1 < prefix.size()) {
    try {
      const int failed_index = std::stoi(prefix.substr(at + 1));
      py::dict details;
      details["failed_index"] = py::int_(failed_index);
      parsed.details = details;
    } catch (const std::exception &) {
    }
  }

  return parsed;
}

template<typename ErrorT>
py::dict make_error_result_base(const ErrorT & error)
{
  py::dict result;
  result["ok"] = false;
  result["path"] = py::none();
  result["error_code"] = py::str(error.codeString());
  result["error_message"] = py::str(error.what());
  result["error_reason"] = py::none();
  result["error_details"] = py::none();
  return result;
}

template<typename ErrorT>
py::dict make_error_result(const ErrorT & error)
{
  return make_error_result_base(error);
}

py::dict make_error_result(const kinematic_smoother::FailedToSmoothPath & error)
{
  py::dict result = make_error_result_base(error);
  const ParsedSmoothingFailure parsed = parse_smoothing_failure_message(error.what());
  result["error_message"] = py::str(parsed.message);
  result["error_reason"] = parsed.reason;
  result["error_details"] = parsed.details;
  return result;
}

py::dict make_error_result(
  const kinematic_smoother::SmoothingFailureInfo & failure,
  const py::object & path = py::none())
{
  py::dict result;
  result["ok"] = false;
  result["path"] = path;
  result["error_code"] = py::str(
    kinematic_smoother::toErrorCodeString(kinematic_smoother::ErrorCode::FailedToSmoothPath));
  result["error_message"] = py::str(failure.message);
  result["error_reason"] = py::str(
    kinematic_smoother::toSmoothingFailureReasonString(failure.reason));
  if (
    failure.failed_index >= 0 ||
    std::isfinite(failure.actual_curvature) ||
    std::isfinite(failure.max_curvature) ||
    std::isfinite(failure.turning_radius) ||
    std::isfinite(failure.goal_longitudinal_error) ||
    std::isfinite(failure.goal_lateral_error) ||
    std::isfinite(failure.goal_longitudinal_tolerance) ||
    std::isfinite(failure.goal_lateral_tolerance))
  {
    py::dict details;
    if (failure.failed_index >= 0) {
      details["failed_index"] = py::int_(failure.failed_index);
    }
    if (std::isfinite(failure.actual_curvature)) {
      details["actual_curvature"] = py::float_(failure.actual_curvature);
    }
    if (std::isfinite(failure.max_curvature)) {
      details["max_curvature"] = py::float_(failure.max_curvature);
    }
    if (std::isfinite(failure.turning_radius)) {
      details["turning_radius"] = py::float_(failure.turning_radius);
    }
    if (std::isfinite(failure.actual_curvature) && std::isfinite(failure.max_curvature)) {
      details["curvature_excess"] = py::float_(failure.actual_curvature - failure.max_curvature);
    }
    if (std::isfinite(failure.goal_longitudinal_error)) {
      details["goal_longitudinal_error"] = py::float_(failure.goal_longitudinal_error);
    }
    if (std::isfinite(failure.goal_lateral_error)) {
      details["goal_lateral_error"] = py::float_(failure.goal_lateral_error);
    }
    if (std::isfinite(failure.goal_longitudinal_tolerance)) {
      details["goal_longitudinal_tolerance"] = py::float_(failure.goal_longitudinal_tolerance);
    }
    if (std::isfinite(failure.goal_lateral_tolerance)) {
      details["goal_lateral_tolerance"] = py::float_(failure.goal_lateral_tolerance);
    }
    result["error_details"] = details;
  } else {
    result["error_details"] = py::none();
  }
  return result;
}

PyObject * make_python_smoothing_failure(const kinematic_smoother::SmoothingFailureInfo & failure)
{
  PyErr_SetString(
    PyExc_RuntimeError,
    (std::string(kinematic_smoother::toErrorCodeString(
       kinematic_smoother::ErrorCode::FailedToSmoothPath)) +
    ": " + failure.formattedMessage()).c_str());
  return nullptr;
}

// ---- Exception-safe result wrapper for try_* style APIs ----

template<typename Fn>
py::dict invoke_with_result(Fn && fn)
{
  try {
    py::dict result;
    result["ok"] = true;
    result["path"] = fn();
    result["error_code"] = py::none();
    result["error_message"] = py::none();
    result["error_reason"] = py::none();
    result["error_details"] = py::none();
    return result;
  } catch (const kinematic_smoother::InvalidPath & error) {
    return make_error_result(error);
  } catch (const kinematic_smoother::FailedToSmoothPath & error) {
    return make_error_result(error);
  } catch (const kinematic_smoother::InvalidCostmap & error) {
    return make_error_result(error);
  } catch (const kinematic_smoother::PrecomputedEsdfSizeMismatch & error) {
    return make_error_result(error);
  } catch (const std::exception & error) {
    if (const auto * invalid_path = dynamic_cast<const kinematic_smoother::InvalidPath *>(&error)) {
      return make_error_result(*invalid_path);
    }
    if (const auto * failed = dynamic_cast<const kinematic_smoother::FailedToSmoothPath *>(&error)) {
      return make_error_result(*failed);
    }
    if (const auto * invalid_costmap = dynamic_cast<const kinematic_smoother::InvalidCostmap *>(&error)) {
      return make_error_result(*invalid_costmap);
    }
    if (
      const auto * size_mismatch =
      dynamic_cast<const kinematic_smoother::PrecomputedEsdfSizeMismatch *>(&error))
    {
      return make_error_result(*size_mismatch);
    }

    py::dict result;
    result["ok"] = false;
    result["path"] = py::none();
    result["error_code"] = py::none();
    result["error_message"] = py::str(error.what());
    result["error_reason"] = py::none();
    result["error_details"] = py::none();
    return result;
  }
}

}  // namespace

PYBIND11_MODULE(py_kinematic_smoother, m)
{
  m.doc() = "Python bindings for the kinematic_smoother C++ library";

  // ---- Stable error-code surface ----

  py::enum_<kinematic_smoother::ErrorCode>(m, "ErrorCode")
    .value("INVALID_PATH", kinematic_smoother::ErrorCode::InvalidPath)
    .value("FAILED_TO_SMOOTH_PATH", kinematic_smoother::ErrorCode::FailedToSmoothPath)
    .value("INVALID_COSTMAP", kinematic_smoother::ErrorCode::InvalidCostmap)
    .value(
      "PRECOMPUTED_ESDF_SIZE_MISMATCH",
      kinematic_smoother::ErrorCode::PrecomputedEsdfSizeMismatch);

  m.def(
    "error_code_to_string",
    [](kinematic_smoother::ErrorCode code) {
      return kinematic_smoother::toErrorCodeString(code);
    },
    py::arg("code"));

  m.attr("ERROR_INVALID_PATH") = py::str(
    kinematic_smoother::toErrorCodeString(kinematic_smoother::ErrorCode::InvalidPath));
  m.attr("ERROR_FAILED_TO_SMOOTH_PATH") = py::str(
    kinematic_smoother::toErrorCodeString(kinematic_smoother::ErrorCode::FailedToSmoothPath));
  m.attr("ERROR_INVALID_COSTMAP") = py::str(
    kinematic_smoother::toErrorCodeString(kinematic_smoother::ErrorCode::InvalidCostmap));
  m.attr("ERROR_PRECOMPUTED_ESDF_SIZE_MISMATCH") = py::str(
    kinematic_smoother::toErrorCodeString(
      kinematic_smoother::ErrorCode::PrecomputedEsdfSizeMismatch));

  // ---- Core value types and planning utilities ----

  py::class_<kinematic_smoother::Costmap2D>(m, "Costmap2D")
    .def(py::init<>())
    .def(
    py::init<unsigned int, unsigned int, double, double, double>(),
    py::arg("size_x"), py::arg("size_y"), py::arg("resolution"),
    py::arg("origin_x"), py::arg("origin_y"))
    .def("getSizeInCellsX", &kinematic_smoother::Costmap2D::getSizeInCellsX)
    .def("getSizeInCellsY", &kinematic_smoother::Costmap2D::getSizeInCellsY)
    .def("getResolution", &kinematic_smoother::Costmap2D::getResolution)
    .def("getOriginX", &kinematic_smoother::Costmap2D::getOriginX)
    .def("getOriginY", &kinematic_smoother::Costmap2D::getOriginY)
    .def("getCost", &kinematic_smoother::Costmap2D::getCost)
    .def("setCost", &kinematic_smoother::Costmap2D::setCost)
    .def_readonly_static("NO_INFORMATION", &kinematic_smoother::Costmap2D::NO_INFORMATION)
    .def_readonly_static("LETHAL_OBSTACLE", &kinematic_smoother::Costmap2D::LETHAL_OBSTACLE)
    .def_readonly_static(
    "INSCRIBED_INFLATED_OBSTACLE",
    &kinematic_smoother::Costmap2D::INSCRIBED_INFLATED_OBSTACLE)
    .def_readonly_static("FREE_SPACE", &kinematic_smoother::Costmap2D::FREE_SPACE);

  // --- SmootherParams ---
  py::class_<kinematic_smoother::SmootherParams>(m, "SmootherParams")
    .def(py::init<>())
    .def_readwrite("smooth_weight_sqrt", &kinematic_smoother::SmootherParams::smooth_weight_sqrt)
    .def_readwrite("model_weight_sqrt", &kinematic_smoother::SmootherParams::model_weight_sqrt)
    .def_readwrite(
    "costmap_weight_sqrt",
    &kinematic_smoother::SmootherParams::costmap_weight_sqrt)
    .def_readwrite(
    "cusp_costmap_weight_sqrt",
    &kinematic_smoother::SmootherParams::cusp_costmap_weight_sqrt)
    .def_readwrite("cusp_zone_length", &kinematic_smoother::SmootherParams::cusp_zone_length)
    .def_readwrite(
    "distance_weight_sqrt",
    &kinematic_smoother::SmootherParams::distance_weight_sqrt)
    .def_readwrite(
    "reference_point_max_deviation",
    &kinematic_smoother::SmootherParams::reference_point_max_deviation)
    .def_readwrite(
    "curvature_weight_sqrt",
    &kinematic_smoother::SmootherParams::curvature_weight_sqrt)
    .def_readwrite(
    "curvature_rate_weight_sqrt",
    &kinematic_smoother::SmootherParams::curvature_rate_weight_sqrt)
    .def_readwrite(
    "kinematic_curvature_weight_sqrt",
    &kinematic_smoother::SmootherParams::kinematic_curvature_weight_sqrt)
    .def_readwrite(
    "kinematic_curvature_rate_weight_sqrt",
    &kinematic_smoother::SmootherParams::kinematic_curvature_rate_weight_sqrt)
    .def_readwrite("max_curvature", &kinematic_smoother::SmootherParams::max_curvature)
    .def_readwrite("max_time", &kinematic_smoother::SmootherParams::max_time)
    .def_readwrite("use_exact_esdf", &kinematic_smoother::SmootherParams::use_exact_esdf)
    .def_readwrite(
    "obstacle_safe_distance",
    &kinematic_smoother::SmootherParams::obstacle_safe_distance)
    .def_readwrite(
    "cost_check_radius",
    &kinematic_smoother::SmootherParams::cost_check_radius)
    .def_readwrite(
    "path_downsampling_factor",
    &kinematic_smoother::SmootherParams::path_downsampling_factor)
    .def_readwrite(
    "path_upsampling_factor",
    &kinematic_smoother::SmootherParams::path_upsampling_factor)
    .def_readwrite(
    "goal_longitudinal_tolerance",
    &kinematic_smoother::SmootherParams::goal_longitudinal_tolerance)
    .def_readwrite(
    "goal_lateral_tolerance",
    &kinematic_smoother::SmootherParams::goal_lateral_tolerance)
    .def_readwrite(
    "goal_orientation_tolerance",
    &kinematic_smoother::SmootherParams::goal_orientation_tolerance)
    .def_readwrite("reversing_enabled", &kinematic_smoother::SmootherParams::reversing_enabled)
    .def_readwrite(
    "keep_goal_orientation",
    &kinematic_smoother::SmootherParams::keep_goal_orientation)
    .def_readwrite(
    "keep_start_orientation",
    &kinematic_smoother::SmootherParams::keep_start_orientation)
    .def_readwrite(
    "cost_check_points",
    &kinematic_smoother::SmootherParams::cost_check_points);

  // --- OptimizerParams ---
  py::class_<kinematic_smoother::OptimizerParams>(m, "OptimizerParams")
    .def(py::init<>())
    .def_readwrite("debug", &kinematic_smoother::OptimizerParams::debug)
    .def_readwrite(
    "linear_solver_type",
    &kinematic_smoother::OptimizerParams::linear_solver_type)
    .def_readwrite("max_iterations", &kinematic_smoother::OptimizerParams::max_iterations)
    .def_readwrite("param_tol", &kinematic_smoother::OptimizerParams::param_tol)
    .def_readwrite("fn_tol", &kinematic_smoother::OptimizerParams::fn_tol)
    .def_readwrite("gradient_tol", &kinematic_smoother::OptimizerParams::gradient_tol);

  py::class_<kinematic_smoother::AStarPlannerParams>(m, "AStarPlannerParams")
    .def(py::init<>())
    .def_readwrite("lethal_cost", &kinematic_smoother::AStarPlannerParams::lethal_cost)
    .def_readwrite("use_exact_esdf", &kinematic_smoother::AStarPlannerParams::use_exact_esdf)
    .def_readwrite("safe_distance", &kinematic_smoother::AStarPlannerParams::safe_distance)
    .def_readwrite("cost_penalty_weight", &kinematic_smoother::AStarPlannerParams::cost_penalty_weight)
    .def_readwrite("point_radius", &kinematic_smoother::AStarPlannerParams::point_radius)
    .def_readwrite(
    "collision_check_radius",
    &kinematic_smoother::AStarPlannerParams::collision_check_radius)
    .def_readwrite(
    "collision_check_points",
    &kinematic_smoother::AStarPlannerParams::collision_check_points)
    .def_readwrite(
    "use_rectangular_footprint",
    &kinematic_smoother::AStarPlannerParams::use_rectangular_footprint)
    .def_readwrite("rectangular_length", &kinematic_smoother::AStarPlannerParams::rectangular_length)
    .def_readwrite("rectangular_width", &kinematic_smoother::AStarPlannerParams::rectangular_width);

  py::class_<kinematic_smoother::AStarPlanner>(m, "AStarPlanner")
    .def(py::init<>())
    .def(
      "plan",
      [](kinematic_smoother::AStarPlanner & self,
      const kinematic_smoother::Costmap2D & costmap,
      double start_x, double start_y,
      double goal_x, double goal_y,
      const kinematic_smoother::AStarPlannerParams & params)
      {
        return self.plan(&costmap, start_x, start_y, goal_x, goal_y, params);
      },
      py::arg("costmap"), py::arg("start_x"), py::arg("start_y"),
      py::arg("goal_x"), py::arg("goal_y"), py::arg("params"))
    .def("get_esdf", &kinematic_smoother::AStarPlanner::getESDF);

  m.def(
    "compute_esdf",
    [](const kinematic_smoother::Costmap2D & costmap, unsigned char lethal_cost, bool use_exact)
    {
      return kinematic_smoother::ESDF::ComputeESDF(
        &costmap,
        lethal_cost,
        use_exact ? kinematic_smoother::ESDFAlgorithm::Exact :
        kinematic_smoother::ESDFAlgorithm::Approximate);
    },
    py::arg("costmap"),
    py::arg("lethal_cost") = kinematic_smoother::Costmap2D::LETHAL_OBSTACLE,
    py::arg("use_exact") = true);

  // ---- Geometric smoother bindings ----

  // ---- Kinematic smoother bindings ----
  // These bindings intentionally mirror the geometric smoother API above:
  // same exception-style entrypoints, same try_* methods, and the same
  // ok/error_* structured result schema. That keeps Python and web callers
  // backend-agnostic even though the underlying optimizer model differs.

  py::class_<kinematic_smoother::KinematicSmoother>(m, "KinematicSmoother")
    .def(py::init<>())
    .def("initialize", &kinematic_smoother::KinematicSmoother::initialize)
    .def(
      "get_last_optimized_knot_count",
      &kinematic_smoother::KinematicSmoother::getLastOptimizedKnotCount)
    .def(
      "smooth",
      [](kinematic_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const py::handle & costmap_handle,
      const kinematic_smoother::SmootherParams & params) -> PyObject *
      {
        std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
        const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
        const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
        const auto * costmap = copy_optional_costmap(costmap_handle);
        kinematic_smoother::SmoothingFailureInfo failure;
        if (!self.smooth(path, start_dir, end_dir, costmap, params, nullptr, &failure)) {
          return make_python_smoothing_failure(failure);
        }
        return py::cast(path).release().ptr();
      },
      py::return_value_policy::take_ownership,
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap") = py::none(), py::arg("params"),
      // 异常式接口：失败时抛 Python 异常，成功时返回运动学平滑后的路径。
      "Smooth a path using the kinematic backend. Input path z must encode direction sign (+1/-1); returned path z is yaw in radians.")
    .def(
      "try_smooth",
      [](kinematic_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const py::handle & costmap_handle,
      const kinematic_smoother::SmootherParams & params) -> py::dict
      {
        try {
          std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
          const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
          const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
          const auto * costmap = copy_optional_costmap(costmap_handle);
          kinematic_smoother::SmoothingFailureInfo failure;
          if (!self.smooth(path, start_dir, end_dir, costmap, params, nullptr, &failure)) {
            return make_error_result(failure, py::cast(path));
          }

          py::dict result;
          result["ok"] = true;
          result["path"] = path;
          result["error_code"] = py::none();
          result["error_message"] = py::none();
          result["error_reason"] = py::none();
          result["error_details"] = py::none();
          return result;
        } catch (const kinematic_smoother::InvalidPath & error) {
          return make_error_result(error);
        } catch (const kinematic_smoother::InvalidCostmap & error) {
          return make_error_result(error);
        } catch (const kinematic_smoother::PrecomputedEsdfSizeMismatch & error) {
          return make_error_result(error);
        } catch (const py::error_already_set &) {
          throw;
        } catch (const std::exception & error) {
          py::dict result;
          result["ok"] = false;
          result["path"] = py::none();
          result["error_code"] = py::none();
          result["error_message"] = py::str(error.what());
          result["error_reason"] = py::none();
          result["error_details"] = py::none();
          return result;
        }
      },
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap") = py::none(), py::arg("params"),
      // 结构化接口：把运动学后端失败统一折叠成 ok/error_* 字段。
      "Try to smooth a path with the kinematic backend and return a structured result.")
    .def(
      "smooth_with_planner_esdf",
      [](kinematic_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const kinematic_smoother::Costmap2D & costmap,
      const kinematic_smoother::SmootherParams & params,
      const kinematic_smoother::AStarPlanner & planner) -> PyObject *
      {
        std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
        const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
        const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
        kinematic_smoother::SmoothingFailureInfo failure;
        if (!self.smooth(path, start_dir, end_dir, &costmap, params, &planner.getESDF(), &failure)) {
          return make_python_smoothing_failure(failure);
        }
        return py::cast(path).release().ptr();
      },
      py::return_value_policy::take_ownership,
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"), py::arg("planner"),
      // 异常式接口：复用 planner 已算好的 ESDF，失败时抛 Python 异常。
      "Smooth a path with the kinematic backend while reusing the ESDF previously computed by an A* planner.")
    .def(
      "try_smooth_with_planner_esdf",
      [](kinematic_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const kinematic_smoother::Costmap2D & costmap,
      const kinematic_smoother::SmootherParams & params,
      const kinematic_smoother::AStarPlanner & planner) -> py::dict
      {
        try {
          std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
          const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
          const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
          kinematic_smoother::SmoothingFailureInfo failure;
          if (!self.smooth(path, start_dir, end_dir, &costmap, params, &planner.getESDF(), &failure)) {
            return make_error_result(failure, py::cast(path));
          }

          py::dict result;
          result["ok"] = true;
          result["path"] = path;
          result["error_code"] = py::none();
          result["error_message"] = py::none();
          result["error_reason"] = py::none();
          result["error_details"] = py::none();
          return result;
        } catch (const kinematic_smoother::InvalidPath & error) {
          return make_error_result(error);
        } catch (const kinematic_smoother::InvalidCostmap & error) {
          return make_error_result(error);
        } catch (const kinematic_smoother::PrecomputedEsdfSizeMismatch & error) {
          return make_error_result(error);
        } catch (const py::error_already_set &) {
          throw;
        } catch (const std::exception & error) {
          py::dict result;
          result["ok"] = false;
          result["path"] = py::none();
          result["error_code"] = py::none();
          result["error_message"] = py::str(error.what());
          result["error_reason"] = py::none();
          result["error_details"] = py::none();
          return result;
        }
      },
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"), py::arg("planner"),
      // 结构化接口：复用 planner ESDF，同时保持稳定的 ok/error_* 返回面。
      "Try to smooth a path with the kinematic backend and planner ESDF, returning a structured result.");

  // ---- Native exception translation ----

  py::register_exception<kinematic_smoother::InvalidPath>(m, "InvalidPathError");
  py::register_exception<kinematic_smoother::FailedToSmoothPath>(m, "FailedToSmoothPathError");
  py::register_exception<kinematic_smoother::InvalidCostmap>(m, "InvalidCostmapError");
  py::register_exception<kinematic_smoother::PrecomputedEsdfSizeMismatch>(
    m,
    "PrecomputedEsdfSizeMismatchError");
}
