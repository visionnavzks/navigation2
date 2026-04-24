// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "constrained_smoother/astar_esdf.hpp"
#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/kinematic_smoother.hpp"
#include "constrained_smoother/options.hpp"
#include "constrained_smoother/smoother.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/esdf.hpp"

#include <vector>
#include <cmath>
#include <typeinfo>

namespace py = pybind11;

namespace
{

struct ParsedSmoothingFailure
{
  py::object reason{py::none()};
  py::object details{py::none()};
  std::string message;
};

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
    reason == "footprint_collision";
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

py::dict make_error_result(const constrained_smoother::FailedToSmoothPath & error)
{
  py::dict result = make_error_result_base(error);
  const ParsedSmoothingFailure parsed = parse_smoothing_failure_message(error.what());
  result["error_message"] = py::str(parsed.message);
  result["error_reason"] = parsed.reason;
  result["error_details"] = parsed.details;
  return result;
}

py::dict make_error_result(const constrained_smoother::SmoothingFailureInfo & failure)
{
  py::dict result;
  result["ok"] = false;
  result["path"] = py::none();
  result["error_code"] = py::str(
    constrained_smoother::toErrorCodeString(constrained_smoother::ErrorCode::FailedToSmoothPath));
  result["error_message"] = py::str(failure.message);
  result["error_reason"] = py::str(
    constrained_smoother::toSmoothingFailureReasonString(failure.reason));
  if (failure.failed_index >= 0) {
    py::dict details;
    details["failed_index"] = py::int_(failure.failed_index);
    result["error_details"] = details;
  } else {
    result["error_details"] = py::none();
  }
  return result;
}

PyObject * make_python_smoothing_failure(const constrained_smoother::SmoothingFailureInfo & failure)
{
  PyErr_SetString(
    PyExc_RuntimeError,
    (std::string(constrained_smoother::toErrorCodeString(
       constrained_smoother::ErrorCode::FailedToSmoothPath)) +
    ": " + failure.formattedMessage()).c_str());
  return nullptr;
}

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
  } catch (const constrained_smoother::InvalidPath & error) {
    return make_error_result(error);
  } catch (const constrained_smoother::FailedToSmoothPath & error) {
    return make_error_result(error);
  } catch (const constrained_smoother::InvalidCostmap & error) {
    return make_error_result(error);
  } catch (const constrained_smoother::PrecomputedEsdfSizeMismatch & error) {
    return make_error_result(error);
  } catch (const std::exception & error) {
    if (const auto * invalid_path = dynamic_cast<const constrained_smoother::InvalidPath *>(&error)) {
      return make_error_result(*invalid_path);
    }
    if (const auto * failed = dynamic_cast<const constrained_smoother::FailedToSmoothPath *>(&error)) {
      return make_error_result(*failed);
    }
    if (const auto * invalid_costmap = dynamic_cast<const constrained_smoother::InvalidCostmap *>(&error)) {
      return make_error_result(*invalid_costmap);
    }
    if (
      const auto * size_mismatch =
      dynamic_cast<const constrained_smoother::PrecomputedEsdfSizeMismatch *>(&error))
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

PYBIND11_MODULE(py_constrained_smoother, m)
{
  m.doc() = "Python bindings for the constrained_smoother C++ library";

  py::enum_<constrained_smoother::ErrorCode>(m, "ErrorCode")
    .value("INVALID_PATH", constrained_smoother::ErrorCode::InvalidPath)
    .value("FAILED_TO_SMOOTH_PATH", constrained_smoother::ErrorCode::FailedToSmoothPath)
    .value("INVALID_COSTMAP", constrained_smoother::ErrorCode::InvalidCostmap)
    .value(
      "PRECOMPUTED_ESDF_SIZE_MISMATCH",
      constrained_smoother::ErrorCode::PrecomputedEsdfSizeMismatch);

  m.def(
    "error_code_to_string",
    [](constrained_smoother::ErrorCode code) {
      return constrained_smoother::toErrorCodeString(code);
    },
    py::arg("code"));

  m.attr("ERROR_INVALID_PATH") = py::str(
    constrained_smoother::toErrorCodeString(constrained_smoother::ErrorCode::InvalidPath));
  m.attr("ERROR_FAILED_TO_SMOOTH_PATH") = py::str(
    constrained_smoother::toErrorCodeString(constrained_smoother::ErrorCode::FailedToSmoothPath));
  m.attr("ERROR_INVALID_COSTMAP") = py::str(
    constrained_smoother::toErrorCodeString(constrained_smoother::ErrorCode::InvalidCostmap));
  m.attr("ERROR_PRECOMPUTED_ESDF_SIZE_MISMATCH") = py::str(
    constrained_smoother::toErrorCodeString(
      constrained_smoother::ErrorCode::PrecomputedEsdfSizeMismatch));

  // --- Costmap2D ---
  py::class_<constrained_smoother::Costmap2D>(m, "Costmap2D")
    .def(py::init<>())
    .def(
    py::init<unsigned int, unsigned int, double, double, double>(),
    py::arg("size_x"), py::arg("size_y"), py::arg("resolution"),
    py::arg("origin_x"), py::arg("origin_y"))
    .def("getSizeInCellsX", &constrained_smoother::Costmap2D::getSizeInCellsX)
    .def("getSizeInCellsY", &constrained_smoother::Costmap2D::getSizeInCellsY)
    .def("getResolution", &constrained_smoother::Costmap2D::getResolution)
    .def("getOriginX", &constrained_smoother::Costmap2D::getOriginX)
    .def("getOriginY", &constrained_smoother::Costmap2D::getOriginY)
    .def("getCost", &constrained_smoother::Costmap2D::getCost)
    .def("setCost", &constrained_smoother::Costmap2D::setCost)
    .def_readonly_static("NO_INFORMATION", &constrained_smoother::Costmap2D::NO_INFORMATION)
    .def_readonly_static("LETHAL_OBSTACLE", &constrained_smoother::Costmap2D::LETHAL_OBSTACLE)
    .def_readonly_static(
    "INSCRIBED_INFLATED_OBSTACLE",
    &constrained_smoother::Costmap2D::INSCRIBED_INFLATED_OBSTACLE)
    .def_readonly_static("FREE_SPACE", &constrained_smoother::Costmap2D::FREE_SPACE);

  // --- SmootherParams ---
  py::class_<constrained_smoother::SmootherParams>(m, "SmootherParams")
    .def(py::init<>())
    .def_readwrite("smooth_weight_sqrt", &constrained_smoother::SmootherParams::smooth_weight_sqrt)
    .def_readwrite(
    "costmap_weight_sqrt",
    &constrained_smoother::SmootherParams::costmap_weight_sqrt)
    .def_readwrite(
    "cusp_costmap_weight_sqrt",
    &constrained_smoother::SmootherParams::cusp_costmap_weight_sqrt)
    .def_readwrite("cusp_zone_length", &constrained_smoother::SmootherParams::cusp_zone_length)
    .def_readwrite(
    "distance_weight_sqrt",
    &constrained_smoother::SmootherParams::distance_weight_sqrt)
    .def_readwrite(
    "curvature_weight_sqrt",
    &constrained_smoother::SmootherParams::curvature_weight_sqrt)
    .def_readwrite(
    "curvature_rate_weight_sqrt",
    &constrained_smoother::SmootherParams::curvature_rate_weight_sqrt)
    .def_readwrite("max_curvature", &constrained_smoother::SmootherParams::max_curvature)
    .def_readwrite("max_time", &constrained_smoother::SmootherParams::max_time)
    .def_readwrite("use_exact_esdf", &constrained_smoother::SmootherParams::use_exact_esdf)
    .def_readwrite(
    "obstacle_safe_distance",
    &constrained_smoother::SmootherParams::obstacle_safe_distance)
    .def_readwrite(
    "cost_check_radius",
    &constrained_smoother::SmootherParams::cost_check_radius)
    .def_readwrite(
    "path_downsampling_factor",
    &constrained_smoother::SmootherParams::path_downsampling_factor)
    .def_readwrite(
    "path_upsampling_factor",
    &constrained_smoother::SmootherParams::path_upsampling_factor)
    .def_readwrite("reversing_enabled", &constrained_smoother::SmootherParams::reversing_enabled)
    .def_readwrite(
    "keep_goal_orientation",
    &constrained_smoother::SmootherParams::keep_goal_orientation)
    .def_readwrite(
    "keep_start_orientation",
    &constrained_smoother::SmootherParams::keep_start_orientation)
    .def_readwrite(
    "cost_check_points",
    &constrained_smoother::SmootherParams::cost_check_points);

  // --- OptimizerParams ---
  py::class_<constrained_smoother::OptimizerParams>(m, "OptimizerParams")
    .def(py::init<>())
    .def_readwrite("debug", &constrained_smoother::OptimizerParams::debug)
    .def_readwrite(
    "linear_solver_type",
    &constrained_smoother::OptimizerParams::linear_solver_type)
    .def_readwrite("max_iterations", &constrained_smoother::OptimizerParams::max_iterations)
    .def_readwrite("param_tol", &constrained_smoother::OptimizerParams::param_tol)
    .def_readwrite("fn_tol", &constrained_smoother::OptimizerParams::fn_tol)
    .def_readwrite("gradient_tol", &constrained_smoother::OptimizerParams::gradient_tol);

  py::class_<constrained_smoother::AStarPlannerParams>(m, "AStarPlannerParams")
    .def(py::init<>())
    .def_readwrite("lethal_cost", &constrained_smoother::AStarPlannerParams::lethal_cost)
    .def_readwrite("use_exact_esdf", &constrained_smoother::AStarPlannerParams::use_exact_esdf)
    .def_readwrite("safe_distance", &constrained_smoother::AStarPlannerParams::safe_distance)
    .def_readwrite("cost_penalty_weight", &constrained_smoother::AStarPlannerParams::cost_penalty_weight)
    .def_readwrite("point_radius", &constrained_smoother::AStarPlannerParams::point_radius)
    .def_readwrite(
    "collision_check_radius",
    &constrained_smoother::AStarPlannerParams::collision_check_radius)
    .def_readwrite(
    "collision_check_points",
    &constrained_smoother::AStarPlannerParams::collision_check_points)
    .def_readwrite(
    "use_rectangular_footprint",
    &constrained_smoother::AStarPlannerParams::use_rectangular_footprint)
    .def_readwrite("rectangular_length", &constrained_smoother::AStarPlannerParams::rectangular_length)
    .def_readwrite("rectangular_width", &constrained_smoother::AStarPlannerParams::rectangular_width);

  py::class_<constrained_smoother::AStarPlanner>(m, "AStarPlanner")
    .def(py::init<>())
    .def(
      "plan",
      [](constrained_smoother::AStarPlanner & self,
      const constrained_smoother::Costmap2D & costmap,
      double start_x, double start_y,
      double goal_x, double goal_y,
      const constrained_smoother::AStarPlannerParams & params)
      {
        return self.plan(&costmap, start_x, start_y, goal_x, goal_y, params);
      },
      py::arg("costmap"), py::arg("start_x"), py::arg("start_y"),
      py::arg("goal_x"), py::arg("goal_y"), py::arg("params"))
    .def("get_esdf", &constrained_smoother::AStarPlanner::getESDF);

  m.def(
    "compute_esdf",
    [](const constrained_smoother::Costmap2D & costmap, unsigned char lethal_cost, bool use_exact)
    {
      return constrained_smoother::ESDF::ComputeESDF(
        &costmap,
        lethal_cost,
        use_exact ? constrained_smoother::ESDFAlgorithm::Exact :
        constrained_smoother::ESDFAlgorithm::Approximate);
    },
    py::arg("costmap"),
    py::arg("lethal_cost") = constrained_smoother::Costmap2D::LETHAL_OBSTACLE,
    py::arg("use_exact") = true);

  // --- Smoother ---
  py::class_<constrained_smoother::Smoother>(m, "Smoother")
    .def(py::init<>())
    .def("initialize", &constrained_smoother::Smoother::initialize)
    .def("get_last_optimized_knot_count", &constrained_smoother::Smoother::getLastOptimizedKnotCount)
    .def(
    "smooth",
    [](constrained_smoother::Smoother & self,
    const py::handle & path_handle,
    const py::handle & start_dir_handle,
    const py::handle & end_dir_handle,
    const constrained_smoother::Costmap2D & costmap,
    const constrained_smoother::SmootherParams & params) -> PyObject *
    {
      std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
      const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
      const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
      constrained_smoother::SmoothingFailureInfo failure;
      if (!self.smooth(path, start_dir, end_dir, &costmap, params, nullptr, &failure)) {
        return make_python_smoothing_failure(failure);
      }
      return py::cast(path).release().ptr();
    },
    py::return_value_policy::take_ownership,
    py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
    py::arg("costmap"), py::arg("params"),
    "Smooth a path. Input path z must encode direction sign (+1/-1); returned path z is yaw in radians.")
    .def(
    "try_smooth",
    [](constrained_smoother::Smoother & self,
    const py::handle & path_handle,
    const py::handle & start_dir_handle,
    const py::handle & end_dir_handle,
    const constrained_smoother::Costmap2D & costmap,
    const constrained_smoother::SmootherParams & params) -> py::dict
    {
      try {
        std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
        const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
        const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
        constrained_smoother::SmoothingFailureInfo failure;
        if (!self.smooth(path, start_dir, end_dir, &costmap, params, nullptr, &failure)) {
          return make_error_result(failure);
        }

        py::dict result;
        result["ok"] = true;
        result["path"] = path;
        result["error_code"] = py::none();
        result["error_message"] = py::none();
        result["error_reason"] = py::none();
        result["error_details"] = py::none();
        return result;
      } catch (const constrained_smoother::InvalidPath & error) {
        return make_error_result(error);
      } catch (const constrained_smoother::InvalidCostmap & error) {
        return make_error_result(error);
      } catch (const constrained_smoother::PrecomputedEsdfSizeMismatch & error) {
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
    py::arg("costmap"), py::arg("params"),
    "Try to smooth a path and return a structured result with ok/path/error_code/error_message.")
    .def(
    "smooth_with_planner_esdf",
    [](constrained_smoother::Smoother & self,
    const py::handle & path_handle,
    const py::handle & start_dir_handle,
    const py::handle & end_dir_handle,
    const constrained_smoother::Costmap2D & costmap,
    const constrained_smoother::SmootherParams & params,
    const constrained_smoother::AStarPlanner & planner) -> PyObject *
    {
      std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
      const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
      const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
      constrained_smoother::SmoothingFailureInfo failure;
      if (!self.smooth(path, start_dir, end_dir, &costmap, params, &planner.getESDF(), &failure)) {
        return make_python_smoothing_failure(failure);
      }
      return py::cast(path).release().ptr();
    },
    py::return_value_policy::take_ownership,
    py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
    py::arg("costmap"), py::arg("params"), py::arg("planner"),
    "Smooth a path while reusing the ESDF previously computed by an A* planner.")
    .def(
    "try_smooth_with_planner_esdf",
    [](constrained_smoother::Smoother & self,
    const py::handle & path_handle,
    const py::handle & start_dir_handle,
    const py::handle & end_dir_handle,
    const constrained_smoother::Costmap2D & costmap,
    const constrained_smoother::SmootherParams & params,
    const constrained_smoother::AStarPlanner & planner) -> py::dict
    {
      try {
        std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
        const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
        const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
        constrained_smoother::SmoothingFailureInfo failure;
        if (!self.smooth(path, start_dir, end_dir, &costmap, params, &planner.getESDF(), &failure)) {
          return make_error_result(failure);
        }

        py::dict result;
        result["ok"] = true;
        result["path"] = path;
        result["error_code"] = py::none();
        result["error_message"] = py::none();
        result["error_reason"] = py::none();
        result["error_details"] = py::none();
        return result;
      } catch (const constrained_smoother::InvalidPath & error) {
        return make_error_result(error);
      } catch (const constrained_smoother::InvalidCostmap & error) {
        return make_error_result(error);
      } catch (const constrained_smoother::PrecomputedEsdfSizeMismatch & error) {
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
    "Try to smooth a path with a planner ESDF and return a structured result.");

  py::class_<constrained_smoother::KinematicSmoother>(m, "KinematicSmoother")
    .def(py::init<>())
    .def("initialize", &constrained_smoother::KinematicSmoother::initialize)
    .def(
      "get_last_optimized_knot_count",
      &constrained_smoother::KinematicSmoother::getLastOptimizedKnotCount)
    .def(
      "smooth",
      [](constrained_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const constrained_smoother::Costmap2D & costmap,
      const constrained_smoother::SmootherParams & params) -> PyObject *
      {
        std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
        const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
        const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
        constrained_smoother::SmoothingFailureInfo failure;
        if (!self.smooth(path, start_dir, end_dir, &costmap, params, nullptr, &failure)) {
          return make_python_smoothing_failure(failure);
        }
        return py::cast(path).release().ptr();
      },
      py::return_value_policy::take_ownership,
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"),
      "Smooth a path using the kinematic backend. Input path z must encode direction sign (+1/-1); returned path z is yaw in radians.")
    .def(
      "try_smooth",
      [](constrained_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const constrained_smoother::Costmap2D & costmap,
      const constrained_smoother::SmootherParams & params) -> py::dict
      {
        try {
          std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
          const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
          const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
          constrained_smoother::SmoothingFailureInfo failure;
          if (!self.smooth(path, start_dir, end_dir, &costmap, params, nullptr, &failure)) {
            return make_error_result(failure);
          }

          py::dict result;
          result["ok"] = true;
          result["path"] = path;
          result["error_code"] = py::none();
          result["error_message"] = py::none();
          result["error_reason"] = py::none();
          result["error_details"] = py::none();
          return result;
        } catch (const constrained_smoother::InvalidPath & error) {
          return make_error_result(error);
        } catch (const constrained_smoother::InvalidCostmap & error) {
          return make_error_result(error);
        } catch (const constrained_smoother::PrecomputedEsdfSizeMismatch & error) {
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
      py::arg("costmap"), py::arg("params"),
      "Try to smooth a path with the kinematic backend and return a structured result.")
    .def(
      "smooth_with_planner_esdf",
      [](constrained_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const constrained_smoother::Costmap2D & costmap,
      const constrained_smoother::SmootherParams & params,
      const constrained_smoother::AStarPlanner & planner) -> PyObject *
      {
        std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
        const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
        const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
        constrained_smoother::SmoothingFailureInfo failure;
        if (!self.smooth(path, start_dir, end_dir, &costmap, params, &planner.getESDF(), &failure)) {
          return make_python_smoothing_failure(failure);
        }
        return py::cast(path).release().ptr();
      },
      py::return_value_policy::take_ownership,
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"), py::arg("planner"),
      "Smooth a path with the kinematic backend while reusing the ESDF previously computed by an A* planner.")
    .def(
      "try_smooth_with_planner_esdf",
      [](constrained_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const constrained_smoother::Costmap2D & costmap,
      const constrained_smoother::SmootherParams & params,
      const constrained_smoother::AStarPlanner & planner) -> py::dict
      {
        try {
          std::vector<Eigen::Vector3d> path = copy_path3d(path_handle, "path");
          const Eigen::Vector2d start_dir = copy_vector2d(start_dir_handle, "start_dir");
          const Eigen::Vector2d end_dir = copy_vector2d(end_dir_handle, "end_dir");
          constrained_smoother::SmoothingFailureInfo failure;
          if (!self.smooth(path, start_dir, end_dir, &costmap, params, &planner.getESDF(), &failure)) {
            return make_error_result(failure);
          }

          py::dict result;
          result["ok"] = true;
          result["path"] = path;
          result["error_code"] = py::none();
          result["error_message"] = py::none();
          result["error_reason"] = py::none();
          result["error_details"] = py::none();
          return result;
        } catch (const constrained_smoother::InvalidPath & error) {
          return make_error_result(error);
        } catch (const constrained_smoother::InvalidCostmap & error) {
          return make_error_result(error);
        } catch (const constrained_smoother::PrecomputedEsdfSizeMismatch & error) {
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
      "Try to smooth a path with the kinematic backend and planner ESDF, returning a structured result.");

  // --- Exceptions ---
  py::register_exception<constrained_smoother::InvalidPath>(m, "InvalidPathError");
  py::register_exception<constrained_smoother::FailedToSmoothPath>(m, "FailedToSmoothPathError");
  py::register_exception<constrained_smoother::InvalidCostmap>(m, "InvalidCostmapError");
  py::register_exception<constrained_smoother::PrecomputedEsdfSizeMismatch>(
    m,
    "PrecomputedEsdfSizeMismatchError");
}
