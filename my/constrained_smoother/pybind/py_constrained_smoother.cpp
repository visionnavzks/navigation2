// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "constrained_smoother/astar_esdf.hpp"
#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/kinematic_smoother.hpp"
#include "constrained_smoother/options.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/esdf.hpp"

#include <vector>
#include <cmath>

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

const constrained_smoother::Costmap2D * copy_optional_costmap(const py::handle & handle)
{
  if (handle.is_none()) {
    return nullptr;
  }

  return &py::cast<const constrained_smoother::Costmap2D &>(handle);
}

bool run_smooth_request(
  constrained_smoother::KinematicSmoother & smoother,
  std::vector<Eigen::Vector3d> & path,
  const Eigen::Vector2d & start_dir,
  const Eigen::Vector2d & end_dir,
  const constrained_smoother::Costmap2D * costmap,
  const constrained_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf,
  constrained_smoother::SmoothingFailureInfo * failure)
{
  const constrained_smoother::SmootherRequest request{
    path,
    start_dir,
    end_dir,
    costmap,
    params,
    precomputed_esdf,
    failure,
  };
  return smoother.smooth(request);
}

struct SmoothBindingInput
{
  std::vector<Eigen::Vector3d> path;
  Eigen::Vector2d start_dir;
  Eigen::Vector2d end_dir;
  const constrained_smoother::Costmap2D * costmap;
};

PyObject * make_python_smoothing_failure(const constrained_smoother::SmoothingFailureInfo & failure);

template<typename ErrorT>
py::dict make_error_result(const ErrorT & error);

py::dict make_error_result(const constrained_smoother::FailedToSmoothPath & error);
py::dict make_error_result(
  const constrained_smoother::SmoothingFailureInfo & failure,
  const py::object & path = py::none());

py::dict make_ok_result(const std::vector<Eigen::Vector3d> & path);

template<typename Fn>
py::dict invoke_try_smooth(Fn && fn);

SmoothBindingInput parse_smooth_input(
  const py::handle & path_handle,
  const py::handle & start_dir_handle,
  const py::handle & end_dir_handle,
  const py::handle & costmap_handle)
{
  return SmoothBindingInput{
    copy_path3d(path_handle, "path"),
    copy_vector2d(start_dir_handle, "start_dir"),
    copy_vector2d(end_dir_handle, "end_dir"),
    copy_optional_costmap(costmap_handle),
  };
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

py::dict make_error_result(const constrained_smoother::FailedToSmoothPath & error)
{
  py::dict result = make_error_result_base(error);
  const ParsedSmoothingFailure parsed = parse_smoothing_failure_message(error.what());
  result["error_message"] = py::str(parsed.message);
  result["error_reason"] = parsed.reason;
  result["error_details"] = parsed.details;
  return result;
}

py::dict make_error_result(
  const constrained_smoother::SmoothingFailureInfo & failure,
  const py::object & path)
{
  py::dict result;
  result["ok"] = false;
  result["path"] = path;
  result["error_code"] = py::str(
    constrained_smoother::toErrorCodeString(constrained_smoother::ErrorCode::FailedToSmoothPath));
  result["error_message"] = py::str(failure.message);
  result["error_reason"] = py::str(
    constrained_smoother::toSmoothingFailureReasonString(failure.reason));
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

PyObject * make_python_smoothing_failure(const constrained_smoother::SmoothingFailureInfo & failure)
{
  PyErr_SetString(
    PyExc_RuntimeError,
    (std::string(constrained_smoother::toErrorCodeString(
       constrained_smoother::ErrorCode::FailedToSmoothPath)) +
    ": " + failure.formattedMessage()).c_str());
  return nullptr;
}

py::dict make_ok_result(const std::vector<Eigen::Vector3d> & path)
{
  py::dict result;
  result["ok"] = true;
  result["path"] = path;
  result["error_code"] = py::none();
  result["error_message"] = py::none();
  result["error_reason"] = py::none();
  result["error_details"] = py::none();
  return result;
}

PyObject * run_smooth_or_raise(
  constrained_smoother::KinematicSmoother & smoother,
  SmoothBindingInput && input,
  const constrained_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf)
{
  constrained_smoother::SmoothingFailureInfo failure;
  if (!run_smooth_request(
      smoother,
      input.path,
      input.start_dir,
      input.end_dir,
      input.costmap,
      params,
      precomputed_esdf,
      &failure))
  {
    return make_python_smoothing_failure(failure);
  }

  return py::cast(input.path).release().ptr();
}

py::dict run_try_smooth_result(
  constrained_smoother::KinematicSmoother & smoother,
  SmoothBindingInput && input,
  const constrained_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf)
{
  constrained_smoother::SmoothingFailureInfo failure;
  if (!run_smooth_request(
      smoother,
      input.path,
      input.start_dir,
      input.end_dir,
      input.costmap,
      params,
      precomputed_esdf,
      &failure))
  {
    return make_error_result(failure, py::cast(input.path));
  }

  return make_ok_result(input.path);
}

PyObject * run_smooth_binding(
  constrained_smoother::KinematicSmoother & smoother,
  const py::handle & path_handle,
  const py::handle & start_dir_handle,
  const py::handle & end_dir_handle,
  const py::handle & costmap_handle,
  const constrained_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf)
{
  return run_smooth_or_raise(
    smoother,
    parse_smooth_input(path_handle, start_dir_handle, end_dir_handle, costmap_handle),
    params,
    precomputed_esdf);
}

py::dict run_try_smooth_binding(
  constrained_smoother::KinematicSmoother & smoother,
  const py::handle & path_handle,
  const py::handle & start_dir_handle,
  const py::handle & end_dir_handle,
  const py::handle & costmap_handle,
  const constrained_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf)
{
  return invoke_try_smooth([&]() -> py::dict {
    return run_try_smooth_result(
      smoother,
      parse_smooth_input(path_handle, start_dir_handle, end_dir_handle, costmap_handle),
      params,
      precomputed_esdf);
  });
}

template<typename Fn>
py::dict invoke_try_smooth(Fn && fn)
{
  try {
    return fn();
  } catch (const constrained_smoother::InvalidPath & error) {
    return make_error_result(error);
  } catch (const constrained_smoother::FailedToSmoothPath & error) {
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
}

}  // namespace

PYBIND11_MODULE(py_constrained_smoother, m)
{
  m.doc() = "Python bindings for the constrained_smoother C++ library";

  // ---- Stable error-code surface ----

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

  // ---- Core value types and planning utilities ----

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
    .def_readwrite("model_weight_sqrt", &constrained_smoother::SmootherParams::model_weight_sqrt)
    .def_readwrite(
    "costmap_weight_sqrt",
    &constrained_smoother::SmootherParams::costmap_weight_sqrt)
    .def_readwrite(
    "cusp_costmap_weight_sqrt",
    &constrained_smoother::SmootherParams::cusp_costmap_weight_sqrt)
    .def_readwrite("cusp_zone_length", &constrained_smoother::SmootherParams::cusp_zone_length)
    .def_readwrite(
    "reference_path_weight_sqrt",
    &constrained_smoother::SmootherParams::reference_path_weight_sqrt)
    .def_readwrite(
    "reference_point_max_deviation_m",
    &constrained_smoother::SmootherParams::reference_point_max_deviation_m)
    .def_readwrite(
    "kinematic_curvature_weight_sqrt",
    &constrained_smoother::SmootherParams::kinematic_curvature_weight_sqrt)
    .def_readwrite(
    "kinematic_curvature_rate_weight_sqrt",
    &constrained_smoother::SmootherParams::kinematic_curvature_rate_weight_sqrt)
    .def_readwrite(
    "kinematic_spacing_weight_sqrt",
    &constrained_smoother::SmootherParams::kinematic_spacing_weight_sqrt)
    .def_readwrite(
    "path_length_weight_sqrt",
    &constrained_smoother::SmootherParams::path_length_weight_sqrt)
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
    .def_readwrite(
    "goal_longitudinal_tolerance",
    &constrained_smoother::SmootherParams::goal_longitudinal_tolerance)
    .def_readwrite(
    "goal_lateral_tolerance",
    &constrained_smoother::SmootherParams::goal_lateral_tolerance)
    .def_readwrite(
    "goal_orientation_tolerance",
    &constrained_smoother::SmootherParams::goal_orientation_tolerance)
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
    .def_property(
    "linear_solver_type",
    [](const constrained_smoother::OptimizerParams & params) {
      return std::string(
        constrained_smoother::OptimizerParams::linearSolverToString(params.linear_solver));
    },
    [](constrained_smoother::OptimizerParams & params, const std::string & solver_name) {
      params.linear_solver =
        constrained_smoother::OptimizerParams::linearSolverFromString(solver_name);
    })
    .def_readwrite("max_iterations", &constrained_smoother::OptimizerParams::max_iterations)
    .def_readwrite("parameter_tolerance", &constrained_smoother::OptimizerParams::parameter_tolerance)
    .def_readwrite("function_tolerance", &constrained_smoother::OptimizerParams::function_tolerance)
    .def_readwrite("gradient_tolerance", &constrained_smoother::OptimizerParams::gradient_tolerance);

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

  // ---- Kinematic smoother bindings ----
  // This is now the only smoothing backend exposed by the standalone module.

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
      const py::handle & costmap_handle,
      const constrained_smoother::SmootherParams & params) -> PyObject *
      {
        return run_smooth_binding(
          self,
          path_handle,
          start_dir_handle,
          end_dir_handle,
          costmap_handle,
          params,
          nullptr);
      },
      py::return_value_policy::take_ownership,
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap") = py::none(), py::arg("params"),
      // 异常式接口：失败时抛 Python 异常，成功时返回运动学平滑后的路径。
      "Smooth a path using the kinematic backend. Input path z must encode direction sign (+1/-1); returned path z is yaw in radians.")
    .def(
      "try_smooth",
      [](constrained_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const py::handle & costmap_handle,
      const constrained_smoother::SmootherParams & params) -> py::dict
      {
        return run_try_smooth_binding(
          self,
          path_handle,
          start_dir_handle,
          end_dir_handle,
          costmap_handle,
          params,
          nullptr);
      },
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap") = py::none(), py::arg("params"),
      // 结构化接口：把运动学后端失败统一折叠成 ok/error_* 字段。
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
        return run_smooth_binding(
          self,
          path_handle,
          start_dir_handle,
          end_dir_handle,
          py::cast(costmap),
          params,
          &planner.getESDF());
      },
      py::return_value_policy::take_ownership,
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"), py::arg("planner"),
      // 异常式接口：复用 planner 已算好的 ESDF，失败时抛 Python 异常。
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
        return run_try_smooth_binding(
          self,
          path_handle,
          start_dir_handle,
          end_dir_handle,
          py::cast(costmap),
          params,
          &planner.getESDF());
      },
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"), py::arg("planner"),
      // 结构化接口：复用 planner ESDF，同时保持稳定的 ok/error_* 返回面。
      "Try to smooth a path with the kinematic backend and planner ESDF, returning a structured result.");

  // ---- Native exception translation ----

  py::register_exception<constrained_smoother::InvalidPath>(m, "InvalidPathError");
  py::register_exception<constrained_smoother::FailedToSmoothPath>(m, "FailedToSmoothPathError");
  py::register_exception<constrained_smoother::InvalidCostmap>(m, "InvalidCostmapError");
  py::register_exception<constrained_smoother::PrecomputedEsdfSizeMismatch>(
    m,
    "PrecomputedEsdfSizeMismatchError");
}
