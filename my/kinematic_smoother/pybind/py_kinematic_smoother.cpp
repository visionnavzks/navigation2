// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "kinematic_smoother/astar_esdf.hpp"
#include "kinematic_smoother/costmap2d.hpp"
#include "kinematic_smoother/footprint.hpp"
#include "kinematic_smoother/kinematic_smoother.hpp"
#include "kinematic_smoother/options.hpp"
#include "kinematic_smoother/exceptions.hpp"
#include "kinematic_smoother/esdf.hpp"

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

const kinematic_smoother::Costmap2D * copy_optional_costmap(const py::handle & handle)
{
  if (handle.is_none()) {
    return nullptr;
  }

  return &py::cast<const kinematic_smoother::Costmap2D &>(handle);
}

kinematic_smoother::SmootherResult run_smooth_request(
  kinematic_smoother::KinematicSmoother & smoother,
  const std::vector<Eigen::Vector3d> & path,
  const Eigen::Vector2d & start_dir,
  const Eigen::Vector2d & end_dir,
  const kinematic_smoother::Costmap2D * costmap,
  const kinematic_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf,
  kinematic_smoother::SmoothingFailureInfo * failure)
{
  const kinematic_smoother::SmootherRequest request{
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
  const kinematic_smoother::Costmap2D * costmap;
};

PyObject * make_python_smoothing_failure(const kinematic_smoother::SmoothingFailureInfo & failure);

template<typename ErrorT>
py::dict make_error_result(const ErrorT & error);

py::dict make_error_result(const kinematic_smoother::FailedToSmoothPath & error);
py::dict make_error_result(
  const kinematic_smoother::SmoothingFailureInfo & failure,
  const kinematic_smoother::SmootherResult & result);

py::dict make_ok_result(const kinematic_smoother::SmootherResult & result);

template<typename Fn>
py::dict invoke_try_smooth(Fn && fn);

py::list make_optional_float_list(const std::vector<double> & values)
{
  py::list result;
  for (const double value : values) {
    if (std::isfinite(value)) {
      result.append(py::float_(value));
    } else {
      result.append(py::none());
    }
  }
  return result;
}

py::object make_optional_float(double value)
{
  if (std::isfinite(value)) {
    return py::float_(value);
  }
  return py::none();
}

// 把 SmootherResult.cost_terms 折叠为稳定的 [{name, initial_cost, final_cost}] 列表；
// 未评估的阶段（如求解失败时的 final_cost）以 None 表示。
py::object make_cost_terms_payload(
  const std::vector<kinematic_smoother::SmootherCostTerm> & cost_terms)
{
  if (cost_terms.empty()) {
    return py::none();
  }
  py::list result;
  for (const auto & term : cost_terms) {
    py::dict entry;
    entry["name"] = py::str(term.name);
    entry["initial_cost"] = make_optional_float(term.initial_cost);
    entry["final_cost"] = make_optional_float(term.final_cost);
    result.append(entry);
  }
  return result;
}

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
//     "smoothed_path": list | None,
//     "candidate_path": list | None,
//     "optimized_knot_count": int,
//     "target_spacing_m": float,
//     "cost_terms": [{"name": str, "initial_cost": float | None, "final_cost": float | None}] | None,
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
  result["smoothed_path"] = py::none();
  result["candidate_path"] = py::none();
  result["optimized_knot_count"] = py::int_(0);
  result["target_spacing_m"] = py::float_(0.0);
  result["smoothed_curvatures"] = py::none();
  result["smoothed_curvature_rates"] = py::none();
  result["cost_terms"] = py::none();
  result["error_code"] = py::str(error.codeString());
  result["error_message"] = py::str(error.what());
  result["error_reason"] = py::none();
  result["error_details"] = py::none();
  return result;
}

void fill_result_payload(
  py::dict & result_dict,
  const kinematic_smoother::SmootherResult & smooth_result)
{
  const py::object candidate_path = smooth_result.candidate_path.empty() ?
    py::none() : py::cast(smooth_result.candidate_path);
  const py::object smoothed_path = smooth_result.success ?
    py::cast(smooth_result.smoothed_path) : py::none();

  result_dict["path"] = smooth_result.success ? smoothed_path : candidate_path;
  result_dict["smoothed_path"] = smoothed_path;
  result_dict["candidate_path"] = candidate_path;
  result_dict["optimized_knot_count"] = py::int_(smooth_result.optimized_knot_count);
  result_dict["target_spacing_m"] = py::float_(smooth_result.target_spacing);
  result_dict["cost_terms"] = make_cost_terms_payload(smooth_result.cost_terms);
  if (smooth_result.success) {
    result_dict["smoothed_curvatures"] =
      make_optional_float_list(smooth_result.smoothed_curvatures);
    result_dict["smoothed_curvature_rates"] =
      make_optional_float_list(smooth_result.smoothed_curvature_rates);
  } else {
    result_dict["smoothed_curvatures"] = py::none();
    result_dict["smoothed_curvature_rates"] = py::none();
  }
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

py::dict make_error_result(const kinematic_smoother::InvalidCostmap & error)
{
  py::dict result;
  result["ok"] = false;
  result["path"] = py::none();
  result["smoothed_path"] = py::none();
  result["candidate_path"] = py::none();
  result["optimized_knot_count"] = py::int_(0);
  result["target_spacing_m"] = py::float_(0.0);
  result["smoothed_curvatures"] = py::none();
  result["smoothed_curvature_rates"] = py::none();
  result["cost_terms"] = py::none();
  result["error_code"] = py::str(
    kinematic_smoother::toErrorCodeString(kinematic_smoother::ErrorCode::InvalidCostmap));
  result["error_message"] = py::str(error.what());
  result["error_reason"] = py::none();
  result["error_details"] = py::none();
  return result;
}

py::dict make_error_result(
  const kinematic_smoother::PrecomputedEsdfSizeMismatch & error)
{
  py::dict result;
  result["ok"] = false;
  result["path"] = py::none();
  result["smoothed_path"] = py::none();
  result["candidate_path"] = py::none();
  result["optimized_knot_count"] = py::int_(0);
  result["target_spacing_m"] = py::float_(0.0);
  result["smoothed_curvatures"] = py::none();
  result["smoothed_curvature_rates"] = py::none();
  result["cost_terms"] = py::none();
  result["error_code"] = py::str(
    kinematic_smoother::toErrorCodeString(
      kinematic_smoother::ErrorCode::PrecomputedEsdfSizeMismatch));
  result["error_message"] = py::str(error.what());
  result["error_reason"] = py::none();
  result["error_details"] = py::none();
  return result;
}

py::dict make_error_result(
  const kinematic_smoother::SmoothingFailureInfo & failure,
  const kinematic_smoother::SmootherResult & smooth_result)
{
  py::dict result;
  result["ok"] = false;
  fill_result_payload(result, smooth_result);
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

py::dict make_ok_result(const kinematic_smoother::SmootherResult & smooth_result)
{
  py::dict result;
  result["ok"] = true;
  fill_result_payload(result, smooth_result);
  result["error_code"] = py::none();
  result["error_message"] = py::none();
  result["error_reason"] = py::none();
  result["error_details"] = py::none();
  return result;
}

PyObject * run_smooth_or_raise(
  kinematic_smoother::KinematicSmoother & smoother,
  SmoothBindingInput && input,
  const kinematic_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf)
{
  kinematic_smoother::SmoothingFailureInfo failure;
  const kinematic_smoother::SmootherResult result = run_smooth_request(
    smoother,
    input.path,
    input.start_dir,
    input.end_dir,
    input.costmap,
    params,
    precomputed_esdf,
    &failure);
  if (!result.success) {
    return make_python_smoothing_failure(failure);
  }

  return py::cast(result).release().ptr();
}

py::dict run_try_smooth_result(
  kinematic_smoother::KinematicSmoother & smoother,
  SmoothBindingInput && input,
  const kinematic_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf)
{
  kinematic_smoother::SmoothingFailureInfo failure;
  const kinematic_smoother::SmootherResult result = run_smooth_request(
    smoother,
    input.path,
    input.start_dir,
    input.end_dir,
    input.costmap,
    params,
    precomputed_esdf,
    &failure);
  if (!result.success) {
    return make_error_result(failure, result);
  }

  return make_ok_result(result);
}

PyObject * run_smooth_binding(
  kinematic_smoother::KinematicSmoother & smoother,
  const py::handle & path_handle,
  const py::handle & start_dir_handle,
  const py::handle & end_dir_handle,
  const py::handle & costmap_handle,
  const kinematic_smoother::SmootherParams & params,
  const std::vector<double> * precomputed_esdf)
{
  return run_smooth_or_raise(
    smoother,
    parse_smooth_input(path_handle, start_dir_handle, end_dir_handle, costmap_handle),
    params,
    precomputed_esdf);
}

py::dict run_try_smooth_binding(
  kinematic_smoother::KinematicSmoother & smoother,
  const py::handle & path_handle,
  const py::handle & start_dir_handle,
  const py::handle & end_dir_handle,
  const py::handle & costmap_handle,
  const kinematic_smoother::SmootherParams & params,
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
  } catch (const kinematic_smoother::InvalidPath & error) {
    return make_error_result(error);
  } catch (const kinematic_smoother::FailedToSmoothPath & error) {
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
    result["smoothed_path"] = py::none();
    result["candidate_path"] = py::none();
    result["optimized_knot_count"] = py::int_(0);
    result["target_spacing_m"] = py::float_(0.0);
    result["smoothed_curvatures"] = py::none();
    result["smoothed_curvature_rates"] = py::none();
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

  // ---- Footprint model (geometric precomputation) ----
  // shared by smoother / validator / A*. Pure geometry, no costmap/ESDF read.

  py::enum_<kinematic_smoother::FootprintMode>(m, "FootprintMode")
    .value("POINT", kinematic_smoother::FootprintMode::Point)
    .value("CAPSULE", kinematic_smoother::FootprintMode::Capsule);

  py::enum_<kinematic_smoother::CapsuleMode>(m, "CapsuleMode")
    .value("CONSERVATIVE", kinematic_smoother::CapsuleMode::Conservative)
    .value("EXACT", kinematic_smoother::CapsuleMode::Exact);

  py::class_<kinematic_smoother::FootprintSpec>(m, "FootprintSpec")
    .def(py::init<>())
    .def_readwrite("mode", &kinematic_smoother::FootprintSpec::mode)
    .def_readwrite("capsule_mode", &kinematic_smoother::FootprintSpec::capsule_mode)
    .def_readwrite("length_m", &kinematic_smoother::FootprintSpec::length_m)
    .def_readwrite("width_m", &kinematic_smoother::FootprintSpec::width_m)
    .def_readwrite(
    "point_radius_m", &kinematic_smoother::FootprintSpec::point_radius_m)
    .def_readwrite(
    "sampling_tolerance_m", &kinematic_smoother::FootprintSpec::sampling_tolerance_m)
    .def_readwrite(
    "min_resolution_m", &kinematic_smoother::FootprintSpec::min_resolution_m);

  py::class_<kinematic_smoother::FootprintModel>(m, "FootprintModel")
    .def(py::init<>())
    .def_readwrite(
    "check_radius", &kinematic_smoother::FootprintModel::check_radius)
    .def_readwrite(
    "check_points", &kinematic_smoother::FootprintModel::check_points);

  m.def(
    "build_footprint_model",
    &kinematic_smoother::buildFootprintModel,
    py::arg("spec"),
    "Build a discrete (radius, check_points) footprint model from a FootprintSpec. "
    "Pure geometric precomputation; does not read costmap/ESDF.");

  py::class_<kinematic_smoother::SmootherResult>(m, "SmootherResult")
    .def_readonly("success", &kinematic_smoother::SmootherResult::success)
    .def_readonly("candidate_path", &kinematic_smoother::SmootherResult::candidate_path)
    .def_readonly("smoothed_path", &kinematic_smoother::SmootherResult::smoothed_path)
    .def_readonly(
      "smoothed_curvatures",
      &kinematic_smoother::SmootherResult::smoothed_curvatures)
    .def_readonly(
      "smoothed_curvature_rates",
      &kinematic_smoother::SmootherResult::smoothed_curvature_rates)
    .def_readonly(
      "optimized_knot_count",
      &kinematic_smoother::SmootherResult::optimized_knot_count)
    .def_property_readonly(
      "cost_terms",
      [](const kinematic_smoother::SmootherResult & result) {
        return make_cost_terms_payload(result.cost_terms);
      })
    .def_property_readonly(
      "target_spacing_m",
      [](const kinematic_smoother::SmootherResult & result) {
        return result.target_spacing;
      })
    .def_property_readonly(
      "path",
      [](const kinematic_smoother::SmootherResult & result) {
        return result.smoothed_path;
      });

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

  // --- ValidationTolerances ---
  // 求解后硬验收的容差表；默认接受起终点 2 cm 位置偏差、1° 终点朝向偏差。
  py::class_<kinematic_smoother::ValidationTolerances>(m, "ValidationTolerances")
    .def(py::init<>())
    .def_readwrite(
    "start_position_m",
    &kinematic_smoother::ValidationTolerances::start_position_m)
    .def_readwrite(
    "goal_position_m",
    &kinematic_smoother::ValidationTolerances::goal_position_m)
    .def_readwrite(
    "goal_position_numerical_slack_m",
    &kinematic_smoother::ValidationTolerances::goal_position_numerical_slack_m)
    .def_readwrite(
    "cusp_position_m",
    &kinematic_smoother::ValidationTolerances::cusp_position_m)
    .def_readwrite(
    "min_segment_displacement_m",
    &kinematic_smoother::ValidationTolerances::min_segment_displacement_m)
    .def_readwrite(
    "start_orientation_rad",
    &kinematic_smoother::ValidationTolerances::start_orientation_rad)
    .def_readwrite(
    "goal_orientation_rad",
    &kinematic_smoother::ValidationTolerances::goal_orientation_rad)
    .def_readwrite(
    "cusp_orientation_rad",
    &kinematic_smoother::ValidationTolerances::cusp_orientation_rad)
    .def_readwrite(
    "curvature_tolerance",
    &kinematic_smoother::ValidationTolerances::curvature_tolerance);

  // --- SmootherParams ---
  py::class_<kinematic_smoother::SmootherParams>(m, "SmootherParams")
    .def(py::init<>())
    .def_readwrite("model_weight", &kinematic_smoother::SmootherParams::model_weight)
    .def_readwrite(
    "obstacle_weight",
    &kinematic_smoother::SmootherParams::obstacle_weight)
    .def_readwrite(
    "reference_path_weight",
    &kinematic_smoother::SmootherParams::reference_path_weight)
    .def_readwrite(
    "reference_point_max_deviation_m",
    &kinematic_smoother::SmootherParams::reference_point_max_deviation_m)
    .def_readwrite(
    "kinematic_curvature_weight",
    &kinematic_smoother::SmootherParams::kinematic_curvature_weight)
    .def_readwrite(
    "kinematic_curvature_rate_weight",
    &kinematic_smoother::SmootherParams::kinematic_curvature_rate_weight)
    .def_readwrite(
    "kinematic_spacing_weight",
    &kinematic_smoother::SmootherParams::kinematic_spacing_weight)
    .def_readwrite(
    "kinematic_max_spacing",
    &kinematic_smoother::SmootherParams::kinematic_max_spacing)
    .def_readwrite(
    "path_length_weight",
    &kinematic_smoother::SmootherParams::path_length_weight)
    .def_readwrite("fix_weight", &kinematic_smoother::SmootherParams::fix_weight)
    .def_readwrite("max_curvature", &kinematic_smoother::SmootherParams::max_curvature)
    .def_readwrite("max_time", &kinematic_smoother::SmootherParams::max_time)
    .def_readwrite("use_exact_esdf", &kinematic_smoother::SmootherParams::use_exact_esdf)
    .def_readwrite(
    "obstacle_safe_distance",
    &kinematic_smoother::SmootherParams::obstacle_safe_distance)
    .def_readwrite(
    "costmap_boundary_margin",
    &kinematic_smoother::SmootherParams::costmap_boundary_margin)
    .def_readwrite(
    "cost_check_radius",
    &kinematic_smoother::SmootherParams::cost_check_radius)
    .def_readwrite(
    "path_target_spacing",
    &kinematic_smoother::SmootherParams::path_target_spacing)
    .def_readwrite(
    "path_downsampling_factor",
    &kinematic_smoother::SmootherParams::path_downsampling_factor)
    .def_readwrite(
    "path_upsampling_factor",
    &kinematic_smoother::SmootherParams::path_upsampling_factor)
    .def_readwrite(
    "path_output_spacing",
    &kinematic_smoother::SmootherParams::path_output_spacing)
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
    &kinematic_smoother::SmootherParams::cost_check_points)
    .def_readwrite("validation", &kinematic_smoother::SmootherParams::validation);

  // --- OptimizerParams ---
  py::class_<kinematic_smoother::OptimizerParams>(m, "OptimizerParams")
    .def(py::init<>())
    .def_readwrite("debug", &kinematic_smoother::OptimizerParams::debug)
    .def_readwrite("max_iterations", &kinematic_smoother::OptimizerParams::max_iterations)
    .def_readwrite("parameter_tolerance", &kinematic_smoother::OptimizerParams::parameter_tolerance)
    .def_readwrite("function_tolerance", &kinematic_smoother::OptimizerParams::function_tolerance)
    .def_readwrite("gradient_tolerance", &kinematic_smoother::OptimizerParams::gradient_tolerance);

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

  // ---- Kinematic smoother bindings ----
  // This is now the only smoothing backend exposed by the standalone module.

  py::class_<kinematic_smoother::KinematicSmoother>(m, "KinematicSmoother")
    .def(py::init<>())
    .def("initialize", &kinematic_smoother::KinematicSmoother::initialize)
    .def(
      "smooth",
      [](kinematic_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const py::handle & costmap_handle,
      const kinematic_smoother::SmootherParams & params) -> PyObject *
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
      // costmap 必须在必填的 params 之前给出实参；保留默认值会让 params 沦为
      // “可选实参后的必填实参”，positional 调用 smooth(path, sd, ed, params) 会把
      // params 误绑到 costmap 槽位。不需要 costmap 时显式传 None。
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"),
      // 异常式接口：失败时抛 Python 异常，成功时返回显式结果对象。
      "Smooth a path using the kinematic backend. Input path z must encode direction sign (+1/-1); the returned result carries both candidate/final paths and optimization diagnostics.")
    .def(
      "try_smooth",
      [](kinematic_smoother::KinematicSmoother & self,
      const py::handle & path_handle,
      const py::handle & start_dir_handle,
      const py::handle & end_dir_handle,
      const py::handle & costmap_handle,
      const kinematic_smoother::SmootherParams & params) -> py::dict
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
      // costmap 必须在必填的 params 之前给出实参；保留默认值会让 params 沦为
      // “可选实参后的必填实参”，positional 调用 smooth(path, sd, ed, params) 会把
      // params 误绑到 costmap 槽位。不需要 costmap 时显式传 None。
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"),
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
      "Smooth a path with the kinematic backend while reusing the ESDF previously computed by an A* planner, returning a structured result object on success.")
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

  py::register_exception<kinematic_smoother::InvalidPath>(m, "InvalidPathError");
  py::register_exception<kinematic_smoother::FailedToSmoothPath>(m, "FailedToSmoothPathError");
  py::register_exception<kinematic_smoother::InvalidCostmap>(m, "InvalidCostmapError");
  py::register_exception<kinematic_smoother::PrecomputedEsdfSizeMismatch>(
    m,
    "PrecomputedEsdfSizeMismatchError");
}
