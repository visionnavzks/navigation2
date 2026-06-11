#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>

#include "smoother_clothoid/costmap2d.hpp"
#include "smoother_clothoid/exceptions.hpp"
#include "smoother_clothoid/esdf.hpp"
#include "smoother_clothoid/options.hpp"
#include "smoother_clothoid/smoother.hpp"

#include <vector>
#include <cmath>

namespace nb = nanobind;

namespace
{

nb::sequence requireSeq(const nb::handle & h, const char * name)
{
  if (!nb::isinstance<nb::sequence>(h) || nb::isinstance<nb::str>(h))
    throw nb::value_error((std::string(name) + " must be a numeric sequence").c_str());
  return nb::cast<nb::sequence>(h);
}

Eigen::Vector2d toVec2(const nb::handle & h, const char * name)
{
  nb::sequence s = requireSeq(h, name);
  if (nb::len(s) != 2) throw nb::value_error((std::string(name) + " must have 2 values").c_str());
  return {nb::cast<double>(s[0]), nb::cast<double>(s[1])};
}

std::vector<Eigen::Vector3d> toPath3d(const nb::handle & h, const char * name)
{
  nb::sequence outer = requireSeq(h, name);
  std::vector<Eigen::Vector3d> path;
  path.reserve(nb::len(outer));
  for (size_t i = 0; i < static_cast<size_t>(nb::len(outer)); ++i) {
    nb::sequence pt = requireSeq(outer[i], name);
    if (nb::len(pt) != 3) throw nb::value_error((std::string(name) + " entries must have 3 values").c_str());
    path.emplace_back(nb::cast<double>(pt[0]), nb::cast<double>(pt[1]), nb::cast<double>(pt[2]));
  }
  return path;
}

const smoother_clothoid::Costmap2D * toCostmap(const nb::handle & h)
{
  return h.is_none() ? nullptr : &nb::cast<const smoother_clothoid::Costmap2D &>(h);
}

nb::str toNbStr(const std::string & s) { return nb::str(s.c_str()); }

template<typename E>
nb::dict errorResult(const E &)
{
  nb::dict r;
  r["ok"] = false;
  r["path"] = nb::none(); r["smoothed_path"] = nb::none(); r["candidate_path"] = nb::none();
  r["optimized_knot_count"] = 0; r["target_spacing_m"] = 0.0;
  r["error_code"] = nb::none();
  r["error_message"] = nb::none();
  r["error_reason"] = nb::none(); r["error_details"] = nb::none();
  return r;
}

nb::dict errorResult(const smoother_clothoid::SmoothingFailureInfo & f, const smoother_clothoid::SmootherResult & r)
{
  nb::dict d;
  d["ok"] = false;
  const nb::object cand = r.candidate_path.empty() ? nb::none() : nb::cast(r.candidate_path);
  const nb::object smooth = r.success ? nb::cast(r.smoothed_path) : nb::none();
  d["path"] = r.success ? smooth : cand;
  d["smoothed_path"] = smooth; d["candidate_path"] = cand;
  d["optimized_knot_count"] = nb::int_(r.optimized_knot_count);
  d["target_spacing_m"] = nb::float_(r.target_spacing);
  d["error_code"] = toNbStr(smoother_clothoid::toErrorCodeString(smoother_clothoid::ErrorCode::FailedToSmoothPath));
  d["error_message"] = toNbStr(f.message);
  d["error_reason"] = toNbStr(smoother_clothoid::toSmoothingFailureReasonString(f.reason));
  if (f.failed_index >= 0 || std::isfinite(f.actual_curvature) || std::isfinite(f.goal_longitudinal_error)) {
    nb::dict det;
    if (f.failed_index >= 0) det["failed_index"] = nb::int_(f.failed_index);
    if (std::isfinite(f.actual_curvature)) det["actual_curvature"] = nb::float_(f.actual_curvature);
    if (std::isfinite(f.max_curvature)) det["max_curvature"] = nb::float_(f.max_curvature);
    if (std::isfinite(f.turning_radius)) det["turning_radius"] = nb::float_(f.turning_radius);
    if (std::isfinite(f.goal_longitudinal_error)) det["goal_longitudinal_error"] = nb::float_(f.goal_longitudinal_error);
    if (std::isfinite(f.goal_lateral_error)) det["goal_lateral_error"] = nb::float_(f.goal_lateral_error);
    d["error_details"] = det;
  } else {
    d["error_details"] = nb::none();
  }
  return d;
}

nb::dict okResult(const smoother_clothoid::SmootherResult & r)
{
  nb::dict d;
  d["ok"] = true;
  const nb::object cand = r.candidate_path.empty() ? nb::none() : nb::cast(r.candidate_path);
  d["path"] = nb::cast(r.smoothed_path);
  d["smoothed_path"] = nb::cast(r.smoothed_path);
  d["candidate_path"] = cand;
  d["optimized_knot_count"] = nb::int_(r.optimized_knot_count);
  d["target_spacing_m"] = nb::float_(r.target_spacing);
  d["error_code"] = nb::none(); d["error_message"] = nb::none();
  d["error_reason"] = nb::none(); d["error_details"] = nb::none();
  return d;
}

template<typename Fn>
nb::dict trySmooth(Fn && fn)
{
  try { return fn(); }
  catch (const smoother_clothoid::InvalidPath & e) {
    nb::dict r = errorResult(e);
    r["error_code"] = toNbStr(smoother_clothoid::toErrorCodeString(smoother_clothoid::ErrorCode::InvalidPath));
    r["error_message"] = toNbStr(e.what());
    return r;
  }
  catch (const smoother_clothoid::FailedToSmoothPath & e) {
    nb::dict r = errorResult(e);
    r["error_code"] = toNbStr(smoother_clothoid::toErrorCodeString(smoother_clothoid::ErrorCode::FailedToSmoothPath));
    r["error_message"] = toNbStr(e.what());
    return r;
  }
  catch (const smoother_clothoid::InvalidCostmap & e) {
    nb::dict r;
    r["ok"] = false; r["path"] = nb::none(); r["smoothed_path"] = nb::none();
    r["candidate_path"] = nb::none(); r["optimized_knot_count"] = 0; r["target_spacing_m"] = 0.0;
    r["error_code"] = toNbStr(smoother_clothoid::toErrorCodeString(smoother_clothoid::ErrorCode::InvalidCostmap));
    r["error_message"] = toNbStr(e.what());
    r["error_reason"] = nb::none(); r["error_details"] = nb::none();
    return r;
  }
  catch (const smoother_clothoid::PrecomputedEsdfSizeMismatch & e) {
    nb::dict r;
    r["ok"] = false; r["path"] = nb::none(); r["smoothed_path"] = nb::none();
    r["candidate_path"] = nb::none(); r["optimized_knot_count"] = 0; r["target_spacing_m"] = 0.0;
    r["error_code"] = toNbStr(smoother_clothoid::toErrorCodeString(smoother_clothoid::ErrorCode::PrecomputedEsdfSizeMismatch));
    r["error_message"] = toNbStr(e.what());
    r["error_reason"] = nb::none(); r["error_details"] = nb::none();
    return r;
  }
  catch (const nb::python_error &) { throw; }
  catch (const std::exception & e) {
    nb::dict r;
    r["ok"] = false; r["path"] = nb::none(); r["smoothed_path"] = nb::none();
    r["candidate_path"] = nb::none(); r["optimized_knot_count"] = 0; r["target_spacing_m"] = 0.0;
    r["error_code"] = nb::none(); r["error_message"] = toNbStr(e.what());
    r["error_reason"] = nb::none(); r["error_details"] = nb::none();
    return r;
  }
}

}  // namespace

NB_MODULE(nb_smoother_clothoid, m)
{
  m.doc() = "nanobind bindings for smoother_clothoid C++ library";

  nb::enum_<smoother_clothoid::ErrorCode>(m, "ErrorCode")
    .value("INVALID_PATH", smoother_clothoid::ErrorCode::InvalidPath)
    .value("FAILED_TO_SMOOTH_PATH", smoother_clothoid::ErrorCode::FailedToSmoothPath)
    .value("INVALID_COSTMAP", smoother_clothoid::ErrorCode::InvalidCostmap)
    .value("PRECOMPUTED_ESDF_SIZE_MISMATCH", smoother_clothoid::ErrorCode::PrecomputedEsdfSizeMismatch);

  nb::class_<smoother_clothoid::SmootherResult>(m, "SmootherResult")
    .def_ro("success", &smoother_clothoid::SmootherResult::success)
    .def_ro("candidate_path", &smoother_clothoid::SmootherResult::candidate_path)
    .def_ro("smoothed_path", &smoother_clothoid::SmootherResult::smoothed_path)
    .def_ro("optimized_knot_count", &smoother_clothoid::SmootherResult::optimized_knot_count)
    .def_prop_ro("target_spacing_m",
      [](const smoother_clothoid::SmootherResult & r) { return r.target_spacing; })
    .def_prop_ro("path",
      [](const smoother_clothoid::SmootherResult & r) { return r.smoothed_path; });

  nb::class_<smoother_clothoid::Costmap2D>(m, "Costmap2D")
    .def(nb::init<>())
    .def(nb::init<unsigned int, unsigned int, double, double, double>(),
      nb::arg("size_x"), nb::arg("size_y"), nb::arg("resolution"), nb::arg("origin_x"), nb::arg("origin_y"))
    .def("getSizeInCellsX", &smoother_clothoid::Costmap2D::getSizeInCellsX)
    .def("getSizeInCellsY", &smoother_clothoid::Costmap2D::getSizeInCellsY)
    .def("getResolution", &smoother_clothoid::Costmap2D::getResolution)
    .def("getOriginX", &smoother_clothoid::Costmap2D::getOriginX)
    .def("getOriginY", &smoother_clothoid::Costmap2D::getOriginY)
    .def("getCost", &smoother_clothoid::Costmap2D::getCost)
    .def("setCost", &smoother_clothoid::Costmap2D::setCost)
    .def_prop_ro_static("NO_INFORMATION", [](const nb::object &) { return smoother_clothoid::Costmap2D::NO_INFORMATION; })
    .def_prop_ro_static("LETHAL_OBSTACLE", [](const nb::object &) { return smoother_clothoid::Costmap2D::LETHAL_OBSTACLE; })
    .def_prop_ro_static("INSCRIBED_INFLATED_OBSTACLE", [](const nb::object &) { return smoother_clothoid::Costmap2D::INSCRIBED_INFLATED_OBSTACLE; })
    .def_prop_ro_static("FREE_SPACE", [](const nb::object &) { return smoother_clothoid::Costmap2D::FREE_SPACE; });

  nb::class_<smoother_clothoid::SmootherParams>(m, "SmootherParams")
    .def(nb::init<>())
    .def_rw("model_weight_sqrt", &smoother_clothoid::SmootherParams::model_weight_sqrt)
    .def_rw("costmap_weight_sqrt", &smoother_clothoid::SmootherParams::costmap_weight_sqrt)
    .def_rw("cusp_costmap_weight_sqrt", &smoother_clothoid::SmootherParams::cusp_costmap_weight_sqrt)
    .def_rw("cusp_zone_length", &smoother_clothoid::SmootherParams::cusp_zone_length)
    .def_rw("reference_path_weight_sqrt", &smoother_clothoid::SmootherParams::reference_path_weight_sqrt)
    .def_rw("reference_point_max_deviation_m", &smoother_clothoid::SmootherParams::reference_point_max_deviation_m)
    .def_rw("kinematic_curvature_weight_sqrt", &smoother_clothoid::SmootherParams::kinematic_curvature_weight_sqrt)
    .def_rw("kinematic_curvature_rate_weight_sqrt", &smoother_clothoid::SmootherParams::kinematic_curvature_rate_weight_sqrt)
    .def_rw("kinematic_spacing_weight_sqrt", &smoother_clothoid::SmootherParams::kinematic_spacing_weight_sqrt)
    .def_rw("kinematic_max_spacing", &smoother_clothoid::SmootherParams::kinematic_max_spacing)
    .def_rw("path_length_weight_sqrt", &smoother_clothoid::SmootherParams::path_length_weight_sqrt)
    .def_rw("fix_weight", &smoother_clothoid::SmootherParams::fix_weight)
    .def_rw("max_curvature", &smoother_clothoid::SmootherParams::max_curvature)
    .def_rw("max_time", &smoother_clothoid::SmootherParams::max_time)
    .def_rw("use_exact_esdf", &smoother_clothoid::SmootherParams::use_exact_esdf)
    .def_rw("obstacle_safe_distance", &smoother_clothoid::SmootherParams::obstacle_safe_distance)
    .def_rw("cost_check_radius", &smoother_clothoid::SmootherParams::cost_check_radius)
    .def_rw("path_downsampling_factor", &smoother_clothoid::SmootherParams::path_downsampling_factor)
    .def_rw("path_upsampling_factor", &smoother_clothoid::SmootherParams::path_upsampling_factor)
    .def_rw("goal_longitudinal_tolerance", &smoother_clothoid::SmootherParams::goal_longitudinal_tolerance)
    .def_rw("goal_lateral_tolerance", &smoother_clothoid::SmootherParams::goal_lateral_tolerance)
    .def_rw("goal_orientation_tolerance", &smoother_clothoid::SmootherParams::goal_orientation_tolerance)
    .def_rw("reversing_enabled", &smoother_clothoid::SmootherParams::reversing_enabled)
    .def_rw("keep_goal_orientation", &smoother_clothoid::SmootherParams::keep_goal_orientation)
    .def_rw("keep_start_orientation", &smoother_clothoid::SmootherParams::keep_start_orientation)
    .def_rw("cost_check_points", &smoother_clothoid::SmootherParams::cost_check_points);

  nb::class_<smoother_clothoid::OptimizerParams>(m, "OptimizerParams")
    .def(nb::init<>())
    .def_rw("debug", &smoother_clothoid::OptimizerParams::debug)
    .def_prop_ro("linear_solver_type",
      [](const smoother_clothoid::OptimizerParams & p) {
        return std::string(smoother_clothoid::OptimizerParams::linearSolverToString(p.linear_solver));
      })
    .def("set_linear_solver_type", [](smoother_clothoid::OptimizerParams & p, const std::string & s) {
      p.linear_solver = smoother_clothoid::OptimizerParams::linearSolverFromString(s);
    })
    .def_rw("max_iterations", &smoother_clothoid::OptimizerParams::max_iterations)
    .def_rw("parameter_tolerance", &smoother_clothoid::OptimizerParams::parameter_tolerance)
    .def_rw("function_tolerance", &smoother_clothoid::OptimizerParams::function_tolerance)
    .def_rw("gradient_tolerance", &smoother_clothoid::OptimizerParams::gradient_tolerance);

  m.def("compute_esdf",
    [](const smoother_clothoid::Costmap2D & c, unsigned char lc, bool exact) {
      return smoother_clothoid::ESDF::ComputeESDF(&c, lc,
        exact ? smoother_clothoid::ESDFAlgorithm::Exact : smoother_clothoid::ESDFAlgorithm::Approximate);
    },
    nb::arg("costmap"), nb::arg("lethal_cost") = smoother_clothoid::Costmap2D::LETHAL_OBSTACLE,
    nb::arg("use_exact") = true);

  nb::class_<smoother_clothoid::ClothoidSmoother>(m, "ClothoidSmoother")
    .def(nb::init<>())
    .def("initialize", &smoother_clothoid::ClothoidSmoother::initialize)
    .def("smooth",
      [](smoother_clothoid::ClothoidSmoother & self,
         const nb::handle & ph, const nb::handle & sh, const nb::handle & eh,
         const nb::handle & ch, const smoother_clothoid::SmootherParams & params) -> nb::object
      {
        smoother_clothoid::SmoothingFailureInfo f;
        const auto r = self.smooth({toPath3d(ph,"path"), toVec2(sh,"start_dir"), toVec2(eh,"end_dir"),
          toCostmap(ch), params, nullptr, &f});
        if (r.success) return nb::cast(r);
        PyErr_SetString(PyExc_RuntimeError,
          (std::string(smoother_clothoid::toErrorCodeString(smoother_clothoid::ErrorCode::FailedToSmoothPath))
           + ": " + f.formattedMessage()).c_str());
        return nb::none();
      },
      nb::arg("path"), nb::arg("start_dir"), nb::arg("end_dir"),
      nb::arg("costmap") = nb::none(), nb::arg("params"))
    .def("try_smooth",
      [](smoother_clothoid::ClothoidSmoother & self,
         const nb::handle & ph, const nb::handle & sh, const nb::handle & eh,
         const nb::handle & ch, const smoother_clothoid::SmootherParams & params) -> nb::dict
      {
        return trySmooth([&]() {
          smoother_clothoid::SmoothingFailureInfo f;
          const auto r = self.smooth({toPath3d(ph,"path"), toVec2(sh,"start_dir"), toVec2(eh,"end_dir"),
            toCostmap(ch), params, nullptr, &f});
          return r.success ? okResult(r) : errorResult(f, r);
        });
      },
      nb::arg("path"), nb::arg("start_dir"), nb::arg("end_dir"),
      nb::arg("costmap") = nb::none(), nb::arg("params"));

  nb::exception<smoother_clothoid::InvalidPath>(m, "InvalidPathError");
  nb::exception<smoother_clothoid::FailedToSmoothPath>(m, "FailedToSmoothPathError");
  nb::exception<smoother_clothoid::InvalidCostmap>(m, "InvalidCostmapError");
  nb::exception<smoother_clothoid::PrecomputedEsdfSizeMismatch>(m, "PrecomputedEsdfSizeMismatchError");
}
