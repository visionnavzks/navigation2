// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0
//
// Minimal pybind11 bindings for the hybrid_astar planner used by the
// constrained_smoother web demo. Only the subset of the API the demo needs is
// exposed - the goal is to feed the planner a costmap and a (start, goal)
// pair and return a Path.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "hybrid_astar/constants.hpp"
#include "hybrid_astar/costmap_2d.hpp"
#include "hybrid_astar/smac_planner_hybrid.hpp"
#include "hybrid_astar/types.hpp"

namespace py = pybind11;

namespace
{

// Convert a numpy / sequence costmap into a hybrid_astar::Costmap2D so the
// caller doesn't have to call setCost in a Python loop. Returned by unique_ptr
// because Costmap2D owns a recursive_mutex and is therefore non-copyable /
// non-movable.
std::unique_ptr<hybrid_astar::Costmap2D> make_costmap_from_buffer(
  unsigned int size_x,
  unsigned int size_y,
  double resolution,
  double origin_x,
  double origin_y,
  const py::sequence & data)
{
  const std::size_t expected = static_cast<std::size_t>(size_x) * size_y;
  if (static_cast<std::size_t>(py::len(data)) != expected) {
    throw py::value_error(
            "costmap data length must equal size_x * size_y");
  }
  auto costmap = std::make_unique<hybrid_astar::Costmap2D>(
    size_x, size_y, resolution, origin_x, origin_y);
  unsigned char * cells = costmap->getCharMap();
  for (std::size_t i = 0; i < expected; ++i) {
    cells[i] = static_cast<unsigned char>(py::cast<int>(data[i]));
  }
  return costmap;
}

// The web demo always treats the path as a list of (x, y, theta) tuples, so
// we convert hybrid_astar::Path into that representation directly.
std::vector<std::tuple<double, double, double>> path_to_tuples(
  const hybrid_astar::Path & path)
{
  std::vector<std::tuple<double, double, double>> result;
  result.reserve(path.size());
  for (const auto & p : path) {
    result.emplace_back(p.x, p.y, p.theta);
  }
  return result;
}

}  // namespace

PYBIND11_MODULE(py_hybrid_astar, m)
{
  m.doc() = "Python bindings for the hybrid_astar planner (web demo subset)";

  // ---- Cost constants ----
  m.attr("FREE_COST") = py::int_(static_cast<int>(hybrid_astar::FREE_COST));
  m.attr("MAX_NON_OBSTACLE_COST") =
    py::int_(static_cast<int>(hybrid_astar::MAX_NON_OBSTACLE_COST));
  m.attr("INSCRIBED_COST") = py::int_(static_cast<int>(hybrid_astar::INSCRIBED_COST));
  m.attr("OCCUPIED_COST") = py::int_(static_cast<int>(hybrid_astar::OCCUPIED_COST));
  m.attr("UNKNOWN_COST") = py::int_(static_cast<int>(hybrid_astar::UNKNOWN_COST));

  // ---- Pose ----
  py::class_<hybrid_astar::Pose>(m, "Pose")
    .def(py::init<>())
    .def(
      py::init([](double x, double y, double theta) {
        return hybrid_astar::Pose{x, y, theta};
      }),
      py::arg("x"), py::arg("y"), py::arg("theta"))
    .def_readwrite("x", &hybrid_astar::Pose::x)
    .def_readwrite("y", &hybrid_astar::Pose::y)
    .def_readwrite("theta", &hybrid_astar::Pose::theta)
    .def(
      "__repr__",
      [](const hybrid_astar::Pose & p) {
        return "<Pose x=" + std::to_string(p.x) +
               " y=" + std::to_string(p.y) +
               " theta=" + std::to_string(p.theta) + ">";
      });

  // ---- Costmap2D ----
  py::class_<hybrid_astar::Costmap2D>(m, "Costmap2D")
    .def(py::init<>())
    .def(
      py::init<unsigned int, unsigned int, double, double, double, unsigned char>(),
      py::arg("size_x"), py::arg("size_y"), py::arg("resolution"),
      py::arg("origin_x"), py::arg("origin_y"), py::arg("default_cost") = 0)
    .def("getCost", py::overload_cast<unsigned int, unsigned int>(
        &hybrid_astar::Costmap2D::getCost, py::const_))
    .def("setCost", &hybrid_astar::Costmap2D::setCost)
    .def("getSizeInCellsX", &hybrid_astar::Costmap2D::getSizeInCellsX)
    .def("getSizeInCellsY", &hybrid_astar::Costmap2D::getSizeInCellsY)
    .def("getResolution", &hybrid_astar::Costmap2D::getResolution)
    .def("getOriginX", &hybrid_astar::Costmap2D::getOriginX)
    .def("getOriginY", &hybrid_astar::Costmap2D::getOriginY);

  m.def(
    "make_costmap",
    &make_costmap_from_buffer,
    py::arg("size_x"), py::arg("size_y"), py::arg("resolution"),
    py::arg("origin_x"), py::arg("origin_y"), py::arg("data"),
    "Build a hybrid_astar.Costmap2D from a flat row-major sequence of cell costs.");

  // ---- SearchInfo (only the knobs the demo can flip) ----
  py::class_<hybrid_astar::SearchInfo>(m, "SearchInfo")
    .def(py::init<>())
    .def_readwrite("minimum_turning_radius", &hybrid_astar::SearchInfo::minimum_turning_radius)
    .def_readwrite("non_straight_penalty", &hybrid_astar::SearchInfo::non_straight_penalty)
    .def_readwrite("change_penalty", &hybrid_astar::SearchInfo::change_penalty)
    .def_readwrite("reverse_penalty", &hybrid_astar::SearchInfo::reverse_penalty)
    .def_readwrite("cost_penalty", &hybrid_astar::SearchInfo::cost_penalty)
    .def_readwrite("retrospective_penalty", &hybrid_astar::SearchInfo::retrospective_penalty)
    .def_readwrite("rotation_penalty", &hybrid_astar::SearchInfo::rotation_penalty)
    .def_readwrite("analytic_expansion_ratio", &hybrid_astar::SearchInfo::analytic_expansion_ratio)
    .def_readwrite(
      "analytic_expansion_max_length",
      &hybrid_astar::SearchInfo::analytic_expansion_max_length)
    .def_readwrite(
      "analytic_expansion_max_cost",
      &hybrid_astar::SearchInfo::analytic_expansion_max_cost)
    .def_readwrite(
      "analytic_expansion_max_cost_override",
      &hybrid_astar::SearchInfo::analytic_expansion_max_cost_override)
    .def_readwrite(
      "cache_obstacle_heuristic",
      &hybrid_astar::SearchInfo::cache_obstacle_heuristic)
    .def_readwrite(
      "allow_reverse_expansion",
      &hybrid_astar::SearchInfo::allow_reverse_expansion)
    .def_readwrite(
      "allow_primitive_interpolation",
      &hybrid_astar::SearchInfo::allow_primitive_interpolation)
    .def_readwrite(
      "downsample_obstacle_heuristic",
      &hybrid_astar::SearchInfo::downsample_obstacle_heuristic)
    .def_readwrite(
      "use_quadratic_cost_penalty",
      &hybrid_astar::SearchInfo::use_quadratic_cost_penalty);

  // ---- SmootherParams (native hybrid_astar smoother, not the Ceres one) ----
  py::class_<hybrid_astar::SmootherParams>(m, "NativeSmootherParams")
    .def(py::init<>())
    .def_readwrite("tolerance", &hybrid_astar::SmootherParams::tolerance_)
    .def_readwrite("max_its", &hybrid_astar::SmootherParams::max_its_)
    .def_readwrite("w_data", &hybrid_astar::SmootherParams::w_data_)
    .def_readwrite("w_smooth", &hybrid_astar::SmootherParams::w_smooth_)
    .def_readwrite("holonomic", &hybrid_astar::SmootherParams::holonomic_)
    .def_readwrite("do_refinement", &hybrid_astar::SmootherParams::do_refinement_)
    .def_readwrite("refinement_num", &hybrid_astar::SmootherParams::refinement_num_);

  // ---- SmacPlannerHybridConfig ----
  py::class_<hybrid_astar::SmacPlannerHybridConfig>(m, "SmacPlannerHybridConfig")
    .def(py::init<>())
    .def_readwrite(
      "downsample_costmap",
      &hybrid_astar::SmacPlannerHybridConfig::downsample_costmap)
    .def_readwrite(
      "downsampling_factor",
      &hybrid_astar::SmacPlannerHybridConfig::downsampling_factor)
    .def_readwrite(
      "angle_quantization_bins",
      &hybrid_astar::SmacPlannerHybridConfig::angle_quantization_bins)
    .def_readwrite("tolerance", &hybrid_astar::SmacPlannerHybridConfig::tolerance)
    .def_readwrite("allow_unknown", &hybrid_astar::SmacPlannerHybridConfig::allow_unknown)
    .def_readwrite("max_iterations", &hybrid_astar::SmacPlannerHybridConfig::max_iterations)
    .def_readwrite(
      "max_on_approach_iterations",
      &hybrid_astar::SmacPlannerHybridConfig::max_on_approach_iterations)
    .def_readwrite(
      "terminal_checking_interval",
      &hybrid_astar::SmacPlannerHybridConfig::terminal_checking_interval)
    .def_readwrite("smooth_path", &hybrid_astar::SmacPlannerHybridConfig::smooth_path)
    .def_readwrite("max_planning_time", &hybrid_astar::SmacPlannerHybridConfig::max_planning_time)
    .def_readwrite("lookup_table_size", &hybrid_astar::SmacPlannerHybridConfig::lookup_table_size)
    .def_readwrite(
      "debug_visualizations",
      &hybrid_astar::SmacPlannerHybridConfig::debug_visualizations)
    .def_readwrite(
      "motion_model_for_search",
      &hybrid_astar::SmacPlannerHybridConfig::motion_model_for_search)
    .def_readwrite(
      "goal_heading_mode",
      &hybrid_astar::SmacPlannerHybridConfig::goal_heading_mode)
    .def_readwrite(
      "coarse_search_resolution",
      &hybrid_astar::SmacPlannerHybridConfig::coarse_search_resolution)
    .def_readwrite("search_info", &hybrid_astar::SmacPlannerHybridConfig::search_info)
    .def_readwrite("smoother_params", &hybrid_astar::SmacPlannerHybridConfig::smoother_params)
    .def_property(
      "robot_footprint",
      [](const hybrid_astar::SmacPlannerHybridConfig & self) {
        std::vector<std::pair<double, double>> footprint;
        footprint.reserve(self.robot_footprint.size());
        for (const auto & point : self.robot_footprint) {
          footprint.emplace_back(point.x, point.y);
        }
        return footprint;
      },
      [](hybrid_astar::SmacPlannerHybridConfig & self,
         const std::vector<std::pair<double, double>> & footprint) {
        self.robot_footprint.clear();
        self.robot_footprint.reserve(footprint.size());
        for (const auto & point : footprint) {
          self.robot_footprint.push_back({point.first, point.second});
        }
      })
    .def_readwrite("use_radius", &hybrid_astar::SmacPlannerHybridConfig::use_radius)
    .def_readwrite(
      "circumscribed_cost",
      &hybrid_astar::SmacPlannerHybridConfig::circumscribed_cost)
    .def_readwrite("inflation_radius", &hybrid_astar::SmacPlannerHybridConfig::inflation_radius)
    .def_readwrite(
      "circumscribed_radius",
      &hybrid_astar::SmacPlannerHybridConfig::circumscribed_radius)
    .def_readwrite(
      "use_esdf_footprint",
      &hybrid_astar::SmacPlannerHybridConfig::use_esdf_footprint)
    .def_readwrite("use_exact_esdf", &hybrid_astar::SmacPlannerHybridConfig::use_exact_esdf)
    .def_readwrite(
      "cost_check_points",
      &hybrid_astar::SmacPlannerHybridConfig::cost_check_points)
    .def_readwrite("robot_radius", &hybrid_astar::SmacPlannerHybridConfig::robot_radius)
    .def_readwrite("safe_distance", &hybrid_astar::SmacPlannerHybridConfig::safe_distance);

  // ---- SmacPlannerHybrid ----
  py::class_<hybrid_astar::SmacPlannerHybrid>(m, "SmacPlannerHybrid")
    .def(py::init<>())
    .def(
      "configure",
      [](hybrid_astar::SmacPlannerHybrid & self,
         hybrid_astar::Costmap2D & costmap,
         const hybrid_astar::SmacPlannerHybridConfig & config) {
        self.configure(&costmap, config);
      },
      py::arg("costmap"), py::arg("config"),
      py::keep_alive<1, 2>())
    .def(
      "create_plan",
      [](hybrid_astar::SmacPlannerHybrid & self,
         const hybrid_astar::Pose & start,
         const hybrid_astar::Pose & goal) {
        // SmacPlannerHybrid::createPlan throws std::runtime_error from inside
        // the C++ search loop on conditions like "Start occupied" or
        // "No valid path found". We translate those failures to a Python dict
        // {ok, message, path} rather than letting the exception escape, because
        // when py_hybrid_astar and py_constrained_smoother are co-loaded into a
        // single Python interpreter the C++ runtime / pybind11 internals do not
        // share typeinfo across modules and cross-module exception unwinding
        // calls std::terminate(). The Python wrapper inspects ok/message and
        // raises an ApiError with the proper code from there.
        py::dict result;
        try {
          hybrid_astar::Path path = self.createPlan(
            start, goal, []() { return false; });
          result["ok"] = py::bool_(true);
          result["message"] = py::str("");
          result["path"] = py::cast(path_to_tuples(path));
        } catch (const std::exception & exc) {
          result["ok"] = py::bool_(false);
          result["message"] = py::str(exc.what());
          result["path"] = py::list();
        } catch (...) {
          result["ok"] = py::bool_(false);
          result["message"] = py::str("hybrid_astar: unknown failure");
          result["path"] = py::list();
        }
        return result;
      },
      py::arg("start"), py::arg("goal"),
      "Run Hybrid A*. Returns {ok, message, path} where path is a list of "
      "(x, y, theta) tuples; on failure ok is False, message is the C++ "
      "diagnostic, and path is empty.");
}
