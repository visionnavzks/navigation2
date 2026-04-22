// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "constrained_smoother/astar_esdf.hpp"
#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/kinematic_smoother_simple.hpp"
#include "constrained_smoother/options.hpp"
#include "constrained_smoother/smoother.hpp"
#include "constrained_smoother/exceptions.hpp"
#include "constrained_smoother/esdf.hpp"

#include <vector>
#include <cmath>

namespace py = pybind11;

PYBIND11_MODULE(py_constrained_smoother, m)
{
  m.doc() = "Python bindings for the constrained_smoother C++ library";

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
    std::vector<Eigen::Vector3d> path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const constrained_smoother::Costmap2D & costmap,
    const constrained_smoother::SmootherParams & params) -> std::vector<Eigen::Vector3d>
    {
      self.smooth(path, start_dir, end_dir, &costmap, params);
      return path;
    },
    py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
    py::arg("costmap"), py::arg("params"),
    "Smooth a path. Input path z must encode direction sign (+1/-1); returned path z is yaw in radians.")
    .def(
    "smooth_with_planner_esdf",
    [](constrained_smoother::Smoother & self,
    std::vector<Eigen::Vector3d> path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const constrained_smoother::Costmap2D & costmap,
    const constrained_smoother::SmootherParams & params,
    const constrained_smoother::AStarPlanner & planner) -> std::vector<Eigen::Vector3d>
    {
      self.smooth(path, start_dir, end_dir, &costmap, params, &planner.getESDF());
      return path;
    },
    py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
    py::arg("costmap"), py::arg("params"), py::arg("planner"),
    "Smooth a path while reusing the ESDF previously computed by an A* planner.");

  py::class_<constrained_smoother::SimpleKinematicSmoother>(m, "SimpleKinematicSmoother")
    .def(py::init<>())
    .def("initialize", &constrained_smoother::SimpleKinematicSmoother::initialize)
    .def(
      "get_last_optimized_knot_count",
      &constrained_smoother::SimpleKinematicSmoother::getLastOptimizedKnotCount)
    .def(
      "smooth",
      [](constrained_smoother::SimpleKinematicSmoother & self,
      std::vector<Eigen::Vector3d> path,
      const Eigen::Vector2d & start_dir,
      const Eigen::Vector2d & end_dir,
      const constrained_smoother::Costmap2D & costmap,
      const constrained_smoother::SmootherParams & params) -> std::vector<Eigen::Vector3d>
      {
        self.smooth(path, start_dir, end_dir, &costmap, params);
        return path;
      },
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"),
      "Smooth a path using the simple kinematic backend. Input path z must encode direction sign (+1/-1); returned path z is yaw in radians.")
    .def(
      "smooth_with_planner_esdf",
      [](constrained_smoother::SimpleKinematicSmoother & self,
      std::vector<Eigen::Vector3d> path,
      const Eigen::Vector2d & start_dir,
      const Eigen::Vector2d & end_dir,
      const constrained_smoother::Costmap2D & costmap,
      const constrained_smoother::SmootherParams & params,
      const constrained_smoother::AStarPlanner & planner) -> std::vector<Eigen::Vector3d>
      {
        self.smooth(path, start_dir, end_dir, &costmap, params, &planner.getESDF());
        return path;
      },
      py::arg("path"), py::arg("start_dir"), py::arg("end_dir"),
      py::arg("costmap"), py::arg("params"), py::arg("planner"),
      "Smooth a path with the simple kinematic backend while reusing the ESDF previously computed by an A* planner.");

  // --- Exceptions ---
  py::register_exception<constrained_smoother::InvalidPath>(m, "InvalidPathError");
  py::register_exception<constrained_smoother::FailedToSmoothPath>(m, "FailedToSmoothPathError");
}
