/**
 * nanobind module for Ceres 2D Path Smoother with ESDF.
 *
 * Exposes ESDFMap, SmootherParams, SmootherResult, and PathSmoother2D
 * to Python for easy integration and visualization.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "ceres_smoother_2d.hpp"
#include "astar.hpp"

namespace nb = nanobind;
using namespace ceres_smoother_2d;

NB_MODULE(ceres_smoother_2d, m)
{
  m.doc() = "Ceres-based 2D Path Smoother with ESDF obstacle avoidance";

  // ========================================================================
  // ESDFMap
  // ========================================================================
  nb::class_<ESDFMap>(m, "ESDFMap",
    "Euclidean Signed Distance Field map computed from an occupancy grid.\n"
    "Positive distance = free space, negative = inside obstacle.")
    .def(nb::init<const std::string &, double, double, double, int>(),
      nb::arg("path"),
      nb::arg("resolution"),
      nb::arg("origin_x") = 0.0,
      nb::arg("origin_y") = 0.0,
      nb::arg("obstacle_thresh") = 127,
      "Load occupancy grid from PNG and compute ESDF.\n\n"
      "Args:\n"
      "    path: Path to grayscale PNG (0=obstacle, 255=free).\n"
      "    resolution: Meters per pixel.\n"
      "    origin_x: World x of grid pixel (0,0).\n"
      "    origin_y: World y of grid pixel (0,0).\n"
      "    obstacle_thresh: Pixels <= this value are treated as obstacles.")
    .def(nb::init<const std::vector<uint8_t> &, int, int, double, double, double>(),
      nb::arg("occupancy"),
      nb::arg("width"),
      nb::arg("height"),
      nb::arg("resolution"),
      nb::arg("origin_x") = 0.0,
      nb::arg("origin_y") = 0.0,
      "Construct from raw occupancy data (0=free, 1=obstacle).")
    .def("get_distance", &ESDFMap::getDistance,
      nb::arg("wx"), nb::arg("wy"),
      "Get signed distance at world coordinate (bilinear interpolation).")
    .def("in_bounds", &ESDFMap::inBounds,
      nb::arg("wx"), nb::arg("wy"), nb::arg("margin") = 0.0,
      "Check if world point is inside map bounds.")
    .def("esdf_at_grid", &ESDFMap::esdfAtGrid,
      nb::arg("col"), nb::arg("row"),
      "Get ESDF value at fractional grid coordinates.")
    .def_prop_ro("width", &ESDFMap::width, "Grid width in pixels.")
    .def_prop_ro("height", &ESDFMap::height, "Grid height in pixels.")
    .def_prop_ro("resolution", &ESDFMap::resolution, "Meters per pixel.")
    .def_prop_ro("origin_x", &ESDFMap::originX, "World x of grid origin.")
    .def_prop_ro("origin_y", &ESDFMap::originY, "World y of grid origin.")
    .def_prop_ro("world_width", &ESDFMap::worldWidth, "Map width in meters.")
    .def_prop_ro("world_height", &ESDFMap::worldHeight, "Map height in meters.")
    .def("get_esdf_array", [](const ESDFMap & self) {
        const auto & grid = self.esdfGrid();
        return std::vector<double>(grid.begin(), grid.end());
      },
      "Get ESDF grid as flat array (row-major, height x width).")
    .def("get_occupancy_array", [](const ESDFMap & self) {
        const auto & grid = self.occupancyGrid();
        return std::vector<uint8_t>(grid.begin(), grid.end());
      },
      "Get occupancy grid as flat array (row-major, height x width).");

  // ========================================================================
  // SmootherParams
  // ========================================================================
  nb::class_<SmootherParams>(m, "SmootherParams",
    "Parameters for the Ceres-based 2D path smoother.")
    .def(nb::init<>())
    .def_rw("max_iterations", &SmootherParams::max_iterations,
      "Ceres solver max iterations.")
    .def_rw("max_time_seconds", &SmootherParams::max_time_seconds,
      "Ceres solver max time in seconds.")
    .def_rw("verbose", &SmootherParams::verbose,
      "Enable Ceres verbose output.")
    .def_rw("w_smooth", &SmootherParams::w_smooth,
      "Smoothness weight (jerk penalty).")
    .def_rw("w_curvature", &SmootherParams::w_curvature,
      "Curvature constraint weight.")
    .def_rw("min_turning_radius", &SmootherParams::min_turning_radius,
      "Minimum turning radius in meters.")
    .def_rw("w_reference", &SmootherParams::w_reference,
      "Reference path tracking weight.")
    .def_rw("w_length", &SmootherParams::w_length,
      "Elastic-band length weight (penalises Σ‖p_next-p_curr‖², i.e. sum of "
      "squared inter-point distances). Replaces the old target_spacing spring "
      "for faster convergence and no rest-length conflict with fixed start/goal.")
    .def_rw("w_obstacle", &SmootherParams::w_obstacle,
      "ESDF obstacle avoidance weight (soft hinge outside obstacles).")
    .def_rw("w_penetration", &SmootherParams::w_penetration,
      "ESDF penetration weight: penalizes points that are *inside* an "
      "obstacle (dist < 0). Default 0 (disabled) reproduces the old "
      "single-hinge behavior. Set > 0 to make wall-interior states "
      "strictly suboptimal and prevent the optimizer from stalling "
      "inside a wall.")
    .def_rw("safety_margin", &SmootherParams::safety_margin,
      "Minimum clearance from robot edge to obstacles in meters.")
    .def_rw("robot_radius", &SmootherParams::robot_radius,
      "Robot inscribed radius in meters. Effective clearance threshold = "
      "safety_margin + robot_radius.")
    .def_rw("target_spacing", &SmootherParams::target_spacing,
      "Desired inter-point spacing in meters (used by w_length and the "
      "optional post-processing resample).")
    .def_rw("resample_after_smooth", &SmootherParams::resample_after_smooth,
      "If true, uniformly resample the smoothed path along arc length "
      "so adjacent output points are ~target_spacing meters apart.")
    .def_rw("resample_before_smooth", &SmootherParams::resample_before_smooth,
      "If true, resample the input reference path to uniform spacing "
      "BEFORE optimization. Recommended when the upstream path (e.g. A*) "
      "has uneven point density.");

  // ========================================================================
  // SmootherResult
  // ========================================================================
  nb::class_<SmootherResult>(m, "SmootherResult",
    "Result of the path smoothing optimization.")
    .def(nb::init<>())
    .def_ro("success", &SmootherResult::success, "Whether optimization succeeded.")
    .def_ro("x", &SmootherResult::x, "Smoothed x coordinates.")
    .def_ro("y", &SmootherResult::y, "Smoothed y coordinates.")
    .def_ro("final_cost", &SmootherResult::final_cost, "Final optimization cost.")
    .def_ro("solve_time_ms", &SmootherResult::solve_time_ms, "Solve time in milliseconds.")
    .def_ro("iterations", &SmootherResult::iterations, "Number of iterations.")
    .def_ro("report", &SmootherResult::report, "Ceres solver report.");

  // ========================================================================
  // PathSmoother2D
  // ========================================================================
  nb::class_<PathSmoother2D>(m, "PathSmoother2D",
    "Ceres-based 2D path smoother with ESDF obstacle avoidance.")
    .def(nb::init<SmootherParams>(),
      nb::arg("params") = SmootherParams{},
      "Create smoother with given parameters.")
    .def("smooth", &PathSmoother2D::smooth,
      nb::arg("x"), nb::arg("y"), nb::arg("map"),
      "Smooth a 2D path using ESDF-based obstacle avoidance.\n\n"
      "Args:\n"
      "    x: Input path x coordinates.\n"
      "    y: Input path y coordinates.\n"
      "    map: ESDF map for obstacle queries.\n\n"
      "Returns:\n"
      "    SmootherResult with smoothed path and diagnostics.");

  // ========================================================================
  // resamplePathByArcLength (free function)
  // ========================================================================
  m.def("resample_path_by_arc_length",
    [](const std::vector<double> & xs, const std::vector<double> & ys,
       double target_spacing)
    -> std::pair<std::vector<double>, std::vector<double>> {
      std::vector<double> rx, ry;
      resamplePathByArcLength(xs, ys, target_spacing, rx, ry);
      return {std::move(rx), std::move(ry)};
    },
    nb::arg("x"), nb::arg("y"), nb::arg("target_spacing"),
    "Uniformly resample a polyline along its arc length.\n\n"
    "Args:\n"
    "    x, y: Input polyline coordinates (N >= 2).\n"
    "    target_spacing: Desired average inter-point distance in meters.\n\n"
    "Returns:\n"
    "    Tuple (xs, ys) of the resampled polyline. Endpoints are preserved\n"
    "    exactly; intermediate points are linearly interpolated within each\n"
    "    input segment. Output count is max(2, round(L/target)+1).");

  // ========================================================================
  // A* (C++ implementation, ~100x faster than Python on 1k-class grids)
  // ========================================================================
  nb::class_<AStarResult>(m, "AStarResult",
    "Result of A* pathfinding on the ESDFMap's occupancy grid.")
    .def(nb::init<>())
    .def_ro("success", &AStarResult::success, "Whether a path was found.")
    .def_ro("x", &AStarResult::x, "Path x coordinates (world meters).")
    .def_ro("y", &AStarResult::y, "Path y coordinates (world meters).")
    .def_ro("expansions", &AStarResult::expansions, "Number of nodes popped + expanded.")
    .def_ro("time_ms", &AStarResult::time_ms, "Wall-clock search time in milliseconds.");

  m.def("astar_solve", &astarSolve,
    nb::arg("map"),
    nb::arg("sx"), nb::arg("sy"),
    nb::arg("gx"), nb::arg("gy"),
    nb::arg("robot_radius") = 0.0,
    "A* shortest path on the ESDFMap's occupancy grid.\n\n"
    "8-connected, Euclidean step cost (1 cardinal, sqrt(2) diagonal),\n"
    "Euclidean distance heuristic. ~100x faster than the Python fallback.\n\n"
    "Args:\n"
    "    map: ESDFMap (occupancy grid is read internally).\n"
    "    sx, sy: start point in world coordinates (meters).\n"
    "    gx, gy: goal point in world coordinates (meters).\n"
    "    robot_radius: circular robot radius in meters. Obstacles are\n"
    "        pre-inflated by this amount (any cell with ESDF distance\n"
    "        below robot_radius is treated as an obstacle), so the\n"
    "        returned path is feasible for a robot of that radius.\n"
    "        0 (default) = no inflation.\n\n"
    "Returns:\n"
    "    AStarResult with success flag, world-coord path, expansion count,\n"
    "    and solve time. Returns success=false if start/goal is in an\n"
    "    (inflated) obstacle or no path exists.");
}
