#ifndef SMOOTHER_CLOTHOID__OPTIONS_HPP_
#define SMOOTHER_CLOTHOID__OPTIONS_HPP_

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace smoother_clothoid
{

struct SmootherParams
{
  bool obstacleTermsEnabled() const
  {
    return std::max(costmap_weight_sqrt, cusp_costmap_weight_sqrt) > 1e-9;
  }

  // --- Weights ---
  double model_weight_sqrt{0.0};
  double costmap_weight_sqrt{0.0};
  double cusp_costmap_weight_sqrt{0.0};
  double cusp_zone_length{0.0};
  double reference_path_weight_sqrt{0.0};
  double reference_point_max_deviation_m{0.0};
  double kinematic_curvature_weight_sqrt{0.0};
  double kinematic_curvature_rate_weight_sqrt{0.0};
  double kinematic_spacing_weight_sqrt{1.0};
  double kinematic_max_spacing{0.0};
  double path_length_weight_sqrt{0.0};
  double fix_weight{100.0};
  double max_curvature{0.0};
  double max_time{10.0};

  // --- Obstacle handling ---
  bool use_exact_esdf{true};
  double obstacle_safe_distance{0.5};
  double cost_check_radius{0.0};
  std::vector<double> cost_check_points{};

  // --- Path resampling ---
  int path_downsampling_factor{1};
  int path_upsampling_factor{1};
  bool reversing_enabled{true};

  // --- Boundary handling ---
  double goal_longitudinal_tolerance{0.0};
  double goal_lateral_tolerance{0.0};
  double goal_orientation_tolerance{0.0};
  bool keep_goal_orientation{true};
  bool keep_start_orientation{true};
};

struct OptimizerParams
{
  enum class LinearSolver
  {
    DenseQr,
    SparseNormalCholesky,
  };

  static const char * linearSolverToString(LinearSolver solver)
  {
    switch (solver) {
      case LinearSolver::DenseQr: return "DENSE_QR";
      case LinearSolver::SparseNormalCholesky: return "SPARSE_NORMAL_CHOLESKY";
    }
    return "SPARSE_NORMAL_CHOLESKY";
  }

  static LinearSolver linearSolverFromString(const std::string & name)
  {
    if (name == "DENSE_QR") return LinearSolver::DenseQr;
    if (name == "SPARSE_NORMAL_CHOLESKY") return LinearSolver::SparseNormalCholesky;
    throw std::invalid_argument("Unsupported linear_solver_type: " + name);
  }

  bool debug{false};
  LinearSolver linear_solver{LinearSolver::SparseNormalCholesky};
  int max_iterations{50};
  double parameter_tolerance{1e-8};
  double function_tolerance{1e-6};
  double gradient_tolerance{1e-10};
};

}  // namespace smoother_clothoid

#endif  // SMOOTHER_CLOTHOID__OPTIONS_HPP_
