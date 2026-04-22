// Copyright (c) 2021 RoboTech Vision
// Copyright (c) 2020, Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. Reserved.

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_HPP_

#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <queue>
#include <utility>
#include <deque>
#include <limits>
#include <algorithm>

#include "constrained_smoother/smoother_cost_function.hpp"
#include "constrained_smoother/utils.hpp"
#include "constrained_smoother/exceptions.hpp"

#include "ceres/ceres.h"
#include "Eigen/Core"

namespace constrained_smoother
{

/**
 * @class constrained_smoother::Smoother
 * @brief A Ceres-based constrained path smoother, independent of ROS.
 */
class Smoother
{
public:
  Smoother() {}
  ~Smoother() {}

  size_t getLastOptimizedKnotCount() const
  {
    return last_optimized_knot_count_;
  }

  /**
   * @brief Initialization of the smoother
   * @param params OptimizerParam struct
   */
  void initialize(const OptimizerParams params)
  {
    debug_ = params.debug;

    options_.linear_solver_type = params.solver_types.at(params.linear_solver_type);

    options_.max_num_iterations = params.max_iterations;

    options_.function_tolerance = params.fn_tol;
    options_.gradient_tolerance = params.gradient_tol;
    options_.parameter_tolerance = params.param_tol;

    if (debug_) {
      options_.minimizer_progress_to_stdout = true;
      options_.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
    } else {
      options_.logging_type = ceres::SILENT;
    }
  }

  /**
   * @brief Smoother method
    * @param path Reference to path. On input it stores (x, y, direction_sign);
    * on output the third component is overwritten with yaw in radians.
   * @param start_dir Orientation of the first pose
   * @param end_dir Orientation of the last pose
   * @param costmap Pointer to Costmap2D
   * @param params Smoother parameters
   * @return If smoothing was successful
   */
  bool smooth(
    std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const Costmap2D * costmap,
    const SmootherParams & params)
  {
    return smooth(path, start_dir, end_dir, costmap, params, nullptr);
  }

  bool smooth(
    std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
    // Path has always at least 2 points
    if (path.size() < 2) {
      throw InvalidPath("Constrained smoother: Path must have at least 2 points");
    }

    options_.max_solver_time_in_seconds = params.max_time;

    ceres::Problem problem;
    std::vector<Eigen::Vector3d> path_optim;
    std::vector<bool> optimized;
    if (buildProblem(
      path, start_dir, end_dir, costmap, params, precomputed_esdf, problem, path_optim,
      optimized))
    {
      last_optimized_knot_count_ = std::count(optimized.begin(), optimized.end(), true);
      // solve the problem
      ceres::Solver::Summary summary;
      ceres::Solve(options_, &problem, &summary);
      if (debug_) {
        std::cout << summary.FullReport() << std::endl;
      }
      if (!summary.IsSolutionUsable() || summary.initial_cost - summary.final_cost < 0.0) {
        throw FailedToSmoothPath("Solution is not usable");
      }
    } else {
      last_optimized_knot_count_ = std::count(optimized.begin(), optimized.end(), true);
      if (debug_) {
        std::cout << "[smoother] Path too short to optimize" << std::endl;
      }
    }

    upsampleAndPopulate(path_optim, optimized, start_dir, end_dir, params, path);

    return true;
  }

private:
  using EsdfInterpolator = ceres::BiCubicInterpolator<ceres::Grid2D<double>>;

  struct BuildProblemState
  {
    explicit BuildProblemState(double initial_direction)
    : last_direction(initial_direction)
    {
    }

    int preprelast_i{-1};
    int prelast_i{-1};
    int last_i{0};
    double last_direction;
    bool last_was_cusp{false};
    bool last_is_reversing{false};
    std::deque<std::pair<double, SmootherCostFunction *>> potential_cusp_funcs{};
    double last_segment_len{EPSILON};
    double potential_cusp_funcs_len{0.0};
    double len_since_cusp{std::numeric_limits<double>::infinity()};
  };

  static double interpolateCuspZoneWeight(
    double distance_from_cusp,
    double cusp_half_length,
    const SmootherParams & params)
  {
    return params.cusp_costmap_weight_sqrt * (1.0 - distance_from_cusp / cusp_half_length) +
           params.costmap_weight_sqrt * distance_from_cusp / cusp_half_length;
  }

  std::shared_ptr<EsdfInterpolator> initializeEsdfInterpolator(
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf)
  {
    // Build or validate the ESDF backing the obstacle residuals.
    const size_t expected_esdf_size =
      static_cast<size_t>(costmap->getSizeInCellsX()) * costmap->getSizeInCellsY();
    if (precomputed_esdf != nullptr) {
      if (precomputed_esdf->size() != expected_esdf_size) {
        throw std::runtime_error("Precomputed ESDF size does not match costmap dimensions");
      }
      esdf_values_ = *precomputed_esdf;
    } else {
      esdf_values_ = ESDF::ComputeESDF(
        costmap,
        Costmap2D::LETHAL_OBSTACLE,
        params.use_exact_esdf ? ESDFAlgorithm::Exact : ESDFAlgorithm::Approximate);
    }

    esdf_grid_ = std::make_shared<ceres::Grid2D<double>>(
      esdf_values_.data(), 0, costmap->getSizeInCellsY(), 0, costmap->getSizeInCellsX());
    return std::make_shared<EsdfInterpolator>(*esdf_grid_);
  }

  void initializeOptimizationPath(
    const std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized) const
  {
    path_optim = path;
    applyEndpointOrientationAnchors(path_optim, start_dir, end_dir, params);
    optimized = std::vector<bool>(path.size(), false);
    optimized[0] = true;
  }

  void addPathResidualBlocks(
    const std::vector<Eigen::Vector3d> & path,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::shared_ptr<EsdfInterpolator> & esdf_interpolator,
    ceres::Problem & problem,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized) const
  {
    // Walk the path once to downsample, detect cusps, and wire the local residuals.
    const double cusp_half_length = params.cusp_zone_length / 2;
    ceres::LossFunction * loss_function = nullptr;
    BuildProblemState state(path_optim[0][2]);

    for (size_t i = 1; i < path_optim.size(); i++) {
      auto & pt = path_optim[i];

      // A cusp is a direction-sign flip between consecutive path points.
      // Those points must be preserved, so they bypass the normal downsampling skip.
      bool is_cusp = false;
      if (i != path_optim.size() - 1) {
        is_cusp = pt[2] * state.last_direction < 0;
        state.last_direction = pt[2];

        if (!is_cusp &&
          i > (params.keep_start_orientation ? 1 : 0) &&
          i < path_optim.size() - (params.keep_goal_orientation ? 2 : 1) &&
          static_cast<int>(i - state.last_i) < params.path_downsampling_factor)
        {
          continue;
        }
      }

      double current_segment_len =
        (path_optim[i] - path_optim[state.last_i]).block<2, 1>(0, 0).norm();

      // Retain only recent cost blocks that can still fall inside the backward half
      // of a cusp zone. Older ones can no longer be reweighted by a future cusp.
      state.potential_cusp_funcs_len += current_segment_len;
      while (!state.potential_cusp_funcs.empty() &&
        state.potential_cusp_funcs_len > cusp_half_length)
      {
        state.potential_cusp_funcs_len -= state.potential_cusp_funcs.front().first;
        state.potential_cusp_funcs.pop_front();
      }

      // When a cusp is found, retroactively raise the obstacle weight on the nearby
      // preceding blocks, then start the forward-side ramp from zero arc length.
      if (is_cusp) {
        double len_to_cusp = current_segment_len;
        for (int i_cusp = state.potential_cusp_funcs.size() - 1; i_cusp >= 0; i_cusp--) {
          auto & f = state.potential_cusp_funcs[i_cusp];
          double new_weight = interpolateCuspZoneWeight(len_to_cusp, cusp_half_length, params);
          if (std::abs(new_weight - params.cusp_costmap_weight_sqrt) <
            std::abs(f.second->getCostmapWeightSqrt() - params.cusp_costmap_weight_sqrt))
          {
            f.second->setCostmapWeightSqrt(new_weight);
          }
          len_to_cusp += f.first;
        }
        state.potential_cusp_funcs_len = 0;
        state.potential_cusp_funcs.clear();
        state.len_since_cusp = 0;
      }

      // Register the main 3-point smoothing/cost block once we have a predecessor,
      // and optionally add the 4-point D3 curvature-rate proxy on same-direction runs.
      optimized[i] = true;
      if (state.prelast_i != -1) {
        double costmap_weight_sqrt = params.costmap_weight_sqrt;
        if (state.len_since_cusp <= cusp_half_length) {
          costmap_weight_sqrt =
            interpolateCuspZoneWeight(state.len_since_cusp, cusp_half_length, params);
        }

        SmootherCostFunction * cost_function = new SmootherCostFunction(
          path[state.last_i].template block<2, 1>(0, 0),
          (state.last_was_cusp ? -1 : 1) * state.last_segment_len / current_segment_len,
          state.last_is_reversing,
          costmap,
          esdf_interpolator,
          params,
          costmap_weight_sqrt);
        problem.AddResidualBlock(
          cost_function->AutoDiff(), loss_function,
          path_optim[state.last_i].data(), pt.data(), path_optim[state.prelast_i].data());

        if (params.curvature_rate_weight_sqrt > 0.0 &&
          state.preprelast_i != -1 &&
          path_optim[state.preprelast_i][2] * path_optim[state.prelast_i][2] > 0.0 &&
          path_optim[state.prelast_i][2] * path_optim[state.last_i][2] > 0.0 &&
          path_optim[state.last_i][2] * pt[2] > 0.0)
        {
          // Do not connect this D3 term across a cusp. The finite-difference proxy
          // is only meaningful when all three consecutive segments share one direction.
          CurvatureRateCostFunction * curvature_rate_cost_function =
            new CurvatureRateCostFunction(params.curvature_rate_weight_sqrt);
          problem.AddResidualBlock(
            curvature_rate_cost_function->AutoDiff(), loss_function,
            path_optim[state.preprelast_i].data(), path_optim[state.prelast_i].data(),
            path_optim[state.last_i].data(), pt.data());
        }

        state.potential_cusp_funcs.emplace_back(current_segment_len, cost_function);
      }

      state.last_was_cusp = is_cusp;
      state.last_is_reversing = state.last_direction < 0;
      state.preprelast_i = state.prelast_i;
      state.prelast_i = state.last_i;
      state.last_i = i;
      state.len_since_cusp += current_segment_len;
      state.last_segment_len = std::max(EPSILON, current_segment_len);
    }
  }

  bool finalizeOptimizationProblem(
    ceres::Problem & problem,
    const std::vector<Eigen::Vector3d> & path_optim,
    const SmootherParams & params) const
  {
    // If every interior point was skipped or fixed, there is no optimization problem to solve.
    int posesToOptimize = problem.NumParameterBlocks() - 2;  // minus start and goal
    if (params.keep_goal_orientation) {
      posesToOptimize -= 1;
    }
    if (params.keep_start_orientation) {
      posesToOptimize -= 1;
    }
    if (posesToOptimize <= 0) {
      return false;  // nothing to optimize
    }

    // Freeze the endpoint anchors after all residuals are wired.
    problem.SetParameterBlockConstant(path_optim.front().data());
    if (params.keep_start_orientation) {
      problem.SetParameterBlockConstant(path_optim[1].data());
    }
    if (params.keep_goal_orientation) {
      problem.SetParameterBlockConstant(path_optim[path_optim.size() - 2].data());
    }
    problem.SetParameterBlockConstant(path_optim.back().data());
    return true;
  }

  /**
   * @brief Build problem method
   */
  bool buildProblem(
    const std::vector<Eigen::Vector3d> & path,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const Costmap2D * costmap,
    const SmootherParams & params,
    const std::vector<double> * precomputed_esdf,
    ceres::Problem & problem,
    std::vector<Eigen::Vector3d> & path_optim,
    std::vector<bool> & optimized)
  {
    auto esdf_interpolator = initializeEsdfInterpolator(costmap, params, precomputed_esdf);
    initializeOptimizationPath(path, start_dir, end_dir, params, path_optim, optimized);
    addPathResidualBlocks(path, costmap, params, esdf_interpolator, problem, path_optim, optimized);
    return finalizeOptimizationProblem(problem, path_optim, params);
  }

  void applyEndpointOrientationAnchors(
    std::vector<Eigen::Vector3d> & path_optim,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params) const
  {
    if (path_optim.size() < 3) {
      return;
    }

    const auto normalized_dir = [](const Eigen::Vector2d & dir) -> Eigen::Vector2d {
        const double norm = dir.norm();
        if (norm <= EPSILON) {
          return Eigen::Vector2d(1.0, 0.0);
        }
        return Eigen::Vector2d(dir / norm);
      };

    if (params.keep_start_orientation) {
      const double start_segment_len =
        (path_optim[1] - path_optim[0]).template block<2, 1>(0, 0).norm();
      path_optim[1].template block<2, 1>(0, 0) =
        path_optim[0].template block<2, 1>(0, 0) + normalized_dir(start_dir) * start_segment_len;
    }

    if (params.keep_goal_orientation) {
      const size_t goal_index = path_optim.size() - 1;
      const size_t pregoal_index = goal_index - 1;
      const double goal_segment_len =
        (path_optim[goal_index] - path_optim[pregoal_index]).template block<2, 1>(0, 0).norm();
      Eigen::Vector2d anchored_pregoal =
        path_optim[goal_index].template block<2, 1>(0, 0) - normalized_dir(end_dir) * goal_segment_len;

      if (params.keep_start_orientation && pregoal_index == 1) {
        path_optim[pregoal_index].template block<2, 1>(0, 0) =
          0.5 * (path_optim[pregoal_index].template block<2, 1>(0, 0) + anchored_pregoal);
      } else {
        path_optim[pregoal_index].template block<2, 1>(0, 0) = anchored_pregoal;
      }
    }
  }

  /**
   * @brief Populate optimized points to path, assigning orientations and
   *        upsampling poses using cubic bezier
   */
  void upsampleAndPopulate(
    const std::vector<Eigen::Vector3d> & path_optim,
    const std::vector<bool> & optimized,
    const Eigen::Vector2d & start_dir,
    const Eigen::Vector2d & end_dir,
    const SmootherParams & params,
    std::vector<Eigen::Vector3d> & path)
  {
    path.clear();
    if (params.path_upsampling_factor > 1) {
      path.reserve(params.path_upsampling_factor * (path_optim.size() - 1) + 1);
    }
    int last_i = 0;
    int prelast_i = -1;
    Eigen::Vector2d prelast_dir = {0, 0};
    for (int i = 1; i <= static_cast<int>(path_optim.size()); i++) {
      if (i == static_cast<int>(path_optim.size()) || optimized[i]) {
        if (prelast_i != -1) {
          Eigen::Vector2d last_dir;
          auto & prelast = path_optim[prelast_i];
          auto & last = path_optim[last_i];

          // Compute orientation of last
          if (i < static_cast<int>(path_optim.size())) {
            auto & current = path_optim[i];
            Eigen::Vector2d tangent_dir_val = tangentDir<double>(
              prelast.block<2, 1>(0, 0),
              last.block<2, 1>(0, 0),
              current.block<2, 1>(0, 0),
              prelast[2] * last[2] < 0);

            last_dir =
              tangent_dir_val.dot((current - last).block<2, 1>(0, 0) * last[2]) >= 0 ?
              tangent_dir_val :
              -tangent_dir_val;
            last_dir.normalize();
          } else if (params.keep_goal_orientation) {
            last_dir = end_dir;
          } else {
            last_dir = (last - prelast).block<2, 1>(0, 0) * last[2];
            last_dir.normalize();
          }
          double last_angle = atan2(last_dir[1], last_dir[0]);

          // Interpolate poses between prelast and last
          int interp_cnt = (last_i - prelast_i) * params.path_upsampling_factor - 1;
          if (interp_cnt > 0) {
            Eigen::Vector2d last_pt = last.block<2, 1>(0, 0);
            Eigen::Vector2d prelast_pt = prelast.block<2, 1>(0, 0);
            double dist = (last_pt - prelast_pt).norm();
            Eigen::Vector2d pt1 = prelast_pt + prelast_dir * dist * 0.4 * prelast[2];
            Eigen::Vector2d pt2 = last_pt - last_dir * dist * 0.4 * prelast[2];
            for (int j = 1; j <= interp_cnt; j++) {
              double interp = j / static_cast<double>(interp_cnt + 1);
              Eigen::Vector2d pt = cubicBezier(prelast_pt, pt1, pt2, last_pt, interp);
              path.emplace_back(pt[0], pt[1], 0.0);
            }
          }
          path.emplace_back(last[0], last[1], last_angle);

          // Assign orientations to interpolated points
          for (size_t j = path.size() - 1 - interp_cnt; j < path.size() - 1; j++) {
            Eigen::Vector2d tangent_dir_val = tangentDir<double>(
              path[j - 1].block<2, 1>(0, 0),
              path[j].block<2, 1>(0, 0),
              path[j + 1].block<2, 1>(0, 0),
              false);
            tangent_dir_val =
              tangent_dir_val.dot(
              (path[j + 1] - path[j]).block<2, 1>(0, 0) * prelast[2]) >= 0 ?
              tangent_dir_val :
              -tangent_dir_val;
            path[j][2] = atan2(tangent_dir_val[1], tangent_dir_val[0]);
          }

          prelast_dir = last_dir;
        } else {  // start pose
          auto & start = path_optim[0];
          Eigen::Vector2d dir = params.keep_start_orientation ?
            start_dir :
            ((path_optim[i] - start).block<2, 1>(0, 0) * start[2]).normalized();
          path.emplace_back(start[0], start[1], atan2(dir[1], dir[0]));
          prelast_dir = dir;
        }
        prelast_i = last_i;
        last_i = i;
      }
    }
  }

  /*
    Piecewise cubic bezier curve as defined by Adobe in Postscript
    The two end points are pt0 and pt3
    Their associated control points are pt1 and pt2
  */
  static Eigen::Vector2d cubicBezier(
    Eigen::Vector2d & pt0, Eigen::Vector2d & pt1,
    Eigen::Vector2d & pt2, Eigen::Vector2d & pt3, double mu)
  {
    Eigen::Vector2d a, b, c, pt;

    c[0] = 3 * (pt1[0] - pt0[0]);
    c[1] = 3 * (pt1[1] - pt0[1]);
    b[0] = 3 * (pt2[0] - pt1[0]) - c[0];
    b[1] = 3 * (pt2[1] - pt1[1]) - c[1];
    a[0] = pt3[0] - pt0[0] - c[0] - b[0];
    a[1] = pt3[1] - pt0[1] - c[1] - b[1];

    pt[0] = a[0] * mu * mu * mu + b[0] * mu * mu + c[0] * mu + pt0[0];
    pt[1] = a[1] * mu * mu * mu + b[1] * mu * mu + c[1] * mu + pt0[1];

    return pt;
  }

  bool debug_;
  ceres::Solver::Options options_;
  std::vector<double> esdf_values_;
  std::shared_ptr<ceres::Grid2D<double>> esdf_grid_;
  size_t last_optimized_knot_count_{0};
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_HPP_
