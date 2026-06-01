// Copyright (c) 2021, Samsung Research America
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

#ifndef SMAC_PLANNER__SMOOTHER_HPP_
#define SMAC_PLANNER__SMOOTHER_HPP_

#include <vector>

#include "my/smac_planner/types.hpp"
#include "my/smac_planner/constants.hpp"
#include "my/smac_planner/costmap_2d.hpp"
#include "ompl/base/StateSpace.h"

namespace smac_planner
{

struct BoundaryPoints
{
  BoundaryPoints(double & x_in, double & y_in, double & theta_in)
  : x(x_in), y(y_in), theta(theta_in)
  {}

  double x;
  double y;
  double theta;
};

struct BoundaryExpansion
{
  double path_end_idx{0.0};
  double expansion_path_length{0.0};
  double original_path_length{0.0};
  std::vector<BoundaryPoints> pts;
  bool in_collision{false};
};

typedef std::vector<BoundaryExpansion> BoundaryExpansions;

class Smoother
{
public:
  explicit Smoother(const SmootherParams & params);

  ~Smoother() {}

  void initialize(
    const double & min_turning_radius);

  bool smooth(
    Path & path,
    const Costmap2D * costmap,
    const double & max_time);

protected:
  bool smoothImpl(
    Path & path,
    bool & reversing_segment,
    const Costmap2D * costmap,
    const double & max_time);

  inline double getFieldByDim(
    const Pose & msg,
    const unsigned int & dim);

  inline void setFieldByDim(
    Pose & msg, const unsigned int dim,
    const double & value);

  void enforceStartBoundaryConditions(
    const Pose & start_pose,
    Path & path,
    const Costmap2D * costmap,
    const bool & reversing_segment);

  void enforceEndBoundaryConditions(
    const Pose & end_pose,
    Path & path,
    const Costmap2D * costmap,
    const bool & reversing_segment);

  unsigned int findShortestBoundaryExpansionIdx(const BoundaryExpansions & boundary_expansions);

  void findBoundaryExpansion(
    const Pose & start,
    const Pose & end,
    BoundaryExpansion & expansion,
    const Costmap2D * costmap);

  template<typename IteratorT>
  BoundaryExpansions generateBoundaryExpansionPoints(IteratorT start, IteratorT end);

  double min_turning_rad_, tolerance_, data_w_, smooth_w_;
  int max_its_, refinement_ctr_, refinement_num_;
  bool is_holonomic_, do_refinement_;
  MotionModel motion_model_;
  ompl::base::StateSpacePtr state_space_;
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__SMOOTHER_HPP_
