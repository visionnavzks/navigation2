#include <ompl/base/ScopedState.h>

#include <chrono>
#include <memory>
#include <vector>
#include <cmath>

#include "my/hybrid_astar/smoother.hpp"

#define SMAC_DEBUG(fmt, ...) fprintf(stderr, "[hybrid_astar] " fmt "\n", ##__VA_ARGS__)

namespace hybrid_astar
{
using namespace std::chrono;  // NOLINT

struct PathSegment { int start; int end; };

inline std::vector<PathSegment> findDirectionalPathSegments(
  const Path & path, bool is_holonomic)
{
  std::vector<PathSegment> segments;
  if (path.empty()) return segments;
  int seg_start = 0;
  for (size_t i = 1; i < path.size(); ++i) {
    double dy = path[i].y - path[i-1].y;
    double dx = path[i].x - path[i-1].x;
    if (!is_holonomic && (std::abs(dx) > 1e-6 || std::abs(dy) > 1e-6)) {
      double angle = std::atan2(dy, dx);
      double prev_angle = (i > 1) ? std::atan2(
        path[i-1].y - path[i-2].y, path[i-1].x - path[i-2].x) : angle;
      double angle_diff = std::fmod(std::abs(angle - prev_angle), 2.0 * M_PI);
      if (angle_diff > M_PI) angle_diff = 2.0 * M_PI - angle_diff;
      if (angle_diff > M_PI_2) {
        segments.push_back({seg_start, static_cast<int>(i - 1)});
        seg_start = i - 1;
      }
    }
  }
  segments.push_back({seg_start, static_cast<int>(path.size() - 1)});
  return segments;
}

inline void updateApproximatePathOrientations(Path & path, bool is_holonomic)
{
  if (path.size() < 2) return;
  for (size_t i = 0; i < path.size() - 1; ++i) {
    double dx = path[i + 1].x - path[i].x;
    double dy = path[i + 1].y - path[i].y;
    if (std::abs(dx) > 1e-6 || std::abs(dy) > 1e-6) {
      path[i].theta = std::atan2(dy, dx);
    }
  }
  if (path.size() > 1) path.back().theta = path[path.size() - 2].theta;
}

Smoother::Smoother(const SmootherParams & params)
{
  tolerance_ = params.tolerance_;
  max_its_ = params.max_its_;
  data_w_ = params.w_data_;
  smooth_w_ = params.w_smooth_;
  is_holonomic_ = params.holonomic_;
  do_refinement_ = params.do_refinement_;
  refinement_num_ = params.refinement_num_;
}

void Smoother::initialize(const double & min_turning_radius)
{
  min_turning_rad_ = min_turning_radius;
  state_space_ = createStateSpace(MotionModel::DUBIN, min_turning_rad_);
}

bool Smoother::smooth(
  Path & path,
  const Costmap2D * costmap,
  const double & max_time)
{
  if (max_its_ == 0) {
    return false;
  }

  steady_clock::time_point start = steady_clock::now();
  double time_remaining = max_time;
  bool success = true, reversing_segment = false;
  Path curr_path_segment;
  std::vector<PathSegment> path_segments = findDirectionalPathSegments(
    path,
    is_holonomic_);

  for (unsigned int i = 0; i != path_segments.size(); i++) {
    if (path_segments[i].end - path_segments[i].start > 10) {
      curr_path_segment.clear();
      std::copy(
        path.begin() + path_segments[i].start,
        path.begin() + path_segments[i].end + 1,
        std::back_inserter(curr_path_segment));

      steady_clock::time_point now = steady_clock::now();
      time_remaining = max_time - duration_cast<duration<double>>(now - start).count();
      refinement_ctr_ = 0;

      const Pose start_pose = curr_path_segment.front();
      const Pose goal_pose = curr_path_segment.back();
      bool local_success =
        smoothImpl(curr_path_segment, reversing_segment, costmap, time_remaining);
      success = success && local_success;

      if (!is_holonomic_ && local_success) {
        enforceStartBoundaryConditions(start_pose, curr_path_segment, costmap, reversing_segment);
        enforceEndBoundaryConditions(goal_pose, curr_path_segment, costmap, reversing_segment);
      }

      std::copy(
        curr_path_segment.begin(),
        curr_path_segment.end(),
        path.begin() + path_segments[i].start);
    }
  }

  return success;
}

bool Smoother::smoothImpl(
  Path & path,
  bool & reversing_segment,
  const Costmap2D * costmap,
  const double & max_time)
{
  steady_clock::time_point a = steady_clock::now();
  double max_dur = max_time;

  int its = 0;
  double change = tolerance_;
  const unsigned int & path_size = path.size();
  double x_i, y_i, y_m1, y_ip1, y_i_org;
  unsigned int mx, my;

  Path new_path = path;
  Path last_path = path;

  while (change >= tolerance_) {
    its += 1;
    change = 0.0;

    if (its >= max_its_) {
      SMAC_DEBUG(
        "Number of iterations has exceeded limit of %i.", max_its_);
      path = last_path;
      updateApproximatePathOrientations(path, is_holonomic_);
      return false;
    }

    steady_clock::time_point b = steady_clock::now();
    double timespan = duration_cast<duration<double>>(b - a).count();
    if (timespan > max_dur) {
      SMAC_DEBUG(
        "Smoothing time exceeded allowed duration of %0.2f.", max_time);
      path = last_path;
      updateApproximatePathOrientations(path, is_holonomic_);
      return false;
    }

    for (unsigned int i = 1; i != path_size - 1; i++) {
      for (unsigned int j = 0; j != 2; j++) {
        x_i = getFieldByDim(path[i], j);
        y_i = getFieldByDim(new_path[i], j);
        y_m1 = getFieldByDim(new_path[i - 1], j);
        y_ip1 = getFieldByDim(new_path[i + 1], j);
        y_i_org = y_i;

        y_i += data_w_ * (x_i - y_i) + smooth_w_ * (y_ip1 + y_m1 - (2.0 * y_i));
        setFieldByDim(new_path[i], j, y_i);
        change += std::fabs(y_i - y_i_org);
      }

      float cost = 0.0;
      if (costmap) {
        costmap->worldToMap(
          getFieldByDim(new_path[i], 0),
          getFieldByDim(new_path[i], 1),
          mx, my);
        cost = static_cast<float>(costmap->getCost(mx, my));
      }

      if (cost > MAX_NON_OBSTACLE_COST && cost != UNKNOWN_COST) {
        SMAC_DEBUG(
          "Smoothing process resulted in an infeasible collision. "
          "Returning the last path before the infeasibility was introduced.");
        path = last_path;
        updateApproximatePathOrientations(path, is_holonomic_);
        return false;
      }
    }

    last_path = new_path;
  }

  if (do_refinement_ && refinement_ctr_ < refinement_num_) {
    refinement_ctr_++;
    smoothImpl(new_path, reversing_segment, costmap, max_time);
  }

  updateApproximatePathOrientations(new_path, is_holonomic_);
  path = new_path;
  return true;
}

double Smoother::getFieldByDim(
  const Pose & msg, const unsigned int & dim) const
{
  if (dim == 0) {
    return msg.x;
  } else if (dim == 1) {
    return msg.y;
  } else {
    return 0.0;
  }
}

void Smoother::setFieldByDim(
  Pose & msg, const unsigned int dim,
  const double & value)
{
  if (dim == 0) {
    msg.x = value;
  } else if (dim == 1) {
    msg.y = value;
  }
}

unsigned int Smoother::findShortestBoundaryExpansionIdx(
  const BoundaryExpansions & boundary_expansions)
{
  double min_length = 1e9;
  int shortest_boundary_expansion_idx = 1e9;
  for (unsigned int idx = 0; idx != boundary_expansions.size(); idx++) {
    if (boundary_expansions[idx].expansion_path_length<min_length &&
      !boundary_expansions[idx].in_collision &&
      boundary_expansions[idx].path_end_idx>0.0 &&
      boundary_expansions[idx].expansion_path_length > 0.0)
    {
      min_length = boundary_expansions[idx].expansion_path_length;
      shortest_boundary_expansion_idx = idx;
    }
  }

  return shortest_boundary_expansion_idx;
}

void Smoother::findBoundaryExpansion(
  const Pose & start,
  const Pose & end,
  BoundaryExpansion & expansion,
  const Costmap2D * costmap)
{
  ompl::base::ScopedState<> from(state_space_), to(state_space_), s(state_space_);

  from[0] = start.x;
  from[1] = start.y;
  from[2] = start.theta;
  to[0] = end.x;
  to[1] = end.y;
  to[2] = end.theta;

  double d = state_space_->distance(from(), to());
  if (d > 2.0 * expansion.original_path_length) {
    return;
  }

  std::vector<double> reals;
  double theta(0.0), x(0.0), y(0.0);
  double x_m = start.x;
  double y_m = start.y;

  for (double i = 0; i <= expansion.path_end_idx; i++) {
    state_space_->interpolate(from(), to(), i / expansion.path_end_idx, s());
    reals = s.reals();
    theta = (reals[2] < 0.0) ? (reals[2] + 2.0 * M_PI) : reals[2];
    theta = (theta > 2.0 * M_PI) ? (theta - 2.0 * M_PI) : theta;
    x = reals[0];
    y = reals[1];

    unsigned int mx, my;
    costmap->worldToMap(x, y, mx, my);
    if (static_cast<float>(costmap->getCost(mx, my)) >= INSCRIBED_COST) {
      expansion.in_collision = true;
    }

    expansion.expansion_path_length += hypot(x - x_m, y - y_m);
    x_m = x;
    y_m = y;

    expansion.pts.emplace_back(x, y, theta);
  }
}

template<typename IteratorT>
BoundaryExpansions Smoother::generateBoundaryExpansionPoints(IteratorT start, IteratorT end)
{
  std::vector<double> distances = {
    min_turning_rad_,
    2.0 * min_turning_rad_,
    M_PI * min_turning_rad_,
    2.0 * M_PI * min_turning_rad_
  };

  BoundaryExpansions boundary_expansions;
  boundary_expansions.resize(distances.size());
  double curr_dist = 0.0;
  double x_last = start->x;
  double y_last = start->y;
  unsigned int curr_dist_idx = 0;

  for (IteratorT iter = start; iter != end; iter++) {
    curr_dist += hypot(iter->x - x_last, iter->y - y_last);
    x_last = iter->x;
    y_last = iter->y;

    if (curr_dist >= distances[curr_dist_idx]) {
      boundary_expansions[curr_dist_idx].path_end_idx = iter - start;
      boundary_expansions[curr_dist_idx].original_path_length = curr_dist;
      curr_dist_idx++;
    }

    if (curr_dist_idx == boundary_expansions.size()) {
      break;
    }
  }

  return boundary_expansions;
}

void Smoother::enforceStartBoundaryConditions(
  const Pose & start_pose,
  Path & path,
  const Costmap2D * costmap,
  const bool & reversing_segment)
{
  BoundaryExpansions boundary_expansions =
    generateBoundaryExpansionPoints<Path::iterator>(path.begin(), path.end());

  for (unsigned int i = 0; i != boundary_expansions.size(); i++) {
    BoundaryExpansion & expansion = boundary_expansions[i];
    if (expansion.path_end_idx == 0.0) {
      continue;
    }

    if (!reversing_segment) {
      findBoundaryExpansion(
        start_pose, path[expansion.path_end_idx], expansion,
        costmap);
    } else {
      findBoundaryExpansion(
        path[expansion.path_end_idx], start_pose, expansion,
        costmap);
    }
  }

  unsigned int best_expansion_idx = findShortestBoundaryExpansionIdx(boundary_expansions);
  if (best_expansion_idx >= boundary_expansions.size()) {
    return;
  }

  BoundaryExpansion & best_expansion = boundary_expansions[best_expansion_idx];
  if (reversing_segment) {
    std::reverse(best_expansion.pts.begin(), best_expansion.pts.end());
  }
  for (unsigned int i = 0; i != best_expansion.pts.size(); i++) {
    path[i].x = best_expansion.pts[i].x;
    path[i].y = best_expansion.pts[i].y;
    path[i].theta = best_expansion.pts[i].theta;
  }
}

void Smoother::enforceEndBoundaryConditions(
  const Pose & end_pose,
  Path & path,
  const Costmap2D * costmap,
  const bool & reversing_segment)
{
  BoundaryExpansions boundary_expansions =
    generateBoundaryExpansionPoints<Path::reverse_iterator>(path.rbegin(), path.rend());

  unsigned int expansion_starting_idx;
  for (unsigned int i = 0; i != boundary_expansions.size(); i++) {
    BoundaryExpansion & expansion = boundary_expansions[i];
    if (expansion.path_end_idx == 0.0) {
      continue;
    }
    expansion_starting_idx = path.size() - expansion.path_end_idx - 1;
    if (!reversing_segment) {
      findBoundaryExpansion(path[expansion_starting_idx], end_pose, expansion, costmap);
    } else {
      findBoundaryExpansion(end_pose, path[expansion_starting_idx], expansion, costmap);
    }
  }

  unsigned int best_expansion_idx = findShortestBoundaryExpansionIdx(boundary_expansions);
  if (best_expansion_idx >= boundary_expansions.size()) {
    return;
  }

  BoundaryExpansion & best_expansion = boundary_expansions[best_expansion_idx];
  if (reversing_segment) {
    std::reverse(best_expansion.pts.begin(), best_expansion.pts.end());
  }
  expansion_starting_idx = path.size() - best_expansion.path_end_idx - 1;
  for (unsigned int i = 0; i != best_expansion.pts.size(); i++) {
    path[expansion_starting_idx + i].x = best_expansion.pts[i].x;
    path[expansion_starting_idx + i].y = best_expansion.pts[i].y;
    path[expansion_starting_idx + i].theta = best_expansion.pts[i].theta;
  }
}

}  // namespace hybrid_astar
