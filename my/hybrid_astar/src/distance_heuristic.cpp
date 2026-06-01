#include "my/hybrid_astar/distance_heuristic.hpp"
#include "my/hybrid_astar/node_hybrid.hpp"
#include "my/hybrid_astar/steering_state_space.hpp"

namespace hybrid_astar
{

template<>
template<typename MotionTableT>
void DistanceHeuristic<NodeHybrid>::precomputeDistanceHeuristic(
  const float & lookup_table_dim,
  const MotionModel & motion_model,
  const unsigned int & dim_3_size,
  const SearchInfo & search_info,
  MotionTableT & motion_table)
{
  motion_table.state_space = createSteeringStateSpace(motion_model, search_info.minimum_turning_radius);

  SteeringState from, to;
  to[0] = 0.0;
  to[1] = 0.0;
  to[2] = 0.0;
  size_lookup_ = lookup_table_dim;
  float motion_heuristic = 0.0;
  unsigned int index = 0;
  int dim_3_size_int = static_cast<int>(dim_3_size);
  float angular_bin_size = 2 * M_PI / static_cast<float>(dim_3_size);

  dist_heuristic_lookup_table_.resize(
    (static_cast<int>(floor(size_lookup_ / 2.0)) -
     static_cast<int>(ceil(-size_lookup_ / 2.0)) + 1) *
    (static_cast<int>(floor(size_lookup_ / 2.0)) + 1) * dim_3_size_int);
  for (float x = ceil(-size_lookup_ / 2.0); x <= floor(size_lookup_ / 2.0); x += 1.0) {
    for (float y = 0.0; y <= floor(size_lookup_ / 2.0); y += 1.0) {
      for (int heading = 0; heading != dim_3_size_int; heading++) {
        from[0] = x;
        from[1] = y;
        from[2] = heading * angular_bin_size;
        motion_heuristic = motion_table.state_space->distance(from, to);
        dist_heuristic_lookup_table_[index] = motion_heuristic;
        index++;
      }
    }
  }
}

template<typename NodeT>
template<typename MotionTableT>
float DistanceHeuristic<NodeT>::getDistanceHeuristic(
  const Coordinates & node_coords,
  const Coordinates & goal_coords,
  const float & obstacle_heuristic,
  MotionTableT & motion_table)
{
  const TrigValues & trig_vals = motion_table.trig_values[goal_coords.theta];
  const float cos_th = trig_vals.first;
  const float sin_th = -trig_vals.second;
  const float dx = node_coords.x - goal_coords.x;
  const float dy = node_coords.y - goal_coords.y;

  double dtheta_bin = node_coords.theta - goal_coords.theta;
  dtheta_bin = wrapBinIndex(static_cast<int>(dtheta_bin), motion_table.num_angle_quantization);

  Coordinates node_coords_relative(
    round(dx * cos_th - dy * sin_th),
    round(dx * sin_th + dy * cos_th),
    round(dtheta_bin));

  float motion_heuristic = 0.0;
  const int floored_size = floor(size_lookup_ / 2.0);
  const int y_size = floored_size + 1;
  const float mirrored_relative_y = abs(node_coords_relative.y);
  if (abs(node_coords_relative.x) <= floored_size && mirrored_relative_y <= floored_size) {
    int theta_pos;
    if (node_coords_relative.y < 0.0) {
      theta_pos = motion_table.num_angle_quantization - node_coords_relative.theta;
      theta_pos %= motion_table.num_angle_quantization;
    } else {
      theta_pos = node_coords_relative.theta;
    }
    const int x_pos = node_coords_relative.x + floored_size;
    const int y_pos = static_cast<int>(mirrored_relative_y);
    const int index =
      x_pos * y_size * motion_table.num_angle_quantization +
      y_pos * motion_table.num_angle_quantization +
      theta_pos;
    motion_heuristic = dist_heuristic_lookup_table_[index];
  } else if (obstacle_heuristic <= 0.0) {
    SteeringState from, to;
    to[0] = goal_coords.x;
    to[1] = goal_coords.y;
    from[0] = node_coords.x;
    from[1] = node_coords.y;
    to[2] = motion_table.getAngleFromBin(goal_coords.theta);
    from[2] = motion_table.getAngleFromBin(node_coords.theta);
    motion_heuristic = motion_table.state_space->distance(from, to);
  }

  return motion_heuristic;
}

template void DistanceHeuristic<NodeHybrid>::precomputeDistanceHeuristic<HybridMotionTable>(
  const float &, const MotionModel &, const unsigned int &, const SearchInfo &,
  HybridMotionTable &);
template float DistanceHeuristic<NodeHybrid>::getDistanceHeuristic<HybridMotionTable>(
  const Coordinates &, const Coordinates &, const float &, HybridMotionTable &);

}  // namespace hybrid_astar
