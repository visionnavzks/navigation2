#ifndef SMAC_PLANNER__UTILS_HPP_
#define SMAC_PLANNER__UTILS_HPP_

#include <vector>
#include <memory>
#include <string>
#include <cmath>

#include "nlohmann/json.hpp"
#include "my/smac_planner/types.hpp"
#include "my/smac_planner/constants.hpp"

namespace smac_planner
{

class Costmap2D;

inline Pose getWorldCoords(
  const float & mx, const float & my, const Costmap2D * costmap)
{
  Pose p;
  p.x = costmap->getOriginX() + mx * costmap->getResolution();
  p.y = costmap->getOriginY() + my * costmap->getResolution();
  p.theta = 0.0;
  return p;
}

inline double findCircumscribedCost(
  Costmap2D * costmap,
  double circumscribed_radius,
  double inflation_radius)
{
  if (inflation_radius < circumscribed_radius) {
    return 0.0;
  }
  double resolution = costmap->getResolution();
  double distance_cells = circumscribed_radius / resolution;
  double inflation_cells = inflation_radius / resolution;
  double cost = INSCRIBED_COST * (1.0 - distance_cells / inflation_cells);
  return std::max(0.0, cost);
}

inline void fromJsonToMetaData(const nlohmann::json & json, LatticeMetadata & lattice_metadata)
{
  json.at("turning_radius").get_to(lattice_metadata.min_turning_radius);
  json.at("grid_resolution").get_to(lattice_metadata.grid_resolution);
  json.at("num_of_headings").get_to(lattice_metadata.number_of_headings);
  json.at("heading_angles").get_to(lattice_metadata.heading_angles);
  json.at("number_of_trajectories").get_to(lattice_metadata.number_of_trajectories);
  json.at("motion_model").get_to(lattice_metadata.motion_model);
}

inline void fromJsonToPose(const nlohmann::json & json, MotionPose & pose)
{
  pose._x = json[0];
  pose._y = json[1];
  pose._theta = json[2];
}

inline void fromJsonToMotionPrimitive(
  const nlohmann::json & json, MotionPrimitive & motion_primitive)
{
  json.at("trajectory_id").get_to(motion_primitive.trajectory_id);
  json.at("start_angle_index").get_to(motion_primitive.start_angle);
  json.at("end_angle_index").get_to(motion_primitive.end_angle);
  json.at("trajectory_radius").get_to(motion_primitive.turning_radius);
  json.at("trajectory_length").get_to(motion_primitive.trajectory_length);
  json.at("arc_length").get_to(motion_primitive.arc_length);
  json.at("straight_length").get_to(motion_primitive.straight_length);
  json.at("left_turn").get_to(motion_primitive.left_turn);

  for (unsigned int i = 0; i < json["poses"].size(); i++) {
    MotionPose pose;
    fromJsonToPose(json["poses"][i], pose);
    motion_primitive.poses.push_back(pose);
  }
}

}  // namespace smac_planner

#endif  // SMAC_PLANNER__UTILS_HPP_
