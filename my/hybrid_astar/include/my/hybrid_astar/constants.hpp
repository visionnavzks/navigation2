#ifndef HYBRID_ASTAR__CONSTANTS_HPP_
#define HYBRID_ASTAR__CONSTANTS_HPP_

#include <string>

namespace hybrid_astar
{
enum class MotionModel
{
  UNKNOWN = 0,
  DUBIN = 2,
  REEDS_SHEPP = 3,
};

enum class GoalHeadingMode
{
  UNKNOWN = 0,
  DEFAULT = 1,
  BIDIRECTIONAL = 2,
  ALL_DIRECTION = 3,
};

inline std::string toString(const MotionModel & n)
{
  switch (n) {
    case MotionModel::DUBIN:
      return "Dubin";
    case MotionModel::REEDS_SHEPP:
      return "Reeds-Shepp";
    default:
      return "Unknown";
  }
}

inline MotionModel fromString(const std::string & n)
{
  if (n == "DUBIN") {
    return MotionModel::DUBIN;
  } else if (n == "REEDS_SHEPP") {
    return MotionModel::REEDS_SHEPP;
  } else {
    return MotionModel::UNKNOWN;
  }
}

inline std::string toString(const GoalHeadingMode & n)
{
  switch (n) {
    case GoalHeadingMode::DEFAULT:
      return "DEFAULT";
    case GoalHeadingMode::BIDIRECTIONAL:
      return "BIDIRECTIONAL";
    case GoalHeadingMode::ALL_DIRECTION:
      return "ALL_DIRECTION";
    default:
      return "Unknown";
  }
}

inline GoalHeadingMode toGoalHeadingMode(const std::string & n)
{
  if (n == "DEFAULT") {
    return GoalHeadingMode::DEFAULT;
  } else if (n == "BIDIRECTIONAL") {
    return GoalHeadingMode::BIDIRECTIONAL;
  } else if (n == "ALL_DIRECTION") {
    return GoalHeadingMode::ALL_DIRECTION;
  } else {
    return GoalHeadingMode::UNKNOWN;
  }
}

inline constexpr float UNKNOWN_COST = 255.0f;
inline constexpr float OCCUPIED_COST = 254.0f;
inline constexpr float INSCRIBED_COST = 253.0f;
inline constexpr float MAX_NON_OBSTACLE_COST = 252.0f;
inline constexpr float MAX_NON_OBSTACLE_COST_SQ = MAX_NON_OBSTACLE_COST * MAX_NON_OBSTACLE_COST;
inline constexpr float FREE_COST = 0.0f;

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__CONSTANTS_HPP_
