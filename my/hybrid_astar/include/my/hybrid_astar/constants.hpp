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

inline GoalHeadingMode fromStringToGH(const std::string & n)
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

const float UNKNOWN_COST = 255.0;
const float OCCUPIED_COST = 254.0;
const float INSCRIBED_COST = 253.0;
const float MAX_NON_OBSTACLE_COST = 252.0;
const float FREE_COST = 0;

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__CONSTANTS_HPP_
