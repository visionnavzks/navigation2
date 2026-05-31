#ifndef HYBRID_ASTAR__TYPES_HPP_
#define HYBRID_ASTAR__TYPES_HPP_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <cmath>

#include "ompl/base/spaces/DubinsStateSpace.h"
#include "ompl/base/spaces/ReedsSheppStateSpace.h"

#include "my/hybrid_astar/constants.hpp"

namespace hybrid_astar
{

struct Pose { double x, y, theta; };
using Path = std::vector<Pose>;
struct Point2D { double x, y; };
inline bool operator==(const Point2D & a, const Point2D & b)
{
  return std::abs(a.x - b.x) < 1e-9 && std::abs(a.y - b.y) < 1e-9;
}
using Footprint = std::vector<Point2D>;

typedef std::pair<float, uint64_t> NodeHeuristicPair;
typedef std::vector<float> LookupTable;
typedef std::pair<double, double> TrigValues;

struct NodeHeuristicComparator
{
  bool operator()(const NodeHeuristicPair & a, const NodeHeuristicPair & b) const
  {
    return a.first > b.first;
  }
};

struct SearchInfo
{
  float minimum_turning_radius{8.0};
  float non_straight_penalty{1.05};
  float change_penalty{0.0};
  float reverse_penalty{2.0};
  float cost_penalty{2.0};
  float retrospective_penalty{0.015};
  float rotation_penalty{5.0};
  float analytic_expansion_ratio{3.5};
  float analytic_expansion_max_length{60.0};
  float analytic_expansion_max_cost{200.0};
  bool analytic_expansion_max_cost_override{false};
  bool cache_obstacle_heuristic{false};
  bool allow_reverse_expansion{false};
  bool allow_primitive_interpolation{false};
  bool downsample_obstacle_heuristic{true};
  bool use_quadratic_cost_penalty{false};
};

struct SmootherParams
{
  SmootherParams()
  : tolerance_(1e-3),
    max_its_(1000),
    w_data_(0.32),
    w_smooth_(0.25),
    holonomic_(false),
    do_refinement_(true),
    refinement_num_(3)
  {
  }

  void set(
    const double & tolerance,
    const int & max_its,
    const double & w_data,
    const double & w_smooth,
    const bool & do_refinement,
    const int & refinement_num)
  {
    tolerance_ = tolerance;
    max_its_ = max_its;
    w_data_ = w_data;
    w_smooth_ = w_smooth;
    do_refinement_ = do_refinement;
    refinement_num_ = refinement_num;
  }

  double tolerance_;
  int max_its_;
  double w_data_;
  double w_smooth_;
  bool holonomic_;
  bool do_refinement_;
  int refinement_num_;
};

enum class TurnDirection
{
  UNKNOWN = 0,
  FORWARD = 1,
  LEFT = 2,
  RIGHT = 3,
  REVERSE = 4,
  REV_LEFT = 5,
  REV_RIGHT = 6
};

struct MotionPose
{
  MotionPose() = default;

  MotionPose(const float & x, const float & y, const float & theta, const TurnDirection & turn_dir)
  : _x(x), _y(y), _theta(theta), _turn_dir(turn_dir)
  {}

  MotionPose operator-(const MotionPose & p2)
  {
    return MotionPose(
      this->_x - p2._x, this->_y - p2._y, this->_theta - p2._theta, TurnDirection::UNKNOWN);
  }

  float _x;
  float _y;
  float _theta;
  TurnDirection _turn_dir;
};

typedef std::vector<MotionPose> MotionPoses;

template<typename NodeT>
struct GoalState
{
  NodeT * goal = nullptr;
  bool is_valid = true;
};

struct Coordinates
{
  Coordinates() = default;

  Coordinates(const float & x_in, const float & y_in, const float & theta_in)
  : x(x_in), y(y_in), theta(theta_in)
  {}

  inline bool operator==(const Coordinates & rhs) const
  {
    return std::abs(x - rhs.x) < 1e-6f && std::abs(y - rhs.y) < 1e-6f &&
           std::abs(theta - rhs.theta) < 1e-6f;
  }

  inline bool operator!=(const Coordinates & rhs) const
  {
    return !(*this == rhs);
  }

  float x, y, theta;
};

inline unsigned int wrapBinIndex(int bin, unsigned int num_bins)
{
  bin %= static_cast<int>(num_bins);
  if (bin < 0) {bin += static_cast<int>(num_bins);}
  return static_cast<unsigned int>(bin);
}

inline double wrapAngle(double angle)
{
  angle = std::fmod(angle, 2.0 * M_PI);
  if (angle < 0.0) {angle += 2.0 * M_PI;}
  return angle;
}

inline ompl::base::StateSpacePtr createStateSpace(
  const MotionModel & model, double turning_radius)
{
  if (model == MotionModel::DUBIN) {
    return std::make_shared<ompl::base::DubinsStateSpace>(turning_radius);
  } else if (model == MotionModel::REEDS_SHEPP) {
    return std::make_shared<ompl::base::ReedsSheppStateSpace>(turning_radius);
  }
  throw std::runtime_error("Invalid motion model for state space creation");
}

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__TYPES_HPP_
