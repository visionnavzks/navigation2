#ifndef COSTMAP_2D__GEOMETRY_HPP_
#define COSTMAP_2D__GEOMETRY_HPP_

#include <cmath>

namespace costmap_2d {

struct Point {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct Pose2D {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct Transform2D {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};

  Point apply(const Point &point) const {
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    return Point{x + cos_yaw * point.x - sin_yaw * point.y,
                 y + sin_yaw * point.x + cos_yaw * point.y, point.z};
  }

  Pose2D apply(const Pose2D &pose) const {
    const Point transformed = apply(Point{pose.x, pose.y, 0.0});
    return Pose2D{transformed.x, transformed.y, pose.yaw + yaw};
  }
};

} // namespace costmap_2d

#endif // COSTMAP_2D__GEOMETRY_HPP_