#pragma once

#include <cmath>
#include <utility>

namespace steering_lite
{

    constexpr double HALF_PI = 1.5707963267948966192;
    constexpr double TWO_PI = 6.2831853071795864770;
    constexpr double SQRT_PI = 1.7724538509055160273;
    constexpr double SQRT_PI_INV = 0.56418958354775628695;
    constexpr double SQRT_TWO_PI_INV = 0.39894228040143267794;
    constexpr double EPSILON = 1e-4;
    constexpr double SEGMENT_EPS = 1e-6;

    double sgn(double x);
    std::pair<double, double> polar(double x, double y);
    double twopify(double alpha);
    double pify(double alpha);

    std::pair<double, double> fresnel(double s);

    void end_of_clothoid(double x_i, double y_i, double theta_i, double kappa_i,
                         double sigma, double direction, double length,
                         double& x_f, double& y_f, double& theta_f, double& kappa_f);

    void end_of_circular_arc(double x_i, double y_i, double theta_i,
                             double kappa, double direction, double length,
                             double& x_f, double& y_f, double& theta_f);

    void end_of_straight_line(double x_i, double y_i, double theta,
                              double direction, double length,
                              double& x_f, double& y_f);

} // namespace steering_lite
