#pragma once

#include <cmath>
#include <ostream>
#include <string>

namespace dubins_rs
{

    struct State
    {
        double x{0};
        double y{0};
        double theta{0};
        double kappa{0};
        double sigma{0};

        State() = default;
        State(double x_val, double y_val, double theta_val, double kappa_val = 0, double sigma_val = 0)
            : x(x_val), y(y_val), theta(theta_val), kappa(kappa_val), sigma(sigma_val)
        {
        }

        bool nearly_equal(const State& other) const
        {
            constexpr double EPS = 1e-6;
            return std::fabs(x - other.x) < EPS &&
                   std::fabs(y - other.y) < EPS &&
                   std::fabs(theta - other.theta) < EPS &&
                   std::fabs(kappa - other.kappa) < EPS &&
                   std::fabs(sigma - other.sigma) < EPS;
        }

        bool operator==(const State& other) const { return nearly_equal(other); }
        bool operator!=(const State& other) const { return !(*this == other); }

        friend std::ostream& operator<<(std::ostream& os, const State& s)
        {
            os << "State(x=" << s.x << ", y=" << s.y
               << ", theta=" << s.theta << ", kappa=" << s.kappa
               << ", sigma=" << s.sigma << ")";
            return os;
        }

        std::string to_string() const
        {
            return "State(x=" + std::to_string(x) + ", y=" + std::to_string(y) +
                   ", theta=" + std::to_string(theta) + ", kappa=" + std::to_string(kappa) +
                   ", sigma=" + std::to_string(sigma) + ")";
        }
    };

    struct Control
    {
        double delta_s{0};
        double kappa{0};
        double sigma{0};

        Control() = default;
        Control(double delta_s_val, double kappa_val, double sigma_val = 0)
            : delta_s(delta_s_val), kappa(kappa_val), sigma(sigma_val)
        {
        }

        std::string to_string() const
        {
            return "Control(delta_s=" + std::to_string(delta_s) +
                   ", kappa=" + std::to_string(kappa) +
                   ", sigma=" + std::to_string(sigma) + ")";
        }
    };

} // namespace dubins_rs
