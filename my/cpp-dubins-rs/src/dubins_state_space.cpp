#include "steering_functions_lite/dubins_state_space.h"
#include "steering_functions_lite/math_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace steering_lite
{

    namespace
    {
        const int DUBINS_LEFT = 0;
        const int DUBINS_STRAIGHT = 1;
        const int DUBINS_RIGHT = 2;

        const int DUBINS_PATH_TYPE[6][3] = {
            {DUBINS_LEFT, DUBINS_STRAIGHT, DUBINS_LEFT},
            {DUBINS_RIGHT, DUBINS_STRAIGHT, DUBINS_RIGHT},
            {DUBINS_RIGHT, DUBINS_STRAIGHT, DUBINS_LEFT},
            {DUBINS_LEFT, DUBINS_STRAIGHT, DUBINS_RIGHT},
            {DUBINS_RIGHT, DUBINS_LEFT, DUBINS_RIGHT},
            {DUBINS_LEFT, DUBINS_RIGHT, DUBINS_LEFT}};

        const double DUBINS_ZERO = -1e-9;
    } // anonymous namespace

    DubinsStateSpace::DubinsStateSpace(double kappa, double discretization, DubinsDirectionMode direction_mode)
        : kappa_(kappa), discretization_(discretization), direction_mode_(direction_mode)
    {
        assert(kappa > 0.0 && discretization > 0.0);
    }

    DubinsStateSpace::DubinsPath DubinsStateSpace::dubins_lsl(double d, double alpha, double beta) const
    {
        double ca = std::cos(alpha), sa = std::sin(alpha);
        double cb = std::cos(beta), sb = std::sin(beta);
        double tmp = 2.0 + d * d - 2.0 * (ca * cb + sa * sb - d * (sa - sb));
        if (tmp >= DUBINS_ZERO)
        {
            double theta = std::atan2(cb - ca, d + sa - sb);
            double t = twopify(-alpha + theta);
            double p = std::sqrt(std::max(tmp, 0.0));
            double q = twopify(beta - theta);
            DubinsPath path;
            for (int i = 0; i < 3; ++i) path.type_[i] = DUBINS_PATH_TYPE[0][i];
            path.length_[0] = t;
            path.length_[1] = p;
            path.length_[2] = q;
            return path;
        }
        return DubinsPath{};
    }

    DubinsStateSpace::DubinsPath DubinsStateSpace::dubins_rsr(double d, double alpha, double beta) const
    {
        double ca = std::cos(alpha), sa = std::sin(alpha);
        double cb = std::cos(beta), sb = std::sin(beta);
        double tmp = 2.0 + d * d - 2.0 * (ca * cb + sa * sb - d * (sb - sa));
        if (tmp >= DUBINS_ZERO)
        {
            double theta = std::atan2(ca - cb, d - sa + sb);
            double t = twopify(alpha - theta);
            double p = std::sqrt(std::max(tmp, 0.0));
            double q = twopify(-beta + theta);
            DubinsPath path;
            for (int i = 0; i < 3; ++i) path.type_[i] = DUBINS_PATH_TYPE[1][i];
            path.length_[0] = t;
            path.length_[1] = p;
            path.length_[2] = q;
            return path;
        }
        return DubinsPath{};
    }

    DubinsStateSpace::DubinsPath DubinsStateSpace::dubins_rsl(double d, double alpha, double beta) const
    {
        double ca = std::cos(alpha), sa = std::sin(alpha);
        double cb = std::cos(beta), sb = std::sin(beta);
        double tmp = d * d - 2.0 + 2.0 * (ca * cb + sa * sb - d * (sa + sb));
        if (tmp >= DUBINS_ZERO)
        {
            double p = std::sqrt(std::max(tmp, 0.0));
            double theta = std::atan2(ca + cb, d - sa - sb) - std::atan2(2.0, p);
            double t = twopify(alpha - theta);
            double q = twopify(beta - theta);
            DubinsPath path;
            for (int i = 0; i < 3; ++i) path.type_[i] = DUBINS_PATH_TYPE[2][i];
            path.length_[0] = t;
            path.length_[1] = p;
            path.length_[2] = q;
            return path;
        }
        return DubinsPath{};
    }

    DubinsStateSpace::DubinsPath DubinsStateSpace::dubins_lsr(double d, double alpha, double beta) const
    {
        double ca = std::cos(alpha), sa = std::sin(alpha);
        double cb = std::cos(beta), sb = std::sin(beta);
        double tmp = -2.0 + d * d + 2.0 * (ca * cb + sa * sb + d * (sa + sb));
        if (tmp >= DUBINS_ZERO)
        {
            double p = std::sqrt(std::max(tmp, 0.0));
            double theta = std::atan2(-ca - cb, d + sa + sb) - std::atan2(-2.0, p);
            double t = twopify(-alpha + theta);
            double q = twopify(-beta + theta);
            DubinsPath path;
            for (int i = 0; i < 3; ++i) path.type_[i] = DUBINS_PATH_TYPE[3][i];
            path.length_[0] = t;
            path.length_[1] = p;
            path.length_[2] = q;
            return path;
        }
        return DubinsPath{};
    }

    DubinsStateSpace::DubinsPath DubinsStateSpace::dubins_rlr(double d, double alpha, double beta) const
    {
        double ca = std::cos(alpha), sa = std::sin(alpha);
        double cb = std::cos(beta), sb = std::sin(beta);
        double tmp = 0.125 * (6.0 - d * d + 2.0 * (ca * cb + sa * sb + d * (sa - sb)));
        if (std::fabs(tmp) <= 1.0)
        {
            double p = twopify(TWO_PI - std::acos(tmp));
            double theta = std::atan2(ca - cb, d - sa + sb);
            double t = twopify(alpha - theta + 0.5 * p);
            double q = twopify(alpha - beta - t + p);
            DubinsPath path;
            for (int i = 0; i < 3; ++i) path.type_[i] = DUBINS_PATH_TYPE[4][i];
            path.length_[0] = t;
            path.length_[1] = p;
            path.length_[2] = q;
            return path;
        }
        return DubinsPath{};
    }

    DubinsStateSpace::DubinsPath DubinsStateSpace::dubins_lrl(double d, double alpha, double beta) const
    {
        double ca = std::cos(alpha), sa = std::sin(alpha);
        double cb = std::cos(beta), sb = std::sin(beta);
        double tmp = 0.125 * (6.0 - d * d + 2.0 * (ca * cb + sa * sb - d * (sa - sb)));
        if (std::fabs(tmp) <= 1.0)
        {
            double p = twopify(TWO_PI - std::acos(tmp));
            double theta = std::atan2(-ca + cb, d + sa - sb);
            double t = twopify(-alpha + theta + 0.5 * p);
            double q = twopify(beta - alpha - t + p);
            DubinsPath path;
            for (int i = 0; i < 3; ++i) path.type_[i] = DUBINS_PATH_TYPE[5][i];
            path.length_[0] = t;
            path.length_[1] = p;
            path.length_[2] = q;
            return path;
        }
        return DubinsPath{};
    }

    DubinsStateSpace::DubinsPath DubinsStateSpace::dubins_word(int path_type, double d, double alpha, double beta) const
    {
        switch (path_type)
        {
        case 0: return dubins_lsl(d, alpha, beta);
        case 1: return dubins_rsr(d, alpha, beta);
        case 2: return dubins_rsl(d, alpha, beta);
        case 3: return dubins_lsr(d, alpha, beta);
        case 4: return dubins_rlr(d, alpha, beta);
        case 5: return dubins_lrl(d, alpha, beta);
        default: return DubinsPath{};
        }
    }

    std::tuple<double, double, double> DubinsStateSpace::dubins_parameters(
        const State& q0, const State& q1, double rho, bool forward) const
    {
        double dx = q1.x - q0.x;
        double dy = q1.y - q0.y;
        double d = std::sqrt(dx * dx + dy * dy) / rho;
        double theta0 = forward ? q0.theta : q0.theta + M_PI;
        double theta1 = forward ? q1.theta : q1.theta + M_PI;
        double heading = std::atan2(dy, dx);
        double alpha = twopify(theta0 - heading);
        double beta = twopify(theta1 - heading);
        return {d, alpha, beta};
    }

    std::pair<DubinsStateSpace::DubinsPath, double> DubinsStateSpace::best_dubins_path(
        double d, double alpha, double beta) const
    {
        DubinsPath best;
        double best_len = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 6; ++i)
        {
            DubinsPath candidate = dubins_word(i, d, alpha, beta);
            double candidate_len = candidate.length();
            if (candidate_len < best_len)
            {
                best_len = candidate_len;
                best = candidate;
            }
        }
        return {best, best_len};
    }

    std::pair<DubinsStateSpace::DubinsPath, double> DubinsStateSpace::shortest_dubins_path(
        const State& q0, const State& q1, double rho, bool forward) const
    {
        auto [d, alpha, beta] = dubins_parameters(q0, q1, rho, forward);
        auto [best, best_len] = best_dubins_path(d, alpha, beta);
        return {best, best_len * rho};
    }

    std::vector<Control> DubinsStateSpace::controls_from_dubins(const DubinsPath& path, double rho, bool forward) const
    {
        double d = forward ? 1.0 : -1.0;
        double kappa_dir = forward ? 1.0 : -1.0;
        std::vector<Control> controls;

        for (int i = 0; i < 3; ++i)
        {
            double length = path.length_[i] * rho;
            if (std::fabs(length) < SEGMENT_EPS) continue;

            double base_kappa, sigma;
            switch (path.type_[i])
            {
            case DUBINS_LEFT:
                base_kappa = 1.0 / rho;
                sigma = 0.0;
                break;
            case DUBINS_STRAIGHT:
                base_kappa = 0.0;
                sigma = 0.0;
                break;
            default:
                base_kappa = -1.0 / rho;
                sigma = 0.0;
                break;
            }
            double kappa = kappa_dir * base_kappa;
            controls.emplace_back(d * length, kappa, sigma);
        }
        return controls;
    }

    std::vector<std::vector<Control>> DubinsStateSpace::all_controls_for_direction(
        const State& s1, const State& s2, double rho, bool forward) const
    {
        auto [d, alpha, beta] = dubins_parameters(s1, s2, rho, forward);
        std::vector<std::vector<Control>> all;
        for (int i = 0; i < 6; ++i)
        {
            DubinsPath path = dubins_word(i, d, alpha, beta);
            if (path.is_valid())
                all.push_back(controls_from_dubins(path, rho, forward));
        }
        return all;
    }

    std::vector<bool> DubinsStateSpace::direction_candidates() const
    {
        switch (direction_mode_)
        {
        case DubinsDirectionMode::ForwardOnly:
            return {true};
        case DubinsDirectionMode::ReverseOnly:
            return {false};
        case DubinsDirectionMode::ForwardOrReverse:
            return {true, false};
        }
        return {true};
    }

    std::pair<DubinsStateSpace::DubinsPath, bool> DubinsStateSpace::best_path_over_directions(
        const State& s1, const State& s2, double rho) const
    {
        DubinsPath best_path;
        double best_len = std::numeric_limits<double>::infinity();
        bool best_forward = true;

        for (bool forward : direction_candidates())
        {
            auto [path, len] = shortest_dubins_path(s1, s2, rho, forward);
            if (len < best_len)
            {
                best_len = len;
                best_path = path;
                best_forward = forward;
            }
        }
        return {best_path, best_forward};
    }

    std::vector<Control> DubinsStateSpace::get_controls(const State& s1, const State& s2) const
    {
        double rho = 1.0 / kappa_;
        auto [path, forward] = best_path_over_directions(s1, s2, rho);
        return controls_from_dubins(path, rho, forward);
    }

    std::vector<std::vector<Control>> DubinsStateSpace::get_all_controls(const State& s1, const State& s2) const
    {
        double rho = 1.0 / kappa_;
        std::vector<std::vector<Control>> all;
        for (bool forward : direction_candidates())
        {
            auto controls = all_controls_for_direction(s1, s2, rho, forward);
            all.insert(all.end(), controls.begin(), controls.end());
        }
        return all;
    }

    double DubinsStateSpace::discretization() const
    {
        return discretization_;
    }

} // namespace steering_lite
