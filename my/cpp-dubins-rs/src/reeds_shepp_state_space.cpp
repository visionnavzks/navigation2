#include "steering_functions_lite/reeds_shepp_state_space.h"
#include "steering_functions_lite/math_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace dubins_rs
{

    namespace
    {
        const int RS_NOP = 0;
        const int RS_LEFT = 1;
        const int RS_STRAIGHT = 2;
        const int RS_RIGHT = 3;

        const int RS_PATH_TYPE[18][5] = {
            {RS_LEFT, RS_RIGHT, RS_LEFT, RS_NOP, RS_NOP},
            {RS_RIGHT, RS_LEFT, RS_RIGHT, RS_NOP, RS_NOP},
            {RS_LEFT, RS_RIGHT, RS_LEFT, RS_RIGHT, RS_NOP},
            {RS_RIGHT, RS_LEFT, RS_RIGHT, RS_LEFT, RS_NOP},
            {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_NOP},
            {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_NOP},
            {RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_LEFT, RS_NOP},
            {RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_RIGHT, RS_NOP},
            {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_NOP},
            {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_NOP},
            {RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_LEFT, RS_NOP},
            {RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_RIGHT, RS_NOP},
            {RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_NOP, RS_NOP},
            {RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_NOP, RS_NOP},
            {RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_NOP, RS_NOP},
            {RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_NOP, RS_NOP},
            {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_RIGHT},
            {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_LEFT}};

        const double RS_ZERO = 10.0 * std::numeric_limits<double>::epsilon();
    } // anonymous namespace

    ReedsSheppStateSpace::ReedsSheppStateSpace(double kappa, double discretization)
        : kappa_(kappa), discretization_(discretization), kappa_inv_(1.0 / kappa)
    {
        assert(kappa > 0.0 && discretization > 0.0);
    }

    void ReedsSheppStateSpace::tau_omega(double u, double v, double xi, double eta, double phi,
                                         double& tau, double& omega) const
    {
        double delta = pify(u - v);
        double a = std::sin(u) - std::sin(delta);
        double b = std::cos(u) - std::cos(delta) - 1.0;
        double t1 = std::atan2(eta * a - xi * b, xi * a + eta * b);
        double t2 = 2.0 * (std::cos(delta) - std::cos(v) - std::cos(u)) + 3.0;
        tau = (t2 < 0.0) ? pify(t1 + M_PI) : pify(t1);
        omega = pify(tau - u + v - phi);
    }

    bool ReedsSheppStateSpace::lp_sp_lp(double x, double y, double phi, double& t, double& u, double& v) const
    {
        auto [r, theta] = polar(x - std::sin(phi), y - 1.0 + std::cos(phi));
        u = r;
        t = theta;
        if (t >= -RS_ZERO)
        {
            v = pify(phi - t);
            if (v >= -RS_ZERO)
                return true;
        }
        return false;
    }

    bool ReedsSheppStateSpace::lp_sp_rp(double x, double y, double phi, double& t, double& u, double& v) const
    {
        double u1, t1;
        auto [r1, theta1] = polar(x + std::sin(phi), y - 1.0 - std::cos(phi));
        u1 = r1;
        t1 = theta1;
        double u1sq = u1 * u1;
        if (u1sq >= 4.0)
        {
            double theta;
            u = std::sqrt(u1sq - 4.0);
            theta = std::atan2(2.0, u);
            t = pify(t1 + theta);
            v = pify(t - phi);
            if (t >= -RS_ZERO && v >= -RS_ZERO)
                return true;
        }
        return false;
    }

    bool ReedsSheppStateSpace::lp_rm_l(double x, double y, double phi, double& t, double& u, double& v) const
    {
        double xi = x - std::sin(phi);
        double eta = y - 1.0 + std::cos(phi);
        auto [u1, theta] = polar(xi, eta);
        if (u1 <= 4.0)
        {
            u = -2.0 * std::asin(0.25 * u1);
            t = pify(theta + 0.5 * u + M_PI);
            v = pify(phi - t + u);
            if (t >= -RS_ZERO && u <= RS_ZERO)
                return true;
        }
        return false;
    }

    bool ReedsSheppStateSpace::lp_rup_lum_rm(double x, double y, double phi, double& t, double& u, double& v) const
    {
        double xi = x + std::sin(phi);
        double eta = y - 1.0 - std::cos(phi);
        double rho = 0.25 * (2.0 + std::sqrt(xi * xi + eta * eta));
        if (rho <= 1.0)
        {
            u = std::acos(rho);
            tau_omega(u, -u, xi, eta, phi, t, v);
            if (t >= -RS_ZERO && v <= RS_ZERO)
                return true;
        }
        return false;
    }

    bool ReedsSheppStateSpace::lp_rum_lum_rp(double x, double y, double phi, double& t, double& u, double& v) const
    {
        double xi = x + std::sin(phi);
        double eta = y - 1.0 - std::cos(phi);
        double rho = (20.0 - xi * xi - eta * eta) / 16.0;
        if (rho >= 0.0 && rho <= 1.0)
        {
            u = -std::acos(rho);
            if (u >= -0.5 * M_PI)
            {
                tau_omega(u, u, xi, eta, phi, t, v);
                if (t >= -RS_ZERO && v >= -RS_ZERO)
                    return true;
            }
        }
        return false;
    }

    bool ReedsSheppStateSpace::lp_rm_sm_lm(double x, double y, double phi, double& t, double& u, double& v) const
    {
        double xi = x - std::sin(phi);
        double eta = y - 1.0 + std::cos(phi);
        auto [rho, theta] = polar(xi, eta);
        if (rho >= 2.0)
        {
            double r = std::sqrt(rho * rho - 4.0);
            u = 2.0 - r;
            t = pify(theta + std::atan2(r, -2.0));
            v = pify(phi - 0.5 * M_PI - t);
            if (t >= -RS_ZERO && u <= RS_ZERO && v <= RS_ZERO)
                return true;
        }
        return false;
    }

    bool ReedsSheppStateSpace::lp_rm_sm_rm(double x, double y, double phi, double& t, double& u, double& v) const
    {
        double xi = x + std::sin(phi);
        double eta = y - 1.0 - std::cos(phi);
        auto [rho, theta] = polar(-eta, xi);
        if (rho >= 2.0)
        {
            t = theta;
            u = 2.0 - rho;
            v = pify(t + 0.5 * M_PI - phi);
            if (t >= -RS_ZERO && u <= RS_ZERO && v <= RS_ZERO)
                return true;
        }
        return false;
    }

    bool ReedsSheppStateSpace::lp_rm_s_lm_rp(double x, double y, double phi, double& t, double& u, double& v) const
    {
        double xi = x + std::sin(phi);
        double eta = y - 1.0 - std::cos(phi);
        auto [rho, _theta] = polar(xi, eta);
        if (rho >= 2.0)
        {
            u = 4.0 - std::sqrt(rho * rho - 4.0);
            if (u <= RS_ZERO)
            {
                t = pify(std::atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta));
                v = pify(t - phi);
                if (t >= -RS_ZERO && v >= -RS_ZERO)
                    return true;
            }
        }
        return false;
    }

    std::vector<ReedsSheppStateSpace::RsPath> ReedsSheppStateSpace::collect_all_paths(
        double x, double y, double phi) const
    {
        std::vector<RsPath> paths;
        double xb = x * std::cos(phi) + y * std::sin(phi);
        double yb = x * std::sin(phi) - y * std::cos(phi);

        double t, u, v;

        // CSC
        struct CSCEntry { double nx, ny, nphi; int pidx; double sign; };
        for (const auto& e : std::vector<CSCEntry>{
                 {x, y, phi, 14, 1.0}, {-x, y, -phi, 14, -1.0},
                 {x, -y, -phi, 15, 1.0}, {-x, -y, phi, 15, -1.0}})
        {
            if (lp_sp_lp(e.nx, e.ny, e.nphi, t, u, v))
            {
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * t;
                p.length_[1] = e.sign * u;
                p.length_[2] = e.sign * v;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) + std::fabs(p.length_[2]);
            }
        }
        for (const auto& e : std::vector<CSCEntry>{
                 {x, y, phi, 12, 1.0}, {-x, y, -phi, 12, -1.0},
                 {x, -y, -phi, 13, 1.0}, {-x, -y, phi, 13, -1.0}})
        {
            if (lp_sp_rp(e.nx, e.ny, e.nphi, t, u, v))
            {
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * t;
                p.length_[1] = e.sign * u;
                p.length_[2] = e.sign * v;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) + std::fabs(p.length_[2]);
            }
        }

        // CCC
        struct CCCEntry { double nx, ny, nphi; int pidx; double sign; bool rev; };
        for (const auto& e : std::vector<CCCEntry>{
                 {x, y, phi, 0, 1.0, false}, {-x, y, -phi, 0, -1.0, false},
                 {x, -y, -phi, 1, 1.0, false}, {-x, -y, phi, 1, -1.0, false},
                 {xb, yb, phi, 0, 1.0, true}, {-xb, yb, -phi, 0, -1.0, true},
                 {xb, -yb, -phi, 1, 1.0, true}, {-xb, -yb, phi, 1, -1.0, true}})
        {
            if (lp_rm_l(e.nx, e.ny, e.nphi, t, u, v))
            {
                double a = e.rev ? v : t;
                double b = e.rev ? u : u;
                double c = e.rev ? t : v;
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * a;
                p.length_[1] = e.sign * b;
                p.length_[2] = e.sign * c;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) + std::fabs(p.length_[2]);
            }
        }

        // CCCC
        struct CCCCEntry { double nx, ny, nphi; int pidx; double sign; bool neg_mid; };
        for (const auto& e : std::vector<CCCCEntry>{
                 {x, y, phi, 2, 1.0, true}, {-x, y, -phi, 2, -1.0, true},
                 {x, -y, -phi, 3, 1.0, true}, {-x, -y, phi, 3, -1.0, true}})
        {
            if (lp_rup_lum_rm(e.nx, e.ny, e.nphi, t, u, v))
            {
                double u2 = e.neg_mid ? -u : u;
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * t;
                p.length_[1] = e.sign * u;
                p.length_[2] = e.sign * u2;
                p.length_[3] = e.sign * v;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) +
                           std::fabs(p.length_[2]) + std::fabs(p.length_[3]);
            }
        }
        for (const auto& e : std::vector<CCCCEntry>{
                 {x, y, phi, 2, 1.0, false}, {-x, y, -phi, 2, -1.0, false},
                 {x, -y, -phi, 3, 1.0, false}, {-x, -y, phi, 3, -1.0, false}})
        {
            if (lp_rum_lum_rp(e.nx, e.ny, e.nphi, t, u, v))
            {
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * t;
                p.length_[1] = e.sign * u;
                p.length_[2] = e.sign * u;
                p.length_[3] = e.sign * v;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) +
                           std::fabs(p.length_[2]) + std::fabs(p.length_[3]);
            }
        }

        // CCSC forward
        struct CCSCE { double nx, ny, nphi; int pidx; double sign; };
        for (const auto& e : std::vector<CCSCE>{
                 {x, y, phi, 4, 1.0}, {-x, y, -phi, 4, -1.0},
                 {x, -y, -phi, 5, 1.0}, {-x, -y, phi, 5, -1.0}})
        {
            if (lp_rm_sm_lm(e.nx, e.ny, e.nphi, t, u, v))
            {
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * t;
                p.length_[1] = e.sign * (-0.5 * M_PI);
                p.length_[2] = e.sign * u;
                p.length_[3] = e.sign * v;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) +
                           std::fabs(p.length_[2]) + std::fabs(p.length_[3]);
            }
        }
        for (const auto& e : std::vector<CCSCE>{
                 {x, y, phi, 8, 1.0}, {-x, y, -phi, 8, -1.0},
                 {x, -y, -phi, 9, 1.0}, {-x, -y, phi, 9, -1.0}})
        {
            if (lp_rm_sm_rm(e.nx, e.ny, e.nphi, t, u, v))
            {
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * t;
                p.length_[1] = e.sign * (-0.5 * M_PI);
                p.length_[2] = e.sign * u;
                p.length_[3] = e.sign * v;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) +
                           std::fabs(p.length_[2]) + std::fabs(p.length_[3]);
            }
        }

        // CCSC backwards
        for (const auto& e : std::vector<CCSCE>{
                 {xb, yb, phi, 6, 1.0}, {-xb, yb, -phi, 6, -1.0},
                 {xb, -yb, -phi, 7, 1.0}, {-xb, -yb, phi, 7, -1.0}})
        {
            if (lp_rm_sm_lm(e.nx, e.ny, e.nphi, t, u, v))
            {
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * v;
                p.length_[1] = e.sign * u;
                p.length_[2] = e.sign * (-0.5 * M_PI);
                p.length_[3] = e.sign * t;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) +
                           std::fabs(p.length_[2]) + std::fabs(p.length_[3]);
            }
        }
        for (const auto& e : std::vector<CCSCE>{
                 {xb, yb, phi, 10, 1.0}, {-xb, yb, -phi, 10, -1.0},
                 {xb, -yb, -phi, 11, 1.0}, {-xb, -yb, phi, 11, -1.0}})
        {
            if (lp_rm_sm_rm(e.nx, e.ny, e.nphi, t, u, v))
            {
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * v;
                p.length_[1] = e.sign * u;
                p.length_[2] = e.sign * (-0.5 * M_PI);
                p.length_[3] = e.sign * t;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) +
                           std::fabs(p.length_[2]) + std::fabs(p.length_[3]);
            }
        }

        // CCSCC
        for (const auto& e : std::vector<CCSCE>{
                 {x, y, phi, 16, 1.0}, {-x, y, -phi, 16, -1.0},
                 {x, -y, -phi, 17, 1.0}, {-x, -y, phi, 17, -1.0}})
        {
            if (lp_rm_s_lm_rp(e.nx, e.ny, e.nphi, t, u, v))
            {
                paths.push_back(RsPath{});
                auto& p = paths.back();
                for (int i = 0; i < 5; ++i) p.type_[i] = RS_PATH_TYPE[e.pidx][i];
                p.length_[0] = e.sign * t;
                p.length_[1] = e.sign * (-0.5 * M_PI);
                p.length_[2] = e.sign * u;
                p.length_[3] = e.sign * (-0.5 * M_PI);
                p.length_[4] = e.sign * v;
                p.total_ = std::fabs(p.length_[0]) + std::fabs(p.length_[1]) +
                           std::fabs(p.length_[2]) + std::fabs(p.length_[3]) +
                           std::fabs(p.length_[4]);
            }
        }

        return paths;
    }

    ReedsSheppStateSpace::RsPath ReedsSheppStateSpace::shortest_rs_path(double x, double y, double phi) const
    {
        auto paths = collect_all_paths(x, y, phi);
        RsPath best;
        for (const auto& p : paths)
        {
            if (p.length() < best.length())
                best = p;
        }
        return best;
    }

    std::tuple<double, double, double> ReedsSheppStateSpace::normalize(
        const State& s1, const State& s2) const
    {
        double dx = s2.x - s1.x;
        double dy = s2.y - s1.y;
        double c = std::cos(s1.theta);
        double s = std::sin(s1.theta);
        double x = (c * dx + s * dy) * kappa_;
        double y = (-s * dx + c * dy) * kappa_;
        double phi = s2.theta - s1.theta;
        return {x, y, phi};
    }

    std::vector<Control> ReedsSheppStateSpace::controls_from_rs(const RsPath& path) const
    {
        std::vector<Control> controls;
        for (int i = 0; i < 5; ++i)
        {
            int seg = path.type_[i];
            if (seg == RS_NOP) break;
            double delta_s = kappa_inv_ * path.length_[i];
            if (std::fabs(delta_s) <= SEGMENT_EPS) continue;
            double kappa;
            switch (seg)
            {
            case RS_LEFT:
                kappa = 1.0 / kappa_inv_;
                break;
            case RS_RIGHT:
                kappa = -1.0 / kappa_inv_;
                break;
            default:
                kappa = 0.0;
                break;
            }
            controls.emplace_back(delta_s, kappa, 0.0);
        }
        return controls;
    }

    std::vector<Control> ReedsSheppStateSpace::get_controls(const State& s1, const State& s2) const
    {
        auto [x, y, phi] = normalize(s1, s2);
        RsPath path = shortest_rs_path(x, y, phi);
        return controls_from_rs(path);
    }

    std::vector<std::vector<Control>> ReedsSheppStateSpace::get_all_controls(const State& s1, const State& s2) const
    {
        auto [x, y, phi] = normalize(s1, s2);
        auto all_rs_paths = collect_all_paths(x, y, phi);
        std::vector<std::vector<Control>> all_controls;
        for (const auto& rs_path : all_rs_paths)
        {
            auto controls = controls_from_rs(rs_path);
            if (!controls.empty())
                all_controls.push_back(controls);
        }
        return all_controls;
    }

    double ReedsSheppStateSpace::discretization() const
    {
        return discretization_;
    }

    double ReedsSheppStateSpace::get_distance(const State& s1, const State& s2) const
    {
        auto [x, y, phi] = normalize(s1, s2);
        return shortest_rs_path(x, y, phi).length() / kappa_;
    }

} // namespace dubins_rs
