#pragma once

#include <vector>
#include "state_space.h"

namespace steering_lite
{

    class ReedsSheppStateSpace : public StateSpace
    {
    public:
        ReedsSheppStateSpace(double kappa, double discretization);

        std::vector<Control> get_controls(const State& s1, const State& s2) const override;
        std::vector<std::vector<Control>> get_all_controls(const State& s1, const State& s2) const override;
        double discretization() const override;

        double get_distance(const State& s1, const State& s2) const;

    private:
        struct RsPath
        {
            int type_[5]{0, 0, 0, 0, 0};
            double length_[5]{0.0, 0.0, 0.0, 0.0, 0.0};
            double total_{1e30};

            double length() const { return total_; }
            bool is_valid() const { return total_ < 1e29; }
        };

        void tau_omega(double u, double v, double xi, double eta, double phi, double& tau, double& omega) const;
        bool lp_sp_lp(double x, double y, double phi, double& t, double& u, double& v) const;
        bool lp_sp_rp(double x, double y, double phi, double& t, double& u, double& v) const;
        bool lp_rm_l(double x, double y, double phi, double& t, double& u, double& v) const;
        bool lp_rup_lum_rm(double x, double y, double phi, double& t, double& u, double& v) const;
        bool lp_rum_lum_rp(double x, double y, double phi, double& t, double& u, double& v) const;
        bool lp_rm_sm_lm(double x, double y, double phi, double& t, double& u, double& v) const;
        bool lp_rm_sm_rm(double x, double y, double phi, double& t, double& u, double& v) const;
        bool lp_rm_s_lm_rp(double x, double y, double phi, double& t, double& u, double& v) const;

        std::vector<RsPath> collect_all_paths(double x, double y, double phi) const;
        RsPath shortest_rs_path(double x, double y, double phi) const;
        std::tuple<double, double, double> normalize(const State& s1, const State& s2) const;
        std::vector<Control> controls_from_rs(const RsPath& path) const;

        double kappa_;
        double discretization_;
        double kappa_inv_;
    };

} // namespace steering_lite
