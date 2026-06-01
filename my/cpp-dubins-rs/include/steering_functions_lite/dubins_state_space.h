#pragma once

#include <vector>
#include "state_space.h"

namespace dubins_rs
{

    enum class DubinsDirectionMode
    {
        ForwardOnly,
        ReverseOnly,
        ForwardOrReverse
    };

    class DubinsStateSpace : public StateSpace
    {
    public:
        DubinsStateSpace(double kappa, double discretization, DubinsDirectionMode direction_mode);

        std::vector<Control> get_controls(const State& s1, const State& s2) const override;
        std::vector<std::vector<Control>> get_all_controls(const State& s1, const State& s2) const override;
        double discretization() const override;

    private:
        struct DubinsPath
        {
            int type_[3]{0, 0, 0};
            double length_[3]{0.0, 1e30, 0.0};

            double length() const { return length_[0] + length_[1] + length_[2]; }
            bool is_valid() const { return length() < 1e29; }
        };

        DubinsPath dubins_lsl(double d, double alpha, double beta) const;
        DubinsPath dubins_rsr(double d, double alpha, double beta) const;
        DubinsPath dubins_rsl(double d, double alpha, double beta) const;
        DubinsPath dubins_lsr(double d, double alpha, double beta) const;
        DubinsPath dubins_rlr(double d, double alpha, double beta) const;
        DubinsPath dubins_lrl(double d, double alpha, double beta) const;

        DubinsPath dubins_word(int path_type, double d, double alpha, double beta) const;
        std::tuple<double, double, double> dubins_parameters(const State& q0, const State& q1, double rho, bool forward) const;
        std::pair<DubinsPath, double> best_dubins_path(double d, double alpha, double beta) const;
        std::pair<DubinsPath, double> shortest_dubins_path(const State& q0, const State& q1, double rho, bool forward) const;
        std::vector<Control> controls_from_dubins(const DubinsPath& path, double rho, bool forward) const;
        std::vector<std::vector<Control>> all_controls_for_direction(const State& s1, const State& s2, double rho, bool forward) const;

        std::vector<bool> direction_candidates() const;
        std::pair<DubinsPath, bool> best_path_over_directions(const State& s1, const State& s2, double rho) const;

        double kappa_;
        double discretization_;
        DubinsDirectionMode direction_mode_;
    };

} // namespace dubins_rs
