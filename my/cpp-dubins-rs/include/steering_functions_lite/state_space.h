#pragma once

#include <vector>
#include "state.h"

namespace steering_lite
{

    class StateSpace
    {
    public:
        virtual ~StateSpace() = default;

        virtual std::vector<Control> get_controls(const State& s1, const State& s2) const = 0;
        virtual std::vector<std::vector<Control>> get_all_controls(const State& s1, const State& s2) const = 0;
        virtual double discretization() const = 0;

        std::vector<State> get_path(const State& state1, const State& state2) const;
        std::vector<State> integrate(const State& state, const std::vector<Control>& controls) const;
        std::vector<State> integrate_with_disc(const State& state, const std::vector<Control>& controls, double disc) const;
        State interpolate(const State& state, const std::vector<Control>& controls, double t) const;
        std::vector<std::vector<State>> get_all_paths(const State& s1, const State& s2) const;

    private:
        static State integrate_ode_step(const State& state, const Control& control, double integration_step);
    };

} // namespace steering_lite
