#include <algorithm>

#include "steering_functions_lite/state_space.h"
#include "steering_functions_lite/math_utils.h"

namespace steering_lite
{

    State StateSpace::integrate_ode_step(const State& state, const Control& control, double integration_step)
    {
        State next;
        double kappa = control.kappa;
        double sigma = control.sigma;
        double d = sgn(control.delta_s);

        if (std::fabs(sigma) > EPSILON)
        {
            double x_f, y_f, theta_f, kappa_f;
            end_of_clothoid(state.x, state.y, state.theta, state.kappa,
                            sigma, d, integration_step,
                            x_f, y_f, theta_f, kappa_f);
            next.x = x_f;
            next.y = y_f;
            next.theta = theta_f;
            next.kappa = kappa_f;
            next.sigma = sigma;
        }
        else if (std::fabs(kappa) > EPSILON)
        {
            double x_f, y_f, theta_f;
            end_of_circular_arc(state.x, state.y, state.theta,
                                kappa, d, integration_step,
                                x_f, y_f, theta_f);
            next.x = x_f;
            next.y = y_f;
            next.theta = theta_f;
            next.kappa = kappa;
        }
        else
        {
            double x_f, y_f;
            end_of_straight_line(state.x, state.y, state.theta,
                                 d, integration_step,
                                 x_f, y_f);
            next.x = x_f;
            next.y = y_f;
            next.theta = state.theta;
        }

        return next;
    }

    std::vector<State> StateSpace::get_path(const State& state1, const State& state2) const
    {
        auto controls = get_controls(state1, state2);
        return integrate(state1, controls);
    }

    std::vector<State> StateSpace::integrate(const State& state, const std::vector<Control>& controls) const
    {
        return integrate_with_disc(state, controls, discretization());
    }

    std::vector<State> StateSpace::integrate_with_disc(const State& state, const std::vector<Control>& controls, double disc) const
    {
        std::vector<State> path;
        if (controls.empty())
            return path;

        State curr = state;
        curr.kappa = controls[0].kappa;
        curr.sigma = controls[0].sigma;
        path.push_back(curr);

        for (const auto& control : controls)
        {
            double abs_ds = std::fabs(control.delta_s);
            size_t n = static_cast<size_t>(std::max(2.0, std::ceil(abs_ds / disc)));
            double step = abs_ds / static_cast<double>(n);
            double s_seg = 0.0;

            for (size_t i = 0; i < n; ++i)
            {
                s_seg += step;
                double integration_step = (s_seg > abs_ds) ? step - (s_seg - abs_ds) : step;
                State next = integrate_ode_step(curr, control, integration_step);
                path.push_back(next);
                curr = next;
            }
        }
        return path;
    }

    State StateSpace::interpolate(const State& state, const std::vector<Control>& controls, double t) const
    {
        t = std::clamp(t, 0.0, 1.0);
        if (controls.empty())
            return state;

        double s_path = 0.0;
        for (const auto& c : controls)
            s_path += std::fabs(c.delta_s);

        double s_inter = t * s_path;
        double s_accum = 0.0;
        State curr = state;
        State result = state;

        for (const auto& control : controls)
        {
            double abs_ds = std::fabs(control.delta_s);
            if (s_inter - s_accum > abs_ds)
            {
                curr = integrate_ode_step(curr, control, abs_ds);
                s_accum += abs_ds;
            }
            else
            {
                result = integrate_ode_step(curr, control, s_inter - s_accum);
                return result;
            }
        }
        return result;
    }

    std::vector<std::vector<State>> StateSpace::get_all_paths(const State& s1, const State& s2) const
    {
        auto all_controls = get_all_controls(s1, s2);
        std::vector<std::vector<State>> all_paths;
        all_paths.reserve(all_controls.size());
        for (const auto& controls : all_controls)
            all_paths.push_back(integrate(s1, controls));
        return all_paths;
    }

} // namespace steering_lite
