#include "steering_functions_lite/steering_path.h"
#include "steering_functions_lite/dubins_state_space.h"
#include "steering_functions_lite/reeds_shepp_state_space.h"

#include <cassert>

namespace dubins_rs
{

    SteeringPath::SteeringPath(PathType path_type, double kappa_max, double discretization)
        : path_type_(path_type), kappa_max_(kappa_max), discretization_(discretization),
          dubins_direction_mode_(DubinsDirectionMode::ForwardOnly)
    {
        assert(kappa_max > 0.0 && discretization > 0.0);
        planner_ = build_planner();
    }

    std::unique_ptr<StateSpace> SteeringPath::build_planner() const
    {
        switch (path_type_)
        {
        case PathType::Dubins:
            return std::make_unique<DubinsStateSpace>(kappa_max_, discretization_, dubins_direction_mode_);
        case PathType::Rs:
            return std::make_unique<ReedsSheppStateSpace>(kappa_max_, discretization_);
        }
        return nullptr;
    }

    void SteeringPath::rebuild_planner()
    {
        planner_ = build_planner();
    }

    void SteeringPath::set_path_type(PathType path_type)
    {
        path_type_ = path_type;
        rebuild_planner();
    }

    void SteeringPath::set_kappa_max(double kappa_max)
    {
        assert(kappa_max > 0.0);
        kappa_max_ = kappa_max;
        rebuild_planner();
    }

    void SteeringPath::set_discretization(double discretization)
    {
        assert(discretization > 0.0);
        discretization_ = discretization;
        rebuild_planner();
    }

    void SteeringPath::set_dubins_direction_mode(DubinsDirectionMode direction_mode)
    {
        dubins_direction_mode_ = direction_mode;
        rebuild_planner();
    }

    std::vector<Control> SteeringPath::compute_shortest_control_sequence(const State& start, const State& goal) const
    {
        return planner_->get_controls(start, goal);
    }

    std::vector<State> SteeringPath::compute_shortest_path(const State& start, const State& goal) const
    {
        return planner_->get_path(start, goal);
    }

    std::vector<std::vector<Control>> SteeringPath::compute_all_control_sequences(const State& start, const State& goal) const
    {
        return planner_->get_all_controls(start, goal);
    }

    std::vector<std::vector<State>> SteeringPath::compute_all_paths(const State& start, const State& goal) const
    {
        return planner_->get_all_paths(start, goal);
    }

} // namespace dubins_rs
