#pragma once

#include <memory>
#include <vector>
#include "state.h"
#include "state_space.h"
#include "dubins_state_space.h"

namespace steering_lite
{

    enum class PathType
    {
        Dubins,
        Rs
    };

    class SteeringPath
    {
    public:
        SteeringPath(PathType path_type, double kappa_max, double discretization);

        PathType path_type() const { return path_type_; }
        double kappa_max() const { return kappa_max_; }
        double discretization() const { return discretization_; }
        DubinsDirectionMode dubins_direction_mode() const { return dubins_direction_mode_; }

        void set_path_type(PathType path_type);
        void set_kappa_max(double kappa_max);
        void set_discretization(double discretization);
        void set_dubins_direction_mode(DubinsDirectionMode direction_mode);

        std::vector<Control> compute_shortest_control_sequence(const State& start, const State& goal) const;
        std::vector<State> compute_shortest_path(const State& start, const State& goal) const;
        std::vector<std::vector<Control>> compute_all_control_sequences(const State& start, const State& goal) const;
        std::vector<std::vector<State>> compute_all_paths(const State& start, const State& goal) const;

    private:
        void rebuild_planner();
        std::unique_ptr<StateSpace> build_planner() const;

        PathType path_type_;
        double kappa_max_;
        double discretization_;
        DubinsDirectionMode dubins_direction_mode_;
        std::unique_ptr<StateSpace> planner_;
    };

} // namespace steering_lite
