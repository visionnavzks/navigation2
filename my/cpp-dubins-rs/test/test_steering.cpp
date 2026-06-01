#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "steering_functions_lite/steering_path.h"
#include "steering_functions_lite/math_utils.h"
#include "steering_functions_lite/dubins_state_space.h"
#include "steering_functions_lite/reeds_shepp_state_space.h"
#include "steering_functions_lite/state_space.h"

using namespace steering_lite;

static int passed = 0;
static int failed = 0;

#define TEST(name) \
    void name(); \
    struct name##_reg { name##_reg() { name(); } } name##_instance; \
    void name()

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "FAIL: " << #a << " != " << #b << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        failed++; return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    if (std::fabs((a) - (b)) > (eps)) { \
        std::cerr << "FAIL: |" << #a << " - " << #b << "| > " << eps \
                  << " (" << (a) << " vs " << b << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        failed++; return; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        failed++; return; \
    } \
} while(0)

#define TEST_END(name) do { \
    std::cout << "  " << #name << ": PASSED" << std::endl; \
    passed++; \
} while(0)

// ==================== Math Utils Tests ====================

void test_sgn_positive()
{
    ASSERT_EQ(sgn(5.0), 1.0);
    TEST_END(test_sgn_positive);
}

void test_sgn_negative()
{
    ASSERT_EQ(sgn(-3.0), -1.0);
    TEST_END(test_sgn_negative);
}

void test_sgn_zero()
{
    ASSERT_EQ(sgn(0.0), 1.0);
    TEST_END(test_sgn_zero);
}

void test_polar_origin()
{
    auto [r, theta] = polar(0.0, 0.0);
    ASSERT_NEAR(r, 0.0, 1e-10);
    ASSERT_NEAR(theta, 0.0, 1e-10);
    TEST_END(test_polar_origin);
}

void test_polar_positive_x()
{
    auto [r, theta] = polar(3.0, 0.0);
    ASSERT_NEAR(r, 3.0, 1e-10);
    ASSERT_NEAR(theta, 0.0, 1e-10);
    TEST_END(test_polar_positive_x);
}

void test_polar_positive_y()
{
    auto [r, theta] = polar(0.0, 4.0);
    ASSERT_NEAR(r, 4.0, 1e-10);
    ASSERT_NEAR(theta, M_PI / 2.0, 1e-10);
    TEST_END(test_polar_positive_y);
}

void test_polar_quadrant1()
{
    auto [r, theta] = polar(3.0, 4.0);
    ASSERT_NEAR(r, 5.0, 1e-10);
    ASSERT_NEAR(theta, std::atan2(4.0, 3.0), 1e-10);
    TEST_END(test_polar_quadrant1);
}

void test_twopify_zero()
{
    ASSERT_NEAR(twopify(0.0), 0.0, 1e-10);
    TEST_END(test_twopify_zero);
}

void test_twopify_positive()
{
    double result = twopify(M_PI);
    ASSERT_NEAR(result, M_PI, 1e-10);
    TEST_END(test_twopify_positive);
}

void test_twopify_large()
{
    double result = twopify(5.0 * M_PI);
    ASSERT_TRUE(result >= 0.0 && result < TWO_PI);
    TEST_END(test_twopify_large);
}

void test_twopify_negative()
{
    double result = twopify(-M_PI);
    ASSERT_TRUE(result >= 0.0 && result < TWO_PI);
    TEST_END(test_twopify_negative);
}

void test_pify_zero()
{
    ASSERT_NEAR(pify(0.0), 0.0, 1e-10);
    TEST_END(test_pify_zero);
}

void test_pify_positive_pi()
{
    double result = pify(M_PI);
    ASSERT_NEAR(result, M_PI, 1e-10);
    TEST_END(test_pify_positive_pi);
}

void test_pify_negative_pi()
{
    double result = pify(-M_PI);
    ASSERT_TRUE((std::fabs(result - M_PI) < 1e-10) || (std::fabs(result + M_PI) < 1e-10));
    TEST_END(test_pify_negative_pi);
}

void test_pify_beyond_pi()
{
    double result = pify(3.0 * M_PI);
    ASSERT_TRUE(result > -M_PI && result <= M_PI);
    TEST_END(test_pify_beyond_pi);
}

void test_pify_negative_beyond()
{
    double result = pify(-3.0 * M_PI);
    ASSERT_TRUE((std::fabs(result - M_PI) < 1e-10) || (std::fabs(result + M_PI) < 1e-10));
    TEST_END(test_pify_negative_beyond);
}

void test_fresnel_zero()
{
    auto [s, c] = fresnel(0.0);
    ASSERT_NEAR(s, 0.0, 1e-10);
    ASSERT_NEAR(c, 0.0, 1e-10);
    TEST_END(test_fresnel_zero);
}

void test_fresnel_positive()
{
    auto [s, c] = fresnel(1.0);
    ASSERT_TRUE(s > 0.0 && s < 1.0);
    ASSERT_TRUE(c > 0.0 && c < 1.0);
    TEST_END(test_fresnel_positive);
}

void test_fresnel_negative()
{
    auto [s_pos, c_pos] = fresnel(1.0);
    auto [s_neg, c_neg] = fresnel(-1.0);
    ASSERT_NEAR(s_pos + s_neg, 0.0, 1e-10);
    ASSERT_NEAR(c_pos + c_neg, 0.0, 1e-10);
    TEST_END(test_fresnel_negative);
}

void test_fresnel_large()
{
    auto [s, c] = fresnel(10.0);
    ASSERT_NEAR(s, 0.5, 0.1);
    ASSERT_NEAR(c, 0.5, 0.1);
    TEST_END(test_fresnel_large);
}

void test_end_of_straight_line_forward()
{
    double x_f, y_f;
    end_of_straight_line(0.0, 0.0, 0.0, 1.0, 5.0, x_f, y_f);
    ASSERT_NEAR(x_f, 5.0, 1e-10);
    ASSERT_NEAR(y_f, 0.0, 1e-10);
    TEST_END(test_end_of_straight_line_forward);
}

void test_end_of_straight_line_backward()
{
    double x_f, y_f;
    end_of_straight_line(5.0, 0.0, 0.0, -1.0, 5.0, x_f, y_f);
    ASSERT_NEAR(x_f, 0.0, 1e-10);
    ASSERT_NEAR(y_f, 0.0, 1e-10);
    TEST_END(test_end_of_straight_line_backward);
}

void test_end_of_straight_line_angle()
{
    double x_f, y_f;
    end_of_straight_line(0.0, 0.0, M_PI / 2.0, 1.0, 3.0, x_f, y_f);
    ASSERT_NEAR(x_f, 0.0, 1e-10);
    ASSERT_NEAR(y_f, 3.0, 1e-10);
    TEST_END(test_end_of_straight_line_angle);
}

void test_end_of_circular_arc_zero_length()
{
    double x_f, y_f, theta_f;
    end_of_circular_arc(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, x_f, y_f, theta_f);
    ASSERT_NEAR(x_f, 0.0, 1e-6);
    ASSERT_NEAR(y_f, 0.0, 1e-6);
    TEST_END(test_end_of_circular_arc_zero_length);
}

void test_end_of_circular_arc_half_circle()
{
    double x_f, y_f, theta_f;
    end_of_circular_arc(0.0, 0.0, 0.0, 1.0, 1.0, M_PI, x_f, y_f, theta_f);
    ASSERT_NEAR(x_f, 0.0, 1e-6);
    ASSERT_NEAR(y_f, 2.0, 1e-6);
    TEST_END(test_end_of_circular_arc_half_circle);
}

void test_end_of_clothoid_zero_length()
{
    double x_f, y_f, theta_f, kappa_f;
    end_of_clothoid(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, x_f, y_f, theta_f, kappa_f);
    ASSERT_NEAR(x_f, 0.0, 1e-10);
    ASSERT_NEAR(y_f, 0.0, 1e-10);
    ASSERT_NEAR(theta_f, 0.0, 1e-10);
    ASSERT_NEAR(kappa_f, 0.0, 1e-10);
    TEST_END(test_end_of_clothoid_zero_length);
}

void test_end_of_clothoid_kappa_change()
{
    double x_f, y_f, theta_f, kappa_f;
    double sigma = 0.5;
    double length = 2.0;
    end_of_clothoid(0.0, 0.0, 0.0, 0.0, sigma, 1.0, length, x_f, y_f, theta_f, kappa_f);
    ASSERT_NEAR(kappa_f, sigma * length, 1e-10);
    TEST_END(test_end_of_clothoid_kappa_change);
}

// ==================== State/Control Tests ====================

void test_state_default()
{
    State s;
    ASSERT_EQ(s.x, 0.0);
    ASSERT_EQ(s.y, 0.0);
    ASSERT_EQ(s.theta, 0.0);
    ASSERT_EQ(s.kappa, 0.0);
    ASSERT_EQ(s.sigma, 0.0);
    TEST_END(test_state_default);
}

void test_state_nearly_equal_same()
{
    State s1(1.0, 2.0, 3.0, 4.0, 5.0);
    State s2(1.0, 2.0, 3.0, 4.0, 5.0);
    ASSERT_TRUE(s1.nearly_equal(s2));
    TEST_END(test_state_nearly_equal_same);
}

void test_state_nearly_equal_small_diff()
{
    State s1(1.0, 2.0, 3.0, 4.0, 5.0);
    State s2(1.0 + 1e-7, 2.0, 3.0, 4.0, 5.0);
    ASSERT_TRUE(s1.nearly_equal(s2));
    TEST_END(test_state_nearly_equal_small_diff);
}

void test_state_nearly_equal_large_diff()
{
    State s1(1.0, 2.0, 3.0, 4.0, 5.0);
    State s2(1.1, 2.0, 3.0, 4.0, 5.0);
    ASSERT_TRUE(!s1.nearly_equal(s2));
    TEST_END(test_state_nearly_equal_large_diff);
}

void test_state_eq()
{
    State s1(1.0, 2.0, 3.0, 4.0, 5.0);
    State s2(1.0, 2.0, 3.0, 4.0, 5.0);
    ASSERT_TRUE(s1 == s2);
    TEST_END(test_state_eq);
}

void test_state_eq_with_tolerance()
{
    State s1(1.0, 2.0, 3.0, 4.0, 5.0);
    State s2(1.0 + 1e-7, 2.0, 3.0, 4.0, 5.0);
    ASSERT_TRUE(s1 == s2);
    TEST_END(test_state_eq_with_tolerance);
}

void test_control_default()
{
    Control c;
    ASSERT_EQ(c.delta_s, 0.0);
    ASSERT_EQ(c.kappa, 0.0);
    ASSERT_EQ(c.sigma, 0.0);
    TEST_END(test_control_default);
}

void test_control_new()
{
    Control c(1.0, 2.0, 3.0);
    ASSERT_EQ(c.delta_s, 1.0);
    ASSERT_EQ(c.kappa, 2.0);
    ASSERT_EQ(c.sigma, 3.0);
    TEST_END(test_control_new);
}

// ==================== Dubins Tests ====================

void dubins_path_ends_near(const State& start, const State& goal, const std::vector<State>& path,
                           double pos_eps, double heading_eps)
{
    ASSERT_TRUE(!path.empty());
    const State& end = path.back();
    ASSERT_TRUE(std::fabs(end.x - goal.x) < pos_eps);
    ASSERT_TRUE(std::fabs(end.y - goal.y) < pos_eps);
    ASSERT_TRUE(std::fabs(pify(end.theta - goal.theta)) < heading_eps);
}

void test_dubins_forward_straight()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ForwardOnly);
    State start(0.0, 0.0, 0.0);
    State goal(5.0, 0.0, 0.0);
    auto path = planner.get_path(start, goal);
    dubins_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_dubins_forward_straight);
}

void test_dubins_forward_turn()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ForwardOnly);
    State start(0.0, 0.0, 0.0);
    State goal(2.0, 2.0, M_PI / 2.0);
    auto path = planner.get_path(start, goal);
    dubins_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_dubins_forward_turn);
}

void test_dubins_reverse_reaches_goal()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ReverseOnly);
    State start(-3.0, 0.0, 0.0);
    State goal(3.0, 2.0, M_PI / 4.0);
    auto path = planner.get_path(start, goal);
    dubins_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_dubins_reverse_reaches_goal);
}

void test_dubins_forward_or_reverse()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ForwardOrReverse);
    State start(0.0, 0.0, 0.0);
    State goal(0.5, 0.0, M_PI);
    auto path = planner.get_path(start, goal);
    dubins_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_dubins_forward_or_reverse);
}

void test_dubins_path_length_positive()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ForwardOnly);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto path = planner.get_path(start, goal);
    ASSERT_TRUE(path.size() > 1);
    TEST_END(test_dubins_path_length_positive);
}

void test_dubins_get_controls_nonempty()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ForwardOnly);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto controls = planner.get_controls(start, goal);
    ASSERT_TRUE(!controls.empty());
    TEST_END(test_dubins_get_controls_nonempty);
}

void test_dubins_get_all_controls_nonempty()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ForwardOrReverse);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto all_controls = planner.get_all_controls(start, goal);
    ASSERT_TRUE(!all_controls.empty());
    TEST_END(test_dubins_get_all_controls_nonempty);
}

void test_dubins_discretization()
{
    DubinsStateSpace planner(1.0, 0.1, DubinsDirectionMode::ForwardOnly);
    ASSERT_NEAR(planner.discretization(), 0.1, 1e-10);
    TEST_END(test_dubins_discretization);
}

void test_dubins_various_orientations()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ForwardOrReverse);
    State start(0.0, 0.0, 0.0);

    for (double angle : {0.0, M_PI / 4.0, M_PI / 2.0})
    {
        State goal(4.0, 0.0, angle);
        auto path = planner.get_path(start, goal);
        if (!path.empty())
        {
            dubins_path_ends_near(start, goal, path, 1e-2, 1e-2);
        }
    }
    TEST_END(test_dubins_various_orientations);
}

void test_dubins_integrate_empty_controls()
{
    DubinsStateSpace planner(1.0, 0.05, DubinsDirectionMode::ForwardOnly);
    State start(0.0, 0.0, 0.0);
    std::vector<Control> controls;
    auto path = planner.integrate(start, controls);
    ASSERT_TRUE(path.empty());
    TEST_END(test_dubins_integrate_empty_controls);
}

// ==================== Reeds-Shepp Tests ====================

void rs_path_ends_near(const State& start, const State& goal, const std::vector<State>& path,
                       double pos_eps, double heading_eps)
{
    ASSERT_TRUE(!path.empty());
    const State& end = path.back();
    ASSERT_TRUE(std::fabs(end.x - goal.x) < pos_eps);
    ASSERT_TRUE(std::fabs(end.y - goal.y) < pos_eps);
    ASSERT_TRUE(std::fabs(pify(end.theta - goal.theta)) < heading_eps);
}

void test_rs_straight_forward()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(5.0, 0.0, 0.0);
    auto path = planner.get_path(start, goal);
    rs_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_rs_straight_forward);
}

void test_rs_straight_reverse()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(-5.0, 0.0, 0.0);
    auto path = planner.get_path(start, goal);
    rs_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_rs_straight_reverse);
}

void test_rs_turn_left()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(0.0, 2.0, M_PI / 2.0);
    auto path = planner.get_path(start, goal);
    rs_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_rs_turn_left);
}

void test_rs_turn_right()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(0.0, -2.0, -M_PI / 2.0);
    auto path = planner.get_path(start, goal);
    rs_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_rs_turn_right);
}

void test_rs_reverse_with_turn()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(-2.0, 2.0, M_PI);
    auto path = planner.get_path(start, goal);
    rs_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_rs_reverse_with_turn);
}

void test_rs_complex_maneuver()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 2.0, M_PI / 4.0);
    auto path = planner.get_path(start, goal);
    rs_path_ends_near(start, goal, path, 1e-4, 1e-4);
    TEST_END(test_rs_complex_maneuver);
}

void test_rs_get_controls_nonempty()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto controls = planner.get_controls(start, goal);
    ASSERT_TRUE(!controls.empty());
    TEST_END(test_rs_get_controls_nonempty);
}

void test_rs_get_all_controls_nonempty()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto all_controls = planner.get_all_controls(start, goal);
    ASSERT_TRUE(!all_controls.empty());
    TEST_END(test_rs_get_all_controls_nonempty);
}

void test_rs_get_distance_positive()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    double dist = planner.get_distance(start, goal);
    ASSERT_TRUE(dist > 0.0);
    TEST_END(test_rs_get_distance_positive);
}

void test_rs_discretization()
{
    ReedsSheppStateSpace planner(1.0, 0.1);
    ASSERT_NEAR(planner.discretization(), 0.1, 1e-10);
    TEST_END(test_rs_discretization);
}

void test_rs_various_orientations()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);

    for (double angle : {0.0, M_PI / 4.0, M_PI / 2.0})
    {
        State goal(4.0, 0.0, angle);
        auto path = planner.get_path(start, goal);
        if (!path.empty())
        {
            rs_path_ends_near(start, goal, path, 1e-2, 1e-2);
        }
    }
    TEST_END(test_rs_various_orientations);
}

void test_rs_integrate_empty_controls()
{
    ReedsSheppStateSpace planner(1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    std::vector<Control> controls;
    auto path = planner.integrate(start, controls);
    ASSERT_TRUE(path.empty());
    TEST_END(test_rs_integrate_empty_controls);
}

// ==================== SteeringPath Tests ====================

void test_planner_new_dubins()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    ASSERT_EQ(p.path_type(), PathType::Dubins);
    ASSERT_NEAR(p.kappa_max(), 1.0, 1e-10);
    ASSERT_NEAR(p.discretization(), 0.05, 1e-10);
    TEST_END(test_planner_new_dubins);
}

void test_planner_new_rs()
{
    SteeringPath p(PathType::Rs, 1.0, 0.05);
    ASSERT_EQ(p.path_type(), PathType::Rs);
    TEST_END(test_planner_new_rs);
}

void test_planner_set_kappa_max_valid()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    p.set_kappa_max(2.0);
    ASSERT_NEAR(p.kappa_max(), 2.0, 1e-10);
    TEST_END(test_planner_set_kappa_max_valid);
}

void test_planner_set_discretization_valid()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    p.set_discretization(0.1);
    ASSERT_NEAR(p.discretization(), 0.1, 1e-10);
    TEST_END(test_planner_set_discretization_valid);
}

void test_planner_set_path_type()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    p.set_path_type(PathType::Rs);
    ASSERT_EQ(p.path_type(), PathType::Rs);
    TEST_END(test_planner_set_path_type);
}

void test_planner_dubins_compute_path()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto path = p.compute_shortest_path(start, goal);
    ASSERT_TRUE(!path.empty());
    const State& end = path.back();
    ASSERT_TRUE(std::fabs(end.x - goal.x) < 1e-3);
    ASSERT_TRUE(std::fabs(end.y - goal.y) < 1e-3);
    TEST_END(test_planner_dubins_compute_path);
}

void test_planner_rs_compute_path()
{
    SteeringPath p(PathType::Rs, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto path = p.compute_shortest_path(start, goal);
    ASSERT_TRUE(!path.empty());
    const State& end = path.back();
    ASSERT_TRUE(std::fabs(end.x - goal.x) < 1e-3);
    ASSERT_TRUE(std::fabs(end.y - goal.y) < 1e-3);
    TEST_END(test_planner_rs_compute_path);
}

void test_planner_dubins_compute_controls()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto controls = p.compute_shortest_control_sequence(start, goal);
    ASSERT_TRUE(!controls.empty());
    TEST_END(test_planner_dubins_compute_controls);
}

void test_planner_rs_compute_controls()
{
    SteeringPath p(PathType::Rs, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto controls = p.compute_shortest_control_sequence(start, goal);
    ASSERT_TRUE(!controls.empty());
    TEST_END(test_planner_rs_compute_controls);
}

void test_planner_dubins_compute_all_paths()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto all_paths = p.compute_all_paths(start, goal);
    ASSERT_TRUE(!all_paths.empty());
    TEST_END(test_planner_dubins_compute_all_paths);
}

void test_planner_rs_compute_all_paths()
{
    SteeringPath p(PathType::Rs, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto all_paths = p.compute_all_paths(start, goal);
    ASSERT_TRUE(!all_paths.empty());
    TEST_END(test_planner_rs_compute_all_paths);
}

void test_planner_dubins_compute_all_controls()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto all_controls = p.compute_all_control_sequences(start, goal);
    ASSERT_TRUE(!all_controls.empty());
    TEST_END(test_planner_dubins_compute_all_controls);
}

void test_planner_rs_compute_all_controls()
{
    SteeringPath p(PathType::Rs, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);
    State goal(3.0, 3.0, M_PI / 4.0);
    auto all_controls = p.compute_all_control_sequences(start, goal);
    ASSERT_TRUE(!all_controls.empty());
    TEST_END(test_planner_rs_compute_all_controls);
}

void test_planner_various_orientations_dubins()
{
    SteeringPath p(PathType::Dubins, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);

    for (double angle : {0.0, M_PI / 4.0, M_PI / 2.0})
    {
        State goal(4.0, 0.0, angle);
        auto path = p.compute_shortest_path(start, goal);
        if (!path.empty())
        {
            const State& end = path.back();
            ASSERT_TRUE(std::fabs(end.x - goal.x) < 1e-2);
            ASSERT_TRUE(std::fabs(end.y - goal.y) < 1e-2);
        }
    }
    TEST_END(test_planner_various_orientations_dubins);
}

void test_planner_various_orientations_rs()
{
    SteeringPath p(PathType::Rs, 1.0, 0.05);
    State start(0.0, 0.0, 0.0);

    for (double angle : {0.0, M_PI / 4.0, M_PI / 2.0})
    {
        State goal(4.0, 0.0, angle);
        auto path = p.compute_shortest_path(start, goal);
        if (!path.empty())
        {
            const State& end = path.back();
            ASSERT_TRUE(std::fabs(end.x - goal.x) < 1e-2);
            ASSERT_TRUE(std::fabs(end.y - goal.y) < 1e-2);
        }
    }
    TEST_END(test_planner_various_orientations_rs);
}

// ==================== Main ====================

int main()
{
    std::cout << "=== Math Utils Tests ===" << std::endl;
    test_sgn_positive();
    test_sgn_negative();
    test_sgn_zero();
    test_polar_origin();
    test_polar_positive_x();
    test_polar_positive_y();
    test_polar_quadrant1();
    test_twopify_zero();
    test_twopify_positive();
    test_twopify_large();
    test_twopify_negative();
    test_pify_zero();
    test_pify_positive_pi();
    test_pify_negative_pi();
    test_pify_beyond_pi();
    test_pify_negative_beyond();
    test_fresnel_zero();
    test_fresnel_positive();
    test_fresnel_negative();
    test_fresnel_large();
    test_end_of_straight_line_forward();
    test_end_of_straight_line_backward();
    test_end_of_straight_line_angle();
    test_end_of_circular_arc_zero_length();
    test_end_of_circular_arc_half_circle();
    test_end_of_clothoid_zero_length();
    test_end_of_clothoid_kappa_change();

    std::cout << "\n=== State/Control Tests ===" << std::endl;
    test_state_default();
    test_state_nearly_equal_same();
    test_state_nearly_equal_small_diff();
    test_state_nearly_equal_large_diff();
    test_state_eq();
    test_state_eq_with_tolerance();
    test_control_default();
    test_control_new();

    std::cout << "\n=== Dubins Tests ===" << std::endl;
    test_dubins_forward_straight();
    test_dubins_forward_turn();
    test_dubins_reverse_reaches_goal();
    test_dubins_forward_or_reverse();
    test_dubins_path_length_positive();
    test_dubins_get_controls_nonempty();
    test_dubins_get_all_controls_nonempty();
    test_dubins_discretization();
    test_dubins_various_orientations();
    test_dubins_integrate_empty_controls();

    std::cout << "\n=== Reeds-Shepp Tests ===" << std::endl;
    test_rs_straight_forward();
    test_rs_straight_reverse();
    test_rs_turn_left();
    test_rs_turn_right();
    test_rs_reverse_with_turn();
    test_rs_complex_maneuver();
    test_rs_get_controls_nonempty();
    test_rs_get_all_controls_nonempty();
    test_rs_get_distance_positive();
    test_rs_discretization();
    test_rs_various_orientations();
    test_rs_integrate_empty_controls();

    std::cout << "\n=== SteeringPath Tests ===" << std::endl;
    test_planner_new_dubins();
    test_planner_new_rs();
    test_planner_set_kappa_max_valid();
    test_planner_set_discretization_valid();
    test_planner_set_path_type();
    test_planner_dubins_compute_path();
    test_planner_rs_compute_path();
    test_planner_dubins_compute_controls();
    test_planner_rs_compute_controls();
    test_planner_dubins_compute_all_paths();
    test_planner_rs_compute_all_paths();
    test_planner_dubins_compute_all_controls();
    test_planner_rs_compute_all_controls();
    test_planner_various_orientations_dubins();
    test_planner_various_orientations_rs();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total: " << passed + failed << " tests, " << passed << " passed, " << failed << " failed" << std::endl;

    if (failed > 0)
    {
        std::cerr << "\nSome tests FAILED!" << std::endl;
        return 1;
    }

    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
