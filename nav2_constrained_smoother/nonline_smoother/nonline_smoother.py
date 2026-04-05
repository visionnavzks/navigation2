import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from rs_curve import ReedsSheppPlanner

def generate_reference_path(start_x=0.0, start_y=0.0, start_theta=0.0, 
                            goal_x=20.0, goal_y=0.0, goal_theta=0.0, target_ds=0.3, turning_radius=5.0):
    """Generates a Reeds-Shepp reference path between start and goal states"""
    planner = ReedsSheppPlanner(turning_radius=turning_radius)
    result = planner.plan(start_x, start_y, start_theta, goal_x, goal_y, goal_theta)
    
    if result.best_path:
        x_ref, y_ref, theta_ref, dir_ref = result.best_path.generate_trajectory(
            start_x, start_y, start_theta, step_size=target_ds
        )
        return np.array(x_ref), np.array(y_ref), np.unwrap(np.array(theta_ref)), np.array(dir_ref)
    else:
        raise ValueError("Failed to generate Reeds-Shepp path")
        # Fallback to linear interpolation (ensure theta endpoints match)
        N = max(int(np.hypot(goal_x - start_x, goal_y - start_y) / target_ds), 20)
        x_ref = np.linspace(start_x, goal_x, N)
        y_ref = np.linspace(start_y, goal_y, N)
        # Interpolate orientation between endpoints
        theta_ref = np.linspace(start_theta, goal_theta, N)
        dir_ref = np.ones(N)
        return x_ref, y_ref, np.unwrap(theta_ref), dir_ref


class NonlinearPathSmoother:
    def __init__(self, params=None):
        """Initialize the smoother with given parameters or defaults"""
        if params is None:
            params = {}
        self.params = params
        self._load_parameters()
        
    def _load_parameters(self):
        """Extract algorithm parameters into class attributes"""
        self.max_kappa = self.params.get('max_kappa', 0.5)      # Maximum curvature
        self.w_ref = self.params.get('w_ref', 10.0)             # Weight for reference tracking
        self.w_dkappa = self.params.get('w_dkappa', 10.0)       # Weight for subjective smoothness
        self.w_kappa = self.params.get('w_kappa', 0.1)          # Weight for curvature
        self.w_ds = self.params.get('w_ds', 1.0)                # Weight for step size (uniformity)
        self.target_ds = self.params.get('target_ds', 0.0)      # Desired spacing (0.0 means auto)

    def solve(self, x_ref, y_ref, theta_ref, dir_ref):
        """Run nonlinear mathematical optimization on given reference path"""
        N = len(x_ref)
        if self.target_ds > 0.01:
            target_ds_mag = self.target_ds
        else:
            target_ds_mag = np.mean(np.linalg.norm(np.diff(np.c_[x_ref, y_ref], axis=0), axis=1))
        
        # --- Setup Opti stack ---
        opti = ca.Opti()

        # --- Decision Variables ---
        X = opti.variable(4, N)
        x     = X[0, :]
        y     = X[1, :]
        theta = X[2, :]
        kappa = X[3, :]

        U = opti.variable(2, N-1)
        ds     = U[0, :]
        dkappa = U[1, :]

        # --- Objective Function ---
        obj = 0
        for i in range(N):
            obj += self.w_ref * ((x[i] - x_ref[i])**2 + (y[i] - y_ref[i])**2)
            # Use w_ref for theta error to keep heading aligned. 1-cos handles angle wrapping.
            obj += self.w_ref * (1.0 - ca.cos(theta[i] - theta_ref[i]))
            obj += self.w_kappa * kappa[i]**2
            
        for i in range(N-1):
            # Use absolute value for weighting the smoothness term to ensure positive cost
            obj += self.w_dkappa * dkappa[i]**2 * ca.fabs(ds[i])
            # Match the signed target step size based on reference direction
            # obj += self.w_ds * (ds[i] - target_ds_mag * dir_ref[i])**2
            obj += self.w_ds * (ds[i])**2

        opti.minimize(obj)

        # --- Kinematic Constraints (Single-track bicycle model) ---
        for i in range(N-1):
            # Exact integration for curvature and heading (assuming piecewise constant dkappa)
            kappa_next = kappa[i] + ds[i] * dkappa[i]
            theta_next = theta[i] + ds[i] * kappa[i] + 0.5 * ds[i]**2 * dkappa[i]

            # Simpson's rule for x and y integration (better approximation of Fresnel integrals)
            theta_mid = theta[i] + 0.5 * ds[i] * kappa[i] + 0.125 * ds[i]**2 * dkappa[i]
            
            x_next = x[i] + (ds[i] / 6.0) * (ca.cos(theta[i]) + 4.0 * ca.cos(theta_mid) + ca.cos(theta_next))
            y_next = y[i] + (ds[i] / 6.0) * (ca.sin(theta[i]) + 4.0 * ca.sin(theta_mid) + ca.sin(theta_next))

            opti.subject_to(x[i+1] == x_next)
            opti.subject_to(y[i+1] == y_next)
            opti.subject_to(theta[i+1] == theta_next)
            opti.subject_to(kappa[i+1] == kappa_next)

        # --- State and Control Bounds ---
        opti.subject_to(opti.bounded(-self.max_kappa, kappa, self.max_kappa))
        
        # Add topology constraints: ds must maintain its sign and stay within reasonable bounds
        for i in range(N-1):
            # Constraint ds to be within [0.05, 2.0] times the target step size with correct direction
            # Tightened upper bound to prevent looping
            opti.subject_to(ds[i] * dir_ref[i] >= 0.05 * target_ds_mag)
            opti.subject_to(ds[i] * dir_ref[i] <= 2.0 * target_ds_mag)

        # --- Boundary Conditions ---
        # Start point
        opti.subject_to(x[0] == x_ref[0])
        opti.subject_to(y[0] == y_ref[0])
        opti.subject_to(theta[0] == theta_ref[0])
        opti.subject_to(kappa[0] == 0.0)

        # End point
        opti.subject_to(x[N-1] == x_ref[N-1])
        opti.subject_to(y[N-1] == y_ref[N-1])
        opti.subject_to(theta[N-1] == theta_ref[N-1])


        # --- Initial Guess ---
        opti.set_initial(x, x_ref)
        opti.set_initial(y, y_ref)
        opti.set_initial(theta, theta_ref)
        opti.set_initial(kappa, np.zeros(N))
        # Important: Initialize ds with the correct sign!
        opti.set_initial(ds, dir_ref[:-1] * target_ds_mag)
        opti.set_initial(dkappa, np.zeros(N-1))

        # --- Setup Solver ---
        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0}
        opti.solver("ipopt", p_opts, s_opts)

        # --- Solve ---
        try:
            sol = opti.solve()
            print("Optimization successfully formulated and solved!")
            # Return optimized path and directions (sign of ds)
            ds_val = sol.value(ds)
            dir_opt = np.ones(N)
            dir_opt[:-1] = np.sign(ds_val)
            dir_opt[-1] = dir_opt[-2] if N > 1 else 1
            
            return (sol.value(x), sol.value(y), sol.value(theta), 
                    sol.value(kappa), ds_val, sol.value(dkappa), dir_opt)
        except Exception as e:
            print(f"Optimization failed: {e}")
            return (opti.debug.value(x), opti.debug.value(y), 
                    opti.debug.value(theta), None, None, None, None)
