import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from dubins_curve import DubinsPlanner, Command

def generate_reference_path(start_x=0.0, start_y=0.0, start_theta=0.0, 
                            goal_x=20.0, goal_y=0.0, goal_theta=0.0, target_ds=0.3, turning_radius=5.0):
    """Generates a Reeds-Shepp reference path between start and goal states"""
    planner = DubinsPlanner(turning_radius=turning_radius)
    result = planner.plan(start_x, start_y, start_theta, goal_x, goal_y, goal_theta)
    
    if False:
        x_ref, y_ref, theta_ref, gears = result.best_path.generate_trajectory(
            start_x, start_y, start_theta, step_size=target_ds
        )
        return np.array(x_ref), np.array(y_ref), np.unwrap(np.array(theta_ref)), np.array(gears), result.best_commands

    # Fallback to straight line if planning fails
    dist = np.hypot(goal_x - start_x, goal_y - start_y)
    steps = max(int(dist / target_ds), 1)
    x_ref = np.linspace(start_x, goal_x, steps + 1)
    y_ref = np.linspace(start_y, goal_y, steps + 1)
    theta_ref = np.linspace(start_theta, goal_theta, steps + 1)
    gears = np.ones(steps)
    commands = [Command(dist, 0.0)]
    
    return x_ref, y_ref, np.unwrap(theta_ref), gears, commands


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
        self.kappa_start = self.params.get('kappa_start', 0.0)  # Start curvature (float or None)
        
        # Solver Parameters
        self.ipopt_max_iter = int(self.params.get('ipopt_max_iter', 500))
        self.ipopt_tol = float(self.params.get('ipopt_tol', 1e-6))
        self.ipopt_print_level = int(self.params.get('ipopt_print_level', 0))

    def solve(self, x_ref, y_ref, theta_ref, gears):
        """Run nonlinear mathematical optimization on given reference path"""
        # Ensure angles are unwrapped for continuity in optimization
        theta_ref = np.unwrap(theta_ref)
        
        # Augment path to support cusps: detect gear shifts and insert an extra point 
        # to create a zero-length "virtual" segment at the cusp.
        aug_x_ref = [x_ref[0]]
        aug_y_ref = [y_ref[0]]
        aug_theta_ref = [theta_ref[0]]
        aug_gears = []
        is_virtual = [] # True if segment i -> i+1 is a virtual jump (zero-length)
        
        for i in range(len(x_ref) - 1):
            if i > 0 and gears[i] != gears[i-1]:
                # Cusp found at current index i. Insert virtual transition.
                aug_x_ref.append(x_ref[i])
                aug_y_ref.append(y_ref[i])
                aug_theta_ref.append(theta_ref[i])
                aug_gears.append(gears[i])
                is_virtual.append(True)
                
            # Add regular segment i -> i+1
            aug_x_ref.append(x_ref[i+1])
            aug_y_ref.append(y_ref[i+1])
            aug_theta_ref.append(theta_ref[i+1])
            aug_gears.append(gears[i])
            is_virtual.append(False)

        x_ref = np.array(aug_x_ref)
        y_ref = np.array(aug_y_ref)
        theta_ref = np.array(aug_theta_ref)
        gears = np.array(aug_gears)
        N = len(x_ref)

        if self.target_ds > 0.01:
            target_ds_mag = self.target_ds
        else:
            # Estimate DS from non-virtual segments
            diffs = np.linalg.norm(np.diff(np.c_[x_ref, y_ref], axis=0), axis=1)
            target_ds_mag = np.mean(diffs[diffs > 1e-4])
        
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
            if is_virtual[i]:
                # No smoothness or step-size cost for virtual jumps
                continue
            # Use positive ds for weighting the smoothness term
            obj += self.w_dkappa * dkappa[i]**2 * ds[i]
            # Match the target step size (always positive now)
            obj += self.w_ds * (ds[i] - target_ds_mag)**2

        opti.minimize(obj)

        # --- Kinematic Constraints (Single-track bicycle model) ---
        for i in range(N-1):
            if is_virtual[i]:
                # Virtual transition: same pose, decoupled kappa
                opti.subject_to(x[i+1] == x[i])
                opti.subject_to(y[i+1] == y[i])
                opti.subject_to(theta[i+1] == theta[i])
                opti.subject_to(ds[i] == 0)
                # Note: kappa[i+1] == kappa_next is NOT enforced here, allowing the jump
                continue

            # Exact integration for curvature and heading (assuming piecewise constant dkappa)
            kappa_next = kappa[i] + ds[i] * dkappa[i]
            theta_next = theta[i] + gears[i] * (ds[i] * kappa[i] + 0.5 * ds[i]**2 * dkappa[i])

            # Simpson's rule for x and y integration (better approximation of Fresnel integrals)
            theta_mid = theta[i] + gears[i] * (0.5 * ds[i] * kappa[i] + 0.125 * ds[i]**2 * dkappa[i])
            
            x_next = x[i] + gears[i] * (ds[i] / 6.0) * (ca.cos(theta[i]) + 4.0 * ca.cos(theta_mid) + ca.cos(theta_next))
            y_next = y[i] + gears[i] * (ds[i] / 6.0) * (ca.sin(theta[i]) + 4.0 * ca.sin(theta_mid) + ca.sin(theta_next))

            opti.subject_to(x[i+1] == x_next)
            opti.subject_to(y[i+1] == y_next)
            opti.subject_to(theta[i+1] == theta_next)
            opti.subject_to(kappa[i+1] == kappa_next)

        # --- State and Control Bounds ---
        opti.subject_to(opti.bounded(-self.max_kappa, kappa, self.max_kappa))
        
        # Add topology constraints: ds must be strictly positive and within bounds
        for i in range(N-1):
            if not is_virtual[i]:
                # Constraint ds to be within [0.05, 2.0] times the target step size
                opti.subject_to(ds[i] >= 0.05 * target_ds_mag)
                opti.subject_to(ds[i] <= 2.0 * target_ds_mag)

        # --- Boundary Conditions ---
        # Start point
        opti.subject_to(x[0] == x_ref[0])
        opti.subject_to(y[0] == y_ref[0])
        opti.subject_to(theta[0] == theta_ref[0])
        if self.kappa_start is not None:
            opti.subject_to(kappa[0] == self.kappa_start)

        # End point
        opti.subject_to(x[N-1] == x_ref[N-1])
        opti.subject_to(y[N-1] == y_ref[N-1])
        opti.subject_to(theta[N-1] == theta_ref[N-1])


        # --- Initial Guess ---
        opti.set_initial(x, x_ref)
        opti.set_initial(y, y_ref)
        opti.set_initial(theta, theta_ref)
        opti.set_initial(kappa, np.zeros(N))
        
        # Initial guess for ds needs to handle virtual segments
        initial_ds = np.ones(N-1) * target_ds_mag
        for i in range(N-1):
            if is_virtual[i]:
                initial_ds[i] = 0.0
        opti.set_initial(ds, initial_ds)
        opti.set_initial(dkappa, np.zeros(N-1))

        # --- Setup Solver ---
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": self.ipopt_max_iter, 
            "tol": self.ipopt_tol,
            "print_level": self.ipopt_print_level
        }
        opti.solver("ipopt", p_opts, s_opts)

        # --- Solve ---
        start_time = time.time()
        try:
            sol = opti.solve()
            solve_time = (time.time() - start_time) * 1000.0 # ms
            print(f"Optimization successfully formulated and solved in {solve_time:.2f}ms")
            # Return optimized path and re-apply sign to ds for compatibility
            ds_val = sol.value(ds)
            signed_ds = ds_val * gears
            gears_opt = np.sign(signed_ds)
            
            return (sol.value(x), sol.value(y), sol.value(theta), 
                    sol.value(kappa), signed_ds, sol.value(dkappa), gears_opt, solve_time)
        except Exception as e:
            solve_time = (time.time() - start_time) * 1000.0 # ms
            print(f"Optimization failed after {solve_time:.2f}ms: {e}")
            return (opti.debug.value(x), opti.debug.value(y), 
                    opti.debug.value(theta), None, None, None, None, solve_time)
