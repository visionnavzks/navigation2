
import math
import numpy as np
from scipy.optimize import least_squares
from zhengli.kinematic_smoother.esdf_map import ESDFMap

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def angle_diff(a, b):
    d = a - b
    return normalize_angle(d)

class KinematicSmoother:
    def __init__(self, 
                 robot_params={'length': 4.0, 'width': 2.0},
                 esdf_map=None,
                 w_model=10.0, 
                 w_smooth=10.0, 
                 w_obs=20.0,
                 w_danger=10.0,
                 d_safe=0.1,
                 w_s=1.0, 
                 ref_weight=1.0,
                 w_fix=100.0,
                 target_spacing=0.2, 
                 max_kappa=0.5,
                 max_iter=50):
        """
        w_model: Weight for kinematic consistency.
        w_smooth: Weight for minimizing Jerk (d_kappa).
        w_obs: Weight for obstacle cost (Collision).
        w_danger: Weight for obstacle cost (Danger zone).
        d_safe: Safe distance margin.
        w_s: Weight for spacing regularization.
        w_fix: Weight for hard boundary constraints.
        ref_weight: Weight to stay close to original path (optional).
        robot_params: dict with 'length' and 'width' of the footprint.
        esdf_map: Instance of ESDFMap.
        target_spacing: Desired distance between points.
        max_kappa: Hard limit for curvature (1/R).
        """
        self.w_model = w_model
        self.w_smooth = w_smooth
        self.w_obs = w_obs
        self.w_danger = w_danger
        self.d_safe = d_safe
        self.w_s = w_s
        self.w_fix = w_fix
        self.ref_weight = ref_weight
        self.target_spacing = target_spacing
        self.max_kappa = max_kappa
        self.max_iter = max_iter
        
        self.esdf_map = esdf_map
        
        # Initialize Multi-Circle Decomposition
        self._init_circle_decomposition(robot_params.get('length', 4.0), 
                                        robot_params.get('width', 2.0),
                                        robot_params.get('center_x_offset', 0.0))

    def _init_circle_decomposition(self, length, width, center_offset):
        # Heuristic: use N circles to cover the length
        # Radius R = width / 2 * slightly less to be safe or slightly more?
        # Standard conservative: R = sqrt((L/2)^2 + (W/2)^2) is one big circle.
        # Decomposition:
        # We want circles with radius r roughly equal to width/2 (or slightly larger).
        # Number of circles n_circles approx Length / (2*r) * overla_factor
        
        # Let's use simple covering:
        self.rob_width = width
        self.rob_length = length
        self.center_offset = center_offset
        
        # Radius of covering circles.
        # Setting R = width / 2 makes sense for tight fits width-wise.
        self.circle_radius = width / 2.0
        
        # Distance between circle centers.
        # Geometric center is at (center_offset, 0) in base frame.
        # Robot extends from [center_offset - L/2, center_offset + L/2].
        
        if self.circle_radius < 1e-3: 
             self.circle_offsets = np.array([center_offset])
             self.n_circles = 1
             return

        # Simple approach: Place circles such that they touch or overlap to cover L.
        # We distribute centers from x_min + r to x_max - r.
        
        min_x = center_offset - length / 2.0
        max_x = center_offset + length / 2.0
        
        start_x = min_x + self.circle_radius
        end_x = max_x - self.circle_radius
        
        if start_x > end_x: # Length < width case or Length approx Width
            self.circle_offsets = np.array([center_offset])
        else:
            # How many circles? 
            # Distribute roughly every R?
            # Range size = end_x - start_x.
            # We want gaps to be small. 
            num_circles = int(np.ceil((end_x - start_x) / self.circle_radius)) + 1
            if num_circles < 2: num_circles = 2
            
            self.circle_offsets = np.linspace(start_x, end_x, num_circles)
        
        self.n_circles = len(self.circle_offsets)
        # print(f"[KinematicSmoother] Robot {length}x{width} offset={center_offset} decomposed into {self.n_circles} circles")

    def optimize(self, raw_path, gear_directions=None, return_result=False, verbose=1):
        """
        raw_path: (N, 2) or (N, 3) [x, y, (theta)]
        gear_directions: (N-1,) 1 or -1.
        return_result: if True, return the full scipy OptimizeResult.
        verbose: scipy least_squares verbosity.
        """
        raw_path = np.array(raw_path)
        N_orig = len(raw_path)
        if N_orig < 2: return raw_path
        
        if gear_directions is None:
            gear_directions = np.ones(N_orig - 1)
        
        # 1. Preprocess: Inject cusps
        processed_path = [raw_path[0]]
        processed_gears = []
        is_cusp_segment = []
        orig_indices_map = [0]
        
        for i in range(N_orig - 1):
            curr_gear = gear_directions[i]
            next_gear = gear_directions[i+1] if i + 1 < len(gear_directions) else curr_gear
            
            # Segment
            processed_gears.append(curr_gear)
            is_cusp_segment.append(False)
            processed_path.append(raw_path[i+1])
            orig_indices_map.append(i+1)
            
            # Cusp handling
            if i < N_orig - 2 and curr_gear != next_gear:
                # Add duplicate point for cusp
                processed_gears.append(0) # 0 indicates transition? Or just placeholder.
                is_cusp_segment.append(True)
                processed_path.append(raw_path[i+1])
                orig_indices_map.append(i+1)
        
        processed_path = np.array(processed_path)
        processed_gears = np.array(processed_gears)
        is_cusp_segment = np.array(is_cusp_segment, dtype=bool)
        N = len(processed_path)
        
        # 2. Initial Guess
        x_init = processed_path[:, 0]
        y_init = processed_path[:, 1]
        
        theta_init = np.zeros(N)
        # Compute headings
        for i in range(N-1):
            dx = processed_path[i+1,0] - processed_path[i,0]
            dy = processed_path[i+1,1] - processed_path[i,1]
            if is_cusp_segment[i]:
                 theta_init[i] = theta_init[i-1] if i > 0 else 0
            else:
                 norm = np.hypot(dx, dy)
                 if norm > 1e-6:
                     th = np.arctan2(dy, dx)
                     if processed_gears[i] < 0: th += np.pi
                     theta_init[i] = normalize_angle(th)
                 else:
                     theta_init[i] = theta_init[i-1] if i>0 else 0
        theta_init[-1] = theta_init[-2]
        
        if raw_path.shape[1] >= 3:
            # If input has orientation, guide the start/end
            theta_init[0] = raw_path[0, 2] # Force start yaw
            # For intermediate, we trust geometry unless explicit?
            # Let's trust geometry for smooth start.
            pass

        kappa_init = np.zeros(N)
        ds_init = np.zeros(N)
        for i in range(N-1):
            if is_cusp_segment[i]:
                ds_init[i] = 0.0
            else:
                ds_init[i] = np.hypot(processed_path[i+1,0]-processed_path[i,0], 
                                      processed_path[i+1,1]-processed_path[i,1])
        
        # Flatten state: [x0, y0, th0, k0, ds0, x1, ...]
        initial_guess = np.column_stack((x_init, y_init, theta_init, kappa_init, ds_init)).flatten()
        
        # Boundary Values
        start_pose = np.zeros(3)
        start_pose[:2] = processed_path[0, :2]
        start_pose[2] = theta_init[0] if raw_path.shape[1] < 3 else raw_path[0, 2]
        
        end_pose = np.zeros(3)
        end_pose[:2] = processed_path[-1, :2]
        end_pose[2] = theta_init[-1] if raw_path.shape[1] < 3 else raw_path[-1, 2]
        
        # Bounds
        low = np.full(5*N, -np.inf)
        up = np.full(5*N, np.inf)
        
        # Apply ds >= 0
        ds_indices = np.arange(4, 5*N, 5)
        low[ds_indices] = 0.0 
        
        # Apply max_kappa constraints
        kappa_indices = np.arange(3, 5*N, 5)
        low[kappa_indices] = -self.max_kappa
        up[kappa_indices] = self.max_kappa

        # 3. Optimize
        res = least_squares(
            self._residuals,
            initial_guess,
            bounds=(low, up),
            args=(processed_path, processed_gears, is_cusp_segment, start_pose, end_pose),
            verbose=verbose,
            max_nfev=self.max_iter,
            # Explicitly set x_scale to make variables more uniform
            x_scale='jac'
        )

        if return_result:
            return res

        return res.x.reshape((N, 5))

    def _residuals(self, vars, ref_path, gears, is_cusp, start_pose, end_pose):
        N = len(ref_path)
        state = vars.reshape((N, 5))
        
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        kappa = state[:, 3]
        ds = state[:, 4]
        
        res = []
        
        # A. Kinematic Model
        res.extend(self._kinematic_residuals(x, y, theta, kappa, ds, gears, is_cusp))
        
        # B. Smoothness
        res.extend(self._smoothness_residuals(kappa, ds, is_cusp))
        
        # C. Obstacles (ESDF)
        if self.esdf_map is not None:
             res.extend(self._obstacle_residuals(x, y, theta))
        
        # D. Spacing / Regularization
        res.extend(self._spacing_residuals(ds, is_cusp))
        
        # E. Boundary & Ref
        res.extend(self._boundary_residuals(x, y, theta, ds, start_pose, end_pose))
        
        # Optional: Reference deviation (soft)
        if self.ref_weight > 1e-5:
             dist_err = np.hypot(x - ref_path[:,0], y - ref_path[:,1])
             res.extend((self.ref_weight * dist_err).tolist())

        return np.array(res)

    def _kinematic_residuals(self, x, y, theta, kappa, ds, gears, is_cusp):
        res = []
        N = len(x)
        for i in range(N - 1):
            if is_cusp[i]:
                # Force continuity at cusp (ds=0 usually, but positions must match)
                res.append(self.w_fix * (x[i+1] - x[i]))
                res.append(self.w_fix * (y[i+1] - y[i]))
                res.append(self.w_fix * angle_diff(theta[i+1], theta[i]))
            else:
                d = 1.0 if gears[i] >= 0 else -1.0
                step = ds[i]
                k = kappa[i]
                k_next = kappa[i+1]
                
                # Traperzoidal / Midpoint for theta
                th_pred = theta[i] + d * step * (k + k_next) * 0.5
                
                # Midpoint for position
                th_mid = theta[i] + d * step * k * 0.5
                x_pred = x[i] + d * step * np.cos(th_mid)
                y_pred = y[i] + d * step * np.sin(th_mid)
                
                res.append(self.w_model * (x[i+1] - x_pred))
                res.append(self.w_model * (y[i+1] - y_pred))
                res.append(self.w_model * angle_diff(theta[i+1], th_pred))
        return res

    def _smoothness_residuals(self, kappa, ds, is_cusp):
        res = []
        N = len(kappa)
        for i in range(N - 1):
            if not is_cusp[i]:
                # Minimize change in curvature (Jerky)
                # dK/ds ~ (K_next - K_curr) / ds
                # Optimization usually likes constant stepping.
                # Using (k2 - k1) is simple. 
                # If ds varies, maybe (k2-k1)/sqrt(ds) for energy?
                denom = np.sqrt(ds[i]) if ds[i] > 1e-3 else 0.03 # avoid div 0
                val = (kappa[i+1] - kappa[i]) / denom
                res.append(self.w_smooth * val)
        return res

    def _obstacle_residuals(self, x, y, theta):
        # res = []
        # Vectorized implementation
        N = len(x)
        M = self.n_circles
        
        # Prepare all circle centers
        # x shape: (N,)
        # offsets shape: (M,)
        # We want result shape (N*M,) or processing all N*M points.
        
        # Using broadcasting
        # Centers X: x[:, None] + offsets[None, :] * cos(theta)[:, None]
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        
        # (N, M)
        cx = x[:, np.newaxis] + self.circle_offsets[np.newaxis, :] * cos_th[:, np.newaxis]
        cy = y[:, np.newaxis] + self.circle_offsets[np.newaxis, :] * sin_th[:, np.newaxis]
        
        # Flatten to query (N*M,)
        cx_flat = cx.flatten()
        cy_flat = cy.flatten()
        
        # Batch Query ESDF
        dists, _ = self.esdf_map.get_distance_and_gradient(cx_flat, cy_flat)
        
        # Helper: Hinge Loss is max(0, radius - dist)
        # Note: If dist is negative (inside obstacle), surf_dist is even more negative.
        # Wait, ESDF definition: Inside is Negative. 
        # Collision if dist < radius.
        
        # dists is distance to obstacle boundary.
        # Surface distance = dists - radius.
        # Collision if Surface distance < 0.
        # Cost = -Surface Distance = radius - dists.
        
        surf_dists = dists - self.circle_radius
        
        # 3-Stage Cost Function:
        # 1. d > d_safe: 0
        # 2. 0 < d <= d_safe: w_danger * (d - d_safe)
        # 3. d <= 0: w_collision * (d - d_safe) (using w_obs as w_collision)
        
        weights = np.zeros_like(surf_dists)
        
        # Mask for Danger Zone (approaching wall)
        mask_danger = (surf_dists > 0) & (surf_dists <= self.d_safe)
        weights[mask_danger] = self.w_danger
        
        # Mask for Collision (hit wall)
        mask_collision = surf_dists <= 0
        weights[mask_collision] = self.w_obs
        
        # Calculate weighted residuals
        # The residual is weight * (d - d_safe). 
        # Since d <= d_safe in active regions, (d - d_safe) is negative. 
        # least_squares minimizes sum(res^2), so sign doesn't matter for magnitude.
        costs = weights * (surf_dists - self.d_safe)
        
        # Only return non-zero terms to save computation? 
        # No, structure structure must be consistent? 
        # optimize handles zeros fine.
        
        return costs.tolist()

    def _spacing_residuals(self, ds, is_cusp):
        res = []
        scale = max(self.target_spacing, 1e-4)
        for i in range(len(ds) - 1):
            if is_cusp[i]:
                # Cusp spacing should be 0
                res.append(self.w_s * 10.0 * ds[i]) 
            else:
                # Regular limits
                res.append(self.w_s * (ds[i] - self.target_spacing) / scale)
        return res

    def _boundary_residuals(self, x, y, theta, ds, start, end):
        res = []
        # Start
        res.append(self.w_fix * (x[0] - start[0]))
        res.append(self.w_fix * (y[0] - start[1]))
        res.append(self.w_fix * angle_diff(theta[0], start[2]))
        
        # End
        res.append(self.w_fix * (x[-1] - end[0]))
        res.append(self.w_fix * (y[-1] - end[1]))
        res.append(self.w_fix * angle_diff(theta[-1], end[2]))
        
        # Constrain last ds to 0 (unused control input)
        # This prevents it from drifting to large values
        res.append(self.w_fix * ds[-1])
        
        return res
