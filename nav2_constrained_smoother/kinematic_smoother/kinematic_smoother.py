
import numpy as np
from scipy.optimize import least_squares
import math

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def angle_diff(a, b):
    d = a - b
    return normalize_angle(d)

class KinematicSmoother:
    def __init__(self, w_model=10.0, w_ref=1.0, w_smooth=10.0, w_s=1.0, w_fix=100.0, 
                 target_spacing=0.2, max_iter=50):
        """
        w_model: Weight for kinematic model constraints.
        w_ref: Weight for reference path deviation.
        w_smooth: Weight for curvature smoothness (d_kappa/ds).
        w_s: Weight for point spacing regularization.
        w_fix: Weight for start/end pose constraints.
        target_spacing: Desired distance between points (ds).
        """
        self.w_model = w_model
        self.w_ref = w_ref
        self.w_smooth = w_smooth
        self.w_s = w_s
        self.w_fix = w_fix
        self.target_spacing = target_spacing
        self.max_iter = max_iter

    def optimize(self, raw_path, gear_directions=None):
        """
        raw_path: np.array of shape (N, 2) or (N, 3). [x, y, (theta)]
        gear_directions: np.array of shape (N-1,), values +1 (fwd) or -1 (bwd).
                         If None, assumes all +1.
        """
        raw_path = np.array(raw_path)
        N_orig = len(raw_path)
        if N_orig < 2:
            return raw_path

        if gear_directions is None:
            gear_directions = np.ones(N_orig - 1)
        else:
            gear_directions = np.array(gear_directions)

        # 1. Preprocess: Inject cusps (duplicate nodes where gear flips)
        processed_path = [raw_path[0]]
        processed_gears = []
        is_cusp_segment = [] # True if segment is a zero-length cusp transition

        # We will reconstruct the path and gears.
        # Original: Point 0 --(gear0)--> Point 1 --(gear1)--> Point 2
        # If gear0 != gear1, we insert Point 1' (duplicate of 1).
        # New: Point 0 --(gear0)--> Point 1 --(0)--> Point 1' --(gear1)--> Point 2
        
        orig_indices_map = [0] # Map new index to original index (for ref cost)

        for i in range(N_orig - 1):
            # Add current segment
            # Current point is already in processed_path (from init or prev loop)
            # We are preparing to add next point.
            
            curr_gear = gear_directions[i]
            next_gear = gear_directions[i+1] if i + 1 < len(gear_directions) else curr_gear
            
            # Add the segment to next point
            processed_gears.append(curr_gear)
            is_cusp_segment.append(False)
            processed_path.append(raw_path[i+1])
            orig_indices_map.append(i+1)
            
            # Check for cusp
            if i < N_orig - 2 and curr_gear != next_gear:
                # Insert duplicate node
                processed_gears.append(0) # Gear 0 for "switch"
                is_cusp_segment.append(True)
                processed_path.append(raw_path[i+1]) # Duplicate
                orig_indices_map.append(i+1) # Maps to same ref point

        processed_path = np.array(processed_path)
        processed_gears = np.array(processed_gears)
        is_cusp_segment = np.array(is_cusp_segment, dtype=bool)
        
        N = len(processed_path)
        
        # 2. Initial Guess
        # Need: x, y, theta, kappa, ds
        x_init = processed_path[:, 0]
        y_init = processed_path[:, 1]
        
        # Estimate theta
        theta_init = np.zeros(N)
        # Use simple difference for initial theta, respecting gear
        for i in range(N - 1):
            dx = processed_path[i+1, 0] - processed_path[i, 0]
            dy = processed_path[i+1, 1] - processed_path[i, 1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if is_cusp_segment[i]:
                # In-place switch, keep theta same as prev (or average?)
                # Actually, usually theta is continuous at cusp
                theta_init[i] = theta_init[i-1] if i > 0 else 0
            else:
                if dist > 1e-6:
                    angle = np.arctan2(dy, dx)
                    if processed_gears[i] < 0:
                        angle += np.pi
                    theta_init[i] = normalize_angle(angle)
                else:
                    theta_init[i] = theta_init[i-1] if i > 0 else 0.0

        theta_init[-1] = theta_init[-2]
        
        # Use provided theta if available in raw_path?
        # If raw_path has 3 cols, we should guide theta_init using it.
        if raw_path.shape[1] >= 3:
             # simple nearest neighbor or use map
             pass
             # For now rely on geom.
        
        kappa_init = np.zeros(N)
        ds_init = np.zeros(N)
        
        for i in range(N - 1):
            if is_cusp_segment[i]:
                ds_init[i] = 0.0
            else:
                dist = np.linalg.norm(processed_path[i+1, :2] - processed_path[i, :2])
                ds_init[i] = dist
        
        # Flatten
        # x, y, theta, kappa, ds
        initial_guess = np.column_stack((x_init, y_init, theta_init, kappa_init, ds_init)).flatten()
        
        start_yaw = theta_init[0]
        if raw_path.shape[1] >= 3:
            start_yaw = raw_path[0, 2]
            
        # End yaw
        end_yaw = theta_init[-1]
        if raw_path.shape[1] >= 3:
            end_yaw = raw_path[-1, 2]

        # 3. Optimize
        res = least_squares(
            self._residuals,
            initial_guess,
            args=(processed_path, processed_gears, is_cusp_segment, start_yaw, end_yaw, orig_indices_map),
            verbose=1,
            max_nfev=self.max_iter
        )
        
        # 4. Post-process
        opt_vars = res.x.reshape((N, 5))
        
        # Filter out cusp duplicate nodes (where ds ~ 0 and is_cusp_segment)
        # Or just return full path? 
        # For control, full path is fine, but maybe we want to merge them back?
        # User usually expects same number of points? 
        # But we changed N by inserting points.
        # We should return the dense path, containing all info.
        
        return opt_vars

    def _residuals(self, vars, ref_path_orig_nodes, gears, is_cusp, start_yaw, end_yaw, orig_map):
        N = len(ref_path_orig_nodes)
        state = vars.reshape((N, 5))
        
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        kappa = state[:, 3]
        ds = state[:, 4]
        
        residuals = []
        
        # A. Kinematic Consistency
        residuals.extend(self._calculate_kinematic_residuals(x, y, theta, kappa, ds, gears, is_cusp))
        
        # B. Smoothness
        residuals.extend(self._calculate_smoothness_residuals(kappa, ds, is_cusp))
        
        # C. Reference Follow
        residuals.extend(self._calculate_reference_residuals(x, y, ref_path_orig_nodes))
        
        # D. Spacing Regularization
        residuals.extend(self._calculate_spacing_residuals(ds, is_cusp))

        # E. Boundary Constraints
        residuals.extend(self._calculate_boundary_residuals(x, y, theta, start_yaw, end_yaw, ref_path_orig_nodes))
        
        return np.array(residuals)

    def _calculate_kinematic_residuals(self, x, y, theta, kappa, ds, gears, is_cusp):
        residuals = []
        N = len(x)
        for i in range(N - 1):
            if is_cusp[i]:
                # Cusp segment: Ignore kinematic model, enforce strict state continuity.
                # Use w_fix (or high weight) to ensure numerical "identity".
                residuals.append(self.w_fix * (x[i+1] - x[i]))
                residuals.append(self.w_fix * (y[i+1] - y[i]))
                residuals.append(self.w_fix * angle_diff(theta[i+1], theta[i]))
            else:
                direction = gears[i] # 1, -1
                dir_val = 1.0 if direction >= 0 else -1.0
                
                # Mid-point integration
                # theta_mid = theta + dir * ds * k / 2
                mid_theta = theta[i] + dir_val * ds[i] * kappa[i] / 2.0
                
                x_pred = x[i] + dir_val * ds[i] * np.cos(mid_theta)
                y_pred = y[i] + dir_val * ds[i] * np.sin(mid_theta)
                theta_pred = theta[i] + dir_val * ds[i] * kappa[i]
                
                # Calculate errors
                residuals.append(self.w_model * (x[i+1] - x_pred))
                residuals.append(self.w_model * (y[i+1] - y_pred))
                residuals.append(self.w_model * angle_diff(theta[i+1], theta_pred))
        return residuals

    def _calculate_smoothness_residuals(self, kappa, ds, is_cusp):
        residuals = []
        N = len(kappa)
        for i in range(N - 1):
            if not is_cusp[i]:
                # Standard segment
                # Prevent division by zero
                denom = ds[i] if ds[i] > 1e-4 else 1e-4
                d_k = (kappa[i+1] - kappa[i]) / denom
                residuals.append(self.w_smooth * d_k)
            else:
                # Cusp segment: No kappa continuity enforced
                pass
        return residuals

    def _calculate_reference_residuals(self, x, y, ref_path_orig_nodes):
        # state[i] corresponds to orig_map[i] in ref_path_orig_nodes
        # ref_path_orig_nodes has the shape of processed path, but its values are from raw_path.
        # Actually passed `ref_path_orig_nodes` is already the processed array (with duplicates).
        # So we just compare 1-to-1.
        
        dist_err = np.sqrt((x - ref_path_orig_nodes[:, 0])**2 + (y - ref_path_orig_nodes[:, 1])**2)
        return (self.w_ref * dist_err).tolist()

    def _calculate_spacing_residuals(self, ds, is_cusp):
        residuals = []
        N = len(ds)
        for i in range(N - 1):
            if is_cusp[i]:
                residuals.append(self.w_s * 10 * (ds[i] - 0.0)) # Strong force to 0
            else:
                residuals.append(self.w_s * (ds[i] - self.target_spacing))
        
        # Constrain the last ds to 0 (no segment after last point)
        residuals.append(self.w_s * 10 * (ds[-1] - 0.0))
        return residuals

    def _calculate_boundary_residuals(self, x, y, theta, start_yaw, end_yaw, ref_path_orig_nodes):
        residuals = []
        residuals.append(self.w_fix * angle_diff(theta[0], start_yaw))
        residuals.append(self.w_fix * angle_diff(theta[-1], end_yaw))
        
        # Start/End position could also be fixed if needed, but ref cost handles it softly.
        residuals.append(self.w_fix * (x[0] - ref_path_orig_nodes[0, 0]))
        residuals.append(self.w_fix * (y[0] - ref_path_orig_nodes[0, 1]))
        residuals.append(self.w_fix * (x[-1] - ref_path_orig_nodes[-1, 0]))
        residuals.append(self.w_fix * (y[-1] - ref_path_orig_nodes[-1, 1]))
        return residuals
