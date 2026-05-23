
import numpy as np
import scipy.optimize
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass
class SmootherParams:
    start_time: float = 0.0
    max_time: float = 0.1
    debug: bool = False
    max_iterations: int = 70
    min_turning_radius: float = 0.40
    w_curve: float = 30.0
    w_dist: float = 0.0
    w_smooth: float = 2000000.0
    w_danger: float = 0.0
    w_collision: float = 0.0
    danger_zone_dist: float = 0.5 # d_safe
    
    # Deprecated or unused in new SDF mode unless we want to keep it? 
    # The user asked to change the function, implying replacement.
    # We'll keep w_cost but it might not be used if the new logic relies on w_danger/w_collision
    w_cost: float = 0.015 # Legacy costmap weight
    w_cost_cusp_multiplier: float = 3.0
    cusp_zone_length: float = 2.5
    path_downsampling_factor: int = 3
    path_upsampling_factor: int = 1
    keep_start_orientation: bool = True
    keep_goal_orientation: bool = True
    cusp_costmap_weight: float = 0.0 # Will be calculated as w_cost * w_cost_cusp_multiplier
    # Footprint check points [x1, y1, w1, x2, y2, w2, ...] relative to robot center
    cost_check_points: List[float] = None
    
    def __post_init__(self):
        if self.cost_check_points is None:
            self.cost_check_points = []
        # Derived parameter
        self.cusp_costmap_weight = self.w_cost * self.w_cost_cusp_multiplier
        self.max_curvature = 1.0 / self.min_turning_radius if self.min_turning_radius > 0 else float('inf')

class ConstrainedSmoother:
    def __init__(self, params: SmootherParams = None):
        self.params = params if params else SmootherParams()

    def smooth(self, path: List[List[float]], costmap_func=None) -> List[List[float]]:
        """
        Smooth the given path.
        :param path: List of [x, y, theta] or [x, y] points.
        :param costmap_func: Function to get cost at (x, y). Returns cost (0-255).
        :return: Smoothed path.
        """
        if len(path) < 2:
            return path
        
        path_np = np.array(path)
        path_xy = path_np[:, :2] # Extract x, y
        
        # Determine cusps and direction
        # C++ impl calculates directions and cusps here.
        # For simplicity in this first pass, we assume forward motion or handle simplified case.
        # But to match C++, we should process orientations.
        
        # Optimization
        optimized_path_xy = self._optimize(path_xy, costmap_func)
        
        # Upsample and populate orientations
        final_path = self._upsample_and_populate(optimized_path_xy, self.params)
        
        return final_path

    def _optimize(self, path_xy: np.ndarray, costmap_func) -> np.ndarray:
        """
        Run the optimization.
        """
        N = len(path_xy)
        if N < 3:
            return path_xy

        # Flatten path for scipy [x0, y0, x1, y1, ...]
        x0 = path_xy.flatten()
        
        # Bounds? Unconstrained L-BFGS-B or Least Squares?
        # C++ uses Ceres. Least Squares is appropriate.
        
        # We need to fix start and end points
        # In scipy least_squares, we can't easily fix variables directly other than bounds or structure.
        # A common trick is to remove fixed variables from the optimization vector.
        # Optimization variables: points 1 to N-2 (0-indexed)
        # Fixed: 0 and N-1.
        
        # If keeping orientation, we fix the second point (P1) and second-to-last point (Pn-2)
        # to preserve the direction of the first and last segments.
        idx_start = 1
        idx_end = N - 1
        
        if self.params.keep_start_orientation and N > 2:
            idx_start = 2
            
        if self.params.keep_goal_orientation and N > 2:
            idx_end = N - 2
        
        # Check if we have enough points to optimize
        if idx_start >= idx_end:
            return path_xy
        
        # For this implementation, let's optimize all points except first and last
        x_init = path_xy[idx_start:idx_end].flatten()
        
        if len(x_init) == 0:
            return path_xy
            
        def residuals_func(x_flat):
            # Reconstruct full path
            current_inner = x_flat.reshape((-1, 2))
            full_path = np.vstack([path_xy[:idx_start], current_inner, path_xy[idx_end:]])
            
            res = []
            
            # Smoothness (acceleration)
            # w * || (p_i+1 - p_i) - (p_i - p_i-1) ||^2
            # For least squares, we return the vector components.
            # (p_i+1 - 2p_i + p_i-1)
            
            # Vectorized calculation
            # p_prev = full_path[:-2]
            # p_curr = full_path[1:-1]
            # p_next = full_path[2:]
            
            # Smoothness
            # diff = (p_next - p_curr) - (p_curr - p_prev) = p_next - 2*p_curr + p_prev
            # We assume equal spacing for simplicity or stick to standard simple smoothness.
            # C++ uses `next_to_last_length_ratio_ * d_next - d_prev`.
            # If we assume mostly uniform distribution initially, this simplifies to standard term.
            
            p_prev = full_path[:-2]
            p_curr = full_path[1:-1]
            p_next = full_path[2:]
            
            smooth_res = self.params.w_smooth * (p_next - 2 * p_curr + p_prev)
            res.append(smooth_res.flatten())
            
            # Distance from original
            # w * || p_i - p_orig_i ||
            # Need to match indices.
            # We are optimizing points from idx_start to idx_end
            # Corresponds to path_xy[idx_start:idx_end]
            
            dist_res = self.params.w_dist * (current_inner - path_xy[idx_start:idx_end])
            res.append(dist_res.flatten())
            
            # Curvature
            if self.params.w_curve > 0 and self.params.min_turning_radius > 0:
                # Calculate curvature for each triplet p_prev, p_curr, p_next
                # Using Menger curvature formula or circle fitting
                # curvature k = 4 * Area / (a * b * c) where a, b, c are side lengths
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                
                # Vectorized calculation
                # A = p_prev, B = p_curr, C = p_next
                # a = norm(B - C)
                # b = norm(A - C)
                # c = norm(A - B)
                
                A = p_prev
                B = p_curr
                C = p_next
                
                a = np.linalg.norm(B - C, axis=1)
                b = np.linalg.norm(A - C, axis=1)
                c = np.linalg.norm(A - B, axis=1)
                
                # Area using shoelace formula for triangle
                # 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area = 0.5 * np.abs(
                    A[:, 0] * (B[:, 1] - C[:, 1]) +
                    B[:, 0] * (C[:, 1] - A[:, 1]) +
                    C[:, 0] * (A[:, 1] - B[:, 1])
                )
                
                # Curvature k = 4 * Area / (a * b * c)
                # Avoid division by zero
                denominator = a * b * c
                mask = denominator > 1e-6
                
                k = np.zeros_like(area)
                k[mask] = 4 * area[mask] / denominator[mask]
                
                # Residual: w * (k - k_max)^2 only if k > k_max
                # Since least_squares minimizes sum of squares, we return sqrt(w) * max(0, k - k_max)
                k_residual = np.maximum(0, k - self.params.max_curvature)
                
                # Always append residuals to maintain constant size for least_squares
                # If weight is 0, this will just be 0
                res.append(np.sqrt(self.params.w_curve) * k_residual)

            # Costmap / SDF
            if costmap_func is not None and (self.params.w_danger > 0 or self.params.w_collision > 0 or self.params.w_cost > 0):
                # We assume costmap_func now returns signed distance if using w_danger/w_collision
                # Or returns cost (0-255) if using w_cost. 
                # The user instruction implies replacing the function.
                # But to preserve backward compatibility if needed, we might check params.
                
                # If using new params
                use_sdf = (self.params.w_danger > 0 or self.params.w_collision > 0)
                
                # If no check points, use center
                points_to_check = current_inner
                weights_to_check = np.ones(len(current_inner)) # Default weight 1.0 per point
                
                if self.params.cost_check_points:
                     # Calculate directions and transform check points
                     # (Same logic as before to get pts_world)
                     tangents = p_next - p_prev
                     norms = np.linalg.norm(tangents, axis=1)
                     mask = norms > 1e-6
                     tangents[mask] /= norms[mask][:, np.newaxis]
                     tangents[~mask] = np.array([1.0, 0.0])
                     
                     check_pts_local = []
                     check_w = []
                     for i in range(0, len(self.params.cost_check_points), 3):
                         check_pts_local.append([self.params.cost_check_points[i], self.params.cost_check_points[i+1]])
                         check_w.append(self.params.cost_check_points[i+2])
                     
                     check_pts_local = np.array(check_pts_local)
                     check_w = np.array(check_w)
                     
                     # Broadphase or per-point transform
                     # We have M points and K check points. Result M*K points.
                     # efficient broadcasting:
                     # Rot matrices M x 2 x 2
                     cos_theta = tangents[:, 0]
                     sin_theta = tangents[:, 1]
                     
                     # pts_world[i, k] = R[i] * local[k] + center[i]
                     # local[k] is (x, y)
                     # R[i] * local[k] = [c -s; s c] * [x; y] = [cx - sy; sx + cy]
                     
                     # x_rot = local_x * cos - local_y * sin
                     # y_rot = local_x * sin + local_y * cos
                     
                     # Expand dims for broadcasting
                     # check_pts_local: (K, 2)
                     # cos_theta: (M,)
                     
                     local_x = check_pts_local[:, 0] # (K,)
                     local_y = check_pts_local[:, 1] # (K,)
                     
                     # (M, K)
                     x_rot = np.outer(cos_theta, local_x) - np.outer(sin_theta, local_y)
                     y_rot = np.outer(sin_theta, local_x) + np.outer(cos_theta, local_y)
                     
                     # Add centers
                     # current_inner: (M, 2)
                     pts_world_x = x_rot + current_inner[:, 0][:, np.newaxis]
                     pts_world_y = y_rot + current_inner[:, 1][:, np.newaxis]
                     
                     points_to_check = np.stack([pts_world_x.flatten(), pts_world_y.flatten()], axis=1)
                     
                     # Flatten weights: repeating check_w for each path point
                     # check_w is (K,)
                     # We need (M*K,)
                     weights_to_check = np.tile(check_w, len(current_inner))
                
                # Evaluate function
                # costmap_func should accept vectorized input ideally, but let's loop if unsure or simple wrapper
                if use_sdf:
                    # Expecting signed distance
                    dists = []
                    # Try to see if costmap_func supports array
                    # If not, loop
                    # Assuming costmap_func is like the one in esdf.py (get_distance), it supports scalar or array?
                    # The user prompt passed `costmap_func` which in test is `dummy_costmap`.
                    # Let's assume it can take x, y scalars.
                    pts_x = points_to_check[:, 0]
                    pts_y = points_to_check[:, 1]
                    
                    # If the function is from our ESDF class, it supports arrays.
                    # But generic loop for safety unless we know.
                    # Let's do list comprehension
                    d_vals = np.array([costmap_func(p[0], p[1]) for p in points_to_check])
                    
                    # Apply Piecewise Cost
                    # r = 0 if d > d_safe
                    # r = w_danger * (d - d_safe) if 0 < d <= d_safe
                    # r = w_collision * (d - d_safe) if d <= 0
                    
                    sdf_res = np.zeros_like(d_vals)
                    d_safe = self.params.danger_zone_dist
                    
                    # Mask 1: Danger zone (0 < d <= d_safe)
                    mask_danger = (d_vals > 0) & (d_vals <= d_safe)
                    if np.any(mask_danger):
                         sdf_res[mask_danger] = self.params.w_danger * (d_vals[mask_danger] - d_safe)
                         
                    # Mask 2: Collision (d <= 0)
                    mask_collision = d_vals <= 0
                    if np.any(mask_collision):
                        sdf_res[mask_collision] = self.params.w_collision * (d_vals[mask_collision] - d_safe)
                        
                    # Weight by check point weight (sqrt, because least_squares squares it)
                    # res = sqrt(point_weight) * residual
                    combined_res = np.sqrt(weights_to_check) * sdf_res
                    res.append(combined_res)
                    
                else:
                    # Legacy mode (Cost 0-255)
                    costs = np.array([costmap_func(p[0], p[1]) for p in points_to_check])
                    combined_res = np.sqrt(self.params.w_cost) * np.sqrt(weights_to_check) * costs
                    res.append(combined_res)
                
            return np.concatenate(res)

        res = scipy.optimize.least_squares(
            residuals_func, 
            x_init, 
            method='lm', # Levenberg-Marquardt is usually fast for unconstrained
            max_nfev=self.params.max_iterations
        )
        
        optimized_inner = res.x.reshape((-1, 2))
        return np.vstack([path_xy[:idx_start], optimized_inner, path_xy[idx_end:]])

    def _upsample_and_populate(self, path_xy: np.ndarray, params: SmootherParams) -> List[List[float]]:
        """
        Upsample the path using Cubic Bezier interpolation and assign orientations.
        """
        if len(path_xy) < 2:
            return [[x, y, 0.0] for x, y in path_xy]

        upsampled_path = []
        path_optim = path_xy

        # Helper for Cubic Bezier
        def cubic_bezier(pt0, pt1, pt2, pt3, mu):
            # pt = (1-mu)^3 * pt0 + 3*mu*(1-mu)^2 * pt1 + 3*mu^2*(1-mu) * pt2 + mu^3 * pt3
            # OR matrix form as in C++
            c = 3 * (pt1 - pt0)
            b = 3 * (pt2 - pt1) - c
            a = pt3 - pt0 - c - b
            
            return a * mu**3 + b * mu**2 + c * mu + pt0

        for i in range(len(path_optim) - 1):
            curr_pt = path_optim[i]
            next_pt = path_optim[i+1]
            
            # Add current point
            # If it's not the first point, we might have already added it as the last point of previous segment?
            # C++ implementation adds points carefully.
            
            # Simple approach:
            # For each segment, generate N points.
            # If upsampling_factor = 1, we just return original points? 
            # No, C++ says: "0 - path remains downsampled, 1 - path is upsampled back to original granularity using cubic bezier"
            # Here we just assume upsampling_factor means "number of segments between points" or similar.
            # C++ code: int interp_cnt = (last_i - prelast_i) * params.path_upsampling_factor - 1;
            
            # Let's simplify: if upsampling_factor > 0, we interpolate.
            
            # In C++ optimization, points might be downsampled (skipped). 
            # `path_optim` passed here seems to be the result of optimization which presumably has same number of points as input to optimize.
            # But the input to optimize might have been downsampled?
            # In my python impl, I haven't implemented downsampling yet. So `path_optim` is full path.
            
            # If so, upsampling might just be smoothing between points.
            # C++ implementation calculates control points based on tangents.
            
            # Let's just do linear interpolation if factor > 1 for now, or just return points if factor <= 1
            # But wait, the previous code had a placeholder.
            # I will implement a basic Catmull-Rom or similar if tangents are needed, or just standard Bezier if I have control points.
            # C++ code calculates control points:
            # pt1 = prelast_pt + prelast_dir * dist * 0.4
            # pt2 = last_pt - last_dir * dist * 0.4
            
            # We need directions (tangents).
            pass # We will calculate orientations in the loop

        # For now, let's just return the optimized path with calculated orientations
        final_path = []
        for i in range(len(path_optim)):
            x, y = path_optim[i]
            theta = 0.0
            if i < len(path_optim) - 1:
                dx = path_optim[i+1][0] - x
                dy = path_optim[i+1][1] - y
                theta = math.atan2(dy, dx)
            elif i > 0:
                dx = x - path_optim[i-1][0]
                dy = y - path_optim[i-1][1]
                theta = math.atan2(dy, dx)
            
            final_path.append([x, y, theta])
            
        return final_path

