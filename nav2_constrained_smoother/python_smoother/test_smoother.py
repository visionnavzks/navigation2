
import numpy as np
import matplotlib.pyplot as plt
from constrained_smoother import ConstrainedSmoother, SmootherParams


def calculate_metrics(path):
    path = np.array(path)
    # Length
    diffs = np.diff(path[:, :2], axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    length = np.sum(dists)
    
    # Curvature (approx) using circle fitting on triplets
    # Use the same logic as in smoother or simplified
    curvatures = []
    for i in range(1, len(path)-1):
        A = path[i-1, :2]
        B = path[i, :2]
        C = path[i+1, :2]
        
        a = np.linalg.norm(B - C)
        b = np.linalg.norm(A - C)
        c = np.linalg.norm(A - B)
        
        area = 0.5 * np.abs(
            A[0] * (B[1] - C[1]) +
            B[0] * (C[1] - A[1]) +
            C[0] * (A[1] - B[1])
        )
        
        if a*b*c > 1e-6:
            k = 4 * area / (a * b * c)
            curvatures.append(k)
        else:
            curvatures.append(0.0)
            
    max_k = np.max(curvatures) if curvatures else 0.0
    mean_k = np.mean(curvatures) if curvatures else 0.0
    
    return length, max_k, mean_k

def test_straight_line_with_noise():
    print("Testing Straight Line with Noise...")
    # Generate straight line
    x = np.linspace(0, 10, 20)
    y = np.zeros_like(x)
    
    # Add noise
    np.random.seed(42)
    y_noisy = y + np.random.normal(0, 0.1, size=x.shape)
    y_noisy[0] = 0 # Fix start
    y_noisy[-1] = 0 # Fix end
    
    path = np.column_stack((x, y_noisy))
    
    params = SmootherParams()
    params.w_smooth = 100.0 # High smoothing weight
    params.w_dist = 0.0 # Low distance weight to allow smoothing
    params.w_curve = 0.0
    params.keep_start_orientation = True
    params.keep_goal_orientation = True
    
    smoother = ConstrainedSmoother(params)
    smoothed_path = smoother.smooth(path.tolist())
    smoothed_path = np.array(smoothed_path)
    
    l_orig, mk_orig, _ = calculate_metrics(path)
    l_smooth, mk_smooth, _ = calculate_metrics(smoothed_path)
    
    print(f"  Length: {l_orig:.3f} -> {l_smooth:.3f}")
    print(f"  Max Curvature: {mk_orig:.3f} -> {mk_smooth:.3f}")
    
    plt.figure()
    plt.plot(path[:, 0], path[:, 1], 'r--', label='Noisy')
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'b-', label='Smoothed')
    
    # Orientation Arrows
    # Start
    start_dx = path[1][0] - path[0][0]
    start_dy = path[1][1] - path[0][1]
    norm_s = np.hypot(start_dx, start_dy)
    plt.arrow(path[0][0], path[0][1], start_dx/norm_s, start_dy/norm_s, 
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='Start Orient')
    # End
    end_dx = path[-1][0] - path[-2][0]
    end_dy = path[-1][1] - path[-2][1]
    norm_e = np.hypot(end_dx, end_dy)
    plt.arrow(path[-1][0], path[-1][1], end_dx/norm_e, end_dy/norm_e, 
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='Goal Orient')
    plt.legend()
    plt.title(f'Straight Line (Len: {l_smooth:.2f})')
    plt.savefig('test_straight.png')
    print("Saved test_straight.png")

def test_corner_rounding():
    print("\nTesting Corner Rounding...")
    # L-shape
    x = np.concatenate([np.linspace(0, 5, 10), np.linspace(5, 5, 10)])
    y = np.concatenate([np.zeros(10), np.linspace(0, 5, 10)])
    
    path = np.column_stack((x, y))
    
    # Add indices to separate segments slightly to avoid duplicate point at corner
    # OR just keep as is, smoother handles duplicates (though C++ says "not supported" for in-place rotations)
    # Let's ensure no duplicate
    x = np.concatenate([np.linspace(0, 5, 11)[:-1], np.linspace(5, 5, 11)])
    y = np.concatenate([np.zeros(11)[:-1], np.linspace(0, 5, 11)])
    path = np.column_stack((x, y))

    # Force 180 degree start: (0,0) -> (-1, 0) -> ...
    # Insert (-1, 0) at index 1
    # P0 is (0,0)
    # P1 becomes (-1, 0)
    # This forces the first segment to point Left (180 deg)
    path = np.insert(path, 1, [-1.0, 0.0], axis=0)

    params = SmootherParams()
    params.min_turning_radius = 2.0
    params.w_curve = 50.0 # Enforce curvature
    params.w_smooth = 10.0
    params.w_dist = 0.5 
    params.keep_start_orientation = True
    params.keep_goal_orientation = True 
    
    smoother = ConstrainedSmoother(params)
    smoothed_path = smoother.smooth(path.tolist())
    smoothed_path = np.array(smoothed_path)
    
    l_orig, mk_orig, _ = calculate_metrics(path)
    l_smooth, mk_smooth, _ = calculate_metrics(smoothed_path)
    
    print(f"  Length: {l_orig:.3f} -> {l_smooth:.3f}")
    print(f"  Max Curvature: {mk_orig:.3f} -> {mk_smooth:.3f} (Target < {1.0/params.min_turning_radius:.3f})")

    plt.figure()
    plt.plot(path[:, 0], path[:, 1], 'r--', label='Original')
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'b-', label='Smoothed')
    
    # Arrows
    start_dx = path[1][0] - path[0][0]
    start_dy = path[1][1] - path[0][1]
    norm_s = np.hypot(start_dx, start_dy)
    plt.arrow(path[0][0], path[0][1], start_dx/norm_s, start_dy/norm_s, 
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='Start Orient')
    
    end_dx = path[-1][0] - path[-2][0]
    end_dy = path[-1][1] - path[-2][1]
    norm_e = np.hypot(end_dx, end_dy)
    plt.arrow(path[-1][0], path[-1][1], end_dx/norm_e, end_dy/norm_e, 
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='Goal Orient')
    plt.axis('equal')
    plt.legend()
    plt.title(f'Corner Rounding (Max K: {mk_smooth:.2f})')
    plt.savefig('test_corner.png')
    print("Saved test_corner.png")

def dummy_costmap(x, y):
    # Gaussian obstacle at (5, 0)
    dist = np.sqrt((x - 5)**2 + (y - 0)**2)
    # Cost roughly 0-254
    cost = 254.0 * np.exp(-dist**2 / 1.0)
    return cost

def test_obstacle_avoidance():
    print("\nTesting Obstacle Avoidance...")
    # Straight line passing through (5, 0)
    x = np.linspace(0, 10, 20)
    y = np.zeros_like(x)
    path = np.column_stack((x, y))
    
    params = SmootherParams()
    params.w_cost = 0.05 # Cost weight
    params.w_smooth = 5.0
    params.w_dist = 0.05
    params.keep_start_orientation = True
    params.keep_goal_orientation = True
    
    smoother = ConstrainedSmoother(params)
    smoothed_path = smoother.smooth(path.tolist(), costmap_func=dummy_costmap)
    smoothed_path = np.array(smoothed_path)
    
    # Calculate max cost on path
    costs_orig = [dummy_costmap(p[0], p[1]) for p in path]
    costs_smooth = [dummy_costmap(p[0], p[1]) for p in smoothed_path]
    
    print(f"  Max Cost: {np.max(costs_orig):.1f} -> {np.max(costs_smooth):.1f}")
    
    plt.figure()
    
    # Visualize costmap
    X, Y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(-2, 2, 50))
    Z = dummy_costmap(X, Y)
    plt.contourf(X, Y, Z, cmap='Reds', alpha=0.3)
    
    plt.plot(path[:, 0], path[:, 1], 'r--', label='Original')
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'b-', label='Smoothed')
    
    # Arrows
    start_dx = path[1][0] - path[0][0]
    start_dy = path[1][1] - path[0][1]
    norm_s = np.hypot(start_dx, start_dy)
    plt.arrow(path[0][0], path[0][1], start_dx/norm_s, start_dy/norm_s, 
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='Start Orient')
    
    end_dx = path[-1][0] - path[-2][0]
    end_dy = path[-1][1] - path[-2][1]
    norm_e = np.hypot(end_dx, end_dy)
    plt.arrow(path[-1][0], path[-1][1], end_dx/norm_e, end_dy/norm_e, 
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='Goal Orient')
    plt.axis('equal')
    plt.legend()
    plt.title(f'Obstacle Avoidance (Max Cost: {np.max(costs_smooth):.1f})')
    plt.savefig('test_obstacle.png')
    print("Saved test_obstacle.png")


if __name__ == "__main__":
    test_straight_line_with_noise()
    test_corner_rounding()
    test_obstacle_avoidance()
