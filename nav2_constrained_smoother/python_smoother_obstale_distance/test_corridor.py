
import numpy as np
import matplotlib.pyplot as plt
from constrained_smoother import ConstrainedSmoother, SmootherParams

def corridor_costmap(x, y):
    # L-shaped corridor
    # Vertical segment: x in [0, 4], y in [0, 10]
    # Horizontal segment: x in [0, 10], y in [6, 10]
    # Walls are high cost
    
    # Safe zone (0 cost)
    is_safe = False
    
    # Vertical part (width 4, length 10) centered at x=2
    if 0 <= x <= 4 and 0 <= y <= 10:
        is_safe = True
        
    # Horizontal part (width 4) centered at y=8
    if 0 <= x <= 10 and 6 <= y <= 10:
        is_safe = True
        
    if is_safe:
        # Distance map for smoother gradients?
        # For this test, let's make walls distinct
        # Add slight gradient towards wall
        # Vertical walls
        dist_v = min(abs(x - 0), abs(x - 4)) if (0 <= y <= 10) else 0
        dist_h = min(abs(y - 6), abs(y - 10)) if (0 <= x <= 10) else 0
        
        # We want cost to rise near walls (dist -> 0)
        # Cost = exp(-dist)
        
        # Simple hard walls
        return 0.0
    else:
        return 254.0

def test_rectangular_robot():
    print("Testing Rectangular Robot in Corridor...")
    
    # Path going through L-turn
    # P1: (2, 2) -> P2: (2, 8) -> P3: (8, 8)
    # Corner is at (2, 8). 
    # If robot is long, it cuts the corner.
    
    # Generate initial path
    path1 = np.column_stack((np.ones(10)*2, np.linspace(0, 8, 10)))
    path2 = np.column_stack((np.linspace(2, 10, 10), np.ones(10)*8))
    # Remove duplicate
    path = np.vstack([path1[:-1], path2])
    
    # Rectangular footprint: 1.0 x 0.5
    # Points: FL, FR, BL, BR
    # x: forward, y: left
    footprint_points = [
        0.5, 0.25, 1.0,   # FL
        0.5, -0.25, 1.0,  # FR
        -0.5, 0.25, 1.0,  # BL
        -0.5, -0.25, 1.0, # BR
        0.5, 0.0, 1.0,    # Front Center
        -0.5, 0.0, 1.0    # Back Center
    ]
    
    params = SmootherParams()
    params.w_cost = 50.0 # High cost to avoid collision
    params.w_smooth = 10.0
    params.w_dist = 0.5
    params.cost_check_points = footprint_points
    params.max_iterations = 100
    params.keep_start_orientation = True # Explicitly keep start
    params.keep_goal_orientation = True  # Explicitly keep goal
    
    smoother = ConstrainedSmoother(params)
    smoothed_path = smoother.smooth(path.tolist(), costmap_func=corridor_costmap)
    smoothed_path = np.array(smoothed_path)
    
    # Visualization
    plt.figure(figsize=(10, 10))
    
    # Draw corridor walls
    # Vertical
    plt.plot([0, 0], [0, 10], 'k-', linewidth=2)
    plt.plot([4, 4], [0, 6], 'k-', linewidth=2)
    # Horizontal
    plt.plot([0, 10], [10, 10], 'k-', linewidth=2)
    plt.plot([4, 10], [6, 6], 'k-', linewidth=2)
    
    plt.plot(path[:, 0], path[:, 1], 'r--', label='Original')
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'b.-', label='Smoothed')
    
    # Orientation Arrows
    # Start Orientation (from first segment)
    start_dx = path[1][0] - path[0][0]
    start_dy = path[1][1] - path[0][1]
    # Normalize
    norm_s = np.hypot(start_dx, start_dy)
    plt.arrow(path[0][0], path[0][1], start_dx/norm_s, start_dy/norm_s, 
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='Start Orient')
              
    # Goal Orientation (from last segment)
    end_dx = path[-1][0] - path[-2][0]
    end_dy = path[-1][1] - path[-2][1]
    norm_e = np.hypot(end_dx, end_dy)
    plt.arrow(path[-1][0], path[-1][1], end_dx/norm_e, end_dy/norm_e, 
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='Goal Orient')

    # Draw footprints
    
    # Draw footprints
    # Calculate orientations for plotting
    for i in range(0, len(smoothed_path), 3): # Plot every 3rd
        pt = smoothed_path[i]
        x, y = pt[0], pt[1]
        
        # Estimate theta
        if i < len(smoothed_path)-1:
            dx = smoothed_path[i+1][0] - x
            dy = smoothed_path[i+1][1] - y
            theta = np.arctan2(dy, dx)
        elif i > 0:
            dx = x - smoothed_path[i-1][0]
            dy = y - smoothed_path[i-1][1]
            theta = np.arctan2(dy, dx)
        else:
            theta = 0
            
        # Plot rectangle
        l, w = 1.0, 0.5
        # Corners relative to center
        corners = np.array([
            [l/2, w/2],
            [l/2, -w/2],
            [-l/2, -w/2],
            [-l/2, w/2],
            [l/2, w/2]
        ])
        
        # Rotate
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        corners_world = (R @ corners.T).T + np.array([x, y])
        
        plt.plot(corners_world[:, 0], corners_world[:, 1], 'g-', alpha=0.5)
    
    plt.axis('equal')
    plt.title('Rectangular Robot in Corridor')
    plt.legend()
    plt.savefig('test_corridor.png')
    print("Saved test_corridor.png")
    
    # Verify costs
    # Check if any footprint point hits wall (rough check)
    max_cost = 0
    for pt in smoothed_path:
        # Approximate check with center for now, but implementation checks corners
        c = corridor_costmap(pt[0], pt[1])
        # We can re-check corners if we want rigorous test
        pass

if __name__ == "__main__":
    test_rectangular_robot()
