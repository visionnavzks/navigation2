
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from zhengli.kinematic_smoother.esdf_map import ESDFMap
from zhengli.kinematic_smoother.kinematic_smoother import KinematicSmoother

def create_occupancy_grid(width=100, height=100, res=0.2):
    # 0 = Free, 1 = Obstacle
    # 20m x 20m grid
    grid = np.ones((height, width), dtype=int)
    
    # Define "回" (Hui) Corridor Geometry
    # Outer bounds: [2, 18]
    # Inner bounds (obstacle): [6, 14]
    # Corridor width approx 4m.
    # Center of path approx at 4m offset from edges.
    
    for y in range(height):
        for x in range(width):
            wx = x * res
            wy = y * res
            
            # Carve out the loop
            # Outer limit
            inside_outer = (2.0 <= wx <= 18.0) and (2.0 <= wy <= 18.0)
            # Inner limit (Obstacle island)
            inside_inner = (6.0 <= wx <= 14.0) and (6.0 <= wy <= 14.0)
            
            if inside_outer and not inside_inner:
                grid[y, x] = 0 # Free
    
    return grid

def draw_robot(x, y, yaw, length, width, center_offset=0.0):
    # Simple box
    # Geometric center is at (center_offset, 0) in base frame
    # Corners relative to Geometric center are +/- length/2, +/- width/2
    # So corners relative to base are +/- length/2 + center_offset
    
    half_l = length / 2.0
    half_w = width / 2.0
    
    # Define corners in Base Frame
    # FL, BL, BR, FR, FL
    # Front is positive X
    
    # If offset=0, range is [-L/2, L/2]
    # If offset=1, range is [-L/2+1, L/2+1]
    
    corners_x = np.array([half_l, -half_l, -half_l, half_l, half_l]) + center_offset
    corners_y = np.array([half_w, half_w, -half_w, -half_w, half_w])
    
    outline = np.column_stack((corners_x, corners_y))
    
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    outline_rot = outline @ rot.T
    outline_rot += np.array([x, y])
    plt.plot(outline_rot[:,0], outline_rot[:,1], 'b-', alpha=0.3)

def main():
    res = 0.2
    grid_w, grid_h = 100, 100 # 20m x 20m
    
    # Create Occupancy Grid
    occ_grid = create_occupancy_grid(grid_w, grid_h, res)
    
    # Create ESDF
    esdf = ESDFMap(occ_grid, res, origin_x=0.0, origin_y=0.0, use_bicubic=False)
    
    # Path: "回" Loop
    # A(4,4) -> B(16,4) -> C(16,16) -> D(4,16) -> A(4,4)
    # 4 segments
    
    pts_per_leg = 20
    
    # 1. Bottom: (8,4) -> (16,4)
    x1 = np.linspace(8, 16, pts_per_leg)
    y1 = np.full_like(x1, 4.0)
    th1 = np.zeros_like(x1)
    
    # 2. Right: (16,4) -> (16,16)
    y2 = np.linspace(4, 16, pts_per_leg)
    x2 = np.full_like(y2, 16.0)
    th2 = np.full_like(y2, np.pi/2)
    
    # 3. Top: (16,16) -> (4,16)
    x3 = np.linspace(16, 4, pts_per_leg)
    y3 = np.full_like(x3, 16.0)
    th3 = np.full_like(x3, np.pi)
    
    # 4. Left: (4,16) -> (4,8)
    y4 = np.linspace(16, 8, pts_per_leg)
    x4 = np.full_like(y4, 4.0)
    th4 = np.full_like(y4, -np.pi/2)
    
    # Concatenate (handling overlaps crudely by just stacking, clean enough for demo)
    path_x = np.concatenate([x1[:-1], x2[:-1], x3[:-1], y4]) # Wait x4 is constant 4.0
    # Fix x/y concatenation
    raw_x = np.concatenate([x1[:-1], x2[:-1], x3[:-1], x4])
    raw_y = np.concatenate([y1[:-1], y2[:-1], y3[:-1], y4])
    raw_th = np.concatenate([th1[:-1], th2[:-1], th3[:-1], th4])
    
    # Unwrap angles to avoid jumps?
    # Actually simple 0->pi/2 is fine. But pi -> -pi/2 is a jump of -3pi/2 (270 deg).
    # Ideally should be continuous: 0 -> pi/2 -> pi -> 3pi/2 (or -pi/2 mapped).
    # Since the smoother might use diffs, we should ensure continuity if possible,
    # or rely on the smoother's angle normalization handling.
    # Let's clean up angles: 0, pi/2, pi, 3pi/2 (for 4th leg)
    raw_th[-pts_per_leg:] = 3 * np.pi / 2
    
    raw_path = np.column_stack((raw_x, raw_y, raw_th))
    gears = np.ones(len(raw_path)-1)
    
    print("Optimizing '回' Loop Path...")
    print(f"Map Size: {grid_w*res}m x {grid_h*res}m")
    
    # Smoother
    # TEST OFFSET: Base is 2.0m BEHIND the center (or center is 2.0m in front of base)
    # Robot Length 7.0. Half length 3.5.
    # If offset = 2.0. Front is at 2.0 + 3.5 = 5.5. Back is at 2.0 - 3.5 = -1.5.
    center_offset = 0.0 # Let's try 1.0m forward offset
    
    smoother = KinematicSmoother(
        robot_params={'length': 7.0, 'width': 2.0, 'center_x_offset': center_offset},
        esdf_map=esdf,
        w_obs=200.0,
        w_danger=50.0,
        d_safe=0.3,
        w_smooth=2.0,
        w_model=1000.0,
        w_fix=1000.0,
        target_spacing=0.25,
        max_iter=100
    )
    
    start_time = time.time()
    opt_vars = smoother.optimize(raw_path, gears)
    print(f"Optimization took {time.time() - start_time:.4f}s")
    
    # Verification: Check ds >= 0
    ds_vals = opt_vars[:, 4]
    print(f"Minimum ds: {np.min(ds_vals):.6f}")
    if np.min(ds_vals) < -1e-6:
        print("WARNING: ds < 0 detected!")
    else:
        print("VERIFIED: ds >= 0 constraint hold.")
    
    # Plot
    plt.figure(figsize=(10, 10))
    
    # Plot ESDF Background
    extent = [0, grid_w*res, 0, grid_h*res]
    plt.imshow(esdf.esdf_field, origin='lower', extent=extent, cmap='RdBu', vmin=-2.0, vmax=5.0)
    plt.colorbar(label='Distance (m)')

    # Plot 0-distance line (Obstacle Boundary)
    plt.contour(esdf.esdf_field, origin='lower', extent=extent, levels=[0], colors='k', linewidths=2)
    
    # Plot Raw
    plt.plot(raw_path[:,0], raw_path[:,1], 'k--', label='Raw Path', linewidth=2, alpha=0.5)
    
    # Plot Opt
    ox = opt_vars[:, 0]
    oy = opt_vars[:, 1]
    oth = opt_vars[:, 2]
    plt.plot(ox, oy, 'g-', label='Optimized', linewidth=3)

    # Plot Start and End Arrows
    arrow_len = 1.5
    # Start (Blue)
    plt.arrow(ox[0], oy[0], arrow_len*np.cos(oth[0]), arrow_len*np.sin(oth[0]), 
              head_width=0.6, head_length=0.6, fc='blue', ec='blue', alpha=0.5, width=0.15)
    # End (Red)
    plt.arrow(ox[-1], oy[-1], arrow_len*np.cos(oth[-1]), arrow_len*np.sin(oth[-1]), 
              head_width=0.6, head_length=0.6, fc='red', ec='red', alpha=0.5, width=0.15)
    
    # Plot Robot Footprint at intervals
    indices = np.arange(0, len(ox), 1)
    # 2. 如果最后一个索引不是 len(ox)-1，则补上终点索引
    if indices[-1] != len(ox) - 1:
        indices = np.append(indices, len(ox) - 1)
    for i in indices:
        draw_robot(ox[i], oy[i], oth[i], smoother.rob_length, smoother.rob_width, smoother.center_offset)

    # VISUALIZATION: Draw covering circles for the first robot (index 0)
    # This helps verify the collision model
    idx_first = 0
    cx_base = ox[idx_first]
    cy_base = oy[idx_first]
    cth = oth[idx_first]
    
    for offset in smoother.circle_offsets:
        # Calculate circle center in global frame
        # offset is along the robot's x-axis (heading)
        cur_cx = cx_base + offset * np.cos(cth)
        cur_cy = cy_base + offset * np.sin(cth)
        
        # Draw Circle
        circle = plt.Circle((cur_cx, cur_cy), smoother.circle_radius, 
                            color='m', fill=False, linestyle='-', linewidth=1.5, alpha=0.8, label='Coverage Limit')
        plt.gca().add_patch(circle)
        # Plot center
        plt.plot(cur_cx, cur_cy, 'm.', markersize=3)

    plt.title("Kinematic Smoothing in '' Corridor")
    plt.legend()
    # plt.axis('equal') 
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kinematic_corridor_result.png")
    print("Saved kinematic_corridor_result.png")
    plt.show()

if __name__ == "__main__":
    main()
