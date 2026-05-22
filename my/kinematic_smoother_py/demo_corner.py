
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from zhengli.kinematic_smoother.kinematic_smoother import KinematicSmoother

def create_corner_path():
    # Segment 1: (0,0) to (10,0)
    x1 = np.linspace(0, 10, 20)
    y1 = np.zeros_like(x1)
    
    # Segment 2: (10,0) to (10, 10)
    # We start slightly after 10 to simulate a discrete corner, 
    # but let's connect them at (10,0).
    # NOTE: x remains 10, y goes 0 -> 10
    x2 = np.ones(20) * 10
    y2 = np.linspace(0, 10, 20)
    
    # Concatenate. Remove duplicate at (10,0)
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    
    # Create orientation (yaw)
    # We only care about start and end for constraints
    thetas = np.zeros_like(x)
    thetas[0] = np.pi # Start facing 180 degrees (Left)
    thetas[-1] = np.pi / 2 # End facing 90 degrees (Up)
    
    path = np.column_stack([x, y, thetas])
    
    # Gears: All forward (+1)
    gears = np.ones(len(path) - 1)
    
    return path, gears

def main():
    path, gears = create_corner_path()
    
    # A generic "Ackermann-like" or differential constraints
    smoother = KinematicSmoother(
        w_model=100.0,       # Strict kinematic model
        ref_weight=0.01,     # Low weight to allow U-turn deviation
        w_smooth=50.0,       # High smoothing for corners
        w_s=1.0,             # Regularize spacing
        w_fix=100.0,         # Fix start/end
        target_spacing=0.5,
        max_iter=500
    )
    
    print("Optimizing Corner Path...")
    opt_vars = smoother.optimize(path, gears)
    print("Optimization done.")
    
    # Extract results
    x = opt_vars[:, 0]
    y = opt_vars[:, 1]
    theta = opt_vars[:, 2]
    kappa = opt_vars[:, 3]
    ds = opt_vars[:, 4]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Path X-Y
    plt.subplot(2, 2, 1)
    plt.plot(path[:, 0], path[:, 1], 'k--', label='Raw Path (Corner)', marker='x', alpha=0.5)
    plt.plot(x, y, 'b.-', label='Optimized')
    
    # Plot orientation quivers
    step = 2
    plt.quiver(x[::step], y[::step], np.cos(theta[::step]), np.sin(theta[::step]), 
               scale=20, color='r', width=0.005, label='Orientation')
    
    plt.title("Path & Orientation")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    # 2. Theta
    plt.subplot(2, 2, 2)
    plt.plot(theta, '.-')
    plt.title("Theta profile (Rad)")
    plt.grid(True)
    
    # 3. Kappa
    plt.subplot(2, 2, 3)
    plt.plot(kappa, '.-')
    plt.grid(True)
    # Draw limits
    min_turning_radius = 2.0
    max_k = 1.0 / min_turning_radius
    plt.hlines([max_k, -max_k], 0, len(kappa), colors='r', linestyles='--', label='Max Curvature')
    plt.legend()
    plt.title(f"Curvature (Max K={max_k:.2f})")
    
    # 4. ds (Spacing)
    plt.subplot(2, 2, 4)
    plt.plot(ds, '.-')
    plt.title("Spacing (Delta S)")
    plt.hlines(0.5, 0, len(ds), colors='r', linestyles='--')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("kinematic_corner_result.png")
    plt.show() # Non-blocking for script
    print("Result saved to kinematic_corner_result.png")

if __name__ == "__main__":
    main()
