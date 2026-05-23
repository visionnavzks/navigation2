
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from zhengli.kinematic_smoother.kinematic_smoother import KinematicSmoother

def create_cusp_path():
    # Forward segment: (0,0) to (10,0)
    x1 = np.linspace(0, 10, 20)
    y1 = np.zeros_like(x1)
    
    # Backward segment: (10,0) to (0, 5)
    # Linearly interpolate
    x2 = np.linspace(10, 0, 20)
    y2 = np.linspace(0, 5, 20)
    
    # Combine (remove duplicate point at 10,0 to mimic raw path connection)
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    
    path = np.column_stack([x, y])
    
    # Gears
    # Segment 0..18 (19 points) -> Forward
    # Segment 19..37 (19 points) -> Backward
    # Total points: 20 + 20 - 1 = 39. Segments: 38.
    # First 19 segments are +1. Next 19 segments are -1.
    
    gears = np.ones(len(path) - 1)
    gears[19:] = -1
    
    return path, gears

def main():
    path, gears = create_cusp_path()
    
    smoother = KinematicSmoother(
        w_model=50.0,
        ref_weight=0.5,
        w_smooth=20.0,
        w_s=1.0,
        w_fix=100.0,
        target_spacing=0.5,
        max_iter=100
    )
    
    print("Optimizing...")
    opt_vars = smoother.optimize(path, gears)
    print("Optimization done.")
    
    # Extract results
    x = opt_vars[:, 0]
    y = opt_vars[:, 1]
    theta = opt_vars[:, 2]
    kappa = opt_vars[:, 3]
    ds = opt_vars[:, 4]
    
    # Identify cusp point for plotting (where ds ~ 0)
    is_cusp = ds < 1e-3
    
    plt.figure(figsize=(15, 10))
    
    # 1. Path X-Y
    plt.subplot(2, 2, 1)
    plt.plot(path[:, 0], path[:, 1], 'k--', label='Raw Path', marker='x', alpha=0.5)
    plt.plot(x, y, 'b.-', label='Optimized')
    # Plot orientation quivers
    step = 2
    plt.quiver(x[::step], y[::step], np.cos(theta[::step]), np.sin(theta[::step]), scale=20, color='r', width=0.005)
    plt.title("Path & Orientation (Red Quivers)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    # 2. Theta
    plt.subplot(2, 2, 2)
    plt.plot(theta, '.-')
    plt.title("Theta profile")
    plt.grid(True)
    
    # 3. Kappa
    plt.subplot(2, 2, 3)
    plt.plot(kappa, '.-')
    plt.title("Curvature (Kappa)")
    plt.grid(True)
    
    # 4. ds (Spacing)
    plt.subplot(2, 2, 4)
    plt.plot(ds, '.-')
    plt.title("Spacing (Delta S)")
    plt.hlines(0.5, 0, len(ds), colors='r', linestyles='--')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("kinematic_demo_result.png")
    plt.show()
    print("Result saved to kinematic_demo_result.png")

if __name__ == "__main__":
    main()
