import numpy as np
import matplotlib.pyplot as plt
from nonlinear_smoother import NonlinearPathSmoother

def test_cusp_smoother():
    # Construct a reference path with a cusp at (10, 0)
    # Forward right turn
    t_f = np.linspace(0, np.pi/2, 20)
    R = 5.0
    x_f = R * np.sin(t_f)
    y_f = R * (1 - np.cos(t_f))
    theta_f = t_f
    gears_f = np.ones(19)
    # Target kappa_f = 1/R = 0.2
    
    # Backward left turn (starting from endpoint of f)
    t_b = np.linspace(np.pi/2, 0, 20)
    R2 = 2.0
    # Re-calculate x_b, y_b so it starts at endpoint of f
    # Endpoint of f: (5, 5)
    # x_f(pi/2) = 5*sin(pi/2) = 5
    # y_f(pi/2) = 5*(1-cos(pi/2)) = 5
    # Now circle with R2=2.0 starting at (5,5) with orientation pi/2
    # Moving backwards.
    # Center of circle for backward left turn: (5 - R2*cos(pi/2), 5 - R2*sin(pi/2)) = (5, 3)
    x_b = 5 + R2 * (np.sin(t_b) - 1.0)
    y_b = 3 + R2 * (1.0 - np.cos(t_b))
    theta_b = t_b 
    gears_b = -np.ones(19)
    # Target kappa_b should be approx 1/R2 = 0.5
    
    x_ref = np.concatenate([x_f, x_b[1:]])
    y_ref = np.concatenate([y_f, y_b[1:]])
    theta_ref = np.concatenate([theta_f, theta_b[1:]])
    gears = np.concatenate([gears_f, gears_b])
    
    print(f"Total reference points: {len(x_ref)}")
    print(f"Total gears: {len(gears)}")
    
    planner = NonlinearPathSmoother(params={'w_ref': 100.0, 'w_kappa': 0.1, 'w_dkappa': 10.0})
    result = planner.solve(x_ref, y_ref, theta_ref, gears)
    
    x_opt, y_opt, theta_opt, kappa_opt, ds_opt, dkappa_opt, gears_opt = result
    
    if x_opt is not None:
        print("Optimization succeeded!")
        print(f"Optimized path length: {len(x_opt)}")
        
        # Check for cusp jump
        # Find where ds is near 0
        cusp_indices = np.where(np.abs(ds_opt) < 1e-6)[0]
        print(f"Cusp indices found: {cusp_indices}")
        
        for idx in cusp_indices:
            print(f"At cusp index {idx}:")
            print(f"  Pos: ({x_opt[idx]:.3f}, {y_opt[idx]:.3f}) -> ({x_opt[idx+1]:.3f}, {y_opt[idx+1]:.3f})")
            print(f"  Theta: {theta_opt[idx]:.3f} -> {theta_opt[idx+1]:.3f}")
            print(f"  Kappa: {kappa_opt[idx]:.3f} -> {kappa_opt[idx+1]:.3f} (Jump: {kappa_opt[idx+1]-kappa_opt[idx]:.3f})")

        # Plotting
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_ref, y_ref, 'r--', label='Reference')
        plt.plot(x_opt, y_opt, 'b-', label='Optimized')
        plt.scatter(x_opt[cusp_indices], y_opt[cusp_indices], color='green', s=100, label='Cusp')
        plt.title("Path")
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(2, 1, 2)
        plt.plot(kappa_opt, 'g-', label='Curvature')
        plt.title("Curvature Profile")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('cusp_test_result.png')
        print("Saved result to cusp_test_result.png")
    else:
        print("Optimization failed.")

if __name__ == "__main__":
    test_cusp_smoother()
