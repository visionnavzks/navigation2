from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
for path_entry in (REPO_ROOT, CURRENT_DIR):
    if path_entry not in sys.path:
        sys.path.append(path_entry)

from my.teb_local_controller_ds.demo_support import demo_problem, run_random_demo


def _control_axis(solution: dict[str, np.ndarray]) -> np.ndarray:
    return 0.5 * (solution["s"][:-1] + solution["s"][1:])


def _plot_scene(initial_state, reference, solution) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    path_ax, theta_ax, kappa_ax, control_ax = axes.flat

    path_ax.plot(reference.x, reference.y, "--", color="#0f766e", linewidth=2.0, label="reference")
    path_ax.plot(solution["x"], solution["y"], color="#ca5a34", linewidth=2.2, marker="o", markersize=3.5, label="optimized")
    path_ax.scatter([initial_state.x], [initial_state.y], color="#d97706", s=70, label="initial")
    path_ax.set_title("Path")
    path_ax.set_xlabel("x [m]")
    path_ax.set_ylabel("y [m]")
    path_ax.axis("equal")
    path_ax.grid(True, alpha=0.25)
    path_ax.legend(loc="best")

    theta_ax.plot(reference.s, reference.theta, "--", color="#0f766e", linewidth=2.0, label="ref theta")
    theta_ax.plot(solution["s"], solution["theta"], color="#ca5a34", linewidth=2.2, marker="o", markersize=3.5, label="theta")
    theta_ax.set_title("Heading")
    theta_ax.set_xlabel("s [m]")
    theta_ax.set_ylabel("theta [rad]")
    theta_ax.grid(True, alpha=0.25)
    theta_ax.legend(loc="best")

    kappa_ax.plot(reference.s, reference.kappa, "--", color="#0f766e", linewidth=2.0, label="ref kappa")
    kappa_ax.plot(solution["s"], solution["kappa"], color="#d97706", linewidth=2.2, marker="o", markersize=3.5, label="kappa")
    kappa_ax.set_title("Curvature")
    kappa_ax.set_xlabel("s [m]")
    kappa_ax.set_ylabel("kappa [1/m]")
    kappa_ax.grid(True, alpha=0.25)
    kappa_ax.legend(loc="best")

    control_s = _control_axis(solution)
    control_ax.plot(control_s, solution["ds"], color="#8b5cf6", linewidth=2.2, marker="o", markersize=3.5, label="ds")
    control_ax.plot(control_s, solution["dkappa"], color="#0b4f6c", linewidth=2.2, marker="o", markersize=3.5, label="dkappa")
    control_ax.set_title("Controls")
    control_ax.set_xlabel("s [m]")
    control_ax.set_ylabel("value")
    control_ax.grid(True, alpha=0.25)
    control_ax.legend(loc="best")

    total_s = float(solution["s"][-1])
    avg_ds = float(np.mean(solution["ds"]))
    figure.suptitle(
        f"DS MPC Demo | solve {solution['solve_time_ms']:.2f} ms | total s {total_s:.2f} m | avg ds {avg_ds:.3f} m",
        fontsize=13,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="DS-parameterized MPC matplotlib demo")
    parser.add_argument("--random", action="store_true", help="Use a random initial state instead of the fixed demo state")
    parser.add_argument("--seed", type=int, default=None, help="Random seed when --random is used")
    parser.add_argument("--save", type=str, default=None, help="Save figure to a file instead of showing it")
    args = parser.parse_args()

    if args.random:
        initial_state, reference, solution = run_random_demo(seed=args.seed)
    else:
        initial_state, reference, solution = demo_problem()

    figure = _plot_scene(initial_state, reference, solution)
    if args.save:
        figure.savefig(args.save, dpi=180)
        print(f"Saved figure to {args.save}")
        plt.close(figure)
        return

    plt.show()


if __name__ == "__main__":
    main()