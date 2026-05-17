from __future__ import annotations

import os
import sys
import traceback

from flask import Flask, jsonify, render_template


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
for path_entry in (REPO_ROOT, CURRENT_DIR):
    if path_entry not in sys.path:
        sys.path.append(path_entry)

from my.teb_local_controller.demo_support import describe_demo_configuration, run_random_demo


app = Flask(__name__)


def _state_to_dict(state):
    return {
        "x": float(state.x),
        "y": float(state.y),
        "theta": float(state.theta),
        "v": float(state.v),
        "a": float(state.a),
        "kappa": float(state.kappa),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/random_demo", methods=["POST"])
def random_demo():
    try:
        initial_state, reference, solution = run_random_demo()
        return jsonify(
            {
                "success": True,
                "config": describe_demo_configuration(),
                "initial_state": _state_to_dict(initial_state),
                "reference": {
                    "x": reference.x.tolist(),
                    "y": reference.y.tolist(),
                    "theta": reference.theta.tolist(),
                    "v": reference.v.tolist(),
                    "a": reference.a.tolist(),
                    "kappa": reference.kappa.tolist(),
                    "s": reference.s.tolist(),
                    "dt_ref": float(reference.dt_ref),
                },
                "solution": {
                    "x": solution["x"].tolist(),
                    "y": solution["y"].tolist(),
                    "theta": solution["theta"].tolist(),
                    "v": solution["v"].tolist(),
                    "a": solution["a"].tolist(),
                    "kappa": solution["kappa"].tolist(),
                    "dt": solution["dt"].tolist(),
                    "jerk": solution["jerk"].tolist(),
                    "dkappa": solution["dkappa"].tolist(),
                    "time": solution["time"].tolist(),
                    "solve_time_ms": float(solution["solve_time_ms"]),
                    "costs": solution["costs"],
                },
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)