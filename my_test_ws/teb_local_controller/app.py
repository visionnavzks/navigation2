from __future__ import annotations

import os
import sys
import traceback

from flask import Flask, jsonify, render_template, request


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
for path_entry in (REPO_ROOT, CURRENT_DIR):
    if path_entry not in sys.path:
        sys.path.append(path_entry)

from demo_support import default_demo_reference, describe_demo_configuration, run_random_demo, solve_demo
from teb_mpc import VehicleState


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


def _dict_to_state(payload):
    return VehicleState(
        x=float(payload["x"]),
        y=float(payload["y"]),
        theta=float(payload["theta"]),
        v=float(payload.get("v", 0.0)),
        a=float(payload.get("a", 0.0)),
        kappa=float(payload.get("kappa", 0.0)),
    )


def _reference_to_dict(reference):
    return {
        "x": reference.x.tolist(),
        "y": reference.y.tolist(),
        "theta": reference.theta.tolist(),
        "v": reference.v.tolist(),
        "a": reference.a.tolist(),
        "kappa": reference.kappa.tolist(),
        "s": reference.s.tolist(),
        "dt_ref": float(reference.dt_ref),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/random_demo", methods=["POST"])
def random_demo():
    try:
        payload = request.get_json(silent=True) or {}
        controller_params = payload.get("controller_params") or {}
        reference_config = payload.get("reference_config") or {}
        sampling_config = payload.get("sampling_config") or {}
        seed = payload.get("seed")
        initial_state_override = payload.get("initial_state_override")
        display_reference = default_demo_reference(reference_config=reference_config)

        if initial_state_override is not None:
            initial_state, reference, solution = solve_demo(
                initial_state=_dict_to_state(initial_state_override),
                params=controller_params,
                reference_config=reference_config,
            )
        else:
            initial_state, reference, solution = run_random_demo(
                seed=seed,
                params=controller_params,
                reference_config=reference_config,
                sampling_config=sampling_config,
            )
        return jsonify(
            {
                "success": True,
                "optimization": {
                    "succeeded": True,
                    "message": str(solution.get("solver_status", "Optimization succeeded")),
                },
                "config": describe_demo_configuration(
                    params=controller_params,
                    reference_config=reference_config,
                    sampling_config=sampling_config,
                ),
                "initial_state": _state_to_dict(initial_state),
                "reference": _reference_to_dict(reference),
                "display_reference": _reference_to_dict(display_reference),
                "reference_meta": solution.get("reference_meta", {}),
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
                    "solver_status": str(solution.get("solver_status", "Optimization succeeded")),
                    "costs": solution["costs"],
                    "cost_items": solution.get("cost_items", []),
                },
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify(
            {
                "success": False,
                "message": str(exc),
                "optimization": {
                    "succeeded": False,
                    "message": str(exc),
                },
            }
        ), 500


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "1") not in {"0", "false", "False"},
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5002")),
    )
