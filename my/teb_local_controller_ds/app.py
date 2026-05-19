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

from my.teb_local_controller_ds.demo_support import describe_demo_configuration, run_random_demo, solve_demo
from my.teb_local_controller_ds.teb_mpc import SpatialState


app = Flask(__name__)


def _state_to_dict(state: SpatialState) -> dict[str, float]:
    return {
        "x": float(state.x),
        "y": float(state.y),
        "theta": float(state.theta),
        "kappa": float(state.kappa),
    }


def _dict_to_state(payload: dict[str, float]) -> SpatialState:
    return SpatialState(
        x=float(payload["x"]),
        y=float(payload["y"]),
        theta=float(payload["theta"]),
        kappa=float(payload.get("kappa", 0.0)),
    )


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
                "config": describe_demo_configuration(
                    params=controller_params,
                    reference_config=reference_config,
                    sampling_config=sampling_config,
                ),
                "initial_state": _state_to_dict(initial_state),
                "reference": {
                    "x": reference.x.tolist(),
                    "y": reference.y.tolist(),
                    "theta": reference.theta.tolist(),
                    "kappa": reference.kappa.tolist(),
                    "s": reference.s.tolist(),
                },
                "solution": {
                    "x": solution["x"].tolist(),
                    "y": solution["y"].tolist(),
                    "theta": solution["theta"].tolist(),
                    "kappa": solution["kappa"].tolist(),
                    "ds": solution["ds"].tolist(),
                    "dkappa": solution["dkappa"].tolist(),
                    "s": solution["s"].tolist(),
                    "solve_time_ms": float(solution["solve_time_ms"]),
                    "costs": solution["costs"],
                },
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "1") not in {"0", "false", "False"},
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5003")),
    )