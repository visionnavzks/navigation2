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
from teb_mpc import GoalPoint, TEBMPCController, VehicleState, build_goal_reference


app = Flask(__name__)

DEFAULT_GOAL_INITIAL_STATE = {
    "x": 0.0,
    "y": 0.0,
    "theta": 0.0,
    "v": 0.1,
    "a": 0.0,
    "kappa": 0.0,
}

GOAL_CONFIG_DEFAULTS = {
    "x": 2.0,
    "y": 1.0,
    "theta": 0.3490658504,
    "v": 0.0,
    "a": 0.0,
    "kappa": 0.0,
    "ds": 0.2,
    "cruise_speed": 0.8,
    "dt_ref": 0.1,
    "sample_count": 0,
}


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


def _solution_to_dict(solution):
    return {
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
        "resize_log": solution.get("resize_log", []),
        "resize_iterations": int(solution.get("resize_iterations", 0)),
        "playback": solution.get("playback"),
    }


def _error_response(exc):
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


def _merged_goal_config(goal_config):
    merged = {**GOAL_CONFIG_DEFAULTS, **(goal_config or {})}
    if merged.get("theta") == "":
        merged["theta"] = None
    if merged.get("sample_count") == "":
        merged["sample_count"] = 0
    return merged


def _goal_from_config(goal_config):
    theta = goal_config.get("theta")
    return GoalPoint(
        x=float(goal_config["x"]),
        y=float(goal_config["y"]),
        theta=None if theta is None else float(theta),
        v=float(goal_config["v"]),
        a=float(goal_config["a"]),
        kappa=float(goal_config["kappa"]),
    )


def _goal_sample_count(goal_config):
    sample_count = int(goal_config["sample_count"])
    return None if sample_count <= 0 else sample_count


def _goal_config_to_dict(goal, goal_config, reference_size):
    return {
        "x": goal.x,
        "y": goal.y,
        "theta": goal.theta,
        "v": goal.v,
        "a": goal.a,
        "kappa": goal.kappa,
        "ds": float(goal_config["ds"]),
        "cruise_speed": float(goal_config["cruise_speed"]),
        "dt_ref": float(goal_config["dt_ref"]),
        "sample_count": int(goal_config["sample_count"]),
        "resolved_sample_count": int(reference_size),
    }


def _goal_response_config(controller_params, goal, goal_config, reference):
    config = describe_demo_configuration(params=controller_params)
    config["reference"] = {
        **config["reference"],
        "ds": float(goal_config["ds"]),
        "cruise_speed": float(goal_config["cruise_speed"]),
        "dt_ref": float(goal_config["dt_ref"]),
        "params": {
            **config["reference"].get("params", {}),
            "ds": float(goal_config["ds"]),
            "cruise_speed": float(goal_config["cruise_speed"]),
            "dt_ref": float(goal_config["dt_ref"]),
        },
    }
    config["goal"] = _goal_config_to_dict(goal, goal_config, reference.size)
    return config


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
        terminal_theta_override = payload.get("terminal_theta_override")
        if terminal_theta_override == "":
            terminal_theta_override = None
        if terminal_theta_override is not None:
            terminal_theta_override = float(terminal_theta_override)
        record = bool(payload.get("record_iterations"))
        display_reference = default_demo_reference(reference_config=reference_config)

        if initial_state_override is not None:
            initial_state, reference, solution = solve_demo(
                initial_state=_dict_to_state(initial_state_override),
                params=controller_params,
                reference_config=reference_config,
                terminal_theta_override=terminal_theta_override,
                record=record,
            )
        else:
            initial_state, reference, solution = run_random_demo(
                seed=seed,
                params=controller_params,
                reference_config=reference_config,
                sampling_config=sampling_config,
                terminal_theta_override=terminal_theta_override,
                record=record,
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
                "solution": _solution_to_dict(solution),
            }
        )
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/goal_demo", methods=["POST"])
def goal_demo():
    try:
        payload = request.get_json(silent=True) or {}
        controller_params = payload.get("controller_params") or {}
        goal_config = _merged_goal_config(payload.get("goal_config"))
        initial_state_payload = payload.get("initial_state_override") or DEFAULT_GOAL_INITIAL_STATE
        initial_state = _dict_to_state(initial_state_payload)
        goal = _goal_from_config(goal_config)
        record = bool(payload.get("record_iterations"))

        controller = TEBMPCController(params=controller_params)
        reference = build_goal_reference(
            start=initial_state,
            goal=goal,
            ds=float(goal_config["ds"]),
            cruise_speed=float(goal_config["cruise_speed"]),
            dt_ref=float(goal_config["dt_ref"]),
            sample_count=_goal_sample_count(goal_config),
        )
        solution = controller.solve_with_resize(initial_state=initial_state, reference=reference, record=record)
        reference = solution["resampled_reference"]
        solution["reference_meta"] = {
            "mode": "point_goal",
            "is_stopping_reference": False,
            "goal": {
                "x": goal.x,
                "y": goal.y,
                "theta": float(reference.theta[-1]),
                "v": goal.v,
                "a": goal.a,
                "kappa": goal.kappa,
            },
            "reference_size": int(len(solution["x"])),
            "reference_length": float(reference.s[-1]),
        }

        return jsonify(
            {
                "success": True,
                "optimization": {
                    "succeeded": True,
                    "message": str(solution.get("solver_status", "Optimization succeeded")),
                },
                "config": _goal_response_config(controller_params, goal, goal_config, reference),
                "initial_state": _state_to_dict(initial_state),
                "reference": _reference_to_dict(reference),
                "display_reference": _reference_to_dict(reference),
                "reference_meta": solution.get("reference_meta", {}),
                "solution": _solution_to_dict(solution),
            }
        )
    except Exception as exc:
        return _error_response(exc)


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "1") not in {"0", "false", "False"},
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5002")),
    )
