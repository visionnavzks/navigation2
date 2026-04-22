"""
Flask web application for path smoother tuning.
Supports two backends:
  1. Python Nonlinear Smoother (CasADi / IPOPT)
  2. C++ Kinematic Smoother (Ceres)
"""
import json
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Try to import the Python nonlinear smoother from the sibling directory.
# ---------------------------------------------------------------------------
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _repo_root)

try:
    from my.nonline_smoother.nonlinear_smoother import (
        NonlinearPathSmoother,
        generate_reference_path,
    )
    _PYTHON_SMOOTHER_AVAILABLE = True
except ImportError:
    _PYTHON_SMOOTHER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path to the C++ kinematic_smoother executable.
# The user should build it and either place it in the build/ directory or
# set the KINEMATIC_SMOOTHER_BIN environment variable.
# ---------------------------------------------------------------------------
_DEFAULT_BIN = os.path.join(os.path.dirname(__file__), '..', 'build', 'kinematic_smoother_node')
_CERES_BIN = os.environ.get('KINEMATIC_SMOOTHER_BIN', _DEFAULT_BIN)
_CERES_AVAILABLE = os.path.isfile(_CERES_BIN) and os.access(_CERES_BIN, os.X_OK)

app = Flask(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _to_float_list(values):
    if not isinstance(values, list):
        return None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def _compute_reference_headings(x_ref, y_ref, gears, start_theta, goal_theta):
    num = len(x_ref)
    if num == 0:
        return np.array([])
    if num == 1:
        return np.array([start_theta])

    seg = []
    prev = start_theta
    for i in range(num - 1):
        dx = x_ref[i + 1] - x_ref[i]
        dy = y_ref[i + 1] - y_ref[i]
        if np.hypot(dx, dy) > 1e-9:
            h = np.arctan2(dy, dx)
            if gears[i] < 0:
                h += np.pi
            prev = h
        seg.append(prev)

    theta = np.zeros(num)
    theta[0] = start_theta
    theta[-1] = goal_theta
    for i in range(1, num - 1):
        theta[i] = np.arctan2(
            np.sin(seg[i - 1]) + np.sin(seg[i]),
            np.cos(seg[i - 1]) + np.cos(seg[i]),
        )
    return np.unwrap(theta)


def _parse_custom_reference(custom_ref, fallback_gears, start_theta, goal_theta):
    if not isinstance(custom_ref, dict):
        return None
    x = _to_float_list(custom_ref.get('x'))
    y = _to_float_list(custom_ref.get('y'))
    th = _to_float_list(custom_ref.get('theta'))
    g = _to_float_list(custom_ref.get('gears'))
    if x is None or y is None or len(x) != len(y) or len(x) < 2:
        return None

    n = len(x) - 1
    if g is not None and len(g) == n:
        gears = np.where(np.array(g) >= 0, 1.0, -1.0)
    elif fallback_gears is not None and len(fallback_gears) == n:
        gears = np.where(np.array(fallback_gears) >= 0, 1.0, -1.0)
    else:
        gears = np.ones(n)

    x_ref = np.array(x)
    y_ref = np.array(y)
    if th is not None and len(th) == len(x):
        theta_ref = np.unwrap(np.array(th))
    else:
        theta_ref = _compute_reference_headings(x_ref, y_ref, gears, start_theta, goal_theta)
    return x_ref, y_ref, theta_ref, gears


# ============================================================================
# Backend: Python (CasADi / IPOPT)
# ============================================================================

def _run_python_smoother(params, x_ref, y_ref, theta_ref, gears, dubins_commands):
    smoother = NonlinearPathSmoother(params)
    res = smoother.solve(x_ref, y_ref, theta_ref, gears)
    (x_opt, y_opt, theta_opt, kappa_opt,
     ds_opt, dkappa_opt, gears_opt, solve_time, target_ds_mag, costs) = res

    formatted_cmds = _format_dubins(dubins_commands)

    if kappa_opt is None:
        return {
            'success': False,
            'message': 'Optimization failed to converge.',
            'solve_time_ms': float(solve_time),
            'target_ds_mag': float(target_ds_mag),
            'x_ref': x_ref.tolist(), 'y_ref': y_ref.tolist(),
            'theta_ref': theta_ref.tolist(), 'gears': gears.tolist(),
            'x_opt': np.asarray(x_opt).tolist() if x_opt is not None else [],
            'y_opt': np.asarray(y_opt).tolist() if y_opt is not None else [],
            'dubins_commands': formatted_cmds,
        }

    return {
        'success': True,
        'solve_time_ms': float(solve_time),
        'target_ds_mag': float(target_ds_mag),
        'x_ref': x_ref.tolist(), 'y_ref': y_ref.tolist(),
        'theta_ref': theta_ref.tolist(), 'gears': gears.tolist(),
        'x_opt': np.asarray(x_opt).tolist(),
        'y_opt': np.asarray(y_opt).tolist(),
        'theta_opt': np.asarray(theta_opt).tolist(),
        'gears_opt': np.asarray(gears_opt).tolist(),
        'kappa_opt': np.asarray(kappa_opt).tolist(),
        'ds_opt': np.asarray(ds_opt).tolist(),
        'dkappa_opt': np.asarray(dkappa_opt).tolist(),
        'dubins_commands': formatted_cmds,
        'costs': costs,
    }


# ============================================================================
# Backend: C++ / Ceres (via subprocess)
# ============================================================================

def _run_ceres_smoother(params, x_ref, y_ref, theta_ref, gears,
                        dubins_commands, obstacles=None):
    """Call the C++ kinematic_smoother_node executable via subprocess."""
    # Build input JSON
    payload = {
        'x_ref': x_ref.tolist() if hasattr(x_ref, 'tolist') else list(x_ref),
        'y_ref': y_ref.tolist() if hasattr(y_ref, 'tolist') else list(y_ref),
        'theta_ref': theta_ref.tolist() if hasattr(theta_ref, 'tolist') else list(theta_ref),
        'gears': gears.tolist() if hasattr(gears, 'tolist') else list(gears),
        'params': {
            'max_kappa': params.get('max_kappa', 0.5),
            'w_ref': params.get('w_ref', 10.0),
            'w_dkappa': params.get('w_dkappa', 10.0),
            'w_kappa': params.get('w_kappa', 0.1),
            'w_ds': params.get('w_ds', 1.0),
            'w_kinematic': params.get('w_kinematic', 1000.0),
            'target_ds': params.get('target_ds', 0.0),
            'ds_min_ratio': params.get('ds_min_ratio', 0.05),
            'ds_max_ratio': params.get('ds_max_ratio', 2.0),
            'max_iterations': int(params.get('max_iterations', 500)),
            'tolerance': float(params.get('tolerance', 1e-6)),
            'fix_start_kappa': bool(params.get('fix_start_kappa', True)),
            'kappa_start': params.get('kappa_start', 0.0),
            'w_esdf': params.get('w_esdf', 0.0),
            'esdf_safe_distance': params.get('esdf_safe_distance', 0.5),
        },
    }
    if obstacles:
        payload['obstacles'] = obstacles

    input_json = json.dumps(payload)

    try:
        proc = subprocess.run(
            [_CERES_BIN],
            input=input_json, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return {
                'success': False,
                'message': f'C++ smoother exited with code {proc.returncode}: {proc.stderr[:500]}',
                'x_ref': payload['x_ref'], 'y_ref': payload['y_ref'],
                'theta_ref': payload['theta_ref'], 'gears': payload['gears'],
                'dubins_commands': _format_dubins(dubins_commands),
            }
        result = json.loads(proc.stdout)
        result['x_ref'] = payload['x_ref']
        result['y_ref'] = payload['y_ref']
        result['theta_ref'] = payload['theta_ref']
        result['gears'] = payload['gears']
        result['dubins_commands'] = _format_dubins(dubins_commands)
        return result
    except FileNotFoundError:
        return {'success': False,
                'message': f'C++ smoother binary not found at {_CERES_BIN}'}
    except subprocess.TimeoutExpired:
        return {'success': False, 'message': 'C++ smoother timed out.'}
    except json.JSONDecodeError as e:
        return {'success': False, 'message': f'Invalid JSON from C++ smoother: {e}'}


def _format_dubins(commands):
    if not commands:
        return []
    out = []
    for cmd in commands:
        if hasattr(cmd, 'curvature'):
            k = cmd.curvature
            t = 'S' if abs(k) < 1e-6 else ('L' if k > 0 else 'R')
            out.append({'length': float(cmd.length), 'curvature': float(k), 'type': t})
        elif isinstance(cmd, dict):
            out.append(cmd)
    return out


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    return render_template(
        'index.html',
        python_available=_PYTHON_SMOOTHER_AVAILABLE,
        ceres_available=_CERES_AVAILABLE,
    )


@app.route('/api/smooth', methods=['POST'])
def run_smoother():
    try:
        req = request.get_json(silent=True) or {}
        params = req.get('params', {})
        backend = req.get('backend', 'python')  # 'python' or 'ceres'
        obstacles = req.get('obstacles', None)

        # Cast numeric params
        for k in params:
            try:
                params[k] = float(params[k])
            except (TypeError, ValueError):
                pass

        # --- Endpoint parameters ---
        start_x = params.get('start_x', 0.0)
        start_y = params.get('start_y', 0.0)
        start_theta = params.get('start_theta', 0.0)
        goal_x = params.get('goal_x', 20.0)
        goal_y = params.get('goal_y', 0.0)
        goal_theta = params.get('goal_theta', 0.0)
        target_ds = params.get('target_ds', 0.4)
        ref_ds = target_ds if target_ds > 0.05 else 0.4

        max_kappa = params.get('max_kappa', 0.5)
        turning_radius = 1.0 / max_kappa if max_kappa > 0.01 else 5.0

        fix_start_kappa = params.get('fix_start_kappa', True)
        if fix_start_kappa:
            params['kappa_start'] = params.get('start_kappa', 0.0)
        else:
            params['kappa_start'] = None

        use_dubins = params.get('use_dubins', True)

        # --- Generate reference path ---
        if not _PYTHON_SMOOTHER_AVAILABLE:
            return jsonify({'success': False,
                            'message': 'Python smoother not available for reference generation.'})

        nominal_x, nominal_y, nominal_theta, nominal_gears, dubins_commands = \
            generate_reference_path(
                start_x, start_y, start_theta,
                goal_x, goal_y, goal_theta,
                target_ds=ref_ds, turning_radius=turning_radius,
                use_dubins=use_dubins,
            )
        if nominal_x is None:
            return jsonify({'success': False, 'message': 'Reference path generation failed.'})

        custom = _parse_custom_reference(
            req.get('custom_reference'), nominal_gears, start_theta, goal_theta)
        if custom is not None:
            x_ref, y_ref, theta_ref, gears = custom
            ref_source = 'custom'
        else:
            x_ref, y_ref, theta_ref, gears = nominal_x, nominal_y, nominal_theta, nominal_gears
            ref_source = 'generated'

        # --- Dispatch to backend ---
        if backend == 'ceres':
            if not _CERES_AVAILABLE:
                return jsonify({'success': False,
                                'message': f'C++ smoother not found at {_CERES_BIN}. '
                                           'Build it or set KINEMATIC_SMOOTHER_BIN.'})
            result = _run_ceres_smoother(
                params, x_ref, y_ref, theta_ref, gears, dubins_commands, obstacles)
        else:
            result = _run_python_smoother(
                params, x_ref, y_ref, theta_ref, gears, dubins_commands)

        result['reference_source'] = ref_source
        result.setdefault('dubins_commands', _format_dubins(dubins_commands))
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/status')
def status():
    return jsonify({
        'python_available': _PYTHON_SMOOTHER_AVAILABLE,
        'ceres_available': _CERES_AVAILABLE,
        'ceres_bin': _CERES_BIN,
    })


if __name__ == '__main__':
    print(f'Python smoother: {"available" if _PYTHON_SMOOTHER_AVAILABLE else "NOT available"}')
    print(f'Ceres smoother : {"available" if _CERES_AVAILABLE else "NOT available"} ({_CERES_BIN})')
    app.run(debug=True, port=5001)
