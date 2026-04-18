import os
import sys
import numpy as np
import traceback
from flask import Flask, request, jsonify, render_template

# Add current directory to path to import nonlinear_smoother
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nonlinear_smoother import generate_reference_path, NonlinearPathSmoother

app = Flask(__name__)


def _to_float_list(values):
    if not isinstance(values, list):
        return None

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def _compute_reference_headings(x_ref, y_ref, gears, start_theta, goal_theta):
    num_points = len(x_ref)
    if num_points == 0:
        return np.array([])
    if num_points == 1:
        return np.array([start_theta], dtype=float)

    segment_headings = []
    previous_heading = start_theta
    for idx in range(num_points - 1):
        dx = x_ref[idx + 1] - x_ref[idx]
        dy = y_ref[idx + 1] - y_ref[idx]
        if np.hypot(dx, dy) > 1e-9:
            heading = np.arctan2(dy, dx)
            if gears[idx] < 0:
                heading += np.pi
            previous_heading = heading
        segment_headings.append(previous_heading)

    theta_ref = np.zeros(num_points, dtype=float)
    theta_ref[0] = start_theta
    theta_ref[-1] = goal_theta
    for idx in range(1, num_points - 1):
        prev_heading = segment_headings[idx - 1]
        next_heading = segment_headings[idx]
        theta_ref[idx] = np.arctan2(
            np.sin(prev_heading) + np.sin(next_heading),
            np.cos(prev_heading) + np.cos(next_heading),
        )

    return np.unwrap(theta_ref)


def _parse_custom_reference(custom_reference, fallback_gears, start_theta, goal_theta):
    if not isinstance(custom_reference, dict):
        return None

    x_values = _to_float_list(custom_reference.get('x'))
    y_values = _to_float_list(custom_reference.get('y'))
    theta_values = _to_float_list(custom_reference.get('theta'))
    gear_values = _to_float_list(custom_reference.get('gears'))

    if x_values is None or y_values is None or len(x_values) != len(y_values) or len(x_values) < 2:
        return None

    expected_gear_count = len(x_values) - 1
    if gear_values is not None and len(gear_values) == expected_gear_count:
        gears = np.where(np.array(gear_values, dtype=float) >= 0.0, 1.0, -1.0)
    elif fallback_gears is not None and len(fallback_gears) == expected_gear_count:
        gears = np.where(np.array(fallback_gears, dtype=float) >= 0.0, 1.0, -1.0)
    else:
        gears = np.ones(expected_gear_count, dtype=float)

    x_ref = np.array(x_values, dtype=float)
    y_ref = np.array(y_values, dtype=float)
    if theta_values is not None and len(theta_values) == len(x_values):
        theta_ref = np.unwrap(np.array(theta_values, dtype=float))
    else:
        theta_ref = _compute_reference_headings(x_ref, y_ref, gears, start_theta, goal_theta)

    return x_ref, y_ref, theta_ref, gears

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/smooth', methods=['POST'])
def run_smoother():
    try:
        req_data = request.get_json(silent=True) or {}
        params = req_data.get('params', {})
        
        # Ensure parameter values are float
        for k in params:
            try:
                params[k] = float(params[k])
            except (TypeError, ValueError):
                pass
                
        # Generate reference path with user endpoints
        start_x = params.get('start_x', 0.0)
        start_y = params.get('start_y', 0.0)
        start_theta = params.get('start_theta', 0.0)
        goal_x = params.get('goal_x', 20.0)
        goal_y = params.get('goal_y', 0.0)
        goal_theta = params.get('goal_theta', 0.0)
        target_ds = params.get('target_ds', 0.4)
        
        # If target_ds is 0, use a default for reference generation
        ref_ds = target_ds if target_ds > 0.05 else 0.4
        
        # Calculate turning radius from max_kappa parameter
        max_kappa = params.get('max_kappa', 0.5)
        turning_radius = 1.0 / max_kappa if max_kappa > 0.01 else 5.0
        
        # Handle optional start curvature
        fix_start_kappa = params.get('fix_start_kappa', True)
        if fix_start_kappa:
            params['kappa_start'] = params.get('start_kappa', 0.0)
        else:
            params['kappa_start'] = None
            
        # Handle initialization strategy
        use_dubins = params.get('use_dubins', True)
        
        nominal_x_ref, nominal_y_ref, nominal_theta_ref, nominal_gears, dubins_commands = generate_reference_path(
            start_x, start_y, start_theta, goal_x, goal_y, goal_theta, 
            target_ds=ref_ds, turning_radius=turning_radius, use_dubins=use_dubins
        )
        
        if nominal_x_ref is None:
            return jsonify({'success': False, 'message': 'Failed to generate initial Dubins path.'})

        custom_reference = _parse_custom_reference(
            req_data.get('custom_reference'),
            nominal_gears,
            start_theta,
            goal_theta,
        )
        if custom_reference is not None:
            x_ref, y_ref, theta_ref, gears = custom_reference
            reference_source = 'custom'
        else:
            x_ref, y_ref, theta_ref, gears = nominal_x_ref, nominal_y_ref, nominal_theta_ref, nominal_gears
            reference_source = 'generated'
            
        # Initialize smoother object and run NLP smoother
        smoother = NonlinearPathSmoother(params)
        res = smoother.solve(x_ref, y_ref, theta_ref, gears)
        x_opt, y_opt, theta_opt, kappa_opt, ds_opt, dkappa_opt, gears_opt, solve_time, target_ds_mag, costs = res
        
        formatted_commands = []
        if dubins_commands:
            for cmd in dubins_commands:
                cmd_type = "S"
                if abs(cmd.curvature) > 1e-6:
                    cmd_type = "L" if cmd.curvature > 0 else "R"
                formatted_commands.append({
                    "length": float(cmd.length),
                    "curvature": float(cmd.curvature),
                    "type": cmd_type
                })
                
        if kappa_opt is None:
            # When it fails, it returns debug values but None for kappa
            return jsonify({
                'success': False, 
                'message': 'Optimization failed to converge.',
                'solve_time_ms': float(solve_time),
                'target_ds_mag': float(target_ds_mag),
                'x_ref': x_ref.tolist(),
                'y_ref': y_ref.tolist(),
                'theta_ref': theta_ref.tolist(),
                'gears': gears.tolist(),
                'reference_source': reference_source,
                'x_opt': np.array(x_opt).tolist() if x_opt is not None else [],
                'y_opt': np.array(y_opt).tolist() if y_opt is not None else [],
                'dubins_commands': formatted_commands
            })
            
        return jsonify({
            'success': True,
            'solve_time_ms': float(solve_time),
            'target_ds_mag': float(target_ds_mag),
            'x_ref': x_ref.tolist(),
            'y_ref': y_ref.tolist(),
            'theta_ref': theta_ref.tolist(),
            'gears': gears.tolist(),
            'reference_source': reference_source,
            'x_opt': np.array(x_opt).tolist(),
            'y_opt': np.array(y_opt).tolist(),
            'theta_opt': np.array(theta_opt).tolist(),
            'gears_opt': np.array(gears_opt).tolist(),
            'kappa_opt': np.array(kappa_opt).tolist(),
            'ds_opt': np.array(ds_opt).tolist(),
            'dkappa_opt': np.array(dkappa_opt).tolist(),
            'dubins_commands': formatted_commands,
            'costs': costs
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
