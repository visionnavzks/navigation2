import os
import sys
import numpy as np
import traceback
from flask import Flask, request, jsonify, render_template

# Add current directory to path to import nonlinear_smoother
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nonlinear_smoother import generate_reference_path, NonlinearPathSmoother

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/smooth', methods=['POST'])
def run_smoother():
    try:
        req_data = request.json
        params = req_data.get('params', {})
        
        # Ensure parameter values are float
        for k in params:
            try:
                params[k] = float(params[k])
            except ValueError:
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
        
        x_ref, y_ref, theta_ref, gears, dubins_commands = generate_reference_path(
            start_x, start_y, start_theta, goal_x, goal_y, goal_theta, 
            target_ds=ref_ds, turning_radius=turning_radius, use_dubins=use_dubins
        )
        
        if x_ref is None:
            return jsonify({'success': False, 'message': 'Failed to generate initial Dubins path.'})
            
        # Initialize smoother object and run NLP smoother
        smoother = NonlinearPathSmoother(params)
        res = smoother.solve(x_ref, y_ref, theta_ref, gears)
        x_opt, y_opt, theta_opt, kappa_opt, ds_opt, dkappa_opt, gears_opt, solve_time, target_ds_mag = res
        
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
                'gears': gears.tolist(),
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
            'gears': gears.tolist(),
            'x_opt': np.array(x_opt).tolist(),
            'y_opt': np.array(y_opt).tolist(),
            'gears_opt': np.array(gears_opt).tolist(),
            'kappa_opt': np.array(kappa_opt).tolist(),
            'ds_opt': np.array(ds_opt).tolist(),
            'dkappa_opt': np.array(dkappa_opt).tolist(),
            'dubins_commands': formatted_commands
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
