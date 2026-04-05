import os
import sys
import numpy as np
import traceback
from flask import Flask, request, jsonify, render_template

# Add current directory to path to import nonline_smoother
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nonline_smoother import generate_reference_path, NonlinearPathSmoother

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
        
        x_ref, y_ref, theta_ref, dir_ref = generate_reference_path(
            start_x, start_y, start_theta, goal_x, goal_y, goal_theta, 
            target_ds=ref_ds, turning_radius=turning_radius
        )
        
        # Add small noise to middle points for visual jitter if desired (optional)
        # x_ref[1:-1] += np.random.normal(0, 0.02, len(x_ref)-2)
        # y_ref[1:-1] += np.random.normal(0, 0.02, len(y_ref)-2)
        
        # Initialize smoother object and run NLP smoother
        smoother = NonlinearPathSmoother(params)
        x_opt, y_opt, theta_opt, kappa_opt, ds_opt, dkappa_opt, dir_opt = smoother.solve(x_ref, y_ref, theta_ref, dir_ref)
        
        if kappa_opt is None:
            # When it fails, it returns debug values but None for kappa
            return jsonify({
                'success': False, 
                'message': 'Optimization failed to converge.',
                'x_ref': x_ref.tolist(),
                'y_ref': y_ref.tolist(),
                'dir_ref': dir_ref.tolist(),
                'x_opt': np.array(x_opt).tolist() if x_opt is not None else [],
                'y_opt': np.array(y_opt).tolist() if y_opt is not None else []
            })
            
        return jsonify({
            'success': True,
            'x_ref': x_ref.tolist(),
            'y_ref': y_ref.tolist(),
            'dir_ref': dir_ref.tolist(),
            'x_opt': np.array(x_opt).tolist(),
            'y_opt': np.array(y_opt).tolist(),
            'dir_opt': np.array(dir_opt).tolist(),
            'kappa_opt': np.array(kappa_opt).tolist(),
            'ds_opt': np.array(ds_opt).tolist(),
            'dkappa_opt': np.array(dkappa_opt).tolist()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
