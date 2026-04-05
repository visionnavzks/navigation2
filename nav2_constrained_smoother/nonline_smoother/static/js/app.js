document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const statusMsg = document.getElementById('status-msg');

    const params = [
        'start_x', 'start_y', 'start_theta',
        'goal_x', 'goal_y', 'goal_theta',
        'max_kappa', 'target_ds', 'w_ref', 'w_dkappa', 'w_kappa', 'w_ds'
    ];

    const plotCharts = (data) => {
        // Shared layout styling
        const chartLayout = {
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f1f5f9', family: 'Outfit' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.1)', zerolinecolor: 'rgba(255,255,255,0.2)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.1)', zerolinecolor: 'rgba(255,255,255,0.2)' },
            margin: { t: 40, b: 40, l: 50, r: 20 },
            legend: { orientation: 'h', y: 1.1 }
        };

        // 1. Path Plot
        const traceRef = {
            x: data.x_ref,
            y: data.y_ref,
            mode: 'lines+markers',
            name: 'Reference Path',
            line: {color: '#ef4444', dash: 'dash', width: 2},
            marker: {size: 4}
        };
        const traceOpt = {
            x: data.x_opt,
            y: data.y_opt,
            mode: 'lines+markers',
            name: 'Smoothed Path',
            line: {color: '#3b82f6', width: 3},
            marker: {size: 6}
        };
        Plotly.newPlot('path-chart', [traceRef, traceOpt], {
            ...chartLayout,
            title: 'Path Geometry (X vs Y)',
            yaxis: { ...chartLayout.yaxis, scaleanchor: 'x', scaleratio: 1 }
        });

        // 2. Curvature Plot
        if (data.kappa_opt && data.kappa_opt.length > 0) {
            const traceKappa = {
                y: data.kappa_opt,
                type: 'scatter',
                mode: 'lines',
                name: 'Curvature κ',
                line: {color: '#10b981', width: 2}
            };
            Plotly.newPlot('kappa-chart', [traceKappa], {
                ...chartLayout,
                title: 'Curvature Profile'
            });
        } else {
            document.getElementById('kappa-chart').innerHTML = '';
        }

        // 3. Control Outputs Plot
        if (data.ds_opt && data.dkappa_opt && data.ds_opt.length > 0) {
            const traceDs = {
                y: data.ds_opt,
                type: 'scatter',
                mode: 'lines',
                name: 'Step Size ds',
                line: {color: '#8b5cf6', width: 2}
            };
            const traceDkappa = {
                y: data.dkappa_opt,
                type: 'scatter',
                mode: 'lines',
                name: 'Curvature Deriv dκ',
                line: {color: '#f59e0b', width: 2}
            };
            Plotly.newPlot('control-chart', [traceDs, traceDkappa], {
                ...chartLayout,
                title: 'Control Outputs'
            });
        } else {
            document.getElementById('control-chart').innerHTML = '';
        }
    };

    const runOptimization = async () => {
        runBtn.textContent = 'Optimizing...';
        runBtn.disabled = true;
        statusMsg.textContent = 'Running nonlinear solver...';
        statusMsg.className = 'status-msg';

        const payload = { params: {} };
        params.forEach(param => {
            payload.params[param] = parseFloat(document.getElementById(param).value);
        });

        try {
            const res = await fetch('/api/smooth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success) {
                statusMsg.textContent = 'Optimization solved successfully!';
                statusMsg.className = 'status-msg success';
                plotCharts(data);
            } else {
                statusMsg.textContent = data.message || 'Optimization failed.';
                statusMsg.className = 'status-msg error';
                // Still plot whatever we have (maybe reference path)
                if (data.x_ref) {
                    plotCharts(data);
                }
            }
        } catch (err) {
            console.error(err);
            statusMsg.textContent = 'Network or server error occurred.';
            statusMsg.className = 'status-msg error';
        }

        runBtn.textContent = 'Run Optimization';
        runBtn.disabled = false;
    };

    runBtn.addEventListener('click', runOptimization);
    
    // Update span values when slider moves and run optimization on release
    params.forEach(param => {
        const input = document.getElementById(param);
        const span = document.getElementById(`val_${param}`);
        input.addEventListener('input', () => {
            span.textContent = parseFloat(input.value).toFixed(2);
        });
        input.addEventListener('change', runOptimization);
    });
    
    // Initial run
    runOptimization();
});
