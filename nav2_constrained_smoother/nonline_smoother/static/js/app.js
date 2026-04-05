document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const statusMsg = document.getElementById('status-msg');

    const params = [
        'start_x', 'start_y', 'start_theta',
        'goal_x', 'goal_y', 'goal_theta',
        'max_kappa', 'target_ds', 'w_ref', 'w_dkappa', 'w_kappa', 'w_ds'
    ];

    const getSegmentTraces = (x, y, directions, namePrefix, isRef) => {
        if (!x || x.length === 0) return [];
        const traces = [];
        let currentDir = directions[0];
        let startIdx = 0;

        for (let i = 1; i <= x.length; i++) {
            if (i === x.length || directions[i] !== currentDir) {
                const segmentX = x.slice(startIdx, i + 1);
                const segmentY = y.slice(startIdx, i + 1);
                
                const isForward = currentDir >= 0;
                
                // Premium Color Palette
                const colors = {
                    smoothFwd: '#6366f1', // Indigo 500
                    smoothBwd: '#f43f5e', // Rose 500
                    refFwd: '#94a3b8',    // Slate 400
                    refBwd: '#fda4af'     // Rose 200 (muted)
                };

                let color;
                if (isRef) {
                    color = isForward ? colors.refFwd : colors.refBwd;
                } else {
                    color = isForward ? colors.smoothFwd : colors.smoothBwd;
                }

                const label = `${namePrefix} (${isForward ? 'Forward' : 'Backward'})`;
                
                traces.push({
                    x: segmentX,
                    y: segmentY,
                    mode: isRef ? 'lines' : 'lines+markers',
                    name: label,
                    line: {
                        color: color, 
                        dash: isRef ? 'dot' : 'solid',
                        width: isRef ? 2 : 4,
                        shape: 'spline',
                        smoothing: 1.3
                    },
                    marker: isRef ? {} : { 
                        size: 6, 
                        color: color,
                        line: { color: '#ffffff', width: 1 }
                    },
                    legendgroup: label,
                    showlegend: traces.findIndex(t => t.name === label) === -1
                });
                
                if (i < x.length) {
                    startIdx = i;
                    currentDir = directions[i];
                }
            }
        }
        return traces;
    };

    const plotCharts = (data) => {
        const chartLayout = {
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f1f5f9', family: 'Outfit' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
            margin: { t: 60, b: 40, l: 50, r: 20 },
            legend: { 
                orientation: 'h', 
                y: -0.2, 
                x: 0.5, 
                xanchor: 'center',
                font: { size: 10 }
            }
        };

        const refTraces = getSegmentTraces(data.x_ref, data.y_ref, data.dir_ref || new Array(data.x_ref.length).fill(1), 'Ref', true);
        const optTraces = getSegmentTraces(data.x_opt, data.y_opt, data.dir_opt || new Array(data.x_opt.length).fill(1), 'Smooth', false);

        Plotly.newPlot('path-chart', [...refTraces, ...optTraces], {
            ...chartLayout,
            title: 'Path Geometry (Indigo: Fwd, Rose: Bwd)',
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
