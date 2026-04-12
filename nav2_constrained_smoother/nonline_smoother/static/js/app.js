document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const statusMsg = document.getElementById('status-msg');

    const params = [
        'start_x', 'start_y', 'start_theta',
        'goal_x', 'goal_y', 'goal_theta',
        'max_kappa', 'target_ds', 'w_ref', 'w_dkappa', 'w_kappa', 'w_ds',
        'start_kappa', 'ipopt_max_iter', 'ipopt_print_level'
    ];

    const getSegmentTraces = (x, y, gears, namePrefix, isRef) => {
        if (!x || x.length < 2 || !gears || gears.length === 0) return [];
        const traces = [];
        let currentDir = gears[0];
        let startIdx = 0;

        for (let i = 1; i < x.length; i++) {
            // Check if we are at the last point, or if the next segment's gear is different
            if (i === x.length - 1 || gears[i] !== currentDir) {
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
                
                if (i < x.length - 1) {
                    startIdx = i;
                    currentDir = gears[i];
                }
            }
        }
        return traces;
    };

    const plotCharts = (data, params) => {
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

        const refTraces = getSegmentTraces(data.x_ref, data.y_ref, data.gears, 'Ref', true);
        const optTraces = getSegmentTraces(data.x_opt, data.y_opt, data.gears_opt, 'Smooth', false);

        const annotations = [
            {
                x: params.start_x + Math.cos(params.start_theta) * 1.5,
                y: params.start_y + Math.sin(params.start_theta) * 1.5,
                ax: params.start_x,
                ay: params.start_y,
                xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 4,
                arrowcolor: 'rgba(99, 102, 241, 0.5)' // Indigo with 0.5 alpha
            },
            {
                x: params.goal_x + Math.cos(params.goal_theta) * 1.5,
                y: params.goal_y + Math.sin(params.goal_theta) * 1.5,
                ax: params.goal_x,
                ay: params.goal_y,
                xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 4,
                arrowcolor: 'rgba(244, 63, 94, 0.5)' // Rose with 0.5 alpha
            }
        ];

        Plotly.newPlot('path-chart', [...refTraces, ...optTraces], {
            ...chartLayout,
            title: 'Path Geometry (Indigo: Fwd, Rose: Bwd)',
            yaxis: { ...chartLayout.yaxis, scaleanchor: 'x', scaleratio: 1 },
            annotations: annotations
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

        // 3. Step Size Plot
        if (data.ds_opt && data.ds_opt.length > 0) {
            const ds_values = data.ds_opt.map(d => Math.abs(d));
            const nonZeroDs = ds_values.filter(d => d > 1e-4);
            const avgDs = nonZeroDs.length > 0 ? (nonZeroDs.reduce((a, b) => a + b, 0) / nonZeroDs.length) : 0;
            
            const traceDs = {
                y: ds_values,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Step Size ds',
                line: {color: '#8b5cf6', width: 2},
                marker: {size: 4}
            };

            const traceAvg = {
                y: Array(ds_values.length).fill(avgDs),
                type: 'scatter',
                mode: 'lines',
                name: `Avg (${avgDs.toFixed(4)})`,
                line: {color: '#ec4899', width: 2, dash: 'dash'}
            };

            const traceTarget = {
                y: Array(ds_values.length).fill(data.target_ds_mag),
                type: 'scatter',
                mode: 'lines',
                name: `Target (${data.target_ds_mag.toFixed(4)})`,
                line: {color: '#10b981', width: 2, dash: 'dot'}
            };

            Plotly.newPlot('ds-chart', [traceDs, traceAvg, traceTarget], {
                ...chartLayout,
                title: 'Step Size (ds)'
            });
        } else {
            document.getElementById('ds-chart').innerHTML = '';
        }

        // 4. Curvature Derivative Plot
        if (data.dkappa_opt && data.dkappa_opt.length > 0) {
            const traceDkappa = {
                y: data.dkappa_opt,
                type: 'scatter',
                mode: 'lines',
                name: 'Curvature Deriv dκ',
                line: {color: '#f59e0b', width: 2}
            };
            Plotly.newPlot('dk-chart', [traceDkappa], {
                ...chartLayout,
                title: 'Curvature Derivative (dκ/ds)'
            });
        } else {
            document.getElementById('dk-chart').innerHTML = '';
        }
    };

    const renderDubinsCommands = (commands, currentMaxKappa) => {
        const container = document.getElementById('dubins-commands-list');
        const radiusDisplay = document.getElementById('display-turning-radius');
        
        // Update turning radius display
        if (currentMaxKappa > 0.01) {
            radiusDisplay.textContent = (1.0 / currentMaxKappa).toFixed(2);
        } else {
            radiusDisplay.textContent = '--';
        }

        if (!commands || commands.length === 0) {
            container.innerHTML = '<p class="empty-msg">No path generated yet.</p>';
            return;
        }

        container.innerHTML = '';
        commands.forEach((cmd, i) => {
            if (cmd.length < 0.01) return; // Skip tiny segments

            const pill = document.createElement('div');
            pill.className = `command-pill type-${cmd.type}`;
            
            let label = cmd.type; // L, R, or S
            
            pill.innerHTML = `
                <span class="type-icon">${label}</span>
                <span class="dist">${cmd.length.toFixed(2)}m</span>
            `;
            container.appendChild(pill);
        });
    };

    const runOptimization = async () => {
        runBtn.textContent = 'Optimizing...';
        runBtn.disabled = true;
        statusMsg.textContent = 'Running nonlinear solver...';
        statusMsg.className = 'status-msg';

        const payload = { params: {} };
        params.forEach(param => {
            const el = document.getElementById(param);
            if (el) payload.params[param] = parseFloat(el.value);
        });
        
        // Add manual curvature flag
        payload.params['fix_start_kappa'] = document.getElementById('fix_start_kappa').checked;
        payload.params['use_dubins'] = document.getElementById('use_dubins').checked;

        const currentMaxKappa = payload.params.max_kappa;

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
                document.getElementById('display-solve-time').textContent = data.solve_time_ms.toFixed(1);
                
                // Calculate average ds (exclude zero-length virtual segments)
                if (data.ds_opt && data.ds_opt.length > 0) {
                    const nonZeroDs = data.ds_opt.map(d => Math.abs(d)).filter(d => d > 1e-4);
                    const avgDs = nonZeroDs.length > 0 ? (nonZeroDs.reduce((a, b) => a + b, 0) / nonZeroDs.length) : 0;
                    document.getElementById('display-avg-ds').textContent = avgDs.toFixed(3);
                } else {
                    document.getElementById('display-avg-ds').textContent = '--';
                }
                
                document.getElementById('display-target-ds').textContent = data.target_ds_mag.toFixed(3);
                plotCharts(data, payload.params);
                renderDubinsCommands(data.dubins_commands, currentMaxKappa);
            } else {
                statusMsg.textContent = data.message || 'Optimization failed.';
                statusMsg.className = 'status-msg error';
                if (data.solve_time_ms !== undefined) {
                    document.getElementById('display-solve-time').textContent = data.solve_time_ms.toFixed(1);
                }
                if (data.target_ds_mag !== undefined) {
                    document.getElementById('display-target-ds').textContent = data.target_ds_mag.toFixed(3);
                }
                if (data.x_ref) {
                    plotCharts(data, payload.params);
                    renderDubinsCommands(data.dubins_commands, currentMaxKappa);
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
        if (input && span) {
            input.addEventListener('input', () => {
                const val = parseFloat(input.value);
                if (input.step >= 1.0 || (input.max > 10 && input.step === "1")) {
                    span.textContent = val.toFixed(0);
                } else {
                    span.textContent = val.toFixed(2);
                }
            });
            input.addEventListener('change', runOptimization);
        }
    });

    // Special Handler for Log10 Tolerance
    const tolLogInput = document.getElementById('ipopt_tol_log');
    const tolInput = document.getElementById('ipopt_tol');
    const tolSpan = document.getElementById('val_ipopt_tol');
    if (tolLogInput && tolInput && tolSpan) {
        const updateTol = () => {
            const exponent = parseInt(tolLogInput.value);
            const tolValue = Math.pow(10, exponent);
            tolInput.value = tolValue;
            // Use scientific notation for display
            tolSpan.textContent = tolValue.toExponential(1);
        };
        tolLogInput.addEventListener('input', updateTol);
        tolLogInput.addEventListener('change', runOptimization);
        // Set initial
        updateTol();
    }

    const fixKappaToggle = document.getElementById('fix_start_kappa');
    const kappaContainer = document.getElementById('start_kappa_container');
    
    fixKappaToggle.addEventListener('change', () => {
        kappaContainer.style.opacity = fixKappaToggle.checked ? '1' : '0.4';
        kappaContainer.style.pointerEvents = fixKappaToggle.checked ? 'auto' : 'none';
        runOptimization();
    });
    
    const useDubinsToggle = document.getElementById('use_dubins');
    useDubinsToggle.addEventListener('change', runOptimization);
    
    // Initial run
    runOptimization();
});
