document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const statusMsg = document.getElementById('status-msg');
    const referenceEditorChart = document.getElementById('reference-editor-chart');
    const referenceEditorState = document.getElementById('reference-editor-state');
    const resetReferenceBtn = document.getElementById('reset-reference-btn');
    const backendSelect = document.getElementById('backend');
    const clearObstaclesBtn = document.getElementById('clear-obstacles-btn');

    // Parameters shared between backends
    const params = [
        'start_x', 'start_y', 'start_theta',
        'goal_x', 'goal_y', 'goal_theta',
        'max_kappa', 'target_ds', 'ds_min_ratio', 'ds_max_ratio',
        'w_ref', 'w_dkappa', 'w_kappa', 'w_ds',
        'start_kappa', 'ipopt_max_iter', 'ipopt_print_level',
        'w_kinematic', 'ceres_max_iter',
        'w_esdf', 'esdf_safe_distance'
    ];

    const generatorParams = new Set([
        'start_x', 'start_y', 'start_theta',
        'goal_x', 'goal_y', 'goal_theta',
        'max_kappa', 'target_ds'
    ]);

    const sharedPlotConfig = {
        responsive: true, displayModeBar: false, scrollZoom: false,
        doubleClick: false, displaylogo: false
    };

    const editorPlotConfig = { ...sharedPlotConfig, staticPlot: false };

    const state = {
        activeParams: {},
        obstacles: [],          // [{x_min, y_min, x_max, y_max}, ...]
        reference: {
            base: null, current: null, isCustom: false,
            dragIndex: null, selectedIndex: null, movedDuringDrag: false
        }
    };

    // ===== Helpers =====

    const cloneRef = (r) => r ? { x: [...r.x], y: [...r.y], theta: [...r.theta], gears: [...r.gears] } : null;

    const unwrapAngles = (angles) => {
        if (!angles || !angles.length) return [];
        const u = [angles[0]];
        for (let i = 1; i < angles.length; i++) {
            let c = angles[i], d = c - u[i - 1];
            while (d > Math.PI) { c -= 2 * Math.PI; d = c - u[i - 1]; }
            while (d < -Math.PI) { c += 2 * Math.PI; d = c - u[i - 1]; }
            u.push(c);
        }
        return u;
    };

    const averageAngles = (a) => Math.atan2(
        a.reduce((s, v) => s + Math.sin(v), 0),
        a.reduce((s, v) => s + Math.cos(v), 0));

    const computeRefHeadings = (x, y, gears) => {
        if (!x || !x.length) return [];
        if (x.length === 1) return [state.activeParams.start_theta ?? 0];
        const st = state.activeParams.start_theta ?? 0;
        const gt = state.activeParams.goal_theta ?? 0;
        const seg = [];
        let prev = st;
        for (let i = 0; i < x.length - 1; i++) {
            const dx = x[i + 1] - x[i], dy = y[i + 1] - y[i];
            if (Math.hypot(dx, dy) > 1e-9) {
                let h = Math.atan2(dy, dx);
                if ((gears[i] ?? 1) < 0) h += Math.PI;
                prev = h;
            }
            seg.push(prev);
        }
        const th = new Array(x.length).fill(0);
        th[0] = st; th[x.length - 1] = gt;
        for (let i = 1; i < x.length - 1; i++)
            th[i] = averageAngles([seg[i - 1], seg[i]]);
        return unwrapAngles(th);
    };

    const setSliderValue = (id, value) => {
        const input = document.getElementById(id);
        const span = document.getElementById(`val_${id}`);
        if (!input || !span) return;
        input.value = value;
        const v = parseFloat(value);
        span.textContent = (input.step >= 1.0 || (input.max > 10 && input.step === '1'))
            ? v.toFixed(0) : v.toFixed(2);
    };

    const syncEndpoints = () => {
        const r = state.reference.current;
        if (!r || r.x.length < 2) return;
        setSliderValue('start_x', r.x[0]);
        setSliderValue('start_y', r.y[0]);
        setSliderValue('goal_x', r.x[r.x.length - 1]);
        setSliderValue('goal_y', r.y[r.y.length - 1]);
    };

    const updateBadge = () => {
        referenceEditorState.textContent = state.reference.isCustom ? 'Custom Reference' : 'Generated Reference';
        resetReferenceBtn.disabled = !state.reference.base || !state.reference.isCustom;
    };

    const getRefFromResponse = (d) =>
        (d.x_ref && d.y_ref && d.theta_ref && d.gears)
            ? { x: [...d.x_ref], y: [...d.y_ref], theta: [...d.theta_ref], gears: [...d.gears] }
            : null;

    // ===== Backend visibility =====

    const updateBackendUI = () => {
        const be = backendSelect.value;
        document.getElementById('display-backend').textContent =
            be === 'ceres' ? 'C++/Ceres' : 'Python/CasADi';

        // Show/hide backend-specific sections
        const ceresSec = document.getElementById('ceres-params-section');
        const ipoptSec = document.getElementById('ipopt-params-section');
        if (ceresSec) ceresSec.style.display = be === 'ceres' ? '' : 'none';
        if (ipoptSec) ipoptSec.style.display = be === 'python' ? '' : 'none';
    };

    backendSelect.addEventListener('change', () => {
        updateBackendUI();
        runOptimization({ keepCustomReference: state.reference.isCustom });
    });

    // ===== Reference Editor =====

    const getRefBounds = (r) => {
        const minX = Math.min(...r.x), maxX = Math.max(...r.x);
        const minY = Math.min(...r.y), maxY = Math.max(...r.y);
        const pad = Math.max(maxX - minX, maxY - minY, 1) * 0.15;
        return { x: [minX - pad, maxX + pad], y: [minY - pad, maxY + pad] };
    };

    const buildHeadingTrace = (r) => {
        const scale = Math.max(0.25, Math.min(1.2,
            Math.max(Math.max(...r.x) - Math.min(...r.x),
                     Math.max(...r.y) - Math.min(...r.y)) * 0.08));
        const hx = [], hy = [];
        for (let i = 0; i < r.x.length; i++) {
            hx.push(r.x[i], r.x[i] + Math.cos(r.theta[i]) * scale, null);
            hy.push(r.y[i], r.y[i] + Math.sin(r.theta[i]) * scale, null);
        }
        return { x: hx, y: hy, type: 'scatter', mode: 'lines', name: 'Heading',
                 line: { color: '#fbbf24', width: 2 }, hoverinfo: 'skip', showlegend: false };
    };

    const renderRefEditor = () => {
        const r = state.reference.current;
        updateBadge();
        if (!r || !r.x.length) { referenceEditorChart.innerHTML = ''; return; }
        const bounds = getRefBounds(r);
        const sizes = r.x.map((_, i) =>
            i === state.reference.selectedIndex ? 15 : (i === 0 || i === r.x.length - 1 ? 13 : 11));
        const colors = r.x.map((_, i) =>
            i === state.reference.selectedIndex ? '#fbbf24' :
            i === 0 ? '#60a5fa' : i === r.x.length - 1 ? '#fb7185' : '#e2e8f0');

        const pointTrace = {
            x: r.x, y: r.y, type: 'scatter', mode: 'lines+markers+text',
            text: r.x.map((_, i) => `${i}`), textposition: 'top center',
            textfont: { color: '#cbd5e1', size: 10 },
            line: { color: '#94a3b8', width: 2, dash: 'dot' },
            marker: { size: sizes, color: colors, line: { color: '#0f172a', width: 1.5 } },
            hovertemplate: 'idx=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
            showlegend: false
        };

        Plotly.react(referenceEditorChart,
            [buildHeadingTrace(r), pointTrace],
            { plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#f1f5f9', family: 'Outfit' },
              margin: { t: 20, b: 40, l: 50, r: 20 }, dragmode: false, hovermode: 'closest',
              xaxis: { title: 'x (m)', range: bounds.x, fixedrange: true,
                       gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
              yaxis: { title: 'y (m)', range: bounds.y, fixedrange: true,
                       scaleanchor: 'x', scaleratio: 1,
                       gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' }
            }, editorPlotConfig);
    };

    // ===== Payload =====

    const collectParams = () => {
        const p = {};
        params.forEach((k) => {
            const el = document.getElementById(k);
            if (el) p[k] = parseFloat(el.value);
        });
        p.fix_start_kappa = document.getElementById('fix_start_kappa').checked;
        p.use_dubins = document.getElementById('use_dubins').checked;
        return p;
    };

    const buildCustomRefPayload = () => {
        if (!state.reference.isCustom || !state.reference.current) return null;
        const c = state.reference.current;
        return { x: [...c.x], y: [...c.y], theta: [...c.theta], gears: [...c.gears] };
    };

    // ===== Plotting =====

    const segmentTraces = (x, y, gears, prefix, isRef) => {
        if (!x || x.length < 2 || !gears || !gears.length) return [];
        const traces = [];
        let dir = gears[0], s = 0;
        for (let i = 1; i < x.length; i++) {
            if (i === x.length - 1 || gears[i] !== dir) {
                const sx = x.slice(s, i + 1), sy = y.slice(s, i + 1);
                const fwd = dir >= 0;
                const col = isRef ? (fwd ? '#94a3b8' : '#fda4af') : (fwd ? '#6366f1' : '#f43f5e');
                const lbl = `${prefix} (${fwd ? 'Forward' : 'Backward'})`;
                traces.push({
                    x: sx, y: sy, mode: isRef ? 'lines' : 'lines+markers', name: lbl,
                    line: { color: col, dash: isRef ? 'dot' : 'solid', width: isRef ? 2 : 4, shape: 'spline', smoothing: 1.3 },
                    marker: isRef ? {} : { size: 6, color: col, line: { color: '#fff', width: 1 } },
                    legendgroup: lbl, showlegend: traces.findIndex(t => t.name === lbl) === -1
                });
                if (i < x.length - 1) { s = i; dir = gears[i]; }
            }
        }
        return traces;
    };

    const obstacleShapes = () => state.obstacles.map(o => ({
        type: 'rect', x0: o.x_min, y0: o.y_min, x1: o.x_max, y1: o.y_max,
        fillcolor: 'rgba(239,68,68,0.25)', line: { color: 'rgba(239,68,68,0.6)', width: 2 }
    }));

    const plotCharts = (data, cp) => {
        const layout = {
            plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f1f5f9', family: 'Outfit' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
            margin: { t: 60, b: 40, l: 50, r: 20 },
            legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center', font: { size: 10 } }
        };

        const refT = segmentTraces(data.x_ref, data.y_ref, data.gears, 'Ref', true);
        const optT = segmentTraces(data.x_opt, data.y_opt, data.gears_opt, 'Smooth', false);

        const arrows = [
            { x: cp.start_x + Math.cos(cp.start_theta) * 1.5, y: cp.start_y + Math.sin(cp.start_theta) * 1.5,
              ax: cp.start_x, ay: cp.start_y, xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
              showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 4, arrowcolor: 'rgba(99,102,241,0.5)' },
            { x: cp.goal_x + Math.cos(cp.goal_theta) * 1.5, y: cp.goal_y + Math.sin(cp.goal_theta) * 1.5,
              ax: cp.goal_x, ay: cp.goal_y, xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
              showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 4, arrowcolor: 'rgba(244,63,94,0.5)' }
        ];

        Plotly.newPlot('path-chart', [...refT, ...optT], {
            ...layout, title: 'Path Geometry (Indigo: Fwd, Rose: Bwd)',
            yaxis: { ...layout.yaxis, scaleanchor: 'x', scaleratio: 1 },
            annotations: arrows, shapes: obstacleShapes()
        }, sharedPlotConfig);

        // Attach click handler for placing obstacles
        const pathChart = document.getElementById('path-chart');
        if (pathChart._obstacleHandler) {
            pathChart.removeListener('plotly_click', pathChart._obstacleHandler);
        }
        pathChart._obstacleHandler = (ev) => {
            if (!ev.points || !ev.points.length) return;
            const pt = ev.points[0];
            const hw = 0.5;
            state.obstacles.push({ x_min: pt.x - hw, y_min: pt.y - hw, x_max: pt.x + hw, y_max: pt.y + hw });
            runOptimization({ keepCustomReference: state.reference.isCustom });
        };
        pathChart.on('plotly_click', pathChart._obstacleHandler);

        if (data.kappa_opt && data.kappa_opt.length) {
            Plotly.newPlot('kappa-chart', [{
                y: data.kappa_opt, type: 'scatter', mode: 'lines', name: 'Curvature κ',
                line: { color: '#10b981', width: 2 }
            }], { ...layout, title: 'Curvature Profile' }, sharedPlotConfig);
        } else { document.getElementById('kappa-chart').innerHTML = ''; }

        if (data.ds_opt && data.ds_opt.length) {
            const ds = data.ds_opt.map(v => Math.abs(v));
            const nz = ds.filter(v => v > 1e-4);
            const avg = nz.length ? nz.reduce((a, b) => a + b, 0) / nz.length : 0;
            Plotly.newPlot('ds-chart', [
                { y: ds, type: 'scatter', mode: 'lines+markers', name: 'Step Size ds',
                  line: { color: '#8b5cf6', width: 2 }, marker: { size: 4 } },
                { y: Array(ds.length).fill(avg), type: 'scatter', mode: 'lines',
                  name: `Avg (${avg.toFixed(4)})`, line: { color: '#ec4899', width: 2, dash: 'dash' } },
                { y: Array(ds.length).fill(data.target_ds_mag), type: 'scatter', mode: 'lines',
                  name: `Target (${data.target_ds_mag.toFixed(4)})`, line: { color: '#10b981', width: 2, dash: 'dot' } }
            ], { ...layout, title: 'Step Size (ds)' }, sharedPlotConfig);
        } else { document.getElementById('ds-chart').innerHTML = ''; }

        if (data.dkappa_opt && data.dkappa_opt.length) {
            Plotly.newPlot('dk-chart', [{
                y: data.dkappa_opt, type: 'scatter', mode: 'lines', name: 'Curvature Deriv dκ',
                line: { color: '#f59e0b', width: 2 }
            }], { ...layout, title: 'Curvature Derivative (dκ/ds)' }, sharedPlotConfig);
        } else { document.getElementById('dk-chart').innerHTML = ''; }
    };

    const renderDubins = (cmds, maxK) => {
        const rd = document.getElementById('display-turning-radius');
        rd.textContent = maxK > 0.01 ? (1 / maxK).toFixed(2) : '--';
        const c = document.getElementById('dubins-commands-list');
        if (!cmds || !cmds.length) { c.innerHTML = '<p class="empty-msg">No path generated yet.</p>'; return; }
        c.innerHTML = '';
        cmds.forEach(cmd => {
            if (cmd.length < 0.01) return;
            const p = document.createElement('div');
            p.className = `command-pill type-${cmd.type}`;
            p.innerHTML = `<span class="type-icon">${cmd.type}</span><span class="dist">${cmd.length.toFixed(2)}m</span>`;
            c.appendChild(p);
        });
    };

    const updateMetrics = (d) => {
        if (d.solve_time_ms !== undefined) document.getElementById('display-solve-time').textContent = d.solve_time_ms.toFixed(1);
        if (d.target_ds_mag !== undefined) document.getElementById('display-target-ds').textContent = d.target_ds_mag.toFixed(3);
        if (d.x_ref) document.getElementById('display-num-points').textContent = d.x_ref.length;
        if (d.dubins_commands) {
            const rl = d.dubins_commands.reduce((s, c) => s + c.length, 0);
            document.getElementById('display-ref-length').textContent = rl.toFixed(2);
        }
        if (d.ds_opt && d.ds_opt.length) {
            const dm = d.ds_opt.map(v => Math.abs(v));
            const nz = dm.filter(v => v > 1e-4);
            const avg = nz.length ? nz.reduce((a, b) => a + b) / nz.length : 0;
            document.getElementById('display-avg-ds').textContent = avg.toFixed(3);
            document.getElementById('display-path-length').textContent = dm.reduce((a, b) => a + b).toFixed(2);
        } else {
            document.getElementById('display-avg-ds').textContent = '--';
            document.getElementById('display-path-length').textContent = '--';
        }
        if (d.theta_opt && d.theta_opt.length > 1) {
            let tr = 0;
            for (let i = 1; i < d.theta_opt.length; i++) tr += Math.abs(d.theta_opt[i] - d.theta_opt[i - 1]);
            document.getElementById('display-total-rotation').textContent = (tr * 180 / Math.PI).toFixed(1);
        } else { document.getElementById('display-total-rotation').textContent = '--'; }
        const costFields = ['total', 'ref', 'smooth', 'kappa', 'ds'];
        if (d.costs) {
            costFields.forEach(f => document.getElementById(`display-cost-${f}`).textContent = (d.costs[f] ?? 0).toFixed(2));
        } else {
            costFields.forEach(f => document.getElementById(`display-cost-${f}`).textContent = '--');
        }
    };

    // ===== Optimization =====

    const runOptimization = async ({ keepCustomReference = state.reference.isCustom } = {}) => {
        const payload = { params: collectParams(), backend: backendSelect.value };
        state.activeParams = { ...payload.params };

        if (keepCustomReference) {
            const cr = buildCustomRefPayload();
            if (cr) payload.custom_reference = cr;
        }

        if (state.obstacles.length && backendSelect.value === 'ceres') {
            payload.obstacles = state.obstacles;
        }

        runBtn.textContent = 'Optimizing...';
        runBtn.disabled = true;
        statusMsg.textContent = `Running ${backendSelect.value === 'ceres' ? 'Ceres' : 'CasADi'} solver...`;
        statusMsg.className = 'status-msg';

        try {
            const resp = await fetch('/api/smooth', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            const maxK = payload.params.max_kappa;
            const rr = getRefFromResponse(data);

            if (rr) {
                if (payload.custom_reference) {
                    state.reference.current = cloneRef(rr);
                    state.reference.isCustom = true;
                    if (!state.reference.base) state.reference.base = cloneRef(rr);
                } else {
                    state.reference.base = cloneRef(rr);
                    state.reference.current = cloneRef(rr);
                    state.reference.isCustom = false;
                }
                renderRefEditor();
            }

            updateMetrics(data);
            renderDubins(data.dubins_commands, maxK);

            if (data.success) {
                statusMsg.textContent = payload.custom_reference
                    ? 'Solved with custom reference path.' : 'Optimization solved successfully!';
                statusMsg.className = 'status-msg success';
            } else {
                statusMsg.textContent = data.message || 'Optimization failed.';
                statusMsg.className = 'status-msg error';
            }

            if (data.x_ref) plotCharts(data, payload.params);
        } catch (err) {
            console.error(err);
            statusMsg.textContent = 'Network or server error.';
            statusMsg.className = 'status-msg error';
        }

        runBtn.textContent = 'Run Optimization';
        runBtn.disabled = false;
    };

    const clearCustomRef = ({ syncInputs = true } = {}) => {
        if (!state.reference.base) return;
        state.reference.current = cloneRef(state.reference.base);
        state.reference.isCustom = false;
        state.reference.selectedIndex = null;
        if (syncInputs) syncEndpoints();
        renderRefEditor();
    };

    // ===== Drag handlers =====

    const plotCoords = (ev) => {
        const fl = referenceEditorChart._fullLayout;
        if (!fl || !fl.xaxis || !fl.yaxis) return null;
        const b = referenceEditorChart.getBoundingClientRect();
        return { fullLayout: fl, plotX: ev.clientX - b.left - fl.xaxis._offset,
                 plotY: ev.clientY - b.top - fl.yaxis._offset };
    };

    const nearestPoint = (ev) => {
        const r = state.reference.current, pc = plotCoords(ev);
        if (!r || !pc) return null;
        const { fullLayout: fl, plotX, plotY } = pc;
        if (plotX < 0 || plotY < 0 || plotX > fl.xaxis._length || plotY > fl.yaxis._length) return null;
        let best = null, bd = Infinity;
        for (let i = 0; i < r.x.length; i++) {
            const d = Math.hypot(fl.xaxis.l2p(r.x[i]) - plotX, fl.yaxis.l2p(r.y[i]) - plotY);
            if (d < bd) { bd = d; best = i; }
        }
        return bd > 18 ? null : best;
    };

    const updateDrag = (ev) => {
        const r = state.reference.current, di = state.reference.dragIndex, pc = plotCoords(ev);
        if (!r || di === null || !pc) return;
        const { fullLayout: fl, plotX, plotY } = pc;
        r.x[di] = fl.xaxis.p2l(Math.max(0, Math.min(fl.xaxis._length, plotX)));
        r.y[di] = fl.yaxis.p2l(Math.max(0, Math.min(fl.yaxis._length, plotY)));
        r.theta = computeRefHeadings(r.x, r.y, r.gears);
        state.reference.isCustom = true;
        state.reference.selectedIndex = di;
        if (di === 0 || di === r.x.length - 1) syncEndpoints();
        renderRefEditor();
    };

    referenceEditorChart.addEventListener('pointerdown', (ev) => {
        const idx = nearestPoint(ev);
        if (idx === null) return;
        ev.preventDefault();
        state.reference.dragIndex = idx;
        state.reference.selectedIndex = idx;
        state.reference.movedDuringDrag = false;
        referenceEditorChart.classList.add('is-dragging');
        renderRefEditor();
    });

    window.addEventListener('pointermove', (ev) => {
        if (state.reference.dragIndex === null) return;
        state.reference.movedDuringDrag = true;
        updateDrag(ev);
    });

    window.addEventListener('pointerup', () => {
        if (state.reference.dragIndex === null) return;
        const moved = state.reference.movedDuringDrag;
        state.reference.dragIndex = null;
        state.reference.movedDuringDrag = false;
        referenceEditorChart.classList.remove('is-dragging');
        updateBadge();
        if (moved) runOptimization({ keepCustomReference: true });
    });

    // ===== Event listeners =====

    runBtn.addEventListener('click', () => runOptimization({ keepCustomReference: state.reference.isCustom }));
    resetReferenceBtn.addEventListener('click', () => { clearCustomRef(); runOptimization({ keepCustomReference: false }); });
    clearObstaclesBtn.addEventListener('click', () => { state.obstacles = []; runOptimization({ keepCustomReference: state.reference.isCustom }); });

    params.forEach((p) => {
        const input = document.getElementById(p);
        const span = document.getElementById(`val_${p}`);
        if (!input || !span) return;
        input.addEventListener('input', () => {
            const v = parseFloat(input.value);
            span.textContent = (input.step >= 1.0 || (input.max > 10 && input.step === '1')) ? v.toFixed(0) : v.toFixed(2);
        });
        input.addEventListener('change', () => {
            if (generatorParams.has(p)) { clearCustomRef({ syncInputs: false }); runOptimization({ keepCustomReference: false }); return; }
            runOptimization({ keepCustomReference: state.reference.isCustom });
        });
    });

    // Tolerance (log10) for IPOPT
    const tolLog = document.getElementById('ipopt_tol_log');
    const tolIn = document.getElementById('ipopt_tol');
    const tolSp = document.getElementById('val_ipopt_tol');
    if (tolLog && tolIn && tolSp) {
        const upTol = () => { const e = parseInt(tolLog.value); tolIn.value = Math.pow(10, e); tolSp.textContent = Math.pow(10, e).toExponential(1); };
        tolLog.addEventListener('input', upTol);
        tolLog.addEventListener('change', () => runOptimization({ keepCustomReference: state.reference.isCustom }));
        upTol();
    }

    // Tolerance (log10) for Ceres
    const ctolLog = document.getElementById('ceres_tol_log');
    const ctolIn = document.getElementById('ceres_tol');
    const ctolSp = document.getElementById('val_ceres_tol');
    if (ctolLog && ctolIn && ctolSp) {
        const upCTol = () => { const e = parseInt(ctolLog.value); ctolIn.value = Math.pow(10, e); ctolSp.textContent = Math.pow(10, e).toExponential(1); };
        ctolLog.addEventListener('input', upCTol);
        ctolLog.addEventListener('change', () => runOptimization({ keepCustomReference: state.reference.isCustom }));
        upCTol();
    }

    const fixKappa = document.getElementById('fix_start_kappa');
    const kappaCont = document.getElementById('start_kappa_container');
    fixKappa.addEventListener('change', () => {
        kappaCont.style.opacity = fixKappa.checked ? '1' : '0.4';
        kappaCont.style.pointerEvents = fixKappa.checked ? 'auto' : 'none';
        runOptimization({ keepCustomReference: state.reference.isCustom });
    });

    document.getElementById('use_dubins').addEventListener('change', () => {
        clearCustomRef({ syncInputs: false });
        runOptimization({ keepCustomReference: false });
    });

    // ===== Init =====
    updateBackendUI();
    state.activeParams = collectParams();
    runOptimization({ keepCustomReference: false });
});
