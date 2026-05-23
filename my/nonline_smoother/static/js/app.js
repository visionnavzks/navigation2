document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const statusMsg = document.getElementById('status-msg');
    const referenceEditorChart = document.getElementById('reference-editor-chart');
    const referenceEditorState = document.getElementById('reference-editor-state');
    const resetReferenceBtn = document.getElementById('reset-reference-btn');

    const params = [
        'start_x', 'start_y', 'start_theta',
        'goal_x', 'goal_y', 'goal_theta',
        'max_kappa', 'target_ds', 'ds_min_ratio', 'ds_max_ratio', 'w_ref', 'w_dkappa', 'w_kappa', 'w_ds',
        'start_kappa', 'ipopt_max_iter', 'ipopt_print_level'
    ];

    const generatorParams = new Set([
        'start_x', 'start_y', 'start_theta',
        'goal_x', 'goal_y', 'goal_theta',
        'max_kappa', 'target_ds'
    ]);

    const sharedPlotConfig = {
        responsive: true,
        displayModeBar: false,
        scrollZoom: false,
        doubleClick: false,
        displaylogo: false
    };

    const editorPlotConfig = {
        ...sharedPlotConfig,
        staticPlot: false
    };

    const state = {
        activeParams: {},
        reference: {
            base: null,
            current: null,
            isCustom: false,
            dragIndex: null,
            selectedIndex: null,
            movedDuringDrag: false
        }
    };

    const cloneReference = (reference) => {
        if (!reference) {
            return null;
        }

        return {
            x: [...reference.x],
            y: [...reference.y],
            theta: [...reference.theta],
            gears: [...reference.gears]
        };
    };

    const unwrapAngles = (angles) => {
        if (!angles || angles.length === 0) {
            return [];
        }

        const unwrapped = [angles[0]];
        for (let i = 1; i < angles.length; i += 1) {
            let candidate = angles[i];
            let delta = candidate - unwrapped[i - 1];
            while (delta > Math.PI) {
                candidate -= 2 * Math.PI;
                delta = candidate - unwrapped[i - 1];
            }
            while (delta < -Math.PI) {
                candidate += 2 * Math.PI;
                delta = candidate - unwrapped[i - 1];
            }
            unwrapped.push(candidate);
        }
        return unwrapped;
    };

    const averageAngles = (angles) => {
        const sinSum = angles.reduce((sum, angle) => sum + Math.sin(angle), 0);
        const cosSum = angles.reduce((sum, angle) => sum + Math.cos(angle), 0);
        return Math.atan2(sinSum, cosSum);
    };

    const computeReferenceHeadings = (x, y, gears) => {
        if (!x || x.length === 0) {
            return [];
        }

        if (x.length === 1) {
            return [state.activeParams.start_theta ?? 0];
        }

        const startTheta = state.activeParams.start_theta ?? 0;
        const goalTheta = state.activeParams.goal_theta ?? 0;
        const segmentHeadings = [];
        let previousHeading = startTheta;

        for (let i = 0; i < x.length - 1; i += 1) {
            const dx = x[i + 1] - x[i];
            const dy = y[i + 1] - y[i];
            if (Math.hypot(dx, dy) > 1e-9) {
                let heading = Math.atan2(dy, dx);
                if ((gears[i] ?? 1) < 0) {
                    heading += Math.PI;
                }
                previousHeading = heading;
            }
            segmentHeadings.push(previousHeading);
        }

        const theta = new Array(x.length).fill(0);
        theta[0] = startTheta;
        theta[x.length - 1] = goalTheta;

        for (let i = 1; i < x.length - 1; i += 1) {
            theta[i] = averageAngles([segmentHeadings[i - 1], segmentHeadings[i]]);
        }

        return unwrapAngles(theta);
    };

    const setSliderValue = (id, value) => {
        const input = document.getElementById(id);
        const span = document.getElementById(`val_${id}`);
        if (!input || !span) {
            return;
        }

        input.value = value;
        const numericValue = parseFloat(value);
        if (input.step >= 1.0 || (input.max > 10 && input.step === '1')) {
            span.textContent = numericValue.toFixed(0);
        } else {
            span.textContent = numericValue.toFixed(2);
        }
    };

    const syncEndpointInputs = () => {
        const reference = state.reference.current;
        if (!reference || reference.x.length < 2) {
            return;
        }

        setSliderValue('start_x', reference.x[0]);
        setSliderValue('start_y', reference.y[0]);
        setSliderValue('goal_x', reference.x[reference.x.length - 1]);
        setSliderValue('goal_y', reference.y[reference.y.length - 1]);
    };

    const updateReferenceBadge = () => {
        referenceEditorState.textContent = state.reference.isCustom ? 'Custom Reference' : 'Generated Reference';
        resetReferenceBtn.disabled = !state.reference.base || !state.reference.isCustom;
    };

    const getReferenceFromResponse = (data) => {
        if (!data.x_ref || !data.y_ref || !data.theta_ref || !data.gears) {
            return null;
        }

        return {
            x: [...data.x_ref],
            y: [...data.y_ref],
            theta: [...data.theta_ref],
            gears: [...data.gears]
        };
    };

    const getReferenceBounds = (reference) => {
        const allX = reference.x;
        const allY = reference.y;
        const minX = Math.min(...allX);
        const maxX = Math.max(...allX);
        const minY = Math.min(...allY);
        const maxY = Math.max(...allY);
        const spanX = Math.max(maxX - minX, 1.0);
        const spanY = Math.max(maxY - minY, 1.0);
        const padding = Math.max(spanX, spanY) * 0.15;

        return {
            x: [minX - padding, maxX + padding],
            y: [minY - padding, maxY + padding]
        };
    };

    const buildHeadingTrace = (reference) => {
        const xSpan = Math.max(...reference.x) - Math.min(...reference.x);
        const ySpan = Math.max(...reference.y) - Math.min(...reference.y);
        const headingScale = Math.max(0.25, Math.min(1.2, Math.max(xSpan, ySpan) * 0.08));
        const headingX = [];
        const headingY = [];

        for (let i = 0; i < reference.x.length; i += 1) {
            headingX.push(reference.x[i], reference.x[i] + Math.cos(reference.theta[i]) * headingScale, null);
            headingY.push(reference.y[i], reference.y[i] + Math.sin(reference.theta[i]) * headingScale, null);
        }

        return {
            x: headingX,
            y: headingY,
            type: 'scatter',
            mode: 'lines',
            name: 'Heading',
            line: {
                color: '#fbbf24',
                width: 2
            },
            hoverinfo: 'skip',
            showlegend: false
        };
    };

    const renderReferenceEditor = () => {
        const reference = state.reference.current;
        updateReferenceBadge();

        if (!reference || reference.x.length === 0) {
            referenceEditorChart.innerHTML = '';
            return;
        }

        const bounds = getReferenceBounds(reference);
        const markerSizes = reference.x.map((_, index) => {
            if (index === state.reference.selectedIndex) {
                return 15;
            }
            if (index === 0 || index === reference.x.length - 1) {
                return 13;
            }
            return 11;
        });

        const markerColors = reference.x.map((_, index) => {
            if (index === state.reference.selectedIndex) {
                return '#fbbf24';
            }
            if (index === 0) {
                return '#60a5fa';
            }
            if (index === reference.x.length - 1) {
                return '#fb7185';
            }
            return '#e2e8f0';
        });

        const pointTrace = {
            x: reference.x,
            y: reference.y,
            type: 'scatter',
            mode: 'lines+markers+text',
            text: reference.x.map((_, index) => `${index}`),
            textposition: 'top center',
            textfont: {
                color: '#cbd5e1',
                size: 10
            },
            line: {
                color: '#94a3b8',
                width: 2,
                dash: 'dot'
            },
            marker: {
                size: markerSizes,
                color: markerColors,
                line: {
                    color: '#0f172a',
                    width: 1.5
                }
            },
            hovertemplate: 'idx=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
            showlegend: false
        };

        Plotly.react(
            referenceEditorChart,
            [buildHeadingTrace(reference), pointTrace],
            {
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#f1f5f9', family: 'Outfit' },
                margin: { t: 20, b: 40, l: 50, r: 20 },
                dragmode: false,
                hovermode: 'closest',
                xaxis: {
                    title: 'x (m)',
                    range: bounds.x,
                    fixedrange: true,
                    gridcolor: 'rgba(255,255,255,0.05)',
                    zerolinecolor: 'rgba(255,255,255,0.1)'
                },
                yaxis: {
                    title: 'y (m)',
                    range: bounds.y,
                    fixedrange: true,
                    scaleanchor: 'x',
                    scaleratio: 1,
                    gridcolor: 'rgba(255,255,255,0.05)',
                    zerolinecolor: 'rgba(255,255,255,0.1)'
                }
            },
            editorPlotConfig
        );
    };

    const buildCustomReferencePayload = () => {
        if (!state.reference.isCustom || !state.reference.current) {
            return null;
        }

        return {
            x: [...state.reference.current.x],
            y: [...state.reference.current.y],
            theta: [...state.reference.current.theta],
            gears: [...state.reference.current.gears]
        };
    };

    const collectPayloadParams = () => {
        const payloadParams = {};
        params.forEach((param) => {
            const el = document.getElementById(param);
            if (el) {
                payloadParams[param] = parseFloat(el.value);
            }
        });
        payloadParams.fix_start_kappa = document.getElementById('fix_start_kappa').checked;
        payloadParams.use_dubins = document.getElementById('use_dubins').checked;
        return payloadParams;
    };

    const getSegmentTraces = (x, y, gears, namePrefix, isRef) => {
        if (!x || x.length < 2 || !gears || gears.length === 0) {
            return [];
        }

        const traces = [];
        let currentDir = gears[0];
        let startIdx = 0;

        for (let i = 1; i < x.length; i += 1) {
            if (i === x.length - 1 || gears[i] !== currentDir) {
                const segmentX = x.slice(startIdx, i + 1);
                const segmentY = y.slice(startIdx, i + 1);
                const isForward = currentDir >= 0;
                const colors = {
                    smoothFwd: '#6366f1',
                    smoothBwd: '#f43f5e',
                    refFwd: '#94a3b8',
                    refBwd: '#fda4af'
                };

                const color = isRef
                    ? (isForward ? colors.refFwd : colors.refBwd)
                    : (isForward ? colors.smoothFwd : colors.smoothBwd);
                const label = `${namePrefix} (${isForward ? 'Forward' : 'Backward'})`;

                traces.push({
                    x: segmentX,
                    y: segmentY,
                    mode: isRef ? 'lines' : 'lines+markers',
                    name: label,
                    line: {
                        color,
                        dash: isRef ? 'dot' : 'solid',
                        width: isRef ? 2 : 4,
                        shape: 'spline',
                        smoothing: 1.3
                    },
                    marker: isRef ? {} : {
                        size: 6,
                        color,
                        line: { color: '#ffffff', width: 1 }
                    },
                    legendgroup: label,
                    showlegend: traces.findIndex((trace) => trace.name === label) === -1
                });

                if (i < x.length - 1) {
                    startIdx = i;
                    currentDir = gears[i];
                }
            }
        }

        return traces;
    };

    const plotCharts = (data, currentParams) => {
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
                x: currentParams.start_x + Math.cos(currentParams.start_theta) * 1.5,
                y: currentParams.start_y + Math.sin(currentParams.start_theta) * 1.5,
                ax: currentParams.start_x,
                ay: currentParams.start_y,
                xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 4,
                arrowcolor: 'rgba(99, 102, 241, 0.5)'
            },
            {
                x: currentParams.goal_x + Math.cos(currentParams.goal_theta) * 1.5,
                y: currentParams.goal_y + Math.sin(currentParams.goal_theta) * 1.5,
                ax: currentParams.goal_x,
                ay: currentParams.goal_y,
                xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 4,
                arrowcolor: 'rgba(244, 63, 94, 0.5)'
            }
        ];

        Plotly.newPlot('path-chart', [...refTraces, ...optTraces], {
            ...chartLayout,
            title: 'Path Geometry (Indigo: Fwd, Rose: Bwd)',
            yaxis: { ...chartLayout.yaxis, scaleanchor: 'x', scaleratio: 1 },
            annotations
        }, sharedPlotConfig);

        if (data.kappa_opt && data.kappa_opt.length > 0) {
            Plotly.newPlot('kappa-chart', [{
                y: data.kappa_opt,
                type: 'scatter',
                mode: 'lines',
                name: 'Curvature κ',
                line: { color: '#10b981', width: 2 }
            }], {
                ...chartLayout,
                title: 'Curvature Profile'
            }, sharedPlotConfig);
        } else {
            document.getElementById('kappa-chart').innerHTML = '';
        }

        if (data.ds_opt && data.ds_opt.length > 0) {
            const dsValues = data.ds_opt.map((value) => Math.abs(value));
            const nonZeroDs = dsValues.filter((value) => value > 1e-4);
            const avgDs = nonZeroDs.length > 0
                ? (nonZeroDs.reduce((sum, value) => sum + value, 0) / nonZeroDs.length)
                : 0;

            Plotly.newPlot('ds-chart', [{
                y: dsValues,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Step Size ds',
                line: { color: '#8b5cf6', width: 2 },
                marker: { size: 4 }
            }, {
                y: Array(dsValues.length).fill(avgDs),
                type: 'scatter',
                mode: 'lines',
                name: `Avg (${avgDs.toFixed(4)})`,
                line: { color: '#ec4899', width: 2, dash: 'dash' }
            }, {
                y: Array(dsValues.length).fill(data.target_ds_mag),
                type: 'scatter',
                mode: 'lines',
                name: `Target (${data.target_ds_mag.toFixed(4)})`,
                line: { color: '#10b981', width: 2, dash: 'dot' }
            }], {
                ...chartLayout,
                title: 'Step Size (ds)'
            }, sharedPlotConfig);
        } else {
            document.getElementById('ds-chart').innerHTML = '';
        }

        if (data.dkappa_opt && data.dkappa_opt.length > 0) {
            Plotly.newPlot('dk-chart', [{
                y: data.dkappa_opt,
                type: 'scatter',
                mode: 'lines',
                name: 'Curvature Deriv dκ',
                line: { color: '#f59e0b', width: 2 }
            }], {
                ...chartLayout,
                title: 'Curvature Derivative (dκ/ds)'
            }, sharedPlotConfig);
        } else {
            document.getElementById('dk-chart').innerHTML = '';
        }
    };

    const renderDubinsCommands = (commands, currentMaxKappa) => {
        const container = document.getElementById('dubins-commands-list');
        const radiusDisplay = document.getElementById('display-turning-radius');

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
        commands.forEach((cmd) => {
            if (cmd.length < 0.01) {
                return;
            }

            const pill = document.createElement('div');
            pill.className = `command-pill type-${cmd.type}`;
            pill.innerHTML = `
                <span class="type-icon">${cmd.type}</span>
                <span class="dist">${cmd.length.toFixed(2)}m</span>
            `;
            container.appendChild(pill);
        });
    };

    const updateMetrics = (data) => {
        if (data.solve_time_ms !== undefined) {
            document.getElementById('display-solve-time').textContent = data.solve_time_ms.toFixed(1);
        }

        if (data.target_ds_mag !== undefined) {
            document.getElementById('display-target-ds').textContent = data.target_ds_mag.toFixed(3);
        }

        if (data.x_ref) {
            document.getElementById('display-num-points').textContent = data.x_ref.length;
        }

        if (data.dubins_commands) {
            const refLength = data.dubins_commands.reduce((sum, cmd) => sum + cmd.length, 0);
            document.getElementById('display-ref-length').textContent = refLength.toFixed(2);
        }

        if (data.ds_opt && data.ds_opt.length > 0) {
            const dsMagnitudes = data.ds_opt.map((value) => Math.abs(value));
            const nonZeroDs = dsMagnitudes.filter((value) => value > 1e-4);
            const avgDs = nonZeroDs.length > 0
                ? (nonZeroDs.reduce((sum, value) => sum + value, 0) / nonZeroDs.length)
                : 0;
            const totalLength = dsMagnitudes.reduce((sum, value) => sum + value, 0);
            document.getElementById('display-avg-ds').textContent = avgDs.toFixed(3);
            document.getElementById('display-path-length').textContent = totalLength.toFixed(2);
        } else {
            document.getElementById('display-avg-ds').textContent = '--';
            document.getElementById('display-path-length').textContent = '--';
        }

        if (data.theta_opt && data.theta_opt.length > 1) {
            let totalRotation = 0;
            for (let i = 1; i < data.theta_opt.length; i += 1) {
                totalRotation += Math.abs(data.theta_opt[i] - data.theta_opt[i - 1]);
            }
            document.getElementById('display-total-rotation').textContent = ((totalRotation * 180) / Math.PI).toFixed(1);
        } else {
            document.getElementById('display-total-rotation').textContent = '--';
        }

        if (data.costs) {
            document.getElementById('display-cost-total').textContent = data.costs.total.toFixed(2);
            document.getElementById('display-cost-ref').textContent = data.costs.ref.toFixed(2);
            document.getElementById('display-cost-smooth').textContent = data.costs.smooth.toFixed(2);
            document.getElementById('display-cost-kappa').textContent = data.costs.kappa.toFixed(2);
            document.getElementById('display-cost-ds').textContent = data.costs.ds.toFixed(2);
        } else {
            document.getElementById('display-cost-total').textContent = '--';
            document.getElementById('display-cost-ref').textContent = '--';
            document.getElementById('display-cost-smooth').textContent = '--';
            document.getElementById('display-cost-kappa').textContent = '--';
            document.getElementById('display-cost-ds').textContent = '--';
        }
    };

    const runOptimization = async ({ keepCustomReference = state.reference.isCustom } = {}) => {
        const payload = { params: collectPayloadParams() };
        state.activeParams = { ...payload.params };

        if (keepCustomReference) {
            const customReference = buildCustomReferencePayload();
            if (customReference) {
                payload.custom_reference = customReference;
            }
        }

        runBtn.textContent = 'Optimizing...';
        runBtn.disabled = true;
        statusMsg.textContent = 'Running nonlinear solver...';
        statusMsg.className = 'status-msg';

        try {
            const response = await fetch('/api/smooth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            const currentMaxKappa = payload.params.max_kappa;
            const responseReference = getReferenceFromResponse(data);

            if (responseReference) {
                if (payload.custom_reference) {
                    state.reference.current = cloneReference(responseReference);
                    state.reference.isCustom = true;
                    if (!state.reference.base) {
                        state.reference.base = cloneReference(responseReference);
                    }
                } else {
                    state.reference.base = cloneReference(responseReference);
                    state.reference.current = cloneReference(responseReference);
                    state.reference.isCustom = false;
                }
                renderReferenceEditor();
            }

            updateMetrics(data);
            renderDubinsCommands(data.dubins_commands, currentMaxKappa);

            if (data.success) {
                statusMsg.textContent = payload.custom_reference
                    ? 'Optimization solved with custom reference path.'
                    : 'Optimization solved successfully!';
                statusMsg.className = 'status-msg success';
            } else {
                statusMsg.textContent = data.message || 'Optimization failed.';
                statusMsg.className = 'status-msg error';
            }

            if (data.x_ref) {
                plotCharts(data, payload.params);
            }
        } catch (error) {
            console.error(error);
            statusMsg.textContent = 'Network or server error occurred.';
            statusMsg.className = 'status-msg error';
        }

        runBtn.textContent = 'Run Optimization';
        runBtn.disabled = false;
    };

    const clearCustomReference = ({ syncInputs = true } = {}) => {
        if (!state.reference.base) {
            return;
        }

        state.reference.current = cloneReference(state.reference.base);
        state.reference.isCustom = false;
        state.reference.selectedIndex = null;
        if (syncInputs) {
            syncEndpointInputs();
        }
        renderReferenceEditor();
    };

    const getPlotCoordinates = (event) => {
        const fullLayout = referenceEditorChart._fullLayout;
        if (!fullLayout || !fullLayout.xaxis || !fullLayout.yaxis) {
            return null;
        }

        const bounds = referenceEditorChart.getBoundingClientRect();
        const plotX = event.clientX - bounds.left - fullLayout.xaxis._offset;
        const plotY = event.clientY - bounds.top - fullLayout.yaxis._offset;

        return {
            fullLayout,
            plotX,
            plotY
        };
    };

    const findNearestReferencePoint = (event) => {
        const reference = state.reference.current;
        const plotCoords = getPlotCoordinates(event);
        if (!reference || !plotCoords) {
            return null;
        }

        const { fullLayout, plotX, plotY } = plotCoords;
        if (plotX < 0 || plotY < 0 || plotX > fullLayout.xaxis._length || plotY > fullLayout.yaxis._length) {
            return null;
        }

        let bestIndex = null;
        let bestDistance = Infinity;
        for (let i = 0; i < reference.x.length; i += 1) {
            const pointX = fullLayout.xaxis.l2p(reference.x[i]);
            const pointY = fullLayout.yaxis.l2p(reference.y[i]);
            const distance = Math.hypot(pointX - plotX, pointY - plotY);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        if (bestDistance > 18) {
            return null;
        }

        return bestIndex;
    };

    const updateDraggedPoint = (event) => {
        const reference = state.reference.current;
        const dragIndex = state.reference.dragIndex;
        const plotCoords = getPlotCoordinates(event);
        if (!reference || dragIndex === null || !plotCoords) {
            return;
        }

        const { fullLayout, plotX, plotY } = plotCoords;
        const clampedX = Math.max(0, Math.min(fullLayout.xaxis._length, plotX));
        const clampedY = Math.max(0, Math.min(fullLayout.yaxis._length, plotY));

        reference.x[dragIndex] = fullLayout.xaxis.p2l(clampedX);
        reference.y[dragIndex] = fullLayout.yaxis.p2l(clampedY);
        reference.theta = computeReferenceHeadings(reference.x, reference.y, reference.gears);
        state.reference.isCustom = true;
        state.reference.selectedIndex = dragIndex;

        if (dragIndex === 0 || dragIndex === reference.x.length - 1) {
            syncEndpointInputs();
        }

        renderReferenceEditor();
    };

    referenceEditorChart.addEventListener('pointerdown', (event) => {
        const nearestIndex = findNearestReferencePoint(event);
        if (nearestIndex === null) {
            return;
        }

        event.preventDefault();
        state.reference.dragIndex = nearestIndex;
        state.reference.selectedIndex = nearestIndex;
        state.reference.movedDuringDrag = false;
        referenceEditorChart.classList.add('is-dragging');
        renderReferenceEditor();
    });

    window.addEventListener('pointermove', (event) => {
        if (state.reference.dragIndex === null) {
            return;
        }

        state.reference.movedDuringDrag = true;
        updateDraggedPoint(event);
    });

    window.addEventListener('pointerup', () => {
        if (state.reference.dragIndex === null) {
            return;
        }

        const shouldRun = state.reference.movedDuringDrag;
        state.reference.dragIndex = null;
        state.reference.movedDuringDrag = false;
        referenceEditorChart.classList.remove('is-dragging');
        updateReferenceBadge();

        if (shouldRun) {
            runOptimization({ keepCustomReference: true });
        }
    });

    runBtn.addEventListener('click', () => runOptimization({ keepCustomReference: state.reference.isCustom }));
    resetReferenceBtn.addEventListener('click', () => {
        clearCustomReference();
        runOptimization({ keepCustomReference: false });
    });

    params.forEach((param) => {
        const input = document.getElementById(param);
        const span = document.getElementById(`val_${param}`);
        if (!input || !span) {
            return;
        }

        input.addEventListener('input', () => {
            const value = parseFloat(input.value);
            if (input.step >= 1.0 || (input.max > 10 && input.step === '1')) {
                span.textContent = value.toFixed(0);
            } else {
                span.textContent = value.toFixed(2);
            }
        });

        input.addEventListener('change', () => {
            if (generatorParams.has(param)) {
                clearCustomReference({ syncInputs: false });
                runOptimization({ keepCustomReference: false });
                return;
            }

            runOptimization({ keepCustomReference: state.reference.isCustom });
        });
    });

    const tolLogInput = document.getElementById('ipopt_tol_log');
    const tolInput = document.getElementById('ipopt_tol');
    const tolSpan = document.getElementById('val_ipopt_tol');
    if (tolLogInput && tolInput && tolSpan) {
        const updateTol = () => {
            const exponent = parseInt(tolLogInput.value, 10);
            const tolValue = Math.pow(10, exponent);
            tolInput.value = tolValue;
            tolSpan.textContent = tolValue.toExponential(1);
        };

        tolLogInput.addEventListener('input', updateTol);
        tolLogInput.addEventListener('change', () => runOptimization({ keepCustomReference: state.reference.isCustom }));
        updateTol();
    }

    const fixKappaToggle = document.getElementById('fix_start_kappa');
    const kappaContainer = document.getElementById('start_kappa_container');
    fixKappaToggle.addEventListener('change', () => {
        kappaContainer.style.opacity = fixKappaToggle.checked ? '1' : '0.4';
        kappaContainer.style.pointerEvents = fixKappaToggle.checked ? 'auto' : 'none';
        runOptimization({ keepCustomReference: state.reference.isCustom });
    });

    const useDubinsToggle = document.getElementById('use_dubins');
    useDubinsToggle.addEventListener('change', () => {
        clearCustomReference({ syncInputs: false });
        runOptimization({ keepCustomReference: false });
    });

    state.activeParams = collectPayloadParams();
    runOptimization({ keepCustomReference: false });
});
