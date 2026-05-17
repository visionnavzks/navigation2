const randomizeBtn = document.getElementById('randomize-btn');
const applyParamsBtn = document.getElementById('apply-params-btn');
const resetParamsBtn = document.getElementById('reset-params-btn');
const paramForm = document.getElementById('param-form');
const statusText = document.getElementById('status-text');
const statusBadge = document.getElementById('status-badge');
const pathCanvas = document.getElementById('path-canvas');
const plotlyChart = document.getElementById('plotly-chart');
const hoverOverlay = document.getElementById('hover-overlay');
const legendToggles = Array.from(document.querySelectorAll('.legend-toggle'));
const axisButtons = Array.from(document.querySelectorAll('.axis-btn'));
const pathCtx = pathCanvas.getContext('2d');

const statsEls = {
    solveTime: document.getElementById('solve-time'),
    totalTime: document.getElementById('total-time'),
    avgDt: document.getElementById('avg-dt'),
    dtRange: document.getElementById('dt-range'),
    pointCount: document.getElementById('point-count'),
    pathLength: document.getElementById('path-length'),
    terminalError: document.getElementById('terminal-error'),
    totalCost: document.getElementById('total-cost'),
    initialState: document.getElementById('initial-state'),
    referenceConfig: document.getElementById('reference-config'),
    limitsConfig: document.getElementById('limits-config'),
    weightsConfig: document.getElementById('weights-config'),
    solverConfig: document.getElementById('solver-config'),
};

let currentScene = null;
let currentData = null;
let activeHoverKey = null;
let chartAxisMode = 'time';
let defaultParameterSnapshot = null;
const layerVisibility = {
    reference: true,
    optimized: true,
    correspondence: true,
    initial: true,
};

function formatNumber(value, digits = 3) {
    return Number(value).toFixed(digits);
}

function setStatus(message, tone = 'idle') {
    statusText.textContent = message;
    statusBadge.className = `status-badge ${tone}`;
    statusBadge.textContent = tone === 'loading' ? 'Solving' : tone === 'success' ? 'Ready' : tone === 'error' ? 'Error' : 'Idle';
}

function setButtonLoading(isLoading) {
    randomizeBtn.disabled = isLoading;
    applyParamsBtn.disabled = isLoading;
    resetParamsBtn.disabled = isLoading;
    randomizeBtn.textContent = isLoading ? '求解中...' : '随机初始化并求解';
    applyParamsBtn.textContent = isLoading ? '应用中...' : '应用当前参数并求解';
}

function formatSigned(value, digits = 3) {
    return `${value >= 0 ? '+' : ''}${formatNumber(value, digits)}`;
}

function clonePlainObject(value) {
    return JSON.parse(JSON.stringify(value));
}

function getControllerConfigFromResponse(config) {
    return {
        ...(config?.limits || {}),
        ...(config?.weights || {}),
        ...(config?.solver || {}),
    };
}

function applyConfigToForm(config) {
    const groupedValues = {
        reference: config?.reference?.params || {},
        sampling: config?.sampling || {},
        controller: getControllerConfigFromResponse(config),
    };

    paramForm.querySelectorAll('[data-param-group][data-param-key]').forEach((input) => {
        const group = input.dataset.paramGroup;
        const key = input.dataset.paramKey;
        if (Object.prototype.hasOwnProperty.call(groupedValues[group] || {}, key)) {
            input.value = groupedValues[group][key];
        }
    });
}

function collectParameterPayload() {
    const payload = {
        controller_params: {},
        reference_config: {},
        sampling_config: {},
    };
    const targetMap = {
        controller: 'controller_params',
        reference: 'reference_config',
        sampling: 'sampling_config',
    };

    paramForm.querySelectorAll('[data-param-group][data-param-key]').forEach((input) => {
        const rawValue = input.value.trim();
        if (rawValue === '') {
            return;
        }
        const value = input.dataset.paramType === 'int' ? Number.parseInt(rawValue, 10) : Number.parseFloat(rawValue);
        if (Number.isNaN(value)) {
            return;
        }
        payload[targetMap[input.dataset.paramGroup]][input.dataset.paramKey] = value;
    });
    return payload;
}

function resetParameterForm() {
    if (!defaultParameterSnapshot) {
        return;
    }
    applyConfigToForm(defaultParameterSnapshot);
    setStatus('已恢复默认参数，可以点击按钮重新求解。', 'idle');
}

function drawGrid(ctx, width, height) {
    ctx.save();
    ctx.strokeStyle = 'rgba(26, 34, 48, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= width; x += 48) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y <= height; y += 48) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    ctx.restore();
}

function buildViewport(points, width, height) {
    const padding = 48;
    const xs = points.map((item) => item[0]);
    const ys = points.map((item) => item[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = Math.max(maxX - minX, 1.0);
    const spanY = Math.max(maxY - minY, 1.0);
    const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);

    return {
        project(x, y) {
            const px = padding + (x - minX) * scale;
            const py = height - padding - (y - minY) * scale;
            return [px, py];
        },
    };
}

function drawPolyline(ctx, points, viewport, color, lineWidth, dashed = false) {
    if (!points.length) {
        return;
    }
    ctx.save();
    ctx.beginPath();
    const [x0, y0] = viewport.project(points[0][0], points[0][1]);
    ctx.moveTo(x0, y0);
    for (let index = 1; index < points.length; index += 1) {
        const [px, py] = viewport.project(points[index][0], points[index][1]);
        ctx.lineTo(px, py);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    if (dashed) {
        ctx.setLineDash([10, 8]);
    }
    ctx.stroke();
    ctx.restore();
}

function drawCorrespondence(ctx, referencePoints, solutionPoints, viewport) {
    const pairCount = Math.min(referencePoints.length, solutionPoints.length);
    ctx.save();
    ctx.strokeStyle = 'rgba(141, 133, 120, 0.42)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 7]);
    for (let index = 0; index < pairCount; index += 1) {
        const [rx, ry] = viewport.project(referencePoints[index][0], referencePoints[index][1]);
        const [sx, sy] = viewport.project(solutionPoints[index][0], solutionPoints[index][1]);
        ctx.beginPath();
        ctx.moveTo(rx, ry);
        ctx.lineTo(sx, sy);
        ctx.stroke();
    }
    ctx.restore();
}

function drawReferenceMarkers(ctx, points, viewport) {
    ctx.save();
    ctx.strokeStyle = 'rgba(15, 118, 110, 0.7)';
    ctx.lineWidth = 1.6;
    points.forEach(([x, y]) => {
        const [px, py] = viewport.project(x, y);
        ctx.beginPath();
        ctx.moveTo(px - 5, py - 5);
        ctx.lineTo(px + 5, py + 5);
        ctx.moveTo(px + 5, py - 5);
        ctx.lineTo(px - 5, py + 5);
        ctx.stroke();
    });
    ctx.restore();
}

function drawOptimizedMarkers(ctx, points, viewport) {
    ctx.save();
    points.forEach(([x, y]) => {
        const [px, py] = viewport.project(x, y);
        ctx.beginPath();
        ctx.fillStyle = 'rgba(202, 90, 52, 0.5)';
        ctx.strokeStyle = 'rgba(255, 247, 242, 0.9)';
        ctx.lineWidth = 1;
        ctx.arc(px, py, 4.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    });
    ctx.restore();
}

function drawHeadingArrows(ctx, points, headings, viewport, color, step, alpha) {
    for (let index = 0; index < points.length; index += step) {
        drawArrow(ctx, viewport, points[index][0], points[index][1], headings[index], color, 14, alpha, 1.3);
    }
    if (points.length > 1) {
        const last = points.length - 1;
        drawArrow(ctx, viewport, points[last][0], points[last][1], headings[last], color, 14, alpha, 1.3);
    }
}

function updateLegendToggleStyles() {
    legendToggles.forEach((toggle) => {
        toggle.classList.toggle('active', layerVisibility[toggle.dataset.layer]);
    });
}

function updateAxisButtonStyles() {
    axisButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.axisMode === chartAxisMode);
    });
}

function drawArrow(ctx, viewport, x, y, theta, color, arrowLength = 20, alpha = 1, lineWidth = 2) {
    const [px, py] = viewport.project(x, y);
    const tipX = px + arrowLength * Math.cos(theta);
    const tipY = py - arrowLength * Math.sin(theta);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - 8 * Math.cos(theta - 0.4), tipY + 8 * Math.sin(theta - 0.4));
    ctx.lineTo(tipX - 8 * Math.cos(theta + 0.4), tipY + 8 * Math.sin(theta + 0.4));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
}

function buildHoverItems(data, viewport) {
    const items = [];
    const referenceCount = data.reference.x.length;
    const solutionCount = data.solution.x.length;
    const solutionS = [0];
    for (let index = 1; index < solutionCount; index += 1) {
        const ds = Math.hypot(
            data.solution.x[index] - data.solution.x[index - 1],
            data.solution.y[index] - data.solution.y[index - 1],
        );
        solutionS.push(solutionS[index - 1] + ds);
    }

    if (layerVisibility.reference) {
        for (let index = 0; index < referenceCount; index += 1) {
            const [sx, sy] = viewport.project(data.reference.x[index], data.reference.y[index]);
            items.push({
                key: `reference-${index}`,
                label: 'Reference',
                index,
                sx,
                sy,
                x: data.reference.x[index],
                y: data.reference.y[index],
                s: data.reference.s[index],
                theta: data.reference.theta[index],
                v: data.reference.v[index],
                a: data.reference.a[index],
                kappa: data.reference.kappa[index],
            });
        }
    }

    if (layerVisibility.optimized) {
        for (let index = 0; index < solutionCount; index += 1) {
            const matchIndex = Math.min(index, referenceCount - 1);
            const [sx, sy] = viewport.project(data.solution.x[index], data.solution.y[index]);
            items.push({
                key: `solution-${index}`,
                label: 'Optimized',
                index,
                sx,
                sy,
                x: data.solution.x[index],
                y: data.solution.y[index],
                s: solutionS[index],
                theta: data.solution.theta[index],
                v: data.solution.v[index],
                a: data.solution.a[index],
                kappa: data.solution.kappa[index],
                time: data.solution.time[index],
                dt: index < data.solution.dt.length ? data.solution.dt[index] : NaN,
                jerk: index < data.solution.jerk.length ? data.solution.jerk[index] : NaN,
                dkappa: index < data.solution.dkappa.length ? data.solution.dkappa[index] : NaN,
                trackError: Math.hypot(data.solution.x[index] - data.reference.x[matchIndex], data.solution.y[index] - data.reference.y[matchIndex]),
                headingError: data.solution.theta[index] - data.reference.theta[matchIndex],
            });
        }
    }

    if (layerVisibility.initial) {
        const [initialSx, initialSy] = viewport.project(data.initial_state.x, data.initial_state.y);
        items.push({
            key: 'initial-0',
            label: 'Initial',
            index: 0,
            sx: initialSx,
            sy: initialSy,
            x: data.initial_state.x,
            y: data.initial_state.y,
            theta: data.initial_state.theta,
            v: data.initial_state.v,
            a: data.initial_state.a,
            kappa: data.initial_state.kappa,
            trackError: Math.hypot(data.initial_state.x - data.reference.x[0], data.initial_state.y - data.reference.y[0]),
            headingError: data.initial_state.theta - data.reference.theta[0],
        });
    }
    return items;
}

function renderHoverDetails(item) {
    if (!item) {
        hoverOverlay.classList.add('hidden');
        return;
    }

    const rows = [
        ['点类型', `${item.label} #${item.index}`],
        ['x', `${formatNumber(item.x, 3)} m`],
        ['y', `${formatNumber(item.y, 3)} m`],
        ['s', `${formatNumber(item.s ?? 0, 3)} m`],
        ['theta', `${formatNumber(item.theta, 3)} rad`],
        ['v', `${formatNumber(item.v, 3)} m/s`],
        ['a', `${formatNumber(item.a, 3)} m/s²`],
        ['kappa', `${formatNumber(item.kappa, 3)} 1/m`],
    ];
    if (Number.isFinite(item.time)) rows.push(['t', `${formatNumber(item.time, 3)} s`]);
    if (Number.isFinite(item.dt)) rows.push(['dt', `${formatNumber(item.dt, 3)} s`]);
    if (Number.isFinite(item.jerk)) rows.push(['jerk', `${formatNumber(item.jerk, 3)} m/s³`]);
    if (Number.isFinite(item.dkappa)) rows.push(['dkappa', `${formatNumber(item.dkappa, 3)} 1/(m*s)`]);
    if (Number.isFinite(item.trackError)) rows.push(['track err', `${formatNumber(item.trackError, 3)} m`]);
    if (Number.isFinite(item.headingError)) rows.push(['heading err', `${formatSigned(item.headingError, 3)} rad`]);

    hoverOverlay.innerHTML = `
        <div class="hover-title">${item.label} #${item.index}</div>
        <div class="hover-grid">
            ${rows.map(([label, value]) => `<span>${label}</span><strong>${value}</strong>`).join('')}
        </div>
    `;
    hoverOverlay.classList.remove('hidden');
}

function positionHoverOverlay(event) {
    const rect = pathCanvas.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const maxX = rect.width - hoverOverlay.offsetWidth - 10;
    const maxY = rect.height - hoverOverlay.offsetHeight - 10;
    hoverOverlay.style.left = `${Math.max(10, Math.min(localX + 12, maxX))}px`;
    hoverOverlay.style.top = `${Math.max(10, Math.min(localY + 12, maxY))}px`;
}

function renderPathView(data, activeKey = null) {
    const { reference, solution, initial_state: initialState } = data;
    const referencePoints = reference.x.map((x, index) => [x, reference.y[index]]);
    const solutionPoints = solution.x.map((x, index) => [x, solution.y[index]]);
    const allPoints = [...referencePoints, ...solutionPoints, [initialState.x, initialState.y]];
    const viewport = buildViewport(allPoints, pathCanvas.width, pathCanvas.height);

    pathCtx.clearRect(0, 0, pathCanvas.width, pathCanvas.height);
    drawGrid(pathCtx, pathCanvas.width, pathCanvas.height);
    if (layerVisibility.correspondence && layerVisibility.reference && layerVisibility.optimized) {
        drawCorrespondence(pathCtx, referencePoints, solutionPoints, viewport);
    }
    if (layerVisibility.reference) {
        drawPolyline(pathCtx, referencePoints, viewport, 'rgba(15, 118, 110, 0.62)', 3.6, true);
        drawReferenceMarkers(pathCtx, referencePoints, viewport);
        drawHeadingArrows(pathCtx, referencePoints, reference.theta, viewport, '#0f766e', 5, 0.45);
    }
    if (layerVisibility.optimized) {
        drawPolyline(pathCtx, solutionPoints, viewport, 'rgba(202, 90, 52, 0.78)', 3.8, false);
        drawOptimizedMarkers(pathCtx, solutionPoints, viewport);
        drawHeadingArrows(pathCtx, solutionPoints, solution.theta, viewport, '#ca5a34', 5, 0.52);
    }
    if (layerVisibility.initial) {
        drawArrow(pathCtx, viewport, initialState.x, initialState.y, initialState.theta, '#d97706', 18, 0.9, 1.7);
    }

    const hoverItems = buildHoverItems(data, viewport);
    const hoveredItem = activeKey ? hoverItems.find((item) => item.key === activeKey) : null;
    if (hoveredItem) {
        pathCtx.save();
        pathCtx.fillStyle = 'rgba(217, 119, 6, 0.22)';
        pathCtx.strokeStyle = '#d97706';
        pathCtx.lineWidth = 2;
        pathCtx.beginPath();
        pathCtx.arc(hoveredItem.sx, hoveredItem.sy, 11, 0, Math.PI * 2);
        pathCtx.fill();
        pathCtx.stroke();
        pathCtx.restore();
    }

    currentScene = { viewport, hoverItems };
}

function buildSolutionDistance(data) {
    const result = [0];
    for (let index = 1; index < data.solution.x.length; index += 1) {
        const dx = data.solution.x[index] - data.solution.x[index - 1];
        const dy = data.solution.y[index] - data.solution.y[index - 1];
        result.push(result[index - 1] + Math.hypot(dx, dy));
    }
    return result;
}

function buildMidpoints(values) {
    const result = [];
    for (let index = 0; index < values.length - 1; index += 1) {
        result.push((values[index] + values[index + 1]) * 0.5);
    }
    return result;
}

function renderPlotlyCharts(data) {
    const solutionDistance = buildSolutionDistance(data);
    const stateAxis = chartAxisMode === 'time' ? data.solution.time : solutionDistance;
    const controlAxis = chartAxisMode === 'time' ? buildMidpoints(data.solution.time) : buildMidpoints(solutionDistance);
    const referenceAxis = chartAxisMode === 'time'
        ? data.reference.s.map((value) => value / Math.max(data.config.reference.cruise_speed, 1e-6))
        : data.reference.s;
    const xTitle = chartAxisMode === 'time' ? 'time [s]' : 'distance [m]';
    const dtRef = data.reference.dt_ref;

    const traces = [
        {
            x: stateAxis,
            y: data.solution.v,
            mode: 'lines+markers',
            name: 'optimized v',
            line: { color: '#ca5a34', width: 2.4 },
            marker: { size: 6 },
            xaxis: 'x',
            yaxis: 'y',
        },
        {
            x: referenceAxis,
            y: data.reference.v,
            mode: 'lines',
            name: 'ref v',
            line: { color: '#0f766e', width: 2, dash: 'dash' },
            xaxis: 'x',
            yaxis: 'y',
        },
        {
            x: stateAxis,
            y: data.solution.a,
            mode: 'lines+markers',
            name: 'optimized a',
            line: { color: '#d97706', width: 2.2 },
            marker: { size: 6 },
            xaxis: 'x2',
            yaxis: 'y2',
        },
        {
            x: referenceAxis,
            y: data.reference.a,
            mode: 'lines',
            name: 'ref a',
            line: { color: '#0f766e', width: 2, dash: 'dash' },
            xaxis: 'x2',
            yaxis: 'y2',
        },
        {
            x: stateAxis,
            y: data.solution.kappa,
            mode: 'lines+markers',
            name: 'optimized kappa',
            line: { color: '#8b5cf6', width: 2.2 },
            marker: { size: 6 },
            xaxis: 'x3',
            yaxis: 'y3',
        },
        {
            x: referenceAxis,
            y: data.reference.kappa,
            mode: 'lines',
            name: 'ref kappa',
            line: { color: '#0f766e', width: 2, dash: 'dash' },
            xaxis: 'x3',
            yaxis: 'y3',
        },
        {
            x: controlAxis,
            y: data.solution.dt,
            mode: 'lines+markers',
            name: 'dt',
            line: { color: '#7b655a', width: 2.2 },
            marker: {
                size: 6,
                color: data.solution.dt.map((value) => (value >= dtRef ? '#ca5a34' : '#0f766e')),
            },
            xaxis: 'x4',
            yaxis: 'y4',
        },
        {
            x: [controlAxis[0] ?? 0, controlAxis[controlAxis.length - 1] ?? 1],
            y: [dtRef, dtRef],
            mode: 'lines',
            name: 'dt_ref',
            line: { color: '#0b4f6c', width: 2, dash: 'dot' },
            xaxis: 'x4',
            yaxis: 'y4',
        },
        {
            x: controlAxis,
            y: data.solution.jerk,
            mode: 'lines+markers',
            name: 'jerk',
            line: { color: '#0b4f6c', width: 2.2 },
            marker: { size: 6 },
            xaxis: 'x5',
            yaxis: 'y5',
        },
        {
            x: controlAxis,
            y: data.solution.dkappa,
            mode: 'lines+markers',
            name: 'dkappa',
            line: { color: '#0f766e', width: 2.2 },
            marker: { size: 6 },
            xaxis: 'x6',
            yaxis: 'y6',
        },
    ];

    const axisStyle = {
        showgrid: true,
        gridcolor: 'rgba(26, 34, 48, 0.08)',
        zeroline: false,
        linecolor: 'rgba(26, 34, 48, 0.14)',
        ticks: 'outside',
        tickfont: { family: 'Space Grotesk, Noto Sans SC, sans-serif', size: 11 },
        titlefont: { family: 'Space Grotesk, Noto Sans SC, sans-serif', size: 12 },
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 56, r: 24, t: 44, b: 42 },
        font: { family: 'Space Grotesk, Noto Sans SC, sans-serif', color: '#1a2230' },
        grid: { rows: 2, columns: 3, pattern: 'independent' },
        legend: {
            orientation: 'h',
            x: 0,
            xanchor: 'left',
            y: 1.08,
            yanchor: 'bottom',
            bgcolor: 'rgba(255,255,255,0.58)',
        },
        xaxis: { ...axisStyle, title: xTitle },
        yaxis: { ...axisStyle, title: 'v [m/s]' },
        xaxis2: { ...axisStyle, title: xTitle },
        yaxis2: { ...axisStyle, title: 'a [m/s²]' },
        xaxis3: { ...axisStyle, title: xTitle },
        yaxis3: { ...axisStyle, title: 'kappa [1/m]' },
        xaxis4: { ...axisStyle, title: xTitle },
        yaxis4: { ...axisStyle, title: 'dt [s]' },
        xaxis5: { ...axisStyle, title: xTitle },
        yaxis5: { ...axisStyle, title: 'jerk [m/s³]' },
        xaxis6: { ...axisStyle, title: xTitle },
        yaxis6: { ...axisStyle, title: 'dkappa [1/(m*s)]' },
    };

    Plotly.react(plotlyChart, traces, layout, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    });
}

function renderStats(data) {
    const dtValues = data.solution.dt;
    const totalTime = data.solution.time[data.solution.time.length - 1];
    const minDt = Math.min(...dtValues);
    const maxDt = Math.max(...dtValues);
    const pathLength = data.solution.x.slice(1).reduce((sum, _value, index) => {
        const dx = data.solution.x[index + 1] - data.solution.x[index];
        const dy = data.solution.y[index + 1] - data.solution.y[index];
        return sum + Math.hypot(dx, dy);
    }, 0);
    const terminalError = Math.hypot(
        data.solution.x[data.solution.x.length - 1] - data.reference.x[data.reference.x.length - 1],
        data.solution.y[data.solution.y.length - 1] - data.reference.y[data.reference.y.length - 1],
    );

    statsEls.solveTime.textContent = `${formatNumber(data.solution.solve_time_ms, 1)} ms`;
    statsEls.totalTime.textContent = `${formatNumber(totalTime, 2)} s`;
    statsEls.avgDt.textContent = `${formatNumber(dtValues.reduce((sum, value) => sum + value, 0) / dtValues.length, 3)} s`;
    statsEls.dtRange.textContent = `${formatNumber(minDt, 3)} - ${formatNumber(maxDt, 3)} s`;
    statsEls.pointCount.textContent = `${data.reference.x.length}`;
    statsEls.pathLength.textContent = `${formatNumber(pathLength, 2)} m`;
    statsEls.terminalError.textContent = `${formatNumber(terminalError, 3)} m`;
    statsEls.totalCost.textContent = formatNumber(data.solution.costs.total, 2);

    const initialState = data.initial_state;
    statsEls.initialState.innerHTML = [
        ['x', `${formatNumber(initialState.x, 3)} m`],
        ['y', `${formatNumber(initialState.y, 3)} m`],
        ['theta', `${formatNumber(initialState.theta, 3)} rad`],
        ['v', `${formatNumber(initialState.v, 3)} m/s`],
        ['a', `${formatNumber(initialState.a, 3)} m/s²`],
        ['kappa', `${formatNumber(initialState.kappa, 3)} 1/m`],
    ].map(([label, value]) => `<div class="state-row"><span>${label}</span><strong>${value}</strong></div>`).join('');
}

function renderConfig(data) {
    const { config } = data;
    const reference = config.reference;
    const limits = config.limits;
    const weights = config.weights;
    const solver = config.solver;

    statsEls.referenceConfig.innerHTML = `
        <div class="config-stack">ds = ${formatNumber(reference.ds, 2)} m, cruise = ${formatNumber(reference.cruise_speed, 2)} m/s, dt_ref = ${formatNumber(reference.dt_ref, 2)} s</div>
        <div class="config-stack">segments (${reference.segment_count})</div>
        ${reference.segment_descriptions.map((segment) => `<div class="config-stack">${segment}</div>`).join('')}
    `;

    statsEls.limitsConfig.innerHTML = [
        ['dt', `[${formatNumber(limits.dt_min, 2)}, ${formatNumber(limits.dt_max, 2)}] s`],
        ['max_speed', `${formatNumber(limits.max_speed, 2)} m/s`],
        ['max_accel', `${formatNumber(limits.max_accel, 2)} m/s²`],
        ['max_jerk', `${formatNumber(limits.max_jerk, 2)} m/s³`],
        ['max_kappa', `${formatNumber(limits.max_kappa, 2)} 1/m`],
        ['max_dkappa', `${formatNumber(limits.max_dkappa, 2)} 1/(m*s)`],
    ].map(([label, value]) => `<div class="config-row"><span>${label}</span><strong>${value}</strong></div>`).join('');

    statsEls.weightsConfig.innerHTML = Object.entries(weights)
        .map(([label, value]) => `<div class="config-row"><span>${label}</span><strong>${formatNumber(value, 2)}</strong></div>`)
        .join('');

    statsEls.solverConfig.innerHTML = [
        ['ipopt_max_iter', solver.ipopt_max_iter],
        ['ipopt_tol', Number(solver.ipopt_tol).toExponential(1)],
        ['ipopt_print_level', solver.ipopt_print_level],
    ].map(([label, value]) => `<div class="config-row"><span>${label}</span><strong>${value}</strong></div>`).join('');
}

function findNearestHover(event) {
    if (!currentScene) {
        return null;
    }
    const rect = pathCanvas.getBoundingClientRect();
    const scaleX = pathCanvas.width / rect.width;
    const scaleY = pathCanvas.height / rect.height;
    const cursorX = (event.clientX - rect.left) * scaleX;
    const cursorY = (event.clientY - rect.top) * scaleY;
    let best = null;
    let bestDistance = 16;
    currentScene.hoverItems.forEach((item) => {
        const distance = Math.hypot(item.sx - cursorX, item.sy - cursorY);
        if (distance < bestDistance) {
            best = item;
            bestDistance = distance;
        }
    });
    return best;
}

function handleCanvasMove(event) {
    if (!currentData) {
        return;
    }
    const nearest = findNearestHover(event);
    if (!nearest) {
        if (activeHoverKey !== null) {
            activeHoverKey = null;
            renderPathView(currentData);
        }
        renderHoverDetails(null);
        return;
    }

    positionHoverOverlay(event);
    if (nearest.key !== activeHoverKey) {
        activeHoverKey = nearest.key;
        renderPathView(currentData, activeHoverKey);
    }
    renderHoverDetails(nearest);
}

function clearCanvasHover() {
    activeHoverKey = null;
    hoverOverlay.classList.add('hidden');
    renderHoverDetails(null);
    if (currentData) {
        renderPathView(currentData);
    }
}

function toggleLayer(layer) {
    layerVisibility[layer] = !layerVisibility[layer];
    updateLegendToggleStyles();
    activeHoverKey = null;
    renderHoverDetails(null);
    if (currentData) {
        renderPathView(currentData);
    }
}

function setChartAxisMode(mode) {
    chartAxisMode = mode;
    updateAxisButtonStyles();
    if (currentData) {
        renderPlotlyCharts(currentData);
    }
}

async function runRandomDemo() {
    setButtonLoading(true);
    setStatus('正在随机生成起点并求解 TEB-MPC...', 'loading');
    try {
        const payload = collectParameterPayload();
        const response = await fetch('/api/random_demo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.message || '求解失败');
        }
        currentData = data;
        activeHoverKey = null;
        renderPathView(data);
        renderPlotlyCharts(data);
        renderStats(data);
        renderConfig(data);
        applyConfigToForm(data.config);
        if (!defaultParameterSnapshot) {
            defaultParameterSnapshot = clonePlainObject(data.config);
        }
        renderHoverDetails(null);
        setStatus('随机初始化完成。可以在画布上悬停查看点信息，再次点击按钮可重新采样。', 'success');
    } catch (error) {
        setStatus(`失败: ${error.message}`, 'error');
    } finally {
        setButtonLoading(false);
    }
}

randomizeBtn.addEventListener('click', runRandomDemo);
applyParamsBtn.addEventListener('click', runRandomDemo);
resetParamsBtn.addEventListener('click', resetParameterForm);
paramForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runRandomDemo();
});
pathCanvas.addEventListener('mousemove', handleCanvasMove);
pathCanvas.addEventListener('mouseleave', clearCanvasHover);
legendToggles.forEach((toggle) => {
    toggle.addEventListener('click', () => toggleLayer(toggle.dataset.layer));
});
axisButtons.forEach((button) => {
    button.addEventListener('click', () => setChartAxisMode(button.dataset.axisMode));
});
updateLegendToggleStyles();
updateAxisButtonStyles();
runRandomDemo();