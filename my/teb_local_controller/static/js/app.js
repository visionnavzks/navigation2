const randomizeBtn = document.getElementById('randomize-btn');
const statusText = document.getElementById('status-text');
const statusBadge = document.getElementById('status-badge');
const pathCanvas = document.getElementById('path-canvas');
const dtCanvas = document.getElementById('dt-canvas');
const hoverOverlay = document.getElementById('hover-overlay');
const legendToggles = Array.from(document.querySelectorAll('.legend-toggle'));
const pathCtx = pathCanvas.getContext('2d');
const dtCtx = dtCanvas.getContext('2d');

const statsEls = {
    solveTime: document.getElementById('solve-time'),
    totalTime: document.getElementById('total-time'),
    avgDt: document.getElementById('avg-dt'),
    dtRange: document.getElementById('dt-range'),
    pointCount: document.getElementById('point-count'),
    pathLength: document.getElementById('path-length'),
    terminalError: document.getElementById('terminal-error'),
    totalCost: document.getElementById('total-cost'),
    hoverDetails: document.getElementById('hover-details'),
    initialState: document.getElementById('initial-state'),
    referenceConfig: document.getElementById('reference-config'),
    limitsConfig: document.getElementById('limits-config'),
    weightsConfig: document.getElementById('weights-config'),
    solverConfig: document.getElementById('solver-config'),
};

let currentScene = null;
let currentData = null;
let activeHoverKey = null;
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
    randomizeBtn.textContent = isLoading ? '求解中...' : '随机初始化并求解';
}

function formatSigned(value, digits = 3) {
    return `${value >= 0 ? '+' : ''}${formatNumber(value, digits)}`;
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
        statsEls.hoverDetails.className = 'detail-card placeholder';
        statsEls.hoverDetails.textContent = '移动鼠标到路径点上查看状态、时间和误差信息。';
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

    const detailHtml = rows
        .map(([label, value]) => `<div class="state-row"><span>${label}</span><strong>${value}</strong></div>`)
        .join('');
    statsEls.hoverDetails.className = 'detail-card';
    statsEls.hoverDetails.innerHTML = detailHtml;
    hoverOverlay.innerHTML = `
        <div class="hover-title">${item.label} #${item.index}</div>
        <div class="hover-grid">
            <span>x</span><strong>${formatNumber(item.x, 3)} m</strong>
            <span>y</span><strong>${formatNumber(item.y, 3)} m</strong>
            <span>theta</span><strong>${formatNumber(item.theta, 3)} rad</strong>
            <span>track err</span><strong>${Number.isFinite(item.trackError) ? `${formatNumber(item.trackError, 3)} m` : '--'}</strong>
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

function renderDtChart(data) {
    const dtValues = data.solution.dt;
    const width = dtCanvas.width;
    const height = dtCanvas.height;
    const padding = 26;
    const innerWidth = width - padding * 2;
    const innerHeight = height - padding * 2;
    const maxDt = Math.max(...dtValues, data.reference.dt_ref);
    const barWidth = innerWidth / dtValues.length;

    dtCtx.clearRect(0, 0, width, height);
    drawGrid(dtCtx, width, height);

    dtCtx.save();
    dtCtx.strokeStyle = '#0b4f6c';
    dtCtx.setLineDash([8, 8]);
    const refY = padding + innerHeight * (1 - data.reference.dt_ref / maxDt);
    dtCtx.beginPath();
    dtCtx.moveTo(padding, refY);
    dtCtx.lineTo(width - padding, refY);
    dtCtx.stroke();
    dtCtx.restore();

    dtValues.forEach((value, index) => {
        const barHeight = innerHeight * (value / maxDt);
        const x = padding + index * barWidth + 2;
        const y = height - padding - barHeight;
        dtCtx.fillStyle = value >= data.reference.dt_ref ? '#ca5a34' : '#0f766e';
        dtCtx.fillRect(x, y, Math.max(barWidth - 4, 2), barHeight);
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

async function runRandomDemo() {
    setButtonLoading(true);
    setStatus('正在随机生成起点并求解 TEB-MPC...', 'loading');
    try {
        const response = await fetch('/api/random_demo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.message || '求解失败');
        }
        currentData = data;
        activeHoverKey = null;
        renderPathView(data);
        renderDtChart(data);
        renderStats(data);
        renderConfig(data);
        renderHoverDetails(null);
        setStatus('随机初始化完成。可以在画布上悬停查看点信息，再次点击按钮可重新采样。', 'success');
    } catch (error) {
        setStatus(`失败: ${error.message}`, 'error');
    } finally {
        setButtonLoading(false);
    }
}

randomizeBtn.addEventListener('click', runRandomDemo);
pathCanvas.addEventListener('mousemove', handleCanvasMove);
pathCanvas.addEventListener('mouseleave', clearCanvasHover);
legendToggles.forEach((toggle) => {
    toggle.addEventListener('click', () => toggleLayer(toggle.dataset.layer));
});
updateLegendToggleStyles();
runRandomDemo();