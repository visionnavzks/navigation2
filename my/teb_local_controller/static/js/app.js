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

const PARAM_HELP_TEXTS = {
    ds: '参考轨迹按弧长离散时的采样间距，单位 m。值越小，参考点越密，跟踪更细，但优化变量更多、求解更慢。',
    cruise_speed: '参考轨迹的名义巡航速度，单位 m/s。它会影响参考速度曲线，也会影响按时间显示时参考曲线的横轴换算。',
    dt_ref: '名义时间步长，单位 s。优化中的 dt 会围绕它变化，w_dt 越大，实际 dt 越不愿意偏离这个值。',
    line_1_length: '第一段直线的长度，单位 m。改变它会直接拉长或缩短参考路径的开头。',
    arc_1_radius: '第一段圆弧的半径，单位 m。半径越小，转弯越急；半径越大，转弯越缓。',
    arc_1_angle: '第一段圆弧的转角，输入单位 rad。正值表示逆时针，负值表示顺时针。0.785 rad 大约等于 45 deg。',
    line_2_length: '第二段直线的长度，单位 m。通常用来调中段过渡的直线长度。',
    arc_2_radius: '第二段圆弧的半径，单位 m。会影响后半段转弯曲率大小。',
    arc_2_angle: '第二段圆弧的转角，输入单位 rad。负值表示顺时针回转，-0.524 rad 约等于 -30 deg。',
    line_3_length: '最后一段直线的长度，单位 m。决定终点前的收尾距离。',
    x_offset_range: '随机初始状态在 x 方向相对参考起点的采样范围，单位 m。实际采样区间是 [-range, +range]。',
    y_offset_range: '随机初始状态在 y 方向相对参考起点的采样范围，单位 m。值越大，初始横向偏差越大。',
    theta_offset_range: '随机初始航向相对参考起点的采样范围，输入单位 rad。实际采样区间是 [-range, +range]，0.7 rad 约等于 40 deg。',
    speed_min: '随机初始速度的最小值，单位 m/s。用于生成新的起始状态。',
    speed_max: '随机初始速度的最大值，单位 m/s。它不等于控制器速度上限，只影响随机起点采样。',
    accel_min: '随机初始加速度的最小值，单位 m/s²。',
    accel_max: '随机初始加速度的最大值，单位 m/s²。',
    kappa_offset_range: '随机初始曲率相对参考起点的扰动范围，单位 1/m。值越大，起步转向偏差越明显。',
    kappa_min: '随机初始曲率的下界，单位 1/m。与 kappa_offset_range 一起决定随机起点的曲率范围。',
    kappa_max: '随机初始曲率的上界，单位 1/m。',
    dt_min: '优化允许的最小时间步长，单位 s。减小它会允许更细的时间伸缩，但可能让问题更难。',
    dt_max: '优化允许的最大时间步长，单位 s。增大它会允许轨迹在某些段上明显放慢。',
    max_speed: '优化状态中的速度上界，单位 m/s。它是硬约束，不是参考速度。',
    max_accel: '优化状态中的加速度绝对值上界，单位 m/s²。',
    max_jerk: '控制量 jerk 的绝对值上界，单位 m/s³。越小表示速度变化更平滑，但机动性更弱。',
    max_kappa: '曲率绝对值上界，单位 1/m。越小表示允许的转弯半径更大。',
    max_dkappa: '曲率变化率绝对值上界，单位 1/(m*s)。越小表示转向变化更平滑。',
    w_pos: '位置跟踪权重。越大，优化越优先贴近参考路径的 x/y 位置，但控制代价和光滑性可能被压制。',
    w_theta: '终端航向误差权重。当前实现中 theta 只在终点代价里使用，所以它主要决定末端朝向是否对齐。',
    w_speed: '速度跟踪权重。越大，优化速度曲线越接近参考速度。',
    w_accel: '当前实现里该权重已保留在参数面板中，但中间跟踪项不再使用加速度参考，所以它目前不会改变结果。',
    w_kappa: '当前实现里该权重已保留在参数面板中，但中间跟踪项不再使用曲率参考，所以它目前不会改变结果。',
    w_dt: '时间弹性权重。越大，dt 越接近 dt_ref；越小，优化越愿意拉伸或压缩时间分配。',
    w_jerk: 'jerk 平滑权重。越大，速度变化更平顺，但响应更保守。',
    w_dkappa: '曲率变化率平滑权重。越大，转向变化更柔和。',
    w_terminal: '终端状态权重。越大，优化越强调最后一个点在位置、速度和航向上贴近参考终点。',
    ipopt_max_iter: 'IPOPT 最大迭代次数。遇到复杂参数组合时可以适当增大。',
    ipopt_tol: 'IPOPT 收敛容差。数值越小，解要求越严格，通常也会更慢。',
    ipopt_print_level: 'IPOPT 日志等级。0 表示几乎不打印，更高值会输出更多求解细节。',
};

let currentScene = null;
let currentData = null;
let activeHoverKey = null;
let chartAxisMode = 'time';
let defaultParameterSnapshot = null;
let autoReplanTimer = null;
let solveInFlight = false;
let pendingAutoReplanOptions = null;
let globalParamTooltip = null;
let activeParamTooltipAnchor = null;
let isDraggingInitialState = false;
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

function shouldPreserveInitialStateForInput(input) {
    return input.dataset.paramGroup !== 'sampling';
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

function ensureGlobalParamTooltip() {
    if (globalParamTooltip) {
        return globalParamTooltip;
    }

    globalParamTooltip = document.createElement('div');
    globalParamTooltip.className = 'param-tooltip';
    globalParamTooltip.setAttribute('role', 'tooltip');
    document.body.appendChild(globalParamTooltip);
    return globalParamTooltip;
}

function hideGlobalParamTooltip() {
    if (!globalParamTooltip) {
        return;
    }
    globalParamTooltip.classList.remove('visible');
    activeParamTooltipAnchor = null;
}

function positionGlobalParamTooltip(anchor) {
    if (!globalParamTooltip || !anchor) {
        return;
    }

    const margin = 12;
    const offset = 10;
    const anchorRect = anchor.getBoundingClientRect();

    globalParamTooltip.style.maxWidth = `${Math.min(320, window.innerWidth - margin * 2)}px`;
    globalParamTooltip.style.left = `${margin}px`;
    globalParamTooltip.style.top = `${margin}px`;

    const tooltipRect = globalParamTooltip.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - tooltipRect.width - margin);
    const left = Math.min(Math.max(anchorRect.left, margin), maxLeft);

    let top = anchorRect.bottom + offset;
    if (top + tooltipRect.height > window.innerHeight - margin) {
        top = anchorRect.top - tooltipRect.height - offset;
    }
    top = Math.max(margin, top);

    globalParamTooltip.style.left = `${left}px`;
    globalParamTooltip.style.top = `${top}px`;
}

function showGlobalParamTooltip(anchor, text) {
    const tooltip = ensureGlobalParamTooltip();
    activeParamTooltipAnchor = anchor;
    tooltip.textContent = text;
    tooltip.classList.add('visible');
    positionGlobalParamTooltip(anchor);
}

function initParameterTooltips() {
    paramForm.querySelectorAll('.param-field').forEach((field) => {
        const input = field.querySelector('input[data-param-key]');
        const labelSpan = field.querySelector('span');
        if (!input || !labelSpan) {
            return;
        }
        const helpText = PARAM_HELP_TEXTS[input.dataset.paramKey];
        if (!helpText || field.querySelector('.param-label-wrap')) {
            return;
        }

        const labelWrap = document.createElement('span');
        labelWrap.className = 'param-label-wrap';

        const labelText = document.createElement('span');
        labelText.textContent = labelSpan.textContent;

        const helpButton = document.createElement('button');
        helpButton.type = 'button';
        helpButton.className = 'param-help-btn';
        helpButton.textContent = '?';
        helpButton.setAttribute('aria-label', `${labelSpan.textContent} 参数说明`);

        labelWrap.addEventListener('mouseenter', () => showGlobalParamTooltip(helpButton, helpText));
        labelWrap.addEventListener('mouseleave', hideGlobalParamTooltip);
        helpButton.addEventListener('focus', () => showGlobalParamTooltip(helpButton, helpText));
        helpButton.addEventListener('blur', hideGlobalParamTooltip);

        labelWrap.appendChild(labelText);
        labelWrap.appendChild(helpButton);
        labelSpan.replaceWith(labelWrap);
    });

    paramForm.addEventListener('scroll', hideGlobalParamTooltip, { passive: true });
    window.addEventListener('resize', () => positionGlobalParamTooltip(activeParamTooltipAnchor), { passive: true });
    window.addEventListener('scroll', () => positionGlobalParamTooltip(activeParamTooltipAnchor), { passive: true });
}

function scheduleAutoReplan(input) {
    if (!currentData) {
        return;
    }
    if (autoReplanTimer !== null) {
        clearTimeout(autoReplanTimer);
    }
    const preserveInitialState = shouldPreserveInitialStateForInput(input);
    setStatus(
        preserveInitialState ? '参数已变更，正在等待基于当前状态自动重规划...' : '采样参数已变更，正在等待自动重新采样并求解...',
        'idle',
    );
    autoReplanTimer = window.setTimeout(() => {
        autoReplanTimer = null;
        runRandomDemo({ preserveInitialState, autoTriggered: true });
    }, 450);
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
        unproject(px, py) {
            const x = minX + (px - padding) / scale;
            const y = minY + (height - padding - py) / scale;
            return [x, y];
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
    const solverReference = data.reference;
    const displayReference = data.display_reference || data.reference;
    const referenceCount = solverReference.x.length;
    const displayReferenceCount = displayReference.x.length;
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
        for (let index = 0; index < displayReferenceCount; index += 1) {
            const [sx, sy] = viewport.project(displayReference.x[index], displayReference.y[index]);
            items.push({
                key: `reference-${index}`,
                label: 'Reference',
                index,
                sx,
                sy,
                x: displayReference.x[index],
                y: displayReference.y[index],
                s: displayReference.s[index],
                theta: displayReference.theta[index],
                v: displayReference.v[index],
                a: displayReference.a[index],
                kappa: displayReference.kappa[index],
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
                trackError: Math.hypot(data.solution.x[index] - solverReference.x[matchIndex], data.solution.y[index] - solverReference.y[matchIndex]),
                headingError: data.solution.theta[index] - solverReference.theta[matchIndex],
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
            s: 0,
            theta: data.initial_state.theta,
            v: data.initial_state.v,
            a: data.initial_state.a,
            kappa: data.initial_state.kappa,
            time: 0,
            dt: data.solution.dt.length > 0 ? data.solution.dt[0] : NaN,
            jerk: data.solution.jerk.length > 0 ? data.solution.jerk[0] : NaN,
            dkappa: data.solution.dkappa.length > 0 ? data.solution.dkappa[0] : NaN,
            trackError: Math.hypot(data.initial_state.x - solverReference.x[0], data.initial_state.y - solverReference.y[0]),
            headingError: data.initial_state.theta - solverReference.theta[0],
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
    const displayReference = data.display_reference || data.reference;
    const referencePoints = displayReference.x.map((x, index) => [x, displayReference.y[index]]);
    const solverReferencePoints = reference.x.map((x, index) => [x, reference.y[index]]);
    const solutionPoints = solution.x.map((x, index) => [x, solution.y[index]]);
    const allPoints = [...referencePoints, ...solutionPoints, [initialState.x, initialState.y]];
    const viewport = buildViewport(allPoints, pathCanvas.width, pathCanvas.height);

    pathCtx.clearRect(0, 0, pathCanvas.width, pathCanvas.height);
    drawGrid(pathCtx, pathCanvas.width, pathCanvas.height);
    if (layerVisibility.correspondence && layerVisibility.reference && layerVisibility.optimized) {
        drawCorrespondence(pathCtx, solverReferencePoints, solutionPoints, viewport);
    }
    if (layerVisibility.reference) {
        drawPolyline(pathCtx, referencePoints, viewport, 'rgba(15, 118, 110, 0.62)', 3.6, true);
        drawReferenceMarkers(pathCtx, referencePoints, viewport);
        drawHeadingArrows(pathCtx, referencePoints, displayReference.theta, viewport, '#0f766e', 5, 0.45);
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
    const { cursorX, cursorY } = getCanvasCursorPosition(event);
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

function findInitialDragTarget(event) {
    if (!currentScene || !layerVisibility.initial) {
        return null;
    }
    const { cursorX, cursorY } = getCanvasCursorPosition(event);
    const initialItem = currentScene.hoverItems.find((item) => item.key === 'initial-0');
    if (!initialItem) {
        return null;
    }
    const distance = Math.hypot(initialItem.sx - cursorX, initialItem.sy - cursorY);
    return distance <= 18 ? initialItem : null;
}

function getCanvasCursorPosition(event) {
    const rect = pathCanvas.getBoundingClientRect();
    const scaleX = pathCanvas.width / rect.width;
    const scaleY = pathCanvas.height / rect.height;
    return {
        cursorX: (event.clientX - rect.left) * scaleX,
        cursorY: (event.clientY - rect.top) * scaleY,
    };
}

function updateCanvasCursor(nearest = null) {
    if (isDraggingInitialState) {
        pathCanvas.style.cursor = 'grabbing';
        return;
    }
    if (nearest?.key === 'initial-0') {
        pathCanvas.style.cursor = 'grab';
        return;
    }
    pathCanvas.style.cursor = 'default';
}

function updateDraggedInitialState(event) {
    if (!isDraggingInitialState || !currentScene?.viewport || !currentData?.initial_state) {
        return null;
    }

    const { cursorX, cursorY } = getCanvasCursorPosition(event);
    const [x, y] = currentScene.viewport.unproject(cursorX, cursorY);
    currentData.initial_state = {
        ...currentData.initial_state,
        x,
        y,
    };
    activeHoverKey = 'initial-0';
    renderPathView(currentData, activeHoverKey);

    const draggedItem = currentScene?.hoverItems.find((item) => item.key === 'initial-0') || null;
    if (draggedItem) {
        renderHoverDetails(draggedItem);
        positionHoverOverlay(event);
    }
    return draggedItem;
}

function beginInitialStateDrag(event) {
    if (!currentData || !layerVisibility.initial) {
        return;
    }
    const nearest = findInitialDragTarget(event);
    if (nearest?.key !== 'initial-0') {
        return;
    }

    isDraggingInitialState = true;
    activeHoverKey = 'initial-0';
    pathCanvas.style.cursor = 'grabbing';
    renderPathView(currentData, activeHoverKey);
    renderHoverDetails(nearest);
    positionHoverOverlay(event);
    setStatus('拖动起点中，松开鼠标后会基于新的位置重规划。', 'idle');
    event.preventDefault();
}

function finishInitialStateDrag() {
    if (!isDraggingInitialState) {
        return;
    }
    isDraggingInitialState = false;
    pathCanvas.style.cursor = 'default';
    runRandomDemo({ preserveInitialState: true, dragTriggered: true });
}

function handleCanvasMove(event) {
    if (!currentData) {
        return;
    }
    if (isDraggingInitialState) {
        updateDraggedInitialState(event);
        updateCanvasCursor();
        return;
    }
    const nearest = findInitialDragTarget(event) || findNearestHover(event);
    updateCanvasCursor(nearest);
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
    if (isDraggingInitialState) {
        return;
    }
    activeHoverKey = null;
    hoverOverlay.classList.add('hidden');
    renderHoverDetails(null);
    updateCanvasCursor();
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

async function runRandomDemo(options = {}) {
    const { preserveInitialState = false, autoTriggered = false, dragTriggered = false } = options;
    if (solveInFlight) {
        pendingAutoReplanOptions = { preserveInitialState, autoTriggered, dragTriggered };
        return;
    }
    solveInFlight = true;
    setButtonLoading(true);
    setStatus(
        dragTriggered
            ? '正在基于拖拽后的起点位置重规划...'
            : autoTriggered
            ? preserveInitialState
                ? '参数变化后，正在基于当前状态自动重规划...'
                : '参数变化后，正在自动重新采样并求解...'
            : preserveInitialState
            ? '正在基于当前起点重规划 TEB-MPC...'
            : '正在随机生成起点并求解 TEB-MPC...',
        'loading',
    );
    try {
        const payload = collectParameterPayload();
        if (preserveInitialState && currentData?.initial_state) {
            payload.initial_state_override = currentData.initial_state;
        }
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
        updateCanvasCursor();
        setStatus(
            dragTriggered
                ? '已按拖拽后的起点位置完成重规划。'
                : autoTriggered
                ? preserveInitialState
                    ? '已根据新的参数自动重规划，当前初始状态保持不变。'
                    : '已根据新的采样参数自动重新采样并完成求解。'
                : preserveInitialState
                ? '已基于当前起点完成重规划。'
                : '随机初始化完成。可以在画布上悬停查看点信息，再次点击按钮可重新采样。',
            'success',
        );
    } catch (error) {
        setStatus(`失败: ${error.message}`, 'error');
    } finally {
        solveInFlight = false;
        setButtonLoading(false);
        if (pendingAutoReplanOptions) {
            const nextOptions = pendingAutoReplanOptions;
            pendingAutoReplanOptions = null;
            runRandomDemo(nextOptions);
        }
    }
}

randomizeBtn.addEventListener('click', runRandomDemo);
applyParamsBtn.addEventListener('click', () => runRandomDemo({ preserveInitialState: Boolean(currentData?.initial_state) }));
resetParamsBtn.addEventListener('click', resetParameterForm);
paramForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runRandomDemo({ preserveInitialState: Boolean(currentData?.initial_state) });
});
pathCanvas.addEventListener('mousemove', handleCanvasMove);
pathCanvas.addEventListener('mousedown', beginInitialStateDrag);
pathCanvas.addEventListener('mouseleave', clearCanvasHover);
window.addEventListener('mouseup', finishInitialStateDrag);
legendToggles.forEach((toggle) => {
    toggle.addEventListener('click', () => toggleLayer(toggle.dataset.layer));
});
axisButtons.forEach((button) => {
    button.addEventListener('click', () => setChartAxisMode(button.dataset.axisMode));
});
paramForm.querySelectorAll('[data-param-group][data-param-key]').forEach((input) => {
    input.addEventListener('input', () => scheduleAutoReplan(input));
});
updateLegendToggleStyles();
updateAxisButtonStyles();
initParameterTooltips();
runRandomDemo();