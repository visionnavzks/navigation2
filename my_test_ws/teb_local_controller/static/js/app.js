const randomizeBtn = document.getElementById('randomize-btn');
const paramForm = document.getElementById('param-form');
const statusText = document.getElementById('status-text');
const statusBadge = document.getElementById('status-badge');
const pathPlot = document.getElementById('path-plot');
const plotlyChart = document.getElementById('plotly-chart');
const hoverOverlay = document.getElementById('hover-overlay');
const legendToggles = Array.from(document.querySelectorAll('.legend-toggle'));
const vizTabButtons = Array.from(document.querySelectorAll('.viz-tab-btn'));
const axisButtons = Array.from(document.querySelectorAll('.axis-btn'));
const plannerModeInputs = Array.from(document.querySelectorAll('input[name="planner-mode"]'));
const modePanels = Array.from(document.querySelectorAll('[data-mode-panel]'));
const initialHeadingSlider = document.getElementById('initial-heading-slider');
const initialHeadingValue = document.getElementById('initial-heading-value');
const PATH_PLOT_CONFIG = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
};
const PATH_WHEEL_ZOOM_IN_FACTOR = 0.82;
const PATH_WHEEL_ZOOM_OUT_FACTOR = 1.22;
const PATH_MIN_ZOOM_SPAN = 0.05;
const MAIN_VIEW_OPACITY = 0.5;
const ENDPOINT_POSE_ARROW_OPACITY = 0.3;
const ENDPOINT_POSE_ARROW_HEAD_SCALE = 0.5;
const START_ARROW_COLOR = '#2563eb';
const END_ARROW_COLOR = '#dc2626';
const MID_ARROW_COLOR = '#7c3aed';
const LOCAL_GOAL_COLOR = '#f59e0b';

const statsEls = {
    optimizationStatus: document.getElementById('optimization-status'),
    optimizationDetail: document.getElementById('optimization-detail'),
    solveTime: document.getElementById('solve-time'),
    totalTime: document.getElementById('total-time'),
    avgDt: document.getElementById('avg-dt'),
    dtRange: document.getElementById('dt-range'),
    pointCount: document.getElementById('point-count'),
    pathLength: document.getElementById('path-length'),
    terminalError: document.getElementById('terminal-error'),
    totalCost: document.getElementById('total-cost'),
    costBreakdown: document.getElementById('cost-breakdown'),
    initialState: document.getElementById('initial-state'),
};

const PARAM_HELP_TEXTS = {
    horizon: 'MPC 局部优化窗口的最大节点数。0 表示不截断，使用从投影点到参考终点的全部剩余点；正数会限制求解规模。',
    ds: '参考轨迹按弧长离散时的采样间距，单位 m。值越小，参考点越密，跟踪更细，但优化变量更多、求解更慢。',
    cruise_speed: '参考轨迹的名义巡航速度，单位 m/s。它会影响参考速度曲线，也会影响按时间显示时参考曲线的横轴换算。',
    dt_ref: '名义时间步长，单位 s。用于初始化优化变量、生成参考时间轴、图表参考线；当 w_dt_ref > 0 时，也作为 dt 参考目标。实际用于优化的值会被夹到 dt_min/dt_max 内。',
    line_1_length: '第一段直线的长度，单位 m。改变它会直接拉长或缩短参考路径的开头。',
    arc_1_radius: '第一段圆弧的半径，单位 m。半径越小，转弯越急；半径越大，转弯越缓。',
    arc_1_angle: '第一段圆弧的转角，输入单位 rad。正值表示逆时针，负值表示顺时针。0.785 rad 大约等于 45 deg。',
    line_2_length: '第二段直线的长度，单位 m。通常用来调中段过渡的直线长度。',
    arc_2_radius: '第二段圆弧的半径，单位 m。会影响后半段转弯曲率大小。',
    arc_2_angle: '第二段圆弧的转角，输入单位 rad。负值表示顺时针回转，-0.524 rad 约等于 -30 deg。',
    line_3_length: '最后一段直线的长度，单位 m。决定终点前的收尾距离。',
    x: '目标点 x 坐标，单位 m。目标点模式下，优化器会生成一条从当前起点到该点的局部轨迹。',
    y: '目标点 y 坐标，单位 m。',
    theta: '目标点航向角，单位 rad。留给目标点模式作为终端朝向参考。',
    v: '目标点终端速度，单位 m/s。通常设为 0 表示到点停车。',
    sample_count: '目标点模式下的局部优化节点数。0 表示按距离和 goal_ds 自动计算；正数表示手动指定。',
    goal_ds: '目标点虚拟参考的采样间距，单位 m。',
    goal_cruise_speed: '目标点模式的名义巡航速度，单位 m/s。',
    goal_dt_ref: '目标点模式的名义时间步长，单位 s。用于初始化 dt 和图表参考线；w_dt_ref > 0 时会参与 dt 参考代价。实际用于优化的值会被夹到 dt_min/dt_max 内。',
    x_offset_range: '随机初始状态在 x 方向相对参考起点的采样范围，单位 m。实际采样区间是 [-range, +range]。',
    y_offset_range: '随机初始状态在 y 方向相对参考起点的采样范围，单位 m。值越大，初始横向偏差越大。',
    speed_min: '随机初始速度的最小值，单位 m/s。用于生成新的起始状态。',
    speed_max: '随机初始速度的最大值，单位 m/s。它不等于控制器速度上限，只影响随机起点采样。',
    accel_min: '随机初始加速度的最小值，单位 m/s²。',
    accel_max: '随机初始加速度的最大值，单位 m/s²。',
    kappa_min: '随机初始曲率的采样下界，单位 1/m。',
    kappa_max: '随机初始曲率的采样上界，单位 1/m。',
    dt_min: '优化允许的最小时间步长，单位 s。减小它会允许更细的时间伸缩，但可能让问题更难。',
    dt_max: '优化允许的最大时间步长，单位 s。增大它会允许轨迹在某些段上明显放慢。',
    max_speed: '优化状态中的速度上界，单位 m/s。它是硬约束，不是参考速度。',
    max_accel: '优化状态中的加速度绝对值上界，单位 m/s²。',
    max_lat_accel: '最大侧向加速度，单位 m/s²。这里约束的是 |v²·kappa|，会同时限制高速急转弯。',
    max_jerk: '控制量 jerk 的绝对值上界，单位 m/s³。越小表示速度变化更平滑，但机动性更弱。',
    max_kappa: '曲率绝对值上界，单位 1/m。越小表示允许的转弯半径更大。',
    max_dkappa: '曲率变化率绝对值上界，单位 1/(m*s)。越小表示转向变化更平滑。',
    w_lat_goal: '终点横向误差权重。误差按参考终点航向投影，越大越优先把末端拉回参考线。',
    w_lon_goal: '终点纵向误差权重。误差按参考终点航向投影，越大越优先让末端前后位置对齐。',
    w_theta_goal: '终点航向误差权重。越大，优化越优先让末端朝向对齐参考终点。',
    w_speed_goal: '终点速度误差权重。越大，末端速度越接近参考终点速度。',
    w_accel_goal: '终点加速度误差权重。越大，末端加速度越接近参考终点加速度。',
    w_time: '总时间代价权重。越大，优化越倾向缩短总时长。',
    w_dt_uniform: '相邻时间步长均匀性权重。越大，相邻 dt 之间的跳变越小。',
    w_dt_ref: 'dt 参考值跟踪权重。0 表示 dt_ref 只用于初值和显示；大于 0 时会惩罚 Σ(dt - dt_ref)²。',
    w_jerk: 'jerk 平滑权重。越大，速度变化更平顺，但响应更保守。',
    w_dkappa: '曲率变化率平滑权重。越大，转向变化更柔和。',
    ipopt_max_iter: 'IPOPT 最大迭代次数。遇到复杂参数组合时可以适当增大。',
    ipopt_tol: 'IPOPT 收敛容差。数值越小，解要求越严格，通常也会更慢。',
    ipopt_print_level: 'IPOPT 日志等级。0 表示几乎不打印，更高值会输出更多求解细节。',
};

const COST_ITEM_LABELS = {
    terminal_lat: '终点横向误差',
    terminal_lon: '终点纵向误差',
    terminal_theta: '终点航向误差',
    terminal_speed: '终点速度误差',
    terminal_accel: '终点加速度误差',
    dt_uniform: '相邻 dt 跳变',
    dt_ref: 'dt 参考误差',
    jerk: 'jerk 平滑',
    dkappa: 'dkappa 平滑',
    time: '总时间',
};

let currentScene = null;
let currentData = null;
let activeHoverKey = null;
let chartAxisMode = 'time';
let activeVizTab = 'main';
let autoReplanTimer = null;
let solveInFlight = false;
let pendingAutoReplanOptions = null;
let globalParamTooltip = null;
let activeParamTooltipAnchor = null;
let isDraggingInitialState = false;
let isDraggingGoalPoint = false;
let isPanningPath = false;
let pathPanStart = null;
const layerVisibility = {
    reference: true,
    stopReference: true,
    optimized: true,
    correspondence: true,
    initial: true,
    arrows: true,
};

function formatNumber(value, digits = 3) {
    return Number(value).toFixed(digits);
}

function formatMetric(value, digits = 3) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
        return '--';
    }
    const absValue = Math.abs(numericValue);
    if ((absValue > 0 && absValue < 0.001) || absValue >= 100000) {
        return numericValue.toExponential(2);
    }
    return numericValue.toFixed(digits);
}

function normalizeAngle(angle) {
    return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function radiansToDegrees(angle) {
    return angle * (180 / Math.PI);
}

function degreesToRadians(angle) {
    return angle * (Math.PI / 180);
}

function setStatus(message, tone = 'idle') {
    statusText.textContent = message;
    statusBadge.className = `status-badge ${tone}`;
    statusBadge.textContent = tone === 'loading' ? 'Solving' : tone === 'success' ? 'Ready' : tone === 'error' ? 'Error' : 'Idle';
}

function setOptimizationIndicator(succeeded, message) {
    statsEls.optimizationStatus.textContent = succeeded ? '成功' : '失败';
    statsEls.optimizationDetail.textContent = message;
    statsEls.optimizationStatus.dataset.tone = succeeded ? 'success' : 'error';
}

function setButtonLoading(isLoading) {
    randomizeBtn.disabled = isLoading;
    const mode = getPlannerMode();
    randomizeBtn.textContent = isLoading
        ? '求解中...'
        : mode === 'goal'
        ? '按目标点求解'
        : '随机初始化并求解';
}

function shouldPreserveInitialStateForInput(input) {
    return input.dataset.paramGroup !== 'sampling';
}

function formatSigned(value, digits = 3) {
    return `${value >= 0 ? '+' : ''}${formatNumber(value, digits)}`;
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
        goal: config?.goal || {},
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
        goal_config: {},
    };
    const targetMap = {
        controller: 'controller_params',
        reference: 'reference_config',
        sampling: 'sampling_config',
        goal: 'goal_config',
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

function getPlannerMode() {
    return plannerModeInputs.find((input) => input.checked)?.value || 'reference';
}

function updatePlannerModeUi() {
    const mode = getPlannerMode();
    modePanels.forEach((panel) => {
        panel.classList.toggle('hidden', panel.dataset.modePanel !== mode);
    });
    setButtonLoading(solveInFlight);
}

function updateInitialHeadingControls(initialState) {
    if (!initialHeadingSlider || !initialHeadingValue) {
        return;
    }

    if (!initialState) {
        initialHeadingSlider.disabled = true;
        initialHeadingSlider.value = '0';
        initialHeadingValue.textContent = '--';
        return;
    }

    const normalizedTheta = normalizeAngle(initialState.theta);
    initialHeadingSlider.disabled = false;
    initialHeadingSlider.value = String(Math.round(radiansToDegrees(normalizedTheta)));
    initialHeadingValue.textContent = `${formatNumber(normalizedTheta, 3)} rad / ${formatNumber(radiansToDegrees(normalizedTheta), 0)} deg`;
}

function scheduleInitialHeadingReplan() {
    if (autoReplanTimer !== null) {
        clearTimeout(autoReplanTimer);
    }
    setStatus('起点朝向已变更，正在等待基于当前状态自动重规划...', 'idle');
    autoReplanTimer = window.setTimeout(() => {
        autoReplanTimer = null;
        runPlanner({ preserveInitialState: true, autoTriggered: true });
    }, 220);
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
        const helpText = PARAM_HELP_TEXTS[labelSpan.textContent] || PARAM_HELP_TEXTS[input.dataset.paramKey];
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
        getPlannerMode() === 'goal'
            ? '目标点参数已变更，正在等待基于当前起点自动重规划...'
            : preserveInitialState
            ? '参数已变更，正在等待基于当前状态自动重规划...'
            : '采样参数已变更，正在等待自动重新采样并求解...',
        'idle',
    );
    autoReplanTimer = window.setTimeout(() => {
        autoReplanTimer = null;
        runPlanner({ preserveInitialState, autoTriggered: true });
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

function setVizTab(tabName) {
    activeVizTab = tabName;
    vizTabButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.vizTab === activeVizTab);
    });
    document.querySelectorAll('[data-viz-tab-panel]').forEach((panel) => {
        panel.classList.toggle('active', panel.dataset.vizTabPanel === activeVizTab);
    });

    if (activeVizTab === 'main') {
        requestAnimationFrame(() => {
            // 仅对已完成绘制(存在 _fullLayout)的图执行 resize;首屏调用时
            // runPlanner 仍在 await,两个图尚未 Plotly.react,跳过以免未处理的 Promise 拒绝。
            if (pathPlot && pathPlot._fullLayout) {
                Plotly.Plots.resize(pathPlot);
            }
            if (plotlyChart && plotlyChart._fullLayout) {
                Plotly.Plots.resize(plotlyChart);
            }
        });
    }
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

function buildHoverItems(data) {
    const items = [];
    const solverReference = data.reference;
    const displayReference = data.display_reference || data.reference;
    const stopReferenceActive = Boolean(data.reference_meta?.is_stopping_reference);
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
            items.push({
                key: `reference-${index}`,
                label: 'Reference',
                index,
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

    if (stopReferenceActive && layerVisibility.stopReference) {
        for (let index = 0; index < referenceCount; index += 1) {
            items.push({
                key: `stop-reference-${index}`,
                label: 'Stop Ref',
                index,
                x: solverReference.x[index],
                y: solverReference.y[index],
                s: solverReference.s[index],
                theta: solverReference.theta[index],
                v: solverReference.v[index],
                a: solverReference.a[index],
                kappa: solverReference.kappa[index],
            });
        }
    }

    if (layerVisibility.optimized) {
        for (let index = 0; index < solutionCount; index += 1) {
            const matchIndex = Math.min(index, referenceCount - 1);
            items.push({
                key: `solution-${index}`,
                label: 'Optimized',
                index,
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
        items.push({
            key: 'initial-0',
            label: 'Initial',
            index: 0,
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
    if (layerVisibility.optimized && referenceCount > 0) {
        const last = referenceCount - 1;
        items.push({
            key: 'local-goal-0',
            label: 'Local Goal',
            index: last,
            x: solverReference.x[last],
            y: solverReference.y[last],
            s: solverReference.s[last],
            theta: solverReference.theta[last],
            v: solverReference.v[last],
            a: solverReference.a[last],
            kappa: solverReference.kappa[last],
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
    const rect = pathPlot.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const maxX = rect.width - hoverOverlay.offsetWidth - 10;
    const maxY = rect.height - hoverOverlay.offsetHeight - 10;
    hoverOverlay.style.left = `${Math.max(10, Math.min(localX + 12, maxX))}px`;
    hoverOverlay.style.top = `${Math.max(10, Math.min(localY + 12, maxY))}px`;
}

function buildSegmentedLineCoords(referencePoints, solutionPoints) {
    const pairCount = Math.min(referencePoints.length, solutionPoints.length);
    const xs = [];
    const ys = [];
    for (let index = 0; index < pairCount; index += 1) {
        xs.push(referencePoints[index][0], solutionPoints[index][0], null);
        ys.push(referencePoints[index][1], solutionPoints[index][1], null);
    }
    return { xs, ys };
}

function buildEndpointMarkerColors(points, defaultColor) {
    return points.map((_point, index) => {
        if (index === 0) {
            return START_ARROW_COLOR;
        }
        if (index === points.length - 1) {
            return END_ARROW_COLOR;
        }
        return defaultColor;
    });
}

function buildPathArrowAnnotation(x, y, theta, color, opacity, arrowLength, lineWidth = 1.4, arrowSize = 1.0, arrowHead = 3) {
    return {
        x: x + arrowLength * Math.cos(theta),
        y: y + arrowLength * Math.sin(theta),
        ax: x,
        ay: y,
        axref: 'x',
        ayref: 'y',
        text: '',
        showarrow: true,
        arrowhead: arrowHead,
        arrowsize: arrowSize,
        arrowwidth: lineWidth,
        arrowcolor: color,
        opacity,
    };
}

function buildHeadingAnnotations(points, headings, step, arrowLength, lineWidth = 1.4) {
    const annotations = [];
    for (let index = 0; index < points.length; index += step) {
        if (index === 0 || index === points.length - 1) {
            continue;
        }
        annotations.push(
            buildPathArrowAnnotation(
                points[index][0],
                points[index][1],
                headings[index],
                MID_ARROW_COLOR,
                0.3,
                arrowLength * 1.45,
                Math.max(lineWidth, 3.0),
                1.55,
            ),
        );
    }
    return annotations;
}

function buildEndpointHeadingAnnotations(points, headings, arrowLength, lineWidth = 5.2, options = {}) {
    const annotations = [];
    if (!points.length) {
        return annotations;
    }
    const endpointArrowSize = 1.15 * ENDPOINT_POSE_ARROW_HEAD_SCALE;
    if (!options.skipStart) {
        annotations.push(
            buildPathArrowAnnotation(
                points[0][0],
                points[0][1],
                headings[0],
                START_ARROW_COLOR,
                ENDPOINT_POSE_ARROW_OPACITY,
                arrowLength,
                lineWidth,
                endpointArrowSize,
            ),
        );
    }
    if (points.length > 1 && !options.skipEnd) {
        const last = points.length - 1;
        annotations.push(
            buildPathArrowAnnotation(
                points[last][0],
                points[last][1],
                headings[last],
                END_ARROW_COLOR,
                ENDPOINT_POSE_ARROW_OPACITY,
                arrowLength,
                lineWidth,
                endpointArrowSize,
            ),
        );
    }
    return annotations;
}

function bindPathPlotInteractions() {
    const nextTarget = pathPlot;
    if (pathPlot.__interactionTarget !== nextTarget) {
        if (pathPlot.__interactionTarget) {
            pathPlot.__interactionTarget.removeEventListener('mousemove', handleCanvasMove, true);
            pathPlot.__interactionTarget.removeEventListener('mousedown', handlePathMouseDown, true);
            pathPlot.__interactionTarget.removeEventListener('mouseleave', clearCanvasHover, true);
            pathPlot.__interactionTarget.removeEventListener('wheel', handlePathWheelZoom, true);
            pathPlot.__interactionTarget.removeEventListener('contextmenu', preventPathContextMenu, true);
        }
        nextTarget.addEventListener('mousemove', handleCanvasMove, true);
        nextTarget.addEventListener('mousedown', handlePathMouseDown, true);
        nextTarget.addEventListener('mouseleave', clearCanvasHover, true);
        nextTarget.addEventListener('wheel', handlePathWheelZoom, { capture: true, passive: false });
        nextTarget.addEventListener('contextmenu', preventPathContextMenu, true);
        pathPlot.__interactionTarget = nextTarget;
    }
}

function updatePathHighlight(key) {
    if (!currentScene || currentScene.highlightTraceIndex == null) {
        return;
    }
    const item = key ? currentScene.itemMap.get(key) : null;
    Plotly.restyle(
        pathPlot,
        {
            x: [item ? [item.x] : []],
            y: [item ? [item.y] : []],
            visible: item ? true : false,
        },
        [currentScene.highlightTraceIndex],
    );
}

function renderPathView(data, activeKey = null, options = {}) {
    const { reference, solution, initial_state: initialState } = data;
    const preservedPathViewRanges = options.pathViewRanges || (options.preserveView ? getCurrentPathViewRanges() : null);
    const displayReference = data.display_reference || data.reference;
    const stopReferenceActive = Boolean(data.reference_meta?.is_stopping_reference);
    const referencePoints = displayReference.x.map((x, index) => [x, displayReference.y[index]]);
    const solverReferencePoints = reference.x.map((x, index) => [x, reference.y[index]]);
    const stopReferencePoints = stopReferenceActive ? solverReferencePoints : [];
    const solutionPoints = solution.x.map((x, index) => [x, solution.y[index]]);
    const allPoints = [...referencePoints, ...stopReferencePoints, ...solutionPoints, [initialState.x, initialState.y]];
    const xs = allPoints.map((point) => point[0]);
    const ys = allPoints.map((point) => point[1]);
    const spanX = Math.max(Math.max(...xs) - Math.min(...xs), 1.0);
    const spanY = Math.max(Math.max(...ys) - Math.min(...ys), 1.0);
    const paddingX = spanX * 0.08;
    const paddingY = spanY * 0.08;
    const arrowLength = Math.max(spanX, spanY) * 0.035;
    const hoverItems = buildHoverItems(data);
    const itemMap = new Map(hoverItems.map((item) => [item.key, item]));
    const annotations = [];
    const traces = [];

    if (layerVisibility.correspondence && layerVisibility.optimized) {
        const { xs: correspondenceX, ys: correspondenceY } = buildSegmentedLineCoords(solverReferencePoints, solutionPoints);
        traces.push({
            x: correspondenceX,
            y: correspondenceY,
            mode: 'lines',
            name: '对应关系',
            line: { color: 'rgba(141, 133, 120, 0.5)', width: 1, dash: 'dot' },
            opacity: MAIN_VIEW_OPACITY,
            hoverinfo: 'skip',
            showlegend: false,
        });
    }

    if (layerVisibility.reference) {
        const referenceKeys = referencePoints.map((_point, index) => `reference-${index}`);
        traces.push({
            x: referencePoints.map((point) => point[0]),
            y: referencePoints.map((point) => point[1]),
            mode: 'lines+markers',
            name: '参考路径',
            customdata: referenceKeys,
            hovertemplate: '<extra></extra>',
            line: { color: 'rgba(15, 118, 110, 0.5)', width: 3.6, dash: 'dash' },
            marker: { color: buildEndpointMarkerColors(referencePoints, '#0f766e'), size: 8, symbol: 'x', opacity: MAIN_VIEW_OPACITY },
            showlegend: false,
        });
        if (layerVisibility.arrows) {
            annotations.push(...buildHeadingAnnotations(referencePoints, displayReference.theta, 5, arrowLength));
        }
        annotations.push(...buildEndpointHeadingAnnotations(
            referencePoints,
            displayReference.theta,
            arrowLength,
            5.2,
            { skipStart: layerVisibility.initial },
        ));
    }

    if (stopReferenceActive && layerVisibility.stopReference) {
        const stopReferenceKeys = stopReferencePoints.map((_point, index) => `stop-reference-${index}`);
        traces.push({
            x: stopReferencePoints.map((point) => point[0]),
            y: stopReferencePoints.map((point) => point[1]),
            mode: 'lines+markers',
            name: '停车参考',
            customdata: stopReferenceKeys,
            hovertemplate: '<extra></extra>',
            line: { color: 'rgba(217, 119, 6, 0.5)', width: 4.2 },
            marker: {
                color: buildEndpointMarkerColors(stopReferencePoints, '#d97706'),
                size: 8,
                symbol: 'diamond',
                opacity: MAIN_VIEW_OPACITY,
                line: { color: buildEndpointMarkerColors(stopReferencePoints, '#d97706'), width: 1.2 },
            },
            showlegend: false,
        });
        if (layerVisibility.arrows) {
            annotations.push(...buildHeadingAnnotations(stopReferencePoints, reference.theta, 4, arrowLength * 0.95, 1.55));
        }
        annotations.push(...buildEndpointHeadingAnnotations(
            stopReferencePoints,
            reference.theta,
            arrowLength * 0.95,
            5.2,
            { skipStart: layerVisibility.initial },
        ));
    }

    if (layerVisibility.optimized) {
        const optimizedKeys = solutionPoints.map((_point, index) => `solution-${index}`);
        const localGoalIndex = solverReferencePoints.length - 1;
        const localGoalPoint = solverReferencePoints[localGoalIndex];
        traces.push({
            x: solutionPoints.map((point) => point[0]),
            y: solutionPoints.map((point) => point[1]),
            mode: 'lines+markers',
            name: '优化路径',
            customdata: optimizedKeys,
            hovertemplate: '<extra></extra>',
            line: { color: 'rgba(202, 90, 52, 0.5)', width: 3.8 },
            marker: {
                color: buildEndpointMarkerColors(solutionPoints, '#ca5a34'),
                size: 9,
                symbol: 'circle-open',
                opacity: MAIN_VIEW_OPACITY,
                line: { color: buildEndpointMarkerColors(solutionPoints, '#ca5a34'), width: 1.8 },
            },
            showlegend: false,
        });
        traces.push({
            x: [localGoalPoint[0]],
            y: [localGoalPoint[1]],
            mode: 'markers',
            name: '局部终点',
            customdata: ['local-goal-0'],
            hovertemplate: '<extra></extra>',
            marker: {
                color: LOCAL_GOAL_COLOR,
                size: 18,
                symbol: 'star',
                opacity: 0.85,
                line: { color: '#92400e', width: 1.6 },
            },
            showlegend: false,
        });
        if (layerVisibility.arrows) {
            annotations.push(...buildHeadingAnnotations(solutionPoints, solution.theta, 5, arrowLength));
        }
        annotations.push(...buildEndpointHeadingAnnotations(
            solutionPoints,
            solution.theta,
            arrowLength,
            5.2,
            { skipStart: layerVisibility.initial },
        ));
    }

    if (layerVisibility.initial) {
        traces.push({
            x: [initialState.x],
            y: [initialState.y],
            mode: 'markers',
            name: '随机起点',
            customdata: ['initial-0'],
            hovertemplate: '<extra></extra>',
            marker: {
                color: START_ARROW_COLOR,
                size: 14,
                opacity: MAIN_VIEW_OPACITY,
                line: { color: START_ARROW_COLOR, width: 1.8 },
            },
            showlegend: false,
        });
        annotations.push(buildPathArrowAnnotation(
            initialState.x,
            initialState.y,
            initialState.theta,
            START_ARROW_COLOR,
            ENDPOINT_POSE_ARROW_OPACITY,
            arrowLength * 1.15,
            5.2,
            1.15 * ENDPOINT_POSE_ARROW_HEAD_SCALE,
        ));
    }

    const highlightTraceIndex = traces.length;
    traces.push({
        x: [],
        y: [],
        mode: 'markers',
        hoverinfo: 'skip',
        visible: false,
        marker: {
            size: 22,
            color: 'rgba(217, 119, 6, 0.18)',
            line: { color: '#d97706', width: 2 },
        },
        showlegend: false,
    });

    const axisStyle = {
        showgrid: true,
        gridcolor: 'rgba(26, 34, 48, 0.08)',
        zeroline: false,
        linecolor: 'rgba(26, 34, 48, 0.14)',
        tickfont: { family: 'Space Grotesk, Noto Sans SC, sans-serif', size: 11 },
        fixedrange: true,
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 32, r: 24, t: 18, b: 30 },
        font: { family: 'Space Grotesk, Noto Sans SC, sans-serif', color: '#1a2230' },
        hovermode: 'closest',
        showlegend: false,
        dragmode: false,
        annotations,
        xaxis: {
            ...axisStyle,
            range: preservedPathViewRanges?.xRange || [Math.min(...xs) - paddingX, Math.max(...xs) + paddingX],
        },
        yaxis: {
            ...axisStyle,
            range: preservedPathViewRanges?.yRange || [Math.min(...ys) - paddingY, Math.max(...ys) + paddingY],
            scaleanchor: 'x',
            scaleratio: 1,
        },
    };

    Plotly.react(pathPlot, traces, layout, PATH_PLOT_CONFIG);
    currentScene = { itemMap, hoverItems, highlightTraceIndex };
    bindPathPlotInteractions();
    updatePathHighlight(activeKey);
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
    const dtRef = data.solution.costs?.dt_ref_used ?? data.reference.dt_ref;

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

    const optimization = data.optimization || {
        succeeded: true,
        message: data.solution.solver_status || 'Optimization succeeded',
    };

    setOptimizationIndicator(Boolean(optimization.succeeded), optimization.message || 'Optimization succeeded');

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

    updateInitialHeadingControls(initialState);
    renderCostBreakdown(data.solution);
}

function renderCostBreakdown(solution) {
    if (!statsEls.costBreakdown) {
        return;
    }

    const costItems = Array.isArray(solution.cost_items) && solution.cost_items.length > 0
        ? solution.cost_items
        : [
            { label: '终点项', residual: null, unit: '', weight: null, cost: solution.costs?.terminal },
            { label: '控制项', residual: null, unit: '', weight: null, cost: solution.costs?.control },
            { label: '时间项', residual: null, unit: '', weight: null, cost: solution.costs?.time },
        ];
    const totalCost = Number(solution.costs?.total);

    statsEls.costBreakdown.innerHTML = `
        <div class="cost-row cost-head">
            <span>项目</span>
            <span>残差</span>
            <span>权重</span>
            <span>代价</span>
            <span>占比</span>
        </div>
        ${costItems.map((item) => {
            const label = COST_ITEM_LABELS[item.key] || item.label || item.key || '--';
            const residual = item.residual === null || item.residual === undefined
                ? '--'
                : `${formatMetric(item.residual, 3)}${item.unit ? ` ${item.unit}` : ''}`;
            const weight = item.weight === null || item.weight === undefined ? '--' : formatMetric(item.weight, 2);
            const itemCost = Number(item.cost);
            const cost = Number.isFinite(itemCost) ? formatMetric(itemCost, 3) : '--';
            const ratioValue = Number.isFinite(itemCost) && Number.isFinite(totalCost) && Math.abs(totalCost) > 1e-12
                ? Math.max(0, (itemCost / totalCost) * 100)
                : null;
            const ratio = ratioValue === null ? '--' : `${formatMetric(ratioValue, 1)}%`;
            const ratioWidth = ratioValue === null ? 0 : Math.min(ratioValue, 100);
            return `
                <div class="cost-row">
                    <span>${label}</span>
                    <strong>${residual}</strong>
                    <strong>${weight}</strong>
                    <strong>${cost}</strong>
                    <div class="cost-share" aria-label="${ratio}">
                        <span class="cost-share-track"><span class="cost-share-fill" style="width: ${ratioWidth}%;"></span></span>
                        <strong>${ratio}</strong>
                    </div>
                </div>
            `;
        }).join('')}
    `;
}

function findNearestHover(event) {
    if (!currentScene) {
        return null;
    }
    const cursor = getCanvasCursorPosition(event);
    if (!cursor) {
        return null;
    }
    let best = null;
    let bestDistance = 18;
    currentScene.hoverItems.forEach((item) => {
        const pixelPosition = projectPlotItemToPixels(item);
        if (!pixelPosition) {
            return;
        }
        const distance = Math.hypot(pixelPosition.px - cursor.clientX, pixelPosition.py - cursor.clientY);
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
    const cursor = getCanvasCursorPosition(event);
    if (!cursor) {
        return null;
    }
    const initialItem = currentScene.hoverItems.find((item) => item.key === 'initial-0');
    if (!initialItem) {
        return null;
    }
    const pixelPosition = projectPlotItemToPixels(initialItem);
    if (!pixelPosition) {
        return null;
    }
    const distance = Math.hypot(pixelPosition.px - cursor.clientX, pixelPosition.py - cursor.clientY);
    return distance <= 18 ? initialItem : null;
}

function findGoalDragTarget(event) {
    if (getPlannerMode() !== 'goal' || !currentScene || !layerVisibility.optimized) {
        return null;
    }
    const cursor = getCanvasCursorPosition(event);
    if (!cursor) {
        return null;
    }
    const goalItem = currentScene.hoverItems.find((item) => item.key === 'local-goal-0');
    if (!goalItem) {
        return null;
    }
    const pixelPosition = projectPlotItemToPixels(goalItem);
    if (!pixelPosition) {
        return null;
    }
    const distance = Math.hypot(pixelPosition.px - cursor.clientX, pixelPosition.py - cursor.clientY);
    return distance <= 22 ? goalItem : null;
}

function getCanvasCursorPosition(event) {
    const axes = getPathPlotAxes();
    if (!axes) {
        return null;
    }
    const plotX = event.clientX - axes.rect.left - axes.xaxis._offset;
    const plotY = event.clientY - axes.rect.top - axes.yaxis._offset;
    return {
        clientX: event.clientX,
        clientY: event.clientY,
        cursorX: axes.xaxis.p2l(plotX),
        cursorY: axes.yaxis.p2l(plotY),
    };
}

function getPathPlotAxes() {
    const fullLayout = pathPlot?._fullLayout;
    if (!fullLayout?.xaxis || !fullLayout?.yaxis) {
        return null;
    }
    return {
        rect: pathPlot.getBoundingClientRect(),
        xaxis: fullLayout.xaxis,
        yaxis: fullLayout.yaxis,
    };
}

function projectPlotItemToPixels(item) {
    const axes = getPathPlotAxes();
    if (!axes) {
        return null;
    }
    return {
        px: axes.rect.left + axes.xaxis._offset + axes.xaxis.l2p(item.x),
        py: axes.rect.top + axes.yaxis._offset + axes.yaxis.l2p(item.y),
    };
}

function isFiniteRange(range) {
    return Array.isArray(range)
        && range.length === 2
        && Number.isFinite(range[0])
        && Number.isFinite(range[1])
        && range[0] !== range[1];
}

function getCurrentPathViewRanges() {
    const axes = getPathPlotAxes();
    if (!axes || !isFiniteRange(axes.xaxis.range) || !isFiniteRange(axes.yaxis.range)) {
        return null;
    }

    return {
        xRange: [...axes.xaxis.range],
        yRange: [...axes.yaxis.range],
    };
}

function zoomRangeAround(range, anchor, factor) {
    const nextRange = [
        anchor - (anchor - range[0]) * factor,
        anchor + (range[1] - anchor) * factor,
    ];
    const span = Math.abs(nextRange[1] - nextRange[0]);
    if (span >= PATH_MIN_ZOOM_SPAN) {
        return nextRange;
    }

    const halfSpan = PATH_MIN_ZOOM_SPAN * 0.5;
    return [anchor - halfSpan, anchor + halfSpan];
}

function getPathPlotPixelPosition(event, axes) {
    return {
        plotX: event.clientX - axes.rect.left - axes.xaxis._offset,
        plotY: event.clientY - axes.rect.top - axes.yaxis._offset,
    };
}

function isInsidePathPlotArea(plotPosition, axes) {
    return plotPosition.plotX >= 0
        && plotPosition.plotX <= axes.xaxis._length
        && plotPosition.plotY >= 0
        && plotPosition.plotY <= axes.yaxis._length;
}

function handlePathWheelZoom(event) {
    if (isDraggingInitialState || isDraggingGoalPoint) {
        return;
    }

    const axes = getPathPlotAxes();
    const cursor = getCanvasCursorPosition(event);
    if (!axes || !cursor || !isFiniteRange(axes.xaxis.range) || !isFiniteRange(axes.yaxis.range)) {
        return;
    }

    if (!isInsidePathPlotArea(getPathPlotPixelPosition(event, axes), axes)) {
        return;
    }

    const zoomFactor = event.deltaY < 0 ? PATH_WHEEL_ZOOM_IN_FACTOR : PATH_WHEEL_ZOOM_OUT_FACTOR;
    event.preventDefault();
    Plotly.relayout(pathPlot, {
        'xaxis.range': zoomRangeAround(axes.xaxis.range, cursor.cursorX, zoomFactor),
        'yaxis.range': zoomRangeAround(axes.yaxis.range, cursor.cursorY, zoomFactor),
    });
}

function handlePathMouseDown(event) {
    if (event.button === 2) {
        beginPathPan(event);
        return;
    }

    beginInitialStateDrag(event);
    if (!isDraggingInitialState) {
        beginGoalPointDrag(event);
    }
}

function preventPathContextMenu(event) {
    if (pathPlot.contains(event.target)) {
        event.preventDefault();
    }
}

function beginPathPan(event) {
    if (isDraggingInitialState || isDraggingGoalPoint) {
        return;
    }

    const axes = getPathPlotAxes();
    if (!axes || !isFiniteRange(axes.xaxis.range) || !isFiniteRange(axes.yaxis.range)) {
        return;
    }

    if (!isInsidePathPlotArea(getPathPlotPixelPosition(event, axes), axes)) {
        return;
    }

    isPanningPath = true;
    pathPanStart = {
        clientX: event.clientX,
        clientY: event.clientY,
        xRange: [...axes.xaxis.range],
        yRange: [...axes.yaxis.range],
        xLength: axes.xaxis._length,
        yLength: axes.yaxis._length,
    };
    activeHoverKey = null;
    renderHoverDetails(null);
    updatePathHighlight(null);
    updateCanvasCursor();
    event.preventDefault();
}

function updatePathPan(event) {
    if (!isPanningPath || !pathPanStart) {
        return;
    }

    const dx = event.clientX - pathPanStart.clientX;
    const dy = event.clientY - pathPanStart.clientY;
    const xSpan = pathPanStart.xRange[1] - pathPanStart.xRange[0];
    const ySpan = pathPanStart.yRange[1] - pathPanStart.yRange[0];
    const xDelta = dx * xSpan / Math.max(pathPanStart.xLength, 1);
    const yDelta = -dy * ySpan / Math.max(pathPanStart.yLength, 1);

    Plotly.relayout(pathPlot, {
        'xaxis.range': [
            pathPanStart.xRange[0] - xDelta,
            pathPanStart.xRange[1] - xDelta,
        ],
        'yaxis.range': [
            pathPanStart.yRange[0] - yDelta,
            pathPanStart.yRange[1] - yDelta,
        ],
    });
    event.preventDefault();
}

function finishPathPan() {
    if (!isPanningPath) {
        return;
    }

    isPanningPath = false;
    pathPanStart = null;
    updateCanvasCursor();
}

function updateCanvasCursor(nearest = null) {
    if (isPanningPath) {
        pathPlot.style.cursor = 'grabbing';
        return;
    }
    if (isDraggingInitialState || isDraggingGoalPoint) {
        pathPlot.style.cursor = 'grabbing';
        return;
    }
    if (nearest?.key === 'initial-0' || nearest?.key === 'local-goal-0') {
        pathPlot.style.cursor = 'grab';
        return;
    }
    pathPlot.style.cursor = 'default';
}

function updateDraggedInitialState(event) {
    if (!isDraggingInitialState || !currentData?.initial_state) {
        return null;
    }

    const cursor = getCanvasCursorPosition(event);
    if (!cursor) {
        return null;
    }
    currentData.initial_state = {
        ...currentData.initial_state,
        x: cursor.cursorX,
        y: cursor.cursorY,
    };
    activeHoverKey = 'initial-0';
    renderPathView(currentData, activeHoverKey, { preserveView: true });

    const draggedItem = currentScene?.hoverItems.find((item) => item.key === 'initial-0') || null;
    if (draggedItem) {
        renderHoverDetails(draggedItem);
        positionHoverOverlay(event);
    }
    return draggedItem;
}

function beginInitialStateDrag(event) {
    if (event.button !== 0) {
        return;
    }
    if (isPanningPath) {
        return;
    }
    if (!currentData || !layerVisibility.initial) {
        return;
    }
    const nearest = findInitialDragTarget(event);
    if (nearest?.key !== 'initial-0') {
        return;
    }

    isDraggingInitialState = true;
    activeHoverKey = 'initial-0';
    pathPlot.style.cursor = 'grabbing';
    renderPathView(currentData, activeHoverKey, { preserveView: true });
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
    pathPlot.style.cursor = 'default';
    runPlanner({ preserveInitialState: true, dragTriggered: true });
}

function goalPositionInputs() {
    return {
        x: paramForm.querySelector('[data-param-group="goal"][data-param-key="x"]'),
        y: paramForm.querySelector('[data-param-group="goal"][data-param-key="y"]'),
    };
}

function updateGoalPositionInputs(x, y) {
    const inputs = goalPositionInputs();
    if (inputs.x) {
        inputs.x.value = formatNumber(x, 3);
    }
    if (inputs.y) {
        inputs.y.value = formatNumber(y, 3);
    }
}

function updateGoalPointInScene(x, y) {
    if (!currentData?.reference?.x?.length) {
        return;
    }
    const referenceLast = currentData.reference.x.length - 1;
    currentData.reference.x[referenceLast] = x;
    currentData.reference.y[referenceLast] = y;

    if (currentData.display_reference?.x?.length) {
        const displayLast = currentData.display_reference.x.length - 1;
        currentData.display_reference.x[displayLast] = x;
        currentData.display_reference.y[displayLast] = y;
    }
    if (currentData.config?.goal) {
        currentData.config.goal.x = x;
        currentData.config.goal.y = y;
    }
    if (currentData.reference_meta?.goal) {
        currentData.reference_meta.goal.x = x;
        currentData.reference_meta.goal.y = y;
    }
}

function updateDraggedGoalPoint(event) {
    if (!isDraggingGoalPoint || !currentData) {
        return null;
    }

    const cursor = getCanvasCursorPosition(event);
    if (!cursor) {
        return null;
    }
    updateGoalPositionInputs(cursor.cursorX, cursor.cursorY);
    updateGoalPointInScene(cursor.cursorX, cursor.cursorY);
    activeHoverKey = 'local-goal-0';
    renderPathView(currentData, activeHoverKey, { preserveView: true });
    renderStats(currentData);

    const draggedItem = currentScene?.hoverItems.find((item) => item.key === 'local-goal-0') || null;
    if (draggedItem) {
        renderHoverDetails(draggedItem);
        positionHoverOverlay(event);
    }
    return draggedItem;
}

function beginGoalPointDrag(event) {
    if (event.button !== 0 || getPlannerMode() !== 'goal') {
        return;
    }
    if (isPanningPath || isDraggingInitialState) {
        return;
    }
    if (!currentData || !layerVisibility.optimized) {
        return;
    }
    const nearest = findGoalDragTarget(event);
    if (nearest?.key !== 'local-goal-0') {
        return;
    }

    isDraggingGoalPoint = true;
    activeHoverKey = 'local-goal-0';
    pathPlot.style.cursor = 'grabbing';
    renderPathView(currentData, activeHoverKey, { preserveView: true });
    renderHoverDetails(nearest);
    positionHoverOverlay(event);
    setStatus('拖动目标点中，松开鼠标后会基于新的目标点重规划。', 'idle');
    event.preventDefault();
}

function finishGoalPointDrag() {
    if (!isDraggingGoalPoint) {
        return;
    }
    isDraggingGoalPoint = false;
    pathPlot.style.cursor = 'default';
    runPlanner({ preserveInitialState: true, goalDragTriggered: true });
}

function handleWindowMouseMove(event) {
    if (isPanningPath) {
        updatePathPan(event);
        return;
    }
    if (isDraggingInitialState || isDraggingGoalPoint) {
        handleCanvasMove(event);
    }
}

function handleCanvasMove(event) {
    if (isPanningPath) {
        updatePathPan(event);
        return;
    }
    if (isDraggingInitialState) {
        updateDraggedInitialState(event);
        updateCanvasCursor();
        return;
    }
    if (isDraggingGoalPoint) {
        updateDraggedGoalPoint(event);
        updateCanvasCursor();
        return;
    }
    const nearest = findInitialDragTarget(event) || findGoalDragTarget(event) || findNearestHover(event);
    updateCanvasCursor(nearest);
    if (!nearest) {
        if (activeHoverKey !== null) {
            activeHoverKey = null;
            updatePathHighlight(null);
        }
        renderHoverDetails(null);
        return;
    }

    positionHoverOverlay(event);
    if (nearest.key !== activeHoverKey) {
        activeHoverKey = nearest.key;
        updatePathHighlight(nearest.key);
    }
    renderHoverDetails(nearest);
}

function handleInitialHeadingInput() {
    if (!currentData?.initial_state || !initialHeadingSlider) {
        return;
    }

    const sliderDegrees = Number.parseFloat(initialHeadingSlider.value);
    if (Number.isNaN(sliderDegrees)) {
        return;
    }

    currentData.initial_state = {
        ...currentData.initial_state,
        theta: normalizeAngle(degreesToRadians(sliderDegrees)),
    };
    activeHoverKey = 'initial-0';
    renderPathView(currentData, activeHoverKey, { preserveView: true });
    renderStats(currentData);

    const currentInitialItem = currentScene?.itemMap.get('initial-0') || null;
    if (currentInitialItem && activeHoverKey === 'initial-0') {
        renderHoverDetails(currentInitialItem);
    }

    scheduleInitialHeadingReplan();
}

function clearCanvasHover() {
    if (isDraggingInitialState || isDraggingGoalPoint || isPanningPath) {
        return;
    }
    activeHoverKey = null;
    hoverOverlay.classList.add('hidden');
    renderHoverDetails(null);
    updatePathHighlight(null);
    updateCanvasCursor();
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

async function runPlanner(options = {}) {
    const {
        preserveInitialState = false,
        autoTriggered = false,
        dragTriggered = false,
        goalDragTriggered = false,
        pathViewRanges = null,
    } = options;
    const plannerMode = getPlannerMode();
    const preservedPathViewRanges = pathViewRanges || (dragTriggered || goalDragTriggered ? getCurrentPathViewRanges() : null);
    if (solveInFlight) {
        pendingAutoReplanOptions = {
            preserveInitialState,
            autoTriggered,
            dragTriggered,
            goalDragTriggered,
            pathViewRanges: preservedPathViewRanges,
        };
        return;
    }
    solveInFlight = true;
    setButtonLoading(true);
    let solvingMessage = '正在随机生成起点并求解 TEB-MPC...';
    if (dragTriggered) {
        solvingMessage = '正在基于拖拽后的起点位置重规划...';
    } else if (goalDragTriggered) {
        solvingMessage = '正在基于拖拽后的目标点重规划...';
    } else if (autoTriggered && plannerMode === 'goal') {
        solvingMessage = '目标点参数变化后，正在基于当前起点自动重规划...';
    } else if (autoTriggered && preserveInitialState) {
        solvingMessage = '参数变化后，正在基于当前状态自动重规划...';
    } else if (autoTriggered) {
        solvingMessage = '参数变化后，正在自动重新采样并求解...';
    } else if (plannerMode === 'goal') {
        solvingMessage = '正在求解当前起点到目标点的局部 TEB-MPC...';
    } else if (preserveInitialState) {
        solvingMessage = '正在基于当前起点重规划 TEB-MPC...';
    }
    setStatus(solvingMessage, 'loading');
    try {
        const payload = collectParameterPayload();
        if ((preserveInitialState || plannerMode === 'goal') && currentData?.initial_state) {
            payload.initial_state_override = currentData.initial_state;
        }
        const response = await fetch(plannerMode === 'goal' ? '/api/goal_demo' : '/api/random_demo', {
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
        renderPathView(data, null, { pathViewRanges: preservedPathViewRanges });
        renderPlotlyCharts(data);
        renderStats(data);
        applyConfigToForm(data.config);
        renderHoverDetails(null);
        updateCanvasCursor();
        setStatus(
            dragTriggered
                ? '已按拖拽后的起点位置完成重规划。'
                : goalDragTriggered
                ? '已按拖拽后的目标点完成重规划。'
                : autoTriggered
                ? plannerMode === 'goal'
                    ? '已根据目标点参数自动重规划。'
                    : preserveInitialState
                    ? '已根据新的参数自动重规划，当前初始状态保持不变。'
                    : '已根据新的采样参数自动重新采样并完成求解。'
                : plannerMode === 'goal'
                ? '目标点局部规划完成。可拖动起点或修改目标点继续重规划。'
                : preserveInitialState
                ? '已基于当前起点完成重规划。'
                : '随机初始化完成。可以在画布上悬停查看点信息，再次点击按钮可重新采样。',
            'success',
        );
    } catch (error) {
        setOptimizationIndicator(false, error.message);
        setStatus(`失败: ${error.message}`, 'error');
    } finally {
        solveInFlight = false;
        setButtonLoading(false);
        if (pendingAutoReplanOptions) {
            const nextOptions = pendingAutoReplanOptions;
            pendingAutoReplanOptions = null;
            runPlanner(nextOptions);
        }
    }
}

randomizeBtn.addEventListener('click', runPlanner);
paramForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runPlanner({ preserveInitialState: Boolean(currentData?.initial_state) });
});
window.addEventListener('mousemove', handleWindowMouseMove);
window.addEventListener('mouseup', finishInitialStateDrag);
window.addEventListener('mouseup', finishGoalPointDrag);
window.addEventListener('mouseup', finishPathPan);
legendToggles.forEach((toggle) => {
    toggle.addEventListener('click', () => toggleLayer(toggle.dataset.layer));
});
axisButtons.forEach((button) => {
    button.addEventListener('click', () => setChartAxisMode(button.dataset.axisMode));
});
vizTabButtons.forEach((button) => {
    button.addEventListener('click', () => setVizTab(button.dataset.vizTab));
});
if (initialHeadingSlider) {
    initialHeadingSlider.addEventListener('input', handleInitialHeadingInput);
}
plannerModeInputs.forEach((input) => {
    input.addEventListener('change', () => {
        updatePlannerModeUi();
        runPlanner({ preserveInitialState: Boolean(currentData?.initial_state) });
    });
});
paramForm.querySelectorAll('[data-param-group][data-param-key]').forEach((input) => {
    input.addEventListener('input', () => scheduleAutoReplan(input));
});
updateLegendToggleStyles();
updateAxisButtonStyles();
updatePlannerModeUi();
setVizTab(activeVizTab);
initParameterTooltips();
updateInitialHeadingControls(null);
runPlanner();
