const randomizeBtn = document.getElementById('randomize-btn');
const applyParamsBtn = document.getElementById('apply-params-btn');
const resetParamsBtn = document.getElementById('reset-params-btn');
const paramForm = document.getElementById('param-form');
const statusText = document.getElementById('status-text');
const statusBadge = document.getElementById('status-badge');
const pathPlot = document.getElementById('path-plot');
const plotlyChart = document.getElementById('plotly-chart');
const hoverOverlay = document.getElementById('hover-overlay');
const legendToggles = Array.from(document.querySelectorAll('.legend-toggle'));
const axisButtons = Array.from(document.querySelectorAll('.axis-btn'));
const initialHeadingSlider = document.getElementById('initial-heading-slider');
const initialHeadingValue = document.getElementById('initial-heading-value');
const initialSpeedSlider = document.getElementById('initial-speed-slider');
const initialSpeedValue = document.getElementById('initial-speed-value');
const initialSpeedMin = document.getElementById('initial-speed-min');
const initialSpeedMid = document.getElementById('initial-speed-mid');
const initialSpeedMax = document.getElementById('initial-speed-max');
const initialAccelSlider = document.getElementById('initial-accel-slider');
const initialAccelValue = document.getElementById('initial-accel-value');
const initialAccelMin = document.getElementById('initial-accel-min');
const initialAccelMid = document.getElementById('initial-accel-mid');
const initialAccelMax = document.getElementById('initial-accel-max');
const initialKappaSlider = document.getElementById('initial-kappa-slider');
const initialKappaValue = document.getElementById('initial-kappa-value');
const initialKappaMin = document.getElementById('initial-kappa-min');
const initialKappaMid = document.getElementById('initial-kappa-mid');
const initialKappaMax = document.getElementById('initial-kappa-max');
const PATH_PLOT_CONFIG = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
};

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
    initialState: document.getElementById('initial-state'),
    referenceConfig: document.getElementById('reference-config'),
    limitsConfig: document.getElementById('limits-config'),
    weightsConfig: document.getElementById('weights-config'),
    terminalWeightsConfig: document.getElementById('terminal-weights-config'),
    realTerminalWeightsConfig: document.getElementById('real-terminal-weights-config'),
    solverConfig: document.getElementById('solver-config'),
};

const dtRefInput = paramForm?.querySelector('[data-param-group="reference"][data-param-key="dt_ref"]');
const dsInput = paramForm?.querySelector('[data-param-group="reference"][data-param-key="ds"]');
const cruiseSpeedInput = paramForm?.querySelector('[data-param-group="reference"][data-param-key="cruise_speed"]');
const dtRefPreview = document.getElementById('dt-ref-preview');
const extraPointsInput = paramForm?.querySelector('[data-param-group="reference"][data-param-key="extra_points"]');
const extraPointsPreview = document.getElementById('extra-points-preview');

const PARAM_HELP_TEXTS = {
    ds: '参考轨迹按弧长离散时的采样间距，单位 m。值越小，参考点越密，跟踪更细，但优化变量更多、求解更慢。',
    cruise_speed: '参考轨迹的名义巡航速度，单位 m/s。它会影响参考速度曲线，也会影响按时间显示时参考曲线的横轴换算。',
    dt_ref: '名义时间步长，单位 s。留空时会按 ds / cruise_speed 自动推导；它主要用于给 dt 提供初始化尺度，并用于构造停车参考。',
    selection_length: '从当前状态投影点开始，最多截取多少米参考路径用于本次优化。0 表示一直取到当前路径终点。',
    near_terminal_s_tol: '近终点 stopping 触发阈值，单位 m，按参考的 s 轴纵向剩余距离判断。0 表示自动取 max(参考采样间距, 制动距离)；当剩余距离小于该阈值时，直接切到停车参考。',
    extra_points: '对齐后的参考轨迹点数调整量。正值会额外插入优化点，负值会减少一些点，但最终至少保留 2 个点。',
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
    max_lat_accel: '最大侧向加速度，单位 m/s²。这里约束的是 |v²·kappa|，会同时限制高速急转弯。',
    max_jerk: '控制量 jerk 的绝对值上界，单位 m/s³。越小表示速度变化更平滑，但机动性更弱。',
    max_kappa: '曲率绝对值上界，单位 1/m。越小表示允许的转弯半径更大。',
    max_dkappa: '曲率变化率绝对值上界，单位 1/(m*s)。越小表示转向变化更平滑。',
    w_pos: '位置跟踪权重。越大，优化越优先贴近参考路径的 x/y 位置，但控制代价和光滑性可能被压制。',
    terminal_cost_mode: '终点代价的位置误差表达方式。world_xy 使用世界坐标下的末端位置二范数；terminal_frame 使用终点切向坐标系下的横向/纵向误差。',
    w_pos_terminal: '过程终点位置权重。只在 terminal_cost_mode = world_xy 时使用。',
    w_pos_terminal_lateral: '过程终点横向误差权重。只在 terminal_cost_mode = terminal_frame 时使用；越大越强调贴近终点切线。',
    w_pos_terminal_longitudinal: '过程终点纵向误差权重。只在 terminal_cost_mode = terminal_frame 时使用；越小越不执着于沿终点切向追满尾部。',
    w_theta: '过程终点航向权重。只在当前优化终点不是原始路径真正终点时使用；与真实路径终点权重二选一，不叠加。',
    w_speed: '速度跟踪权重。越大，优化速度曲线越接近参考速度。',
    w_time: '总时间权重。越大，优化越倾向于减小 sum(dt)，也就是缩短总时域。',
    w_speed_terminal: '过程终点速度权重。只在当前优化终点不是原始路径真正终点时使用；与真实路径终点权重二选一，不叠加。',
    w_pos_terminal_real: '真实路径终点位置权重。只在 terminal_cost_mode = world_xy 时使用。',
    w_pos_terminal_real_lateral: '真实路径终点横向误差权重。只在 terminal_cost_mode = terminal_frame 时使用。',
    w_pos_terminal_real_longitudinal: '真实路径终点纵向误差权重。只在 terminal_cost_mode = terminal_frame 时使用。',
    w_theta_terminal_real: '真实路径终点航向权重。只在当前优化目标就是原始路径真正终点时使用；与过程终点权重二选一，不叠加。',
    w_speed_terminal_real: '真实路径终点速度权重。只在当前优化目标就是原始路径真正终点时使用；与过程终点权重二选一，不叠加。',
    w_dt_smooth: 'dt 平滑权重。惩罚相邻时间步长的差异，越大则 dt 分配越平滑、越不容易突然跳变；越小则允许某些段的 dt 更集中地压缩或拉长。',
    w_jerk: 'jerk 平滑权重。越大，速度变化更平顺，但响应更保守。',
    w_dkappa: '曲率变化率平滑权重。越大，转向变化更柔和。',
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
    stopReference: true,
    optimized: true,
    correspondence: true,
    initial: true,
};

function formatNumber(value, digits = 3) {
    return Number(value).toFixed(digits);
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
        } else if (group === 'reference' && key === 'dt_ref') {
            input.value = '';
        }
    });

    updateDtRefPreview(config?.reference?.dt_ref);
}

function updateDtRefPreview(resolvedDtRef = null) {
    if (!dtRefPreview || !dtRefInput || !dsInput || !cruiseSpeedInput) {
        return;
    }

    const rawDtRef = dtRefInput.value.trim();
    if (rawDtRef !== '') {
        const manualDtRef = Number.parseFloat(rawDtRef);
        if (Number.isFinite(manualDtRef) && manualDtRef > 0) {
            dtRefPreview.textContent = `当前使用: ${formatNumber(manualDtRef, 3)} s`;
            dtRefPreview.dataset.mode = 'manual';
        } else {
            dtRefPreview.textContent = 'dt_ref 需为正数';
            dtRefPreview.dataset.mode = 'error';
        }
        return;
    }

    let autoDtRef = Number.parseFloat(resolvedDtRef);
    if (!(Number.isFinite(autoDtRef) && autoDtRef > 0)) {
        const ds = Number.parseFloat(dsInput.value.trim());
        const cruiseSpeed = Number.parseFloat(cruiseSpeedInput.value.trim());
        autoDtRef = Number.isFinite(ds) && ds > 0 && Number.isFinite(cruiseSpeed) && Math.abs(cruiseSpeed) > 0
            ? ds / Math.abs(cruiseSpeed)
            : Number.NaN;
    }

    if (Number.isFinite(autoDtRef) && autoDtRef > 0) {
        dtRefPreview.textContent = `自动计算: ${formatNumber(autoDtRef, 3)} s`;
        dtRefPreview.dataset.mode = 'auto';
    } else {
        dtRefPreview.textContent = '自动计算值不可用';
        dtRefPreview.dataset.mode = 'error';
    }
}

function updateExtraPointsPreview() {
    if (!extraPointsInput || !extraPointsPreview) {
        return;
    }

    const extraPoints = Number.parseInt(extraPointsInput.value, 10);
    if (Number.isInteger(extraPoints)) {
        extraPointsPreview.textContent = `当前调整: ${extraPoints >= 0 ? '+' : ''}${extraPoints}`;
        extraPointsPreview.dataset.mode = extraPoints === 0 ? 'manual' : extraPoints > 0 ? 'auto' : 'negative';
        return;
    }

    extraPointsPreview.textContent = '当前调整: --';
    delete extraPointsPreview.dataset.mode;
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
        let value;
        if (input.dataset.paramType === 'string') {
            value = rawValue;
        } else if (input.dataset.paramType === 'int') {
            value = Number.parseInt(rawValue, 10);
        } else {
            value = Number.parseFloat(rawValue);
        }
        if (input.dataset.paramType !== 'string' && Number.isNaN(value)) {
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

function resolveInitialSpeedRange(initialState, samplingConfig) {
    const maxCandidate = Number.parseFloat(samplingConfig?.speed_max);

    const minSpeed = 0.0;
    let maxSpeed = Number.isFinite(maxCandidate) ? Math.max(maxCandidate, minSpeed) : 1.2;

    if (initialState && Number.isFinite(initialState.v)) {
        maxSpeed = Math.max(maxSpeed, initialState.v);
    }

    if (Math.abs(maxSpeed - minSpeed) < 1e-6) {
        maxSpeed = minSpeed + 1.0;
    }

    return { minSpeed, maxSpeed };
}

function updateInitialSpeedControls(initialState, samplingConfig = null) {
    if (!initialSpeedSlider || !initialSpeedValue || !initialSpeedMin || !initialSpeedMid || !initialSpeedMax) {
        return;
    }

    const { minSpeed, maxSpeed } = resolveInitialSpeedRange(initialState, samplingConfig);
    const midSpeed = (minSpeed + maxSpeed) * 0.5;
    initialSpeedSlider.min = String(minSpeed);
    initialSpeedSlider.max = String(maxSpeed);
    initialSpeedMin.textContent = `${formatNumber(minSpeed, 2)} m/s`;
    initialSpeedMid.textContent = `${formatNumber(midSpeed, 2)} m/s`;
    initialSpeedMax.textContent = `${formatNumber(maxSpeed, 2)} m/s`;

    if (!initialState) {
        initialSpeedSlider.disabled = true;
        initialSpeedSlider.value = String(minSpeed);
        initialSpeedValue.textContent = '--';
        return;
    }

    initialSpeedSlider.disabled = false;
    initialSpeedSlider.value = String(initialState.v);
    initialSpeedValue.textContent = `${formatNumber(initialState.v, 3)} m/s`;
}

function resolveInitialAccelRange(initialState, samplingConfig) {
    const minCandidate = Number.parseFloat(samplingConfig?.accel_min);
    const maxCandidate = Number.parseFloat(samplingConfig?.accel_max);

    let minAccel = Number.isFinite(minCandidate) ? minCandidate : -0.5;
    let maxAccel = Number.isFinite(maxCandidate) ? maxCandidate : 0.5;
    if (maxAccel < minAccel) {
        [minAccel, maxAccel] = [maxAccel, minAccel];
    }

    if (initialState && Number.isFinite(initialState.a)) {
        minAccel = Math.min(minAccel, initialState.a);
        maxAccel = Math.max(maxAccel, initialState.a);
    }

    if (Math.abs(maxAccel - minAccel) < 1e-6) {
        minAccel -= 0.5;
        maxAccel += 0.5;
    }

    return { minAccel, maxAccel };
}

function updateInitialAccelControls(initialState, samplingConfig = null) {
    if (!initialAccelSlider || !initialAccelValue || !initialAccelMin || !initialAccelMid || !initialAccelMax) {
        return;
    }

    const { minAccel, maxAccel } = resolveInitialAccelRange(initialState, samplingConfig);
    const midAccel = (minAccel + maxAccel) * 0.5;
    initialAccelSlider.min = String(minAccel);
    initialAccelSlider.max = String(maxAccel);
    initialAccelMin.textContent = `${formatNumber(minAccel, 2)} m/s²`;
    initialAccelMid.textContent = `${formatNumber(midAccel, 2)} m/s²`;
    initialAccelMax.textContent = `${formatNumber(maxAccel, 2)} m/s²`;

    if (!initialState) {
        initialAccelSlider.disabled = true;
        initialAccelSlider.value = String(midAccel);
        initialAccelValue.textContent = '--';
        return;
    }

    initialAccelSlider.disabled = false;
    initialAccelSlider.value = String(initialState.a);
    initialAccelValue.textContent = `${formatNumber(initialState.a, 3)} m/s²`;
}

function resolveInitialKappaRange(initialState, samplingConfig) {
    const minCandidate = Number.parseFloat(samplingConfig?.kappa_min);
    const maxCandidate = Number.parseFloat(samplingConfig?.kappa_max);

    let minKappa = Number.isFinite(minCandidate) ? minCandidate : -0.2;
    let maxKappa = Number.isFinite(maxCandidate) ? maxCandidate : 0.2;
    if (maxKappa < minKappa) {
        [minKappa, maxKappa] = [maxKappa, minKappa];
    }

    if (initialState && Number.isFinite(initialState.kappa)) {
        minKappa = Math.min(minKappa, initialState.kappa);
        maxKappa = Math.max(maxKappa, initialState.kappa);
    }

    if (Math.abs(maxKappa - minKappa) < 1e-9) {
        minKappa -= 0.1;
        maxKappa += 0.1;
    }

    return { minKappa, maxKappa };
}

function updateInitialKappaControls(initialState, samplingConfig = null) {
    if (!initialKappaSlider || !initialKappaValue || !initialKappaMin || !initialKappaMid || !initialKappaMax) {
        return;
    }

    const { minKappa, maxKappa } = resolveInitialKappaRange(initialState, samplingConfig);
    const midKappa = (minKappa + maxKappa) * 0.5;
    initialKappaSlider.min = String(minKappa);
    initialKappaSlider.max = String(maxKappa);
    initialKappaMin.textContent = `${formatNumber(minKappa, 3)} 1/m`;
    initialKappaMid.textContent = `${formatNumber(midKappa, 3)} 1/m`;
    initialKappaMax.textContent = `${formatNumber(maxKappa, 3)} 1/m`;

    if (!initialState) {
        initialKappaSlider.disabled = true;
        initialKappaSlider.value = String(midKappa);
        initialKappaValue.textContent = '--';
        return;
    }

    initialKappaSlider.disabled = false;
    initialKappaSlider.value = String(initialState.kappa);
    initialKappaValue.textContent = `${formatNumber(initialState.kappa, 3)} 1/m`;
}

function scheduleInitialStateReplan(message) {
    if (autoReplanTimer !== null) {
        clearTimeout(autoReplanTimer);
    }
    setStatus(message, 'idle');
    autoReplanTimer = window.setTimeout(() => {
        autoReplanTimer = null;
        runRandomDemo({ preserveInitialState: true, autoTriggered: true });
    }, 220);
}

function scheduleInitialHeadingReplan() {
    scheduleInitialStateReplan('起点朝向已变更，正在等待基于当前状态自动重规划...');
}

function scheduleInitialSpeedReplan() {
    scheduleInitialStateReplan('起点速度已变更，正在等待基于当前状态自动重规划...');
}

function scheduleInitialAccelReplan() {
    scheduleInitialStateReplan('起点加速度已变更，正在等待基于当前状态自动重规划...');
}

function scheduleInitialKappaReplan() {
    scheduleInitialStateReplan('起点曲率已变更，正在等待基于当前状态自动重规划...');
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

function buildPathArrowAnnotation(x, y, theta, color, opacity, arrowLength, lineWidth = 1.4) {
    return {
        x: x + arrowLength * Math.cos(theta),
        y: y + arrowLength * Math.sin(theta),
        ax: x,
        ay: y,
        axref: 'x',
        ayref: 'y',
        text: '',
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: lineWidth,
        arrowcolor: color,
        opacity,
    };
}

function buildHeadingAnnotations(points, headings, color, step, opacity, arrowLength, lineWidth = 1.4) {
    const annotations = [];
    for (let index = 0; index < points.length; index += step) {
        annotations.push(buildPathArrowAnnotation(points[index][0], points[index][1], headings[index], color, opacity, arrowLength, lineWidth));
    }
    if (points.length > 1) {
        const last = points.length - 1;
        annotations.push(buildPathArrowAnnotation(points[last][0], points[last][1], headings[last], color, opacity, arrowLength, lineWidth));
    }
    return annotations;
}

function bindPathPlotInteractions() {
    const nextTarget = pathPlot;
    if (pathPlot.__interactionTarget !== nextTarget) {
        if (pathPlot.__interactionTarget) {
            pathPlot.__interactionTarget.removeEventListener('mousemove', handleCanvasMove, true);
            pathPlot.__interactionTarget.removeEventListener('mousedown', beginInitialStateDrag, true);
            pathPlot.__interactionTarget.removeEventListener('mouseleave', clearCanvasHover, true);
        }
        nextTarget.addEventListener('mousemove', handleCanvasMove, true);
        nextTarget.addEventListener('mousedown', beginInitialStateDrag, true);
        nextTarget.addEventListener('mouseleave', clearCanvasHover, true);
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

function renderPathView(data, activeKey = null) {
    const { reference, solution, initial_state: initialState } = data;
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
            line: { color: 'rgba(141, 133, 120, 0.42)', width: 1, dash: 'dot' },
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
            line: { color: 'rgba(15, 118, 110, 0.62)', width: 3.6, dash: 'dash' },
            marker: { color: '#0f766e', size: 8, symbol: 'x' },
            showlegend: false,
        });
        annotations.push(...buildHeadingAnnotations(referencePoints, displayReference.theta, '#0f766e', 5, 0.45, arrowLength));
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
            line: { color: 'rgba(217, 119, 6, 0.9)', width: 4.2 },
            marker: {
                color: 'rgba(217, 119, 6, 0.85)',
                size: 8,
                symbol: 'diamond',
                line: { color: 'rgba(255, 247, 242, 0.95)', width: 1.2 },
            },
            showlegend: false,
        });
        annotations.push(...buildHeadingAnnotations(stopReferencePoints, reference.theta, '#d97706', 4, 0.72, arrowLength * 0.95, 1.55));
    }

    if (layerVisibility.optimized) {
        const optimizedKeys = solutionPoints.map((_point, index) => `solution-${index}`);
        traces.push({
            x: solutionPoints.map((point) => point[0]),
            y: solutionPoints.map((point) => point[1]),
            mode: 'lines+markers',
            name: '优化路径',
            customdata: optimizedKeys,
            hovertemplate: '<extra></extra>',
            line: { color: 'rgba(202, 90, 52, 0.78)', width: 3.8 },
            marker: {
                color: 'rgba(202, 90, 52, 0.65)',
                size: 7,
                line: { color: 'rgba(255, 247, 242, 0.9)', width: 1 },
            },
            showlegend: false,
        });
        annotations.push(...buildHeadingAnnotations(solutionPoints, solution.theta, '#ca5a34', 5, 0.52, arrowLength));
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
                color: '#d97706',
                size: 14,
                line: { color: 'rgba(255, 247, 242, 0.95)', width: 1.5 },
            },
            showlegend: false,
        });
        annotations.push(buildPathArrowAnnotation(initialState.x, initialState.y, initialState.theta, '#d97706', 0.9, arrowLength * 1.15, 1.7));
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
            range: [Math.min(...xs) - paddingX, Math.max(...xs) + paddingX],
        },
        yaxis: {
            ...axisStyle,
            range: [Math.min(...ys) - paddingY, Math.max(...ys) + paddingY],
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
    const dtRef = data.reference.dt_ref;
    const dtMax = data.config.limits.dt_max;

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
            x: [controlAxis[0] ?? 0, controlAxis[controlAxis.length - 1] ?? 1],
            y: [dtMax, dtMax],
            mode: 'lines',
            name: 'dt_max',
            line: { color: '#d97706', width: 2, dash: 'dash' },
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
    updateInitialSpeedControls(initialState, data.config?.sampling);
    updateInitialAccelControls(initialState, data.config?.sampling);
    updateInitialKappaControls(initialState, data.config?.sampling);
}

function renderConfig(data) {
    const { config } = data;
    const reference = config.reference;
    const limits = config.limits;
    const weights = config.weights;
    const solver = config.solver;
    const terminalCostMode = weights.terminal_cost_mode || 'terminal_frame';
    const processWeightKeys = ['w_pos', 'w_speed', 'w_time', 'w_dt_smooth', 'w_jerk', 'w_dkappa'];
    const processTerminalWeightKeys = terminalCostMode === 'terminal_frame'
        ? ['w_pos_terminal_lateral', 'w_pos_terminal_longitudinal', 'w_speed_terminal', 'w_theta']
        : ['w_pos_terminal', 'w_speed_terminal', 'w_theta'];
    const realTerminalWeightKeys = terminalCostMode === 'terminal_frame'
        ? ['w_pos_terminal_real_lateral', 'w_pos_terminal_real_longitudinal', 'w_speed_terminal_real', 'w_theta_terminal_real']
        : ['w_pos_terminal_real', 'w_speed_terminal_real', 'w_theta_terminal_real'];
    const extraPoints = Number.parseInt(reference.params?.extra_points ?? 0, 10) || 0;
    const nearTerminalSTol = Number.parseFloat(reference.params?.near_terminal_s_tol ?? 0) || 0;
    const selectionLength = Number.parseFloat(reference.params?.selection_length ?? 0) || 0;
    const activeReferenceLength = Array.isArray(data.reference?.s) && data.reference.s.length > 0
        ? Number(data.reference.s[data.reference.s.length - 1])
        : 0;
    const renderConfigRows = (rows) => rows
        .map(([label, value]) => `<div class="config-row"><span>${label}</span><strong>${typeof value === 'number' ? formatNumber(value, 2) : value}</strong></div>`)
        .join('');
    const renderConfigGroup = (title, rows) => `
        <div class="config-subgroup">
            <div class="config-subgroup-title">${title}</div>
            <div class="config-subgroup-body">
                ${renderConfigRows(rows)}
            </div>
        </div>
    `;

    statsEls.referenceConfig.innerHTML = `
        <div class="config-stack">ds = ${formatNumber(reference.ds, 2)} m, cruise = ${formatNumber(reference.cruise_speed, 2)} m/s, dt_ref = ${formatNumber(reference.dt_ref, 2)} s</div>
        <div class="config-stack">selection_length = ${selectionLength > 0 ? `${formatNumber(selectionLength, 2)} m` : 'to end'}, active = ${formatNumber(activeReferenceLength, 2)} m</div>
        <div class="config-stack">near_terminal_s_tol = ${nearTerminalSTol > 0 ? `${formatNumber(nearTerminalSTol, 2)} m` : 'auto (max sample spacing / braking distance)'}</div>
        <div class="config-stack">extra_points = ${extraPoints}</div>
        <div class="config-stack">segments (${reference.segment_count})</div>
        ${reference.segment_descriptions.map((segment) => `<div class="config-stack">${segment}</div>`).join('')}
    `;

    statsEls.limitsConfig.innerHTML = [
        ['dt', `[${formatNumber(limits.dt_min, 2)}, ${formatNumber(limits.dt_max, 2)}] s`],
        ['max_speed', `${formatNumber(limits.max_speed, 2)} m/s`],
        ['max_accel', `${formatNumber(limits.max_accel, 2)} m/s²`],
        ['max_lat_accel', `${formatNumber(limits.max_lat_accel, 2)} m/s²`],
        ['max_jerk', `${formatNumber(limits.max_jerk, 2)} m/s³`],
        ['max_kappa', `${formatNumber(limits.max_kappa, 2)} 1/m`],
        ['max_dkappa', `${formatNumber(limits.max_dkappa, 2)} 1/(m*s)`],
    ].map(([label, value]) => `<div class="config-row"><span>${label}</span><strong>${value}</strong></div>`).join('');

    statsEls.weightsConfig.innerHTML = processWeightKeys
        .filter((key) => Object.prototype.hasOwnProperty.call(weights, key))
        .map((key) => [key, weights[key]])
        .map(([label, value]) => `<div class="config-row"><span>${label}</span><strong>${formatNumber(value, 2)}</strong></div>`)
        .join('');

    statsEls.terminalWeightsConfig.innerHTML = [
        renderConfigGroup('模式与收敛', [
            ['terminal_cost_mode', terminalCostMode],
            ['w_speed_terminal', weights.w_speed_terminal],
            ['w_theta', weights.w_theta],
        ].filter(([_label, value]) => value !== undefined)),
        renderConfigGroup('位置权重', processTerminalWeightKeys
            .filter((key) => key.startsWith('w_pos_'))
            .filter((key) => Object.prototype.hasOwnProperty.call(weights, key))
            .map((key) => [key, weights[key]])),
    ].join('');

    statsEls.realTerminalWeightsConfig.innerHTML = [
        renderConfigGroup('模式与收敛', [
            ['terminal_cost_mode', terminalCostMode],
            ['w_speed_terminal_real', weights.w_speed_terminal_real],
            ['w_theta_terminal_real', weights.w_theta_terminal_real],
        ].filter(([_label, value]) => value !== undefined)),
        renderConfigGroup('位置权重', realTerminalWeightKeys
            .filter((key) => key.startsWith('w_pos_'))
            .filter((key) => Object.prototype.hasOwnProperty.call(weights, key))
            .map((key) => [key, weights[key]])),
    ].join('');

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

function updateCanvasCursor(nearest = null) {
    if (isDraggingInitialState) {
        pathPlot.style.cursor = 'grabbing';
        return;
    }
    if (nearest?.key === 'initial-0') {
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
    pathPlot.style.cursor = 'grabbing';
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
    pathPlot.style.cursor = 'default';
    runRandomDemo({ preserveInitialState: true, dragTriggered: true });
}

function handleWindowMouseMove(event) {
    if (isDraggingInitialState) {
        handleCanvasMove(event);
    }
}

function handleCanvasMove(event) {
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
    renderPathView(currentData, activeHoverKey);
    renderStats(currentData);

    const currentInitialItem = currentScene?.itemMap.get('initial-0') || null;
    if (currentInitialItem && activeHoverKey === 'initial-0') {
        renderHoverDetails(currentInitialItem);
    }

    scheduleInitialHeadingReplan();
}

function handleInitialSpeedInput() {
    if (!currentData?.initial_state || !initialSpeedSlider) {
        return;
    }

    const sliderSpeed = Number.parseFloat(initialSpeedSlider.value);
    if (Number.isNaN(sliderSpeed)) {
        return;
    }

    currentData.initial_state = {
        ...currentData.initial_state,
        v: sliderSpeed,
    };
    activeHoverKey = 'initial-0';
    renderPathView(currentData, activeHoverKey);
    renderStats(currentData);

    const currentInitialItem = currentScene?.itemMap.get('initial-0') || null;
    if (currentInitialItem && activeHoverKey === 'initial-0') {
        renderHoverDetails(currentInitialItem);
    }

    scheduleInitialSpeedReplan();
}

function handleInitialAccelInput() {
    if (!currentData?.initial_state || !initialAccelSlider) {
        return;
    }

    const sliderAccel = Number.parseFloat(initialAccelSlider.value);
    if (Number.isNaN(sliderAccel)) {
        return;
    }

    currentData.initial_state = {
        ...currentData.initial_state,
        a: sliderAccel,
    };
    activeHoverKey = 'initial-0';
    renderPathView(currentData, activeHoverKey);
    renderStats(currentData);

    const currentInitialItem = currentScene?.itemMap.get('initial-0') || null;
    if (currentInitialItem && activeHoverKey === 'initial-0') {
        renderHoverDetails(currentInitialItem);
    }

    scheduleInitialAccelReplan();
}

function handleInitialKappaInput() {
    if (!currentData?.initial_state || !initialKappaSlider) {
        return;
    }

    const sliderKappa = Number.parseFloat(initialKappaSlider.value);
    if (Number.isNaN(sliderKappa)) {
        return;
    }

    currentData.initial_state = {
        ...currentData.initial_state,
        kappa: sliderKappa,
    };
    activeHoverKey = 'initial-0';
    renderPathView(currentData, activeHoverKey);
    renderStats(currentData);

    const currentInitialItem = currentScene?.itemMap.get('initial-0') || null;
    if (currentInitialItem && activeHoverKey === 'initial-0') {
        renderHoverDetails(currentInitialItem);
    }

    scheduleInitialKappaReplan();
}

function clearCanvasHover() {
    if (isDraggingInitialState) {
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
        setOptimizationIndicator(false, error.message);
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
window.addEventListener('mousemove', handleWindowMouseMove);
window.addEventListener('mouseup', finishInitialStateDrag);
legendToggles.forEach((toggle) => {
    toggle.addEventListener('click', () => toggleLayer(toggle.dataset.layer));
});
axisButtons.forEach((button) => {
    button.addEventListener('click', () => setChartAxisMode(button.dataset.axisMode));
});
if (initialHeadingSlider) {
    initialHeadingSlider.addEventListener('input', handleInitialHeadingInput);
}
if (initialSpeedSlider) {
    initialSpeedSlider.addEventListener('input', handleInitialSpeedInput);
}
if (initialAccelSlider) {
    initialAccelSlider.addEventListener('input', handleInitialAccelInput);
}
if (initialKappaSlider) {
    initialKappaSlider.addEventListener('input', handleInitialKappaInput);
}
paramForm.querySelectorAll('[data-param-group][data-param-key]').forEach((input) => {
    input.addEventListener('input', () => {
        updateDtRefPreview();
        updateExtraPointsPreview();
        scheduleAutoReplan(input);
    });
});
updateLegendToggleStyles();
updateAxisButtonStyles();
initParameterTooltips();
updateInitialHeadingControls(null);
updateInitialSpeedControls(null, null);
updateInitialAccelControls(null, null);
updateInitialKappaControls(null, null);
updateDtRefPreview();
updateExtraPointsPreview();
runRandomDemo();