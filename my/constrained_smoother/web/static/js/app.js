// A* + Constrained Smoother — interactive map frontend
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('map-canvas');
  const canvasWrap = document.querySelector('.canvas-wrap');
  const ctx = canvas.getContext('2d');
  const curvatureCanvas = document.getElementById('curvature-chart');
  const curvatureCtx = curvatureCanvas ? curvatureCanvas.getContext('2d') : null;
  const loupe = document.getElementById('costmap-loupe');
  const loupeCanvas = document.getElementById('loupe-canvas');
  const loupeCtx = loupeCanvas.getContext('2d');
  const mapDisplayModeSelect = document.getElementById('map-display-mode');
  const esdfColormapSelect = document.getElementById('esdf-colormap');
  const footprintModeSelect = document.getElementById('footprint_mode');
  const runBtn = document.getElementById('run-btn');
  const clearBtn = document.getElementById('clear-btn');
  const resetViewBtn = document.getElementById('reset-view-btn');
  const plannerPenaltySelect = document.getElementById('planner_penalty');
  const statusMsg = document.getElementById('status-msg');

  const sliderConfig = {
    start_yaw_deg: value => `${Math.round(value)} deg`,
    goal_yaw_deg: value => `${Math.round(value)} deg`,
    planner_penalty_weight: value => Number(value).toFixed(1),
    penalty_safe_distance_m: value => Number(value).toFixed(2),
    point_robot_radius_m: value => Number(value).toFixed(2),
    robot_length_m: value => Number(value).toFixed(2),
    robot_width_m: value => Number(value).toFixed(2),
    smooth_weight: value => Math.round(value).toLocaleString(),
    costmap_weight: value => Number(value).toFixed(3),
    distance_weight: value => Number(value).toFixed(1),
    curvature_weight: value => Number(value).toFixed(1),
    max_curvature: value => Number(value).toFixed(1),
    max_iterations: value => String(Math.round(value)),
    path_downsampling_factor: value => String(Math.round(value)),
    path_upsampling_factor: value => String(Math.round(value)),
  };

  const sliders = Object.keys(sliderConfig);
  const layerBindings = {
    layer_costmap: 'costmap',
    layer_axes: 'axes',
    layer_markers: 'markers',
    layer_astar: 'astar',
    layer_reference: 'reference',
    layer_smoothed: 'smoothed',
  };

  const planInfoIds = [
    'info-astar-time', 'info-smooth-time', 'info-astar-pts', 'info-ref-pts', 'info-opt-pts',
    'info-ref-spacing', 'info-raw-length', 'info-ref-length', 'info-opt-length', 'info-length-delta',
  ];
  const AUTO_REPLAN_DELAY_MS = 220;
  const OPTIMIZED_POINT_HOVER_RADIUS_PX = 11;
  const SMOOTHED_FORWARD_COLOR = 'rgba(191, 54, 87, 0.5)';
  const SMOOTHED_REVERSE_COLOR = 'rgba(43, 113, 186, 0.5)';
  const LOUPE_RADIUS_CELLS = 5;
  const LOUPE_CELL_SIZE = Math.floor(loupeCanvas.width / (LOUPE_RADIUS_CELLS * 2 + 1));
  const DEFAULT_ENDPOINTS = {
    start: {x: 1.0, y: 1.0},
    goal: {x: 18.0, y: 18.0},
  };
  const DEFAULT_HEADINGS_DEG = {
    start: 45,
    goal: 45,
  };

  const state = {
    costmap: null,
    start: null,
    goal: null,
    obstacles: [],
    defaultObstacles: [],
    hover: null,
    hoverCanvasPoint: null,
    hoverSample: null,
    hoverOptimizedPoint: null,
    curvatureProfile: null,
    paths: null,
    viewScale: 1,
    viewOffsetX: 0,
    viewOffsetY: 0,
    dragging: false,
    draggingMarker: null,
    draggingObstacleIndex: null,
    dragObstacleOffset: null,
    dragObstacleSize: null,
    hoverMarker: null,
    hoverObstacleIndex: null,
    didDrag: false,
    dragStartX: 0,
    dragStartY: 0,
    dragOffsetX: 0,
    dragOffsetY: 0,
    pendingAutoPlanTimer: null,
    mapDisplayMode: 'costmap',
    esdfColormap: 'diverging',
    layers: {
      costmap: true,
      axes: true,
      markers: true,
      astar: true,
      reference: true,
      smoothed: true,
    },
  };

  let costmapImageData = null;
  let costmapImageCanvas = null;
  let esdfImageData = null;
  let esdfImageCanvas = null;
  let activePlanAbortController = null;
  let activePlanRequestId = 0;
  let activeObstacleUpdateRequestId = 0;

  sliders.forEach(id => {
    const input = document.getElementById(id);
    const label = document.getElementById('val_' + id);
    if (!input || !label) {
      return;
    }

    const sync = () => {
      label.textContent = sliderConfig[id](parseFloat(input.value));
      if (id === 'start_yaw_deg' || id === 'goal_yaw_deg') {
        updateSelectionInfo();
        draw();
      }
      if (id === 'penalty_safe_distance_m' || id === 'point_robot_radius_m' ||
        id === 'robot_length_m' || id === 'robot_width_m') {
        updateRobotConfigUi();
      }
    };

    input.addEventListener('input', sync);
    input.addEventListener('input', () => scheduleAutoPlan());
    sync();
  });

  if (plannerPenaltySelect) {
    plannerPenaltySelect.addEventListener('change', () => scheduleAutoPlan());
  }

  if (mapDisplayModeSelect) {
    mapDisplayModeSelect.addEventListener('change', () => {
      state.mapDisplayMode = mapDisplayModeSelect.value;
      draw();
    });
  }

  if (esdfColormapSelect) {
    esdfColormapSelect.addEventListener('change', () => {
      state.esdfColormap = esdfColormapSelect.value;
      buildCostmapImage();
      draw();
    });
  }

  if (footprintModeSelect) {
    footprintModeSelect.addEventListener('change', () => {
      updateRobotConfigUi();
      scheduleAutoPlan();
    });
  }

  ['keep_start_orientation', 'keep_goal_orientation'].forEach(id => {
    const input = document.getElementById(id);
    if (!input) {
      return;
    }

    input.addEventListener('change', () => {
      updateSelectionInfo();
      draw();
      scheduleAutoPlan();
    });
  });

  function syncDerivedParameterInfo() {
    const maxCurvatureInput = document.getElementById('max_curvature');
    if (!maxCurvatureInput) {
      return;
    }

    const maxCurvature = parseFloat(maxCurvatureInput.value);
    const minTurnRadius = maxCurvature > 0 ? 1 / maxCurvature : null;
    setText(
      'val_min_turn_radius',
      minTurnRadius === null || Number.isNaN(minTurnRadius)
        ? 'Minimum turning radius: --'
        : `Minimum turning radius: ${minTurnRadius.toFixed(2)} m`
    );
  }

  syncDerivedParameterInfo();
  updateRobotConfigUi();
  clearOptimizedPointInspector();
  clearCurvatureChart();

  const maxCurvatureInput = document.getElementById('max_curvature');
  if (maxCurvatureInput) {
    maxCurvatureInput.addEventListener('input', () => {
      syncDerivedParameterInfo();
      drawCurvatureChart();
    });
  }

  window.addEventListener('resize', () => drawCurvatureChart());

  Object.entries(layerBindings).forEach(([id, key]) => {
    const checkbox = document.getElementById(id);
    if (!checkbox) {
      return;
    }

    checkbox.addEventListener('change', () => {
      state.layers[key] = checkbox.checked;
      draw();
    });
  });

  function setText(id, value) {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  }

  function updateRobotConfigUi() {
    const mode = footprintModeSelect ? footprintModeSelect.value : 'point';
    const pointRobotRadiusInput = document.getElementById('point_robot_radius_m');
    const robotLengthInput = document.getElementById('robot_length_m');
    const robotWidthInput = document.getElementById('robot_width_m');
    const penaltySafeDistanceInput = document.getElementById('penalty_safe_distance_m');
    const rectangleEnabled = mode === 'rectangle';
    const pointEnabled = mode === 'point';

    if (pointRobotRadiusInput) {
      pointRobotRadiusInput.disabled = !pointEnabled;
    }

    if (robotLengthInput) {
      robotLengthInput.disabled = !rectangleEnabled;
    }
    if (robotWidthInput) {
      robotWidthInput.disabled = !rectangleEnabled;
    }

    const penaltyValue = penaltySafeDistanceInput ? Number(penaltySafeDistanceInput.value).toFixed(2) : '--';
    const radiusValue = pointRobotRadiusInput ? Number(pointRobotRadiusInput.value).toFixed(2) : '--';
    const lengthValue = robotLengthInput ? Number(robotLengthInput.value).toFixed(2) : '--';
    const widthValue = robotWidthInput ? Number(robotWidthInput.value).toFixed(2) : '--';
    setText(
      'robot-config-summary',
      rectangleEnabled
        ? `Using a rectangular footprint with ${lengthValue} m length and ${widthValue} m width. Shared clearance threshold: ${penaltyValue} m.`
        : `Using a point robot with ${radiusValue} m radius. Effective clearance threshold: ${(Number(penaltyValue) + Number(radiusValue)).toFixed(2)} m.`
    );
  }

  function formatMeters(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} m`;
  }

  function formatDegrees(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} deg`;
  }

  function formatRadians(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} rad`;
  }

  function formatCurvature(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} 1/m`;
  }

  function formatCoord(point) {
    if (!point) {
      return '--';
    }
    return `${point.x.toFixed(2)}, ${point.y.toFixed(2)} m`;
  }

  function normalizeAngleDeg(angleDeg) {
    let normalized = Number(angleDeg);
    if (!Number.isFinite(normalized)) {
      return 0;
    }
    while (normalized > 180) {
      normalized -= 360;
    }
    while (normalized < -180) {
      normalized += 360;
    }
    return normalized;
  }

  function normalizeAngleRad(angleRad) {
    let normalized = Number(angleRad);
    if (!Number.isFinite(normalized)) {
      return 0;
    }
    while (normalized > Math.PI) {
      normalized -= Math.PI * 2;
    }
    while (normalized < -Math.PI) {
      normalized += Math.PI * 2;
    }
    return normalized;
  }

  function getHeadingValue(id, fallbackDeg) {
    const input = document.getElementById(id);
    if (!input) {
      return fallbackDeg;
    }
    return normalizeAngleDeg(parseFloat(input.value));
  }

  function getConstraintEnabled(id, fallback = true) {
    const input = document.getElementById(id);
    if (!input) {
      return fallback;
    }
    return input.checked;
  }

  function setStatus(message, className = '') {
    statusMsg.textContent = message;
    statusMsg.className = 'status-msg ' + className;
    setText('hero-status', message);
  }

  function clonePoint(point) {
    return {x: point.x, y: point.y};
  }

  function getCostColor(cost) {
    if (cost === 255) {
      return [176, 184, 191];
    }

    if (cost >= 254) {
      return [47, 52, 64];
    }

    if (cost > 0) {
      const t = cost / 253;
      return [
        Math.floor(237 + 14 * t),
        Math.floor(193 - 75 * t),
        Math.floor(132 - 60 * t),
      ];
    }

    return [247, 240, 224];
  }

  function interpolatePalette(stops, t) {
    if (t <= stops[0][0]) {
      return stops[0][1];
    }
    if (t >= stops[stops.length - 1][0]) {
      return stops[stops.length - 1][1];
    }

    for (let index = 1; index < stops.length; index += 1) {
      const [stopT, stopColor] = stops[index];
      const [prevT, prevColor] = stops[index - 1];
      if (t <= stopT) {
        const local = (t - prevT) / Math.max(stopT - prevT, 1e-6);
        return [
          Math.round(prevColor[0] + (stopColor[0] - prevColor[0]) * local),
          Math.round(prevColor[1] + (stopColor[1] - prevColor[1]) * local),
          Math.round(prevColor[2] + (stopColor[2] - prevColor[2]) * local),
        ];
      }
    }

    return stops[stops.length - 1][1];
  }

  function getEsdfColor(distance, minDistance, maxDistance, colormapName) {
    if (distance === null || distance === undefined || !Number.isFinite(distance)) {
      return [228, 219, 198];
    }

    const symmetricExtent = Math.max(Math.abs(minDistance), Math.abs(maxDistance), 1e-6);
    const t = Math.max(0, Math.min(1, (distance + symmetricExtent) / (2 * symmetricExtent)));
    const palettes = {
      diverging: [
        [0.0, [70, 32, 107]],
        [0.25, [195, 74, 110]],
        [0.5, [247, 240, 224]],
        [0.75, [91, 170, 161]],
        [1.0, [26, 97, 122]],
      ],
      viridis: [
        [0.0, [68, 1, 84]],
        [0.25, [59, 82, 139]],
        [0.5, [33, 145, 140]],
        [0.75, [94, 201, 98]],
        [1.0, [253, 231, 37]],
      ],
      inferno: [
        [0.0, [0, 0, 4]],
        [0.25, [87, 15, 109]],
        [0.5, [187, 55, 84]],
        [0.75, [249, 142, 8]],
        [1.0, [252, 255, 164]],
      ],
      turbo: [
        [0.0, [48, 18, 59]],
        [0.25, [50, 92, 177]],
        [0.5, [36, 200, 157]],
        [0.75, [240, 190, 45]],
        [1.0, [180, 4, 38]],
      ],
    };

    return interpolatePalette(palettes[colormapName] || palettes.diverging, t);
  }

  function describeCost(cost) {
    if (cost === null || cost === undefined) {
      return {text: 'Outside map', kind: 'outside'};
    }

    if (cost === 255) {
      return {text: 'Unknown space', kind: 'unknown'};
    }

    if (cost >= 254) {
      return {text: 'Lethal obstacle', kind: 'lethal'};
    }

    if (cost === 253) {
      return {text: 'Inscribed inflated obstacle', kind: 'inscribed'};
    }

    if (cost > 0) {
      return {text: 'Inflated cost', kind: 'inflated'};
    }

    return {text: 'Free space', kind: 'free'};
  }

  function cloneObstacleRect(rect) {
    return {x0: rect.x0, y0: rect.y0, x1: rect.x1, y1: rect.y1};
  }

  function cloneObstacleRects(rects) {
    return rects.map(cloneObstacleRect);
  }

  function resetEndpoints() {
    state.start = clonePoint(DEFAULT_ENDPOINTS.start);
    state.goal = clonePoint(DEFAULT_ENDPOINTS.goal);
    const startYawInput = document.getElementById('start_yaw_deg');
    const goalYawInput = document.getElementById('goal_yaw_deg');
    if (startYawInput) {
      startYawInput.value = String(DEFAULT_HEADINGS_DEG.start);
      document.getElementById('val_start_yaw_deg').textContent = sliderConfig.start_yaw_deg(DEFAULT_HEADINGS_DEG.start);
    }
    if (goalYawInput) {
      goalYawInput.value = String(DEFAULT_HEADINGS_DEG.goal);
      document.getElementById('val_goal_yaw_deg').textContent = sliderConfig.goal_yaw_deg(DEFAULT_HEADINGS_DEG.goal);
    }
    const keepStartInput = document.getElementById('keep_start_orientation');
    const keepGoalInput = document.getElementById('keep_goal_orientation');
    if (keepStartInput) {
      keepStartInput.checked = true;
    }
    if (keepGoalInput) {
      keepGoalInput.checked = true;
    }
  }

  function syncObstaclesFromCostmap(costmap) {
    const metadata = costmap?.metadata || {};
    state.obstacles = cloneObstacleRects(metadata.obstacle_rects_cells || []);
    state.defaultObstacles = cloneObstacleRects(metadata.default_obstacle_rects_cells || metadata.obstacle_rects_cells || []);
  }

  function hasEndpoints() {
    return Boolean(state.start && state.goal);
  }

  function cancelPendingPlanning() {
    if (state.pendingAutoPlanTimer !== null) {
      window.clearTimeout(state.pendingAutoPlanTimer);
      state.pendingAutoPlanTimer = null;
    }

    if (activePlanAbortController) {
      activePlanAbortController.abort();
      activePlanAbortController = null;
    }

    runBtn.disabled = !hasEndpoints();
  }

  function scheduleAutoPlan() {
    if (!hasEndpoints()) {
      return;
    }

    if (state.pendingAutoPlanTimer !== null) {
      window.clearTimeout(state.pendingAutoPlanTimer);
    }

    setStatus('Parameter changed. Replanning…', '');
    state.pendingAutoPlanTimer = window.setTimeout(() => {
      state.pendingAutoPlanTimer = null;
      runPlanning({reason: 'slider'});
    }, AUTO_REPLAN_DELAY_MS);
  }

  function syncPhaseUi() {
    setText('phase-indicator', 'Markers ready');
    const pillText = state.draggingMarker
      ? 'Dragging marker'
      : state.draggingObstacleIndex !== null
        ? 'Dragging obstacle'
        : 'Drag scene';
    setText('selection-pill', pillText);
  }

  function updateSelectionInfo() {
    const startHeading = getHeadingValue('start_yaw_deg', DEFAULT_HEADINGS_DEG.start);
    const goalHeading = getHeadingValue('goal_yaw_deg', DEFAULT_HEADINGS_DEG.goal);
    const keepStartOrientation = getConstraintEnabled('keep_start_orientation', true);
    const keepGoalOrientation = getConstraintEnabled('keep_goal_orientation', true);
    setText('start-coord', `${formatCoord(state.start)} | ${Math.round(startHeading)} deg`);
    setText('goal-coord', `${formatCoord(state.goal)} | ${Math.round(goalHeading)} deg`);
    setText('start-heading-readout', `${Math.round(startHeading)} deg`);
    setText('goal-heading-readout', `${Math.round(goalHeading)} deg`);
    setText('start-constraint-readout', keepStartOrientation ? 'Enabled' : 'Disabled');
    setText('goal-constraint-readout', keepGoalOrientation ? 'Enabled' : 'Disabled');
    setText('cursor-coord', formatCoord(state.hover));
    setText('zoom-level', `${state.viewScale.toFixed(2)}x`);
    const gestureText = state.draggingMarker
      ? 'Dragging marker'
      : state.draggingObstacleIndex !== null
        ? 'Dragging obstacle'
        : state.dragging
          ? 'Panning view'
          : 'Left-drag';
    setText('view-mode-label', gestureText);
    syncPhaseUi();
  }

  function hideLoupe() {
    setText('loupe-cost-value', 'ESDF --');
    setText('loupe-cell-cost', '--');
    setText('loupe-esdf-distance', '--');
    setText('loupe-world', 'World: --');
    setText('loupe-cell', 'Cell: --');
    setText('loupe-kind', 'Outside map');
    document.getElementById('loupe-kind')?.setAttribute('data-kind', 'outside');
  }

  function sampleCostmap(worldPoint) {
    if (!state.costmap || !worldPoint) {
      return null;
    }

    const cellX = (worldPoint.x - state.costmap.origin_x) / state.costmap.resolution;
    const cellY = (worldPoint.y - state.costmap.origin_y) / state.costmap.resolution;
    const mx = Math.floor(cellX);
    const my = Math.floor(cellY);
    if (mx < 0 || my < 0 || mx >= state.costmap.size_x || my >= state.costmap.size_y) {
      return null;
    }

    const interpolate = (grid, x, y) => {
      const shiftedX = x - 0.5;
      const shiftedY = y - 0.5;
      const x0 = Math.floor(shiftedX);
      const y0 = Math.floor(shiftedY);
      const tx = shiftedX - x0;
      const ty = shiftedY - y0;
      const sampleAt = (sx, sy) => {
        const clampedX = Math.max(0, Math.min(state.costmap.size_x - 1, sx));
        const clampedY = Math.max(0, Math.min(state.costmap.size_y - 1, sy));
        return grid[clampedY * state.costmap.size_x + clampedX];
      };

      const c00 = sampleAt(x0, y0);
      const c10 = sampleAt(x0 + 1, y0);
      const c01 = sampleAt(x0, y0 + 1);
      const c11 = sampleAt(x0 + 1, y0 + 1);
      const top = c00 * (1 - tx) + c10 * tx;
      const bottom = c01 * (1 - tx) + c11 * tx;
      return top * (1 - ty) + bottom * ty;
    };

    return {
      mx,
      my,
      cellX,
      cellY,
      cost: state.costmap.data[my * state.costmap.size_x + mx],
      esdfDistance: state.costmap.esdf
        ? interpolate(state.costmap.esdf, cellX, cellY)
        : null,
      worldX: worldPoint.x,
      worldY: worldPoint.y,
    };
  }

  function drawLoupe(sample) {
    const diameter = LOUPE_RADIUS_CELLS * 2 + 1;
    const gridPixelSize = diameter * LOUPE_CELL_SIZE;
    const inset = Math.floor((loupeCanvas.width - gridPixelSize) / 2);
    loupeCtx.clearRect(0, 0, loupeCanvas.width, loupeCanvas.height);
    loupeCtx.fillStyle = 'rgba(246, 239, 224, 0.96)';
    loupeCtx.fillRect(0, 0, loupeCanvas.width, loupeCanvas.height);

    for (let offsetY = -LOUPE_RADIUS_CELLS; offsetY <= LOUPE_RADIUS_CELLS; offsetY += 1) {
      for (let offsetX = -LOUPE_RADIUS_CELLS; offsetX <= LOUPE_RADIUS_CELLS; offsetX += 1) {
        const mx = sample.mx + offsetX;
        const my = sample.my + offsetY;
        const drawX = inset + (offsetX + LOUPE_RADIUS_CELLS) * LOUPE_CELL_SIZE;
        const drawY = inset + (LOUPE_RADIUS_CELLS - offsetY) * LOUPE_CELL_SIZE;
        let color = [228, 219, 198];
        let alpha = 0.7;

        if (mx >= 0 && my >= 0 && mx < state.costmap.size_x && my < state.costmap.size_y) {
          color = getCostColor(state.costmap.data[my * state.costmap.size_x + mx]);
          alpha = 1;
        }

        loupeCtx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
        loupeCtx.fillRect(drawX, drawY, LOUPE_CELL_SIZE, LOUPE_CELL_SIZE);
        loupeCtx.strokeStyle = 'rgba(255, 250, 240, 0.45)';
        loupeCtx.lineWidth = 1;
        loupeCtx.strokeRect(drawX + 0.5, drawY + 0.5, LOUPE_CELL_SIZE - 1, LOUPE_CELL_SIZE - 1);
      }
    }

    const center = inset + LOUPE_RADIUS_CELLS * LOUPE_CELL_SIZE;
    loupeCtx.save();
    loupeCtx.strokeStyle = 'rgba(15, 92, 80, 0.95)';
    loupeCtx.lineWidth = 2;
    loupeCtx.strokeRect(center + 1, center + 1, LOUPE_CELL_SIZE - 2, LOUPE_CELL_SIZE - 2);
    loupeCtx.beginPath();
    loupeCtx.moveTo(center + LOUPE_CELL_SIZE / 2, inset - 2);
    loupeCtx.lineTo(center + LOUPE_CELL_SIZE / 2, inset + gridPixelSize + 2);
    loupeCtx.moveTo(inset - 2, center + LOUPE_CELL_SIZE / 2);
    loupeCtx.lineTo(inset + gridPixelSize + 2, center + LOUPE_CELL_SIZE / 2);
    loupeCtx.strokeStyle = 'rgba(15, 92, 80, 0.35)';
    loupeCtx.lineWidth = 1;
    loupeCtx.stroke();
    loupeCtx.restore();
  }

  function updateLoupe() {
    const sample = state.hoverSample;
    if (!sample) {
      hideLoupe();
      return;
    }

    const descriptor = describeCost(sample.cost);
    drawLoupe(sample);
    setText(
      'loupe-cost-value',
      sample.esdfDistance === null ? 'ESDF --' : `ESDF ${sample.esdfDistance.toFixed(2)} m`
    );
    setText('loupe-cell-cost', String(sample.cost));
    setText(
      'loupe-esdf-distance',
      sample.esdfDistance === null ? '--' : `${sample.esdfDistance.toFixed(2)} m`
    );
    setText('loupe-world', `World: ${sample.worldX.toFixed(2)}, ${sample.worldY.toFixed(2)} m`);
    setText('loupe-cell', `Cell: (${sample.mx}, ${sample.my})`);
    setText('loupe-kind', descriptor.text);
    document.getElementById('loupe-kind')?.setAttribute('data-kind', descriptor.kind);
  }

  function clearOptimizedPointInspector() {
    const popup = document.getElementById('opt-point-popup');
    if (popup) {
      popup.hidden = true;
    }
    [
      'opt-point-role', 'opt-point-index', 'opt-point-world', 'opt-point-heading', 'opt-point-tangent',
      'opt-point-arc', 'opt-point-prev-segment', 'opt-point-next-segment', 'opt-point-turn',
      'opt-point-curvature', 'opt-point-esdf', 'opt-point-cost', 'opt-point-cursor-offset',
    ].forEach(id => setText(id, '--'));
    setText(
      'opt-point-note',
      'Hover a point on the rose smoothed path to inspect its geometry, heading, local clearance, and segment context.'
    );
  }

  function positionOptimizedPointPopup() {
    const popup = document.getElementById('opt-point-popup');
    const wrap = document.querySelector('.canvas-wrap');
    if (!popup || !wrap || popup.hidden || !state.hoverCanvasPoint) {
      return;
    }

    const margin = 14;
    const offsetX = 18;
    const offsetY = 18;
    const wrapRect = wrap.getBoundingClientRect();
    const popupRect = popup.getBoundingClientRect();
    const maxLeft = Math.max(margin, wrapRect.width - popupRect.width - margin);
    const maxTop = Math.max(margin, wrapRect.height - popupRect.height - margin);

    let left = state.hoverCanvasPoint.x + offsetX;
    let top = state.hoverCanvasPoint.y + offsetY;
    if (left > maxLeft) {
      left = Math.max(margin, state.hoverCanvasPoint.x - popupRect.width - offsetX);
    }
    if (top > maxTop) {
      top = Math.max(margin, state.hoverCanvasPoint.y - popupRect.height - offsetY);
    }

    popup.style.left = `${Math.min(Math.max(left, margin), maxLeft)}px`;
    popup.style.top = `${Math.min(Math.max(top, margin), maxTop)}px`;
  }

  function getPointRole(index, pointCount) {
    const keepStartOrientation = getConstraintEnabled('keep_start_orientation', true);
    const keepGoalOrientation = getConstraintEnabled('keep_goal_orientation', true);
    if (index === 0) {
      return 'Start endpoint';
    }
    if (index === pointCount - 1) {
      return 'Goal endpoint';
    }
    if (keepStartOrientation && index === 1) {
      return 'Start anchor';
    }
    if (keepGoalOrientation && index === pointCount - 2) {
      return 'Goal anchor';
    }
    return 'Interior point';
  }

  function buildOptimizedPointHoverInfo(index, distancePx) {
    if (!state.paths || !state.hover) {
      return null;
    }

    const xs = state.paths.opt_x || [];
    const ys = state.paths.opt_y || [];
    const thetas = state.paths.opt_theta || [];
    if (index < 0 || index >= xs.length || index >= ys.length) {
      return null;
    }

    const worldX = xs[index];
    const worldY = ys[index];
    const thetaRad = Number.isFinite(thetas[index]) ? thetas[index] : null;
    const prevLen = index > 0 ? Math.hypot(worldX - xs[index - 1], worldY - ys[index - 1]) : null;
    const nextLen = index < xs.length - 1 ? Math.hypot(xs[index + 1] - worldX, ys[index + 1] - worldY) : null;

    let arcLength = 0;
    for (let idx = 1; idx <= index; idx += 1) {
      arcLength += Math.hypot(xs[idx] - xs[idx - 1], ys[idx] - ys[idx - 1]);
    }

    let tangentHeadingRad = null;
    if (index < xs.length - 1) {
      tangentHeadingRad = Math.atan2(ys[index + 1] - worldY, xs[index + 1] - worldX);
    } else if (index > 0) {
      tangentHeadingRad = Math.atan2(worldY - ys[index - 1], worldX - xs[index - 1]);
    }

    let turnAngleRad = null;
    let approxCurvature = null;
    if (index > 0 && index < xs.length - 1 && prevLen && nextLen) {
      const prevVecX = worldX - xs[index - 1];
      const prevVecY = worldY - ys[index - 1];
      const nextVecX = xs[index + 1] - worldX;
      const nextVecY = ys[index + 1] - worldY;
      const cross = prevVecX * nextVecY - prevVecY * nextVecX;
      const dot = prevVecX * nextVecX + prevVecY * nextVecY;
      turnAngleRad = Math.atan2(cross, dot);
      const avgSegment = Math.max((prevLen + nextLen) * 0.5, 1e-6);
      approxCurvature = Math.abs(turnAngleRad) / avgSegment;
    }

    const sample = sampleCostmap({x: worldX, y: worldY});
    const cursorOffset = Math.hypot(state.hover.x - worldX, state.hover.y - worldY);
    return {
      role: getPointRole(index, xs.length),
      index,
      pointCount: xs.length,
      worldX,
      worldY,
      thetaRad,
      tangentHeadingRad,
      arcLength,
      prevLen,
      nextLen,
      turnAngleRad,
      approxCurvature,
      esdfDistance: sample?.esdfDistance ?? null,
      cost: sample?.cost ?? null,
      cursorOffset,
      distancePx,
    };
  }

  function findHoveredOptimizedPoint(canvasX, canvasY) {
    if (!state.paths || !state.layers.smoothed) {
      return null;
    }

    const xs = state.paths.opt_x || [];
    const ys = state.paths.opt_y || [];
    let bestIndex = -1;
    let bestDistanceSq = OPTIMIZED_POINT_HOVER_RADIUS_PX * OPTIMIZED_POINT_HOVER_RADIUS_PX;

    for (let idx = 0; idx < xs.length; idx += 1) {
      const point = worldToCanvas(xs[idx], ys[idx]);
      const dx = point.x - canvasX;
      const dy = point.y - canvasY;
      const distanceSq = dx * dx + dy * dy;
      if (distanceSq <= bestDistanceSq) {
        bestDistanceSq = distanceSq;
        bestIndex = idx;
      }
    }

    if (bestIndex < 0) {
      return null;
    }

    return buildOptimizedPointHoverInfo(bestIndex, Math.sqrt(bestDistanceSq));
  }

  function updateOptimizedPointInspector() {
    const info = state.hoverOptimizedPoint;
    if (!info) {
      clearOptimizedPointInspector();
      return;
    }

    const popup = document.getElementById('opt-point-popup');
    if (popup) {
      popup.hidden = false;
    }

    setText('opt-point-role', info.role);
    setText('opt-point-index', `${info.index + 1} / ${info.pointCount}`);
    setText('opt-point-world', `${info.worldX.toFixed(2)}, ${info.worldY.toFixed(2)} m`);
    setText(
      'opt-point-heading',
      info.thetaRad === null
        ? '--'
        : `${formatDegrees(normalizeAngleDeg(info.thetaRad * 180 / Math.PI))} / ${formatRadians(info.thetaRad)}`
    );
    setText(
      'opt-point-tangent',
      info.tangentHeadingRad === null
        ? '--'
        : `${formatDegrees(normalizeAngleDeg(info.tangentHeadingRad * 180 / Math.PI))} / ${formatRadians(info.tangentHeadingRad)}`
    );
    setText('opt-point-arc', formatMeters(info.arcLength));
    setText('opt-point-prev-segment', formatMeters(info.prevLen));
    setText('opt-point-next-segment', formatMeters(info.nextLen));
    setText(
      'opt-point-turn',
      info.turnAngleRad === null ? '--' : formatDegrees(normalizeAngleDeg(info.turnAngleRad * 180 / Math.PI))
    );
    setText(
      'opt-point-curvature',
      info.approxCurvature === null || Number.isNaN(info.approxCurvature)
        ? '--'
        : `${info.approxCurvature.toFixed(2)} 1/m`
    );
    setText('opt-point-esdf', formatMeters(info.esdfDistance));
    setText('opt-point-cost', info.cost === null || info.cost === undefined ? '--' : String(info.cost));
    setText('opt-point-cursor-offset', `${formatMeters(info.cursorOffset)} / ${info.distancePx.toFixed(1)} px`);
    setText(
      'opt-point-note',
      `Point ${info.index + 1} is ${info.role.toLowerCase()}. Turn angle and curvature are estimated from the local three-point geometry around this optimized pose.`
    );
    positionOptimizedPointPopup();
  }

  function updateCanvasCursor() {
    const isMarkerDrag = Boolean(state.draggingMarker || state.draggingObstacleIndex !== null);
    canvas.classList.toggle('is-pan-mode', true);
    canvas.classList.toggle('is-dragging', state.dragging || isMarkerDrag);
  }

  function clampWorldPoint(point) {
    if (!state.costmap) {
      return point;
    }

    const maxX = state.costmap.origin_x + state.costmap.size_x * state.costmap.resolution;
    const maxY = state.costmap.origin_y + state.costmap.size_y * state.costmap.resolution;
    return {
      x: Math.min(maxX, Math.max(state.costmap.origin_x, point.x)),
      y: Math.min(maxY, Math.max(state.costmap.origin_y, point.y)),
    };
  }

  function getMarkerAtCanvasPoint(cx, cy) {
    if (!state.layers.markers) {
      return null;
    }

    const candidates = [
      ['goal', state.goal],
      ['start', state.start],
    ];
    const hitRadius = 14;

    for (const [name, point] of candidates) {
      if (!point) {
        continue;
      }
      const markerPixel = worldToCanvas(point.x, point.y);
      const distance = Math.hypot(cx - markerPixel.x, cy - markerPixel.y);
      if (distance <= hitRadius) {
        return name;
      }
    }

    return null;
  }

  function worldToCell(wx, wy) {
    if (!state.costmap) {
      return {x: 0, y: 0};
    }
    return {
      x: (wx - state.costmap.origin_x) / state.costmap.resolution,
      y: (wy - state.costmap.origin_y) / state.costmap.resolution,
    };
  }

  function obstacleRectToCanvasBounds(rect) {
    const resolution = state.costmap.resolution;
    const minWorldX = state.costmap.origin_x + rect.x0 * resolution;
    const minWorldY = state.costmap.origin_y + rect.y0 * resolution;
    const maxWorldX = state.costmap.origin_x + rect.x1 * resolution;
    const maxWorldY = state.costmap.origin_y + rect.y1 * resolution;
    const topLeft = worldToCanvas(minWorldX, maxWorldY);
    const bottomRight = worldToCanvas(maxWorldX, minWorldY);

    return {
      left: Math.min(topLeft.x, bottomRight.x),
      right: Math.max(topLeft.x, bottomRight.x),
      top: Math.min(topLeft.y, bottomRight.y),
      bottom: Math.max(topLeft.y, bottomRight.y),
    };
  }

  function getObstacleAtCanvasPoint(cx, cy) {
    if (!state.costmap || !state.obstacles.length) {
      return null;
    }

    for (let index = state.obstacles.length - 1; index >= 0; index -= 1) {
      const bounds = obstacleRectToCanvasBounds(state.obstacles[index]);
      if (cx >= bounds.left && cx <= bounds.right && cy >= bounds.top && cy <= bounds.bottom) {
        return index;
      }
    }

    return null;
  }

  function updateMapInfo(costmap) {
    const meta = costmap.metadata || {};
    const worldWidth = meta.world_width_m ?? (costmap.size_x * costmap.resolution);
    const worldHeight = meta.world_height_m ?? (costmap.size_y * costmap.resolution);

    setText('hero-map-size', `${worldWidth.toFixed(1)} x ${worldHeight.toFixed(1)} m`);
    setText('hero-resolution', `${costmap.resolution.toFixed(2)} m/cell`);
    setText('map-grid', `${costmap.size_x} x ${costmap.size_y}`);
    setText('map-world-size', `${worldWidth.toFixed(1)} x ${worldHeight.toFixed(1)} m`);
    setText('map-world-size-toolbar', `${worldWidth.toFixed(1)} x ${worldHeight.toFixed(1)} m`);
    setText('map-resolution', `${costmap.resolution.toFixed(2)} m/cell`);
    setText('map-origin', `${costmap.origin_x.toFixed(1)}, ${costmap.origin_y.toFixed(1)} m`);
    setText('map-obstacles', String(meta.obstacle_count ?? '--'));
    setText('map-inflation', `${(meta.inflation_radius_m ?? 0).toFixed(2)} m / ${meta.inflation_radius_cells ?? '--'} cells`);
    setText('map-inflation-toolbar', `${(meta.inflation_radius_m ?? 0).toFixed(2)} m`);
    setText('map-free-cells', `${meta.free_cells ?? '--'} / ${meta.cell_count ?? '--'}`);
    setText('map-inflated-cells', `${meta.inflated_cells ?? '--'} / ${meta.cell_count ?? '--'}`);
    setText('map-lethal-cells', `${meta.lethal_cells ?? '--'} / ${meta.cell_count ?? '--'}`);
    setText('map-description', meta.description || 'Fixed synthetic obstacle map used to inspect ESDF-based planner and smoother behavior.');
    setText('map-kind', meta.name || 'Synthetic field');
  }

  function computeCurvatureProfile(pathData) {
    if (!pathData?.opt_x || pathData.opt_x.length < 2) {
      return null;
    }

    const xs = pathData.opt_x;
    const ys = pathData.opt_y;
    const pointCount = Math.min(xs.length, ys.length);
    const arcLengths = new Array(pointCount).fill(0);
    const curvatures = new Array(pointCount).fill(0);

    for (let idx = 1; idx < pointCount; idx += 1) {
      arcLengths[idx] = arcLengths[idx - 1] + Math.hypot(xs[idx] - xs[idx - 1], ys[idx] - ys[idx - 1]);
    }

    for (let idx = 1; idx < pointCount - 1; idx += 1) {
      const prevVecX = xs[idx] - xs[idx - 1];
      const prevVecY = ys[idx] - ys[idx - 1];
      const nextVecX = xs[idx + 1] - xs[idx];
      const nextVecY = ys[idx + 1] - ys[idx];
      const prevLen = Math.hypot(prevVecX, prevVecY);
      const nextLen = Math.hypot(nextVecX, nextVecY);
      if (prevLen <= 1e-6 || nextLen <= 1e-6) {
        curvatures[idx] = 0;
        continue;
      }
      const cross = prevVecX * nextVecY - prevVecY * nextVecX;
      const dot = prevVecX * nextVecX + prevVecY * nextVecY;
      const turnAngle = Math.atan2(cross, dot);
      const avgSegment = Math.max((prevLen + nextLen) * 0.5, 1e-6);
      curvatures[idx] = turnAngle / avgSegment;
    }

    const signedMin = Math.min(...curvatures);
    const signedMax = Math.max(...curvatures);
    const absValues = curvatures.map(value => Math.abs(value));
    const peakAbs = Math.max(...absValues);
    const meanAbs = absValues.reduce((sum, value) => sum + value, 0) / absValues.length;
    return {
      arcLengths,
      curvatures,
      signedMin,
      signedMax,
      peakAbs,
      meanAbs,
      totalLength: arcLengths[arcLengths.length - 1],
    };
  }

  function clearCurvatureChart() {
    state.curvatureProfile = null;
    setText('curvature-state', 'idle');
    setText('curvature-peak', '--');
    setText('curvature-mean', '--');
    setText('curvature-min', '--');
    setText('curvature-max', '--');
    setText('curvature-note', 'Run planning to plot the signed curvature of the optimized path against arc length.');

    if (!curvatureCanvas || !curvatureCtx) {
      return;
    }

    const rect = curvatureCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    curvatureCanvas.width = width;
    curvatureCanvas.height = height;
    curvatureCtx.setTransform(1, 0, 0, 1, 0, 0);
    curvatureCtx.clearRect(0, 0, width, height);
    curvatureCtx.fillStyle = 'rgba(255, 250, 240, 0.96)';
    curvatureCtx.fillRect(0, 0, width, height);
    curvatureCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    curvatureCtx.fillStyle = 'rgba(108, 111, 97, 0.8)';
    curvatureCtx.font = '600 13px "Avenir Next", sans-serif';
    curvatureCtx.textAlign = 'center';
    curvatureCtx.textBaseline = 'middle';
    curvatureCtx.fillText('Curvature chart will appear after a successful plan.', rect.width / 2, rect.height / 2);
  }

  function drawCurvatureChart() {
    if (!curvatureCanvas || !curvatureCtx) {
      return;
    }

    const profile = state.curvatureProfile;
    if (!profile || profile.arcLengths.length < 2) {
      clearCurvatureChart();
      return;
    }

    const rect = curvatureCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    curvatureCanvas.width = width;
    curvatureCanvas.height = height;

    curvatureCtx.setTransform(1, 0, 0, 1, 0, 0);
    curvatureCtx.clearRect(0, 0, width, height);
    curvatureCtx.fillStyle = 'rgba(255, 250, 240, 0.96)';
    curvatureCtx.fillRect(0, 0, width, height);
    curvatureCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const plotWidth = rect.width;
    const plotHeight = rect.height;
    const padding = {left: 52, right: 16, top: 18, bottom: 30};
    const innerWidth = Math.max(1, plotWidth - padding.left - padding.right);
    const innerHeight = Math.max(1, plotHeight - padding.top - padding.bottom);
    const maxCurvatureLimit = parseFloat(document.getElementById('max_curvature')?.value || '0');
    const extent = Math.max(profile.peakAbs, maxCurvatureLimit, 0.1);

    const xToCanvas = value => padding.left + (value / Math.max(profile.totalLength, 1e-6)) * innerWidth;
    const yToCanvas = value => padding.top + innerHeight * (0.5 - value / (2 * extent));

    curvatureCtx.strokeStyle = 'rgba(100, 85, 60, 0.16)';
    curvatureCtx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach(t => {
      const y = padding.top + innerHeight * t;
      curvatureCtx.beginPath();
      curvatureCtx.moveTo(padding.left, y);
      curvatureCtx.lineTo(padding.left + innerWidth, y);
      curvatureCtx.stroke();
    });

    const zeroY = yToCanvas(0);
    curvatureCtx.strokeStyle = 'rgba(35, 48, 40, 0.35)';
    curvatureCtx.lineWidth = 1.2;
    curvatureCtx.beginPath();
    curvatureCtx.moveTo(padding.left, zeroY);
    curvatureCtx.lineTo(padding.left + innerWidth, zeroY);
    curvatureCtx.stroke();

    if (maxCurvatureLimit > 0) {
      [maxCurvatureLimit, -maxCurvatureLimit].forEach(limit => {
        const y = yToCanvas(limit);
        curvatureCtx.save();
        curvatureCtx.setLineDash([6, 6]);
        curvatureCtx.strokeStyle = 'rgba(217, 122, 43, 0.8)';
        curvatureCtx.lineWidth = 1.2;
        curvatureCtx.beginPath();
        curvatureCtx.moveTo(padding.left, y);
        curvatureCtx.lineTo(padding.left + innerWidth, y);
        curvatureCtx.stroke();
        curvatureCtx.restore();
      });
    }

    for (let idx = 0; idx < profile.arcLengths.length - 1; idx += 1) {
      const x0 = xToCanvas(profile.arcLengths[idx]);
      const y0 = yToCanvas(profile.curvatures[idx]);
      const x1 = xToCanvas(profile.arcLengths[idx + 1]);
      const y1 = yToCanvas(profile.curvatures[idx + 1]);
      curvatureCtx.strokeStyle = profile.curvatures[idx] >= 0 ? 'rgba(191, 54, 87, 0.9)' : 'rgba(43, 113, 186, 0.9)';
      curvatureCtx.lineWidth = 2.1;
      curvatureCtx.beginPath();
      curvatureCtx.moveTo(x0, y0);
      curvatureCtx.lineTo(x1, y1);
      curvatureCtx.stroke();
    }

    curvatureCtx.strokeStyle = 'rgba(35, 48, 40, 0.55)';
    curvatureCtx.lineWidth = 1.4;
    curvatureCtx.strokeRect(padding.left, padding.top, innerWidth, innerHeight);

    curvatureCtx.fillStyle = 'rgba(35, 48, 40, 0.82)';
    curvatureCtx.font = '600 12px "Avenir Next", sans-serif';
    curvatureCtx.textAlign = 'left';
    curvatureCtx.textBaseline = 'top';
    curvatureCtx.fillText('curvature (1/m)', padding.left, 2);
    curvatureCtx.textAlign = 'right';
    curvatureCtx.textBaseline = 'bottom';
    curvatureCtx.fillText(`arc length 0 -> ${profile.totalLength.toFixed(2)} m`, padding.left + innerWidth, plotHeight - 4);
    curvatureCtx.textAlign = 'left';
    curvatureCtx.textBaseline = 'middle';
    curvatureCtx.fillText(`${extent.toFixed(2)}`, 8, yToCanvas(extent));
    curvatureCtx.fillText('0', 8, zeroY);
    curvatureCtx.fillText(`${(-extent).toFixed(2)}`, 8, yToCanvas(-extent));

    setText('curvature-state', 'chart ready');
    setText('curvature-peak', formatCurvature(profile.peakAbs));
    setText('curvature-mean', formatCurvature(profile.meanAbs));
    setText('curvature-min', formatCurvature(profile.signedMin));
    setText('curvature-max', formatCurvature(profile.signedMax));
    setText(
      'curvature-note',
      `Signed curvature is estimated from local tangent change per metre along the optimized path. Dashed amber lines mark the current Max Curvature limit (${maxCurvatureLimit.toFixed(2)} 1/m).`
    );
  }

  function updateRunInfo(data) {
    state.curvatureProfile = computeCurvatureProfile(data);
    setText('info-astar-time', `${data.astar_time_ms} ms`);
    setText('info-smooth-time', `${data.smooth_time_ms} ms`);
    setText('info-astar-pts', String(data.num_astar_pts));
    setText('info-ref-pts', String(data.num_ref_pts));
    setText('info-opt-pts', String(data.num_opt_pts));
    setText('info-ref-spacing', formatMeters(data.reference_spacing_target_m));
    setText('info-raw-length', formatMeters(data.raw_path_length_m));
    setText('info-ref-length', formatMeters(data.ref_path_length_m));
    setText('info-opt-length', formatMeters(data.opt_path_length_m));

    const deltaValue = Number(data.opt_vs_ref_delta_m);
    const deltaText = Number.isNaN(deltaValue)
      ? '--'
      : `${deltaValue >= 0 ? '+' : ''}${deltaValue.toFixed(2)} m`;
    setText('info-length-delta', deltaText);

    setText('smooth-state', data.smooth_success ? 'smooth success' : 'fallback to ref');
    setText(
      'run-note',
      data.smooth_success
        ? 'Compare the raw, reference, and smoothed path lengths while toggling layers to inspect how the optimizer changed geometry.'
        : `Smoothing failed and the reference path is being shown instead. ${data.smooth_message || ''}`.trim()
    );
    drawCurvatureChart();
  }

  function clearRunInfo() {
    planInfoIds.forEach(id => setText(id, '--'));
    setText('smooth-state', 'idle');
    setText('run-note', 'Set a start and goal to generate path metrics.');
    clearCurvatureChart();
  }

  function worldToCanvas(wx, wy) {
    if (!state.costmap) {
      return {x: 0, y: 0};
    }

    const costmap = state.costmap;
    const cellX = (wx - costmap.origin_x) / costmap.resolution;
    const cellY = (wy - costmap.origin_y) / costmap.resolution;
    const px = cellX * (canvas.width / costmap.size_x);
    const py = (costmap.size_y - cellY) * (canvas.height / costmap.size_y);

    return {
      x: px * state.viewScale + state.viewOffsetX,
      y: py * state.viewScale + state.viewOffsetY,
    };
  }

  function clientToCanvasPoint(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  function clientDeltaToCanvasDelta(deltaX, deltaY) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: deltaX * (canvas.width / rect.width),
      y: deltaY * (canvas.height / rect.height),
    };
  }

  function canvasToWorld(cx, cy) {
    if (!state.costmap) {
      return {x: 0, y: 0};
    }

    const costmap = state.costmap;
    const px = (cx - state.viewOffsetX) / state.viewScale;
    const py = (cy - state.viewOffsetY) / state.viewScale;
    const cellX = px / (canvas.width / costmap.size_x);
    const cellY = costmap.size_y - py / (canvas.height / costmap.size_y);

    return {
      x: costmap.origin_x + cellX * costmap.resolution,
      y: costmap.origin_y + cellY * costmap.resolution,
    };
  }

  function resetView() {
    state.viewScale = 1;
    state.viewOffsetX = 0;
    state.viewOffsetY = 0;
    updateSelectionInfo();
    draw();
  }

  function buildCostmapImage() {
    if (!state.costmap) {
      return;
    }

    const costmap = state.costmap;
    const image = ctx.createImageData(costmap.size_x, costmap.size_y);

    for (let my = 0; my < costmap.size_y; my += 1) {
      for (let mx = 0; mx < costmap.size_x; mx += 1) {
        const cost = costmap.data[my * costmap.size_x + mx];
        const canvasRow = costmap.size_y - 1 - my;
        const idx = (canvasRow * costmap.size_x + mx) * 4;

        const [red, green, blue] = getCostColor(cost);
        image.data[idx] = red;
        image.data[idx + 1] = green;
        image.data[idx + 2] = blue;
        image.data[idx + 3] = 255;
      }
    }

    costmapImageData = image;
    costmapImageCanvas = document.createElement('canvas');
    costmapImageCanvas.width = costmap.size_x;
    costmapImageCanvas.height = costmap.size_y;
    costmapImageCanvas.getContext('2d').putImageData(costmapImageData, 0, 0);

    const esdfValues = Array.isArray(costmap.esdf) ? costmap.esdf : null;
    if (!esdfValues) {
      esdfImageData = null;
      esdfImageCanvas = null;
      return;
    }

    const finiteEsdfValues = esdfValues.filter(value => Number.isFinite(value));
    const maxDistance = finiteEsdfValues.length ? Math.max(...finiteEsdfValues) : 1.0;
    const minDistance = finiteEsdfValues.length ? Math.min(...finiteEsdfValues) : -1.0;
    const esdfImage = ctx.createImageData(costmap.size_x, costmap.size_y);

    for (let my = 0; my < costmap.size_y; my += 1) {
      for (let mx = 0; mx < costmap.size_x; mx += 1) {
        const distance = esdfValues[my * costmap.size_x + mx];
        const canvasRow = costmap.size_y - 1 - my;
        const idx = (canvasRow * costmap.size_x + mx) * 4;

        const [red, green, blue] = getEsdfColor(
          distance,
          minDistance,
          maxDistance,
          state.esdfColormap,
        );
        esdfImage.data[idx] = red;
        esdfImage.data[idx + 1] = green;
        esdfImage.data[idx + 2] = blue;
        esdfImage.data[idx + 3] = 255;
      }
    }

    esdfImageData = esdfImage;
    esdfImageCanvas = document.createElement('canvas');
    esdfImageCanvas.width = costmap.size_x;
    esdfImageCanvas.height = costmap.size_y;
    esdfImageCanvas.getContext('2d').putImageData(esdfImageData, 0, 0);
  }

  function drawMapFrame() {
    if (!state.costmap) {
      return;
    }

    const costmap = state.costmap;
    const maxX = costmap.origin_x + costmap.size_x * costmap.resolution;
    const maxY = costmap.origin_y + costmap.size_y * costmap.resolution;
    const corners = [
      worldToCanvas(costmap.origin_x, costmap.origin_y),
      worldToCanvas(maxX, costmap.origin_y),
      worldToCanvas(maxX, maxY),
      worldToCanvas(costmap.origin_x, maxY),
    ];

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    corners.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
    ctx.closePath();
    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = 'rgba(20, 122, 106, 0.55)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  function drawArrowHead(fromPoint, toPoint, color) {
    const angle = Math.atan2(toPoint.y - fromPoint.y, toPoint.x - fromPoint.x);
    const arrowLength = 12;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(toPoint.x, toPoint.y);
    ctx.lineTo(
      toPoint.x - arrowLength * Math.cos(angle - Math.PI / 6),
      toPoint.y - arrowLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      toPoint.x - arrowLength * Math.cos(angle + Math.PI / 6),
      toPoint.y - arrowLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
  }

  function drawAxesOverlay() {
    if (!state.costmap) {
      return;
    }

    const costmap = state.costmap;
    const worldWidth = costmap.size_x * costmap.resolution;
    const worldHeight = costmap.size_y * costmap.resolution;
    const maxX = costmap.origin_x + worldWidth;
    const maxY = costmap.origin_y + worldHeight;
    const insetPixels = 18;
    const insetWorldX = (insetPixels * costmap.resolution * costmap.size_x) / (canvas.width * state.viewScale);
    const insetWorldY = (insetPixels * costmap.resolution * costmap.size_y) / (canvas.height * state.viewScale);
    const axisOriginX = costmap.origin_x + insetWorldX;
    const axisOriginY = costmap.origin_y + insetWorldY;
    const axisEndX = maxX - insetWorldX;
    const axisEndY = maxY - insetWorldY;
    const origin = worldToCanvas(axisOriginX, axisOriginY);
    const xEnd = worldToCanvas(axisEndX, axisOriginY);
    const yEnd = worldToCanvas(axisOriginX, axisEndY);
    const axisColor = 'rgba(15, 92, 80, 0.92)';
    const tickColor = 'rgba(35, 48, 40, 0.72)';
    const tickStep = Math.max(1, Math.round(Math.max(worldWidth, worldHeight) / 4));
    const tickLength = 7;

    ctx.save();
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(xEnd.x, xEnd.y);
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(yEnd.x, yEnd.y);
    ctx.stroke();

    drawArrowHead(origin, xEnd, axisColor);
    drawArrowHead(origin, yEnd, axisColor);

    ctx.fillStyle = axisColor;
    ctx.beginPath();
    ctx.arc(origin.x, origin.y, 4.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = '600 12px "Avenir Next", sans-serif';
    ctx.fillStyle = tickColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let xValue = costmap.origin_x; xValue <= maxX + 1e-6; xValue += tickStep) {
      const clampedX = Math.min(axisEndX, Math.max(axisOriginX, xValue));
      const point = worldToCanvas(clampedX, axisOriginY);
      ctx.beginPath();
      ctx.moveTo(point.x, point.y - tickLength);
      ctx.lineTo(point.x, point.y + tickLength);
      ctx.strokeStyle = tickColor;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.fillText(`${(xValue - costmap.origin_x).toFixed(0)}m`, point.x, point.y + 10);
    }

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let yValue = costmap.origin_y; yValue <= maxY + 1e-6; yValue += tickStep) {
      const clampedY = Math.min(axisEndY, Math.max(axisOriginY, yValue));
      const point = worldToCanvas(axisOriginX, clampedY);
      ctx.beginPath();
      ctx.moveTo(point.x - tickLength, point.y);
      ctx.lineTo(point.x + tickLength, point.y);
      ctx.strokeStyle = tickColor;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.fillText(`${(yValue - costmap.origin_y).toFixed(0)}m`, point.x - 10, point.y);
    }

    ctx.fillStyle = axisColor;
    ctx.font = '700 13px "Avenir Next", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('O (0, 0)', origin.x + 10, origin.y - 8);
    ctx.fillText('X', xEnd.x - 6, xEnd.y - 10);
    ctx.fillText('Y', yEnd.x + 8, yEnd.y + 16);
    ctx.restore();
  }

  function drawCostmap() {
    if (!state.costmap) {
      return;
    }

    const imageCanvas = state.mapDisplayMode === 'esdf' ? esdfImageCanvas : costmapImageCanvas;
    if (!imageCanvas) {
      return;
    }

    const costmap = state.costmap;
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(
      state.viewScale * (canvas.width / costmap.size_x), 0,
      0, state.viewScale * (canvas.height / costmap.size_y),
      state.viewOffsetX, state.viewOffsetY
    );
    ctx.drawImage(imageCanvas, 0, 0);
    ctx.restore();
  }

  function drawObstacleOverlay() {
    if (!state.costmap || !state.obstacles.length) {
      return;
    }

    state.obstacles.forEach((rect, index) => {
      const bounds = obstacleRectToCanvasBounds(rect);
      const isActive = state.hoverObstacleIndex === index || state.draggingObstacleIndex === index;

      ctx.save();
      ctx.fillStyle = isActive ? 'rgba(20, 122, 106, 0.14)' : 'rgba(47, 52, 64, 0.10)';
      ctx.strokeStyle = isActive ? 'rgba(15, 92, 80, 0.95)' : 'rgba(47, 52, 64, 0.85)';
      ctx.lineWidth = isActive ? 3 : 2;
      ctx.setLineDash(isActive ? [10, 6] : [6, 5]);
      ctx.beginPath();
      ctx.rect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = isActive ? 'rgba(15, 92, 80, 0.95)' : 'rgba(35, 48, 40, 0.82)';
      ctx.font = '700 12px "Avenir Next", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(`Obs ${index + 1}`, bounds.left + 6, bounds.top + 6);
      ctx.restore();
    });
  }

  function drawPath(xs, ys, color, width, drawDots = false) {
    if (!xs || xs.length < 2) {
      return;
    }

    ctx.save();
    ctx.beginPath();
    const startPoint = worldToCanvas(xs[0], ys[0]);
    ctx.moveTo(startPoint.x, startPoint.y);
    for (let idx = 1; idx < xs.length; idx += 1) {
      const point = worldToCanvas(xs[idx], ys[idx]);
      ctx.lineTo(point.x, point.y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();

    if (drawDots) {
      ctx.fillStyle = color;
      for (let idx = 0; idx < xs.length; idx += 1) {
        const point = worldToCanvas(xs[idx], ys[idx]);
        ctx.beginPath();
        ctx.arc(point.x, point.y, Math.max(width + 0.4, 2.2), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  function getSegmentMotionDirection(thetaRad, dx, dy) {
    if (!Number.isFinite(thetaRad)) {
      return 'forward';
    }

    const segmentNorm = Math.hypot(dx, dy);
    if (segmentNorm <= 1e-6) {
      return 'forward';
    }

    const headingX = Math.cos(thetaRad);
    const headingY = Math.sin(thetaRad);
    const dot = dx * headingX + dy * headingY;
    return dot >= 0 ? 'forward' : 'reverse';
  }

  function drawDirectionalSmoothedPath(xs, ys, thetas, width, drawDots = false) {
    if (!xs || xs.length < 2 || !thetas || thetas.length < 2) {
      drawPath(xs, ys, SMOOTHED_FORWARD_COLOR, width, drawDots);
      return;
    }

    ctx.save();
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (let idx = 0; idx < xs.length - 1; idx += 1) {
      const startPoint = worldToCanvas(xs[idx], ys[idx]);
      const endPoint = worldToCanvas(xs[idx + 1], ys[idx + 1]);
      const dx = xs[idx + 1] - xs[idx];
      const dy = ys[idx + 1] - ys[idx];
      const motionDirection = getSegmentMotionDirection(thetas[idx], dx, dy);
      ctx.beginPath();
      ctx.moveTo(startPoint.x, startPoint.y);
      ctx.lineTo(endPoint.x, endPoint.y);
      ctx.strokeStyle = motionDirection === 'reverse' ? SMOOTHED_REVERSE_COLOR : SMOOTHED_FORWARD_COLOR;
      ctx.stroke();
    }

    if (drawDots) {
      for (let idx = 0; idx < xs.length; idx += 1) {
        const point = worldToCanvas(xs[idx], ys[idx]);
        let motionDirection = 'forward';
        if (idx < xs.length - 1) {
          motionDirection = getSegmentMotionDirection(thetas[idx], xs[idx + 1] - xs[idx], ys[idx + 1] - ys[idx]);
        } else if (idx > 0) {
          motionDirection = getSegmentMotionDirection(thetas[idx], xs[idx] - xs[idx - 1], ys[idx] - ys[idx - 1]);
        }
        ctx.beginPath();
        ctx.arc(point.x, point.y, Math.max(width + 0.4, 2.2), 0, Math.PI * 2);
        ctx.fillStyle = motionDirection === 'reverse' ? SMOOTHED_REVERSE_COLOR : SMOOTHED_FORWARD_COLOR;
        ctx.fill();
      }
    }

    ctx.restore();
  }

  function drawMarker(point, fillColor, text) {
    if (!point) {
      return;
    }

    const pixel = worldToCanvas(point.x, point.y);
    ctx.save();
    ctx.beginPath();
    ctx.fillStyle = fillColor;
    ctx.arc(pixel.x, pixel.y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#fffaf0';
    ctx.stroke();
    ctx.fillStyle = '#fffaf0';
    ctx.font = '700 11px "Avenir Next", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, pixel.x, pixel.y + 0.5);

    const headingDeg = text === 'S'
      ? getHeadingValue('start_yaw_deg', DEFAULT_HEADINGS_DEG.start)
      : getHeadingValue('goal_yaw_deg', DEFAULT_HEADINGS_DEG.goal);
    const orientationConstraintEnabled = text === 'S'
      ? getConstraintEnabled('keep_start_orientation', true)
      : getConstraintEnabled('keep_goal_orientation', true);
    const headingRad = headingDeg * Math.PI / 180;
    const arrowTail = {
      x: pixel.x + Math.cos(headingRad) * 12,
      y: pixel.y - Math.sin(headingRad) * 12,
    };
    const arrowTip = {
      x: pixel.x + Math.cos(headingRad) * 24,
      y: pixel.y - Math.sin(headingRad) * 24,
    };
    ctx.strokeStyle = orientationConstraintEnabled ? '#2b71ba' : 'rgba(43, 113, 186, 0.38)';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(arrowTail.x, arrowTail.y);
    ctx.lineTo(arrowTip.x, arrowTip.y);
    ctx.stroke();
    drawArrowHead(arrowTail, arrowTip, orientationConstraintEnabled ? '#2b71ba' : 'rgba(43, 113, 186, 0.38)');

    if (state.hoverMarker === (text === 'S' ? 'start' : 'goal') || state.draggingMarker === (text === 'S' ? 'start' : 'goal')) {
      ctx.strokeStyle = 'rgba(15, 92, 80, 0.95)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(pixel.x, pixel.y, 14, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawHoveredOptimizedPoint() {
    if (!state.hoverOptimizedPoint) {
      return;
    }

    const point = worldToCanvas(state.hoverOptimizedPoint.worldX, state.hoverOptimizedPoint.worldY);
    ctx.save();
    ctx.beginPath();
    ctx.arc(point.x, point.y, 7.5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 250, 240, 0.92)';
    ctx.fill();
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = 'rgba(191, 54, 87, 0.98)';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(point.x, point.y, 12.5, 0, Math.PI * 2);
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = 'rgba(15, 92, 80, 0.88)';
    ctx.stroke();
    ctx.restore();
  }

  function draw() {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#f6efe0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();

    if (!state.costmap) {
      return;
    }

    if (state.layers.costmap) {
      drawCostmap();
    }
    drawObstacleOverlay();
    drawMapFrame();
    if (state.layers.axes) {
      drawAxesOverlay();
    }

    if (state.paths) {
      if (state.layers.astar) {
        drawPath(state.paths.astar_x, state.paths.astar_y, 'rgba(43, 113, 186, 0.5)', 1.6);
      }
      if (state.layers.reference) {
        drawPath(state.paths.ref_x, state.paths.ref_y, 'rgba(217, 122, 43, 0.5)', 2.2, true);
      }
      if (state.layers.smoothed) {
        drawDirectionalSmoothedPath(state.paths.opt_x, state.paths.opt_y, state.paths.opt_theta, 2.8, true);
      }
    }

    drawHoveredOptimizedPoint();

    if (state.layers.markers) {
      drawMarker(state.start, 'rgba(32, 141, 118, 0.5)', 'S');
      drawMarker(state.goal, 'rgba(217, 79, 52, 0.5)', 'G');
    }
  }

  function getParams() {
    const params = {};
    sliders.forEach(id => {
      const input = document.getElementById(id);
      if (!input) {
        return;
      }
      params[id] = parseFloat(input.value);
    });
    if (plannerPenaltySelect) {
      params.planner_penalty = plannerPenaltySelect.value;
    }
    if (footprintModeSelect) {
      params.footprint_mode = footprintModeSelect.value;
    }
    params.keep_start_orientation = getConstraintEnabled('keep_start_orientation', true);
    params.keep_goal_orientation = getConstraintEnabled('keep_goal_orientation', true);
    updateRobotConfigUi();
    return params;
  }

  async function runPlanning({reason = 'manual'} = {}) {
    if (!hasEndpoints()) {
      return;
    }

    if (state.pendingAutoPlanTimer !== null) {
      window.clearTimeout(state.pendingAutoPlanTimer);
      state.pendingAutoPlanTimer = null;
    }

    if (activePlanAbortController) {
      activePlanAbortController.abort();
    }

    const abortController = new AbortController();
    const requestId = ++activePlanRequestId;
    activePlanAbortController = abortController;

    const statusByReason = {
      manual: 'Planning with A* and constrained smoothing…',
      slider: 'Replanning after parameter change…',
      drag: 'Endpoint moved. Replanning…',
      obstacle: 'Obstacle moved. Replanning…',
      initial: 'Computing the default route…',
    };
    setStatus(statusByReason[reason] || statusByReason.manual, '');
    runBtn.disabled = true;

    const payload = {
      start_x: state.start.x,
      start_y: state.start.y,
      goal_x: state.goal.x,
      goal_y: state.goal.y,
      ...getParams(),
    };

    try {
      const response = await fetch('/api/plan', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });
      const data = await response.json();
      if (abortController.signal.aborted || requestId !== activePlanRequestId) {
        return;
      }
      if (!data.success) {
        state.paths = null;
        state.hoverOptimizedPoint = null;
        updateOptimizedPointInspector();
        clearRunInfo();
        setStatus(data.message || 'Planning failed.', 'error');
        draw();
        return;
      }

      state.paths = data;
      state.hoverOptimizedPoint = null;
      if (state.hover) {
        const hoverCanvas = worldToCanvas(state.hover.x, state.hover.y);
        state.hoverOptimizedPoint = findHoveredOptimizedPoint(hoverCanvas.x, hoverCanvas.y);
      }
      updateOptimizedPointInspector();
      updateRunInfo(data);
      setStatus(
        data.smooth_success
          ? `Run complete. A* ${data.astar_time_ms} ms, smoothing ${data.smooth_time_ms} ms.`
          : `A* succeeded in ${data.astar_time_ms} ms, but smoothing failed so the reference path is shown.`,
        data.smooth_success ? 'ok' : 'error'
      );
      draw();
    } catch (error) {
      if (error.name === 'AbortError') {
        return;
      }
      state.paths = null;
      state.hoverOptimizedPoint = null;
      updateOptimizedPointInspector();
      clearRunInfo();
      setStatus(`Network error: ${error.message}`, 'error');
      draw();
    } finally {
      if (activePlanAbortController === abortController) {
        activePlanAbortController = null;
        runBtn.disabled = false;
      }
    }
  }

  async function updateObstacleLayout() {
    const requestId = ++activeObstacleUpdateRequestId;
    setStatus('Obstacle moved. Rebuilding costmap…', '');
    runBtn.disabled = true;

    try {
      const response = await fetch('/api/obstacles', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({obstacle_rects_cells: state.obstacles}),
      });
      const payload = await response.json();
      if (requestId !== activeObstacleUpdateRequestId) {
        return;
      }
      if (!payload.success) {
        setStatus(payload.message || 'Failed to update obstacles.', 'error');
        return;
      }

      state.costmap = payload;
      syncObstaclesFromCostmap(payload);
      buildCostmapImage();
      updateMapInfo(payload);
      updateSelectionInfo();
      draw();
      runPlanning({reason: 'obstacle'});
    } catch (error) {
      setStatus(`Failed to update obstacles: ${error.message}`, 'error');
      runBtn.disabled = false;
    }
  }

  canvas.addEventListener('mousedown', event => {
    const canvasPoint = clientToCanvasPoint(event.clientX, event.clientY);
    const cx = canvasPoint.x;
    const cy = canvasPoint.y;
    const markerName = event.button === 0 ? getMarkerAtCanvasPoint(cx, cy) : null;
    const obstacleIndex = event.button === 0 && !markerName ? getObstacleAtCanvasPoint(cx, cy) : null;
    const shouldPan = !markerName && obstacleIndex === null && (event.button === 0 || event.button === 1 || event.button === 2);
    if (shouldPan) {
      state.dragging = true;
      state.didDrag = false;
      state.dragStartX = cx;
      state.dragStartY = cy;
      state.dragOffsetX = state.viewOffsetX;
      state.dragOffsetY = state.viewOffsetY;
      updateCanvasCursor();
      event.preventDefault();
      return;
    }

    if (event.button === 0 && markerName) {
      cancelPendingPlanning();
      state.draggingMarker = markerName;
      state.didDrag = false;
      updateSelectionInfo();
      updateCanvasCursor();
      event.preventDefault();
      return;
    }

    if (event.button === 0 && obstacleIndex !== null) {
      cancelPendingPlanning();
      const rect = state.obstacles[obstacleIndex];
      const hoverWorld = canvasToWorld(cx, cy);
      const hoverCell = worldToCell(hoverWorld.x, hoverWorld.y);
      state.draggingObstacleIndex = obstacleIndex;
      state.dragObstacleOffset = {
        x: hoverCell.x - rect.x0,
        y: hoverCell.y - rect.y0,
      };
      state.dragObstacleSize = {
        width: rect.x1 - rect.x0,
        height: rect.y1 - rect.y0,
      };
      state.didDrag = false;
      updateSelectionInfo();
      updateCanvasCursor();
      draw();
      event.preventDefault();
    }
  });

  canvas.addEventListener('contextmenu', event => {
    event.preventDefault();
  });

  canvas.addEventListener('mousemove', event => {
    const canvasPoint = clientToCanvasPoint(event.clientX, event.clientY);
    const cx = canvasPoint.x;
    const cy = canvasPoint.y;
    state.hoverCanvasPoint = {x: cx, y: cy};
    state.hover = canvasToWorld(cx, cy);
    state.hoverSample = sampleCostmap(state.hover);
    state.hoverOptimizedPoint = findHoveredOptimizedPoint(cx, cy);
    state.hoverMarker = state.draggingMarker ? state.draggingMarker : getMarkerAtCanvasPoint(cx, cy);
    state.hoverObstacleIndex = state.draggingObstacleIndex !== null || state.hoverMarker
      ? state.draggingObstacleIndex
      : getObstacleAtCanvasPoint(cx, cy);
    updateSelectionInfo();
    updateCanvasCursor();
    updateLoupe();
    updateOptimizedPointInspector();

    if (state.draggingMarker) {
      state[state.draggingMarker] = clampWorldPoint(state.hover);
      state.didDrag = true;
      draw();
      return;
    }

    if (state.draggingObstacleIndex !== null) {
      const hoverCell = worldToCell(state.hover.x, state.hover.y);
      const width = state.dragObstacleSize.width;
      const height = state.dragObstacleSize.height;
      const maxX0 = state.costmap.size_x - width;
      const maxY0 = state.costmap.size_y - height;
      const nextX0 = Math.max(0, Math.min(maxX0, Math.round(hoverCell.x - state.dragObstacleOffset.x)));
      const nextY0 = Math.max(0, Math.min(maxY0, Math.round(hoverCell.y - state.dragObstacleOffset.y)));
      const rect = state.obstacles[state.draggingObstacleIndex];
      rect.x0 = nextX0;
      rect.y0 = nextY0;
      rect.x1 = nextX0 + width;
      rect.y1 = nextY0 + height;
      state.didDrag = true;
      draw();
      return;
    }

    if (state.dragging) {
      const dx = cx - state.dragStartX;
      const dy = cy - state.dragStartY;
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
        state.didDrag = true;
      }
      state.viewOffsetX = state.dragOffsetX + dx;
      state.viewOffsetY = state.dragOffsetY + dy;
      draw();
    }
  });

  canvas.addEventListener('mouseleave', () => {
    state.hover = null;
    state.hoverCanvasPoint = null;
    state.hoverSample = null;
    state.hoverOptimizedPoint = null;
    if (!state.draggingMarker) {
      state.hoverMarker = null;
    }
    if (state.draggingObstacleIndex === null) {
      state.hoverObstacleIndex = null;
    }
    updateSelectionInfo();
    updateCanvasCursor();
    hideLoupe();
    updateOptimizedPointInspector();
  });

  window.addEventListener('mouseup', async () => {
    const draggedMarker = state.draggingMarker;
    const didMoveMarker = Boolean(state.draggingMarker && state.didDrag);
    const didMoveObstacle = state.draggingObstacleIndex !== null && state.didDrag;
    state.dragging = false;
    state.draggingMarker = null;
    state.draggingObstacleIndex = null;
    state.dragObstacleOffset = null;
    state.dragObstacleSize = null;
    state.didDrag = false;
    state.hoverMarker = null;
    state.hoverObstacleIndex = null;
    updateCanvasCursor();

    if (didMoveMarker) {
      setStatus(`${draggedMarker === 'start' ? 'Start' : 'Goal'} moved. Replanning…`, '');
      updateSelectionInfo();
      draw();
      runPlanning({reason: 'drag'});
      return;
    }

    if (didMoveObstacle) {
      updateSelectionInfo();
      draw();
      await updateObstacleLayout();
      return;
    }

    updateSelectionInfo();
  });

  canvas.addEventListener('dblclick', event => {
    event.preventDefault();
    resetView();
    setStatus('View reset to the full map extent.', '');
  });

  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const canvasPoint = clientToCanvasPoint(event.clientX, event.clientY);
    const mouseX = canvasPoint.x;
    const mouseY = canvasPoint.y;
    const factor = event.deltaY < 0 ? 1.1 : 0.9;
    const newScale = Math.min(8.0, Math.max(0.65, state.viewScale * factor));
    state.viewOffsetX = mouseX - (mouseX - state.viewOffsetX) * (newScale / state.viewScale);
    state.viewOffsetY = mouseY - (mouseY - state.viewOffsetY) * (newScale / state.viewScale);
    state.viewScale = newScale;
    updateSelectionInfo();
    draw();
  }, {passive: false});

  runBtn.addEventListener('click', () => runPlanning({reason: 'manual'}));

  clearBtn.addEventListener('click', () => {
    cancelPendingPlanning();
    state.paths = null;
    resetEndpoints();
    state.obstacles = cloneObstacleRects(state.defaultObstacles);
    state.hover = null;
    state.hoverCanvasPoint = null;
    state.hoverSample = null;
    state.hoverOptimizedPoint = null;
    state.hoverMarker = null;
    state.hoverObstacleIndex = null;
    runBtn.disabled = false;
    clearRunInfo();
    setStatus('Scene reset to the default layout. Rebuilding costmap…', '');
    updateSelectionInfo();
    resetView();
    draw();
    hideLoupe();
    updateOptimizedPointInspector();
    updateObstacleLayout();
  });

  resetViewBtn.addEventListener('click', () => {
    resetView();
    setStatus('View reset to the full map extent.', '');
  });

  async function loadCostmap() {
    setStatus('Loading costmap…', '');
    try {
      const response = await fetch('/api/costmap');
      state.costmap = await response.json();
      syncObstaclesFromCostmap(state.costmap);
      buildCostmapImage();
      updateMapInfo(state.costmap);
      resetEndpoints();
      updateSelectionInfo();
      clearRunInfo();
      runBtn.disabled = false;
      resetView();
      hideLoupe();
      setStatus('Costmap loaded. Left-drag endpoints or obstacle rectangles to update the scene, or left-drag empty space to pan.', '');
      runPlanning({reason: 'initial'});
    } catch (error) {
      setStatus(`Failed to load costmap: ${error.message}`, 'error');
    }
  }

  loadCostmap();
});
