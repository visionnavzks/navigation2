// A* + Constrained Smoother — interactive map frontend
document.addEventListener('DOMContentLoaded', () => {
  const formatScientific = value => Number(value).toExponential(1);
  const canvas = document.getElementById('map-canvas');
  const canvasWrap = document.querySelector('.canvas-wrap');
  const ctx = canvas.getContext('2d');
  const curvatureChart = document.getElementById('curvature-chart');
  const dsChart = document.getElementById('ds-chart');
  const dkdsChart = document.getElementById('dkds-chart');
  const loupe = document.getElementById('costmap-loupe');
  const loupeCanvas = document.getElementById('loupe-canvas');
  const loupeCtx = loupeCanvas.getContext('2d');
  const footprintPreviewCanvas = document.getElementById('footprint-preview-canvas');
  const footprintPreviewCtx = footprintPreviewCanvas ? footprintPreviewCanvas.getContext('2d') : null;
  const mapDisplayModeSelect = document.getElementById('map-display-mode');
  const esdfColormapSelect = document.getElementById('esdf-colormap');
  const footprintModeSelect = document.getElementById('footprint_mode');
  const optimizerTypeSelect = document.getElementById('optimizer_type');
  const linearSolverTypeSelect = document.getElementById('linear_solver_type');
  const runBtn = document.getElementById('run-btn');
  const clearBtn = document.getElementById('clear-btn');
  const resetViewBtn = document.getElementById('reset-view-btn');
  const statusMsg = document.getElementById('status-msg');
  const validationDetailsCard = document.getElementById('footprint-validation-details-card');

  const sliderConfig = {
    start_yaw_deg: value => `${Math.round(value)} deg`,
    goal_yaw_deg: value => `${Math.round(value)} deg`,
    planner_penalty_weight: value => Number(value).toFixed(1),
    hinge_loss_threshold_m: value => Number(value).toFixed(2),
    point_robot_radius_m: value => Number(value).toFixed(2),
    robot_length_m: value => Number(value).toFixed(2),
    robot_width_m: value => Number(value).toFixed(2),
    smooth_weight: value => Math.round(value).toLocaleString(),
    costmap_weight: value => Number(value).toFixed(3),
    cusp_costmap_weight: value => Number(value).toFixed(3),
    cusp_zone_length: value => Number(value).toFixed(2),
    distance_weight: value => Number(value).toFixed(1),
    curvature_weight: value => Number(value).toFixed(1),
    curvature_rate_weight: value => Number(value).toFixed(1),
    max_curvature: value => Number(value).toFixed(1),
    reference_spacing_target_m: value => Number(value).toFixed(2),
    max_iterations: value => String(Math.round(value)),
    max_time: value => Number(value).toFixed(1),
    path_downsampling_factor: value => String(Math.round(value)),
    path_upsampling_factor: value => String(Math.round(value)),
  };
  const numericInputConfig = {
    param_tol: value => formatScientific(value),
    fn_tol: value => formatScientific(value),
    gradient_tol: value => formatScientific(value),
  };
  const numericInputs = Object.keys(numericInputConfig);
  const selectParamIds = ['optimizer_type', 'linear_solver_type'];
  const checkboxParamIds = ['optimizer_debug'];

  const sliders = Object.keys(sliderConfig);
  const layerBindings = {
    layer_costmap: 'costmap',
    layer_axes: 'axes',
    layer_markers: 'markers',
    layer_astar: 'astar',
    layer_reference: 'reference',
    layer_smoothed: 'smoothed',
    layer_robot_projection: 'robotProjection',
  };

  const planInfoIds = [
    'info-optimizer', 'info-astar-time', 'info-smooth-time', 'info-astar-pts', 'info-ref-pts', 'info-opt-knots', 'info-opt-pts',
    'info-ref-spacing', 'info-raw-length', 'info-ref-length', 'info-opt-length', 'info-length-delta',
  ];
  const AUTO_REPLAN_DELAY_MS = 220;
  const OPTIMIZED_POINT_HOVER_RADIUS_PX = 11;
  const SMOOTHED_FORWARD_COLOR = 'rgba(191, 54, 87, 0.5)';
  const SMOOTHED_REVERSE_COLOR = 'rgba(43, 113, 186, 0.5)';
  const ROBOT_PROJECTION_FORWARD_STROKE = 'rgba(191, 54, 87, 0.74)';
  const ROBOT_PROJECTION_FORWARD_FILL = 'rgba(191, 54, 87, 0.12)';
  const ROBOT_PROJECTION_REVERSE_STROKE = 'rgba(43, 113, 186, 0.74)';
  const ROBOT_PROJECTION_REVERSE_FILL = 'rgba(43, 113, 186, 0.12)';
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
      robotProjection: false,
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
      if (id === 'hinge_loss_threshold_m' || id === 'point_robot_radius_m' ||
        id === 'robot_length_m' || id === 'robot_width_m') {
        updateRobotConfigUi();
        draw();
      }
    };

    input.addEventListener('input', sync);
    input.addEventListener('input', () => scheduleAutoPlan());
    sync();
  });

  numericInputs.forEach(id => {
    const input = document.getElementById(id);
    const label = document.getElementById('val_' + id);
    if (!input || !label) {
      return;
    }

    const sync = () => {
      const value = parseFloat(input.value);
      if (!Number.isFinite(value)) {
        return;
      }
      label.textContent = numericInputConfig[id](value);
    };

    input.addEventListener('input', sync);
    input.addEventListener('change', () => {
      sync();
      scheduleAutoPlan();
    });
    sync();
  });

  selectParamIds.forEach(id => {
    const input = document.getElementById(id);
    if (!input) {
      return;
    }

    input.addEventListener('change', () => scheduleAutoPlan());
  });

  checkboxParamIds.forEach(id => {
    const input = document.getElementById(id);
    if (!input) {
      return;
    }

    input.addEventListener('change', () => scheduleAutoPlan());
  });

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
      draw();
      scheduleAutoPlan();
    });
  }

  if (optimizerTypeSelect) {
    optimizerTypeSelect.addEventListener('change', () => {
      updateOptimizerUi();
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

  function updateOptimizerUi() {
    const optimizerType = optimizerTypeSelect ? optimizerTypeSelect.value : 'constrained_smoother';
    const isConstrainedSmoother = optimizerType === 'constrained_smoother';

    if (linearSolverTypeSelect) {
      linearSolverTypeSelect.disabled = !isConstrainedSmoother;
    }

    setText(
      'optimizer-mode-hint',
      isConstrainedSmoother
        ? 'Constrained Smoother uses the existing C++ Ceres objective with curvature, cusp, and ESDF obstacle terms.'
        : 'Kinematic Smoother uses the new C++ bicycle-style state optimizer with ESDF obstacle residuals and footprint sampling.'
    );
    setText(
      'linear-solver-hint',
      isConstrainedSmoother
        ? 'Chooses the Ceres linear solver backend used inside each nonlinear iteration.'
        : 'Only used by Constrained Smoother. Kinematic Smoother solves a single packed state vector with a dense backend.'
    );
  }

  function updateRobotConfigUi() {
    const mode = footprintModeSelect ? footprintModeSelect.value : 'capsule';
    const pointRobotRadiusInput = document.getElementById('point_robot_radius_m');
    const pointEnabled = mode === 'point';

    if (pointRobotRadiusInput) {
      pointRobotRadiusInput.disabled = !pointEnabled;
    }

    const config = getRobotFootprintConfig();
    const badgeText = config.mode === 'capsule' ? 'Capsule' : 'Single circle';
    const summaryText = config.mode === 'capsule'
      ? `Planning and smoothing use ${config.localCheckPoints.length} capsule checkpoints with ${config.checkRadiusM.toFixed(2)} m radius. The dashed ${config.lengthM.toFixed(2)} m × ${config.widthM.toFixed(2)} m rectangle is final validation only.`
      : `Planning and smoothing use one ${config.checkRadiusM.toFixed(2)} m check circle. The dashed ${config.lengthM.toFixed(2)} m × ${config.widthM.toFixed(2)} m rectangle still validates the final path.`;

    setText('footprint-preview-badge', badgeText);
    setText(
      'robot-config-summary',
      summaryText
    );
    if (!state.paths?.final_rectangle_validation) {
      setText('footprint-validation-summary', 'Rectangle validation status will appear after each plan.');
      clearValidationFailureDetails();
    }
    drawFootprintPreview();
  }

  updateOptimizerUi();
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

  function formatValidationPathLabel(validatedPath) {
    if (validatedPath === 'smoothed_candidate') {
      return 'Rejected smoothed candidate';
    }
    if (validatedPath === 'reference_fallback') {
      return 'Returned reference path';
    }
    if (validatedPath === 'smoothed_path') {
      return 'Returned smoothed path';
    }
    return '--';
  }

  function formatValidationReason(reason) {
    const reasonLabels = {
      lethal_overlap: 'Lethal obstacle overlap',
      out_of_bounds: 'Footprint leaves map bounds',
      nonfinite_pose: 'Non-finite pose value',
    };
    return reasonLabels[reason] || '--';
  }

  function formatValidationPose(pose) {
    if (!pose || pose.x === null || pose.y === null || pose.x === undefined || pose.y === undefined) {
      return '--';
    }
    return `${Number(pose.x).toFixed(2)}, ${Number(pose.y).toFixed(2)} m`;
  }

  function formatValidationCell(firstFailure) {
    const collisionCell = firstFailure?.collision_cell;
    if (collisionCell) {
      return `${collisionCell.mx}, ${collisionCell.my}`;
    }

    const bounds = firstFailure?.bounding_box_cells;
    if (bounds) {
      return `mx ${bounds.min_mx}..${bounds.max_mx}, my ${bounds.min_my}..${bounds.max_my}`;
    }

    return '--';
  }

  function formatValidationCellWorld(firstFailure) {
    const collisionCell = firstFailure?.collision_cell;
    if (collisionCell && collisionCell.world_x !== null && collisionCell.world_y !== null && collisionCell.world_x !== undefined && collisionCell.world_y !== undefined) {
      return `${Number(collisionCell.world_x).toFixed(2)}, ${Number(collisionCell.world_y).toFixed(2)} m`;
    }

    const bounds = firstFailure?.bounding_box_cells;
    if (bounds) {
      return `bbox: mx ${bounds.min_mx}..${bounds.max_mx}, my ${bounds.min_my}..${bounds.max_my}`;
    }

    return '--';
  }

  function clearValidationFailureDetails() {
    if (validationDetailsCard) {
      validationDetailsCard.hidden = true;
    }
    setText('validation-detail-path', '--');
    setText('validation-detail-code', '--');
    setText('validation-detail-reason', '--');
    setText('validation-detail-pose-index', '--');
    setText('validation-detail-pose-xy', '--');
    setText('validation-detail-pose-yaw', '--');
    setText('validation-detail-cell', '--');
    setText('validation-detail-cell-world', '--');
    setText('validation-detail-message', '--');
  }

  function showValidationFailureDetails(validation) {
    if (!validationDetailsCard || !validation || validation.valid || !validation.first_failure) {
      clearValidationFailureDetails();
      return;
    }

    const firstFailure = validation.first_failure;
    validationDetailsCard.hidden = false;
    setText('validation-detail-path', formatValidationPathLabel(validation.validated_path));
    setText('validation-detail-code', validation.error_code || '--');
    setText('validation-detail-reason', formatValidationReason(firstFailure.reason));
    setText('validation-detail-pose-index', firstFailure.pose?.index ?? '--');
    setText('validation-detail-pose-xy', formatValidationPose(firstFailure.pose));
    setText('validation-detail-pose-yaw', formatRadians(firstFailure.pose?.yaw));
    setText('validation-detail-cell', formatValidationCell(firstFailure));
    setText('validation-detail-cell-world', formatValidationCellWorld(firstFailure));
    setText('validation-detail-message', validation.message || '--');
  }

  function buildCapsuleCenterOffsets(limitX, radius, tolerance) {
    if (limitX <= 1e-6) {
      return [0];
    }

    const maxGapDepth = Math.min(Math.max(tolerance, 1e-3), Math.max(radius * 0.5, 1e-3));
    const minValue = radius * radius - Math.max(radius - maxGapDepth, 0) ** 2;
    const maxSpacing = Math.max(2 * Math.sqrt(Math.max(minValue, 1e-9)), (state.costmap?.resolution || 0.1) * 0.5);
    const intervalCount = Math.max(1, Math.ceil((2 * limitX) / maxSpacing));
    return Array.from({length: intervalCount + 1}, (_, index) => -limitX + ((2 * limitX * index) / intervalCount));
  }

  function buildLocalFootprintPoints(mode, pointRadiusM, lengthM, widthM) {
    if (mode === 'point') {
      return [{x: 0, y: 0}];
    }

    const halfLength = Math.max(lengthM * 0.5, (state.costmap?.resolution || 0.1) * 0.5);
    const checkRadiusM = Math.max(widthM * 0.5, (state.costmap?.resolution || 0.1) * 0.5);
    return buildCapsuleCenterOffsets(halfLength, checkRadiusM, Math.max((state.costmap?.resolution || 0.1) * 0.35, 0.02))
      .map(offsetX => ({x: offsetX, y: 0}));
  }

  function getRobotFootprintConfig(pathData = null) {
    const readValue = (id, fallback) => {
      const input = document.getElementById(id);
      const value = input ? parseFloat(input.value) : fallback;
      return Number.isFinite(value) ? value : fallback;
    };

    const resolution = state.costmap?.resolution || 0.1;
    const mode = pathData?.footprint_mode || (footprintModeSelect ? footprintModeSelect.value : 'capsule');
    const pointRadiusM = Math.max(0, readValue('point_robot_radius_m', 1.0));
    const lengthM = Math.max(resolution, pathData?.robot_length_m ?? readValue('robot_length_m', 0.8));
    const widthM = Math.max(resolution, pathData?.robot_width_m ?? readValue('robot_width_m', 0.5));
    const localCheckPoints = Array.isArray(pathData?.collision_check_points_local) && pathData.collision_check_points_local.length
      ? pathData.collision_check_points_local.map(point => ({x: Number(point.x), y: Number(point.y)}))
      : buildLocalFootprintPoints(mode, pointRadiusM, lengthM, widthM);
    const checkRadiusM = Number.isFinite(pathData?.collision_check_radius_m)
      ? Math.max(0, Number(pathData.collision_check_radius_m))
      : (mode === 'point' ? pointRadiusM : Math.max(widthM * 0.5, resolution * 0.5));

    return {
      mode,
      pointRadiusM,
      checkRadiusM,
      lengthM,
      widthM,
      localCheckPoints,
    };
  }

  function drawFootprintPreview(pathData = null) {
    if (!footprintPreviewCanvas || !footprintPreviewCtx) {
      return;
    }

    const config = getRobotFootprintConfig(pathData);
    const ctx2d = footprintPreviewCtx;
    const width = footprintPreviewCanvas.width;
    const height = footprintPreviewCanvas.height;
    ctx2d.clearRect(0, 0, width, height);

    const maxExtentX = Math.max(config.lengthM * 0.5 + config.checkRadiusM, 0.15);
    const maxExtentY = Math.max(config.widthM * 0.5, config.checkRadiusM, 0.15);
    const scale = 0.78 * Math.min(width / (2 * maxExtentX), height / (2 * maxExtentY));
    const centerX = width * 0.5;
    const centerY = height * 0.52;
    const toPreview = point => ({
      x: centerX + point.x * scale,
      y: centerY - point.y * scale,
    });

    ctx2d.save();
    ctx2d.strokeStyle = 'rgba(20, 122, 106, 0.24)';
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    ctx2d.moveTo(16, centerY);
    ctx2d.lineTo(width - 16, centerY);
    ctx2d.moveTo(centerX, 12);
    ctx2d.lineTo(centerX, height - 12);
    ctx2d.stroke();

    const halfLengthPx = config.lengthM * 0.5 * scale;
    const halfWidthPx = config.widthM * 0.5 * scale;
    ctx2d.setLineDash([6, 4]);
    ctx2d.lineWidth = 1.5;
    ctx2d.strokeStyle = 'rgba(20, 122, 106, 0.88)';
    ctx2d.strokeRect(centerX - halfLengthPx, centerY - halfWidthPx, halfLengthPx * 2, halfWidthPx * 2);
    ctx2d.setLineDash([]);

    ctx2d.fillStyle = 'rgba(217, 122, 43, 0.16)';
    ctx2d.strokeStyle = 'rgba(217, 122, 43, 0.92)';
    ctx2d.lineWidth = 1.35;
    const circleRadiusPx = Math.max(config.checkRadiusM * scale, 2.2);
    config.localCheckPoints.forEach(point => {
      const previewPoint = toPreview(point);
      ctx2d.beginPath();
      ctx2d.arc(previewPoint.x, previewPoint.y, circleRadiusPx, 0, Math.PI * 2);
      ctx2d.fill();
      ctx2d.stroke();
      ctx2d.beginPath();
      ctx2d.fillStyle = 'rgba(90, 48, 12, 0.96)';
      ctx2d.arc(previewPoint.x, previewPoint.y, 2.3, 0, Math.PI * 2);
      ctx2d.fill();
      ctx2d.fillStyle = 'rgba(217, 122, 43, 0.16)';
    });

    ctx2d.fillStyle = 'rgba(35, 48, 40, 0.74)';
    ctx2d.font = '12px "Avenir Next", "Helvetica Neue", sans-serif';
    ctx2d.fillText('+X', width - 30, centerY - 8);
    ctx2d.fillText('+Y', centerX + 8, 24);
    ctx2d.restore();
  }

  function metersToCanvas(distanceM) {
    if (!state.costmap) {
      return 0;
    }

    const pixelsPerMeter = canvas.width / (state.costmap.size_x * state.costmap.resolution);
    return distanceM * pixelsPerMeter * state.viewScale;
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

  function formatApiError(payload, fallbackMessage) {
    const code = payload?.error?.code;
    const message = payload?.message || payload?.error?.message || fallbackMessage;
    if (code) {
      return `[${code}] ${message}`;
    }
    return message;
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
    const segmentArcLengths = [];
    const segmentLengths = [];
    const curvatures = new Array(pointCount).fill(0);
    const dkDs = new Array(pointCount).fill(0);

    for (let idx = 1; idx < pointCount; idx += 1) {
      const segmentLength = Math.hypot(xs[idx] - xs[idx - 1], ys[idx] - ys[idx - 1]);
      segmentLengths.push(segmentLength);
      arcLengths[idx] = arcLengths[idx - 1] + segmentLength;
      segmentArcLengths.push(arcLengths[idx - 1] + segmentLength * 0.5);
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

    for (let idx = 0; idx < pointCount; idx += 1) {
      const prevIndex = Math.max(0, idx - 1);
      const nextIndex = Math.min(pointCount - 1, idx + 1);
      const deltaS = arcLengths[nextIndex] - arcLengths[prevIndex];
      if (nextIndex === prevIndex || deltaS <= 1e-6) {
        dkDs[idx] = 0;
        continue;
      }
      dkDs[idx] = (curvatures[nextIndex] - curvatures[prevIndex]) / deltaS;
    }

    const computeSignedStats = values => {
      if (!values.length) {
        return {
          signedMin: 0,
          signedMax: 0,
          peakAbs: 0,
          meanAbs: 0,
        };
      }

      let signedMin = Number.POSITIVE_INFINITY;
      let signedMax = Number.NEGATIVE_INFINITY;
      let peakAbs = 0;
      let absSum = 0;

      values.forEach(value => {
        signedMin = Math.min(signedMin, value);
        signedMax = Math.max(signedMax, value);
        const absValue = Math.abs(value);
        peakAbs = Math.max(peakAbs, absValue);
        absSum += absValue;
      });

      return {
        signedMin,
        signedMax,
        peakAbs,
        meanAbs: absSum / values.length,
      };
    };

    const computeRangeStats = values => {
      if (!values.length) {
        return {
          min: 0,
          max: 0,
          mean: 0,
        };
      }

      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      let sum = 0;

      values.forEach(value => {
        min = Math.min(min, value);
        max = Math.max(max, value);
        sum += value;
      });

      return {
        min,
        max,
        mean: sum / values.length,
      };
    };

    const curvatureStats = computeSignedStats(curvatures);
    const dkDsStats = computeSignedStats(dkDs);
    const dsStats = computeRangeStats(segmentLengths);

    return {
      arcLengths,
      segmentArcLengths,
      segmentLengths,
      curvatures,
      dkDs,
      curvatureStats,
      dkDsStats,
      dsStats,
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
    setText('curvature-note', 'Run planning to plot curvature, segment spacing, and curvature rate against the optimized path arc length.');

    const chartElements = [curvatureChart, dsChart, dkdsChart].filter(Boolean);
    if (!chartElements.length) {
      return;
    }

    if (!window.Plotly) {
      chartElements.forEach(element => {
        element.textContent = 'Plotly failed to load. Reload the page to render path profiles.';
      });
      return;
    }

    const createEmptyLayout = (height, title) => ({
      height,
      margin: {l: 56, r: 18, t: 16, b: 46},
      paper_bgcolor: 'rgba(255, 250, 240, 0.96)',
      plot_bgcolor: 'rgba(255, 250, 240, 0.96)',
      font: {
        family: '"Avenir Next", "Helvetica Neue", sans-serif',
        color: 'rgba(35, 48, 40, 0.82)',
      },
      title: undefined,
      xaxis: {
        visible: false,
      },
      yaxis: {
        visible: false,
      },
      annotations: [{
        text: title,
        x: 0.5,
        y: 0.5,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: {
          size: 13,
          color: 'rgba(108, 111, 97, 0.85)',
        },
      }],
    });

    const config = {
      displayModeBar: false,
      responsive: true,
    };

    window.Plotly.react(
      curvatureChart,
      [],
      createEmptyLayout(260, 'Curvature chart will appear after a successful plan.'),
      config
    );
    window.Plotly.react(
      dsChart,
      [],
      createEmptyLayout(220, 'Segment spacing ds will appear after a successful plan.'),
      config
    );
    window.Plotly.react(
      dkdsChart,
      [],
      createEmptyLayout(220, 'Curvature rate dk/ds will appear after a successful plan.'),
      config
    );
  }

  function drawCurvatureChart() {
    if (!curvatureChart || !dsChart || !dkdsChart) {
      return;
    }

    const profile = state.curvatureProfile;
    if (!profile || profile.arcLengths.length < 2) {
      clearCurvatureChart();
      return;
    }

    if (!window.Plotly) {
      setText('curvature-state', 'plotly missing');
      setText('curvature-note', 'Plotly failed to load, so the profile charts could not be rendered.');
      [curvatureChart, dsChart, dkdsChart].forEach(element => {
        element.textContent = 'Plotly failed to load. Reload the page to render path profiles.';
      });
      return;
    }

    const plotBackground = 'rgba(255, 250, 240, 0.96)';
    const gridColor = 'rgba(100, 85, 60, 0.16)';
    const axisColor = 'rgba(35, 48, 40, 0.55)';
    const config = {
      displayModeBar: false,
      responsive: true,
    };
    const makeLayout = (height, xTitle, yTitle) => ({
      height,
      margin: {l: 64, r: 18, t: 18, b: 52},
      paper_bgcolor: plotBackground,
      plot_bgcolor: plotBackground,
      font: {
        family: '"Avenir Next", "Helvetica Neue", sans-serif',
        color: 'rgba(35, 48, 40, 0.82)',
      },
      xaxis: {
        title: xTitle,
        gridcolor: gridColor,
        linecolor: axisColor,
        mirror: true,
        zeroline: false,
      },
      yaxis: {
        title: yTitle,
        gridcolor: gridColor,
        linecolor: axisColor,
        mirror: true,
        zerolinecolor: 'rgba(35, 48, 40, 0.35)',
        zerolinewidth: 1,
      },
      showlegend: false,
    });

    const maxCurvatureLimit = parseFloat(document.getElementById('max_curvature')?.value || '0');
    const curvatureLayout = makeLayout(260, 'Arc length s (m)', 'Curvature k (1/m)');
    if (maxCurvatureLimit > 0) {
      curvatureLayout.shapes = [maxCurvatureLimit, -maxCurvatureLimit].map(limit => ({
        type: 'line',
        x0: 0,
        x1: Math.max(profile.totalLength, 1e-6),
        y0: limit,
        y1: limit,
        line: {
          color: 'rgba(217, 122, 43, 0.8)',
          dash: 'dash',
          width: 1.5,
        },
      }));
    }

    window.Plotly.react(
      curvatureChart,
      [{
        x: profile.arcLengths,
        y: profile.curvatures,
        type: 'scatter',
        mode: 'lines',
        line: {color: 'rgba(191, 54, 87, 0.95)', width: 2.5},
        hovertemplate: 's=%{x:.2f} m<br>k=%{y:.3f} 1/m<extra></extra>',
      }],
      curvatureLayout,
      config
    );

    window.Plotly.react(
      dsChart,
      [{
        x: profile.segmentArcLengths,
        y: profile.segmentLengths,
        type: 'scatter',
        mode: 'lines+markers',
        line: {color: 'rgba(20, 122, 106, 0.95)', width: 2.2},
        marker: {size: 6, color: 'rgba(20, 122, 106, 0.95)'},
        hovertemplate: 's=%{x:.2f} m<br>ds=%{y:.3f} m<extra></extra>',
      }],
      makeLayout(220, 'Segment midpoint s (m)', 'Spacing ds (m)'),
      config
    );

    window.Plotly.react(
      dkdsChart,
      [{
        x: profile.arcLengths,
        y: profile.dkDs,
        type: 'scatter',
        mode: 'lines',
        line: {color: 'rgba(217, 122, 43, 0.95)', width: 2.4},
        hovertemplate: 's=%{x:.2f} m<br>dk/ds=%{y:.3f} 1/m^2<extra></extra>',
      }],
      makeLayout(220, 'Arc length s (m)', 'dk/ds (1/m^2)'),
      config
    );

    setText('curvature-state', 'chart ready');
    setText('curvature-peak', formatCurvature(profile.curvatureStats.peakAbs));
    setText('curvature-mean', formatCurvature(profile.curvatureStats.meanAbs));
    setText('curvature-min', formatCurvature(profile.curvatureStats.signedMin));
    setText('curvature-max', formatCurvature(profile.curvatureStats.signedMax));
    setText(
      'curvature-note',
      `Curvature k(s), returned-point spacing ds, and curvature rate dk/ds are estimated from consecutive optimized path samples. Dashed amber lines mark the current Max Curvature limit (${maxCurvatureLimit.toFixed(2)} 1/m). Mean ds: ${profile.dsStats.mean.toFixed(3)} m, peak |dk/ds|: ${profile.dkDsStats.peakAbs.toFixed(3)} 1/m^2.`
    );
  }

  function updateRunInfo(data) {
    state.curvatureProfile = computeCurvatureProfile(data);
    setText('info-optimizer', data.optimizer_label || '--');
    setText('info-astar-time', `${data.astar_time_ms} ms`);
    setText('info-smooth-time', `${data.smooth_time_ms} ms`);
    setText('info-astar-pts', String(data.num_astar_pts));
    setText('info-ref-pts', String(data.num_ref_pts));
    setText('info-opt-knots', String(data.num_opt_knots));
    setText('info-opt-pts', String(data.num_returned_pts ?? data.num_opt_pts));
    setText('info-ref-spacing', formatMeters(data.reference_spacing_target_m));
    setText('info-raw-length', formatMeters(data.raw_path_length_m));
    setText('info-ref-length', formatMeters(data.ref_path_length_m));
    setText('info-opt-length', formatMeters(data.opt_path_length_m));

    const deltaValue = Number(data.opt_vs_ref_delta_m);
    const deltaText = Number.isNaN(deltaValue)
      ? '--'
      : `${deltaValue >= 0 ? '+' : ''}${deltaValue.toFixed(2)} m`;
    setText('info-length-delta', deltaText);

    setText(
      'smooth-state',
      data.smooth_success
        ? `${data.optimizer_label || 'optimizer'} success`
        : `${data.optimizer_label || 'optimizer'} fallback`
    );
    setText(
      'run-note',
      data.smooth_success
        ? `${data.optimizer_label || 'The selected optimizer'} produced the smoothed path. Compare the raw, reference, and smoothed lengths while toggling layers to inspect how the backend changed geometry.`
        : `${data.optimizer_label || 'The selected optimizer'} failed and the reference path is being shown instead. ${data.smooth_message || ''}`.trim()
    );

    const candidateValidation = data.candidate_rectangle_validation;
    const returnedValidation = data.final_rectangle_validation;
    const failureValidation = candidateValidation && !candidateValidation.valid
      ? candidateValidation
      : returnedValidation && !returnedValidation.valid
        ? returnedValidation
        : null;
    if (candidateValidation && !candidateValidation.valid) {
      const candidateCode = candidateValidation.error_code ? ` [${candidateValidation.error_code}]` : '';
      const returnedSummary = !returnedValidation
        ? ''
        : returnedValidation.collision_free
          ? ' Returned reference path rectangle validation passed.'
          : ` Returned reference path rectangle validation also failed${returnedValidation.error_code ? ` [${returnedValidation.error_code}]` : ''}. ${returnedValidation.message || ''}`;
      setText(
        'footprint-validation-summary',
        `Rejected smoothed path${candidateCode}. ${candidateValidation.message || ''}${returnedSummary}`.trim()
      );
    } else if (returnedValidation) {
      const pathLabel = returnedValidation.validated_path === 'reference_fallback'
        ? 'Returned reference path'
        : 'Returned path';
      const statusText = returnedValidation.collision_free
        ? `${pathLabel} rectangle validation passed on all ${data.num_returned_pts ?? data.num_opt_pts ?? 0} pose(s).`
        : `${pathLabel} rectangle validation failed${returnedValidation.error_code ? ` [${returnedValidation.error_code}]` : ''}. ${returnedValidation.message || ''}`.trim();
      setText('footprint-validation-summary', statusText);
    }
    showValidationFailureDetails(failureValidation);

    drawFootprintPreview(data);
    drawCurvatureChart();
  }

  function clearRunInfo() {
    planInfoIds.forEach(id => setText(id, '--'));
    setText('smooth-state', 'idle');
    setText('run-note', 'Set a start and goal to generate path metrics.');
    setText('footprint-validation-summary', 'Rectangle validation status will appear after each plan.');
    clearValidationFailureDetails();
    drawFootprintPreview();
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

  function resolvePoseHeading(xs, ys, thetas, index) {
    if (thetas && Number.isFinite(thetas[index])) {
      return thetas[index];
    }

    if (index < xs.length - 1) {
      return Math.atan2(ys[index + 1] - ys[index], xs[index + 1] - xs[index]);
    }
    if (index > 0) {
      return Math.atan2(ys[index] - ys[index - 1], xs[index] - xs[index - 1]);
    }
    return 0;
  }

  function buildRobotProjectionSampleIndices(xs, ys, config) {
    if (!xs || xs.length === 0) {
      return [];
    }

    const baseExtentM = Math.max(
      config.lengthM,
      config.widthM,
      config.checkRadiusM * 2,
      state.costmap?.resolution || 0.1
    );
    const targetSpacingM = Math.max(baseExtentM * 0.9, 0.75);
    const indices = [0];
    let accumulated = 0;

    for (let idx = 1; idx < xs.length; idx += 1) {
      accumulated += Math.hypot(xs[idx] - xs[idx - 1], ys[idx] - ys[idx - 1]);
      if (accumulated >= targetSpacingM) {
        indices.push(idx);
        accumulated = 0;
      }
    }

    if (indices[indices.length - 1] !== xs.length - 1) {
      indices.push(xs.length - 1);
    }

    return indices;
  }

  function drawRobotProjectionAtPose(worldX, worldY, thetaRad, motionDirection, config, emphasize = false) {
    const pixel = worldToCanvas(worldX, worldY);
    const strokeColor = motionDirection === 'reverse'
      ? ROBOT_PROJECTION_REVERSE_STROKE
      : ROBOT_PROJECTION_FORWARD_STROKE;
    const fillColor = motionDirection === 'reverse'
      ? ROBOT_PROJECTION_REVERSE_FILL
      : ROBOT_PROJECTION_FORWARD_FILL;
    const headingLengthPx = Math.max(metersToCanvas(Math.max(config.lengthM * 0.5, config.checkRadiusM)) + 8, 12);

    ctx.save();
    ctx.translate(pixel.x, pixel.y);
    ctx.rotate(-thetaRad);
    ctx.lineWidth = emphasize ? 1.8 : 1.35;
    ctx.strokeStyle = strokeColor;
    ctx.fillStyle = fillColor;

    const halfLengthPx = metersToCanvas(config.lengthM) * 0.5;
    const halfWidthPx = metersToCanvas(config.widthM) * 0.5;
    if (halfLengthPx > 1 && halfWidthPx > 1) {
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = 'rgba(20, 122, 106, 0.88)';
      ctx.lineWidth = emphasize ? 1.5 : 1.1;
      ctx.strokeRect(-halfLengthPx, -halfWidthPx, halfLengthPx * 2, halfWidthPx * 2);
      ctx.restore();
    }

    const checkRadiusPx = metersToCanvas(config.checkRadiusM);
    config.localCheckPoints.forEach(point => {
      const circleX = metersToCanvas(point.x);
      const circleY = -metersToCanvas(point.y);
      if (checkRadiusPx > 1.2) {
        ctx.beginPath();
        ctx.arc(circleX, circleY, checkRadiusPx, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(circleX, circleY, 2.6, 0, Math.PI * 2);
        ctx.fillStyle = strokeColor;
        ctx.fill();
      }
    });

    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(headingLengthPx, 0);
    ctx.strokeStyle = strokeColor;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(headingLengthPx, 0);
    ctx.lineTo(headingLengthPx - 5.5, -3.5);
    ctx.lineTo(headingLengthPx - 5.5, 3.5);
    ctx.closePath();
    ctx.fillStyle = strokeColor;
    ctx.fill();
    ctx.restore();
  }

  function drawSmoothedRobotProjection(xs, ys, thetas) {
    if (!state.layers.robotProjection || !xs || xs.length < 1) {
      return;
    }

    const config = getRobotFootprintConfig(state.paths);
    const sampleIndices = buildRobotProjectionSampleIndices(xs, ys, config);
    if (!sampleIndices.length) {
      return;
    }

    sampleIndices.forEach((index, sampleIndex) => {
      const thetaRad = resolvePoseHeading(xs, ys, thetas, index);
      let motionDirection = 'forward';
      if (index < xs.length - 1) {
        motionDirection = getSegmentMotionDirection(thetaRad, xs[index + 1] - xs[index], ys[index + 1] - ys[index]);
      } else if (index > 0) {
        motionDirection = getSegmentMotionDirection(thetaRad, xs[index] - xs[index - 1], ys[index] - ys[index - 1]);
      }

      drawRobotProjectionAtPose(
        xs[index],
        ys[index],
        thetaRad,
        motionDirection,
        config,
        sampleIndex === 0 || sampleIndex === sampleIndices.length - 1
      );
    });
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
      if (state.layers.robotProjection) {
        drawSmoothedRobotProjection(state.paths.opt_x, state.paths.opt_y, state.paths.opt_theta);
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
    numericInputs.forEach(id => {
      const input = document.getElementById(id);
      if (!input) {
        return;
      }

      const value = parseFloat(input.value);
      if (Number.isFinite(value)) {
        params[id] = value;
      }
    });
    selectParamIds.forEach(id => {
      const input = document.getElementById(id);
      if (!input) {
        return;
      }
      params[id] = input.value;
    });
    checkboxParamIds.forEach(id => {
      const input = document.getElementById(id);
      if (!input) {
        return;
      }
      params[id] = input.checked;
    });
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
        setStatus(formatApiError(data, 'Planning failed.'), 'error');
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
      const optimizerLabel = data.optimizer_label || 'Optimizer';
      const smoothErrorLabel = data.smooth_error?.code ? ` [${data.smooth_error.code}]` : '';
      setStatus(
        data.smooth_success
          ? `${optimizerLabel} complete. A* ${data.astar_time_ms} ms, smoothing ${data.smooth_time_ms} ms.`
          : `A* succeeded in ${data.astar_time_ms} ms, but ${optimizerLabel} failed${smoothErrorLabel} so the reference path is shown. ${data.smooth_message || ''}`.trim(),
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
        setStatus(formatApiError(payload, 'Failed to update obstacles.'), 'error');
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
