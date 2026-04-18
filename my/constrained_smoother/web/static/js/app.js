// A* + Constrained Smoother — interactive map frontend
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('map-canvas');
  const canvasWrap = document.querySelector('.canvas-wrap');
  const ctx = canvas.getContext('2d');
  const loupe = document.getElementById('costmap-loupe');
  const loupeCanvas = document.getElementById('loupe-canvas');
  const loupeCtx = loupeCanvas.getContext('2d');
  const runBtn = document.getElementById('run-btn');
  const clearBtn = document.getElementById('clear-btn');
  const resetViewBtn = document.getElementById('reset-view-btn');
  const statusMsg = document.getElementById('status-msg');

  const sliderConfig = {
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
  const LOUPE_RADIUS_CELLS = 5;
  const LOUPE_CELL_SIZE = Math.floor(loupeCanvas.width / (LOUPE_RADIUS_CELLS * 2 + 1));
  const DEFAULT_ENDPOINTS = {
    start: {x: 1.0, y: 1.0},
    goal: {x: 18.0, y: 18.0},
  };

  const state = {
    costmap: null,
    start: null,
    goal: null,
    obstacles: [],
    defaultObstacles: [],
    hover: null,
    hoverSample: null,
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
    };

    input.addEventListener('input', sync);
    input.addEventListener('input', () => scheduleAutoPlan());
    sync();
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

  const maxCurvatureInput = document.getElementById('max_curvature');
  if (maxCurvatureInput) {
    maxCurvatureInput.addEventListener('input', syncDerivedParameterInfo);
  }

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

  function formatMeters(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} m`;
  }

  function formatCoord(point) {
    if (!point) {
      return '--';
    }
    return `${point.x.toFixed(2)}, ${point.y.toFixed(2)} m`;
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
    setText('start-coord', formatCoord(state.start));
    setText('goal-coord', formatCoord(state.goal));
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
    setText('loupe-cost-value', 'Cell --');
    setText('loupe-cell-cost', '--');
    setText('loupe-interp-cost', '--');
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

    const interpolate = (x, y) => {
      const shiftedX = x - 0.5;
      const shiftedY = y - 0.5;
      const x0 = Math.floor(shiftedX);
      const y0 = Math.floor(shiftedY);
      const tx = shiftedX - x0;
      const ty = shiftedY - y0;
      const sampleAt = (sx, sy) => {
        const clampedX = Math.max(0, Math.min(state.costmap.size_x - 1, sx));
        const clampedY = Math.max(0, Math.min(state.costmap.size_y - 1, sy));
        return state.costmap.data[clampedY * state.costmap.size_x + clampedX];
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
      interpolatedCost: interpolate(cellX, cellY),
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
    setText('loupe-cost-value', `Cell ${sample.cost}`);
    setText('loupe-cell-cost', String(sample.cost));
    setText('loupe-interp-cost', sample.interpolatedCost.toFixed(2));
    setText('loupe-world', `World: ${sample.worldX.toFixed(2)}, ${sample.worldY.toFixed(2)} m`);
    setText('loupe-cell', `Cell: (${sample.mx}, ${sample.my})`);
    setText('loupe-kind', descriptor.text);
    document.getElementById('loupe-kind')?.setAttribute('data-kind', descriptor.kind);
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
    setText('map-description', meta.description || 'Fixed synthetic costmap used to inspect optimizer behavior.');
    setText('map-kind', meta.name || 'Synthetic field');
  }

  function updateRunInfo(data) {
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
  }

  function clearRunInfo() {
    planInfoIds.forEach(id => setText(id, '--'));
    setText('smooth-state', 'idle');
    setText('run-note', 'Set a start and goal to generate path metrics.');
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
    if (!costmapImageCanvas || !state.costmap) {
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
    ctx.drawImage(costmapImageCanvas, 0, 0);
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

    if (state.hoverMarker === (text === 'S' ? 'start' : 'goal') || state.draggingMarker === (text === 'S' ? 'start' : 'goal')) {
      ctx.strokeStyle = 'rgba(15, 92, 80, 0.95)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(pixel.x, pixel.y, 14, 0, Math.PI * 2);
      ctx.stroke();
    }
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
        drawPath(state.paths.astar_x, state.paths.astar_y, 'rgba(43, 113, 186, 0.72)', 1.6);
      }
      if (state.layers.reference) {
        drawPath(state.paths.ref_x, state.paths.ref_y, '#d97a2b', 2.2, true);
      }
      if (state.layers.smoothed) {
        drawPath(state.paths.opt_x, state.paths.opt_y, '#bf3657', 2.8, true);
      }
    }

    if (state.layers.markers) {
      drawMarker(state.start, '#208d76', 'S');
      drawMarker(state.goal, '#d94f34', 'G');
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
        clearRunInfo();
        setStatus(data.message || 'Planning failed.', 'error');
        draw();
        return;
      }

      state.paths = data;
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
    state.hover = canvasToWorld(cx, cy);
    state.hoverSample = sampleCostmap(state.hover);
    state.hoverMarker = state.draggingMarker ? state.draggingMarker : getMarkerAtCanvasPoint(cx, cy);
    state.hoverObstacleIndex = state.draggingObstacleIndex !== null || state.hoverMarker
      ? state.draggingObstacleIndex
      : getObstacleAtCanvasPoint(cx, cy);
    updateSelectionInfo();
    updateCanvasCursor();
    updateLoupe();

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
    state.hoverSample = null;
    if (!state.draggingMarker) {
      state.hoverMarker = null;
    }
    if (state.draggingObstacleIndex === null) {
      state.hoverObstacleIndex = null;
    }
    updateSelectionInfo();
    updateCanvasCursor();
    hideLoupe();
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
    state.hoverSample = null;
    state.hoverMarker = null;
    state.hoverObstacleIndex = null;
    runBtn.disabled = false;
    clearRunInfo();
    setStatus('Scene reset to the default layout. Rebuilding costmap…', '');
    updateSelectionInfo();
    resetView();
    draw();
    hideLoupe();
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
