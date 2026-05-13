// A* + Constrained Smoother — interactive map frontend
document.addEventListener('DOMContentLoaded', () => {
  const formatScientific = value => Number(value).toExponential(1);
  const LANGUAGE_STORAGE_KEY = 'constrained-smoother-ui-language';
  const SUPPORTED_LANGUAGES = ['en', 'zh'];
  const canvas = document.getElementById('map-canvas');
  const canvasWrap = document.querySelector('.canvas-wrap');
  const baseCanvasPixelSize = Math.max(1, Number(canvas?.getAttribute('width')) || 800);
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
  const languageSwitch = document.getElementById('language-switch');
  const runBtn = document.getElementById('run-btn');
  const clearBtn = document.getElementById('clear-btn');
  const resetViewBtn = document.getElementById('reset-view-btn');
  const statusMsg = document.getElementById('status-msg');
  const validationDetailsCard = document.getElementById('footprint-validation-details-card');
  const kinematicDiagnosticsCard = document.getElementById('kinematic-diagnostics-card');
  const chartElements = [curvatureChart, dsChart, dkdsChart].filter(Boolean);
  const CHART_HEIGHTS = {
    primary: 190,
    secondary: 136,
  };

  const zhStaticTranslations = {
    'hero.eyebrow': '独立版 Nav2 约束平滑器',
    'hero.title': 'A* + 约束平滑实验台',
    'hero.subtitle': '直接查看合成代价地图，对比原始路径与优化路径，并在接入完整 Nav2 插件前理解每个求解参数会改变什么。',
    'language.label': '语言',
    'hero.summary.mapLabel': '地图',
    'hero.summary.mapCopy': '合成障碍场',
    'hero.summary.resolutionLabel': '分辨率',
    'hero.summary.resolutionCopy': '每格对应的米数',
    'hero.summary.interactionLabel': '交互',
    'session.title': '会话',
    'session.start': '起点',
    'session.goal': '终点',
    'session.cursor': '光标',
    'session.zoom': '缩放',
    'session.gesture': '当前操作',
    'session.startHeading': '起点朝向',
    'session.goalHeading': '终点朝向',
    'session.startConstraint': '起点约束',
    'session.goalConstraint': '终点约束',
    'session.enableStartConstraint': '启用起点朝向约束',
    'session.enableGoalConstraint': '启用终点朝向约束',
    'session.startHeadingLabel': '起点朝向: <span id="val_start_yaw_deg">45 deg</span>',
    'session.startHeadingHint': '设置平滑时起点位姿使用的世界坐标系朝向约束。',
    'session.goalHeadingLabel': '终点朝向: <span id="val_goal_yaw_deg">45 deg</span>',
    'session.goalHeadingHint': '设置平滑时终点位姿使用的世界坐标系朝向约束。',
    'session.goalLongitudinalToleranceLabel': '终点纵向容差 (m): <span id="val_goal_longitudinal_tolerance_m">0.00</span>',
    'session.goalLongitudinalToleranceHint': '允许终点在目标坐标系前后方向内先自由滑动，超出带宽后才触发 hinge 惩罚。若纵向和横向容差都为 0，则终点位置保持固定。',
    'session.goalLateralToleranceLabel': '终点横向容差 (m): <span id="val_goal_lateral_tolerance_m">0.00</span>',
    'session.goalLateralToleranceHint': '允许终点在目标坐标系侧向内先自由偏移，超出带宽后才触发 hinge 惩罚。这就是文档里的终点位置带宽残差。',
    'session.reversingHint': '当前 API 已支持运动学平滑器的 <strong>reversing_enabled</strong>，但 Web Lab 仍以路径点自带的方向符号作为倒车语义来源，因此这个兼容开关依旧不在界面中显示。',
    'session.knownLimitation': '已知限制：当前独立版平滑器不会单独优化转向状态。它先优化 <strong>x/y</strong> 几何，再根据局部切线重建 <strong>yaw</strong>，因此这里的尖点更像几何方向切换，而不是机器人原地不动、只改变转向角的真实停转机动。',
    'session.mapNote': '地图现在显示世界坐标系叠加层，原点位于左下角，<strong>X</strong> 向右增大，<strong>Y</strong> 向上增大。可左键拖拽 <strong>起点</strong>、<strong>终点</strong> 或任意描边障碍块来编辑场景；左键拖拽空白画布可平移；在画布任意处双击可恢复整图视角。滑块变动仍会自动重新规划。',
    'map.title': '地图概览',
    'map.grid': '栅格',
    'map.world': '世界尺寸',
    'map.resolution': '分辨率',
    'map.origin': '原点',
    'map.obstacleBlocks': '障碍块',
    'map.inflationRadius': '膨胀半径',
    'map.freeCells': '空闲栅格',
    'map.inflatedCells': '膨胀栅格',
    'map.lethalCells': '致命栅格',
    'loupe.title': '光标检查器',
    'loupe.liveHover': '实时悬停',
    'loupe.cellCost': '栅格代价值',
    'loupe.esdfDistance': 'ESDF 距离',
    'weights.title': '平滑权重',
    'weights.rawWeightNote': '这些滑块显示的是原始权重，后端会先取 sqrt，再写入求解器里的 *_sqrt 参数；约束平滑器使用 smooth_weight 和几何曲率权重，运动学平滑器使用 model_weight 和独立的运动学曲率权重。',
    'weights.smoothWeightLabel': '平滑权重: <span id="val_smooth_weight">20</span>',
    'weights.smoothWeightHint': '这是约束平滑器三点几何平滑残差的原始权重。调高后，相邻线段会更均匀、更少锯齿，但它并不直接强制自行车模型状态转移。',
    'weights.modelWeightLabel': '模型权重: <span id="val_model_weight">20</span>',
    'weights.modelWeightHint': '这是运动学状态转移一致性残差的原始权重。调高后，每一步状态转移都会更贴近自行车模型预测，而不只是“看起来更平滑”。',
    'weights.obstacleWeightLabel': '障碍权重: <span id="val_costmap_weight">1.000</span>',
    'weights.obstacleWeightHint': '缩放平滑器使用的基于 ESDF 的障碍惩罚。值越大，路径越会被推离障碍物。',
    'weights.cuspObstacleWeightLabel': '尖点障碍权重: <span id="val_cusp_costmap_weight">3.000</span>',
    'weights.cuspObstacleWeightHint': '在尖点邻域覆盖默认障碍权重，使方向切换区域能更强地远离障碍物。',
    'weights.cuspZoneLengthLabel': '尖点区域长度 (m): <span id="val_cusp_zone_length">2.50</span>',
    'weights.cuspZoneLengthHint': '设置方向切换前后尖点障碍权重渐变生效的完整弧长范围。',
    'weights.distanceWeightLabel': '距离权重: <span id="val_distance_weight">0.0</span>',
    'weights.distanceWeightHint': '当你不希望出现大绕路时，用它让优化结果更贴近 A* 参考路径。',
    'weights.curvatureWeightLabel': '曲率权重: <span id="val_curvature_weight">30.0</span>',
    'weights.curvatureWeightHint': '抑制高曲率转弯，尤其是靠近障碍物角点时。',
    'weights.curvatureRateWeightLabel': '曲率变化率权重: <span id="val_curvature_rate_weight">5.0</span>',
    'weights.curvatureRateWeightHint': '使用四点 D3 有限差分惩罚。调高它可以抑制曲率突变，但不会替代最大曲率约束。',
    'weights.kinematicCurvatureWeightLabel': '运动学曲率权重: <span id="val_kinematic_curvature_weight">30.0</span>',
    'weights.kinematicCurvatureWeightHint': '惩罚运动学平滑器里的显式 kappa 状态。调高后会更偏向较小的平均转向曲率，而不只是减少超过最大曲率阈值的片段。',
    'weights.kinematicCurvatureRateWeightLabel': '运动学曲率变化率权重: <span id="val_kinematic_curvature_rate_weight">5.0</span>',
    'weights.kinematicCurvatureRateWeightHint': '惩罚相邻状态之间显式 kappa 的变化率。这和几何平滑器里的四点 D3 代理项不是同一个残差。',
    'weights.maxCurvatureLabel': '最大曲率 (1/m): <span id="val_max_curvature">2.5</span>',
    'weights.maxCurvatureHint': '限制转弯曲率。值越小，最小转弯半径越大。',
    'planner.title': '规划器',
    'planner.penaltyWeightLabel': 'A* 惩罚权重: <span id="val_planner_penalty_weight">1.0</span>',
    'planner.penaltyWeightHint': '缩放共享二次铰链损失下 A* 对低净空栅格的绕行强度。它只影响规划器，不影响平滑器的障碍权重。',
    'planner.hingeThresholdLabel': '铰链损失阈值 (m): <span id="val_hinge_loss_threshold_m">0.50</span>',
    'planner.hingeThresholdHint': 'C++ A* 规划器与约束平滑器共用的铰链边界。ESDF 距离超过该阈值后不再产生惩罚。点机器人模式下，有效阈值等于该值加上机器人半径。',
    'robot.title': '机器人',
    'robot.footprintModel': '足迹模型',
    'robot.footprintCapsule': '胶囊检查点',
    'robot.footprintPoint': '单圆检查',
    'robot.footprintHint': '规划和平滑现在共用同一套“检查点 + 半径”模型。矩形只保留给最终路径验证。',
    'robot.singleCircleRadiusLabel': '单圆半径 (m): <span id="val_point_robot_radius_m">1.00</span>',
    'robot.singleCircleRadiusHint': '仅在单圆模式下使用。胶囊模式会直接从机器人宽度推导圆半径。',
    'robot.lengthLabel': '机器人长度 (m): <span id="val_robot_length_m">0.80</span>',
    'robot.lengthHint': '用于构造胶囊检查点中心，并用真实矩形对最终路径做验证。',
    'robot.widthLabel': '机器人宽度 (m): <span id="val_robot_width_m">0.50</span>',
    'robot.widthHint': '胶囊半径默认取该宽度的一半。最终矩形验证也会使用同一宽度。',
    'robot.previewTitle': '足迹预览',
    'robot.validationDetails': '验证失败详情',
    'robot.failureCode': '失败码',
    'robot.failureReason': '失败原因',
    'robot.poseIndex': '位姿索引',
    'robot.poseXY': '位姿 XY',
    'robot.poseHeading': '位姿朝向',
    'robot.collisionBounds': '碰撞 / 越界',
    'robot.cellWorldContext': '栅格 / 边界世界上下文',
    'solver.title': '求解器',
    'solver.backend': '优化后端',
    'solver.constrained': '约束平滑器',
    'solver.kinematic': '运动学平滑器',
    'solver.linearSolver': '线性求解器',
    'solver.debugLogging': '在 Flask 服务端启用逐迭代求解日志',
    'solver.referenceSpacingLabel': '参考路径目标间距 (m): <span id="val_reference_spacing_target_m">0.30</span>',
    'solver.referenceSpacingHint': '控制在成为参考路径 / 优化器输入前，对稠密 A* 路径做多强的降采样。间距越大，参考点通常越少。',
    'solver.maxIterationsLabel': '最大迭代次数: <span id="val_max_iterations">50</span>',
    'solver.maxIterationsHint': '优化器停止前允许的迭代上限。',
    'solver.maxSolverTimeLabel': '最大求解时间 (s): <span id="val_max_time">10.0</span>',
    'solver.maxSolverTimeHint': '非线性求解的墙钟时间上限。达到该限制或迭代上限后停止。',
    'solver.parameterToleranceLabel': '参数容差: <span id="val_param_tol">1.0e-8</span>',
    'solver.parameterToleranceHint': '当相邻迭代的参数更新足够小时停止。',
    'solver.functionToleranceLabel': '函数容差: <span id="val_fn_tol">1.0e-6</span>',
    'solver.functionToleranceHint': '当目标函数改善小于该阈值时停止。',
    'solver.gradientToleranceLabel': '梯度容差: <span id="val_gradient_tol">1.0e-10</span>',
    'solver.gradientToleranceHint': '当梯度范数足够小、接近驻点时停止。',
    'solver.downsamplingFactorLabel': '降采样因子: <span id="val_path_downsampling_factor">1</span>',
    'solver.downsamplingFactorHint': '在优化前丢弃中间参考点，减少问题规模。',
    'solver.upsamplingFactorLabel': '上采样因子: <span id="val_path_upsampling_factor">1</span>',
    'solver.upsamplingFactorHint': '在优化后重新插入点，恢复用于检查的路径密度。',
    'layers.title': '图层',
    'layers.toggleVisibility': '切换显示',
    'layers.costmap': '代价地图',
    'layers.costmapHint': '空闲、膨胀与致命栅格',
    'layers.mapAxes': '地图坐标轴',
    'layers.mapAxesHint': '世界坐标叠加层',
    'layers.startGoal': '起点 / 终点',
    'layers.startGoalHint': '已选择的端点',
    'layers.astarRawPath': 'A* 原始路径',
    'layers.astarRawPathHint': '稠密的栅格连通规划结果',
    'layers.referencePath': '参考路径',
    'layers.referencePathHint': '降采样后的优化器输入',
    'layers.smoothedPath': '平滑路径',
    'layers.smoothedPathHint': 'Ceres 输出，按前进 / 倒车方向着色',
    'layers.rejectedSmoothedPath': '失败的平滑路径',
    'layers.rejectedSmoothedPathHint': '被后验证拒绝的候选结果，使用警示虚线样式显示',
    'layers.robotProjection': '机器人投影',
    'layers.robotProjectionHint': '沿平滑路径扫过的检查圆与虚线矩形验证轮廓',
    'run.title': '运行统计',
    'run.optimizer': '优化器',
    'run.astarTime': 'A* 时间',
    'run.smoothTime': '平滑时间',
    'run.astarPoints': 'A* 点数',
    'run.referencePoints': '参考点数',
    'run.optimizationKnots': '优化结点数',
    'run.returnedPathPoints': '返回路径点数',
    'run.refSpacingTarget': '参考间距目标',
    'run.rawLength': '原始长度',
    'run.referenceLength': '参考长度',
    'run.optimizedLength': '优化长度',
    'run.optMinusRef': '优化 - 参考',
    'diagnostics.title': '诊断',
    'diagnostics.pending': '当某次运行产生验证或运动学详情时，它们会显示在这里。',
    'run.kinematicDetails': '运动学详情',
    'run.goalSegmentError': '终端段方向误差',
    'run.goalTolerance': '允许容差',
    'run.goalExpected': '期望终点方向',
    'run.goalActualSegment': '实际终端段方向',
    'run.goalPoseHeading': '终端位姿朝向',
    'run.goalPoseError': '终端位姿误差',
    'run.kinematicMaxIterations': '最大迭代次数',
    'run.kinematicMaxTime': '最大求解时间',
    'run.kinematicMaxCurvature': '最大曲率',
    'run.kinematicCurvatureRateWeight': '曲率变化率权重',
    'run.kinematicResampling': '重采样',
    'run.kinematicCeresTolerances': 'Ceres 容差',
    'toolbar.runPlanning': '执行规划',
    'toolbar.resetScene': '重置场景',
    'toolbar.resetView': '重置视图',
    'toolbar.display': '显示',
    'toolbar.originalCostmap': '原始代价地图',
    'toolbar.esdfColormap': 'ESDF 配色',
    'toolbar.diverging': '发散色图',
    'toolbar.world': '世界尺寸',
    'toolbar.inflation': '膨胀',
    'popup.title': '优化点',
    'popup.role': '角色',
    'popup.world': '世界坐标',
    'popup.poseHeading': '位姿朝向',
    'popup.pathTangent': '路径切线',
    'popup.arcLength': '弧长',
    'popup.prevSegment': '上一段',
    'popup.nextSegment': '下一段',
    'popup.turnAngle': '转角',
    'popup.approxCurvature': '近似曲率',
    'popup.esdf': 'ESDF',
    'popup.cellCost': '栅格代价',
    'popup.cursorOffset': '光标偏移',
    'curvature.title': '优化路径剖面',
    'curvature.curvatureKs': '曲率 k(s)',
    'curvature.segmentSpacing': '段间距 ds',
    'curvature.curvatureRate': '曲率变化率 dk/ds',
    'curvature.peak': '峰值 |曲率|',
    'curvature.mean': '平均 |曲率|',
    'curvature.signedMin': '有符号最小值',
    'curvature.signedMax': '有符号最大值',
    'footer.layersHint': '可在左侧面板单独切换各图层，以分别查看代价地图、稠密 A* 路径、降采样参考路径、最终优化轨迹，以及沿该轨迹采样得到的机器人投影。坐标轴叠加层会一直保留，以便在检查轨迹时保持世界坐标可读。',
    'footer.sceneHint': '这张合成地图支持直接编辑场景：拖拽端点标记或描边障碍矩形，左键拖拽空白处可移动相机，双击或使用 <strong>重置视图</strong> 可返回整图视角。',
  };

  const messages = {
    en: {
      'document.title': 'A* + Constrained Smoother Demo',
      'unit.degree': 'deg',
      'unit.meter': 'm',
      'unit.radian': 'rad',
      'unit.curvature': '1/m',
      'unit.curvatureRate': '1/m^2',
      'unit.metersPerCell': 'm/cell',
      'unit.cells': 'cells',
      'unit.ms': 'ms',
      'unit.second': 's',
      'common.enabled': 'Enabled',
      'common.disabled': 'Disabled',
      'common.idle': 'idle',
      'common.outsideMap': 'Outside map',
      'common.chartReady': 'chart ready',
      'common.plotlyMissing': 'plotly missing',
      'status.parameterChangedReplanning': 'Parameter changed. Replanning…',
      'status.manualPlanning': 'Planning with A* and constrained smoothing…',
      'status.sliderPlanning': 'Replanning after parameter change…',
      'status.dragPlanning': 'Endpoint moved. Replanning…',
      'status.obstaclePlanning': 'Obstacle moved. Replanning…',
      'status.initialPlanning': 'Computing the default route…',
      'status.planningFailed': 'Planning failed.',
      'status.networkError': 'Network error: {message}',
      'status.obstacleRebuilding': 'Obstacle moved. Rebuilding costmap…',
      'status.obstacleUpdateFailed': 'Failed to update obstacles.',
      'status.obstacleUpdateError': 'Failed to update obstacles: {message}',
      'status.markerMoved': '{marker} moved. Replanning…',
      'status.viewReset': 'View reset to the full map extent.',
      'status.sceneReset': 'Scene reset to the default layout. Rebuilding costmap…',
      'status.loadingCostmap': 'Loading costmap…',
      'status.costmapLoaded': 'Costmap loaded. Left-drag endpoints or obstacle rectangles to update the scene, or left-drag empty space to pan.',
      'status.costmapLoadFailed': 'Failed to load costmap: {message}',
      'status.planSuccess': '{optimizerLabel} complete. A* {astarTimeMs} ms, smoothing {smoothTimeMs} ms. {statsSummary}',
      'status.pathStats': 'Returned {pointCount} pts, total {pathLength}, mean spacing {meanSpacing}.',
      'status.planFallback': 'A* succeeded in {astarTimeMs} ms, but {optimizerLabel} failed{errorCodeSuffix} so the reference path is shown. {smoothMessage}',
      'status.planRejectedShown': 'A* succeeded in {astarTimeMs} ms, but {optimizerLabel} failed{errorCodeSuffix}. The rejected smoothed candidate is still shown. {smoothMessage}',
      'selection.ready': 'Markers ready',
      'selection.dragMarker': 'Dragging marker',
      'selection.dragObstacle': 'Dragging obstacle',
      'selection.dragScene': 'Drag scene',
      'selection.panning': 'Panning view',
      'selection.leftDrag': 'Left-drag',
      'optimizer.constrained': 'Constrained Smoother',
      'optimizer.kinematic': 'Kinematic Smoother',
      'session.goalLongitudinalToleranceLabel': 'Goal Longitudinal Tolerance (m): <span id="val_goal_longitudinal_tolerance_m">0.00</span>',
      'session.goalLongitudinalToleranceHint': 'Allows the final point to slide forward or backward inside the goal frame before the hinge penalty turns on. Set both goal tolerances to zero to keep the goal position fixed.',
      'session.goalLateralToleranceLabel': 'Goal Lateral Tolerance (m): <span id="val_goal_lateral_tolerance_m">0.00</span>',
      'session.goalLateralToleranceHint': 'Allows the final point to drift sideways inside the goal frame before the hinge penalty turns on. This exposes the goal position bandwidth residual in the web lab.',
      'weights.modelWeightLabel': 'Model Weight: <span id="val_model_weight">20</span>',
      'weights.modelWeightHint': 'Raw weight for the kinematic state-transition consistency residuals. Higher values keep each state transition closer to the predicted bicycle-model motion.',
      'optimizer.mode.constrained': 'Constrained Smoother uses the existing C++ Ceres objective with curvature, cusp, and ESDF obstacle terms.',
      'optimizer.mode.kinematic': 'Kinematic Smoother uses the new C++ bicycle-style state optimizer with ESDF obstacle residuals and footprint sampling.',
      'optimizer.linear.constrained': 'Chooses the Ceres linear solver backend used inside each nonlinear iteration.',
      'optimizer.linear.kinematic': 'Only used by Constrained Smoother. Kinematic Smoother solves a single packed state vector with a dense backend.',
      'robot.badge.capsule': 'Capsule',
      'robot.badge.point': 'Single circle',
      'robot.summary.previewPending': 'Capsule checkpoints are shown in amber; the dashed rectangle is final validation only.',
      'robot.validation.pending': 'Rectangle validation status will appear after each plan.',
      'validation.path.smoothed_candidate': 'Rejected smoothed candidate',
      'validation.path.reference_fallback': 'Returned reference path',
      'validation.path.smoothed_path': 'Returned smoothed path',
      'layers.rejectedSmoothedPath': 'Failed smoothed path',
      'layers.rejectedSmoothedPathHint': 'Rejected candidate, shown with a warning dashed style',
      'validation.reason.lethal_overlap': 'Lethal obstacle overlap',
      'validation.reason.out_of_bounds': 'Footprint leaves map bounds',
      'validation.reason.nonfinite_pose': 'Non-finite pose value',
      'loupe.esdfEmpty': 'ESDF --',
      'loupe.worldEmpty': 'World: --',
      'loupe.cellEmpty': 'Cell: --',
      'map.kind.default': 'Synthetic field',
      'map.description.default': 'Fixed synthetic obstacle map used to inspect ESDF-based planner and smoother behavior.',
      'popup.note.default': 'Hover a point on the rose smoothed path to inspect its geometry, heading, local clearance, and segment context.',
      'popup.role.startEndpoint': 'Start endpoint',
      'popup.role.goalEndpoint': 'Goal endpoint',
      'popup.role.startAnchor': 'Start anchor',
      'popup.role.goalAnchor': 'Goal anchor',
      'popup.role.interiorPoint': 'Interior point',
      'curvature.note.pending': 'Run planning to plot curvature, segment spacing, and curvature rate against the optimized path arc length.',
      'curvature.note.plotlyMissing': 'Plotly failed to load, so the profile charts could not be rendered.',
      'curvature.note.plotlyReload': 'Plotly failed to load. Reload the page to render path profiles.',
      'curvature.empty.curvature': 'Curvature chart will appear after a successful plan.',
      'curvature.empty.spacing': 'Segment spacing ds will appear after a successful plan.',
      'curvature.empty.rate': 'Curvature rate dk/ds will appear after a successful plan.',
      'curvature.axis.arcLength': 'Arc length s (m)',
      'curvature.axis.curvature': 'Curvature k (1/m)',
      'curvature.axis.segmentMidpoint': 'Segment midpoint s (m)',
      'curvature.axis.spacing': 'Spacing ds (m)',
      'curvature.axis.rate': 'dk/ds (1/m^2)',
      'run.note.success': '{optimizerLabel} produced the smoothed path. Compare the raw, reference, and smoothed lengths while toggling layers to inspect how the backend changed geometry.',
      'run.note.fallback': '{optimizerLabel} failed and the reference path is being shown instead. {smoothMessage}',
      'run.note.rejected': '{optimizerLabel} failed validation, but the rejected smoothed candidate is still being shown for inspection. {smoothMessage}',
      'diagnostics.title': 'Diagnostics',
      'diagnostics.pending': 'Validation and kinematic details appear here when a run exposes them.',
      'run.kinematicDetails': 'Kinematic Details',
      'run.goalSegmentError': 'Goal Segment Error',
      'run.goalTolerance': 'Goal Tolerance',
      'run.goalExpected': 'Expected Goal Heading',
      'run.goalActualSegment': 'Actual Terminal Segment',
      'run.goalPoseHeading': 'Terminal Pose Heading',
      'run.goalPoseError': 'Terminal Pose Error',
      'run.kinematicMaxIterations': 'Max Iterations',
      'run.kinematicMaxTime': 'Max Solver Time',
      'run.kinematicMaxCurvature': 'Max Curvature',
      'run.kinematicCurvatureRateWeight': 'Curvature Rate Weight',
      'run.kinematicResampling': 'Resampling',
      'run.kinematicCeresTolerances': 'Ceres Tolerances',
      'run.pipeline.pending': 'Pipeline status will appear after each run.',
      'run.pipeline.summary': 'Pipeline: {summary}',
      'run.smoothState.success': '{optimizerLabel} success',
      'run.smoothState.fallback': '{optimizerLabel} fallback',
      'run.smoothState.rejected': '{optimizerLabel} rejected candidate shown',
      'run.stage.status.ok': 'ok',
      'run.stage.status.error': 'error',
      'run.stage.status.fallback': 'fallback',
      'run.stage.label.validate': 'Rectangle Validate',
      'run.stage.label.web': 'Web',
      'descriptor.outsideMap': 'Outside map',
      'descriptor.unknownSpace': 'Unknown space',
      'descriptor.lethalObstacle': 'Lethal obstacle',
      'descriptor.inscribedObstacle': 'Inscribed inflated obstacle',
      'descriptor.inflatedCost': 'Inflated cost',
      'descriptor.freeSpace': 'Free space',
      'marker.start': 'Start',
      'marker.goal': 'Goal',
      'canvas.obstacleLabel': 'Obs {index}',
      'derived.minTurnRadius': 'Minimum turning radius: {value}',
      'derived.minTurnRadiusEmpty': 'Minimum turning radius: --'
    },
    zh: {
      'document.title': 'A* + 约束平滑演示',
      'unit.degree': '度',
      'unit.meter': '米',
      'unit.radian': '弧度',
      'unit.curvature': '1/米',
      'unit.curvatureRate': '1/米^2',
      'unit.metersPerCell': '米/格',
      'unit.cells': '格',
      'unit.ms': '毫秒',
      'unit.second': '秒',
      'common.enabled': '已启用',
      'common.disabled': '已禁用',
      'common.idle': '空闲',
      'common.outsideMap': '地图外',
      'common.chartReady': '图表就绪',
      'common.plotlyMissing': '缺少 Plotly',
      'status.parameterChangedReplanning': '参数已变化，正在重新规划…',
      'status.manualPlanning': '正在执行 A* 与约束平滑规划…',
      'status.sliderPlanning': '参数变更后重新规划中…',
      'status.dragPlanning': '端点已移动，正在重新规划…',
      'status.obstaclePlanning': '障碍已移动，正在重新规划…',
      'status.initialPlanning': '正在计算默认路径…',
      'status.planningFailed': '规划失败。',
      'status.networkError': '网络错误：{message}',
      'status.obstacleRebuilding': '障碍已移动，正在重建代价地图…',
      'status.obstacleUpdateFailed': '更新障碍失败。',
      'status.obstacleUpdateError': '更新障碍失败：{message}',
      'status.markerMoved': '{marker}已移动，正在重新规划…',
      'status.viewReset': '视图已恢复到整图范围。',
      'status.sceneReset': '场景已重置为默认布局，正在重建代价地图…',
      'status.loadingCostmap': '正在加载代价地图…',
      'status.costmapLoaded': '代价地图已加载。可左键拖拽端点或障碍矩形更新场景，或左键拖拽空白区域平移视图。',
      'status.costmapLoadFailed': '加载代价地图失败：{message}',
      'status.planSuccess': '{optimizerLabel}完成。A* 用时 {astarTimeMs} 毫秒，平滑用时 {smoothTimeMs} 毫秒。{statsSummary}',
      'status.pathStats': '返回 {pointCount} 个点，总长度 {pathLength}，平均间距 {meanSpacing}。',
      'status.planFallback': 'A* 在 {astarTimeMs} 毫秒内成功，但 {optimizerLabel}失败{errorCodeSuffix}，因此当前显示参考路径。{smoothMessage}',
      'status.planRejectedShown': 'A* 在 {astarTimeMs} 毫秒内成功，但 {optimizerLabel}失败{errorCodeSuffix}。当前仍显示被拒绝的平滑候选路径。{smoothMessage}',
      'selection.ready': '标记点就绪',
      'selection.dragMarker': '正在拖拽标记点',
      'selection.dragObstacle': '正在拖拽障碍',
      'selection.dragScene': '拖拽场景',
      'selection.panning': '正在平移视图',
      'selection.leftDrag': '左键拖拽',
      'optimizer.constrained': '约束平滑器',
      'optimizer.kinematic': '运动学平滑器',
      'weights.modelWeightLabel': '模型权重: <span id="val_model_weight">20</span>',
      'weights.modelWeightHint': '这是运动学状态转移一致性残差的原始权重。调高后，每一步状态转移都会更贴近自行车模型预测，而不只是“看起来更平滑”。',
      'optimizer.mode.constrained': '约束平滑器使用现有的 C++ Ceres 目标函数，包含曲率、尖点和 ESDF 障碍项。',
      'optimizer.mode.kinematic': '运动学平滑器使用新的 C++ 自行车模型状态优化器，包含 ESDF 障碍残差与足迹采样。',
      'optimizer.linear.constrained': '选择每次非线性迭代内部使用的 Ceres 线性求解后端。',
      'optimizer.linear.kinematic': '仅约束平滑器会使用该设置。运动学平滑器使用致密后端求解单个打包状态向量。',
      'robot.badge.capsule': '胶囊',
      'robot.badge.point': '单圆',
      'robot.summary.previewPending': '琥珀色显示的是胶囊检查点；虚线矩形仅用于最终验证。',
      'robot.validation.pending': '每次规划后会在这里显示矩形验证状态。',
      'validation.path.smoothed_candidate': '被拒绝的平滑候选路径',
      'validation.path.reference_fallback': '返回的参考路径',
      'validation.path.smoothed_path': '返回的平滑路径',
      'validation.reason.lethal_overlap': '与致命障碍重叠',
      'validation.reason.out_of_bounds': '足迹超出地图边界',
      'validation.reason.nonfinite_pose': '位姿值非有限',
      'loupe.esdfEmpty': 'ESDF --',
      'loupe.worldEmpty': '世界坐标：--',
      'loupe.cellEmpty': '栅格：--',
      'map.kind.default': '合成场景',
      'map.description.default': '固定的合成障碍地图，用于观察基于 ESDF 的规划器和平滑器行为。',
      'popup.note.default': '将鼠标悬停在玫瑰色平滑路径上的某个点，可查看它的几何、朝向、局部净空与相邻线段上下文。',
      'popup.role.startEndpoint': '起点端点',
      'popup.role.goalEndpoint': '终点端点',
      'popup.role.startAnchor': '起点锚点',
      'popup.role.goalAnchor': '终点锚点',
      'popup.role.interiorPoint': '内部点',
      'curvature.note.pending': '执行规划后，可按优化路径弧长绘制曲率、段间距和曲率变化率。',
      'curvature.note.plotlyMissing': 'Plotly 加载失败，因此无法绘制剖面图。',
      'curvature.note.plotlyReload': 'Plotly 加载失败。刷新页面后再渲染路径剖面。',
      'curvature.empty.curvature': '成功规划后会在这里显示曲率图。',
      'curvature.empty.spacing': '成功规划后会在这里显示段间距 ds。',
      'curvature.empty.rate': '成功规划后会在这里显示曲率变化率 dk/ds。',
      'curvature.axis.arcLength': '弧长 s (米)',
      'curvature.axis.curvature': '曲率 k (1/米)',
      'curvature.axis.segmentMidpoint': '段中点 s (米)',
      'curvature.axis.spacing': '间距 ds (米)',
      'curvature.axis.rate': 'dk/ds (1/米^2)',
      'run.note.success': '{optimizerLabel}已生成平滑路径。切换图层并比较原始、参考和平滑路径长度，可以观察后端如何改变路径几何。',
      'run.note.fallback': '{optimizerLabel}失败，因此当前显示参考路径。{smoothMessage}',
      'run.note.rejected': '{optimizerLabel}未通过验证，但当前仍显示被拒绝的平滑候选路径以便检查。{smoothMessage}',
      'run.pipeline.pending': '每次运行后会在这里显示流水线状态。',
      'run.pipeline.summary': '流水线：{summary}',
      'run.smoothState.success': '{optimizerLabel}成功',
      'run.smoothState.fallback': '{optimizerLabel}回退',
      'run.smoothState.rejected': '{optimizerLabel}候选已拒绝但仍显示',
      'run.stage.status.ok': '正常',
      'run.stage.status.error': '错误',
      'run.stage.status.fallback': '回退',
      'run.stage.label.validate': '矩形验证',
      'run.stage.label.web': 'Web',
      'descriptor.outsideMap': '地图外',
      'descriptor.unknownSpace': '未知区域',
      'descriptor.lethalObstacle': '致命障碍',
      'descriptor.inscribedObstacle': '贴边膨胀障碍',
      'descriptor.inflatedCost': '膨胀代价',
      'descriptor.freeSpace': '空闲区域',
      'marker.start': '起点',
      'marker.goal': '终点',
      'canvas.obstacleLabel': '障碍 {index}',
      'derived.minTurnRadius': '最小转弯半径：{value}',
      'derived.minTurnRadiusEmpty': '最小转弯半径：--'
    }
  };

  const staticTextDefaults = new Map();
  const staticHtmlDefaults = new Map();
  const getInitialLanguage = () => {
    const stored = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (SUPPORTED_LANGUAGES.includes(stored)) {
      return stored;
    }
    const browserLanguage = (navigator.language || '').toLowerCase();
    return browserLanguage.startsWith('zh') ? 'zh' : 'en';
  };
  let currentLanguage = getInitialLanguage();

  const interpolateMessage = (template, params = {}) => String(template).replace(/\{(\w+)\}/g, (_, key) => {
    if (Object.prototype.hasOwnProperty.call(params, key)) {
      return params[key];
    }
    return `{${key}}`;
  });

  const t = (key, params = {}) => {
    const dictionary = messages[currentLanguage] || messages.en;
    const template = dictionary[key] ?? messages.en[key] ?? key;
    return interpolateMessage(template, params);
  };

  const captureStaticDefaults = () => {
    document.querySelectorAll('[data-i18n]').forEach(element => {
      staticTextDefaults.set(element.dataset.i18n, element.textContent);
    });
    document.querySelectorAll('[data-i18n-html]').forEach(element => {
      staticHtmlDefaults.set(element.dataset.i18nHtml, element.innerHTML);
    });
  };

  const applyStaticTranslations = () => {
    document.documentElement.lang = currentLanguage === 'zh' ? 'zh-CN' : 'en';
    document.title = t('document.title');

    document.querySelectorAll('[data-i18n]').forEach(element => {
      const key = element.dataset.i18n;
      const defaultValue = staticTextDefaults.get(key) ?? element.textContent;
      element.textContent = currentLanguage === 'zh' ? (zhStaticTranslations[key] ?? defaultValue) : defaultValue;
    });

    document.querySelectorAll('[data-i18n-html]').forEach(element => {
      const key = element.dataset.i18nHtml;
      const defaultValue = staticHtmlDefaults.get(key) ?? element.innerHTML;
      element.innerHTML = currentLanguage === 'zh' ? (zhStaticTranslations[key] ?? defaultValue) : defaultValue;
    });
  };

  const localizeKnownText = (value, mapping) => {
    if (!value) {
      return value;
    }
    return mapping[value] || value;
  };

  const localizeOptimizerLabel = label => localizeKnownText(label, {
    'Constrained Smoother': t('optimizer.constrained'),
    'Kinematic Smoother': t('optimizer.kinematic'),
  });

  const syncAllControlReadouts = () => {
    sliders.forEach(id => {
      const input = document.getElementById(id);
      const label = document.getElementById('val_' + id);
      if (!input || !label) {
        return;
      }
      label.textContent = sliderConfig[id](parseFloat(input.value));
    });

    numericInputs.forEach(id => {
      const input = document.getElementById(id);
      const label = document.getElementById('val_' + id);
      if (!input || !label) {
        return;
      }
      const value = parseFloat(input.value);
      if (Number.isFinite(value)) {
        label.textContent = numericInputConfig[id](value);
      }
    });
  };

  captureStaticDefaults();
  applyStaticTranslations();

  const sliderConfig = {
    start_yaw_deg: value => `${Math.round(value)} ${t('unit.degree')}`,
    goal_yaw_deg: value => `${Math.round(value)} ${t('unit.degree')}`,
    goal_longitudinal_tolerance_m: value => Number(value).toFixed(2),
    goal_lateral_tolerance_m: value => Number(value).toFixed(2),
    planner_penalty_weight: value => Number(value).toFixed(1),
    hinge_loss_threshold_m: value => Number(value).toFixed(2),
    point_robot_radius_m: value => Number(value).toFixed(2),
    robot_length_m: value => Number(value).toFixed(2),
    robot_width_m: value => Number(value).toFixed(2),
    smooth_weight: value => Math.round(value).toLocaleString(),
    model_weight: value => Math.round(value).toLocaleString(),
    costmap_weight: value => Number(value).toFixed(3),
    cusp_costmap_weight: value => Number(value).toFixed(3),
    cusp_zone_length: value => Number(value).toFixed(2),
    distance_weight: value => Number(value).toFixed(1),
    curvature_weight: value => Number(value).toFixed(1),
    curvature_rate_weight: value => Number(value).toFixed(1),
    kinematic_curvature_weight: value => Number(value).toFixed(1),
    kinematic_curvature_rate_weight: value => Number(value).toFixed(1),
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
  const optimizerScopedSliderIds = [
    'smooth_weight', 'model_weight', 'costmap_weight', 'cusp_costmap_weight', 'cusp_zone_length',
    'distance_weight', 'curvature_weight', 'curvature_rate_weight',
    'kinematic_curvature_weight', 'kinematic_curvature_rate_weight', 'max_curvature',
    'reference_spacing_target_m', 'max_iterations', 'max_time',
    'path_downsampling_factor', 'path_upsampling_factor',
  ];
  const optimizerScopedNumericIds = ['param_tol', 'fn_tol', 'gradient_tol'];
  const optimizerScopedSelectIds = ['linear_solver_type'];
  const optimizerScopedCheckboxIds = ['optimizer_debug'];
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
    currentOptimizerType: optimizerTypeSelect ? optimizerTypeSelect.value : 'constrained_smoother',
    optimizerProfiles: {
      constrained_smoother: null,
      kinematic_smoother: null,
    },
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

  function updateSliderReadout(id) {
    const input = document.getElementById(id);
    const label = document.getElementById('val_' + id);
    if (!input || !label) {
      return;
    }
    label.textContent = sliderConfig[id](parseFloat(input.value));
  }

  function updateNumericReadout(id) {
    const input = document.getElementById(id);
    const label = document.getElementById('val_' + id);
    if (!input || !label) {
      return;
    }
    const value = parseFloat(input.value);
    if (!Number.isFinite(value)) {
      return;
    }
    label.textContent = numericInputConfig[id](value);
  }

  function captureOptimizerProfile() {
    return {
      sliders: Object.fromEntries(optimizerScopedSliderIds.map(id => [id, document.getElementById(id)?.value ?? null])),
      numerics: Object.fromEntries(optimizerScopedNumericIds.map(id => [id, document.getElementById(id)?.value ?? null])),
      selects: Object.fromEntries(optimizerScopedSelectIds.map(id => [id, document.getElementById(id)?.value ?? null])),
      checkboxes: Object.fromEntries(optimizerScopedCheckboxIds.map(id => [id, Boolean(document.getElementById(id)?.checked)])),
    };
  }

  function applyOptimizerProfile(profile) {
    if (!profile) {
      return;
    }

    optimizerScopedSliderIds.forEach(id => {
      const input = document.getElementById(id);
      const value = profile.sliders?.[id];
      if (!input || value === null || value === undefined) {
        return;
      }
      input.value = value;
      updateSliderReadout(id);
    });

    optimizerScopedNumericIds.forEach(id => {
      const input = document.getElementById(id);
      const value = profile.numerics?.[id];
      if (!input || value === null || value === undefined) {
        return;
      }
      input.value = value;
      updateNumericReadout(id);
    });

    optimizerScopedSelectIds.forEach(id => {
      const input = document.getElementById(id);
      const value = profile.selects?.[id];
      if (!input || value === null || value === undefined) {
        return;
      }
      input.value = value;
    });

    optimizerScopedCheckboxIds.forEach(id => {
      const input = document.getElementById(id);
      if (!input || !profile.checkboxes || !Object.prototype.hasOwnProperty.call(profile.checkboxes, id)) {
        return;
      }
      input.checked = Boolean(profile.checkboxes[id]);
    });

    syncDerivedParameterInfo();
    drawCurvatureChart();
  }

  function initializeOptimizerProfiles() {
    const initialProfile = captureOptimizerProfile();
    state.optimizerProfiles.constrained_smoother = {
      sliders: {...initialProfile.sliders},
      numerics: {...initialProfile.numerics},
      selects: {...initialProfile.selects},
      checkboxes: {...initialProfile.checkboxes},
    };
    state.optimizerProfiles.kinematic_smoother = {
      sliders: {...initialProfile.sliders},
      numerics: {...initialProfile.numerics},
      selects: {...initialProfile.selects},
      checkboxes: {...initialProfile.checkboxes},
    };
  }

  sliders.forEach(id => {
    const input = document.getElementById(id);
    if (!input || !document.getElementById('val_' + id)) {
      return;
    }

    const sync = () => {
      updateSliderReadout(id);
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
    if (!input || !document.getElementById('val_' + id)) {
      return;
    }

    const sync = () => {
      updateNumericReadout(id);
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
      const previousOptimizerType = state.currentOptimizerType;
      if (previousOptimizerType && state.optimizerProfiles[previousOptimizerType]) {
        state.optimizerProfiles[previousOptimizerType] = captureOptimizerProfile();
      }
      state.currentOptimizerType = optimizerTypeSelect.value;
      applyOptimizerProfile(state.optimizerProfiles[state.currentOptimizerType]);
      updateOptimizerUi();
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
        ? t('derived.minTurnRadiusEmpty')
        : t('derived.minTurnRadius', {value: `${minTurnRadius.toFixed(2)} ${t('unit.meter')}`})
    );
  }

  let pendingCanvasResizeFrame = null;
  let pendingPlotResizeFrame = null;

  function resizeMapCanvas() {
    if (!canvas || !canvasWrap) {
      return;
    }

    const wrapStyle = window.getComputedStyle(canvasWrap);
    const innerWidth = canvasWrap.clientWidth
      - parseFloat(wrapStyle.paddingLeft || '0')
      - parseFloat(wrapStyle.paddingRight || '0');
    const innerHeight = canvasWrap.clientHeight
      - parseFloat(wrapStyle.paddingTop || '0')
      - parseFloat(wrapStyle.paddingBottom || '0');
    const displaySize = Math.max(1, Math.floor(Math.min(innerWidth, innerHeight)));
    if (!Number.isFinite(displaySize) || displaySize < 1) {
      return;
    }

    const displaySizePx = `${displaySize}px`;
    if (
      canvas.width === baseCanvasPixelSize
      && canvas.height === baseCanvasPixelSize
      && canvas.style.width === displaySizePx
      && canvas.style.height === displaySizePx
    ) {
      return;
    }

    canvas.width = baseCanvasPixelSize;
    canvas.height = baseCanvasPixelSize;
    canvas.style.width = displaySizePx;
    canvas.style.height = displaySizePx;
    positionOptimizedPointPopup();
    draw();
  }

  function scheduleMapCanvasResize() {
    if (pendingCanvasResizeFrame !== null) {
      window.cancelAnimationFrame(pendingCanvasResizeFrame);
    }
    pendingCanvasResizeFrame = window.requestAnimationFrame(() => {
      pendingCanvasResizeFrame = null;
      resizeMapCanvas();
    });
  }

  function resizeProfileCharts({rerender = false} = {}) {
    if (!chartElements.length) {
      return;
    }

    if (rerender) {
      drawCurvatureChart();
      return;
    }

    if (!window.Plotly?.Plots) {
      return;
    }

    chartElements.forEach(element => {
      if (element?.clientWidth > 0 && element?.clientHeight > 0 && element.data) {
        window.Plotly.Plots.resize(element);
      }
    });
  }

  function scheduleProfileChartResize(options = {}) {
    if (pendingPlotResizeFrame !== null) {
      window.cancelAnimationFrame(pendingPlotResizeFrame);
    }
    pendingPlotResizeFrame = window.requestAnimationFrame(() => {
      pendingPlotResizeFrame = null;
      resizeProfileCharts(options);
    });
  }

  syncDerivedParameterInfo();
  updateRobotConfigUi();
  initializeOptimizerProfiles();
  clearOptimizedPointInspector();
  clearCurvatureChart();
  resizeMapCanvas();

  const maxCurvatureInput = document.getElementById('max_curvature');
  if (maxCurvatureInput) {
    maxCurvatureInput.addEventListener('input', () => {
      syncDerivedParameterInfo();
      drawCurvatureChart();
    });
  }

  window.addEventListener('resize', () => {
    scheduleMapCanvasResize();
    scheduleProfileChartResize();
  });

  if (window.ResizeObserver) {
    if (canvasWrap) {
      new window.ResizeObserver(() => scheduleMapCanvasResize()).observe(canvasWrap);
    }
    document.querySelectorAll('.curvature-panel').forEach(profilePanel => {
      new window.ResizeObserver(() => scheduleProfileChartResize()).observe(profilePanel);
    });
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

  function initializePanelSwitcher(container) {
    const navButtons = Array.from(container.querySelectorAll('.control-nav-btn[data-panel-target]'));
    const panels = Array.from(container.querySelectorAll('[data-panel-id]'));
    if (!navButtons.length || !panels.length) {
      return;
    }

    const setActivePanel = panelId => {
      const fallbackPanelId = panels[0]?.dataset.panelId;
      const nextPanelId = panels.some(panel => panel.dataset.panelId === panelId) ? panelId : fallbackPanelId;
      if (!nextPanelId) {
        return;
      }

      navButtons.forEach(button => {
        const isActive = button.dataset.panelTarget === nextPanelId;
        button.classList.toggle('is-active', isActive);
        button.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });

      panels.forEach(panel => {
        const isActive = panel.dataset.panelId === nextPanelId;
        panel.classList.toggle('is-active', isActive);
        panel.hidden = !isActive;
      });

      scheduleMapCanvasResize();
      scheduleProfileChartResize();
    };

    navButtons.forEach(button => {
      button.addEventListener('click', () => {
        setActivePanel(button.dataset.panelTarget);
      });
    });

    const defaultPanelId = navButtons.find(button => button.classList.contains('is-active'))?.dataset.panelTarget
      || panels[0]?.dataset.panelId;
    setActivePanel(defaultPanelId);
  }

  function initializeControlPanels() {
    const panelSwitchers = Array.from(document.querySelectorAll('[data-panel-switcher]'));
    if (!panelSwitchers.length) {
      return;
    }

    panelSwitchers.forEach(initializePanelSwitcher);
  }

  function updateOptimizerUi() {
    const optimizerType = optimizerTypeSelect ? optimizerTypeSelect.value : 'constrained_smoother';
    const isConstrainedSmoother = optimizerType === 'constrained_smoother';
    const smoothWeightGroup = document.getElementById('smooth-weight-group');
    const modelWeightGroup = document.getElementById('model-weight-group');
    const constrainedCurvatureWeightGroup = document.getElementById('constrained-curvature-weight-group');
    const constrainedCurvatureRateWeightGroup = document.getElementById('constrained-curvature-rate-weight-group');
    const kinematicCurvatureWeightGroup = document.getElementById('kinematic-curvature-weight-group');
    const kinematicCurvatureRateWeightGroup = document.getElementById('kinematic-curvature-rate-weight-group');
    const weightsOptimizerBadge = document.getElementById('weights-optimizer-badge');

    if (linearSolverTypeSelect) {
      linearSolverTypeSelect.disabled = !isConstrainedSmoother;
    }

    if (smoothWeightGroup) {
      smoothWeightGroup.hidden = !isConstrainedSmoother;
    }

    if (modelWeightGroup) {
      modelWeightGroup.hidden = isConstrainedSmoother;
    }

    if (constrainedCurvatureWeightGroup) {
      constrainedCurvatureWeightGroup.hidden = !isConstrainedSmoother;
    }

    if (constrainedCurvatureRateWeightGroup) {
      constrainedCurvatureRateWeightGroup.hidden = !isConstrainedSmoother;
    }

    if (kinematicCurvatureWeightGroup) {
      kinematicCurvatureWeightGroup.hidden = isConstrainedSmoother;
    }

    if (kinematicCurvatureRateWeightGroup) {
      kinematicCurvatureRateWeightGroup.hidden = isConstrainedSmoother;
    }

    if (weightsOptimizerBadge) {
      weightsOptimizerBadge.textContent = isConstrainedSmoother
        ? t('optimizer.constrained')
        : t('optimizer.kinematic');
    }

    setText(
      'optimizer-mode-hint',
      isConstrainedSmoother
        ? t('optimizer.mode.constrained')
        : t('optimizer.mode.kinematic')
    );
    setText(
      'linear-solver-hint',
      isConstrainedSmoother
        ? t('optimizer.linear.constrained')
        : t('optimizer.linear.kinematic')
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
    const badgeText = config.mode === 'capsule' ? t('robot.badge.capsule') : t('robot.badge.point');
    const summaryText = config.mode === 'capsule'
      ? currentLanguage === 'zh'
        ? `规划和平滑使用 ${config.localCheckPoints.length} 个胶囊检查点，半径为 ${formatMeters(config.checkRadiusM)}。虚线 ${formatMeters(config.lengthM)} × ${formatMeters(config.widthM)} 矩形仅用于最终验证。`
        : `Planning and smoothing use ${config.localCheckPoints.length} capsule checkpoints with ${formatMeters(config.checkRadiusM)} radius. The dashed ${formatMeters(config.lengthM)} × ${formatMeters(config.widthM)} rectangle is final validation only.`
      : currentLanguage === 'zh'
        ? `规划和平滑使用一个半径为 ${formatMeters(config.checkRadiusM)} 的检查圆。虚线 ${formatMeters(config.lengthM)} × ${formatMeters(config.widthM)} 矩形仍用于最终路径验证。`
        : `Planning and smoothing use one ${formatMeters(config.checkRadiusM)} check circle. The dashed ${formatMeters(config.lengthM)} × ${formatMeters(config.widthM)} rectangle still validates the final path.`;

    setText('footprint-preview-badge', badgeText);
    setText(
      'robot-config-summary',
      summaryText
    );
    if (!state.paths?.final_rectangle_validation) {
      setText('footprint-validation-summary', t('robot.validation.pending'));
      clearValidationFailureDetails();
    }
    drawFootprintPreview();
  }

  updateOptimizerUi();
  initializeControlPanels();
  function formatMeters(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} ${t('unit.meter')}`;
  }

  function formatSeconds(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} ${t('unit.second')}`;
  }

  function formatDegrees(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} ${t('unit.degree')}`;
  }

  function formatRadians(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} ${t('unit.radian')}`;
  }

  function formatCurvature(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '--';
    }
    return `${Number(value).toFixed(digits)} ${t('unit.curvature')}`;
  }

  function formatValidationPathLabel(validatedPath) {
    if (validatedPath === 'smoothed_candidate') {
      return t('validation.path.smoothed_candidate');
    }
    if (validatedPath === 'reference_fallback') {
      return t('validation.path.reference_fallback');
    }
    if (validatedPath === 'smoothed_path') {
      return t('validation.path.smoothed_path');
    }
    return '--';
  }

  function isRejectedSmoothedPathVisible(pathData = state.paths) {
    return Boolean(
      pathData
      && !pathData.smooth_success
      && pathData.final_rectangle_validation?.validated_path === 'smoothed_path'
    );
  }

  function updateSmoothedLayerPresentation(pathData = state.paths) {
    const layerName = document.querySelector('[data-i18n="layers.smoothedPath"]');
    const layerHint = document.querySelector('[data-i18n="layers.smoothedPathHint"]');
    const layerSwatch = document.querySelector('.swatch-smoothed');
    const rejected = isRejectedSmoothedPathVisible(pathData);

    if (layerName) {
      layerName.textContent = rejected ? t('layers.rejectedSmoothedPath') : t('layers.smoothedPath');
    }
    if (layerHint) {
      layerHint.textContent = rejected ? t('layers.rejectedSmoothedPathHint') : t('layers.smoothedPathHint');
    }
    if (layerSwatch) {
      layerSwatch.classList.toggle('swatch-smoothed-rejected', rejected);
    }
  }

  function formatValidationReason(reason) {
    const reasonLabels = {
      lethal_overlap: t('validation.reason.lethal_overlap'),
      out_of_bounds: t('validation.reason.out_of_bounds'),
      nonfinite_pose: t('validation.reason.nonfinite_pose'),
    };
    return reasonLabels[reason] || '--';
  }

  function formatValidationPose(pose) {
    if (!pose || pose.x === null || pose.y === null || pose.x === undefined || pose.y === undefined) {
      return '--';
    }
    return `${Number(pose.x).toFixed(2)}, ${Number(pose.y).toFixed(2)} ${t('unit.meter')}`;
  }

  function buildPathStatsSummary(pathData) {
    const pointCount = Number(pathData?.num_returned_pts ?? pathData?.num_opt_pts);
    const totalLength = Number(pathData?.opt_path_length_m);
    const xs = pathData?.opt_x || [];
    const ys = pathData?.opt_y || [];
    let meanSpacing = Number.NaN;

    if (xs.length >= 2 && ys.length >= 2) {
      let spacingSum = 0;
      let segmentCount = 0;
      for (let idx = 1; idx < Math.min(xs.length, ys.length); idx += 1) {
        spacingSum += Math.hypot(xs[idx] - xs[idx - 1], ys[idx] - ys[idx - 1]);
        segmentCount += 1;
      }
      meanSpacing = segmentCount > 0 ? spacingSum / segmentCount : Number.NaN;
    } else if (Number.isFinite(totalLength) && Number.isFinite(pointCount) && pointCount > 1) {
      meanSpacing = totalLength / (pointCount - 1);
    }

    return t('status.pathStats', {
      pointCount: Number.isFinite(pointCount) ? String(pointCount) : '--',
      pathLength: formatMeters(totalLength),
      meanSpacing: formatMeters(meanSpacing, 3),
    });
  }

  function formatAngleDiagnostic(anglePayload) {
    if (!anglePayload || anglePayload.deg === null || anglePayload.deg === undefined) {
      return '--';
    }
    return `${Number(anglePayload.deg).toFixed(2)} ${t('unit.degree')} (${Number(anglePayload.rad).toFixed(4)} ${t('unit.radian')})`;
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

  function clearKinematicDiagnostics() {
    if (kinematicDiagnosticsCard) {
      kinematicDiagnosticsCard.hidden = true;
    }
    [
      'kinematic-detail-mode',
      'kinematic-goal-segment-error',
      'kinematic-goal-tolerance',
      'kinematic-goal-expected',
      'kinematic-goal-actual-segment',
      'kinematic-goal-pose-heading',
      'kinematic-goal-pose-error',
      'kinematic-param-max-iterations',
      'kinematic-param-max-time',
      'kinematic-param-max-curvature',
      'kinematic-param-curvature-rate-weight',
      'kinematic-param-resampling',
      'kinematic-param-ceres-tolerances',
    ].forEach(id => setText(id, '--'));
  }

  function updateKinematicDiagnostics(data) {
    if (!kinematicDiagnosticsCard || data?.optimizer_type !== 'kinematic_smoother') {
      clearKinematicDiagnostics();
      return;
    }

    kinematicDiagnosticsCard.hidden = false;
    const diagnostics = data.goal_orientation_diagnostics || {};
    const optimizerConfig = data.optimizer_config || {};
    setText('kinematic-detail-mode', data.smooth_success ? t('run.stage.status.ok') : t('run.stage.status.error'));
    setText('kinematic-goal-segment-error', formatAngleDiagnostic(diagnostics.terminal_segment_error));
    setText('kinematic-goal-tolerance', formatAngleDiagnostic(diagnostics.tolerance));
    setText('kinematic-goal-expected', formatAngleDiagnostic(diagnostics.expected_goal_heading));
    setText('kinematic-goal-actual-segment', formatAngleDiagnostic(diagnostics.terminal_segment_heading));
    setText('kinematic-goal-pose-heading', formatAngleDiagnostic(diagnostics.terminal_pose_heading));
    setText('kinematic-goal-pose-error', formatAngleDiagnostic(diagnostics.terminal_pose_error));
    setText('kinematic-param-max-iterations', optimizerConfig.max_iterations ?? '--');
    setText('kinematic-param-max-time', formatSeconds(optimizerConfig.max_time_s, 2));
    setText('kinematic-param-max-curvature', formatCurvature(optimizerConfig.max_curvature, 2));
    setText(
      'kinematic-param-curvature-rate-weight',
      optimizerConfig.kinematic_curvature_rate_weight ?? '--'
    );
    setText('kinematic-param-resampling', `${optimizerConfig.path_downsampling_factor ?? '--'} / ${optimizerConfig.path_upsampling_factor ?? '--'}`);
    setText(
      'kinematic-param-ceres-tolerances',
      `p ${Number(optimizerConfig.param_tol ?? NaN).toExponential(1)}, f ${Number(optimizerConfig.fn_tol ?? NaN).toExponential(1)}, g ${Number(optimizerConfig.gradient_tol ?? NaN).toExponential(1)}`
    );
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
    return `${point.x.toFixed(2)}, ${point.y.toFixed(2)} ${t('unit.meter')}`;
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
      return {text: t('descriptor.outsideMap'), kind: 'outside'};
    }

    if (cost === 255) {
      return {text: t('descriptor.unknownSpace'), kind: 'unknown'};
    }

    if (cost >= 254) {
      return {text: t('descriptor.lethalObstacle'), kind: 'lethal'};
    }

    if (cost === 253) {
      return {text: t('descriptor.inscribedObstacle'), kind: 'inscribed'};
    }

    if (cost > 0) {
      return {text: t('descriptor.inflatedCost'), kind: 'inflated'};
    }

    return {text: t('descriptor.freeSpace'), kind: 'free'};
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

    setStatus(t('status.parameterChangedReplanning'), '');
    state.pendingAutoPlanTimer = window.setTimeout(() => {
      state.pendingAutoPlanTimer = null;
      runPlanning({reason: 'slider'});
    }, AUTO_REPLAN_DELAY_MS);
  }

  function syncPhaseUi() {
    setText('phase-indicator', t('selection.ready'));
    const pillText = state.draggingMarker
      ? t('selection.dragMarker')
      : state.draggingObstacleIndex !== null
        ? t('selection.dragObstacle')
        : t('selection.dragScene');
    setText('selection-pill', pillText);
  }

  function updateSelectionInfo() {
    const startHeading = getHeadingValue('start_yaw_deg', DEFAULT_HEADINGS_DEG.start);
    const goalHeading = getHeadingValue('goal_yaw_deg', DEFAULT_HEADINGS_DEG.goal);
    const keepStartOrientation = getConstraintEnabled('keep_start_orientation', true);
    const keepGoalOrientation = getConstraintEnabled('keep_goal_orientation', true);
    setText('start-coord', `${formatCoord(state.start)} | ${Math.round(startHeading)} ${t('unit.degree')}`);
    setText('goal-coord', `${formatCoord(state.goal)} | ${Math.round(goalHeading)} ${t('unit.degree')}`);
    setText('start-heading-readout', `${Math.round(startHeading)} ${t('unit.degree')}`);
    setText('goal-heading-readout', `${Math.round(goalHeading)} ${t('unit.degree')}`);
    setText('start-constraint-readout', keepStartOrientation ? t('common.enabled') : t('common.disabled'));
    setText('goal-constraint-readout', keepGoalOrientation ? t('common.enabled') : t('common.disabled'));
    setText('cursor-coord', formatCoord(state.hover));
    setText('zoom-level', `${state.viewScale.toFixed(2)}x`);
    const gestureText = state.draggingMarker
      ? t('selection.dragMarker')
      : state.draggingObstacleIndex !== null
        ? t('selection.dragObstacle')
        : state.dragging
          ? t('selection.panning')
          : t('selection.leftDrag');
    setText('view-mode-label', gestureText);
    syncPhaseUi();
  }

  function hideLoupe() {
    setText('loupe-cost-value', t('loupe.esdfEmpty'));
    setText('loupe-cell-cost', '--');
    setText('loupe-esdf-distance', '--');
    setText('loupe-world', t('loupe.worldEmpty'));
    setText('loupe-cell', t('loupe.cellEmpty'));
    setText('loupe-kind', t('common.outsideMap'));
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
      sample.esdfDistance === null
        ? t('loupe.esdfEmpty')
        : `ESDF ${sample.esdfDistance.toFixed(2)} ${t('unit.meter')}`
    );
    setText('loupe-cell-cost', String(sample.cost));
    setText(
      'loupe-esdf-distance',
      sample.esdfDistance === null ? '--' : `${sample.esdfDistance.toFixed(2)} ${t('unit.meter')}`
    );
    setText('loupe-world', currentLanguage === 'zh'
      ? `世界坐标：${sample.worldX.toFixed(2)}, ${sample.worldY.toFixed(2)} ${t('unit.meter')}`
      : `World: ${sample.worldX.toFixed(2)}, ${sample.worldY.toFixed(2)} ${t('unit.meter')}`);
    setText('loupe-cell', currentLanguage === 'zh' ? `栅格：(${sample.mx}, ${sample.my})` : `Cell: (${sample.mx}, ${sample.my})`);
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
      t('popup.note.default')
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
      return t('popup.role.startEndpoint');
    }
    if (index === pointCount - 1) {
      return t('popup.role.goalEndpoint');
    }
    if (keepStartOrientation && index === 1) {
      return t('popup.role.startAnchor');
    }
    if (keepGoalOrientation && index === pointCount - 2) {
      return t('popup.role.goalAnchor');
    }
    return t('popup.role.interiorPoint');
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
    setText('opt-point-world', `${info.worldX.toFixed(2)}, ${info.worldY.toFixed(2)} ${t('unit.meter')}`);
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
      currentLanguage === 'zh'
        ? `点 ${info.index + 1} 的角色为${info.role}。转角与曲率由该优化位姿附近的局部三点几何近似得到。`
        : `Point ${info.index + 1} is ${info.role.toLowerCase()}. Turn angle and curvature are estimated from the local three-point geometry around this optimized pose.`
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

    setText('hero-map-size', `${worldWidth.toFixed(1)} x ${worldHeight.toFixed(1)} ${t('unit.meter')}`);
    setText('hero-resolution', `${costmap.resolution.toFixed(2)} ${t('unit.metersPerCell')}`);
    setText('map-grid', `${costmap.size_x} x ${costmap.size_y}`);
    setText('map-world-size', `${worldWidth.toFixed(1)} x ${worldHeight.toFixed(1)} ${t('unit.meter')}`);
    setText('map-world-size-toolbar', `${worldWidth.toFixed(1)} x ${worldHeight.toFixed(1)} ${t('unit.meter')}`);
    setText('map-resolution', `${costmap.resolution.toFixed(2)} ${t('unit.metersPerCell')}`);
    setText('map-origin', `${costmap.origin_x.toFixed(1)}, ${costmap.origin_y.toFixed(1)} ${t('unit.meter')}`);
    setText('map-obstacles', String(meta.obstacle_count ?? '--'));
    setText('map-inflation', `${(meta.inflation_radius_m ?? 0).toFixed(2)} ${t('unit.meter')} / ${meta.inflation_radius_cells ?? '--'} ${t('unit.cells')}`);
    setText('map-inflation-toolbar', `${(meta.inflation_radius_m ?? 0).toFixed(2)} ${t('unit.meter')}`);
    setText('map-free-cells', `${meta.free_cells ?? '--'} / ${meta.cell_count ?? '--'}`);
    setText('map-inflated-cells', `${meta.inflated_cells ?? '--'} / ${meta.cell_count ?? '--'}`);
    setText('map-lethal-cells', `${meta.lethal_cells ?? '--'} / ${meta.cell_count ?? '--'}`);
    const description = localizeKnownText(
      meta.description || t('map.description.default'),
      {
        'Fixed synthetic obstacle map used to inspect ESDF-based planner and smoother behavior.': t('map.description.default'),
        'A draggable 20m x 20m obstacle map with rectangular lethal obstacles and a 5-cell inflated safety buffer for visualization. The C++ A* planner and constrained smoother both optimize ESDF-derived obstacle penalties.': currentLanguage === 'zh'
          ? '一个可拖拽编辑的 20 米 × 20 米障碍地图，包含矩形致命障碍物和 5 格膨胀安全缓冲区，便于可视化。C++ A* 规划器与约束平滑器都会优化基于 ESDF 的障碍惩罚。'
          : 'A draggable 20m x 20m obstacle map with rectangular lethal obstacles and a 5-cell inflated safety buffer for visualization. The C++ A* planner and constrained smoother both optimize ESDF-derived obstacle penalties.',
      }
    );
    const kind = localizeKnownText(meta.name || t('map.kind.default'), {
      'Synthetic field': t('map.kind.default'),
      'Synthetic obstacle field': currentLanguage === 'zh' ? '合成障碍场' : 'Synthetic obstacle field',
    });
    setText('map-description', description);
    setText('map-kind', kind);
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
    setText('curvature-state', t('common.idle'));
    setText('curvature-peak', '--');
    setText('curvature-mean', '--');
    setText('curvature-min', '--');
    setText('curvature-max', '--');
    setText('curvature-note', t('curvature.note.pending'));

    if (!chartElements.length) {
      return;
    }

    if (!window.Plotly) {
      chartElements.forEach(element => {
        element.textContent = t('curvature.note.plotlyReload');
      });
      return;
    }

    const createEmptyLayout = (height, title) => ({
      height,
      margin: {l: 52, r: 14, t: 12, b: 38},
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
      createEmptyLayout(CHART_HEIGHTS.primary, t('curvature.empty.curvature')),
      config
    );
    window.Plotly.react(
      dsChart,
      [],
      createEmptyLayout(CHART_HEIGHTS.secondary, t('curvature.empty.spacing')),
      config
    );
    window.Plotly.react(
      dkdsChart,
      [],
      createEmptyLayout(CHART_HEIGHTS.secondary, t('curvature.empty.rate')),
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
      setText('curvature-state', t('common.plotlyMissing'));
      setText('curvature-note', t('curvature.note.plotlyMissing'));
      [curvatureChart, dsChart, dkdsChart].forEach(element => {
        element.textContent = t('curvature.note.plotlyReload');
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
      margin: {l: 58, r: 14, t: 14, b: 44},
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
    const curvatureLayout = makeLayout(CHART_HEIGHTS.primary, t('curvature.axis.arcLength'), t('curvature.axis.curvature'));
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
      makeLayout(CHART_HEIGHTS.secondary, t('curvature.axis.segmentMidpoint'), t('curvature.axis.spacing')),
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
      makeLayout(CHART_HEIGHTS.secondary, t('curvature.axis.arcLength'), t('curvature.axis.rate')),
      config
    );

    setText('curvature-state', t('common.chartReady'));
    setText('curvature-peak', formatCurvature(profile.curvatureStats.peakAbs));
    setText('curvature-mean', formatCurvature(profile.curvatureStats.meanAbs));
    setText('curvature-min', formatCurvature(profile.curvatureStats.signedMin));
    setText('curvature-max', formatCurvature(profile.curvatureStats.signedMax));
    setText(
      'curvature-note',
      currentLanguage === 'zh'
        ? `曲率 k(s)、返回点间距 ds 和曲率变化率 dk/ds 都是由连续优化路径采样点估算得到。琥珀色虚线表示当前最大曲率限制（${maxCurvatureLimit.toFixed(2)} ${t('unit.curvature')}）。平均 ds：${profile.dsStats.mean.toFixed(3)} ${t('unit.meter')}，峰值 |dk/ds|：${profile.dkDsStats.peakAbs.toFixed(3)} ${t('unit.curvatureRate')}。`
        : `Curvature k(s), returned-point spacing ds, and curvature rate dk/ds are estimated from consecutive optimized path samples. Dashed amber lines mark the current Max Curvature limit (${maxCurvatureLimit.toFixed(2)} ${t('unit.curvature')}). Mean ds: ${profile.dsStats.mean.toFixed(3)} ${t('unit.meter')}, peak |dk/ds|: ${profile.dkDsStats.peakAbs.toFixed(3)} ${t('unit.curvatureRate')}.`
    );
  }

  function updateRunInfo(data) {
    state.curvatureProfile = computeCurvatureProfile(data);
    updateSmoothedLayerPresentation(data);
    updateKinematicDiagnostics(data);
    const showsRejectedCandidate = !data.smooth_success
      && data.final_rectangle_validation?.validated_path === 'smoothed_path';
    const optimizerLabel = localizeOptimizerLabel(data.optimizer_label || '');
    setText('info-optimizer', optimizerLabel || '--');
    setText('info-astar-time', `${data.astar_time_ms} ${t('unit.ms')}`);
    setText('info-smooth-time', `${data.smooth_time_ms} ${t('unit.ms')}`);
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
      : `${deltaValue >= 0 ? '+' : ''}${deltaValue.toFixed(2)} ${t('unit.meter')}`;
    setText('info-length-delta', deltaText);

    setText(
      'smooth-state',
      data.smooth_success
        ? t('run.smoothState.success', {optimizerLabel: optimizerLabel || 'optimizer'})
        : showsRejectedCandidate
          ? t('run.smoothState.rejected', {optimizerLabel: optimizerLabel || 'optimizer'})
          : t('run.smoothState.fallback', {optimizerLabel: optimizerLabel || 'optimizer'})
    );
    setText(
      'run-note',
      data.smooth_success
        ? t('run.note.success', {optimizerLabel: optimizerLabel || 'The selected optimizer'})
        : showsRejectedCandidate
          ? t('run.note.rejected', {optimizerLabel: optimizerLabel || 'The selected optimizer', smoothMessage: data.smooth_message || ''}).trim()
          : t('run.note.fallback', {optimizerLabel: optimizerLabel || 'The selected optimizer', smoothMessage: data.smooth_message || ''}).trim()
    );
    setText(
      'pipeline-summary',
      data.pipeline?.stages?.length
        ? t('run.pipeline.summary', {
          summary: data.pipeline.stages.map(stage => {
            const stageLabel = stage.key === 'validate'
              ? t('run.stage.label.validate')
              : stage.key === 'web'
                ? t('run.stage.label.web')
                : localizeOptimizerLabel(stage.label || '') || stage.label;
            let stageSummary = `${stageLabel}: ${t(`run.stage.status.${stage.status}`)}`;
            if (stage.path) {
              stageSummary += ` (${formatValidationPathLabel(stage.path)})`;
            }
            if (stage.error_code) {
              stageSummary += ` [${stage.error_code}]`;
            }
            return stageSummary;
          }).join(' -> ')
        })
        : t('run.pipeline.pending')
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
          ? currentLanguage === 'zh'
            ? ` ${formatValidationPathLabel(returnedValidation.validated_path)}矩形验证已通过。`
            : ` ${formatValidationPathLabel(returnedValidation.validated_path)} rectangle validation passed.`
          : currentLanguage === 'zh'
            ? ` ${formatValidationPathLabel(returnedValidation.validated_path)}矩形验证也失败了${returnedValidation.error_code ? ` [${returnedValidation.error_code}]` : ''}。${returnedValidation.message || ''}`
            : ` ${formatValidationPathLabel(returnedValidation.validated_path)} rectangle validation also failed${returnedValidation.error_code ? ` [${returnedValidation.error_code}]` : ''}. ${returnedValidation.message || ''}`;
      setText(
        'footprint-validation-summary',
        currentLanguage === 'zh'
          ? `平滑路径被拒绝${candidateCode}。${candidateValidation.message || ''}${returnedSummary}`.trim()
          : `Rejected smoothed path${candidateCode}. ${candidateValidation.message || ''}${returnedSummary}`.trim()
      );
    } else if (returnedValidation) {
      const pathLabel = formatValidationPathLabel(returnedValidation.validated_path);
      const statusText = returnedValidation.collision_free
        ? currentLanguage === 'zh'
          ? `${pathLabel}的矩形验证已在全部 ${data.num_returned_pts ?? data.num_opt_pts ?? 0} 个位姿上通过。`
          : `${pathLabel} rectangle validation passed on all ${data.num_returned_pts ?? data.num_opt_pts ?? 0} pose(s).`
        : currentLanguage === 'zh'
          ? `${pathLabel}的矩形验证失败${returnedValidation.error_code ? ` [${returnedValidation.error_code}]` : ''}。${returnedValidation.message || ''}`.trim()
          : `${pathLabel} rectangle validation failed${returnedValidation.error_code ? ` [${returnedValidation.error_code}]` : ''}. ${returnedValidation.message || ''}`.trim();
      setText('footprint-validation-summary', statusText);
    }
    showValidationFailureDetails(failureValidation);

    drawFootprintPreview(data);
    drawCurvatureChart();
  }

  function clearRunInfo() {
    planInfoIds.forEach(id => setText(id, '--'));
    setText('smooth-state', t('common.idle'));
    setText('run-note', currentLanguage === 'zh' ? '设置起点和终点后即可生成路径指标。' : 'Set a start and goal to generate path metrics.');
    setText('pipeline-summary', t('run.pipeline.pending'));
    setText('footprint-validation-summary', t('robot.validation.pending'));
    updateSmoothedLayerPresentation(null);
    clearValidationFailureDetails();
    clearKinematicDiagnostics();
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
      ctx.fillText(t('canvas.obstacleLabel', {index: index + 1}), bounds.left + 6, bounds.top + 6);
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

  function drawDirectionalSmoothedPath(xs, ys, thetas, width, drawDots = false, rejected = false) {
    if (!xs || xs.length < 2 || !thetas || thetas.length < 2) {
      drawPath(xs, ys, SMOOTHED_FORWARD_COLOR, width, drawDots);
      return;
    }

    ctx.save();
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    if (rejected) {
      ctx.save();
      ctx.setLineDash([10, 7]);
      ctx.lineWidth = width + 3.0;
      ctx.strokeStyle = 'rgba(70, 33, 22, 0.82)';
      ctx.beginPath();
      const firstPoint = worldToCanvas(xs[0], ys[0]);
      ctx.moveTo(firstPoint.x, firstPoint.y);
      for (let idx = 1; idx < xs.length; idx += 1) {
        const point = worldToCanvas(xs[idx], ys[idx]);
        ctx.lineTo(point.x, point.y);
      }
      ctx.stroke();
      ctx.restore();
    }

    for (let idx = 0; idx < xs.length - 1; idx += 1) {
      const startPoint = worldToCanvas(xs[idx], ys[idx]);
      const endPoint = worldToCanvas(xs[idx + 1], ys[idx + 1]);
      const dx = xs[idx + 1] - xs[idx];
      const dy = ys[idx + 1] - ys[idx];
      const motionDirection = getSegmentMotionDirection(thetas[idx], dx, dy);
      ctx.beginPath();
      ctx.moveTo(startPoint.x, startPoint.y);
      ctx.lineTo(endPoint.x, endPoint.y);
      if (rejected) {
        ctx.setLineDash([7, 5]);
      } else {
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = rejected
        ? motionDirection === 'reverse' ? 'rgba(38, 108, 177, 0.88)' : 'rgba(191, 54, 87, 0.88)'
        : motionDirection === 'reverse' ? SMOOTHED_REVERSE_COLOR : SMOOTHED_FORWARD_COLOR;
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
        ctx.arc(point.x, point.y, Math.max(width + (rejected ? 1.8 : 0.4), 2.2), 0, Math.PI * 2);
        ctx.fillStyle = rejected
          ? motionDirection === 'reverse' ? 'rgba(38, 108, 177, 0.94)' : 'rgba(191, 54, 87, 0.94)'
          : motionDirection === 'reverse' ? SMOOTHED_REVERSE_COLOR : SMOOTHED_FORWARD_COLOR;
        ctx.fill();
        if (rejected) {
          ctx.lineWidth = 1.2;
          ctx.strokeStyle = 'rgba(70, 33, 22, 0.82)';
          ctx.stroke();
        }
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
        drawDirectionalSmoothedPath(
          state.paths.opt_x,
          state.paths.opt_y,
          state.paths.opt_theta,
          2.8,
          true,
          isRejectedSmoothedPathVisible(state.paths)
        );
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
      manual: t('status.manualPlanning'),
      slider: t('status.sliderPlanning'),
      drag: t('status.dragPlanning'),
      obstacle: t('status.obstaclePlanning'),
      initial: t('status.initialPlanning'),
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
        setStatus(formatApiError(data, t('status.planningFailed')), 'error');
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
      const optimizerLabel = localizeOptimizerLabel(data.optimizer_label || 'Optimizer');
      const smoothErrorLabel = data.smooth_error?.code ? ` [${data.smooth_error.code}]` : '';
      const showsRejectedCandidate = !data.smooth_success
        && data.final_rectangle_validation?.validated_path === 'smoothed_path';
      setStatus(
        data.smooth_success
          ? t('status.planSuccess', {
            optimizerLabel,
            astarTimeMs: data.astar_time_ms,
            smoothTimeMs: data.smooth_time_ms,
            statsSummary: buildPathStatsSummary(data),
          })
          : showsRejectedCandidate
            ? t('status.planRejectedShown', {
              astarTimeMs: data.astar_time_ms,
              optimizerLabel,
              errorCodeSuffix: smoothErrorLabel,
              smoothMessage: data.smooth_message || '',
            }).trim()
          : t('status.planFallback', {
            astarTimeMs: data.astar_time_ms,
            optimizerLabel,
            errorCodeSuffix: smoothErrorLabel,
            smoothMessage: data.smooth_message || '',
          }).trim(),
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
      setStatus(t('status.networkError', {message: error.message}), 'error');
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
    setStatus(t('status.obstacleRebuilding'), '');
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
        setStatus(formatApiError(payload, t('status.obstacleUpdateFailed')), 'error');
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
      setStatus(t('status.obstacleUpdateError', {message: error.message}), 'error');
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
      setStatus(
        t('status.markerMoved', {marker: t(draggedMarker === 'start' ? 'marker.start' : 'marker.goal')}),
        ''
      );
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
    setStatus(t('status.viewReset'), '');
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
    setStatus(t('status.sceneReset'), '');
    updateSelectionInfo();
    resetView();
    draw();
    hideLoupe();
    updateOptimizedPointInspector();
    updateObstacleLayout();
  });

  resetViewBtn.addEventListener('click', () => {
    resetView();
    setStatus(t('status.viewReset'), '');
  });

  function refreshLocalizedUi() {
    applyStaticTranslations();
    syncAllControlReadouts();
    syncDerivedParameterInfo();
    updateOptimizerUi();
    updateRobotConfigUi();
    updateSelectionInfo();
    if (state.costmap) {
      updateMapInfo(state.costmap);
    }
    if (state.paths) {
      updateRunInfo(state.paths);
      const optimizerLabel = localizeOptimizerLabel(state.paths.optimizer_label || 'Optimizer');
      const smoothErrorLabel = state.paths.smooth_error?.code ? ` [${state.paths.smooth_error.code}]` : '';
      const showsRejectedCandidate = !state.paths.smooth_success
        && state.paths.final_rectangle_validation?.validated_path === 'smoothed_path';
      setStatus(
        state.paths.smooth_success
          ? t('status.planSuccess', {
            optimizerLabel,
            astarTimeMs: state.paths.astar_time_ms,
            smoothTimeMs: state.paths.smooth_time_ms,
            statsSummary: buildPathStatsSummary(state.paths),
          })
          : showsRejectedCandidate
            ? t('status.planRejectedShown', {
              astarTimeMs: state.paths.astar_time_ms,
              optimizerLabel,
              errorCodeSuffix: smoothErrorLabel,
              smoothMessage: state.paths.smooth_message || '',
            }).trim()
          : t('status.planFallback', {
            astarTimeMs: state.paths.astar_time_ms,
            optimizerLabel,
            errorCodeSuffix: smoothErrorLabel,
            smoothMessage: state.paths.smooth_message || '',
          }).trim(),
        state.paths.smooth_success ? 'ok' : 'error'
      );
    } else {
      clearRunInfo();
    }
    if (state.hoverSample) {
      updateLoupe();
    } else {
      hideLoupe();
    }
    if (state.hoverOptimizedPoint) {
      updateOptimizedPointInspector();
    } else {
      clearOptimizedPointInspector();
    }
    draw();
  }

  if (languageSwitch) {
    languageSwitch.value = currentLanguage;
    languageSwitch.addEventListener('change', () => {
      const nextLanguage = languageSwitch.value;
      if (!SUPPORTED_LANGUAGES.includes(nextLanguage)) {
        return;
      }
      currentLanguage = nextLanguage;
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLanguage);
      refreshLocalizedUi();
    });
  }

  async function loadCostmap() {
    setStatus(t('status.loadingCostmap'), '');
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
      setStatus(t('status.costmapLoaded'), '');
      runPlanning({reason: 'initial'});
    } catch (error) {
      setStatus(t('status.costmapLoadFailed', {message: error.message}), 'error');
    }
  }

  loadCostmap();
});
