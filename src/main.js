(function() {
// ============================================================
// CPM Biofilm Simulation — near-wall catheter window
// ============================================================
// Physical interpretation of the grid
// ------------------------------------
//   Domain  : 500 µm × 200 µm  (near-wall slice of a 5 mm catheter lumen)
//   Grid    : 500 × 200 CPM pixels  →  1 px = 1 µm
//   Bacteria: V = 12 px²  →  diameter ≈ 4 µm  ✓
//
//   x-axis (cols, 0→499) : along catheter wall, downstream (+x = flow direction)
//   y-axis (rows, 0→199) : perpendicular to wall
//                          y=0   → open flow (window top, 200 µm from wall)
//                          y=199 → catheter wall surface (y_phys = 0 µm)
//
// NOTE: Artistoo canvas has y=0 at the TOP of the canvas, so:
//   canvas bottom row (y = extents[1]-1) = catheter wall  ← barrier lives here
//   canvas top    row (y = 0)            = open flow window edge  ← cells exit here
//
// The NS server returns velocities indexed [row=0 … row=39] where
//   row 0  → near wall  (low velocity, u_norm ≈ 0)
//   row 39 → window top (u_norm = 1.0)
// Canvas y is INVERTED vs physical y, so we flip the NS row index
// when sampling or drawing the flow field (see sampleFlowVelocity).
//
// Spawning: uniform y across the open window (excluding wall + top margins).
// Flux of suspended cells onto the inlet plane ∝ u(y) × concentration,
// so with a homogeneous upstream suspension, more cells arrive at mid-window
// (higher u) than near the wall (u → 0). Uniform random y is a reasonable
// first approximation; biasing toward the wall would be physically wrong.
// ============================================================

// ============ GLOBAL VARIABLES ============
let sim, meter, gm;
let cellBirthTimes = {};
let cellTimers     = {};
let cellAI         = {};
let workerReady    = false;

let pendingSnapshots = new Map();
const MAX_PENDING_STEPS = 20;

let nsSocket    = null;
let nsConnected = false;
let nsFlowField = null;
let _nsResolve  = null;

const NS_GRID_NX = 80;
const NS_GRID_NY = 40;

let flowForceConstraint = null;
let overlayCanvas       = null;
let overlayCtx          = null;

// Database export is sent through the same local WebSocket as the NS solver.
// The Python server (src/ns_simple.py) writes these snapshots to MySQL/PostgreSQL
// using SQLAlchemy, so the browser never connects directly to the database.
const DB_EXPORT_CONFIG = {
    enabled: true,
    exportInterval: 10,
    attachmentDistance: 8,
    stableCoverageThreshold: 0.6,
    params: {
        flow_rate: 0.5,
        adhesion_wall: 0.6,
        adhesion_cell: 0.5,
        diffusion_rate: 0.25,
        signal_decay: 0.008,
        qs_threshold: 50,
        division_rate: 0.01,
        eps_rate: 0.02,
        runtime: 1000,
        random_seed: 1,
    }
};

let dbSimulationId = null;
let dbRunStarted = false;
let dbFlowFieldSent = false;
let dbLastExportTime = -1;
let dbTimeToFirstCluster = null;
let dbTimeToStableBiofilm = null;

// ============ CONFIGURATION ============
const config = {
    ndim: 2,
    field_size: [500, 200],          // 500 µm × 200 µm, 1 px = 1 µm
    conf: {
        torus: [false, false],
        seed: 1,
        T: 20,
        J: [
        //   bg   wall  k2    k3    k4
            [0,   20,   20,   20,   20  ],
            [20,  100,   5,    5,   50  ],
            [20,   5,   20,   20,   15  ],
            [20,   5,   20,   20,    5  ],
            [20,  50,   15,    5,    3  ],
        ],
        LAMBDA_V: [0, 1000, 50,   50,  80],
        //         bg  wall  k2   k3   k4
        // k2/k3: V=12 px²  → diameter ≈ 4 µm (1 px = 1 µm)
        // k4 EPS: V=24 px² → slightly larger blob, reasonable for EPS capsule
        // wall:  V=500 — wall is 1 px × 500 px wide, fills cleanly
        V:        [0,  500,  12,  12,   24],
    },
    simsettings: {
        NRCELLS: [0, 0, 0, 0],
        CANVASCOLOR: "eaecef",
        // kind 2 is displayed as light gray in the browser. Database export
        // separates it into planktonic vs attached by near-wall position.
        CELLCOLOR:   ["", "808080", "D3D3D3", "D62728", "7B3294"],
        ACTCOLOR:    [false, false, true, true, false],
        SHOWBORDERS: [false, false, true, true, true],
        zoom: 2,
        BURNIN: 500,
        RUNTIME: 1000,
        RUNTIME_BROWSER: "Inf",
        STATSOUT: { browser: false, node: true },
        LOGRATE: 10,
        parentElement: document.getElementById('sim-container')
    }
};

const SIGNAL_CONFIG = {
    productionRate:        8.0,
    diffusionRate:         0.25,
    decayRate:             0.008,
    exposureGain:          0.2,
    internalDecay:         0.002,
    productionRateK3:      30.0,
    activationThreshold:   50,
    deactivationThreshold: 5,
};

const DIVISION_CONFIG = {
    minDivisionAge:       600,
    divideProbK2:         0.005,
    divideVolumeThreshK2: 0.8,
    divideProbK3:         0.01,
    divideVolumeThreshK3: 0.8,
    epsProbK3:            0.02,
    epsVolumeThreshK3:    0.5,
};

// Pixel count threshold for treating a connected cluster as a flow obstacle.
// 1 px = 1 µm², single bacterium ≈ 12 µm².
// 200 px² ≈ ~17 cells touching → a genuine microcolony, not a lone cell pair.
const CLUSTER_THRESHOLD = 200;

// Boundary margins (in CPM pixels = µm) for deletion and spawning.
const OUTLET_MARGIN = 10;   // delete cells within 10 µm of right (downstream) edge
const TOP_MARGIN    = 5;    // delete cells within 5 µm of top (open-flow) edge
const SPAWN_X       = 5;    // µm from inlet — seed new cells just inside left edge

// ============ CLUSTER DETECTION ============
function computeClusters() {
    const pixelsByCell = sim.C.getStat(CPM.PixelsByCell);

    const pixelMap = new Map();
    for (const cellId in pixelsByCell) {
        const kind = sim.C.cellKind(Number(cellId));
        if (kind < 2 || kind > 4) continue;
        for (const [x, y] of pixelsByCell[cellId])
            pixelMap.set(`${x},${y}`, Number(cellId));
    }

    const parent = new Map();
    const rank   = new Map();

    function find(id) {
        if (parent.get(id) !== id) parent.set(id, find(parent.get(id)));
        return parent.get(id);
    }
    function unite(a, b) {
        a = find(a); b = find(b);
        if (a === b) return;
        if ((rank.get(a)||0) < (rank.get(b)||0)) [a,b] = [b,a];
        parent.set(b, a);
        if ((rank.get(a)||0) === (rank.get(b)||0))
            rank.set(a, (rank.get(a)||0) + 1);
    }

    for (const cellId in pixelsByCell) {
        const id = Number(cellId);
        if (sim.C.cellKind(id) < 2 || sim.C.cellKind(id) > 4) continue;
        parent.set(id, id); rank.set(id, 0);
    }

    const dirs = [[1,0],[-1,0],[0,1],[0,-1]];
    for (const [key, cellId] of pixelMap) {
        const [x, y] = key.split(',').map(Number);
        for (const [dx, dy] of dirs) {
            const nid = pixelMap.get(`${x+dx},${y+dy}`);
            if (nid !== undefined && nid !== cellId) unite(cellId, nid);
        }
    }

    const clusterPixelCount = new Map();
    for (const cellId in pixelsByCell) {
        const id   = Number(cellId);
        const kind = sim.C.cellKind(id);
        if (kind < 2 || kind > 4) continue;
        const root = find(id);
        clusterPixelCount.set(root,
            (clusterPixelCount.get(root)||0) + (pixelsByCell[cellId]||[]).length);
    }

    const clusterSizes = new Map();
    for (const cellId in pixelsByCell) {
        const id   = Number(cellId);
        const kind = sim.C.cellKind(id);
        if (kind < 2 || kind > 4) continue;
        clusterSizes.set(id, clusterPixelCount.get(find(id)) || 0);
    }

    return { clusterSizes, clusterRoot: (id) => find(id) };
}

// ============ FLOW FORCE CONSTRAINT ============
class FlowForceConstraint {
    constructor(C) {
        this.C          = C;
        this.name       = "FlowForceConstraint";
        this.cellForces = new Map();   // obstacle cells: Map<cellId, [fx,fy]>
        this.cellDirs   = new Map();   // free cells:     Map<cellId, [vx,vy]>
        this.LAMBDA_DIR_K2  = 500;
        this.FORCE_SCALE_K3 = 500;
        this.FORCE_SCALE_K4 = 500;
    }

    deltaH(sourcei, targeti, src_type, tgt_type) {
        const cellId = src_type;
        if (!cellId || cellId === 0) return 0;
        const kind = this.C.cellKind(cellId);
        if (kind < 2 || kind > 4) return 0;

        let s, t;
        try { s = this.C.grid.i2p(sourcei); t = this.C.grid.i2p(targeti); }
        catch(e) { return 0; }

        const dx = t[0] - s[0];
        const dy = t[1] - s[1];

        if (this.cellForces.has(cellId)) {
            const force = this.cellForces.get(cellId);
            const scale = (kind === 4) ? this.FORCE_SCALE_K4 : this.FORCE_SCALE_K3;
            return -scale * (force[0]*dx + force[1]*dy);
        }
        if (this.cellDirs.has(cellId)) {
            const vel = this.cellDirs.get(cellId);
            if (!vel || (vel[0]===0 && vel[1]===0)) return 0;
            return -this.LAMBDA_DIR_K2 * (vel[0]*dx + vel[1]*dy);
        }
        return 0;
    }

    setForces(forcesObj) {
        this.cellForces.clear();
        for (const [id, force] of Object.entries(forcesObj))
            this.cellForces.set(parseInt(id), force);
    }

    updateCellDir(cellId, vx, vy) {
        const mag = Math.sqrt(vx*vx + vy*vy);
        this.cellDirs.set(cellId, mag < 0.05 ? [0,0] : [vx,vy]);
    }

    removeCell(cellId) {
        this.cellForces.delete(cellId);
        this.cellDirs.delete(cellId);
    }
}

// ============ FLOW OVERLAY ============
function setupOverlay() {
    overlayCanvas = document.getElementById('flow-overlay');
    if (!overlayCanvas) return;
    const zoom = config.simsettings.zoom;
    overlayCanvas.width  = sim.C.extents[0] * zoom;
    overlayCanvas.height = sim.C.extents[1] * zoom;
    overlayCtx = overlayCanvas.getContext('2d');
}

function drawFlowOverlay() {
    if (!overlayCtx || !nsFlowField) return;
    const zoom  = config.simsettings.zoom;
    const W     = sim.C.extents[0];
    const H     = sim.C.extents[1];
    const cellW = (W * zoom) / NS_GRID_NX;
    const cellH = (H * zoom) / NS_GRID_NY;

    let maxSpeed = 0.001;
    for (const v of nsFlowField) {
        const s = Math.sqrt(v[0]*v[0]+v[1]*v[1]);
        if (s > maxSpeed) maxSpeed = s;
    }

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    for (let row = 0; row < NS_GRID_NY; row++) {
        for (let col = 0; col < NS_GRID_NX; col++) {
            const [vx, vy] = nsFlowField[row*NS_GRID_NX+col] || [0,0];
            const t = Math.sqrt(vx*vx+vy*vy) / maxSpeed;
            let r, g, b;
            if      (t < 0.25) { r=0;                             g=Math.floor(255*t/0.25);             b=255; }
            else if (t < 0.5)  { r=0;                             g=255;                                b=Math.floor(255*(1-(t-0.25)/0.25)); }
            else if (t < 0.75) { r=Math.floor(255*(t-0.5)/0.25); g=255;                                b=0; }
            else               { r=255;                            g=Math.floor(255*(1-(t-0.75)/0.25)); b=0; }

            // NS row 0 = near wall = canvas BOTTOM, so flip row for drawing
            const drawRow = NS_GRID_NY - 1 - row;
            overlayCtx.fillStyle = `rgba(${r},${g},${b},0.25)`;
            overlayCtx.fillRect(col*cellW, drawRow*cellH, cellW+1, cellH+1);
        }
    }
}

// ============ NS SERVER ============
function connectNS() {
    if (nsSocket && nsSocket.readyState === WebSocket.OPEN) return;
    try {
        nsSocket = new WebSocket('ws://localhost:8765');

        nsSocket.onopen = () => {
            nsConnected = true;
            console.log('NS server connected');
            document.getElementById('status').innerHTML =
                'NS server connected — simulation running';
            ensureDatabaseRunStarted();
        };

        nsSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'database_export_status') {
                    if (data.ok) console.log('Database export:', data.message, data.counts || {});
                    else console.error('Database export failed:', data.message);
                    return;
                }
                if (data.type === 'flow_forces') {
                    if (data.full_field_velocities) {
                        nsFlowField = data.full_field_velocities;
                        drawFlowOverlay();
                    }
                    if (flowForceConstraint) {
                        if (data.flow_forces)
                            flowForceConstraint.setForces(data.flow_forces);
                        if (data.velocity_forces) {
                            for (const [id, vel] of Object.entries(data.velocity_forces))
                                flowForceConstraint.updateCellDir(parseInt(id), vel[0], vel[1]);
                        }
                    }
                    if (data.flux) {
                        const f   = data.flux;
                        const pct = f.inlet_flux > 0
                            ? ((1 - f.outlet_flux/f.inlet_flux)*100).toFixed(1) : '0.0';
                        document.getElementById('status').innerHTML =
                            `Q_in: ${f.inlet_flux.toFixed(3)} &nbsp; ` +
                            `Q_out: ${f.outlet_flux.toFixed(3)} &nbsp; ` +
                            `Flow blocked: <strong>${pct}%</strong>`;
                    }
                    if (_nsResolve) { const r = _nsResolve; _nsResolve = null; r(); }
                }
            } catch(e) {
                console.error('NS message parse error:', e);
                if (_nsResolve) { _nsResolve(); _nsResolve = null; }
            }
        };

        nsSocket.onclose = () => {
            nsConnected = false;
            if (_nsResolve) { _nsResolve(); _nsResolve = null; }
        };

        nsSocket.onerror = (err) => {
            console.error('NS socket error', err);
            if (_nsResolve) { _nsResolve(); _nsResolve = null; }
        };

    } catch(e) { console.error('NS connect failed:', e); }
}

function sendCellDataToNS() {
    if (!nsConnected || !nsSocket || nsSocket.readyState !== WebSocket.OPEN)
        return Promise.resolve();

    const W = sim.C.extents[0];
    const H = sim.C.extents[1];
    const pixelsByCell = sim.C.getStat(CPM.PixelsByCell);
    const borderByCell = sim.C.getStat(CPM.BorderPixelsByCell);
    const centroids    = sim.C.getStat(CPM.Centroids);

    const { clusterSizes, clusterRoot } = computeClusters();

    const cellsPayload   = [];
    const cellsK2Payload = [];
    const allBoundaryPx  = [];

    for (const cellId in centroids) {
        const id        = Number(cellId);
        const kind      = sim.C.cellKind(id);
        if (kind < 2 || kind > 4) continue;

        const pixels    = pixelsByCell[cellId] || [];
        const borders   = borderByCell[cellId] || [];
        const centroid  = centroids[cellId];
        const clusterSz = clusterSizes.get(id) || 0;

        // Normalise to [0,1].
        // IMPORTANT: y is sent as canvas-y / H.
        // canvas y=0 = window top (fast flow), canvas y=H-1 = wall (no-slip).
        // Python server receives norm_y in [0,1] where 0=fast, 1=wall.
        // Python converts: y_phys_um = (1 - norm_y) * Y_WIN
        const normPixels   = pixels.map(p  => [p[0]/W, p[1]/H]);
        const normBorders  = borders.map(p => [p[0]/W, p[1]/H]);
        const normCentroid = [centroid[0]/W, centroid[1]/H];

        if (clusterSz >= CLUSTER_THRESHOLD) {
            cellsPayload.push({
                id:         id,
                type:       kind,
                pixels:     normPixels,
                borders:    normBorders,
                centroid:   normCentroid,
                cluster_id: clusterRoot(id),
            });
            for (const b of normBorders) allBoundaryPx.push({ x: b[0], y: b[1] });
        } else {
            cellsK2Payload.push({ id, pixels: normPixels, centroid: normCentroid });
        }
    }

    const p = new Promise(resolve => { _nsResolve = resolve; });
    nsSocket.send(JSON.stringify({
        type:          'boundary_conditions',
        mcs:           sim.time,
        boundaries:    allBoundaryPx,
        boundary_only: allBoundaryPx,
        cells:         cellsPayload,
        cells_k2:      cellsK2Payload,
    }));
    return p;
}

// ── sampleFlowVelocity ────────────────────────────────────────────────────────
// Maps a CPM canvas coordinate to the NS flow field.
//
// Canvas y=0    → window top  → NS row 39 (fast flow, u_norm ≈ 1)
// Canvas y=H-1  → wall        → NS row  0 (slow flow, u_norm ≈ 0)
//
// We flip the row so that cells near the wall see low velocity and cells
// near the window top see high velocity — matching physical reality.
function sampleFlowVelocity(cpmX, cpmY) {
    if (!nsFlowField) return null;
    const W = sim.C.extents[0];
    const H = sim.C.extents[1];

    const col = Math.min(NS_GRID_NX-1, Math.max(0,
        Math.floor((cpmX / W) * NS_GRID_NX)));

    // Flip y: canvas y=0 (top, fast) → NS row 39; canvas y=H-1 (wall, slow) → NS row 0
    const rowRaw = Math.floor(((H - 1 - cpmY) / H) * NS_GRID_NY);
    const row    = Math.min(NS_GRID_NY-1, Math.max(0, rowRaw));

    return nsFlowField[row*NS_GRID_NX + col] || [0.0, 0.0];
}

// ============ DATABASE EXPORT ============
function makeSimulationId() {
    const stamp = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14);
    const suffix = Math.random().toString(36).slice(2, 8);
    return `browser_${stamp}_${suffix}`;
}

function sendDatabaseMessage(payload) {
    if (!DB_EXPORT_CONFIG.enabled) return;
    if (!nsConnected || !nsSocket || nsSocket.readyState !== WebSocket.OPEN) return;
    try {
        nsSocket.send(JSON.stringify(payload));
    } catch(e) {
        console.warn("Database export send failed:", e);
    }
}

function ensureDatabaseRunStarted() {
    if (!DB_EXPORT_CONFIG.enabled || dbRunStarted) return;
    if (!nsConnected || !nsSocket || nsSocket.readyState !== WebSocket.OPEN) return;

    if (!dbSimulationId) dbSimulationId = makeSimulationId();
    const params = Object.assign({}, DB_EXPORT_CONFIG.params, {
        diffusion_rate: SIGNAL_CONFIG.diffusionRate,
        signal_decay: SIGNAL_CONFIG.decayRate,
        qs_threshold: SIGNAL_CONFIG.activationThreshold,
        division_rate: DIVISION_CONFIG.divideProbK3,
        eps_rate: DIVISION_CONFIG.epsProbK3,
        runtime: Number.isFinite(config.simsettings.RUNTIME_BROWSER)
            ? config.simsettings.RUNTIME_BROWSER
            : config.simsettings.RUNTIME,
        random_seed: config.conf.seed,
    });
    params.simulation_id = dbSimulationId;
    params.created_at = new Date().toISOString().slice(0, 19).replace("T", " ");

    sendDatabaseMessage({
        type: "simulation_start",
        simulation_id: dbSimulationId,
        simulation: params
    });
    dbRunStarted = true;
}

function stateFromKind(kind, y, H) {
    if (kind === 4) return "eps_producing";
    if (kind === 3) return "qs_active";
    if (kind === 2 && y >= H - DB_EXPORT_CONFIG.attachmentDistance) return "attached";
    if (kind === 2) return "planktonic";
    return "inactive";
}

function estimateFlowFieldRecords() {
    if (!nsFlowField || dbFlowFieldSent || !dbSimulationId) return [];
    const W = sim.C.extents[0];
    const H = sim.C.extents[1];
    const dx = W / NS_GRID_NX;
    const dy = H / NS_GRID_NY;
    const records = [];

    for (let row = 0; row < NS_GRID_NY; row++) {
        for (let col = 0; col < NS_GRID_NX; col++) {
            const idx = row * NS_GRID_NX + col;
            const [vx, vy] = nsFlowField[idx] || [0, 0];
            const nextRow = Math.min(NS_GRID_NY - 1, row + 1);
            const [vxNext] = nsFlowField[nextRow * NS_GRID_NX + col] || [vx, 0];
            records.push({
                simulation_id: dbSimulationId,
                x: (col + 0.5) * dx,
                y: (row + 0.5) * dy,
                velocity_x: vx,
                velocity_y: vy,
                shear: Math.abs(vxNext - vx) / Math.max(dy, 1e-9),
            });
        }
    }
    dbFlowFieldSent = true;
    return records;
}

function buildDatabaseSnapshot(currentTime) {
    if (!dbSimulationId) dbSimulationId = makeSimulationId();

    const W = sim.C.extents[0];
    const H = sim.C.extents[1];
    const centroids = sim.C.getStat(CPM.Centroids);
    const pixelsByCell = sim.C.getStat(CPM.PixelsByCell);
    const { clusterSizes, clusterRoot } = computeClusters();

    const cells = [];
    const clusterAgg = new Map();
    let totalCellCount = 0;
    let qsCount = 0;
    let epsCount = 0;
    let totalBiomassPixels = 0;
    let minY = H;
    const nearWallXs = new Set();
    const topByX = new Map();

    for (const cellId in centroids) {
        const id = Number(cellId);
        const kind = sim.C.cellKind(id);
        if (kind < 2 || kind > 4) continue;

        const centroid = centroids[cellId];
        const pixels = pixelsByCell[cellId] || [];
        const volume = pixels.length;
        const root = clusterRoot(id) || id;
        const clusterId = String(root);
        const state = stateFromKind(kind, centroid[1], H);

        totalCellCount++;
        totalBiomassPixels += volume;
        if (kind === 3) qsCount++;
        if (kind === 4) epsCount++;

        for (const [x, y] of pixels) {
            if (y < minY) minY = y;
            if (y >= H - DB_EXPORT_CONFIG.attachmentDistance) nearWallXs.add(Math.round(x));
            const key = Math.round(x);
            topByX.set(key, Math.min(topByX.get(key) ?? H, y));
        }

        cells.push({
            cell_id: String(id),
            simulation_id: dbSimulationId,
            timepoint: currentTime,
            x: centroid[0],
            y: centroid[1],
            state: state,
            volume: volume,
            local_signal: cellAI[id] || 0,
            cluster_id: clusterId,
        });

        if (!clusterAgg.has(clusterId)) {
            clusterAgg.set(clusterId, {
                cluster_id: clusterId,
                simulation_id: dbSimulationId,
                timepoint: currentTime,
                cluster_size: 0,
                pixel_count: 0,
                sum_x: 0,
                sum_y: 0,
                min_x: W,
                max_x: 0,
                min_y: H,
                max_y: 0,
            });
        }
        const agg = clusterAgg.get(clusterId);
        agg.cluster_size += 1;
        agg.pixel_count += volume;
        agg.sum_x += centroid[0];
        agg.sum_y += centroid[1];
        for (const [x, y] of pixels) {
            agg.min_x = Math.min(agg.min_x, x);
            agg.max_x = Math.max(agg.max_x, x);
            agg.min_y = Math.min(agg.min_y, y);
            agg.max_y = Math.max(agg.max_y, y);
        }
    }

    const clusters = [];
    for (const agg of clusterAgg.values()) {
        const pixelClusterSize = clusterSizes.get(Number(agg.cluster_id)) || agg.pixel_count;
        if (agg.cluster_size < 2 && pixelClusterSize < CLUSTER_THRESHOLD) continue;
        const bboxArea = Math.max(1, (agg.max_x - agg.min_x + 1) * (agg.max_y - agg.min_y + 1));
        clusters.push({
            cluster_id: agg.cluster_id,
            simulation_id: dbSimulationId,
            timepoint: currentTime,
            cluster_size: agg.cluster_size,
            center_x: agg.sum_x / Math.max(agg.cluster_size, 1),
            center_y: agg.sum_y / Math.max(agg.cluster_size, 1),
            density: agg.pixel_count / bboxArea,
        });
    }

    if (clusters.length > 0 && dbTimeToFirstCluster === null) dbTimeToFirstCluster = currentTime;

    const surfaceCoverage = nearWallXs.size / W;
    if (dbTimeToStableBiofilm === null && surfaceCoverage >= DB_EXPORT_CONFIG.stableCoverageThreshold) {
        dbTimeToStableBiofilm = currentTime;
    }

    const topHeights = Array.from(topByX.values()).map(y => H - y);
    const meanHeight = topHeights.length
        ? topHeights.reduce((a, b) => a + b, 0) / topHeights.length
        : 0;
    const roughness = topHeights.length
        ? Math.sqrt(topHeights.reduce((s, h) => s + Math.pow(h - meanHeight, 2), 0) / topHeights.length)
        : 0;
    const clusterSizesForSummary = clusters.map(c => c.cluster_size);

    const summary = {
        simulation_id: dbSimulationId,
        final_cell_count: totalCellCount,
        biofilm_area: totalBiomassPixels,
        biofilm_thickness: totalCellCount ? H - minY : 0,
        surface_coverage: surfaceCoverage,
        mean_cluster_size: clusterSizesForSummary.length
            ? clusterSizesForSummary.reduce((a, b) => a + b, 0) / clusterSizesForSummary.length
            : 0,
        max_cluster_size: clusterSizesForSummary.length ? Math.max(...clusterSizesForSummary) : 0,
        cluster_count: clusters.length,
        roughness: roughness,
        qs_active_ratio: totalCellCount ? qsCount / totalCellCount : 0,
        eps_fraction: totalCellCount ? epsCount / totalCellCount : 0,
        time_to_first_cluster: dbTimeToFirstCluster ?? 0,
        time_to_stable_biofilm: dbTimeToStableBiofilm ?? 0,
    };

    return {
        type: "simulation_snapshot",
        simulation_id: dbSimulationId,
        timepoint: currentTime,
        cells,
        clusters,
        flow_field: estimateFlowFieldRecords(),
        simulation_summary: summary,
    };
}

function maybeExportDatabaseSnapshot(currentTime) {
    if (!DB_EXPORT_CONFIG.enabled) return;
    if (currentTime === dbLastExportTime) return;
    if (currentTime % DB_EXPORT_CONFIG.exportInterval !== 0) return;
    ensureDatabaseRunStarted();
    if (!dbRunStarted) return;

    const snapshot = buildDatabaseSnapshot(currentTime);
    sendDatabaseMessage(snapshot);
    dbLastExportTime = currentTime;
}

// ============ DIFFUSION WORKER ============
let diffusionWorker_obj;

function initWorker() {
    try {
        diffusionWorker_obj = new Worker('/artistoo-worker.js');
        diffusionWorker_obj.onmessage = function(e) {
            switch(e.data.type) {
                case 'worker-ready':
                    diffusionWorker_obj.postMessage({
                        command: 'init',
                        data: {
                            width:  sim.C.extents[0],
                            height: sim.C.extents[1],
                            config: {
                                diffusionRate: SIGNAL_CONFIG.diffusionRate,
                                decayRate:     SIGNAL_CONFIG.decayRate,
                                torus:         sim.C.conf.torus
                            }
                        }
                    });
                    break;
                case 'initialized':
                    workerReady = true;
                    document.getElementById('status').innerHTML =
                        'Ready — connecting to NS server...';
                    break;
                case 'step-complete':
                    handleDiffusionComplete(e.data);
                    break;
            }
        };
        diffusionWorker_obj.onerror = (err) => console.error('Worker error:', err);
    } catch(e) { console.error('Failed to create worker:', e); }
}

function handleDiffusionComplete(data) {
    const { time, concentrations, checkCellIds } = data;
    const snapshot = pendingSnapshots.get(time);
    if (!snapshot) return;
    pendingSnapshots.delete(time);
    if (!concentrations || !checkCellIds) return;

    for (let i = 0; i < checkCellIds.length; i++) {
        const cellId    = checkCellIds[i];
        const fieldConc = concentrations[i] || 0;
        let kind;
        try { kind = sim.C.cellKind(cellId); } catch(e) { continue; }
        if (!kind || kind === 0) continue;

        if (cellAI[cellId] === undefined) cellAI[cellId] = 0;
        cellAI[cellId] *= (1 - SIGNAL_CONFIG.internalDecay);
        cellAI[cellId] += SIGNAL_CONFIG.exposureGain * fieldConc;
        cellAI[cellId]  = Math.max(0, cellAI[cellId]);

        if (kind === 2 && cellAI[cellId] >= SIGNAL_CONFIG.activationThreshold)
            sim.C.setCellKind(cellId, 3);
        if (kind === 3 && cellAI[cellId] < SIGNAL_CONFIG.deactivationThreshold)
            sim.C.setCellKind(cellId, 2);
    }
}

// ============ UTILITY ============
function createCatheterWall() {
    // Wall lives at canvas bottom row = physical catheter surface (y_phys = 0 µm).
    // Single wall cell spans the full 500 px width; V[1]=500 matches exactly.
    const wallCellId = sim.C.makeNewCellID(1);
    const wallY      = sim.C.extents[1] - 1;
    for (let x = 0; x < sim.C.extents[0]; x++)
        sim.C.setpix([x, wallY], wallCellId);
}

// ============ CELL CLEANUP ============
function cleanupCell(cellId) {
    delete cellBirthTimes[cellId];
    delete cellTimers[cellId];
    delete cellAI[cellId];
    flowForceConstraint.removeCell(cellId);
}

// ============ STEP LOOP ============
async function step() {
    sim.step();
    meter.tick();
    await sendCellDataToNS();
    if (sim.conf["RUNTIME_BROWSER"] == "Inf" ||
        sim.time + 1 < sim.conf["RUNTIME_BROWSER"]) {
        requestAnimationFrame(step);
    }
}

// ============ MAIN INITIALIZATION ============
function initialize() {
    document.getElementById('status').innerHTML = 'Initializing...';

    const custommethods = {

        postMCSListener: function() {
            if (!workerReady) return;

            const currentTime = this.time;

            for (const [t] of pendingSnapshots)
                if (currentTime - t > MAX_PENDING_STEPS) pendingSnapshots.delete(t);

            // ── Autoinducer diffusion ─────────────────────────────────────
            const centroids         = this.C.getStat(CPM.Centroids);
            const producerPositions = [];
            const checkEntries      = [];

            for (const cellId in centroids) {
                const kind = this.C.cellKind(Number(cellId));
                if (kind === 2) {
                    producerPositions.push({
                        position: centroids[cellId],
                        rate: SIGNAL_CONFIG.productionRate
                    });
                    checkEntries.push({
                        cellId: Number(cellId),
                        position: centroids[cellId]
                    });
                } else if (kind === 3) {
                    producerPositions.push({
                        position: centroids[cellId],
                        rate: SIGNAL_CONFIG.productionRateK3
                    });
                    checkEntries.push({
                        cellId: Number(cellId),
                        position: centroids[cellId]
                    });
                }
            }

            pendingSnapshots.set(currentTime, { checkEntries });
            diffusionWorker_obj.postMessage({
                command: 'step',
                data: {
                    cellPositions:  producerPositions.map(p => p.position),
                    cellRates:      producerPositions.map(p => p.rate),
                    checkPositions: checkEntries.map(e => e.position),
                    checkCellIds:   checkEntries.map(e => e.cellId),
                    time:           currentTime,
                    flowField:      nsFlowField ? nsFlowField.flat() : null,
                    nsGridNX:       NS_GRID_NX,
                    nsGridNY:       NS_GRID_NY,
                    advectionScale: 0.1
                }
            });

            // ── Cell division and EPS secretion ───────────────────────────
            this.divideAndSecrete();

            // ── Spawn planktonic cells at inlet ───────────────────────────
            this.spawnNewCells();

            // ── Delete cells at open boundaries ───────────────────────────
            this.deleteCellsAtBoundary();

            // ── Fallback: sample flow field when NS not connected ─────────
            if (!nsConnected && nsFlowField) {
                const freshCentroids = this.C.getStat(CPM.Centroids);
                const { clusterSizes } = computeClusters();
                for (const cellId in freshCentroids) {
                    const id = Number(cellId);
                    if (this.C.cellKind(id) < 2 || this.C.cellKind(id) > 4) continue;
                    if ((clusterSizes.get(id)||0) >= CLUSTER_THRESHOLD) continue;
                    const vel = sampleFlowVelocity(...freshCentroids[cellId]);
                    if (vel) flowForceConstraint.updateCellDir(id, vel[0], vel[1]);
                }
            }

            // Export standardized cells/clusters/summary records to DBeaver DB.
            maybeExportDatabaseSnapshot(currentTime);
        },

        divideAndSecrete: function() {
            const targetV2       = this.C.conf.V[2];
            const targetV3       = this.C.conf.V[3];
            const pixelsSnapshot = this.C.getStat(CPM.PixelsByCell);
            const cellIds        = Object.keys(pixelsSnapshot)
                .map(Number)
                .filter(id => pixelsSnapshot[id] && pixelsSnapshot[id].length > 0);

            for (const id of cellIds) {
                let kind;
                try { kind = this.C.cellKind(id); } catch(e) { continue; }
                if (kind < 2 || kind > 3) continue;

                const vol = this.C.getVolume(id);
                if (!vol || vol < 2) continue;

                if (cellTimers[id] === undefined) cellTimers[id] = 0;
                cellTimers[id]++;
                const matureEnough = cellTimers[id] >= DIVISION_CONFIG.minDivisionAge;

                if (kind === 2) {
                    if (matureEnough
                        && vol >= targetV2 * DIVISION_CONFIG.divideVolumeThreshK2
                        && this.C.random() < DIVISION_CONFIG.divideProbK2) {
                        let dId;
                        try { dId = gm.divideCell(id); } catch(e) { continue; }
                        if (!dId) continue;
                        cellTimers[id] = 0; cellTimers[dId] = 0;
                        cellBirthTimes[dId] = this.time;
                        try {
                            const cen = this.C.getStat(CPM.Centroids)[dId];
                            if (cen) {
                                const v = sampleFlowVelocity(cen[0], cen[1]);
                                if (v) flowForceConstraint.updateCellDir(dId, v[0], v[1]);
                            }
                        } catch(e) {}
                    }
                } else if (kind === 3) {
                    if (matureEnough
                        && vol >= targetV3 * DIVISION_CONFIG.divideVolumeThreshK3
                        && this.C.random() < DIVISION_CONFIG.divideProbK3) {
                        let dId;
                        try { dId = gm.divideCell(id); } catch(e) { continue; }
                        if (!dId) continue;
                        this.C.setCellKind(id, 2); this.C.setCellKind(dId, 2);
                        cellTimers[id] = 0; cellTimers[dId] = 0;
                        cellBirthTimes[dId] = this.time;
                    }
                    if (matureEnough
                        && vol >= targetV3 * DIVISION_CONFIG.epsVolumeThreshK3
                        && this.C.random() < DIVISION_CONFIG.epsProbK3) {
                        let dId;
                        try { dId = gm.divideCell(id); } catch(e) { continue; }
                        if (!dId) continue;
                        this.C.setCellKind(dId, 4);
                    }
                }
            }
        },

        spawnNewCells: function() {
            // Uniform y across the open window (TOP_MARGIN → wall-1).
            //
            // Flux of suspended cells arriving at the inlet plane = c × u(y).
            // With a homogeneous upstream suspension (uniform c), more cells
            // arrive per unit time at mid-window (high u) than near the wall
            // (u → 0).  Uniform random y is the simplest approximation of this;
            // biasing toward the wall would be physically backwards.
            //
            // Canvas y=0 = window top (fast); y=H-1 = wall (no-slip, barrier).
            // Safe spawn range: TOP_MARGIN  …  H-2  (leave wall row untouched).
            if (this.C.random() >= 0.05) return;

            const H    = this.C.extents[1];
            const yMin = TOP_MARGIN;
            const yMax = H - 13;       // one pixel clear of the wall row
            const y    = yMin + Math.floor(this.C.random() * (yMax - yMin));

            const newCellId = gm.seedCellAt(2, [SPAWN_X, y]);
            if (!newCellId) return;
            cellBirthTimes[newCellId] = this.time;
            cellTimers[newCellId]     = 0;
            const vel = sampleFlowVelocity(SPAWN_X, y);
            if (vel) flowForceConstraint.updateCellDir(newCellId, vel[0], vel[1]);
        },

        deleteCellsAtBoundary: function() {
            const W          = this.C.extents[0];
            const H          = this.C.extents[1];
            const rightEdge  = W - OUTLET_MARGIN;   // downstream outlet
            const topEdge    = TOP_MARGIN;           // open flow window boundary

            // NOTE: left edge (x≈0) and bottom edge (y≈H-1) are physical boundaries
            // (inlet and wall) — we never delete there.

            const centroids    = this.C.getStat(CPM.Centroids);
            const pixelsByCell = this.C.getStat(CPM.PixelsByCell);
            const toDelete     = [];

            for (const cellId in centroids) {
                const id   = Number(cellId);
                const kind = this.C.cellKind(id);
                if (kind < 2 || kind > 4) continue;

                const [cx, cy] = centroids[cellId];

                const atOutlet  = cx >= rightEdge;   // swept downstream out of window
                const atOpenTop = cy <= topEdge;     // lifted into open flow above window

                if (atOutlet || atOpenTop) toDelete.push(id);
            }

            for (const cellId of toDelete) {
                const pixels = pixelsByCell[cellId];
                if (pixels) for (const pixel of pixels) this.C.setpix(pixel, 0);
                cleanupCell(cellId);
            }
        }
    };

    sim = new CPM.Simulation(config, custommethods);

    sim.C.add(new CPM.BarrierConstraint({
        IS_BARRIER: [false, true, false, false, false]
    }));

    initWorker();

    flowForceConstraint = new FlowForceConstraint(sim.C);
    const _origDeltaH = sim.C.deltaH.bind(sim.C);
    sim.C.deltaH = function(sourcei, targeti, src_type, tgt_type) {
        return _origDeltaH(sourcei, targeti, src_type, tgt_type)
             + flowForceConstraint.deltaH(sourcei, targeti, src_type, tgt_type);
    };

    gm = new CPM.GridManipulator(sim.C);
    createCatheterWall();
    setupOverlay();

    meter = new FPSMeter({ left: "auto", right: "5px" });
    connectNS();

    document.getElementById('status').innerHTML = 'Simulation running';
    step();
}

window.initialize = initialize;

})();
