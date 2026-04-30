// artistoo-worker.js
// Returns raw concentrations per cell — activation decisions are made
// in the main thread using per-cell AI accumulators with hysteresis.

let pixels = null;
let nextPixels = null;
let width, height;
let cfg = {};
let torusX, torusY;

function idx(x, y) {
    return y * width + x;
}

function wrapX(x) {
    if (torusX) return (x + width) % width;
    return x;
}

function wrapY(y) {
    if (torusY) return (y + height) % height;
    return y;
}

function inBounds(x, y) {
    return x >= 0 && x < width && y >= 0 && y < height;
}

function diffuse(rate) {
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const center = pixels[idx(x, y)];

            const xm = wrapX(x - 1);
            const xp = wrapX(x + 1);
            const ym = wrapY(y - 1);
            const yp = wrapY(y + 1);

            let neighborSum = 0;
            let neighborCount = 0;

            if (torusX || x > 0)          { neighborSum += pixels[idx(xm, y)]; neighborCount++; }
            if (torusX || x < width - 1)  { neighborSum += pixels[idx(xp, y)]; neighborCount++; }
            if (torusY || y > 0)          { neighborSum += pixels[idx(x, ym)]; neighborCount++; }
            if (torusY || y < height - 1) { neighborSum += pixels[idx(x, yp)]; neighborCount++; }

            nextPixels[idx(x, y)] = center + rate * (neighborSum - neighborCount * center);
        }
    }

    const tmp = pixels;
    pixels = nextPixels;
    nextPixels = tmp;
}

self.onmessage = function(e) {
    const { command, data } = e.data;

    switch (command) {
        case 'init':
            width  = data.width;
            height = data.height;
            cfg = data.config || {
                diffusionRate:  0.1,
                decayRate:      0.01,
                torus: [false, false]
            };

            torusX = Array.isArray(cfg.torus) ? cfg.torus[0] : !!cfg.torus;
            torusY = Array.isArray(cfg.torus) ? cfg.torus[1] : !!cfg.torus;

            pixels     = new Float32Array(width * height);
            nextPixels = new Float32Array(width * height);

            self.postMessage({ type: 'initialized' });
            break;

        case 'step': {
            if (!pixels) return;

            // 1. PRODUCTION
            const cellPositions = data.cellPositions || [];
            const cellRates     = data.cellRates     || [];
            for (let i = 0; i < cellPositions.length; i++) {
                const [x, y] = cellPositions[i];
                if (inBounds(x, y)) {
                    const rate = cellRates[i] !== undefined ? cellRates[i] : (cfg.productionRate || 1.0);
                    pixels[idx(x, y)] += rate;
                }
            }

            // 2. DIFFUSION
            diffuse(cfg.diffusionRate);

            // 3. ADVECTION — upwind finite difference on the flow velocity field.
            //    The NS grid is coarser than the CPM grid; we map each CPM pixel
            //    to its corresponding NS grid cell to get the local velocity.
            //    Upwind scheme: use upstream neighbour for the gradient to ensure
            //    stability (explicit advection with forward Euler).
            const flowField  = data.flowField;   // flat array [vx,vy, vx,vy, ...]
            const nsNX       = data.nsGridNX || 80;
            const nsNY       = data.nsGridNY || 40;
            const advScale   = data.advectionScale || 0.1;  // tune: fraction of velocity applied

            if (flowField && flowField.length > 0) {
                // Build advected buffer — we write into nextPixels (already zeroed from diffuse swap)
                // Actually after diffuse(), nextPixels is the OLD buffer — reuse it as scratch
                for (let i = 0; i < nextPixels.length; i++) nextPixels[i] = pixels[i];

                for (let py = 0; py < height; py++) {
                    for (let px = 0; px < width; px++) {
                        // Map CPM pixel to NS grid cell
                        const nsCol = Math.min(nsNX - 1, Math.floor((px / width)  * nsNX));
                        const nsRow = Math.min(nsNY - 1, Math.floor((py / height) * nsNY));
                        // NS grid is row-major: row 0 = bottom of domain
                        // CPM y=0 is top, y=height-1 is bottom (wall)
                        // NS row 0 = y=0 in domain = bottom = CPM y=height-1
                        // So NS row for CPM py: nsRow = (height-1-py)/height * nsNY
                        const nsRowFlipped = Math.min(nsNY - 1, Math.floor(((height - 1 - py) / height) * nsNY));
                        const nsIdx = (nsRowFlipped * nsNX + nsCol) * 2;
                        const vx = flowField[nsIdx]     || 0;
                        const vy = flowField[nsIdx + 1] || 0;

                        const c = pixels[idx(px, py)];

                        // Upwind advection: ∂c/∂x approximated from upstream direction
                        // If vx > 0, flow goes right → use left neighbour (px-1)
                        // If vx < 0, flow goes left  → use right neighbour (px+1)
                        let dcdx = 0;
                        if (vx > 0 && (torusX || px > 0)) {
                            dcdx = c - pixels[idx(wrapX(px - 1), py)];
                        } else if (vx < 0 && (torusX || px < width - 1)) {
                            dcdx = pixels[idx(wrapX(px + 1), py)] - c;
                        }

                        // Similarly for y — in CPM, y increases downward
                        // vy > 0 in NS domain means flow upward (away from bottom wall)
                        // which in CPM coords is decreasing py
                        let dcdy = 0;
                        if (vy > 0 && (torusY || py > 0)) {
                            dcdy = c - pixels[idx(px, wrapY(py - 1))];
                        } else if (vy < 0 && (torusY || py < height - 1)) {
                            dcdy = pixels[idx(px, wrapY(py + 1))] - c;
                        }

                        // Subtract advective flux: c -= (vx * dc/dx + vy * dc/dy) * scale
                        nextPixels[idx(px, py)] = Math.max(0,
                            c - advScale * (vx * dcdx + vy * dcdy)
                        );
                    }
                }

                // Copy advected result back to pixels
                for (let i = 0; i < pixels.length; i++) pixels[i] = nextPixels[i];
            }

            // 4. DECAY
            const decayFactor = 1 - cfg.decayRate;
            for (let i = 0; i < pixels.length; i++) {
                pixels[i] *= decayFactor;
            }

            // 5. SAMPLE concentrations at each cell's centroid
            const checkPositions = data.checkPositions || [];
            const checkCellIds   = data.checkCellIds   || [];
            const concentrations = new Array(checkPositions.length);

            for (let i = 0; i < checkPositions.length; i++) {
                const [x, y] = checkPositions[i];
                concentrations[i] = inBounds(x, y) ? pixels[idx(x, y)] : 0;
            }

            self.postMessage({
                type:           'step-complete',
                time:           data.time,
                concentrations,
                checkCellIds
            });
            break;
        }

        case 'get-field':
            self.postMessage({
                type: 'field',
                data: Array.from(pixels)
            });
            break;
    }
};

self.postMessage({ type: 'worker-ready' });
