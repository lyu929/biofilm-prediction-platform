"""
Simple analytic NS server — near-wall catheter window
======================================================
Physical setup
--------------
  Full catheter lumen  : H_full  = 5000 µm  (5 mm inner diameter, plate-plate model)
  CPM zoom window      : 0 – 200 µm from wall  (y_win = 200 µm)
  CPM grid             : 500 µm × 200 µm  →  1 CPM pixel = 1 µm
  Bacteria             : V = 12 px²  →  diameter ≈ 4 µm  ✓
  NS sample grid       : 80 × 40  (matches JS NS_GRID_NX / NS_GRID_NY)
  NS grid spacing      : 6.25 µm × 5 µm  (well above 1 µm px, fine for Poiseuille)

Flow profile
------------
  Full Poiseuille:  u(y) = u_max · 4y(H-y) / H²    (H = H_full = 5000 µm)
  Near-wall regime: u(y) ≈ u_max · 4y / H           (y << H, nearly linear)
  Normalisation:    U_ref = u(y_win) = u_max · 4·y_win·(H-y_win)/H²
  Returned velocity: u_norm(y) = u(y) / U_ref   →  0 at wall, 1.0 at window top

Y-axis convention (critical — must match JS)
--------------------------------------------
  JS canvas y=0   = window top  (fast flow, physical y = Y_WIN = 200 µm)
  JS canvas y=H-1 = wall        (no-slip,   physical y = 0 µm)
  JS sends norm_y = canvas_y / H  →  0 means fast, 1 means wall.
  Python recovers: y_phys = (1 - norm_y) * Y_WIN

Force scaling
-------------
  λ_eff = LAMBDA_BASE · u_norm · sqrt(cell_volume / V_REF_K2)
  V_REF_K2 = 12 px²  (target volume of a single K2 bacterium, matches JS config.conf.V[2])
  A lone bacterium at target size gets v_scaled = u_norm(centroid).
  A larger aggregate or cluster gets proportionally more force (sqrt scaling).

No FEM, no BCs, no penalty terms.
JSON response structure identical to full server — JS needs zero changes.
"""

import asyncio
import json
import traceback

import numpy as np
import websockets

# ── Physical constants ────────────────────────────────────────────────────────
H_FULL   = 5000.0    # µm  — full channel height
Y_WIN    = 200.0     # µm  — CPM window height (near-wall)
X_WIN    = 500.0     # µm  — CPM window width

# Normalisation reference: velocity at top of CPM window
def _u_raw(y_um):
    """Dimensional Poiseuille (arbitrary u_max=1 before normalisation)."""
    return 4.0 * y_um * (H_FULL - y_um) / (H_FULL ** 2)

U_REF = _u_raw(Y_WIN)   # ≈ 0.1584  (we divide everything by this)

# ── NS sample grid ────────────────────────────────────────────────────────────
NS_GRID_NX = 80
NS_GRID_NY = 40

# Grid cell centres in µm within the CPM window
_gx = np.linspace(X_WIN / (2 * NS_GRID_NX),
                  X_WIN - X_WIN / (2 * NS_GRID_NX),
                  NS_GRID_NX)           # x doesn't affect u in pure Poiseuille
_gy = np.linspace(Y_WIN / (2 * NS_GRID_NY),
                  Y_WIN - Y_WIN / (2 * NS_GRID_NY),
                  NS_GRID_NY)           # row 0 = near wall, row 39 = window top

# Pre-compute the full 80×40 velocity field (static — no cells, no obstacles)
# Layout: row-major, index = row * NX + col  (same as JS nsFlowField)
_FLOW_FIELD = []
for row in range(NS_GRID_NY):
    y = _gy[row]
    u_norm = _u_raw(y) / U_REF          # 0 at wall → 1.0 at y = Y_WIN
    for col in range(NS_GRID_NX):
        _FLOW_FIELD.append([float(u_norm), 0.0])   # [vx, vy]


def build_response(mcs: int,
                   cells: list,
                   cells_k2: list) -> dict:
    """
    Compute per-cell forces from the analytic flow field and return the
    same JSON structure the full FEM server returns.

    Obstacle cells  (cells, large clusters)  → flow_forces    [fx, fy]
    Free cells      (cells_k2, planktonic)   → velocity_forces [vx, vy]

    Force scaling
    -------------
    For free cells (velocity steering):
        The JS FlowForceConstraint.deltaH multiplies the velocity by
        LAMBDA_DIR_K2.  We pre-scale here so that LAMBDA_DIR_K2 is the
        tuning knob and this server just delivers a physically motivated
        magnitude:

            v_scaled = u_norm(centroid) · sqrt(vol / V_REF_K2)

        V_REF_K2 is the target volume for a single K2 bacterium (10 px²,
        matching config.conf.V[2]).  A planktonic cell at its target size
        gets v_scaled = u_norm; a larger aggregate gets proportionally more.

    For obstacle cells (pressure steering):
        We integrate the near-wall shear stress over the cell boundary
        as a simple proxy for pressure force.  The full FEM server does a
        proper pressure integral; here we use the local velocity gradient
        (= shear rate) as a stand-in:

            tau(y) = du/dy = u_max · 4(H - 2y) / H²
            tau_norm(y) = tau(y) / tau(y=0)   →  1 at wall

        The net x-force on a boundary segment pointing in the ±x direction
        from a pixel at height y is proportional to tau_norm(y) * sqrt(vol).
        We sum over all boundary pixels.
    """
    V_REF_K2  = 12.0    # target volume single bacterium (px²) — matches JS config.conf.V[2]

    # ── shear rate, normalised to 1 at the wall ───────────────────────────
    def _shear_norm(y_um: float) -> float:
        """du/dy normalised so wall shear = 1."""
        # du/dy = 4·u_max·(H - 2y) / H²  → at y=0: 4·u_max/H
        # normalise by wall value (4·u_max/H)
        return (H_FULL - 2.0 * y_um) / H_FULL

    def _u_norm_at(canvas_norm_y: float) -> float:
        """
        canvas_norm_y = canvas_y / H  in [0,1].
        canvas y=0 → window top (fast, physical y = Y_WIN).
        canvas y=1 → wall       (slow, physical y = 0).
        Invert to get physical y, then evaluate Poiseuille.
        """
        y_um = (1.0 - canvas_norm_y) * Y_WIN
        return float(_u_raw(y_um) / U_REF)

    # ── free-cell velocity forces ─────────────────────────────────────────
    velocity_forces = {}
    for cell in cells_k2:
        cid        = cell['id']
        pixels     = cell.get('pixels', [])
        n_px       = max(len(pixels), 1)
        vol        = float(n_px)

        # centroid[1] = canvas_y / H  (0 = window top/fast, 1 = wall/slow)
        cy_norm    = cell['centroid'][1]
        u_local    = _u_norm_at(cy_norm)

        size_scale = np.sqrt(vol / V_REF_K2)
        vx_scaled  = u_local * size_scale

        velocity_forces[str(cid)] = [float(vx_scaled), 0.0]

    # ── obstacle-cell flow forces ─────────────────────────────────────────
    flow_forces = {}
    for cell in cells:
        cid       = cell['id']
        pixels    = cell.get('pixels', [])
        borders   = cell.get('borders', pixels)
        n_px      = max(len(pixels), 1)
        vol       = float(n_px)
        size_scale = np.sqrt(vol / V_REF_K2)

        fx = 0.0
        for b in borders:
            # b = [norm_x, canvas_norm_y] in [0,1]×[0,1]
            # Invert canvas y: 0=top/fast → y_um=Y_WIN; 1=wall → y_um=0
            y_um    = (1.0 - b[1]) * Y_WIN
            tau     = _shear_norm(y_um)
            # Shear stress acts in +x (downstream) on a surface facing the flow.
            # Simple proxy: treat every border pixel as contributing +tau in x.
            fx     += tau

        # Normalise by border count so a single cell's force doesn't grow
        # just from having more border pixels at the same shear level,
        # then scale by sqrt(volume) for the physical drag term.
        n_border = max(len(borders), 1)
        fx = (fx / n_border) * size_scale

        flow_forces[str(cid)] = [float(fx), 0.0]

    return {
        'type':                  'flow_forces',
        'mcs':                   mcs,
        'grid_shape':            [NS_GRID_NY, NS_GRID_NX],
        'full_field_velocities': _FLOW_FIELD,
        'flow_forces':           flow_forces,
        'velocity_forces':       velocity_forces,
        'flux': {
            'inlet_flux':  round(sum(v[0] for v in _FLOW_FIELD[:NS_GRID_NY]) / NS_GRID_NY, 6),
            'outlet_flux': round(sum(v[0] for v in _FLOW_FIELD[:NS_GRID_NY]) / NS_GRID_NY, 6),
            'net_flux':    0.0,
        },
    }


# ── WebSocket server ──────────────────────────────────────────────────────────
class SimpleNSServer:
    def __init__(self):
        self._active = None
        print("\n" + "=" * 60)
        print("Simple analytic NS server")
        print(f"  Channel height : {H_FULL} µm")
        print(f"  CPM window     : {X_WIN} × {Y_WIN} µm  (near-wall)")
        print(f"  Grid           : {NS_GRID_NX} × {NS_GRID_NY}")
        print(f"  U_ref (norm)   : u at y={Y_WIN:.0f}µm = {U_REF:.5f} × u_max")
        print(f"  Profile at window top  : {_u_raw(Y_WIN)/U_REF:.3f}  (= 1.0 by def)")
        print(f"  Profile at window mid  : {_u_raw(Y_WIN/2)/U_REF:.3f}")
        print(f"  Profile at wall        : {_u_raw(0.0)/U_REF:.3f}  (= 0.0)")
        print("=" * 60)

    async def handler(self, websocket):
        if self._active is not None:
            try:
                await self._active.close(1001, "Replaced")
            except Exception:
                pass
        self._active = websocket
        print(f"\n🔌 Client connected from {websocket.remote_address}")

        try:
            async for message in websocket:
                if self._active is not websocket:
                    break
                try:
                    data   = json.loads(message)
                    if data.get('type') != 'boundary_conditions':
                        continue
                    mcs      = data.get('mcs', 0)
                    cells    = data.get('cells', [])
                    cells_k2 = data.get('cells_k2', [])

                    result = build_response(mcs, cells, cells_k2)

                    n_obs  = len(cells)
                    n_free = len(cells_k2)
                    if mcs % 50 == 0:
                        print(f"MCS {mcs:5d}  obstacle_cells={n_obs:3d}  "
                              f"free_cells={n_free:3d}")

                    await websocket.send(json.dumps(result))

                except json.JSONDecodeError as e:
                    print(f"  JSON error: {e}")
                except Exception as e:
                    print(f"  Handler error: {e}")
                    traceback.print_exc()

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Outer handler error: {e}")
            traceback.print_exc()
        finally:
            if self._active is websocket:
                self._active = None

    async def start(self):
        async with websockets.serve(
            self.handler, "localhost", 8765,
            ping_interval=30,
            ping_timeout=120,
            max_size=100_000_000,
        ):
            print("Listening on ws://localhost:8765")
            await asyncio.Future()


async def main():
    server = SimpleNSServer()
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Fatal: {e}")
        traceback.print_exc()
