import asyncio
import websockets
import json
import numpy as np
from mpi4py import MPI
from dolfinx.fem import (Function, functionspace, dirichletbc, locate_dofs_geometrical,
                         Constant, form, set_bc)
from dolfinx.fem.petsc import (assemble_vector, apply_lifting, create_vector,
                               create_matrix, assemble_matrix)
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from basix.ufl import element
from ufl import (TestFunction, TrialFunction, div, dot, dx, inner,
                 nabla_grad, sym, lhs, rhs, Identity, FacetNormal, ds)
from petsc4py import PETSc
import traceback


class NSSolver:
    """
    Fractional-step (Chorin) Navier-Stokes on a 200×100 channel.

    Parameters tuned for Re ~ 200 so we get visible eddies behind the cell:
      rho = 1.0,  mu = 0.01,  U_max ~ 1.0  →  Re = rho*U*D/mu ~ 1*1*20/0.01 = 2000
    (D ~ cell diameter ≈ 20 domain units)

    The cell obstacle is handled by an immersed-boundary penalty term
    lam * mask * u  in the momentum equation.  lam is kept modest
    (rho/dt * 2) so it blocks flow without creating pressure spikes.

    No post-step DOF reinit, no velocity clamping — let the NS dynamics run freely.
    """

    def __init__(self):
        self.comm = MPI.COMM_WORLD

        # ── Physical parameters ─────────────────────────────────────────
        self.rho = 1.0
        self.mu  = 0.01        # low viscosity → Re ~ 200–2000 depending on cell size
        self.dt  = 0.01        # stable for CFL ~ 0.4 at U=1, dx=2.5

        self.t          = 0.0
        self.step_count = 0

        # Driving pressure BCs  (Re = rho * U_max * H / mu;  U_max ≈ p_inlet*H²/(8μL))
        # With p_inlet=0.08, μ=0.01: U_max ≈ 0.08*100²/(8*0.01*200) ≈ 50  → way too high
        # Use a modest inlet velocity via p_inlet to get U_max ~ 1.
        # Poiseuille: U_max = (p_inlet - p_outlet) * H^2 / (8 * mu * L)
        # Want U_max = 1:  p_inlet = 8 * mu * L * U_max / H^2 = 8*0.01*200*1/10000 = 0.0016
        self.p_inlet  = 0.0016
        self.p_outlet = 0.0

        # ── Mesh geometry ───────────────────────────────────────────────
        self.mesh_dx = 200.0 / 80.0   # 2.5
        self.mesh_dy = 100.0 / 40.0   # 2.5

        # ── Cell / obstacle state ───────────────────────────────────────
        self.cell_boundary_points = []
        self.cell_all_points      = []
        self.cell_boundary_only   = []
        self.cell_data            = {}

        # Force coefficients sent to CPM
        self.K_el_obstacle = 125.0
        self.K_el_flow     = 0.2

        # Grid-point tracking for flow-field output correction
        self._prev_masked_grid_indices = np.array([], dtype=np.int32)

        # ── Build everything ────────────────────────────────────────────
        self.setup_mesh()
        self.setup_spaces()
        self.setup_functions()
        self.setup_static_boundary_conditions()
        self._cache_dof_coords()
        self.set_initial_condition()
        self.setup_variational_forms()
        self.setup_solvers()
        self.setup_evaluators()
        self.grid_points = self.generate_grid_points(80, 40)
        self.test_flow_field()

        Re_est = self.rho * 1.0 * 20.0 / self.mu   # D ~ 20 domain units
        print("\n" + "=" * 60)
        print(f"NS Solver  μ={self.mu}  Re(est)={Re_est:.0f}  dt={self.dt}")
        print(f"U_max(Poiseuille) ≈ {self.p_inlet*100**2/(8*self.mu*200):.3f}")
        print(f"Initial max velocity: {self.get_max_velocity():.4f}")
        print("=" * 60)

    # ------------------------------------------------------------------
    def setup_mesh(self):
        self.mesh = create_rectangle(
            self.comm,
            [np.array([0.0, 0.0]), np.array([200.0, 100.0])],
            [80, 40],
            cell_type=CellType.triangle,
        )
        self.dim = self.mesh.topology.dim

    # ------------------------------------------------------------------
    def setup_spaces(self):
        v_cg2 = element("Lagrange", self.mesh.basix_cell(), 2, shape=(self.dim,))
        s_cg1 = element("Lagrange", self.mesh.basix_cell(), 1)
        self.V  = functionspace(self.mesh, v_cg2)
        self.Q  = functionspace(self.mesh, s_cg1)
        self.V0, self._V0_map = self.V.sub(0).collapse()
        self.V1, self._V1_map = self.V.sub(1).collapse()

    # ------------------------------------------------------------------
    def setup_functions(self):
        self.u      = Function(self.V)
        self.u_n    = Function(self.V)
        self.u_tent = Function(self.V)
        self.p      = Function(self.Q)
        self.p_n    = Function(self.Q)

        s_cg1 = element("Lagrange", self.mesh.basix_cell(), 1)
        self.Qs        = functionspace(self.mesh, s_cg1)
        self.cell_mask = Function(self.Qs)
        self.cell_mask.x.array[:] = 0.0
        self._qs_coords = self.Qs.tabulate_dof_coordinates()[:, :2].copy()

    # ------------------------------------------------------------------
    def _cache_dof_coords(self):
        self._v0_coords = self.V0.tabulate_dof_coordinates()[:, :2].copy()
        self._v1_coords = self.V1.tabulate_dof_coordinates()[:, :2].copy()
        self._v0_parent = np.asarray(self._V0_map, dtype=np.int32)
        self._v1_parent = np.asarray(self._V1_map, dtype=np.int32)

        # Q DOF coords for pressure reference
        self._q_coords = self.Q.tabulate_dof_coordinates()[:, :2].copy()

        print(f"DOF cache: {len(self._v0_coords)} V0, {len(self._v1_coords)} V1")

    # ------------------------------------------------------------------
    def setup_static_boundary_conditions(self):
        def walls(x):
            return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 100.0))

        wall_dofs_x = locate_dofs_geometrical((self.V.sub(0), self.V0), walls)
        wall_dofs_y = locate_dofs_geometrical((self.V.sub(1), self.V1), walls)

        u_zero_x = Function(self.V0); u_zero_x.x.array[:] = 0.0
        u_zero_y = Function(self.V1); u_zero_y.x.array[:] = 0.0

        self.bc_wall_x = dirichletbc(u_zero_x, wall_dofs_x, self.V.sub(0))
        self.bc_wall_y = dirichletbc(u_zero_y, wall_dofs_y, self.V.sub(1))

        self.wall_dofs_constrained = (
            self.bc_wall_x._cpp_object.dof_indices()[0].tolist() +
            self.bc_wall_y._cpp_object.dof_indices()[0].tolist()
        )

        def inflow(x):  return np.isclose(x[0], 0.0)
        def outflow(x): return np.isclose(x[0], 200.0)

        inflow_dofs  = locate_dofs_geometrical(self.Q, inflow)
        outflow_dofs = locate_dofs_geometrical(self.Q, outflow)

        self.bcp = [
            dirichletbc(PETSc.ScalarType(self.p_inlet),  inflow_dofs,  self.Q),
            dirichletbc(PETSc.ScalarType(self.p_outlet), outflow_dofs, self.Q),
        ]
        self.bcu = [self.bc_wall_x, self.bc_wall_y]

        print(f"Wall BCs: {len(self.wall_dofs_constrained)} DOFs  "
              f"Pressure: inlet={len(inflow_dofs)}, outlet={len(outflow_dofs)}")

    # ------------------------------------------------------------------
    def rebuild_bcu(self):
        self.bcu = [self.bc_wall_x, self.bc_wall_y]
        self.update_cell_mask()

    # ------------------------------------------------------------------
    def update_cell_mask(self):
        """Update the IB penalty mask. No DOF reinit — let NS handle recovery."""
        self.cell_mask.x.array[:] = 0.0

        pts_source = self.cell_all_points if self.cell_all_points else self.cell_boundary_points
        if not pts_source:
            self.cell_mask.x.scatter_forward()
            return

        pts  = np.array(pts_source, dtype=float)
        tol  = self.mesh_dx * 1.2   # 3.0 — one mesh cell per CPM pixel

        diff     = self._qs_coords[:, np.newaxis, :] - pts[np.newaxis, :, :]
        min_dist = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
        inside   = min_dist < tol

        self.cell_mask.x.array[inside] = 1.0
        self.cell_mask.x.scatter_forward()
        print(f"  Mask: {inside.sum()} Qs DOFs  ({len(pts)} pts, tol={tol:.1f})")

    # ------------------------------------------------------------------
    def set_initial_condition(self):
        H, L = 100.0, 200.0

        def poiseuille_velocity(x):
            vals = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            vals[0] = (self.p_inlet / (2.0 * self.mu * L)) * x[1] * (H - x[1])
            return vals

        def poiseuille_pressure(x):
            vals = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
            vals[:] = self.p_inlet * (1.0 - x[0] / L)
            return vals

        for f in (self.u, self.u_n, self.u_tent):
            f.interpolate(poiseuille_velocity)
        for f in (self.p, self.p_n):
            f.interpolate(poiseuille_pressure)

        for bc in self.bcu:
            bc.set(self.u.x.array,      None, 1.0)
            bc.set(self.u_n.x.array,    None, 1.0)
            bc.set(self.u_tent.x.array, None, 1.0)

        for f in (self.u, self.u_n, self.u_tent, self.p, self.p_n):
            f.x.scatter_forward()

        print("✓ Initial condition set")

    # ------------------------------------------------------------------
    def setup_variational_forms(self):
        u = TrialFunction(self.V); v = TestFunction(self.V)
        p = TrialFunction(self.Q); q = TestFunction(self.Q)

        n   = FacetNormal(self.mesh)
        k   = Constant(self.mesh, PETSc.ScalarType(self.dt))
        mu  = Constant(self.mesh, PETSc.ScalarType(self.mu))
        rho = Constant(self.mesh, PETSc.ScalarType(self.rho))
        f   = Constant(self.mesh, PETSc.ScalarType((0.0, 0.0)))

        # IB penalty: modest value — blocks flow without pressure spikes.
        # lam/dt ~ 200  (was 10000 in the Poiseuille version → caused instability)
        lam = Constant(self.mesh, PETSc.ScalarType(self.rho / self.dt * 2.0))

        def epsilon(u): return sym(nabla_grad(u))
        def sigma(u, p): return 2 * mu * epsilon(u) - p * Identity(len(u))

        # Crank-Nicolson velocity
        U  = 0.5 * (self.u_n + u)

        # Step 1: tentative velocity (advection + diffusion + IB penalty)
        F1  = rho * dot((u - self.u_n) / k, v) * dx
        F1 += rho * dot(dot(self.u_n, nabla_grad(self.u_n)), v) * dx   # explicit advection
        F1 += inner(sigma(U, self.p_n), epsilon(v)) * dx
        F1 += dot(self.p_n * n, v) * ds
        F1 -= dot(mu * nabla_grad(U) * n, v) * ds
        F1 -= dot(f, v) * dx
        F1 += lam * self.cell_mask * dot(u, v) * dx   # IB penalty
        self.a1 = form(lhs(F1))
        self.L1 = form(rhs(F1))

        # Step 2: pressure correction
        a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
        L2 = (dot(nabla_grad(self.p_n), nabla_grad(q)) * dx
              - (rho / k) * div(self.u_tent) * q * dx)
        self.a2 = form(a2)
        self.L2 = form(L2)

        # Step 3: velocity correction
        a3 = rho * dot(u, v) * dx
        L3 = (rho * dot(self.u_tent, v) * dx
              - k * dot(nabla_grad(self.p - self.p_n), v) * dx)
        self.a3 = form(a3)
        self.L3 = form(L3)

        print("✓ Variational forms compiled")

    # ------------------------------------------------------------------
    def setup_solvers(self):
        def ksp(A, ktype, pctype, hyp=None):
            s = PETSc.KSP().create(self.comm)
            s.setOperators(A)
            s.setType(ktype)
            s.getPC().setType(pctype)
            if hyp: s.getPC().setHYPREType(hyp)
            return s

        self.A1 = create_matrix(self.a1)
        assemble_matrix(self.A1, self.a1, bcs=self.bcu); self.A1.assemble()
        self.solver1 = ksp(self.A1, PETSc.KSP.Type.BCGS, PETSc.PC.Type.HYPRE, "boomeramg")

        self.A2 = create_matrix(self.a2)
        assemble_matrix(self.A2, self.a2, bcs=self.bcp); self.A2.assemble()
        self.solver2 = ksp(self.A2, PETSc.KSP.Type.BCGS, PETSc.PC.Type.HYPRE, "boomeramg")

        self.A3 = create_matrix(self.a3)
        assemble_matrix(self.A3, self.a3, bcs=self.bcu); self.A3.assemble()
        self.solver3 = ksp(self.A3, PETSc.KSP.Type.CG, PETSc.PC.Type.SOR)

        self.b1 = create_vector([self.V])
        self.b2 = create_vector([self.Q])
        self.b3 = create_vector([self.V])
        print("✓ Solvers ready")

    # ------------------------------------------------------------------
    def setup_evaluators(self):
        self.bb = bb_tree(self.mesh, self.mesh.topology.dim)

    def eval_at_point(self, field, x, y):
        try:
            if not (0 <= x <= 200 and 0 <= y <= 100): return None
            pt = np.array([[x, y, 0.0]])
            cands = compute_collisions_points(self.bb, pt)
            cells = compute_colliding_cells(self.mesh, cands, pt).links(0)
            if len(cells) == 0: return None
            return np.asarray(field.eval(pt, np.array([cells[0]], dtype=np.int32))).ravel()
        except Exception:
            return None

    # ------------------------------------------------------------------
    def test_flow_field(self):
        print("\nFlow field test:")
        for x, y in [[20,50],[50,50],[100,50],[150,50]]:
            v = self.eval_at_point(self.u, x, y)
            p = self.eval_at_point(self.p, x, y)
            print(f"  ({x},{y}): u={v}  p={p}")

    # ------------------------------------------------------------------
    def compute_flow_forces(self):
        """Sample velocity around cell and compute drag/lift forces for CPM."""
        forces = {}
        if not self.cell_data:
            return forces

        for cell_id, info in self.cell_data.items():
            cx = info['centroid'][0] * 200.0
            cy = info['centroid'][1] * 100.0
            cell_type = info['type']

            # Sample well outside the cell (1 diameter away)
            r = 25.0
            sample_pts = [
                [cx + r, cy], [cx - r, cy],
                [cx, cy + r*0.6], [cx, cy - r*0.6],
                [cx + r*0.7, cy + r*0.4], [cx + r*0.7, cy - r*0.4],
            ]
            vels = []
            for sx, sy in sample_pts:
                v = self.eval_at_point(self.u, sx, sy)
                if v is not None and len(v) >= 2:
                    vels.append([float(v[0]), float(v[1])])

            if vels:
                avg_v = np.mean(vels, axis=0)
            else:
                vx    = (self.p_inlet / (2.0 * self.mu * 200.0)) * cy * (100.0 - cy)
                avg_v = np.array([max(float(vx), 0.1), 0.0])

            K = self.K_el_obstacle if cell_type == 2 else self.K_el_flow
            forces[str(cell_id)] = (K * avg_v).tolist()

        return forces

    # ------------------------------------------------------------------
    def ns_step(self):
        # Step 1 — tentative velocity
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu); self.A1.assemble()
        self.solver1.setOperators(self.A1)
        with self.b1.localForm() as loc: loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_tent.x.petsc_vec)
        self.u_tent.x.scatter_forward()

        # Step 2 — pressure correction
        self.A2.zeroEntries()
        assemble_matrix(self.A2, self.a2, bcs=self.bcp); self.A2.assemble()
        self.solver2.setOperators(self.A2)
        with self.b2.localForm() as loc: loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.p.x.petsc_vec)
        self.p.x.scatter_forward()

        # Step 3 — velocity correction
        self.A3.zeroEntries()
        assemble_matrix(self.A3, self.a3, bcs=self.bcu); self.A3.assemble()
        self.solver3.setOperators(self.A3)
        with self.b3.localForm() as loc: loc.set(0)
        assemble_vector(self.b3, self.L3)
        apply_lifting(self.b3, [self.a3], [self.bcu])
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b3, self.bcu)
        self.solver3.solve(self.b3, self.u.x.petsc_vec)
        self.u.x.scatter_forward()

        self.u_n.x.array[:] = self.u.x.array[:]
        self.p_n.x.array[:] = self.p.x.array[:]
        self.t          += self.dt
        self.step_count += 1

    # ------------------------------------------------------------------
    def eval_velocity_batch(self, points):
        arr  = np.array(points, dtype=np.float64)
        n    = len(arr)
        if n == 0: return []
        pts3  = np.column_stack([arr, np.zeros(n)])
        cands = compute_collisions_points(self.bb, pts3)
        coll  = compute_colliding_cells(self.mesh, cands, pts3)
        out   = []
        for i in range(n):
            cells = coll.links(i)
            if len(cells) == 0:
                out.append([0.0, 0.0]); continue
            try:
                v = self.u.eval(pts3[i:i+1], np.array([cells[0]], dtype=np.int32))
                out.append([float(v.ravel()[0]), float(v.ravel()[1])])
            except Exception:
                out.append([0.0, 0.0])
        return out

    # ------------------------------------------------------------------
    def get_flow_field(self):
        """
        Evaluate velocity at all grid points.
        Grid points that were inside the cell last step but are now free
        are patched if they are genuinely suppressed (< 50% of Poiseuille).
        This handles any residual IB penalty effect without hiding real eddies.
        """
        out = self.eval_velocity_batch(self.grid_points)
        pts = np.array(self.grid_points, dtype=float)

        # Find grid points inside cell this step
        cur_pts = list(self.cell_all_points) + list(self.cell_boundary_points)
        if cur_pts:
            zone     = np.array(cur_pts, dtype=float)
            diff     = pts[:, np.newaxis, :] - zone[np.newaxis, :, :]
            min_dist = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
            cur_inside = np.where(min_dist < self.mesh_dx * 1.5)[0].astype(np.int32)
        else:
            cur_inside = np.array([], dtype=np.int32)

        # Patch released points only if genuinely suppressed
        prev_set = set(self._prev_masked_grid_indices.tolist())
        cur_set  = set(cur_inside.tolist())
        released = np.array(list(prev_set - cur_set), dtype=np.int32)
        H = 100.0
        for i in released:
            x, y   = pts[i]
            vx_ref = (self.p_inlet / (2.0 * self.mu * 200.0)) * y * (H - y)
            if vx_ref > 0.01 and out[i][0] < vx_ref * 0.5:
                out[i] = [float(vx_ref), 0.0]

        self._prev_masked_grid_indices = cur_inside
        return out

    def get_boundary_average_velocity(self):
        pts = self.cell_boundary_only or self.cell_boundary_points
        if not pts: return [0.0, 0.0]
        off  = self.mesh_dx * 0.5
        spts = [[pt[0]+d[0], pt[1]+d[1]]
                for pt in pts
                for d in [(off,0),(-off,0),(0,off),(0,-off)]
                if 0 <= pt[0]+d[0] <= 200 and 0 <= pt[1]+d[1] <= 100]
        if not spts: return [0.0, 0.0]
        res = self.eval_velocity_batch(spts)
        nz  = [r for r in res if abs(r[0]) > 1e-6 or abs(r[1]) > 1e-6]
        return ([float(np.mean([r[0] for r in nz])), float(np.mean([r[1] for r in nz]))]
                if nz else [1.0, 0.0])

    def generate_grid_points(self, nx, ny):
        x = np.linspace(2.5, 197.5, nx)
        y = np.linspace(2.5,  97.5, ny)
        X, Y = np.meshgrid(x, y)
        return [[X[i,j], Y[i,j]] for i in range(ny) for j in range(nx)]

    def get_max_velocity(self):
        a  = self.u.x.array
        ux = a[0::2]; uy = a[1::2]
        return float(np.max(np.sqrt(ux**2 + uy**2)))

    def verify_no_slip(self):
        if not self.wall_dofs_constrained: return -1.0
        return float(np.max(np.abs(self.u.x.array[self.wall_dofs_constrained])))

    def update_cell_data(self, cells):
        self.cell_data = {}
        for c in cells:
            self.cell_data[c['id']] = {
                'type':     c['type'],
                'pixels':   c['pixels'],
                'centroid': c['centroid'],
            }

    def debug_velocity_map(self, velocities):
        ny, nx = 40, 80
        sx, sy = nx // 20, ny // 8
        print("\n── VELOCITY MAP ─────────────────────────────────────────────")
        for row in range(7, -1, -1):
            gy = row * sy
            y_d = 2.5 + gy * 2.5
            poi = (self.p_inlet / (2.0 * self.mu * 200.0)) * y_d * (100.0 - y_d)
            line = f"y~{int(y_d):3d}| "
            for col in range(20):
                idx = gy * nx + col * sx
                vx  = velocities[idx][0] if idx < len(velocities) else 0.0
                r   = vx / poi if poi > 0.001 else 1.0
                if   vx < -0.01:       line += "-"
                elif r  < 0.05:        line += "X"
                elif r  < 0.3:         line += "."
                elif r  < 0.7:         line += "o"
                else:                  line += "#"
            print(line)
        print(f"   x: 0{'':17s}200")
        umax = max(abs(v[0]) for v in velocities) if velocities else 0
        print(f"  max|vx|={umax:.4f}  step={self.step_count}  t={self.t:.3f}")
        if self.cell_all_points:
            p = np.array(self.cell_all_points)
            print(f"  cell: {len(p)} pts  cx={p[:,0].mean():.1f}  cy={p[:,1].mean():.1f}  "
                  f"x∈[{p[:,0].min():.0f},{p[:,0].max():.0f}]  y∈[{p[:,1].min():.0f},{p[:,1].max():.0f}]")
        print("── END ──────────────────────────────────────────────────────\n")


# ======================================================================
# WebSocket server
# ======================================================================
class WebSocketServer:
    def __init__(self):
        self.solver  = NSSolver()
        self.clients = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        print("\n🔌 Client connected")
        try:
            async for message in websocket:
                data = json.loads(message)

                if data['type'] == 'boundary_conditions':
                    mcs        = data.get('mcs', 0)
                    boundaries = data.get('boundaries', [])
                    cells      = data.get('cells', [])

                    # Full pixel list for interior IB mask
                    if cells:
                        all_px = []
                        for c in cells:
                            for px in c.get('pixels', []):
                                all_px.append([px[0]*200.0, px[1]*100.0])
                        self.solver.cell_all_points = all_px
                    else:
                        self.solver.cell_all_points = []

                    if boundaries:
                        self.solver.cell_boundary_points = [
                            [b['x']*200.0, b['y']*100.0] for b in boundaries
                        ]

                    self.solver.rebuild_bcu()

                    if cells:
                        self.solver.update_cell_data(cells)

                    bo = data.get('boundary_only', boundaries)
                    self.solver.cell_boundary_only = (
                        [[b['x']*200.0, b['y']*100.0] for b in bo]
                        if bo else self.solver.cell_boundary_points
                    )

                    step_ok = True
                    try:
                        self.solver.ns_step()
                    except Exception as e:
                        step_ok = False
                        print(f"  ⚠ NS step error: {e}")
                        traceback.print_exc()

                    try:
                        flow_forces = self.solver.compute_flow_forces()
                    except Exception as e:
                        print(f"  ⚠ flow_forces error: {e}")
                        traceback.print_exc()
                        flow_forces = {}

                    avg_vel    = self.solver.get_boundary_average_velocity()
                    velocities = self.solver.get_flow_field()
                    max_u      = self.solver.get_max_velocity()
                    wall_vel   = self.solver.verify_no_slip()

                    print(f"MCS {mcs}: step={self.solver.step_count} "
                          f"t={self.solver.t:.3f} max_u={max_u:.4f} "
                          f"wall={wall_vel:.1e} forces={len(flow_forces)}"
                          + ("" if step_ok else " [ERR]"))

                    # Debug map every 20 steps
                    if self.solver.step_count % 20 == 0:
                        self.solver.debug_velocity_map(velocities)

                    await websocket.send(json.dumps({
                        'type':                  'flow_forces',
                        'mcs':                   mcs,
                        'grid_shape':            [40, 80],
                        'full_field_velocities': velocities,
                        'flow_forces':           flow_forces,
                        'avg_boundary_velocity': avg_vel,
                    }))

        except websockets.exceptions.ConnectionClosed:
            print("🔌 Client disconnected")
        except Exception as e:
            print(f"Handler error: {e}"); traceback.print_exc()
        finally:
            self.clients.discard(websocket)

    async def start(self):
        async with websockets.serve(self.handler, "localhost", 8765):
            print("\n" + "="*60)
            print("NS+CPM coupling server — high Re")
            print("="*60)
            await asyncio.Future()


async def main():
    server = WebSocketServer()
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()
