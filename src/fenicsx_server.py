import asyncio
import websockets
import json
import numpy as np
import signal as _signal
import traceback
from collections import defaultdict

# Must be before PETSc import so Python's SIG_IGN wins over PETSc's MPI_Abort handler.
_signal.signal(_signal.SIGPIPE, _signal.SIG_IGN)

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


class PoiseuilleNSSolver:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.mu   = 1.0
        self.rho  = 1.0
        self.dt   = 0.005
        self.t    = 0.0
        self.step_count = 0

        self.p_inlet  = 0.16
        self.p_outlet = 0.0

        self.cell_boundary_points = []
        self.cell_all_points      = []
        self.cell_boundary_only   = []
        self.cell_data            = {}

        # FEM domain: [0,200] x [0,100], mesh 80x40
        self.mesh_dx = 200.0 / 80.0   # 2.5 domain units per element
        self.mesh_dy = 100.0 / 40.0

        # CPM grid: 400 x 200 pixels
        # 1 CPM pixel = cpix_x domain units in x, cpix_y in y
        self.cpix_x   = 200.0 / 400.0   # 0.5
        self.cpix_y   = 100.0 / 200.0   # 0.5
        self.mask_tol = self.cpix_x * 1.5

        self._prev_masked_V_dofs       = np.array([], dtype=np.int32)
        self._prev_masked_grid_indices = np.array([], dtype=np.int32)

        self.setup_mesh()
        self.setup_spaces()
        self.setup_functions()
        self.setup_static_boundary_conditions()
        self._cache_dof_coords()
        self.set_initial_condition()
        self.setup_variational_forms()
        self.setup_solvers()
        self.setup_evaluators()
        self.test_flow_field()
        self.grid_points = self.generate_grid_points(80, 40)

        print("\n" + "=" * 60)
        print("NS Solver ready (single-threaded)")
        print(f"cpix=({self.cpix_x},{self.cpix_y})  "
              f"mesh_dx={self.mesh_dx}  sample_offset={1.5*self.mesh_dx:.2f}")
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

    def setup_spaces(self):
        v_cg2 = element("Lagrange", self.mesh.basix_cell(), 2, shape=(self.dim,))
        s_cg1 = element("Lagrange", self.mesh.basix_cell(), 1)
        self.V  = functionspace(self.mesh, v_cg2)
        self.Q  = functionspace(self.mesh, s_cg1)
        self.V0, self._V0_map = self.V.sub(0).collapse()
        self.V1, self._V1_map = self.V.sub(1).collapse()

    def setup_functions(self):
        self.u      = Function(self.V)
        self.u_n    = Function(self.V)
        self.u_tent = Function(self.V)
        self.p      = Function(self.Q)
        self.p_n    = Function(self.Q)

        s_cg1 = element("Lagrange", self.mesh.basix_cell(), 1)
        self.Qs = functionspace(self.mesh, s_cg1)
        self.cell_mask = Function(self.Qs)
        self.cell_mask.x.array[:] = 0.0
        self._qs_coords = self.Qs.tabulate_dof_coordinates()[:, :2].copy()

    def _cache_dof_coords(self):
        raw0 = self.V0.tabulate_dof_coordinates()
        raw1 = self.V1.tabulate_dof_coordinates()
        self._v0_coords = raw0[:, :2].copy()
        self._v1_coords = raw1[:, :2].copy()
        self._v0_parent = np.asarray(self._V0_map, dtype=np.int32)
        self._v1_parent = np.asarray(self._V1_map, dtype=np.int32)

        H   = 100.0
        ref = np.zeros(self.u_n.x.array.shape)
        for i, (x, y) in enumerate(self._v0_coords):
            ref[self._v0_parent[i]] = 4.0 * y * (H - y) / (H * H)
        self._poiseuille_ref = ref

        L        = 200.0
        q_coords = self.Q.tabulate_dof_coordinates()[:, :2]
        self._q_coords         = q_coords.copy()
        self._poiseuille_p_ref = self.p_inlet * (1.0 - q_coords[:, 0] / L)

        print(f"DOF cache: {len(self._v0_coords)} V0, {len(self._v1_coords)} V1")

    def setup_static_boundary_conditions(self):
        H = 100.0

        def walls(x):
            return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], H))

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

        inlet_dofs_x = locate_dofs_geometrical((self.V.sub(0), self.V0), inflow)
        inlet_dofs_y = locate_dofs_geometrical((self.V.sub(1), self.V1), inflow)

        u_inlet_x = Function(self.V0)
        for i, (x, y) in enumerate(self.V0.tabulate_dof_coordinates()[:, :2]):
            u_inlet_x.x.array[i] = 4.0 * y * (H - y) / (H * H)

        u_inlet_y = Function(self.V1); u_inlet_y.x.array[:] = 0.0

        self.bc_inlet_x = dirichletbc(u_inlet_x, inlet_dofs_x, self.V.sub(0))
        self.bc_inlet_y = dirichletbc(u_inlet_y, inlet_dofs_y, self.V.sub(1))

        outflow_dofs = locate_dofs_geometrical(self.Q, outflow)
        self.bcp = [dirichletbc(PETSc.ScalarType(self.p_outlet), outflow_dofs, self.Q)]
        self.bcu = [self.bc_wall_x, self.bc_wall_y, self.bc_inlet_x, self.bc_inlet_y]

        print(f"Wall BCs: {len(self.wall_dofs_constrained)} constrained DOFs")

    def rebuild_bcu(self):
        self.bcu = [self.bc_wall_x, self.bc_wall_y, self.bc_inlet_x, self.bc_inlet_y]
        self.update_cell_mask()

    def update_cell_mask(self):
        pts_source = self.cell_all_points if self.cell_all_points else self.cell_boundary_points

        if not pts_source:
            self._reinit_released_dofs(np.array([], dtype=np.int32))
            self.cell_mask.x.array[:] = 0.0
            self.cell_mask.x.scatter_forward()
            self._prev_masked_V_dofs = np.array([], dtype=np.int32)
            return

        pts = np.array(pts_source, dtype=float)
        tol = self.mask_tol

        diff      = self._qs_coords[:, np.newaxis, :] - pts[np.newaxis, :, :]
        min_dist  = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
        qs_inside = min_dist < tol

        v0_diff     = self._v0_coords[:, np.newaxis, :] - pts[np.newaxis, :, :]
        v0_min_dist = np.sqrt((v0_diff**2).sum(axis=2)).min(axis=1)
        v0_inside   = v0_min_dist < tol

        v1_diff     = self._v1_coords[:, np.newaxis, :] - pts[np.newaxis, :, :]
        v1_min_dist = np.sqrt((v1_diff**2).sum(axis=2)).min(axis=1)
        v1_inside   = v1_min_dist < tol

        cur_masked_V = np.concatenate([
            self._v0_parent[v0_inside],
            self._v1_parent[v1_inside],
        ]).astype(np.int32)

        self._reinit_released_dofs(cur_masked_V)

        self.cell_mask.x.array[:] = 0.0
        self.cell_mask.x.array[qs_inside] = 1.0
        self.cell_mask.x.scatter_forward()
        self._prev_masked_V_dofs = cur_masked_V

    def _reinit_released_dofs(self, cur_masked_V):
        """
        When the mask moves (cluster shifts), DOFs that were penalised
        last step but are free now get immediately reset to Poiseuille
        so the solver doesn't evolve from a near-zero initial state
        and produce a slow-recovery wake.
        """
        self._released_V_dofs = np.array([], dtype=np.int32)
        self._released_Q_dofs = np.array([], dtype=np.int32)

        if len(self._prev_masked_V_dofs) == 0:
            return

        prev_set   = set(self._prev_masked_V_dofs.tolist())
        cur_set    = set(cur_masked_V.tolist()) if len(cur_masked_V) else set()
        released_V = np.array(list(prev_set - cur_set), dtype=np.int32)
        if len(released_V) == 0:
            return

        wall_set   = set(self.wall_dofs_constrained)
        released_V = released_V[~np.isin(released_V, list(wall_set))]

        if len(released_V) > 0:
            self.u_n.x.array[released_V]   = self._poiseuille_ref[released_V]
            self.u.x.array[released_V]     = self._poiseuille_ref[released_V]
            # Also reset tentative velocity so pressure correction starts clean
            self.u_tent.x.array[released_V] = self._poiseuille_ref[released_V]
            self.u_n.x.scatter_forward()
            self.u.x.scatter_forward()
            self.u_tent.x.scatter_forward()
            self._released_V_dofs = released_V

        # Reset pressure in the vacated zone too
        if len(self._prev_masked_V_dofs) > 0:
            prev_v0 = self._prev_masked_V_dofs[
                np.isin(self._prev_masked_V_dofs, self._v0_parent)
            ]
            if len(prev_v0) > 0:
                parent_to_sub = {p: i for i, p in enumerate(self._v0_parent)}
                sub_indices   = np.array([parent_to_sub[p] for p in prev_v0
                                          if p in parent_to_sub], dtype=np.int32)
                if len(sub_indices) > 0:
                    zone_pts  = self._v0_coords[sub_indices]
                    q_diff    = self._q_coords[:, np.newaxis, :] - zone_pts[np.newaxis, :, :]
                    q_mindist = np.sqrt((q_diff**2).sum(axis=2)).min(axis=1)
                    q_release = np.where(q_mindist < self.mesh_dx * 1.5)[0].astype(np.int32)
                    if len(q_release) > 0:
                        self.p_n.x.array[q_release] = self._poiseuille_p_ref[q_release]
                        self.p.x.array[q_release]   = self._poiseuille_p_ref[q_release]
                        self.p_n.x.scatter_forward()
                        self.p.x.scatter_forward()
                        self._released_Q_dofs = q_release

        n_v = len(self._released_V_dofs)
        n_q = len(self._released_Q_dofs)
        if n_v > 0 or n_q > 0:
            print(f"  ↩ Reinitialised {n_v} V + {n_q} Q DOFs → Poiseuille")

    def set_initial_condition(self):
        H, L = 100.0, 200.0

        def poiseuille_velocity(x):
            vals = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            vals[0] = 4.0 * x[1] * (H - x[1]) / (H * H)
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

        self.u.x.scatter_forward()
        self.u_n.x.scatter_forward()
        self.u_tent.x.scatter_forward()
        print("✓ Poiseuille initial condition set")

    def setup_variational_forms(self):
        u = TrialFunction(self.V); v = TestFunction(self.V)
        p = TrialFunction(self.Q); q = TestFunction(self.Q)

        n   = FacetNormal(self.mesh)
        k   = Constant(self.mesh, PETSc.ScalarType(self.dt))
        mu  = Constant(self.mesh, PETSc.ScalarType(self.mu))
        rho = Constant(self.mesh, PETSc.ScalarType(self.rho))
        f   = Constant(self.mesh, PETSc.ScalarType((0.0, 0.0)))
        lam = Constant(self.mesh, PETSc.ScalarType(self.rho / self.dt * 5.0))

        def epsilon(u): return sym(nabla_grad(u))
        def sigma(u, p): return 2 * mu * epsilon(u) - p * Identity(len(u))

        U  = 0.5 * (self.u_n + u)
        F1 = rho * dot((u - self.u_n) / k, v) * dx
        F1 += rho * dot(dot(self.u_n, nabla_grad(self.u_n)), v) * dx
        F1 += inner(sigma(U, self.p_n), epsilon(v)) * dx
        F1 += dot(self.p_n * n, v) * ds
        F1 -= dot(mu * nabla_grad(U) * n, v) * ds
        F1 -= dot(f, v) * dx
        F1 += lam * self.cell_mask * dot(u, v) * dx
        self.a1 = form(lhs(F1)); self.L1 = form(rhs(F1))

        a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
        L2 = (dot(nabla_grad(self.p_n), nabla_grad(q)) * dx
              - (rho / k) * div(self.u_tent) * q * dx)
        self.a2 = form(a2); self.L2 = form(L2)

        a3 = rho * dot(u, v) * dx
        L3 = (rho * dot(self.u_tent, v) * dx
              - k * dot(nabla_grad(self.p - self.p_n), v) * dx)
        self.a3 = form(a3); self.L3 = form(L3)
        print("✓ Variational forms compiled")

    def setup_solvers(self):
        def make_ksp(A, ktype, pctype, hypretype=None):
            ksp = PETSc.KSP().create(self.comm)
            ksp.setOperators(A)
            ksp.setType(ktype)
            ksp.getPC().setType(pctype)
            if hypretype:
                ksp.getPC().setHYPREType(hypretype)
            return ksp

        self.A1 = create_matrix(self.a1)
        assemble_matrix(self.A1, self.a1, bcs=self.bcu); self.A1.assemble()
        self.solver1 = make_ksp(self.A1, PETSc.KSP.Type.BCGS,
                                PETSc.PC.Type.HYPRE, "boomeramg")

        self.A2 = create_matrix(self.a2)
        assemble_matrix(self.A2, self.a2, bcs=self.bcp); self.A2.assemble()
        self.solver2 = make_ksp(self.A2, PETSc.KSP.Type.BCGS,
                                PETSc.PC.Type.HYPRE, "boomeramg")

        self.A3 = create_matrix(self.a3)
        assemble_matrix(self.A3, self.a3, bcs=self.bcu); self.A3.assemble()
        self.solver3 = make_ksp(self.A3, PETSc.KSP.Type.CG, PETSc.PC.Type.SOR)

        self.b1 = create_vector([self.V])
        self.b2 = create_vector([self.Q])
        self.b3 = create_vector([self.V])
        print("✓ Solvers initialised")

    def setup_evaluators(self):
        self.bb = bb_tree(self.mesh, self.mesh.topology.dim)
        print("✓ Evaluators ready")

    def eval_at_point(self, field, x, y):
        try:
            if not (0.0 <= x <= 200.0 and 0.0 <= y <= 100.0):
                return None
            pt        = np.array([[x, y, 0.0]], dtype=np.float64)
            cands     = compute_collisions_points(self.bb, pt)
            colliding = compute_colliding_cells(self.mesh, cands, pt)
            cells     = colliding.links(0)
            if len(cells) == 0:
                return None
            val = field.eval(pt, np.array([cells[0]], dtype=np.int32))
            return np.asarray(val).ravel()
        except Exception:
            return None

    def test_flow_field(self):
        print("\n" + "=" * 60)
        print("FLOW FIELD TEST")
        print("=" * 60)
        for x, y in [[50,25],[50,50],[50,75],[100,50],[150,50]]:
            v = self.eval_at_point(self.u, x, y)
            p = self.eval_at_point(self.p, x, y)
            vs = f"({v[0]:.4f},{v[1]:.4f})" if v is not None else "FAIL"
            ps = f"{p[0]:.4f}" if p is not None else "FAIL"
            print(f"  ({x:3d},{y:2d}): u={vs}  p={ps}")
        print("=" * 60)

    # ------------------------------------------------------------------
    def handle_message(self, data):
        """Full synchronous pipeline for one MCS."""
        mcs           = data.get('mcs', 0)
        boundaries    = data.get('boundaries', [])
        cells         = data.get('cells', [])
        cells_k2      = data.get('cells_k2', [])
        boundary_only = data.get('boundary_only', boundaries)

        if cells:
            self.cell_all_points = [
                [px[0] * 200.0, px[1] * 100.0]
                for c in cells for px in c.get('pixels', [])
            ]
        else:
            self.cell_all_points = []

        if boundaries:
            self.cell_boundary_points = [
                [b['x'] * 200.0, b['y'] * 100.0] for b in boundaries
            ]

        self.rebuild_bcu()

        if cells:
            self.update_cell_data(cells)
        else:
            self.cell_data = {}

        self.cell_boundary_only = (
            [[b['x']*200.0, b['y']*100.0] for b in boundary_only]
            if boundary_only else self.cell_boundary_points
        )

        step_ok = True
        try:
            self.ns_step()
        except Exception as e:
            step_ok = False
            print(f"  ns_step error: {e}")
            traceback.print_exc()

        try:
            flow_forces = self.compute_flow_forces()
        except Exception as e:
            print(f"  compute_flow_forces error: {e}")
            flow_forces = {}

        try:
            velocity_forces = self.compute_velocity_forces(cells_k2)
        except Exception as e:
            print(f"  compute_velocity_forces error: {e}")
            velocity_forces = {}

        velocities = self.get_flow_field()
        max_u      = self.get_max_velocity()
        wall_vel   = self.verify_no_slip()

        try:
            flux = self.compute_flux()
        except Exception as e:
            print(f"  compute_flux error: {e}")
            flux = {'inlet_flux': 0.0, 'outlet_flux': 0.0, 'net_flux': 0.0}

        print(f"MCS {mcs}: step {self.step_count}, t={self.t:.3f}, "
              f"max_u={max_u:.4f}, wall={wall_vel:.2e}, "
              f"pf={len(flow_forces)} vf={len(velocity_forces)} "
              f"Q_in={flux['inlet_flux']:.3f} Q_out={flux['outlet_flux']:.3f}"
              + ("" if step_ok else " [STEP ERR]"))

        if self.step_count % 50 == 0:
            self.debug_velocity_map(velocities)

        return {
            'type':                  'flow_forces',
            'mcs':                   mcs,
            'grid_shape':            [40, 80],
            'full_field_velocities': velocities,
            'flow_forces':           flow_forces,
            'velocity_forces':       velocity_forces,
            'flux':                  flux,
        }

    # ------------------------------------------------------------------
    def compute_flow_forces(self):
        """
        Per-cell pressure force integral (Xu et al. 2008, eq 3.9).

        Key design decisions:

        1. PER-CELL, NOT PER-CLUSTER.
           We compute each cell's own integral over its own exterior membrane
           segments (faces where the neighbour is outside the ENTIRE cluster).
           Interior cells have no exterior faces → zero force naturally.
           Upstream-face cells have many outward-facing fluid-side segments
           at high pressure → large downstream force.
           Downstream-face cells have segments at low pressure → small force.
           This prevents the cluster from blowing apart: interior cells feel
           nothing and stay cohesive; only the boundary cells are pushed.

        2. CORRECT COORDINATE CONVERSION.
           JS sends pixels normalised [0,1]: pt = [cpm_x/400, cpm_y/200].
           Python recovers CPM coords:  (px, py) = (round(pt[0]*400), round(pt[1]*200))
           Domain coords:               dom_x = px * cpix_x = px * 0.5  (range 0..200)
                                        dom_y = py * cpix_y = py * 0.5  (range 0..100)
           DO NOT multiply by 2 — cpix already encodes the full conversion.

        3. SAMPLE OUTSIDE PENALTY REGION.
           Sample pressure 1.5 mesh cells outside the membrane segment so
           we read the uncontaminated fluid pressure, not the penalised interior.
        """
        forces = {}
        if not self.cell_data:
            return forces

        sample_offset_x = 3 * self.mesh_dx   # 3.75 domain units
        sample_offset_y = 3 * self.mesh_dy

        # Build merged pixel set per cluster (needed to identify exterior faces)
        clusters = defaultdict(list)
        for cell_id, info in self.cell_data.items():
            clusters[info.get('cluster_id', cell_id)].append(cell_id)

        cluster_pixel_sets = {}
        for cluster_id, member_ids in clusters.items():
            merged = set()
            for cell_id in member_ids:
                for pt in self.cell_data[cell_id].get('pixels', []):
                    merged.add((round(pt[0] * 400), round(pt[1] * 200)))
            cluster_pixel_sets[cluster_id] = merged

        # Integrate pressure over each cell's own exterior segments
        for cell_id, info in self.cell_data.items():
            cluster_id = info.get('cluster_id', cell_id)
            merged     = cluster_pixel_sets.get(cluster_id, set())

            # This cell's own pixels in CPM coords
            cell_pixels = set()
            for pt in info.get('pixels', []):
                cell_pixels.add((round(pt[0] * 400), round(pt[1] * 200)))

            fx, fy = 0.0, 0.0
            n_segs = 0
            n_none = 0

            for (px, py) in cell_pixels:
                for (ddx, ddy) in [(1,0),(-1,0),(0,1),(0,-1)]:
                    # Exterior face: neighbour is outside the WHOLE cluster
                    if (px+ddx, py+ddy) in merged:
                        continue

                    # Outward unit normal (from cluster interior toward fluid)
                    norm_dx  = float(ddx) * self.cpix_x
                    norm_dy  = float(ddy) * self.cpix_y
                    norm_len = np.sqrt(norm_dx**2 + norm_dy**2)
                    norm_dx /= norm_len
                    norm_dy /= norm_len

                    # CORRECT: px * cpix_x gives domain coords (range 0..200)
                    seg_x    = px * self.cpix_x
                    seg_y    = py * self.cpix_y
                    sample_x = float(np.clip(seg_x + ddx * sample_offset_x, 0.0, 200.0))
                    sample_y = float(np.clip(seg_y + ddy * sample_offset_y, 0.0, 100.0))

                    p_val = self.eval_at_point(self.p, sample_x, sample_y)
                    if p_val is None:
                        n_none += 1
                        continue

                    p = float(p_val[0])

                    # F = -p * n̂_outward * dS
                    # Upstream face: norm_dx < 0, high p → -p*(-|n|) > 0 (rightward) ✓
                    fx -= p * norm_dx * norm_len
                    fy -= p * norm_dy * norm_len
                    n_segs += 1

            forces[str(cell_id)] = [fx, fy]
            if n_segs > 0 or n_none > 0:
                print(f"  [CFF] cell {cell_id}: {n_segs} ext segs {n_none} None "
                      f"F=({fx:.5f},{fy:.5f})")

        return forces

    # ------------------------------------------------------------------
    def compute_velocity_forces(self, cells_k2):
        """Average flow velocity over all pixels of each free cell."""
        if not cells_k2:
            return {}

        all_points, cell_ranges = [], []
        for cell in cells_k2:
            start = len(all_points)
            for pt in cell.get('pixels', []):
                all_points.append([pt[0] * 200.0, pt[1] * 100.0])
            cell_ranges.append((cell['id'], start, len(all_points)))

        if not all_points:
            return {str(c['id']): [0.0, 0.0] for c in cells_k2}

        vels = self.eval_velocity_batch(all_points)
        forces = {}
        for cell_id, start, end in cell_ranges:
            if end <= start:
                forces[str(cell_id)] = [0.0, 0.0]; continue
            nz = [vels[i] for i in range(start, end)
                  if abs(vels[i][0]) > 1e-8 or abs(vels[i][1]) > 1e-8]
            forces[str(cell_id)] = (
                [float(np.mean([v[0] for v in nz])),
                 float(np.mean([v[1] for v in nz]))]
                if nz else [0.0, 0.0]
            )
        return forces

    # ------------------------------------------------------------------
    def compute_flux(self):
        H  = 100.0
        dy = H / 40.0
        iv = self.eval_velocity_batch([self.grid_points[r*80+0]  for r in range(40)])
        ov = self.eval_velocity_batch([self.grid_points[r*80+79] for r in range(40)])
        return {
            'inlet_flux':  round(sum(v[0] for v in iv) * dy, 6),
            'outlet_flux': round(sum(v[0] for v in ov) * dy, 6),
            'net_flux':    round((sum(v[0] for v in ov)-sum(v[0] for v in iv))*dy, 6),
        }

    # ------------------------------------------------------------------
    def ns_step(self):
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu); self.A1.assemble()
        self.solver1.setOperators(self.A1)
        with self.b1.localForm() as loc: loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_tent.x.petsc_vec)
        self.u_tent.x.scatter_forward()

        self.A2.zeroEntries()
        assemble_matrix(self.A2, self.a2, bcs=self.bcp); self.A2.assemble()
        self.solver2.setOperators(self.A2)
        with self.b2.localForm() as loc: loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.p.x.petsc_vec)
        self.p.x.scatter_forward()

        self.A3.zeroEntries()
        assemble_matrix(self.A3, self.a3, bcs=self.bcu); self.A3.assemble()
        self.solver3.setOperators(self.A3)
        with self.b3.localForm() as loc: loc.set(0)
        assemble_vector(self.b3, self.L3)
        apply_lifting(self.b3, [self.a3], [self.bcu])
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b3, self.bcu)
        self.solver3.solve(self.b3, self.u.x.petsc_vec)
        self.u.x.scatter_forward()

        self.u_n.x.array[:] = self.u.x.array[:]
        self.p_n.x.array[:] = self.p.x.array[:]
        self.t          += self.dt
        self.step_count += 1

        # Clamp velocity magnitude
        arr = self.u.x.array
        spd = np.sqrt(arr[0::2]**2 + arr[1::2]**2)
        over = spd > 2.0
        if over.any():
            scale = np.where(over, 2.0/np.maximum(spd, 1e-12), 1.0)
            arr[0::2] *= scale; arr[1::2] *= scale
            self.u.x.scatter_forward()

        # Clamp backflow (unphysical recirculation)
        ux_dofs = self._v0_parent
        neg = self.u.x.array[ux_dofs] < -0.01
        if neg.any():
            self.u.x.array[ux_dofs[neg]] = 0.0
            self.u.x.scatter_forward()

        self.u_n.x.array[:] = self.u.x.array[:]
        self.u_n.x.scatter_forward()

        # Post-step reinit of released DOFs (belt-and-suspenders after solve)
        rv = getattr(self, '_released_V_dofs', np.array([], dtype=np.int32))
        rq = getattr(self, '_released_Q_dofs', np.array([], dtype=np.int32))
        if len(rv) > 0:
            wall_set = set(self.wall_dofs_constrained)
            rv_clean = rv[~np.isin(rv, list(wall_set))]
            if len(rv_clean) > 0:
                self.u.x.array[rv_clean]   = self._poiseuille_ref[rv_clean]
                self.u_n.x.array[rv_clean] = self._poiseuille_ref[rv_clean]
                self.u.x.scatter_forward(); self.u_n.x.scatter_forward()
        if len(rq) > 0:
            self.p.x.array[rq]   = self._poiseuille_p_ref[rq]
            self.p_n.x.array[rq] = self._poiseuille_p_ref[rq]
            self.p.x.scatter_forward(); self.p_n.x.scatter_forward()

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

    def get_flow_field(self):
        """
        Evaluate velocity at all grid points, then unconditionally patch
        grid points that were inside the cluster last step but are free
        now with the analytic Poiseuille value.

        The patch is unconditional (no 0.5 threshold) because the FEM
        solver may still produce suppressed values in the wake region
        even after _reinit_released_dofs — the penalty term's influence
        diffuses outward over a few mesh cells and takes several timesteps
        to fully flush. The direct patch ensures the visualisation and
        any downstream flow sampling always shows physically correct values
        in the vacated zone.
        """
        H   = 100.0
        out = self.eval_velocity_batch(self.grid_points)
        pts = np.array(self.grid_points, dtype=float)

        cur_pts_list = list(self.cell_all_points) + list(self.cell_boundary_points)
        if cur_pts_list:
            cur_zone   = np.array(cur_pts_list, dtype=float)
            diff_cur   = pts[:, np.newaxis, :] - cur_zone[np.newaxis, :, :]
            dist_cur   = np.sqrt((diff_cur**2).sum(axis=2)).min(axis=1)
            cur_inside = np.where(dist_cur < self.mesh_dx*1.5)[0].astype(np.int32)
        else:
            cur_inside = np.array([], dtype=np.int32)

        # Grid points vacated since last step → patch unconditionally
        released = np.array(
            list(set(self._prev_masked_grid_indices.tolist()) -
                 set(cur_inside.tolist())),
            dtype=np.int32
        )
        for i in released:
            x, y   = pts[i]
            vx_ref = 4.0 * y * (H - y) / (H * H)
            if vx_ref > 0.01:
                out[i] = [float(vx_ref), 0.0]   # unconditional restore

        self._prev_masked_grid_indices = cur_inside
        return out

    def generate_grid_points(self, nx, ny):
        x = np.linspace(2.5, 197.5, nx)
        y = np.linspace(2.5,  97.5, ny)
        X, Y = np.meshgrid(x, y)
        return [[X[i,j], Y[i,j]] for i in range(ny) for j in range(nx)]

    def debug_velocity_map(self, velocities):
        ny, nx = 40, 80
        step_x, step_y = nx//20, ny//8
        print("── VELOCITY MAP (vx) ──────────────────────────────────")
        dead_zones = []
        for row in range(7, -1, -1):
            gy   = row * step_y
            line = f"y~{int(2.5+gy*2.5):3d}| "
            for col in range(20):
                gx        = col * step_x
                idx       = gy * nx + gx
                vx        = velocities[idx][0] if idx < len(velocities) else 0.0
                y_domain  = 2.5 + gy * 2.5
                poi       = 4.0 * y_domain * (100.0 - y_domain) / 10000.0
                ratio     = vx / poi if poi > 0.01 else 1.0
                gx_domain = 2.5 + gx * 2.5
                near      = False
                if self.cell_all_points:
                    pts_arr = np.array(self.cell_all_points)
                    near = np.sqrt(((pts_arr - [gx_domain, y_domain])**2)
                                   .sum(axis=1)).min() < self.mesh_dx * 6
                is_dead = (ratio < 0.02 or vx < -0.05) and 5 < y_domain < 95 and not near
                if is_dead:
                    dead_zones.append((int(gx_domain), int(y_domain), vx, poi))
                    ch = "X" if vx >= 0 else "-"
                elif ratio < 0.3: ch = "."
                elif ratio < 0.7: ch = "o"
                else:             ch = "#"
                line += ch
            print(line)
        if dead_zones:
            print(f"  ⚠ {len(dead_zones)} dead zone(s)")
        else:
            print("  ✓ No dead zones")

    def get_max_velocity(self):
        a = self.u.x.array
        return float(np.max(np.sqrt(a[0::2]**2 + a[1::2]**2)))

    def verify_no_slip(self):
        if not self.wall_dofs_constrained: return -1.0
        return float(np.max(np.abs(self.u.x.array[self.wall_dofs_constrained])))

    def update_cell_data(self, cells):
        self.cell_data = {}
        for c in cells:
            raw_pixels  = c.get('pixels') or []
            raw_borders = c.get('borders') or raw_pixels
            self.cell_data[c['id']] = {
                'type':       c['type'],
                'pixels':     raw_pixels,
                'centroid':   c['centroid'],
                'border_pts': [[b[0]*200.0, b[1]*100.0] for b in raw_borders],
                'cluster_id': c.get('cluster_id', c['id']),
            }


# ======================================================================
# WebSocket server — single-threaded, one client at a time
# ======================================================================
class WebSocketServer:
    def __init__(self):
        self.solver            = PoiseuilleNSSolver()
        self._active_websocket = None

    async def handler(self, websocket):
        if self._active_websocket is not None:
            prev = self._active_websocket
            self._active_websocket = None
            try:
                await prev.close(1001, "Replaced by new connection")
            except Exception:
                pass
            print("⚠ Previous client replaced")

        self._active_websocket = websocket
        print(f"\n🔌 Client connected from {websocket.remote_address}")

        try:
            async for message in websocket:
                if self._active_websocket is not websocket:
                    break

                data = json.loads(message)
                if data.get('type') != 'boundary_conditions':
                    continue

                result = self.solver.handle_message(data)

                if self._active_websocket is not websocket:
                    break

                try:
                    await websocket.send(json.dumps(result))
                except (websockets.exceptions.ConnectionClosed,
                        BrokenPipeError, OSError) as e:
                    print(f"  Send failed (client gone): {e}")
                    break

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected normally")
        except Exception as e:
            print(f"Handler error: {e}"); traceback.print_exc()
        finally:
            if self._active_websocket is websocket:
                self._active_websocket = None

    async def start(self):
        async with websockets.serve(
            self.handler, "localhost", 8765,
            ping_interval=30, ping_timeout=120,
            max_size=100_000_000,
        ):
            print("\n" + "="*60)
            print("NS+CPM server ready  (single-threaded, one client)")
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
