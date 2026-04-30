# Simulation Bugfix Notes

These notes summarize the fixes made while preserving the existing model intent from
`src/main.js`, `src/artistoo-worker.js`, and `src/ns_simple.py`.

## Existing model points retained

- Near-wall catheter window geometry: 500 x 200 pixels, with the wall at the
  bottom boundary.
- Poiseuille-like flow with low velocity and low shear near the wall.
- QS signal production, diffusion, advection-inspired dilution, and decay.
- QS activation, EPS-producing cells, cell growth/division, and open-boundary
  deletion.
- Cluster detection based on local cell contact/proximity.

## Attachment fix

The previous browser model had moving gray cells but no explicit stable
cell-wall attachment state. The export runner adds a minimal attachment rule:

```text
if distance_to_wall <= attachment_distance:
    p_attach = adhesion_wall * exp(-local_shear / shear_scale)
```

Successful attachment changes the cell state to `attached`. Attached cells
continue to grow, divide, participate in QS, and produce EPS after QS activation.
This directly addresses the issue where gray cells can spend a long time near
the wall without becoming attached.

To make attached cells easier to observe, increase:

- `adhesion_wall`
- `attachment_distance`
- `shear_scale`

or decrease:

- `flow_rate`

## State colors used by exported analysis

The Python data platform standardizes states for plots and downstream models:

- `planktonic`: light gray
- `attached`: gray
- `active`: blue
- `inactive`: dark blue
- `qs_active`: red
- `eps_producing`: purple
- EPS matrix visualizations: transparent green

## QS/EPS safeguards

- `signal_decay` is applied every timestep.
- `diffusion_rate` is clipped to a stable finite-difference range.
- EPS production is limited to `eps_producing` cells, which are derived from
  QS-active cells.
- EPS fraction is exported in `simulation_summary`.

## Division safeguards

- Cell IDs are generated from a monotonic counter and are unique inside each
  simulation.
- Daughter cells retain the parent's state.
- Parent and daughter volumes are split.
- Cluster IDs are recomputed at each exported timepoint.

## Export guarantees

Every standard table includes `simulation_id`. The `cells` and `clusters` tables
always include `timepoint`. `flow_field` is exported once per simulation, and
`simulation_summary` is exported after the run completes.
