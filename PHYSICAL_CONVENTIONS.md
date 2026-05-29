# Physical Conventions

This document collects the governing-equation, loss-construction, time, coordinate, and boundary-condition conventions used by SpacePINN.

## Governing Equation

The PINN trajectory is parameterized on normalized time `tau in [0, 1]`. The physical time of flight is `T = t_total`, so derivatives with respect to physical time satisfy

- `dr/dt = (1 / T) * dr/dtau`
- `d2r/dt2 = (1 / T^2) * d2r/dtau2`

The governing residual is therefore

- `(1 / T^2) * d2r / dtau2 - G(r) - F = 0`

where `G(r)` is the gravitational acceleration field and `F` is the non-gravitational acceleration required from control or thrust. Equivalently,

- `a - G - F = 0`
- `a = (1 / T^2) * d2r / dtau2`

In the implementation, the learned trajectory is differentiated first and the thrust contribution is reconstructed as the remaining acceleration:

- Cartesian: `F = a - G`
- Polar: `F_rho = a_rho - G_rho` and `F_alpha = a_alpha - G_alpha`

Thus `F` is not a separate network output in the paper experiments. It is the acceleration residual implied by the trajectory after subtracting gravity and any configured external acceleration term.

## Cartesian Coordinates

For Cartesian runs, the state is

- `r = [x, y]` in 2D
- `r = [x, y, z]` in 3D

Velocities and accelerations are physical Cartesian quantities after the normalized-time derivatives have been scaled by `t_total`. The Cartesian gravity model sums point-mass accelerations from the configured gravity sources.

## Polar Coordinates

For polar orbit-transfer and rendezvous runs, the state is

- `r = [rho, alpha]`

where `rho` is orbital radius and `alpha` is the inertial polar angle. The physical velocity components are reported as

- `v_r = drho/dt`
- `v_t = rho * dalpha/dt`

The polar acceleration components used by the dynamics are physical radial and tangential acceleration components:

- `a_rho = d2rho/dt2 - rho * (dalpha/dt)^2`
- `a_alpha = 2 * (drho/dt) * (dalpha/dt) + rho * d2alpha/dt2`

Accordingly, `F_alpha` is a tangential acceleration component, not an angular acceleration. The same convention is used for `G_alpha` and for thrust magnitudes:

- `||F|| = sqrt(F_rho^2 + F_alpha^2)`

`TrajectoryResult` stores both Cartesian outputs for plotting/comparison and polar outputs where applicable. For polar results, `v_polar` is `[v_r, v_t]` in physical units.

## Physics Loss Construction

The optimization loss is a weighted sum of physically distinct terms.

- A physics residual term is always present.
  - Cartesian: `mean(||F||^2) * physics_loss_weight`
  - Polar: `(mean(F_rho^2) + mean(F_alpha^2)) * physics_loss_weight`
- A boundary-condition term is added only when `boundary_loss_weight > 0`.
  - Cartesian boundaries are penalized directly in Cartesian position space.
  - Polar boundary positions are first mapped to Cartesian space, so the endpoint mismatch is measured as a physical position error rather than as independent radius/angle errors.

The total loss is

- `total_loss = physics_loss + optional_boundary_loss + optional_smoothness_loss`

There is also an optional anti-oscillation regularizer for Cartesian runs:

- A tangential-thrust smoothness term can be added when `tangential_thrust_smoothness_weight > 0`.
- It computes the scalar tangential thrust `F dot v_hat`, with `v_hat = v / ||v||`.
- It penalizes strong curvature of that scalar across the collocation grid via a squared finite-difference second derivative.
- This term is intended to suppress rapid forward/backward thrust switching; it is not a hard physical constraint.

Hard-constrained output transforms and soft boundary penalties can coexist. The transform can enforce boundary behavior by construction, while the explicit boundary-loss term remains available for ordinary soft-BC formulations or weighted boundary sweeps.

## Time Convention

- Internally, the PINN optimization and output transforms operate on normalized time `tau in [0, 1]`.
- Transform-side state derivatives are endpoint slopes with respect to normalized time: `dr/dtau`.
- `TrajectoryResult`, saved runs, plots, and physics checks expose physical velocity: `dr/dt`.
- The conversion is `dr/dtau = t_total * dr/dt`.
- For kinematic transforms, pass physical boundary velocities and let the transform convert them using the current model `t_total`.
- Do not precompute normalized-time endpoint velocities from a different run and then reuse them if `t_total` is trainable, because any later drift in `t_total` changes the physical endpoint velocity.

## Boundary-Condition Summary

The paper experiments use the following boundary-condition conventions. This section collects the experiment-specific boundary states in one place. The later sections only describe how the generic transforms interpret these states.

### Swing-By in 2D and 3D

The swing-by examples are formulated in Cartesian coordinates. Boundary positions are prescribed directly as Cartesian vectors.

- 2D state: `r = [x, y]`
- 3D state: `r = [x, y, z]`
- start and end positions are enforced by the selected output transform for exact-BC variants
- ordinary soft-BC variants use the same boundary positions through the boundary-loss term

The geometric exact-BC swing-by variants constrain positions. The kinematic and pre-conditioned 3D workflow additionally uses physical Cartesian endpoint velocities where that transform is selected.

### Circular Orbit Transfer with Fixed Terminal Angle

The fixed-terminal-angle orbit transfer is formulated in inertial polar coordinates.

- start position: `[rho_0, alpha_0] = [R_LEO, 0]`
- terminal position: `[rho_N, alpha_N] = [R_target, alpha_T]`
- start velocity: `v0 = [0, V_LEO]`
- terminal velocity: `vN = [0, V_target]`

Here `V_LEO = sqrt(mu / R_LEO)` and `V_target = sqrt(mu / R_target)` are physical tangential circular-orbit speeds. The transform converts these to normalized polar state slopes internally.

### Circular Orbit Transfer with Free Terminal Angle

The free-terminal-angle orbit transfer keeps the terminal radius and circular terminal speed prescribed, but relaxes the terminal phase angle.

- start position: `[rho_0, alpha_0] = [R_LEO, 0]`
- terminal radius: `rho_N = R_target`
- terminal angle: `alpha_N` is trainable
- start velocity: `v0 = [0, V_LEO]`
- terminal velocity: `vN = [0, V_target]`

Thus the terminal point is constrained to the target circular orbit, while the optimizer may choose the terminal phase angle.

### Rendezvous Hold Point in LEO

The hold-point rendezvous is physically specified in a target-centered Cartesian frame and implemented through polar boundary states for the chaser.

- target: `x_b(t) = R_LEO * e_r(t)`
- target velocity: `dx_b/dt = V_LEO * e_alpha(t)`
- chaser initial position: `x_a(t_0) = x_b(t_0) - 1.0 km * e_r(t_0) - 0.3 km * e_alpha(t_0)`
- chaser initial velocity: `dx_a/dt(t_0) = dx_b/dt(t_0)`
- terminal hold point: `x_a(T) = x_b(T) + h * e_r(T)` with `h = 30 m`
- terminal velocity: `dx_a/dt(T) = (R_LEO + h) * (V_LEO / R_LEO) * e_alpha(T)`

For training, these Cartesian boundary states are converted to polar states `[rho, alpha]` and physical polar velocities `[v_r, v_t]`.

## Cartesian Kinematic Transform Inputs

For `kinematic_fn`, the boundary positions are Cartesian states:

- `x0 = [x_0, y_0]` or `x0 = [x_0, y_0, z_0]`
- `xN = [x_N, y_N]` or `xN = [x_N, y_N, z_N]`

The boundary velocities `v0` and `vN` must be physical Cartesian velocities:

- `v = [dx/dt, dy/dt]` or `v = [dx/dt, dy/dt, dz/dt]`

The transform converts these physical boundary velocities internally to normalized-time state slopes using the current model `t_total`.

Practical implication:

- If a kinematic Cartesian PINN should match endpoint velocities from another saved result, pass `result.v[0]` and `result.v[-1]`.
- Do not pass `result.v[0] * result.t_total` or `result.v[-1] * result.t_total`.

## Polar Kinematic Transform Inputs

For `kinematic_polar_fn`, the boundary positions are polar states:

- `x0 = [rho_0, alpha_0]`
- `xN = [rho_N, alpha_N]`

The boundary velocities `v0` and `vN` must be physical polar velocities:

- `v = [v_r, v_t]`
- `v_r = drho/dt`
- `v_t = rho * dalpha/dt`

The transform converts these physical boundary velocities internally to normalized-time state slopes:

- `rho_tau = t_total * v_r`
- `alpha_tau = t_total * v_t / rho`

Do not pass `[v_r, dalpha/dtau]` or `[v_r, dalpha/dt]` into `kinematic_polar_fn`.

Example for a circular-orbit boundary:

- `v_r = 0`
- `v_t = sqrt(mu / rho)`
- use `v = [0, sqrt(mu / rho)]`

Practical implication:

- For polar kinematic orbit-transfer configs, the natural API is positions in polar coordinates and velocities in physical polar form `[v_r, v_t]`.
- This matches `TrajectoryResult.v_polar`, which is also reported as `[v_r, v_t]` in physical units.

## Rendezvous Implementation Note

The hold-point scenario is specified in a target-centered Cartesian frame because that is the readable physical description. For training, the absolute chaser boundary states from the summary above are converted to inertial polar coordinates and physical polar velocities. The relative notation uses `rho_a_rel` for radial offset and `s_a_rel` for along-track offset in the rotating target frame.

## Final-State Consistency

Saved runs and `TrajectoryResult` must be built from a state recomputed after the last optimizer parameter update. Otherwise, a run can mix

- a pre-update trajectory/state derivative
- with a post-update trainable parameter such as `t_total`

That inconsistency is especially visible in kinematic transforms, where endpoint physical velocities can appear to violate the hard constraint even though the transform itself is correct. The optimization engine therefore refreshes the final state from the final model parameters before constructing the exported `OptimizationRun` and `TrajectoryResult`.
