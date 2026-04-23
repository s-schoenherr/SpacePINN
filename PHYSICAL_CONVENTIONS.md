# Physical Conventions

This document collects the governing-equation, loss-construction, and physical/time/coordinate conventions used by the repository.

## Governing Equation

The PINN trajectory is parameterized on normalized time, so the governing equation should be read in two equivalent ways.

In terms of the normalized trajectory parameter, the residual form is:

- `(1 / T^2) * d^2r / dtau^2 - G(r) - F = 0`

where `T = t_total` is the physical time of flight and `G(r)` is the gravitational acceleration field.

Defining the physical acceleration as

- `a - G - F = 0`

with

- `a = (1 / T^2) * d^2r / dtau^2`

where

- `a` is the total physical acceleration implied by the learned trajectory,
- `G` is the gravitational acceleration from the configured gravity sources,
- `F` is the non-gravitational control or thrust term.

In the implementation, this is rearranged to reconstruct the thrust contribution from the learned trajectory:

- Cartesian: `F = a - G`
- Polar: `F_rho = a_rho - G_rho` and `F_alpha = a_alpha - G_alpha`

So the learned trajectory is differentiated in normalized time, converted to physical acceleration with the `1 / T^2` factor, gravity is computed from the force model, and the remaining residual is interpreted as the thrust required to satisfy the governing equation.

## Physics Loss Construction

The loss is built on top of the governing-equation residual above.

The optimization loss is built as a weighted sum of physically distinct terms:

- A physics residual term is always present.
  - In Cartesian coordinates this is the mean squared thrust residual magnitude, `mean(||F||^2)`, scaled by `physics_loss_weight`.
  - In polar coordinates this is the sum of the mean squared residual components, `mean(F_rho^2) + mean(F_alpha^2)`, also scaled by `physics_loss_weight`.
- A boundary-condition term is added only when `boundary_loss_weight > 0`.
  - In Cartesian form it penalizes squared endpoint position errors at the start and end.
  - In polar form the boundary positions are first mapped to Cartesian space and the endpoint mismatch is penalized there, so the boundary loss is evaluated in physical position space rather than angle/radius components independently.
- A thrust-cap penalty is added only when a thrust cap is configured and `thrust_cap_weight > 0`.
  - It penalizes only the excess above the cap via a squared ReLU on the normalized thrust magnitude.

So the total loss is:

- `total_loss = physics_loss + optional_boundary_loss + optional_thrust_cap_loss`

There is also an optional anti-oscillation regularizer for cartesian runs:

- A tangential-thrust smoothness term can be added when `tangential_thrust_smoothness_weight > 0`.
  - It computes the scalar tangential thrust `F · \hat{v}` with `\hat{v} = v / ||v||`.
  - It penalizes strong curvature of that scalar across the collocation grid via a squared finite-difference second derivative.
  - This term is intended to suppress rapid forward/backward thrust switching; it is not a hard physical constraint.

This means hard-constrained distance-function transforms and soft boundary penalties can coexist: the transform can enforce parts of the boundary behavior by construction, while the explicit boundary-loss term still provides an adjustable soft penalty when that formulation is being used.

## Time Convention

- Internally, the PINN optimization and output transforms operate on normalized time `tau in [0, 1]`.
- Transform-side state derivatives are endpoint slopes with respect to normalized time: `dr/dtau`.
- `TrajectoryResult`, saved runs, plots, and physics checks expose physical velocity: `dr/dt`.
- The conversion is `dr/dtau = t_total * dr/dt`.
- For kinematic transforms, the preferred API is to pass physical boundary velocities and let the transform convert them using the current model `t_total`.
- Do not precompute normalized-time endpoint velocities from a different run and then reuse them if `t_total` is still trainable, because any later drift in `t_total` changes the physical endpoint velocity.

## Cartesian Kinematic Boundary Convention

- For `kinematic_fn`, the boundary positions are Cartesian states:
  - `x0 = [x_0, y_0]` or `x0 = [x_0, y_0, z_0]`
  - `xN = [x_N, y_N]` or `xN = [x_N, y_N, z_N]`
- The boundary velocities `v0` and `vN` must be passed as physical Cartesian velocities:
  - `v = [dx/dt, dy/dt]` or `v = [dx/dt, dy/dt, dz/dt]`
- The transform converts these physical boundary velocities internally to normalized-time state slopes using the current model `t_total`.
- Practical implication:
  - if a kinematic Cartesian PINN should match endpoint velocities from another saved result, pass `result.v[0]` and `result.v[-1]`
  - do not pass `result.v[0] * result.t_total` or `result.v[-1] * result.t_total`

## Final-State Consistency

- Saved runs and `TrajectoryResult` must be built from a state that is recomputed after the last optimizer parameter update.
- Otherwise, a run can mix:
  - a pre-update trajectory/state derivative
  - with a post-update trainable parameter such as `t_total`
- That inconsistency is especially visible in kinematic transforms, where endpoint physical velocities can appear to violate the hard constraint even though the transform itself is correct.
- The optimization engine therefore refreshes the final state from the final model parameters before constructing the exported `OptimizationRun` and `TrajectoryResult`.

## Polar Kinematic Boundary Convention

- For `kinematic_polar_fn`, the boundary positions are polar states:
  - `x0 = [rho_0, alpha_0]`
  - `xN = [rho_N, alpha_N]`
- The boundary velocities `v0` and `vN` must be passed as physical polar velocities:
  - `v = [v_r, v_t]`
  - where `v_r = drho/dt`
  - and `v_t = rho * dalpha/dt`
- The transform converts these physical boundary velocities internally to normalized-time state slopes:
  - `rho_t = t_total * v_r`
  - `alpha_t = t_total * v_t / rho`
- Do not pass `[v_r, dalpha/dtau]` or `[v_r, dalpha/dt]` into `kinematic_polar_fn`.
- Example for a circular-orbit boundary:
  - `v_r = 0`
  - `v_t = sqrt(mu / rho)`
  - use `v = [0, sqrt(mu / rho)]`

Practical implication:

- For polar kinematic orbit-transfer configs, the natural API is positions in polar coordinates and velocities in physical polar form `[v_r, v_t]`.
- This matches `TrajectoryResult.v_polar`, which is also reported as `[v_r, v_t]` in physical units.

Compatibility note:

- Older saved runs created before March 24, 2026 may have used the second polar kinematic boundary component inconsistently as a state-angle slope surrogate instead of physical tangential velocity.
- New runs after that fix use the physically correct interpretation described above.
- Old and new `kinematic_polar` orbit-transfer results are therefore not necessarily directly comparable unless they were regenerated after the fix.

For collection runs, the same applies per entry in `configs/` and `artifacts/result/`.
