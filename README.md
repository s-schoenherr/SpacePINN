# SpacePINN

## Overview
SpacePINN studies spacecraft swingby and orbit-transfer problems with physics-informed neural networks under gravitational and thrust-driven dynamics in Cartesian and polar coordinates. Its main research question is how strongly constrained PINN formulations behave relative to standard soft-penalty approaches: the hard-constrained variants transform the network output with distance functions so boundary conditions are satisfied by construction, while the vanilla formulations enforce the same conditions through loss terms. The repository benchmarks both against OpenGoddard, a direct optimal-control method based on collocation and nonlinear programming, and provides the persistence, plotting, and validation tooling needed to compare the resulting trajectories quantitatively.

This curated release focuses on the current paper-facing experiments:
- swingby benchmarks in 2D and 3D
- Hohmann-style and free-final-angle low-thrust transfers
- descent landing and rendezvous hold-point scenarios
- Monte Carlo robustness studies for the reported stochastic comparisons
- auxiliary paper studies for boundary-condition variation, time-of-flight effects, soft-BC weight selection, and kinematic consistency

In other words, this repository is meant to be a compact, reproducible paper release rather than the full internal research codebase.

## Installation
Clone the repository and install the project together with its development dependencies:

```bash
git clone https://github.com/<user-or-org>/SpacePINN.git
cd SpacePINN
uv sync --extra dev
uv run pytest -q
```

If `uv` is not installed yet, follow the official installation instructions at https://docs.astral.sh/uv/.

## Quick Start
Run one of the included paper experiments directly:

```bash
uv run python -m spacepinn.paper.swingby_2d
uv run python -m spacepinn.paper.swingby_3d
uv run python -m spacepinn.paper.hohnmann_transfer
uv run python -m spacepinn.paper.low_thrust_transfer
uv run python -m spacepinn.paper.descent_landing_2d
uv run python -m spacepinn.paper.rendezvous_hold_point_eci
```

Monte Carlo entry points are available for the experiments used in the paper statistics:

```bash
uv run python -m spacepinn.paper.monte_carlo.swingby_2d
uv run python -m spacepinn.paper.monte_carlo.swingby_3d
uv run python -m spacepinn.paper.monte_carlo.hohnmann_transfer
uv run python -m spacepinn.paper.monte_carlo.low_thrust_transfer
uv run python -m spacepinn.paper.monte_carlo.descent_landing_2d
uv run python -m spacepinn.paper.monte_carlo.rendezvous_hold_point_eci
```

Auxiliary paper studies can be run the same way:

```bash
uv run python -m spacepinn.paper.kinematic_sanity_check_3d
uv run python -m spacepinn.paper.swingby_boundary_condition_variants_3d
uv run python -m spacepinn.paper.swingby_time_of_flight_variants_3d
uv run python -m spacepinn.paper.sweeps.low_thrust_boundary
uv run python -m spacepinn.paper.sweeps.swingby_soft_bc_weight_search_2d
uv run python -m spacepinn.paper.sweeps.swingby_soft_bc_weight_search_3d
```

Each experiment writes a reproducible run directory under:

`runs/YYYY/MM/<run_id>/`

with configs, summaries, serialized results, and generated plots.

## Included Paper Experiments
- `spacepinn.paper.swingby_2d`
  A two-dimensional swingby benchmark comparing PINNs with and without exact boundary conditions against an OpenGoddard baseline.
- `spacepinn.paper.swingby_3d`
  The three-dimensional swingby benchmark, including the pre-conditioning path used for the exact-BC variant.
- `spacepinn.paper.hohnmann_transfer`
  A constrained transfer from a circular LEO to a higher circular MEO-like orbit with fixed terminal angle.
- `spacepinn.paper.low_thrust_transfer`
  A free-final-angle low-thrust orbit-transfer problem used for the larger-radius transfer study.
- `spacepinn.paper.descent_landing_2d`
  A two-dimensional atmospheric descent and landing problem with fixed touchdown angle and OpenGoddard comparison.
- `spacepinn.paper.rendezvous_hold_point_eci`
  A low-thrust rendezvous-to-hold-point problem in an Earth-centered inertial frame, including cold and PINN-warmstarted OpenGoddard references.
- `spacepinn.paper.kinematic_sanity_check_3d`
  A formulation-consistency check comparing position-only exact-BC, kinematic exact-BC, and direct-collocation solutions in the synthetic 3D swingby setup.
- `spacepinn.paper.swingby_boundary_condition_variants_3d`
  A paper-side study of how varying the 3D swingby boundary geometry affects exact-BC trajectories.
- `spacepinn.paper.swingby_time_of_flight_variants_3d`
  A paper-side study of fixed time-of-flight choices for the 3D swingby setup, with selected OpenGoddard references.
- `spacepinn.paper.monte_carlo.*`
  Robustness and variance studies for the main paper experiments, including swingby, transfer, landing, and rendezvous cases.
- `spacepinn.paper.sweeps.*`
  Focused search utilities for soft-BC weights and transfer initialization used in the paper workflow.

## Included Source Layout
```text
spacepinn/
|-- pinn.py           PINN model definition
|-- optimizer.py      Legacy `TrajectoryOptimizer` facade on top of the modular optimization package
|-- optimization/     Training loop, dynamics, losses, and optimizer configuration
|-- result.py         Normalized trajectory/result container used by plotting and persistence
|-- runner/           Experiment execution, saved-run layout, reload, and run summaries
|-- config/           Reusable presets and transform functions
|-- plotter.py        High-level plotting entry point
|-- plotting/         Concrete plot implementations for trajectory, thrust, gravity, loss, and orbit views
|-- paper/            Curated paper experiments, Monte Carlo studies, and sweeps
|-- opengoddard/      Required OpenGoddard baselines for the included experiments
|-- pretraining/      Pre-conditioning helper used by the 3D paper setup
`-- experiment/       Shared experiment execution helpers
```

The intent of this layout is:
- `spacepinn.paper`
  Public paper entry points. If you want to reproduce the headline experiments, start here.
- `spacepinn.paper.monte_carlo`
  Reproducibility and robustness runs used by the paper.
- `spacepinn.paper.sweeps`
  Focused search utilities for initialization and soft-BC weights reported in the paper workflow.
- `spacepinn.opengoddard`
  Direct-collocation baselines required by the included paper experiments.
- `spacepinn.pretraining`
  Auxiliary training helpers that are part of the published 3D workflow.
- everything else
  Shared infrastructure for optimization, plotting, configuration, and run persistence.

## Testing
This release includes a focused, reproducible test set covering the selected experiments and the shared runtime pieces they depend on.

Run the full included suite with:

```bash
uv run --with pytest pytest -q
```

The test suite is intentionally smaller than the one in the original development repository. It is meant to validate the published paper scope, not every exploratory branch that existed during research.

## Physical Conventions
The time normalization, physical velocity conventions, and polar kinematic boundary rules are documented in [PHYSICAL_CONVENTIONS.md](./PHYSICAL_CONVENTIONS.md).

## Notes
This repository is a curated publication-focused subset of the larger research codebase. The included experiments are the paper-facing examples and supporting studies; exploratory branches and development-only scripts were intentionally left out so the release stays easier to read and reproduce.
