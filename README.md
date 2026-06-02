# SpacePINN

SpacePINN studies spacecraft swing-by, orbit-transfer, and rendezvous problems with physics-informed neural networks under gravitational and thrust-driven dynamics in Cartesian and polar coordinates. Its main research question is how strongly constrained PINN formulations behave relative to standard soft-penalty approaches: the hard-constrained variants transform the network output with distance functions so boundary conditions are satisfied by construction, while the vanilla formulations enforce the same conditions through loss terms. The repository benchmarks both against OpenGoddard, a direct optimal-control method based on collocation and nonlinear programming, and provides the persistence, plotting, and validation tooling needed to compare the resulting trajectories quantitatively.

This publication repository is intentionally smaller than the internal research repository: it contains the reusable core functionality, the paper-facing experiment entry points, and the saved paper data used to generate the reported figures.

## Data and Figures

The saved run records are available under `data/runs/`. Each record contains the exported configurations, expected time series, summary files, source plots, and metadata needed to inspect or re-render the published results. For quick access to the figure PDFs used in the manuscript, `data/figure1/` through `data/figure8/` collect the corresponding plots by figure number.

## Installation

```bash
git clone https://github.com/s-schoenherr/SpacePINN.git
cd SpacePINN
uv sync --extra dev
uv run pytest -q
```

If `uv` is not installed, see https://docs.astral.sh/uv/.

## Quick Start

Run the curated smoke tests:

```bash
uv run pytest -q
```

Run a paper experiment:

```bash
uv run python -m spacepinn.paper.swingby_2d --mode single
uv run python -m spacepinn.paper.swingby_3d --mode single
uv run python -m spacepinn.paper.orbit_transfer_fixed_angle --mode single
uv run python -m spacepinn.paper.orbit_transfer_free_angle --mode single
uv run python -m spacepinn.paper.rendezvous_hold_point_eci --mode single
```

Use `--mode mc` on the same entry points to run the full Monte Carlo variant and generate the corresponding boxplots.

Run appendix utilities:

```bash
uv run python -m spacepinn.paper.appendix.boundary_weight_search_2d
uv run python -m spacepinn.paper.appendix.boundary_weight_search_3d
uv run python -m spacepinn.paper.appendix.static_total_time_sweep
```

Inspect exported paper data:

```bash
uv run python examples/load_paper_timeseries.py
```

Re-render the saved static total-time sweep plots from the exported data:

```bash
uv run python examples/rerender_static_sweep.py
```

## Repository Layout

```text
src/spacepinn/
|-- config/            curated boundary-condition presets and output transforms
|-- optimization/      PINN dynamics, loss terms, and training loop
|-- runner/            experiment execution, persistence, and saved-run loading
|-- experiment/        small helpers for composing reusable experiment entries
|-- plotting/          reusable plot helpers and paper plotting style
|-- opengoddard/       direct-collocation baselines used in the paper
|-- pretraining/       3D kinematic-to-geometric pre-conditioning helper
`-- paper/             paper-facing experiment entry points
    |-- swingby_2d.py          single-seed and Monte-Carlo swing-by runs
    |-- swingby_3d.py          single-seed and Monte-Carlo swing-by runs
    |-- appendix/              boundary-weight and static-TOF appendix utilities
    |-- orbit_transfer_fixed_angle.py
    |-- orbit_transfer_free_angle.py
    `-- rendezvous_hold_point_eci.py

data/runs/           exported configurations, time series, summaries, and source plots
data/figure1-8/       figure-numbered copies of the manuscript plot PDFs
examples/             small scripts for loading data and re-rendering saved records
tests/                smoke tests plus core physics and transform validation
```

## Physical Conventions

The time normalization, physical velocity conventions, and polar kinematic boundary rules are documented in [PHYSICAL_CONVENTIONS.md](./PHYSICAL_CONVENTIONS.md).

## Notes

This repository is a reader-facing paper artifact. Exploratory experiments from the internal research repository were intentionally omitted so that the published code path is easier to inspect, run, and reuse.
