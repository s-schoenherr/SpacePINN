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
```

If `uv` is not installed, see https://docs.astral.sh/uv/.

## Quick Start

Run a paper experiment:

```bash
uv run python -m spacepinn.paper.swingby_2d
uv run python -m spacepinn.paper.swingby_3d
uv run python -m spacepinn.paper.orbit_transfer_fixed_angle
uv run python -m spacepinn.paper.orbit_transfer_free_angle
uv run python -m spacepinn.paper.rendezvous_hold_point_eci
```

Use `--mode mc` or `--mc` on the same entry points to run the full Monte Carlo variant and generate the corresponding boxplots.

Run appendix utilities:

```bash
uv run python -m spacepinn.paper.appendix.boundary_weight_search_2d
uv run python -m spacepinn.paper.appendix.boundary_weight_search_3d
uv run python -m spacepinn.paper.appendix.static_total_time_sweep
```

Inspect exported paper data:

```bash
uv run python examples/inspect_paper_record.py
uv run python examples/inspect_paper_record.py data/runs/orbit_transfer_free_angle
```

Run the core physics checks:

```bash
uv run pytest tests/physics/test_core_physics.py -q
```

Validate a specific saved run:

```bash
uv run pytest tests/physics/test_saved_run_physics.py --run-dir runs/YYYY/MM/<run_id>
```

This command expects a generated training run with `manifest.json`; the exported paper records under `data/runs/` are inspected with the example script above.

For maintainers, `uv run pytest -q` also runs smoke checks for the paper entry points. Saved-run tests are skipped unless `--run-dir` or `RUN_DIRS` is provided.

## Repository Structure

The main paper entry points live in `src/spacepinn/paper/`. Exported paper records are stored in `data/runs/`, while `data/figure1/` through `data/figure8/` provide figure-numbered PDF copies for quick inspection and manuscript inclusion. The reusable PINN, plotting, runner, and OpenGoddard code lives under `src/spacepinn/`.

## Physical Conventions

The time normalization, physical velocity conventions, and polar kinematic boundary rules are documented in [PHYSICAL_CONVENTIONS.md](./PHYSICAL_CONVENTIONS.md).

## Notes

This repository is a reader-facing paper artifact. Exploratory experiments from the internal research repository were intentionally omitted so that the published code path is easier to inspect, run, and reuse.
