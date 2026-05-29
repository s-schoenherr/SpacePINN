# Paper Data

This directory contains the exported records used for the paper figures. Each record follows the same structure:

- `expected_configs/`: serialized experiment or baseline configurations
- `expected_timeseries/`: exported trajectory, force, loss, and timing arrays as `.npz` files
- `expected_summary.json`: scalar summary values used for comparisons
- `source_plots/`: the PDF figures generated from the record
- `metadata.json` and `source_run_config.json`: provenance information

Main paper records are grouped by experiment:

- `three_bodies_2d/deterministic` and `three_bodies_2d/monte_carlo`
- `three_bodies_3d/deterministic` and `three_bodies_3d/monte_carlo`
- `orbit_transfer_fixed_angle`
- `orbit_transfer_free_angle`
- `rendezvous_hold_point_eci`

Appendix records are grouped under `appendix/` and contain the boundary-loss sweeps and the static total-time sweep.

A minimal loading example is available at `examples/load_paper_timeseries.py`.

For quick figure-oriented access, the same plot PDFs are copied into `data/figure1/` through `data/figure8/` at the repository root. Those directories are convenience copies; this `data/runs/` tree remains the provenance-preserving export with configurations, time series, summaries, and metadata.
