from __future__ import annotations

import argparse

from spacepinn.paper.sweeps._boundary_weight_sweep import SweepSpec, run_boundary_weight_sweep


def _parse_args():
    parser = argparse.ArgumentParser(description="Dynamic soft-BC weight sweep for the 2D swingby PINN.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--nan-streak-limit", type=int, default=25)
    return parser.parse_args()


def main(*, workers: int = 1, nan_streak_limit: int = 25):
    return run_boundary_weight_sweep(
        spec=SweepSpec(
            label="swingby_soft_bc_weight_search_2d",
            dimension=2,
            dynamic_tof=True,
            selection_metric="min_total_loss",
            plot_title="2D swingby PINN soft-BC weight sweep",
        ),
        workers=workers,
        nan_streak_limit=nan_streak_limit,
    )


if __name__ == "__main__":
    args = _parse_args()
    main(workers=args.workers, nan_streak_limit=args.nan_streak_limit)
