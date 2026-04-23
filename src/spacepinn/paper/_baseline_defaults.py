from __future__ import annotations


PAPER_BASELINE_FTOL = 1e-11
PAPER_BASELINE_MAX_ITERATION = 100
PAPER_BASELINE_SLSQP_MAXITER = 25


def paper_baseline_solver_kwargs(*, smoke_enabled: bool) -> dict[str, float | int]:
    return {
        "ftol": PAPER_BASELINE_FTOL,
        "max_iteration": 1 if smoke_enabled else PAPER_BASELINE_MAX_ITERATION,
        "slsqp_maxiter": PAPER_BASELINE_SLSQP_MAXITER,
    }
