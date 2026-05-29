from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import spacepinn

from spacepinn.paper.appendix._boundary_weight_sweep import SweepSpec, _plot_results, run_boundary_weight_sweep


FIG_PREFIX = "weights3d_dynamic"
DATA_DIR = Path(spacepinn.__file__).resolve().parents[2] / "data" / "runs" / "appendix" / "boundary_weight_search_3d"


def _summary_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "aggregate_summary.json",
        run_dir / "aggregate_summaries" / "aggregate_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No aggregate summary found in {run_dir}")


def replot_saved_run(
    run_dir: str | Path = DATA_DIR,
    *,
    output_dir: str | Path | None = None,
    print_summary: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    target_dir = Path(output_dir) if output_dir is not None else run_dir / "source_plots"
    target_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(_summary_path(run_dir).read_text(encoding="utf-8"))
    output_path = target_dir / f"{FIG_PREFIX}.pdf"
    _plot_results(
        list(summary["rows"]),
        output_path=output_path,
        selection_metric=str(summary.get("selection_metric", "min_total_loss")),
        title="",
        paper_style=True,
        best_label_precision=3,
    )

    if print_summary:
        best = summary.get("best_row") or {}
        print(
            f"[boundary_weight_search_3d_dynamic] wrote {output_path} "
            f"(best lambda_BC={float(best.get('lambda_bc', float('nan'))):.6g})"
        )
    return summary


def main(
    *,
    from_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_sweep: bool = False,
    workers: int = 1,
    print_summary: bool = True,
) -> dict[str, Any]:
    if run_sweep:
        return run_boundary_weight_sweep(
            spec=SweepSpec(
                label="boundary_weight_search_3d_dynamic",
                dimension=3,
                dynamic_tof=True,
                selection_metric="min_total_loss",
            ),
            workers=workers,
        )
    return replot_saved_run(from_run or DATA_DIR, output_dir=output_dir, print_summary=print_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary loss weight search (3D).")
    parser.add_argument("--from-run", type=Path, default=None, help="Existing exported record or run directory to replot.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for re-rendered plot output.")
    parser.add_argument("--run-sweep", action="store_true", help="Run the sweep instead of replotting exported paper data.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        from_run=args.from_run,
        output_dir=args.output_dir,
        run_sweep=args.run_sweep,
        workers=args.workers,
        print_summary=not args.skip_summary,
    )
