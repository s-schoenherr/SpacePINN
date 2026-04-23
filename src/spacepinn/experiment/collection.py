from __future__ import annotations

from ..runner import print_collection_run_summary, run_experiment_collection
from .specs import CollectionSpec


def finalize_collection(
    spec: CollectionSpec,
    *,
    skip_plots: bool = False,
    print_summary: bool = True,
) -> dict:
    collection_run = run_experiment_collection(
        configs=[],
        additional_entries=[entry.as_collection_entry() for entry in spec.entries],
        label=spec.label,
        run_root=spec.run_root,
    )

    if print_summary:
        print_collection_run_summary(collection_run)
        if spec.summary_fn is not None:
            spec.summary_fn(collection_run)

    if not skip_plots and spec.plot_fn is not None:
        spec.plot_fn(collection_run)

    return collection_run
