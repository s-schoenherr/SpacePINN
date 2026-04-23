from __future__ import annotations


def _format_metric(value, precision: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}g}"
    return str(value)


def format_collection_run_summary(collection_run: dict) -> str:
    entries = collection_run.get("entries", [])
    label = collection_run.get("label", "collection")
    run_id = collection_run.get("run_id", "-")
    run_dir = collection_run.get("run_dir", "-")

    label_width = max(len("label"), *(len(str(entry.get("label", "-"))) for entry in entries)) if entries else len("label")
    source_width = max(len("source"), *(len(str(entry.get("source", "-"))) for entry in entries)) if entries else len("source")
    idx_width = max(len("id"), len(str(len(entries) - 1)) if entries else 1)

    lines = [
        "",
        "*" * 108,
        f"[SWINGBY] Run Summary: {label}",
        f"run_id: {run_id}",
        f"run_dir: {run_dir}",
        "-" * 108,
        f"{'id':<{idx_width}}  {'label':<{label_width}}  {'source':<{source_width}}  {'delta_v':>12}  {'t_total':>12}  {'runtime_s':>12}  {'final_loss':>14}",
        "-" * 108,
    ]

    for index, entry in enumerate(entries):
        result = entry.get("result")
        delta_v = _format_metric(getattr(result, "delta_v", None))
        t_total = _format_metric(getattr(result, "t_total", None))
        runtime_seconds = _format_metric(getattr(result, "runtime_seconds", None))
        final_loss = _format_metric(result.loss[-1] if getattr(result, "loss", None) else None)
        lines.append(
            f"{index:<{idx_width}}  {str(entry.get('label', '-')):<{label_width}}  "
            f"{str(entry.get('source', '-')):<{source_width}}  {delta_v:>12}  {t_total:>12}  {runtime_seconds:>12}  {final_loss:>14}"
        )

    lines.append("*" * 108)
    return "\n".join(lines)


def print_collection_run_summary(collection_run: dict) -> None:
    print(format_collection_run_summary(collection_run))
