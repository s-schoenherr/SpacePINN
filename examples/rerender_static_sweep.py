from __future__ import annotations

from pathlib import Path
from spacepinn.paper.appendix.static_total_time_sweep import plot_record

ROOT = Path(__file__).resolve().parents[1]
record = ROOT / "data" / "runs" / "appendix" / "static_total_time_sweep" / "medium"
plot_record(record, output_dir=ROOT / "temp" / "static_sweep_medium")
print("Wrote plots to temp/static_sweep_medium")
