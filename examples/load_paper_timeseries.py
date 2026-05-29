from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
record = ROOT / "data" / "runs" / "swingby_2d" / "deterministic" / "expected_timeseries"
first = sorted(record.glob("*.npz"))[0]
with np.load(first) as data:
    print(f"Loaded {first.relative_to(ROOT)}")
    print("arrays:", sorted(data.files))
    if "r" in data.files:
        print("trajectory shape:", data["r"].shape)
