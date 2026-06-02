from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = ROOT / "data" / "runs" / "swingby_2d" / "deterministic"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _print_record_header(record: Path) -> None:
    metadata = _load_json(record / "metadata.json")
    summary = _load_json(record / "expected_summary.json")
    print(f"Record: {_relative(record)}")
    if metadata:
        print(f"Source run: {metadata.get('source_run_id', metadata.get('run_id', 'unknown'))}")
    if summary:
        print(f"Entries: {len(summary.get('entries', []))}")


def _print_timeseries(record: Path, *, limit: int) -> None:
    timeseries_dir = record / "expected_timeseries"
    files = sorted(timeseries_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {_relative(timeseries_dir)}")
        return

    for index, npz_path in enumerate(files):
        if index >= limit:
            remaining = len(files) - limit
            print(f"... {remaining} more file(s)")
            break

        with np.load(npz_path) as data:
            shapes = {name: tuple(data[name].shape) for name in sorted(data.files)}
        print(f"\n{_relative(npz_path)}")
        for name, shape in shapes.items():
            print(f"  {name}: {shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an exported SpacePINN paper record.")
    parser.add_argument(
        "record",
        nargs="?",
        default=str(DEFAULT_RECORD),
        help="Path to a record directory under data/runs.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of time-series files to show.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = Path(args.record)
    if not record.is_absolute():
        record = ROOT / record
    record = record.resolve()

    _print_record_header(record)
    _print_timeseries(record, limit=max(1, int(args.limit)))


if __name__ == "__main__":
    main()
