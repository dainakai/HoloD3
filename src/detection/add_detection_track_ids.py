#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Assign stable per-detection IDs to a tracks_with_diameter-style CSV. "
            "This pipeline does not track yet, so track_id is just a unique detection id."
        )
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    df = pd.read_csv(args.input)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    else:
        df["row_id"] = np.arange(len(df), dtype=np.int64)

    if "track_id" in df.columns:
        df["track_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
    else:
        insert_at = min(2, len(df.columns))
        df.insert(insert_at, "track_id", np.arange(1, len(df) + 1, dtype=np.int64))

    if "file" in df.columns:
        df["file"] = df["file"].map(lambda x: f"{Path(str(x)).stem}.png")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(
        {
            "input": str(args.input),
            "output": str(args.output),
            "rows": int(len(df)),
            "track_id_policy": "unique_per_detection_no_tracking",
        }
    )


if __name__ == "__main__":
    main()
