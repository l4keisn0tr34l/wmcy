#!/usr/bin/env python3
"""Create context/future indices for world-model training.

This script creates NO attack-classification target. It simply states which chronological
network states are visible to the model and which later states it must predict.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("global_states")
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    states = pd.read_csv(args.global_states)
    states = states.sort_values("state_id").reset_index(drop=True)
    ids = states["state_id"].tolist()

    rows = []
    L, H = args.context, args.horizon
    for end_idx in range(L - 1, len(ids) - H):
        context = ids[end_idx-L+1:end_idx+1]
        future = ids[end_idx+1:end_idx+1+H]
        rows.append({
            "sample_id": len(rows),
            "context_state_ids": ";".join(map(str, context)),
            "future_state_ids": ";".join(map(str, future)),
            "context_end_state": context[-1],
            "target_first_state": future[0],
            "target_last_state": future[-1],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows):,} world-model sequence samples -> {out}")


if __name__ == "__main__":
    main()
