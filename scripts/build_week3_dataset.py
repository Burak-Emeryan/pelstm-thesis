"""Build the Week 3 full-stride walking dataset from the raw .mat files.

Usage: .venv/bin/python scripts/build_week3_dataset.py
"""
from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import dataset  # noqa: E402


def main():
    t0 = time.time()
    X, Y_pos, Y_mom, meta = dataset.build_walking_dataset()
    speeds = np.array([m["speed_mps"] for m in meta])
    rates = collections.Counter(m["rate_hz"] for m in meta)
    feet = collections.Counter(m["foot"] for m in meta)
    seam = np.array([m["seam_ratio"] for m in meta]).ravel()

    print(f"trials                : {len(meta)}")
    print(f"X / Y_pos / Y_mom     : {X.shape} {Y_pos.shape} {Y_mom.shape}")
    print(f"foot                  : {dict(feet)}")
    print(f"rx_consistent         : {sum(m['rx_consistent'] for m in meta)}")
    print(f"EMG rate              : {dict(rates)}")
    print(f"speed m/s             : min {speeds.min():.3f} median {np.median(speeds):.3f} max {speeds.max():.3f}")
    print(f"n_windows             : min {min(m['n_windows'] for m in meta)} max {max(m['n_windows'] for m in meta)}")
    print(f"extra end window      : {sum(m['extra_end_window'] for m in meta)} / {len(meta)} trials")
    print(f"seam ratio            : median {np.median(seam):.2f} p90 {np.percentile(seam, 90):.2f} max {seam.max():.2f}")
    print(f"flagged rate trials   : {[m['trial_id'] for m in meta if m['rate_flags']]}")
    print(f"saved                 : {dataset.NPZ_PATH}")
    print(f"                        {dataset.META_PATH}")
    print(f"wall time             : {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
