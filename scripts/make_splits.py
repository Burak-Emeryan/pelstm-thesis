"""Generate the speed bins and the three pre-declared splits (seed 42).

Usage: .venv/bin/python scripts/make_splits.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import dataset, splits  # noqa: E402


def print_summary(name: str, payload: dict, speed_bins: dict):
    s = payload["summary"]
    print(f"\n--- split {name} ({payload['run_dir']}) ---")
    print(f"  rule      : {payload['rule']}")
    print(f"  n_train / n_val: {s['n_train']} / {s['n_val']}")
    print("  per bin   : " + ", ".join(
        f"b{p['bin']} {p['n_train']}/{p['n_val']}" for p in s["per_bin"]))
    print(f"  train speed range: {s['train_speed_range'][0]:.3f} - {s['train_speed_range'][1]:.3f} m/s")
    print(f"  val   speed range: {s['val_speed_range'][0]:.3f} - {s['val_speed_range'][1]:.3f} m/s")
    if "val_subjects" in payload:
        print(f"  val subjects: {payload['val_subjects']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()

    _, _, _, meta = dataset.load_walking_dataset()
    speed_bins = splits.compute_speed_bins(meta)
    print("speed bins (equal trial count, bin 0 = slowest):")
    for b in speed_bins["bins"]:
        print(f"  bin {b['bin']}: n={b['n_trials']:3d}  speed {b['speed_min']:.3f} - {b['speed_max']:.3f} m/s")

    split_a = splits.make_split_a(meta, speed_bins)
    split_b = splits.make_split_b(meta, speed_bins, split_a)
    split_c = splits.make_split_c(meta, speed_bins)

    splits.write_json(splits.BINS_PATH, speed_bins, force=args.force)
    for payload in (split_a, split_b, split_c):
        splits.write_json(splits.split_path(payload["name"]), payload, force=args.force)
        print_summary(payload["name"], payload, speed_bins)

    print("\nwritten:")
    for p in [splits.BINS_PATH] + [splits.split_path(n) for n in "ABC"]:
        print(f"  {p.relative_to(dataset.data_io.PROJECT_ROOT)}  sha256 {splits.sha256_file(p)[:16]}")


if __name__ == "__main__":
    main()
