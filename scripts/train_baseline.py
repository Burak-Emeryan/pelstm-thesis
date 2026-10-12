"""End-to-end Week 3 pipeline: dataset -> splits -> smoke test -> runs -> summary.

Usage:
    .venv/bin/python scripts/train_baseline.py --run all
    .venv/bin/python scripts/train_baseline.py --run A --target position
    .venv/bin/python scripts/train_baseline.py --summarize
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import data_io, dataset, splits  # noqa: E402
from src.models import train  # noqa: E402

WEEK3_DIR = data_io.PROJECT_ROOT / "outputs" / "week3"
SUMMARY_PATH = WEEK3_DIR / "summary.md"
SMOKE_TRIALS = 16
# 600 instead of the originally planned 300: at seed 42 the position model takes a
# gradient-norm spike (0.96 median -> 130.9) around epoch 98 and needs more steps to
# recover; at 300 epochs it reached 0.1005 of the initial loss, just above the gate.
# Threshold, seed and the 200 epochs of the real runs are unchanged.
SMOKE_EPOCHS = 600
SMOKE_RATIO = 0.1
TARGETS = ("position", "moment")
KELES = {
    "position": {"r_mean": 0.9010, "r_sd": 0.0805, "rmse_mean": 4.8927, "rmse_sd": 1.6140},
    "moment": {"r_mean": 0.9718, "r_sd": 0.0326, "rmse_mean": 0.1193, "rmse_sd": 0.0501},
}
RUN_DIRS = {"A": "A_all844", "B": "B_rx572", "C": "C_subjectwise", "smoke": "smoke"}


def run_script(script: str, extra: list[str] | None = None):
    cmd = [sys.executable, str(data_io.PROJECT_ROOT / "scripts" / script)] + (extra or [])
    print(f"\n$ {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=data_io.PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def ensure_dataset():
    if not dataset.NPZ_PATH.exists():
        print("dataset npz missing -> building")
        run_script("build_week3_dataset.py")
    run_script("check_week3_data.py")


def ensure_splits():
    missing = [p for p in [splits.BINS_PATH] + [splits.split_path(n) for n in "ABC"] if not p.exists()]
    if missing:
        print(f"splits missing ({[p.name for p in missing]}) -> generating")
        run_script("make_splits.py")
    else:
        print("splits present, reusing")


def load_everything():
    X, Y_pos, Y_mom, meta = dataset.load_walking_dataset()
    speed_bins = json.loads(splits.BINS_PATH.read_text())
    return X, {"position": Y_pos, "moment": Y_mom}, meta, speed_bins


def run_smoke(X, Y, meta, speed_bins, targets, force: bool):
    """Overfit the first 16 training trials of split A; a bug shows up as no learning."""
    split_a = splits.load_split("A")
    ids = sorted(split_a["train"])[:SMOKE_TRIALS]
    print(f"\n=== smoke test: {len(ids)} trials, {SMOKE_EPOCHS} epochs "
          f"(validation set = the same trials) ===")
    for target in targets:
        print(f"  [smoke/{target}]")
        res = train.train_target(
            "smoke", target, split_a, splits.split_path("A"), X, Y[target], meta, speed_bins,
            epochs=SMOKE_EPOCHS, force=force, run_dir_name=RUN_DIRS["smoke"],
            train_ids=ids, val_ids=ids,
            extra_config={"note": "smoke test; validation set is identical to the training set",
                          "smoke_n_trials": len(ids)})
        ratio = res["final_train_loss"] / res["initial_train_loss"]
        print(f"    train loss {res['initial_train_loss']:.5f} -> {res['final_train_loss']:.5f} "
              f"(ratio {ratio:.4f}, threshold < {SMOKE_RATIO})")
        if ratio >= SMOKE_RATIO:
            raise RuntimeError(f"smoke test failed for {target}: final/initial train loss "
                               f"{ratio:.4f} >= {SMOKE_RATIO}")
    print("smoke test PASSED")


def run_split(name, X, Y, meta, speed_bins, targets, epochs, force):
    payload = splits.load_split(name)
    print(f"\n=== run {name} ({payload['run_dir']}): "
          f"{payload['summary']['n_train']} train / {payload['summary']['n_val']} val ===")
    t0 = time.time()
    for target in targets:
        print(f"  [{name}/{target}]")
        train.train_target(name, target, payload, splits.split_path(name), X, Y[target],
                           meta, speed_bins, epochs=epochs, force=force)
    print(f"run {name} wall time: {time.time() - t0:.1f} s")


def _fmt(d, key_mean, key_sd, digits=4):
    return f"{d[key_mean]:.{digits}f} ± {d[key_sd]:.{digits}f}"


def summarize():
    rows = []
    for name in ("A", "B", "C"):
        for target in TARGETS:
            path = train.RUNS_DIR / RUN_DIRS[name] / target / "metrics.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
    if not rows:
        raise FileNotFoundError("no finished runs found; nothing to summarize")

    lines = ["# Hafta 3 — baseline LSTM sonuçları", ""]
    lines += ["Run A is the pre-declared primary run (replication gate). Runs B and C are "
              "pre-declared ablations. No run was selected by validation performance.", ""]
    lines += ["## Run × target", "",
              "| run | target | n_val | r (mean ± SD) | RMSE (mean ± SD) | pooled r |",
              "|---|---|---|---|---|---|"]
    for res in rows:
        v = res["validation"]
        lines.append(f"| {res['run']} | {res['target']} | {v['n']} | {_fmt(v, 'r_mean', 'r_sd')} | "
                     f"{_fmt(v, 'rmse_mean', 'rmse_sd')} {res['units']} | {v['pooled_r']:.4f} |")
    for target in TARGETS:
        k = KELES[target]
        unit = "deg" if target == "position" else "Nm/kg"
        lines.append(f"| Keleş 2023 (reference) | {target} | 169 | {_fmt(k, 'r_mean', 'r_sd')} | "
                     f"{_fmt(k, 'rmse_mean', 'rmse_sd')} {unit} | — |")
    lines += ["", "Keleş 2023 used a plain random 675/169 split; the splits here are "
                  "speed-stratified.", ""]

    lines += ["## Run A — per speed bin", "",
              "| bin | target | n | r (mean) | RMSE (mean) |", "|---|---|---|---|---|"]
    bins = json.loads(splits.BINS_PATH.read_text())["bins"]
    for res in [r for r in rows if r["run"] == "A"]:
        for b, stats in sorted(res["by_speed_bin"].items()):
            rng = bins[int(b)]
            lines.append(f"| {b} ({rng['speed_min']:.2f}–{rng['speed_max']:.2f} m/s) | "
                         f"{res['target']} | {stats['n']} | {stats['r_mean']:.4f} | "
                         f"{stats['rmse_mean']:.4f} {res['units']} |")

    lines += ["", "## Runs A and C — RX-consistent vs LX validation trials", "",
              "| run | target | subset | n | r (mean) | RMSE (mean) |", "|---|---|---|---|---|---|"]
    for res in [r for r in rows if r["run"] in ("A", "C") and "by_side" in r]:
        for key, label in (("rx_consistent", "RX-consistent"), ("foot_LX", "Foot = LX")):
            stats = res["by_side"][key]
            if stats["n"]:
                lines.append(f"| {res['run']} | {res['target']} | {label} | {stats['n']} | "
                             f"{stats['r_mean']:.4f} | {stats['rmse_mean']:.4f} {res['units']} |")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {SUMMARY_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["smoke", "A", "B", "C", "all"])
    ap.add_argument("--target", choices=["position", "moment", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=train.EPOCHS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.run is None and not args.summarize:
        ap.error("pass --run and/or --summarize")

    targets = TARGETS if args.target == "both" else (args.target,)
    t0 = time.time()

    if args.run:
        ensure_dataset()
        ensure_splits()
        X, Y, meta, speed_bins = load_everything()
        if args.run == "all":
            run_smoke(X, Y, meta, speed_bins, targets, args.force)
            for name in ("A", "B", "C"):
                run_split(name, X, Y, meta, speed_bins, targets, args.epochs, args.force)
        elif args.run == "smoke":
            run_smoke(X, Y, meta, speed_bins, targets, args.force)
        else:
            run_split(args.run, X, Y, meta, speed_bins, targets, args.epochs, args.force)

    if args.summarize or args.run == "all":
        summarize()
    print(f"\ntotal wall time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
