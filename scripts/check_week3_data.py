"""Verify the Week 3 full-stride dataset (checks C1-C7).

Each check asserts its own conditions, prints PASS/FAIL and contributes to
outputs/week3/data/dataset_report.json. The script exits non-zero if any
check failed; failures are printed with their full traceback.

Usage: .venv/bin/python scripts/check_week3_data.py
"""
from __future__ import annotations

import collections
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import data_io, dataset, features, fullstride  # noqa: E402

GATE1_DIR = data_io.PROJECT_ROOT / "outputs" / "gate1"
REPORT_PATH = dataset.DATA_OUT_DIR / "dataset_report.json"
FIG_PATH = dataset.DATA_OUT_DIR / "fullstride_vs_gate1.png"

EXPECTED = {
    "n_trials": 844,
    "foot_RX": 581,
    "foot_LX": 263,
    "rx_consistent": 572,
    "rates": {960.0: 324, 800.0: 273, 1000.0: 247},
    "flagged": ["S12_D016", "S35_D000"],
    "speed_min_allowed": 0.25,
    "speed_max_allowed": 2.4,
    "n_emg_min": 569,
}
GATE1_TRIALS = ["S02_D001", "S03_D000", "S01_D000", "S41_D001", "S08_D018"]


def naive_window_features(x: np.ndarray, k: int, win: int) -> np.ndarray:
    """Reference implementation, written independently of src/data/fullstride.py.

    Window ending at original sample k covers positions j = k-win+1 .. k;
    a negative j wraps to x[j + (N-1)] because sample N-1 is the same gait
    event as sample 0 one cycle earlier. Returns [IEMG_c0, WL_c0, IEMG_c1, WL_c1].
    """
    n = x.shape[1]
    period = n - 1
    cols = []
    for j in range(k - win + 1, k + 1):
        cols.append(x[:, j + period] if j < 0 else x[:, j])
    w = np.stack(cols, axis=1)                       # (C, win)
    assert w.shape[1] == win
    out = []
    for c in range(w.shape[0]):
        out.append(float(np.sum(np.abs(w[c]))))
        out.append(float(np.sum(np.abs(w[c][1:] - w[c][:-1]))))
    return np.array(out, dtype=np.float64)


def wrap_padded(x: np.ndarray, win: int) -> np.ndarray:
    """Rebuild the wrap-padded signal (documented rule) to query arbitrary window ends."""
    n = x.shape[1]
    return np.concatenate([x[:, n - win:n - 1], x], axis=1)


def load_trial_emg(subject: int, data_idx: int):
    s = data_io.load_subject(subject)
    d = data_io.trials(s)[data_idx]
    ch = [data_io.channel_index(s, "EmgVarName", m) for m in dataset.EMG_MUSCLES]
    return np.asarray(d.EMG, dtype=np.float64)[ch], data_io.derive_rates(s, d).emg


def check_c1(meta, report):
    n = len(meta)
    feet = collections.Counter(m["foot"] for m in meta)
    rates = collections.Counter(m["rate_hz"] for m in meta)
    rx = sum(m["rx_consistent"] for m in meta)
    flagged = sorted(m["trial_id"] for m in meta if m["rate_flags"])
    speeds = np.array([m["speed_mps"] for m in meta])
    s18 = sum(1 for m in meta if m["subject"] == 18)

    print(f"  trials {n} | foot {dict(feet)} | rx_consistent {rx}")
    print(f"  rates {dict(rates)}")
    print(f"  Subject18 walking trials: {s18} (foot RX of those: "
          f"{sum(1 for m in meta if m['subject'] == 18 and m['foot'] == 'RX')})")
    print(f"  rate-flagged trials: {flagged}")
    print(f"  speed m/s: min {speeds.min():.3f} max {speeds.max():.3f} median {np.median(speeds):.3f}")

    assert n == EXPECTED["n_trials"], n
    assert feet["RX"] == EXPECTED["foot_RX"] and feet["LX"] == EXPECTED["foot_LX"], dict(feet)
    assert rx == EXPECTED["rx_consistent"], rx
    assert {k: v for k, v in rates.items()} == EXPECTED["rates"], dict(rates)
    assert flagged == sorted(EXPECTED["flagged"]), flagged
    assert np.isfinite(speeds).all()
    assert speeds.min() >= EXPECTED["speed_min_allowed"], speeds.min()
    assert speeds.max() <= EXPECTED["speed_max_allowed"], speeds.max()

    report["C1"] = {"n_trials": n, "foot": dict(feet), "rx_consistent": rx,
                    "rates": {str(k): v for k, v in rates.items()}, "flagged": flagged,
                    "subject18_trials": s18,
                    "speed_min": float(speeds.min()), "speed_max": float(speeds.max()),
                    "speed_median": float(np.median(speeds))}


def check_c2(X, Y_pos, Y_mom, meta, report):
    assert X.shape == (len(meta), 101, 4), X.shape
    assert Y_pos.shape == (len(meta), 101) and Y_mom.shape == (len(meta), 101)
    assert np.isfinite(X).all() and np.isfinite(Y_pos).all() and np.isfinite(Y_mom).all()
    n_emg = np.array([m["n_emg"] for m in meta])
    assert n_emg.min() >= EXPECTED["n_emg_min"], n_emg.min()
    print(f"  X {X.shape} Y_pos {Y_pos.shape} Y_mom {Y_mom.shape} all finite")
    print(f"  n_emg: min {n_emg.min()} max {n_emg.max()}")
    report["C2"] = {"X_shape": list(X.shape), "n_emg_min": int(n_emg.min()),
                    "n_emg_max": int(n_emg.max()), "all_finite": True}


def check_c3(meta, report):
    """Naive reference vs implementation for 6 trials spanning all three rates."""
    by_rate = collections.defaultdict(list)
    for m in meta:
        by_rate[m["rate_hz"]].append(m)
    picks = []
    for rate in sorted(by_rate):
        trials_r = by_rate[rate]
        picks += [trials_r[0], trials_r[len(trials_r) // 2]]
    assert len(picks) == 6, len(picks)

    rows = []
    for m in picks:
        x, rate = load_trial_emg(m["subject"], m["data_idx"])
        _, info = fullstride.full_stride_features(x, rate)
        win, hop = info["window_samples"], info["hop_samples"]
        n = info["n_samples"]
        feats, ends = info["features"], info["window_end_idx"]
        x_ext = wrap_padded(x, win)

        for k in [0, 1, win - 2, win - 1, int(n // 2), n - 1]:
            naive = naive_window_features(x, k, win)
            hit = np.flatnonzero(ends == k)
            if hit.size:
                impl, src = feats[hit[0]], "feature-matrix row"
            else:
                impl = features.extract_features(x_ext[:, k:k + win], rate)[0]
                src = "window ending at k"
            assert np.allclose(naive, impl, rtol=1e-12, atol=1e-12), (m["trial_id"], k, naive, impl)
            rows.append({"trial_id": m["trial_id"], "rate_hz": rate, "k": int(k), "source": src,
                         "max_abs_diff": float(np.max(np.abs(naive - impl)))})
        print(f"  {m['trial_id']} @{rate:.0f}Hz N={n} W={win} H={hop}: 6/6 window ends match naive reference")

    report["C3"] = {"trials": [m["trial_id"] for m in picks], "comparisons": rows,
                    "max_abs_diff_overall": max(r["max_abs_diff"] for r in rows)}


def check_c4(X, meta, report):
    """Grid endpoints equal the naive features of the windows ending at k=0 and k=N-1."""
    by_rate = collections.defaultdict(list)
    for i, m in enumerate(meta):
        by_rate[m["rate_hz"]].append(i)
    idxs = [by_rate[r][0] for r in sorted(by_rate)] + [by_rate[r][-1] for r in sorted(by_rate)]

    details = []
    for i in idxs:
        m = meta[i]
        x, rate = load_trial_emg(m["subject"], m["data_idx"])
        win = m["window_samples"]
        n = m["n_emg"]
        first = naive_window_features(x, 0, win)
        last = naive_window_features(x, n - 1, win)
        assert np.allclose(X[i, 0], first, rtol=1e-12, atol=1e-12), (m["trial_id"], X[i, 0], first)
        assert np.allclose(X[i, 100], last, rtol=1e-12, atol=1e-12), (m["trial_id"], X[i, 100], last)
        details.append({"trial_id": m["trial_id"],
                        "diff0": float(np.max(np.abs(X[i, 0] - first))),
                        "diff100": float(np.max(np.abs(X[i, 100] - last)))})
    print(f"  {len(idxs)} trials: X[0] and X[100] match the naive k=0 / k=N-1 windows")
    report["C4"] = {"trials": details}


def check_c5(X, meta, report):
    """Soft comparison against the Gate 1 features on their shared grid points."""
    index = {m["trial_id"]: i for i, m in enumerate(meta)}
    results, warnings_ = [], []
    for tid in GATE1_TRIALS:
        f_npy = GATE1_DIR / f"trial_{tid}_features.npy"
        f_json = GATE1_DIR / f"trial_{tid}_meta.json"
        if not (f_npy.exists() and f_json.exists()):
            print(f"  {tid}: Gate 1 files missing, skipped")
            results.append({"trial_id": tid, "skipped": True})
            continue
        g1 = np.load(f_npy)
        g1_meta = json.loads(f_json.read_text())
        grid = np.asarray(g1_meta["pct_stride_grid"], dtype=int)
        assert g1.shape == (grid.size, 4), (g1.shape, grid.size)
        new = X[index[tid]][grid]
        rs = []
        for c in range(4):
            a, b = g1[:, c], new[:, c]
            rs.append(float(np.corrcoef(a, b)[0, 1]))
        print(f"  {tid}: per-column r = {[round(r, 4) for r in rs]} over {grid.size} shared points")
        if min(rs) < 0.98:
            warnings_.append(f"{tid}: min column r {min(rs):.4f} < 0.98")
        results.append({"trial_id": tid, "n_shared": int(grid.size), "r_per_column": rs})
    for w in warnings_:
        print(f"  WARNING {w}")
    report["C5"] = {"results": results, "warnings": warnings_}


def check_c6(X, meta, report):
    index = {m["trial_id"]: i for i, m in enumerate(meta)}
    cols = {"IEMG_GMax": 0, "IEMG_VM": 2}
    fig, axes = plt.subplots(len(GATE1_TRIALS), 2, figsize=(13, 2.6 * len(GATE1_TRIALS)), squeeze=False)
    plotted = 0
    for row, tid in enumerate(GATE1_TRIALS):
        i = index[tid]
        f_npy = GATE1_DIR / f"trial_{tid}_features.npy"
        f_json = GATE1_DIR / f"trial_{tid}_meta.json"
        have_g1 = f_npy.exists() and f_json.exists()
        if have_g1:
            g1 = np.load(f_npy)
            grid = np.asarray(json.loads(f_json.read_text())["pct_stride_grid"], dtype=int)
            plotted += 1
        for col, (name, c) in enumerate(cols.items()):
            ax = axes[row, col]
            ax.axvspan(0, 25, color="0.85", label="0-25 % (wrap-padded region)")
            ax.plot(np.arange(101), X[i, :, c], color="tab:red", lw=1.5, label="full stride (new)")
            if have_g1:
                ax.plot(grid, g1[:, c], color="k", lw=1.2, ls="--", label="Gate 1 (truncated)")
            ax.set_title(f"{tid} — {name}")
            ax.set_xlabel("% stride")
            ax.set_ylabel("mV·samples")
            ax.set_xlim(0, 100)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=110)
    plt.close(fig)
    print(f"  figure written: {FIG_PATH} ({plotted}/{len(GATE1_TRIALS)} trials with Gate 1 overlay)")
    report["C6"] = {"figure": str(FIG_PATH.relative_to(data_io.PROJECT_ROOT)), "gate1_overlays": plotted}


def check_c7(meta, report):
    seam = np.array([m["seam_ratio"] for m in meta], dtype=np.float64)
    flat = seam.ravel()
    stats = {"median": float(np.median(flat)), "p90": float(np.percentile(flat, 90)),
             "max": float(flat.max()),
             "per_channel_median": [float(np.median(seam[:, c])) for c in range(seam.shape[1])]}
    print(f"  seam ratio |x[N-1]-x[0]| / median|diff|: median {stats['median']:.2f} "
          f"p90 {stats['p90']:.2f} max {stats['max']:.2f}")
    print(f"  per-channel median [GMax, VM]: {[round(v, 2) for v in stats['per_channel_median']]}")
    assert np.isfinite(flat).all()
    report["C7"] = stats


def main():
    X, Y_pos, Y_mom, meta = dataset.load_walking_dataset()
    report: dict = {}
    checks = [
        ("C1 counts, rates, flags, speed", lambda: check_c1(meta, report)),
        ("C2 shapes and finiteness", lambda: check_c2(X, Y_pos, Y_mom, meta, report)),
        ("C3 naive reference windows", lambda: check_c3(meta, report)),
        ("C4 grid endpoints", lambda: check_c4(X, meta, report)),
        ("C5 Gate 1 consistency (soft)", lambda: check_c5(X, meta, report)),
        ("C6 comparison figure", lambda: check_c6(X, meta, report)),
        ("C7 seam ratio distribution", lambda: check_c7(meta, report)),
    ]
    failed = []
    for name, fn in checks:
        print(f"\n=== {name} ===")
        try:
            fn()
        except AssertionError:
            traceback.print_exc()
            print(f"FAIL {name}")
            failed.append(name)
            report[name.split()[0]] = {"status": "FAIL"}
        else:
            print(f"PASS {name}")
            report.setdefault(name.split()[0], {})["status"] = "PASS"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1))
    print(f"\nreport written: {REPORT_PATH}")
    if failed:
        print(f"FAILED CHECKS: {failed}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
