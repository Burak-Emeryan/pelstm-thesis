"""Gate 1 end-to-end: explore -> select trials -> load -> features -> align -> save -> plot.

Usage:
    .venv/bin/python scripts/run_gate1.py                  # default trial set
    .venv/bin/python scripts/run_gate1.py --trials 6:0 12:3  # subject:DataIndex (0-based)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import align, data_io, features  # noqa: E402

OUT = data_io.PROJECT_ROOT / "outputs" / "gate1"
WINDOW_MS, HOP_MS = 150.0, 25.0
N_SUBJECTS = 50
EXPECTED_WALKING = 844

GMAX, VM = "Gluteus Maximus", "Vastus Medialis"
ANKLE_ANG, ANKLE_MOM = "AnkleFlx", "AnkleFlxMom"


def trial_id(sub: int, idx: int) -> str:
    return f"S{sub:02d}_D{idx:03d}"


def week1_explore(s):
    print("=== Week 1: struct exploration (", s.name, ") ===")
    print("L = len(s.Data) =", len(data_io.trials(s)))
    for f in ["EmgVarName", "AngVarName", "MomVarName", "GrfVarName"]:
        print(f"{f}: {data_io.var_names(s, f)}")
    print("GMax idx =", data_io.channel_index(s, "EmgVarName", GMAX),
          "| VM idx =", data_io.channel_index(s, "EmgVarName", VM))
    d = data_io.trials(s)[data_io.walking_indices(s)[0]]
    for k, v in data_io.describe_trial(s, d).items():
        print(f"  {k}: {v}")
    print("  -> .Ang/.Mom are %stride (101 cols); .EMG/.Grf are raw samples at their own rates")


def survey_all():
    """Walk every subject: counts, rates, channel order, per-trial flags."""
    walking, flags = [], []
    emg_orders, rate_cfgs, emg_side = collections.Counter(), collections.Counter(), {}
    for sub in range(1, N_SUBJECTS + 1):
        s = data_io.load_subject(sub)
        emg_orders[tuple(data_io.var_names(s, "EmgVarName"))] += 1
        rate_cfgs[(int(s.KinFreq), int(s.GrfFreq), int(s.EMGFreq))] += 1
        emg_side[sub] = str(s.EMGSide).strip()
        for idx in data_io.walking_indices(s):
            d = data_io.trials(s)[idx]
            r = data_io.derive_rates(s, d)
            n = np.shape(d.EMG)[1]
            walking.append(dict(sub=sub, idx=idx, foot=str(d.Foot).strip(), n_emg=n,
                                emg_rate=r.emg, emg_from_marker=round(r.emg_from_marker, 2),
                                emg_from_cadence=round(r.emg_from_cadence, 2),
                                stride_s=round(r.stride_time_s, 4)))
            flags += [f"{trial_id(sub, idx)}: {f}" for f in r.flags]
    return walking, flags, emg_orders, rate_cfgs, emg_side


def build_triplet(s, idx):
    d = data_io.trials(s)[idx]
    rate = data_io.derive_rates(s, d).emg
    ch = [data_io.channel_index(s, "EmgVarName", GMAX), data_io.channel_index(s, "EmgVarName", VM)]
    emg = np.asarray(d.EMG, dtype=np.float64)[ch]           # (2, N): GMax, VM
    F = features.extract_features(emg, rate, WINDOW_MS, HOP_MS)  # (T_win, 4)
    ends = features.window_end_indices(emg.shape[1], rate, WINDOW_MS, HOP_MS)
    targets = np.vstack([d.Ang[data_io.channel_index(s, "AngVarName", ANKLE_ANG)],
                         d.Mom[data_io.channel_index(s, "MomVarName", ANKLE_MOM)]])
    X, Y, grid_idx = align.align_trial(F, ends, emg.shape[1], targets)
    return dict(d=d, rate=rate, emg=emg, F=F, ends=ends, X=X, Y=Y, grid_idx=grid_idx)


def plot_week1(s, sub, idx, path):
    d = data_io.trials(s)[idx]
    rate = data_io.derive_rates(s, d).emg
    g = np.asarray(d.EMG)[data_io.channel_index(s, "EmgVarName", GMAX)]
    ang = d.Ang[data_io.channel_index(s, "AngVarName", ANKLE_ANG)]
    t = np.arange(g.size) / rate
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    ax[0].plot(t, g, lw=0.6)
    ax[0].set(title=f"{trial_id(sub, idx)} raw GMax EMG — own time axis ({g.size} samples @ {rate:.0f} Hz)",
              xlabel="time since heel strike (s)", ylabel="mV")
    ax[1].plot(align.STRIDE_GRID, ang, "k.-", ms=3)
    ax[1].set(title="Ankle dorsiflexion angle — %stride axis (101 points)", xlabel="% stride", ylabel="deg")
    ax[2].plot(align.sample_to_pct_stride(np.arange(g.size), g.size), g, lw=0.6, label="GMax EMG (mV)")
    ax2 = ax[2].twinx()
    ax2.plot(align.STRIDE_GRID, ang, "k", lw=1.5, label="ankle angle (deg)")
    ax[2].set(title="Common axis: EMG sample k -> 100·k/(N−1) % stride", xlabel="% stride", ylabel="mV")
    ax2.set_ylabel("deg")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_week2(results, path):
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 3.2 * len(results)), squeeze=False)
    for row, (tid, res) in enumerate(results):
        n = res["emg"].shape[1]
        t = np.arange(n) / res["rate"]
        t_end = res["ends"] / res["rate"]
        for c, name in enumerate(["GMax", "VM"]):
            a = axes[row, c]
            a.plot(t, res["emg"][c], lw=0.4, color="0.6", label="raw EMG (mV)")
            a.set_ylabel("mV")
            b = a.twinx()
            b.plot(t_end, res["F"][:, 2 * c], "o-", ms=2.5, color="tab:blue", label="IEMG")
            b.plot(t_end, res["F"][:, 2 * c + 1], "s-", ms=2.5, color="tab:orange", label="WL")
            b.set_ylabel("feature (mV·samples)")
            a.set_title(f"{tid} {name} @ {res['rate']:.0f} Hz — 150 ms window, 25 ms hop, causal (end) stamps")
            a.set_xlabel("time since heel strike (s)")
            a.grid(alpha=0.3)
            if row == 0:
                b.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_triplets(results, path):
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.8 * len(results)), squeeze=False)
    for a, (tid, res) in zip(axes[:, 0], results):
        g = res["grid_idx"]
        a.plot(g, res["X"][:, 0] / res["X"][:, 0].max(), label="IEMG GMax (norm)")
        a.plot(g, res["X"][:, 2] / res["X"][:, 2].max(), label="IEMG VM (norm)")
        b = a.twinx()
        b.plot(g, res["Y"][:, 0], "k", lw=1.5, label="ankle angle (deg)")
        b.plot(g, res["Y"][:, 1] * 10, "k--", lw=1, label="ankle moment ×10 (Nm/kg)")
        a.set_xlim(0, 100)
        a.set_title(f"{tid}: aligned triplet on %stride grid ({len(g)} points, {g[0]}–{g[-1]}%)")
        a.set_xlabel("% stride")
        a.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    b.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def default_trials(walking):
    """Cover each rate config + shortest/longest strides; RX foot so EMG side == kinematic side."""
    rx = [w for w in walking if w["foot"] == "RX"]
    picks = []
    for rate in sorted({w["emg_rate"] for w in rx}):
        picks.append(next(w for w in rx if w["emg_rate"] == rate))
    picks.append(min(rx, key=lambda w: w["n_emg"] / w["emg_rate"]))
    picks.append(max(rx, key=lambda w: w["n_emg"] / w["emg_rate"]))
    seen, out = set(), []
    for w in picks:
        if (w["sub"], w["idx"]) not in seen:
            seen.add((w["sub"], w["idx"]))
            out.append((w["sub"], w["idx"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", nargs="*", help="subject:DataIndex pairs, 0-based index into s.Data")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    s6 = data_io.load_subject(6)
    week1_explore(s6)

    walking, flags, emg_orders, rate_cfgs, emg_side = survey_all()
    print("\n=== Survey ===")
    print("Level-walking trials:", len(walking),
          "" if len(walking) == EXPECTED_WALKING else f"!! FLAG: expected {EXPECTED_WALKING}")
    print("Foot:", collections.Counter(w["foot"] for w in walking))
    print("Rate configs (Kin, Grf, EMG) -> #subjects:", dict(rate_cfgs))
    print("EMG rates per walking trial:", collections.Counter(w["emg_rate"] for w in walking))
    print("Distinct EMG channel orders:", len(emg_orders))
    print("EMGSide != RX:", {k: v for k, v in emg_side.items() if v != "RX"})
    print("Rate flags:", flags or "none")
    stride = np.array([w["stride_s"] for w in walking])
    print(f"Stride duration (s): min {stride.min():.3f} median {np.median(stride):.3f} max {stride.max():.3f}")
    json.dump(dict(walking=walking, flags=flags, rate_cfgs={str(k): v for k, v in rate_cfgs.items()},
                   emg_side=emg_side), open(OUT / "gate1_survey.json", "w"), indent=1)

    # Validate the full pipeline on every walking trial (nothing saved).
    cache = {}
    coverage = []
    for w in walking:
        s = cache.setdefault(w["sub"], data_io.load_subject(w["sub"]))
        res = build_triplet(s, w["idx"])
        assert res["X"].shape == (res["Y"].shape[0], 4) and np.isfinite(res["X"]).all() and np.isfinite(res["Y"]).all()
        coverage.append((res["grid_idx"][0], res["grid_idx"][-1], len(res["grid_idx"])))
    cov = np.array(coverage)
    print(f"\nAll {len(walking)} walking trials aligned OK. First grid %: {cov[:, 0].min()}–{cov[:, 0].max()} "
          f"(median {np.median(cov[:, 0]):.0f}); last grid %: {cov[:, 1].min()}–{cov[:, 1].max()}; "
          f"T: {cov[:, 2].min()}–{cov[:, 2].max()} (median {np.median(cov[:, 2]):.0f})")

    picks = [tuple(map(int, t.split(":"))) for t in args.trials] if args.trials else default_trials(walking)
    sub0, idx0 = picks[0]
    plot_week1(cache[sub0], sub0, idx0, OUT / "week1_raw_vs_stride.png")

    results = []
    print("\n=== Saved triplets ===")
    for sub, idx in picks:
        s = cache.setdefault(sub, data_io.load_subject(sub))
        res = build_triplet(s, idx)
        tid = trial_id(sub, idx)
        np.save(OUT / f"trial_{tid}_features.npy", res["X"])
        np.save(OUT / f"trial_{tid}_targets.npy", res["Y"])
        meta = dict(trial=tid, subject=sub, data_index_0based=idx, foot=str(res["d"].Foot).strip(),
                    emg_rate_hz=res["rate"], n_emg_samples=int(res["emg"].shape[1]),
                    stride_time_s=(res["emg"].shape[1] - 1) / res["rate"],
                    window_ms=WINDOW_MS, hop_ms=HOP_MS, n_windows=int(res["F"].shape[0]),
                    feature_columns=["IEMG_GMax", "WL_GMax", "IEMG_VM", "WL_VM"],
                    target_columns=["ankle_flx_angle_deg", "ankle_flx_moment_Nm_per_kg"],
                    pct_stride_grid=res["grid_idx"].tolist())
        json.dump(meta, open(OUT / f"trial_{tid}_meta.json", "w"), indent=1)
        print(f"{tid} foot={meta['foot']} rate={res['rate']:.0f}Hz N={meta['n_emg_samples']} "
              f"stride={meta['stride_time_s']:.3f}s windows={meta['n_windows']} -> X{res['X'].shape} Y{res['Y'].shape} "
              f"grid {res['grid_idx'][0]}–{res['grid_idx'][-1]}%")
        results.append((tid, res))

    plot_week2(results[:3], OUT / "week2_iemg_wl.png")
    plot_triplets(results, OUT / "gate1_triplets.png")
    print("\nPlots written to", OUT)


if __name__ == "__main__":
    main()
