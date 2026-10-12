"""Build the full-stride walking dataset (one row per level-walking trial).

Every trial contributes a (101, 4) feature matrix on the 0-100 % stride grid
(GMax and VM, IEMG and WL) and two (101,) targets taken verbatim from the
stored .Ang / .Mom rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import data_io
from .fullstride import full_stride_features

WEEK3_DIR = data_io.PROJECT_ROOT / "outputs" / "week3"
DATA_OUT_DIR = WEEK3_DIR / "data"
NPZ_PATH = DATA_OUT_DIR / "dataset_walking844.npz"
META_PATH = DATA_OUT_DIR / "dataset_meta.json"

EMG_MUSCLES = ("Gluteus Maximus", "Vastus Medialis")
FEATURE_COLUMNS = ("IEMG_GMax", "WL_GMax", "IEMG_VM", "WL_VM")
ANGLE_NAME = "AnkleFlx"
MOMENT_NAME = "AnkleFlxMom"
WINDOW_MS = 150.0
HOP_MS = 25.0
N_SUBJECTS = 50


def trial_id(subject: int, data_idx: int) -> str:
    """Gate 1 trial id, e.g. S02_D001."""
    return f"S{subject:02d}_D{data_idx:03d}"


def build_walking_dataset(data_dir: Path = data_io.DATA_DIR, save: bool = True):
    """Walk subjects 1..50 in order and build X, Y_pos, Y_mom and per-trial meta."""
    xs, y_pos, y_mom, meta = [], [], [], []

    for subject in range(1, N_SUBJECTS + 1):
        path = Path(data_dir) / f"Subject{subject}.mat"
        if not path.exists():
            raise FileNotFoundError(f"missing data file: {path}")
        s = data_io.load_subject(subject, data_dir=Path(data_dir))
        emg_side = str(s.EMGSide).strip()
        ch_emg = [data_io.channel_index(s, "EmgVarName", name) for name in EMG_MUSCLES]
        i_ang = data_io.channel_index(s, "AngVarName", ANGLE_NAME)
        i_mom = data_io.channel_index(s, "MomVarName", MOMENT_NAME)

        for data_idx in data_io.walking_indices(s):
            d = data_io.trials(s)[data_idx]
            if "speed" not in d._fieldnames:
                raise KeyError(f"no 'speed' field in trial {trial_id(subject, data_idx)}: {d._fieldnames}")
            rates = data_io.derive_rates(s, d)
            rate = rates.emg
            emg = np.asarray(d.EMG, dtype=np.float64)[ch_emg]       # (2, N) = [GMax, VM]

            x101, info = full_stride_features(emg, rate, WINDOW_MS, HOP_MS)

            ang_row = np.asarray(d.Ang, dtype=np.float64)[i_ang]
            mom_row = np.asarray(d.Mom, dtype=np.float64)[i_mom]
            if ang_row.shape != (101,) or mom_row.shape != (101,):
                raise ValueError(f"unexpected target shape in {trial_id(subject, data_idx)}")

            foot = str(d.Foot).strip()
            speed = float(d.speed)
            if not np.isfinite(speed):
                raise ValueError(f"non-finite speed in {trial_id(subject, data_idx)}")

            xs.append(x101)
            y_pos.append(ang_row)
            y_mom.append(mom_row)
            meta.append({
                "trial_id": trial_id(subject, data_idx),
                "subject": subject,
                "data_idx": data_idx,
                "foot": foot,
                "emg_side": emg_side,
                "rx_consistent": bool(foot == "RX" and emg_side == "RX"),
                "speed_mps": speed,
                "rate_hz": float(rate),
                "n_emg": int(info["n_samples"]),
                "stride_time_s": float((info["n_samples"] - 1) / rate),
                "window_samples": info["window_samples"],
                "hop_samples": info["hop_samples"],
                "n_windows": info["n_windows"],
                "extra_end_window": info["extra_end_window"],
                "rate_flags": list(rates.flags),
                "seam_ratio": [float(v) for v in info["seam_ratio"]],
            })

    X = np.stack(xs).astype(np.float64)
    Y_pos = np.stack(y_pos).astype(np.float64)
    Y_mom = np.stack(y_mom).astype(np.float64)
    assert X.shape[1:] == (101, 4) and Y_pos.shape[1:] == (101,) and Y_mom.shape[1:] == (101,)
    assert X.shape[0] == Y_pos.shape[0] == Y_mom.shape[0] == len(meta)

    if save:
        DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(NPZ_PATH, X=X, Y_pos=Y_pos, Y_mom=Y_mom,
                            trial_ids=np.array([m["trial_id"] for m in meta]))
        META_PATH.write_text(json.dumps({
            "feature_columns": list(FEATURE_COLUMNS),
            "emg_muscles": list(EMG_MUSCLES),
            "angle_name": ANGLE_NAME,
            "moment_name": MOMENT_NAME,
            "window_ms": WINDOW_MS,
            "hop_ms": HOP_MS,
            "n_trials": len(meta),
            "trials": meta,
        }, indent=1))
        _verify_saved_targets(data_dir, Y_pos, Y_mom, meta)

    return X, Y_pos, Y_mom, meta


def _verify_saved_targets(data_dir, Y_pos, Y_mom, meta):
    """Reload the npz and assert every target row is bit-identical to the .mat row."""
    saved = np.load(NPZ_PATH, allow_pickle=False)
    assert np.array_equal(saved["Y_pos"], Y_pos) and np.array_equal(saved["Y_mom"], Y_mom)
    assert np.array_equal(saved["X"], np.load(NPZ_PATH)["X"])

    by_subject = {}
    for i, m in enumerate(meta):
        by_subject.setdefault(m["subject"], []).append((i, m["data_idx"]))
    for subject, items in by_subject.items():
        s = data_io.load_subject(subject, data_dir=Path(data_dir))
        i_ang = data_io.channel_index(s, "AngVarName", ANGLE_NAME)
        i_mom = data_io.channel_index(s, "MomVarName", MOMENT_NAME)
        for i, data_idx in items:
            d = data_io.trials(s)[data_idx]
            assert np.array_equal(saved["Y_pos"][i], np.asarray(d.Ang, dtype=np.float64)[i_ang])
            assert np.array_equal(saved["Y_mom"][i], np.asarray(d.Mom, dtype=np.float64)[i_mom])


def load_walking_dataset():
    """Load the saved dataset: (X, Y_pos, Y_mom, meta list)."""
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"{NPZ_PATH} not found; run scripts/build_week3_dataset.py first")
    arrs = np.load(NPZ_PATH, allow_pickle=False)
    meta = json.loads(META_PATH.read_text())["trials"]
    X, Y_pos, Y_mom = arrs["X"], arrs["Y_pos"], arrs["Y_mom"]
    assert len(meta) == X.shape[0]
    assert [m["trial_id"] for m in meta] == list(arrs["trial_ids"])
    return X, Y_pos, Y_mom, meta
