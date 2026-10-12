"""Speed-stratified train/validation splits (seed 42).

Five global speed bins of equal trial count are computed once over the
844-trial pool and are the single source of truth for every split and for
per-bin reporting. Three pre-declared splits are produced:

  A (primary)   all 844 trials, 675 train / 169 validation, validation quota
                per bin by the largest-remainder method.
  B (ablation)  A restricted to RX-consistent trials (Foot == EMGSide == "RX").
  C (ablation)  subject-wise: 40 training subjects / 10 validation subjects,
                2 validation subjects drawn from each of 5 speed groups.

Split files are written once and never silently overwritten; the files are
byte-identical across runs (no timestamps inside them).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import data_io

SEED = 42
N_BINS = 5
VAL_FRACTION = 0.2
SPLITS_DIR = data_io.PROJECT_ROOT / "outputs" / "week3" / "splits"
BINS_PATH = SPLITS_DIR / "speed_bins.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_speed_bins(meta: list[dict]) -> dict:
    """Five equal-count speed bins over the whole pool; bin 0 is the slowest."""
    speed = np.array([m["speed_mps"] for m in meta], dtype=np.float64)
    n = speed.size
    order = np.lexsort((np.arange(n), speed))          # by speed, ties by trial position
    chunks = np.array_split(order, N_BINS)

    trial_bin, bins = {}, []
    for b, chunk in enumerate(chunks):
        for i in chunk:
            trial_bin[meta[i]["trial_id"]] = b
        bins.append({
            "bin": b,
            "n_trials": int(chunk.size),
            "speed_min": float(speed[chunk].min()),
            "speed_max": float(speed[chunk].max()),
        })
    assert len(trial_bin) == n
    return {"seed": SEED, "n_bins": N_BINS, "bins": bins, "trial_bin": trial_bin}


def largest_remainder_quotas(sizes: list[int], total: int) -> list[int]:
    """Distribute `total` units over bins proportionally; remainder to the largest
    fractional parts, ties resolved towards the lower bin index."""
    pool = sum(sizes)
    exact = [total * s / pool for s in sizes]
    quotas = [int(np.floor(e)) for e in exact]
    remaining = total - sum(quotas)
    frac_order = sorted(range(len(sizes)), key=lambda b: (-(exact[b] - quotas[b]), b))
    for b in frac_order[:remaining]:
        quotas[b] += 1
    assert sum(quotas) == total
    return quotas


def _bin_members(meta, speed_bins, trial_ids=None):
    """Trial ids per bin, in pool order (optionally restricted to `trial_ids`)."""
    keep = None if trial_ids is None else set(trial_ids)
    members = [[] for _ in range(N_BINS)]
    for m in meta:
        tid = m["trial_id"]
        if keep is None or tid in keep:
            members[speed_bins["trial_bin"][tid]].append(tid)
    return members


def _speed_range(meta_by_id, ids):
    sp = np.array([meta_by_id[t]["speed_mps"] for t in ids], dtype=np.float64)
    return [float(sp.min()), float(sp.max())]


def _summary(meta_by_id, speed_bins, train, val) -> dict:
    per_bin = []
    for b in range(N_BINS):
        tr = [t for t in train if speed_bins["trial_bin"][t] == b]
        va = [t for t in val if speed_bins["trial_bin"][t] == b]
        assert tr and va, f"bin {b} has {len(tr)} train / {len(va)} val trials"
        per_bin.append({"bin": b, "n_train": len(tr), "n_val": len(va)})
    return {
        "n_train": len(train),
        "n_val": len(val),
        "per_bin": per_bin,
        "train_speed_range": _speed_range(meta_by_id, train),
        "val_speed_range": _speed_range(meta_by_id, val),
    }


def make_split_a(meta: list[dict], speed_bins: dict) -> dict:
    """Primary run: all 844 trials, 80/20 with per-bin validation quotas."""
    meta_by_id = {m["trial_id"]: m for m in meta}
    n_pool = len(meta)
    n_train = int(np.floor((1.0 - VAL_FRACTION) * n_pool))
    n_val = n_pool - n_train
    members = _bin_members(meta, speed_bins)
    quotas = largest_remainder_quotas([len(m) for m in members], n_val)

    rng = np.random.default_rng(SEED)
    train, val = [], []
    for b in range(N_BINS):
        ids = members[b]
        perm = rng.permutation(len(ids))
        chosen = {ids[i] for i in perm[:quotas[b]]}
        val += sorted(chosen)
        train += sorted(set(ids) - chosen)

    train, val = sorted(train), sorted(val)
    assert len(train) == n_train and len(val) == n_val, (len(train), len(val))
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(meta_by_id)
    for b in range(N_BINS):
        got = sum(1 for t in val if speed_bins["trial_bin"][t] == b)
        assert got == quotas[b], (b, got, quotas[b])

    out = {"name": "A", "run_dir": "A_all844", "seed": SEED,
           "rule": "all 844 trials, per-bin validation quota by largest remainder",
           "quotas": quotas, "train": train, "val": val}
    out["summary"] = _summary(meta_by_id, speed_bins, train, val)
    return out


def make_split_b(meta: list[dict], speed_bins: dict, split_a: dict) -> dict:
    """Ablation: split A restricted to RX-consistent trials."""
    meta_by_id = {m["trial_id"]: m for m in meta}
    rx = {m["trial_id"] for m in meta if m["rx_consistent"]}
    train = sorted(t for t in split_a["train"] if t in rx)
    val = sorted(t for t in split_a["val"] if t in rx)
    assert len(train) + len(val) == len(rx) == 572, (len(train), len(val), len(rx))
    assert not set(train) & set(val)

    out = {"name": "B", "run_dir": "B_rx572", "seed": SEED,
           "rule": "split A intersected with rx_consistent (Foot == EMGSide == RX)",
           "train": train, "val": val}
    out["summary"] = _summary(meta_by_id, speed_bins, train, val)
    return out


def make_split_c(meta: list[dict], speed_bins: dict) -> dict:
    """Ablation: subject-wise, 2 validation subjects per speed group of 10."""
    meta_by_id = {m["trial_id"]: m for m in meta}
    by_subject: dict[int, list[dict]] = {}
    for m in meta:
        by_subject.setdefault(m["subject"], []).append(m)

    med = {sub: float(np.median([m["speed_mps"] for m in ms])) for sub, ms in by_subject.items()}
    ordered = sorted(by_subject, key=lambda sub: (med[sub], sub))
    groups = np.array_split(np.array(ordered), N_BINS)

    rng = np.random.default_rng(SEED)
    val_subjects: list[int] = []
    for group in groups:
        perm = rng.permutation(len(group))
        val_subjects += [int(group[i]) for i in perm[:2]]
    val_subjects = sorted(val_subjects)
    train_subjects = sorted(set(by_subject) - set(val_subjects))
    assert len(val_subjects) == 10 and len(train_subjects) == 40
    assert not set(val_subjects) & set(train_subjects)
    assert set(val_subjects) | set(train_subjects) == set(by_subject) == set(range(1, 51))

    train = sorted(m["trial_id"] for m in meta if m["subject"] in set(train_subjects))
    val = sorted(m["trial_id"] for m in meta if m["subject"] in set(val_subjects))
    assert len(train) + len(val) == len(meta)

    out = {"name": "C", "run_dir": "C_subjectwise", "seed": SEED,
           "rule": "subject-wise: subjects ordered by median speed, 2 validation subjects per group of 10",
           "train_subjects": train_subjects, "val_subjects": val_subjects,
           "subject_median_speed": {str(k): med[k] for k in sorted(med)},
           "train": train, "val": val}
    out["summary"] = _summary(meta_by_id, speed_bins, train, val)
    return out


def split_path(name: str) -> Path:
    return SPLITS_DIR / f"split_{name}.json"


def write_json(path: Path, payload: dict, force: bool = False):
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to regenerate")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))


def load_split(name: str) -> dict:
    path = split_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/make_splits.py first")
    return json.loads(path.read_text())
