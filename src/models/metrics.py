"""Validation metrics: per-trial Pearson r and RMSE in physical units."""
from __future__ import annotations

import numpy as np


def per_trial_metrics(pred: np.ndarray, true: np.ndarray):
    """Pearson r and RMSE per trial over the 101 stride points (de-normalized input)."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    assert pred.shape == true.shape and pred.ndim == 2, (pred.shape, true.shape)

    const = np.std(true, axis=1) == 0
    if const.any():
        raise ValueError(f"constant target in {int(const.sum())} validation trial(s); "
                         "Pearson r is undefined")

    r = np.array([np.corrcoef(p, t)[0, 1] for p, t in zip(pred, true)], dtype=np.float64)
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    if not np.isfinite(r).all():
        raise ValueError("non-finite Pearson r in validation metrics")
    return r, rmse


def summarize(r: np.ndarray, rmse: np.ndarray) -> dict:
    """Mean +- sample SD (ddof=1) across trials."""
    assert r.size == rmse.size and r.size > 1
    return {
        "n": int(r.size),
        "r_mean": float(np.mean(r)), "r_sd": float(np.std(r, ddof=1)),
        "rmse_mean": float(np.mean(rmse)), "rmse_sd": float(np.std(rmse, ddof=1)),
    }


def pooled_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Secondary metric: one Pearson r over all validation points pooled."""
    return float(np.corrcoef(np.asarray(pred).ravel(), np.asarray(true).ravel())[0, 1])


def group_metrics(r: np.ndarray, rmse: np.ndarray, mask: np.ndarray) -> dict:
    """Metrics restricted to a boolean subset of the validation trials."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    out = {"n": n, "r_mean": float(np.mean(r[mask])), "rmse_mean": float(np.mean(rmse[mask]))}
    if n > 1:
        out["r_sd"] = float(np.std(r[mask], ddof=1))
        out["rmse_sd"] = float(np.std(rmse[mask], ddof=1))
    return out
