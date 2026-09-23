"""Put EMG-derived features on the 101-point %stride grid used by .Ang/.Mom.

In this dataset each trial's .EMG is already cropped to one stride
(heel strike -> next heel strike of the trial's .Foot), so sample k of an
N-sample EMG maps to 100 * k / (N - 1) % stride. Targets are never resampled.
"""
from __future__ import annotations

import numpy as np

STRIDE_GRID = np.arange(101, dtype=np.float64)  # 0..100 % stride, matches .Ang/.Mom columns


def sample_to_pct_stride(sample_idx: np.ndarray, n_samples: int) -> np.ndarray:
    return 100.0 * np.asarray(sample_idx, dtype=np.float64) / (n_samples - 1)


def resample_to_stride_grid(features: np.ndarray, feature_pct: np.ndarray,
                            grid: np.ndarray = STRIDE_GRID):
    """Linearly interpolate a (T, F) feature sequence onto `grid` without extrapolating.

    Returns (features_on_grid (G, F), grid_idx (G,)): only grid points inside
    [feature_pct[0], feature_pct[-1]] are kept; grid_idx indexes the .Ang/.Mom columns.
    """
    features = np.asarray(features, dtype=np.float64)
    feature_pct = np.asarray(feature_pct, dtype=np.float64)
    assert features.ndim == 2 and features.shape[0] == feature_pct.size
    assert np.all(np.diff(feature_pct) > 0), "feature timestamps must be strictly increasing"

    tol = 1e-9
    keep = (grid >= feature_pct[0] - tol) & (grid <= feature_pct[-1] + tol)
    grid_idx = np.flatnonzero(keep)
    out = np.column_stack([np.interp(grid[grid_idx], feature_pct, features[:, j])
                           for j in range(features.shape[1])])
    assert not np.isnan(out).any()
    return out, grid_idx


def align_trial(features: np.ndarray, window_end_idx: np.ndarray, n_emg_samples: int,
                targets: np.ndarray):
    """Align a window-level feature sequence with stride-normalised targets.

    targets: (K, 101) rows such as ankle angle and moment.
    Returns X (G, F), Y (G, K), grid_idx (G,).
    """
    pct = sample_to_pct_stride(window_end_idx, n_emg_samples)
    X, grid_idx = resample_to_stride_grid(features, pct)
    targets = np.atleast_2d(targets)
    assert targets.shape[1] == STRIDE_GRID.size, targets.shape
    Y = targets[:, grid_idx].T
    assert X.shape[0] == Y.shape[0]
    return X, Y, grid_idx
