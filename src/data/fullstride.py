"""Full-stride (0-100 % of the gait cycle) window features.

Gate 1 kept only windows that fit entirely inside the stored stride, so the
first 6-22 % of every stride had no causal window and was dropped. Here the
signal is pre-padded with its own tail before windowing, which yields a feature
value for every one of the 101 stride-grid points.

The padding is periodic (wrap-around): the trial's EMG is cropped to one gait
cycle, sample N-1 being the same gait event as sample 0 one cycle later, so the
samples preceding sample 0 are taken from the end of the same stride. Zero
padding was rejected because it would artificially lower IEMG over the
0-20 % loading-response phase, exactly where vastus medialis is active, and
would therefore bias the earliest part of every stride towards silence.

The assumption behind wrap padding is that consecutive strides are similar
(steady-state gait); `seam_ratio` in the returned info dict quantifies the
discontinuity at the splice for each channel.
"""
from __future__ import annotations

import numpy as np

from . import align, features


def full_stride_features(emg_2ch: np.ndarray, rate: float, window_ms: float = 150.0,
                         hop_ms: float = 25.0):
    """Causal IEMG/WL features on the full 0-100 % stride grid.

    emg_2ch: (C, N) raw EMG cropped to one stride, float64.
    Returns (X101 (101, 2C) float64, info dict).
    Feature columns follow features.extract_features: [IEMG_c0, WL_c0, IEMG_c1, WL_c1, ...].
    Each feature is stamped at the LAST sample of its window (causal), and the
    window ending at sample k covers samples k-W+1 .. k, wrapping into the end
    of the stride when k-W+1 < 0.
    """
    x = np.asarray(emg_2ch, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (C, N) array, got shape {x.shape}")
    n = x.shape[1]

    win, hop, _ = features.window_layout(n, rate, window_ms, hop_ms)
    if n <= win:
        raise ValueError(f"stride ({n} samples) must be longer than one window ({win})")

    # Periodic pre-padding: the W-1 samples before the final duplicate sample.
    pre = x[:, n - win:n - 1]
    assert pre.shape[1] == win - 1, (pre.shape, win)
    x_ext = np.concatenate([pre, x], axis=1)
    assert x_ext.shape[1] == n + win - 1

    # Original index k lives at x_ext index k+W-1, so the window starting at
    # x_ext index s ends at original index s.
    feats = features.extract_features(x_ext, rate, window_ms, hop_ms)
    ends = np.arange(feats.shape[0], dtype=np.int64) * hop
    assert ends[0] == 0 and ends[-1] <= n - 1, (ends[0], ends[-1], n)
    assert feats.shape[0] == (n - 1) // hop + 1, (feats.shape, n, hop)

    extra_end_window = (n - 1) % hop != 0
    if extra_end_window:
        tail = features.extract_features(x_ext[:, n - 1:n - 1 + win], rate, window_ms, hop_ms)
        assert tail.shape[0] == 1, tail.shape
        feats = np.vstack([feats, tail])
        ends = np.append(ends, n - 1)

    pct = align.sample_to_pct_stride(ends, n)
    assert pct[0] == 0.0 and pct[-1] == 100.0, (pct[0], pct[-1])

    x101, grid_idx = align.resample_to_stride_grid(feats, pct)
    assert np.array_equal(grid_idx, np.arange(101)), grid_idx
    assert x101.shape == (101, feats.shape[1]), x101.shape
    assert np.isfinite(x101).all(), "non-finite value in full-stride features"

    step = np.median(np.abs(np.diff(x, axis=1)), axis=1)
    seam_ratio = np.abs(x[:, n - 1] - x[:, 0]) / step

    info = {
        "n_samples": int(n),
        "window_samples": int(win),
        "hop_samples": int(hop),
        "n_windows": int(feats.shape[0]),
        "extra_end_window": bool(extra_end_window),
        "features": feats,            # pre-interpolation, one row per window
        "window_end_idx": ends,       # original sample index each row ends at
        "seam_ratio": seam_ratio,     # per channel, informational
    }
    return x101, info
