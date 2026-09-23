"""Tool-agnostic sEMG window features. Pure NumPy in, NumPy out."""
from __future__ import annotations

import numpy as np


def window_layout(n_samples: int, rate: float, window_ms: float = 150.0, hop_ms: float = 25.0):
    """Window length/hop in samples and the start index of every full window.

    Window and hop are converted from ms per call, so each trial uses its own rate.
    Raises if either is not (close to) an integer number of samples, so time
    resolution is identical across the 800/960/1000 Hz recordings.
    """
    w_exact, h_exact = window_ms * rate / 1000.0, hop_ms * rate / 1000.0
    win, hop = int(round(w_exact)), int(round(h_exact))
    if abs(w_exact - win) > 1e-6 or abs(h_exact - hop) > 1e-6:
        raise ValueError(f"window {w_exact} / hop {h_exact} samples are not integers at {rate} Hz")
    if n_samples < win:
        raise ValueError(f"signal ({n_samples} samples) shorter than one window ({win})")
    starts = np.arange(0, n_samples - win + 1, hop)
    return win, hop, starts


def window_end_indices(n_samples: int, rate: float, window_ms: float = 150.0, hop_ms: float = 25.0) -> np.ndarray:
    """Sample index of the last sample in each window (the causal timestamp of its feature)."""
    win, _, starts = window_layout(n_samples, rate, window_ms, hop_ms)
    return starts + win - 1


def iemg(windows: np.ndarray) -> np.ndarray:
    """IEMG = sum |x_n| over the last axis."""
    return np.abs(windows).sum(axis=-1)


def waveform_length(windows: np.ndarray) -> np.ndarray:
    """WL = sum |x_{n+1} - x_n| over the last axis."""
    return np.abs(np.diff(windows, axis=-1)).sum(axis=-1)


def extract_features(emg_channels: np.ndarray, rate: float, window_ms: float = 150.0,
                     hop_ms: float = 25.0) -> np.ndarray:
    """Sliding-window IEMG and WL per channel.

    emg_channels: (C, N) raw EMG, one row per channel.
    Returns feature_sequence of shape (T, 2*C) ordered
    [IEMG_ch0, WL_ch0, IEMG_ch1, WL_ch1, ...], one row per window.
    Row t describes samples [starts[t], starts[t] + win); see window_end_indices().
    """
    x = np.asarray(emg_channels, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected (C, N) array, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input EMG contains NaN/inf")

    n_ch, n = x.shape
    win, hop, starts = window_layout(n, rate, window_ms, hop_ms)
    windows = np.lib.stride_tricks.sliding_window_view(x, win, axis=1)[:, starts, :]  # (C, T, win)

    feats = np.stack([iemg(windows), waveform_length(windows)], axis=-1)  # (C, T, 2)
    feature_sequence = feats.transpose(1, 0, 2).reshape(len(starts), 2 * n_ch)

    expected_t = (n - win) // hop + 1
    assert feature_sequence.shape == (expected_t, 2 * n_ch), (feature_sequence.shape, expected_t)
    assert not np.isnan(feature_sequence).any(), "NaN in feature sequence"
    return feature_sequence
