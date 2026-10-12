"""Baseline per-timestep LSTM regressors for ankle position and moment.

Two SEPARATE networks are used: one for the ankle angle, one for the ankle
moment. They never share weights, an optimizer, a scaler or a loss; each is
trained in its own run directory.
"""
from __future__ import annotations

import torch
from torch import nn

INPUT_SIZE = 4           # IEMG_GMax, WL_GMax, IEMG_VM, WL_VM
POSITION_HIDDEN = 200
MOMENT_HIDDEN = 50


class AnkleLSTM(nn.Module):
    """Single-layer unidirectional LSTM with a per-timestep linear read-out.

    Input (B, 101, 4) -> output (B, 101). The initial hidden and cell states
    are zero and no dropout is used.
    """

    def __init__(self, hidden_size: int, input_size: int = INPUT_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.lstm.input_size:
            raise ValueError(f"expected (B, T, {self.lstm.input_size}), got {tuple(x.shape)}")
        seq, _ = self.lstm(x)               # zero initial state
        return self.head(seq).squeeze(-1)   # (B, T)


def build_position_model() -> AnkleLSTM:
    return AnkleLSTM(POSITION_HIDDEN)


def build_moment_model() -> AnkleLSTM:
    return AnkleLSTM(MOMENT_HIDDEN)


BUILDERS = {"position": build_position_model, "moment": build_moment_model}
HIDDEN_SIZES = {"position": POSITION_HIDDEN, "moment": MOMENT_HIDDEN}
TARGET_UNITS = {"position": "deg", "moment": "Nm/kg"}
