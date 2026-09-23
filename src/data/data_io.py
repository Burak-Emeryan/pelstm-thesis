"""Loading and inspection of the Lencioni et al. (2019) .mat files.

One file per subject (data/SubjectK.mat), each holding a struct `s`.
Trials live in `s.Data`; channel/variable names live in `s.*VarName`.
Nominal sampling rates are stored per subject (`s.KinFreq`, `s.GrfFreq`,
`s.EMGFreq`) and are validated per trial against the sample counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Relative disagreement between derived and nominal EMG rate that we flag.
RATE_TOL = 0.01


def load_subject(subject: int | str | Path, data_dir: Path = DATA_DIR):
    """Load struct `s` for one subject (int -> data/Subject<int>.mat)."""
    path = Path(subject) if isinstance(subject, (str, Path)) else data_dir / f"Subject{subject}.mat"
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)["s"]


def var_names(s, field: str) -> list[str]:
    """Stripped variable names, e.g. var_names(s, 'EmgVarName')."""
    return [str(n).strip() for n in np.atleast_1d(getattr(s, field))]


def channel_index(s, field: str, name: str) -> int:
    """Index of `name` in `s.<field>` (exact match after stripping). Raises if absent/ambiguous."""
    names = var_names(s, field)
    hits = [i for i, n in enumerate(names) if n.lower() == name.lower()]
    if len(hits) != 1:
        raise KeyError(f"{name!r} found {len(hits)} times in {field}: {names}")
    return hits[0]


def trials(s) -> np.ndarray:
    return np.atleast_1d(s.Data)


def walking_indices(s, task: str = "Walking") -> list[int]:
    """Indices into s.Data whose .Task equals `task` exactly (so ToeWalking/HeelWalking are excluded)."""
    return [i for i, d in enumerate(trials(s)) if str(d.Task).strip() == task]


@dataclass
class TrialRates:
    emg_nominal: float          # s.EMGFreq
    kin_nominal: float          # s.KinFreq
    grf_nominal: float          # s.GrfFreq
    emg_from_marker: float      # KinFreq * (N_emg-1)/(N_marker-1): both arrays span the same stride
    emg_from_cadence: float     # (N_emg-1) / stride_time, stride_time = 120/cadence
    stride_time_s: float        # (N_emg-1) / emg rate used
    flags: list[str]

    @property
    def emg(self) -> float:
        """Rate used downstream: the nominal header value, after per-trial validation."""
        return self.emg_nominal


def derive_rates(s, d) -> TrialRates:
    """Derive and cross-check the EMG sampling rate of a single trial.

    The TimeStamp* fields are scalars (onset of the stride in the original
    acquisition), not time vectors, so the rate is derived from sample counts:
    .EMG and .Marker are both cropped to the same heel-strike-to-heel-strike
    stride, and `cadence` gives an independent stride duration.
    """
    n_emg = np.shape(d.EMG)[1]
    n_mk = np.shape(d.Marker)[1]
    emg_nom, kin_nom, grf_nom = float(s.EMGFreq), float(s.KinFreq), float(s.GrfFreq)
    from_marker = kin_nom * (n_emg - 1) / (n_mk - 1)
    from_cadence = (n_emg - 1) * float(d.cadence) / 120.0

    flags = []
    # Marker-based estimate is coarse at 60 Hz kinematics (1 frame = ~1.6% of a stride).
    mk_tol = max(RATE_TOL, 1.5 / (n_mk - 1))
    if abs(from_marker / emg_nom - 1) > mk_tol:
        flags.append(f"marker-derived EMG rate {from_marker:.1f} vs nominal {emg_nom:.0f}")
    if abs(from_cadence / emg_nom - 1) > RATE_TOL:
        flags.append(f"cadence-derived EMG rate {from_cadence:.1f} vs nominal {emg_nom:.0f}")
    return TrialRates(emg_nom, kin_nom, grf_nom, from_marker, from_cadence,
                      (n_emg - 1) / emg_nom, flags)


def describe_trial(s, d) -> dict:
    """Shapes and metadata of one trial, for logging."""
    return {
        "Task": str(d.Task).strip(),
        "Foot": str(d.Foot).strip(),
        "EMG": np.shape(d.EMG),
        "Ang": np.shape(d.Ang),
        "Mom": np.shape(d.Mom),
        "Grf": np.shape(d.Grf),
        "Marker": np.shape(d.Marker),
        "TimeStampKin": float(d.TimeStampKin),
        "TimeStampGrf": float(d.TimeStampGrf),
        "TimeStampEmg": float(d.TimeStampEmg),
        "cadence": float(d.cadence),
    }
