"""Training loop for the baseline ankle LSTMs.

Fixed hyper-parameters (Keleş 2023 replication): Adam lr 1e-3, MSE over all
time steps, batch size 32, 200 epochs, no scheduler, no early stopping, no
gradient clipping, no weight decay. The FINAL-epoch model is the one that is
evaluated and saved; the best-validation epoch is recorded for information only.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from ..data import data_io, dataset, splits  # noqa: E402
from . import lstm_baseline, metrics  # noqa: E402

SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 200
STD_FLOOR = 1e-8
RUNS_DIR = data_io.PROJECT_ROOT / "outputs" / "week3" / "runs"
MANIFEST_PATH = data_io.PROJECT_ROOT / "outputs" / "week3" / "MANIFEST.json"


# --------------------------------------------------------------------------- scaling
@dataclass
class Scaler:
    """z-score statistics computed on TRAIN trials only."""
    x_mean: list[float]
    x_std: list[float]
    y_mean: float
    y_std: float

    def transform_x(self, X):
        return (X - np.asarray(self.x_mean)) / np.asarray(self.x_std)

    def transform_y(self, Y):
        return (Y - self.y_mean) / self.y_std

    def inverse_y(self, Y):
        return Y * self.y_std + self.y_mean


def fit_scaler(X_train: np.ndarray, Y_train: np.ndarray) -> Scaler:
    """Per-feature-column stats over train trials x 101 steps; scalar stats for the target."""
    assert X_train.ndim == 3 and Y_train.ndim == 2
    flat = X_train.reshape(-1, X_train.shape[-1])
    x_mean = flat.mean(axis=0)
    x_std = np.maximum(flat.std(axis=0), STD_FLOOR)
    y_mean = float(Y_train.mean())
    y_std = float(max(Y_train.std(), STD_FLOOR))
    return Scaler([float(v) for v in x_mean], [float(v) for v in x_std], y_mean, y_std)


# --------------------------------------------------------------------------- bookkeeping
def set_seeds():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def git_info() -> dict:
    root = data_io.PROJECT_ROOT
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    if head.returncode != 0 or status.returncode != 0:
        raise RuntimeError(f"git failed: {head.stderr}{status.stderr}")
    return {"git_head": head.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip())}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_manifest(entry: dict):
    """Read-modify-write so earlier entries are never lost."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        payload = json.loads(MANIFEST_PATH.read_text())
    else:
        payload = {"runs": []}
    payload["runs"] = [r for r in payload["runs"]
                       if not (r["run"] == entry["run"] and r["target"] == entry["target"])]
    payload["runs"].append(entry)
    payload["runs"].sort(key=lambda r: (r["run"], r["target"]))
    MANIFEST_PATH.write_text(json.dumps(payload, indent=1))


# --------------------------------------------------------------------------- figures
def plot_curves(pred: np.ndarray, true: np.ndarray, target: str, path: Path, title: str):
    grid = np.arange(101)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arr, color, label in ((true, "0.45", "measured"), (pred, "tab:red", "predicted")):
        mean, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        ax.plot(grid, mean, color=color, lw=1.8, label=f"{label} (mean)")
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.25, lw=0,
                        label=f"{label} (±SD)")
    ax.set(xlabel="% stride", xlim=(0, 100),
           ylabel=f"ankle {target} ({lstm_baseline.TARGET_UNITS[target]})", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss(epochs, train_loss, val_loss, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, train_loss, label="train", lw=1.3)
    ax.plot(epochs, val_loss, label="validation", lw=1.3)
    ax.set(xlabel="epoch", ylabel="MSE (normalized units)", yscale="log", title=title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- training
def _subset(X, Y, meta_index, trial_ids):
    idx = np.array([meta_index[t] for t in trial_ids], dtype=np.int64)
    return X[idx], Y[idx], idx


def train_target(run_name: str, target: str, split_payload: dict, split_file: Path,
                 X: np.ndarray, Y: np.ndarray, meta: list[dict], speed_bins: dict,
                 epochs: int = EPOCHS, force: bool = False, run_dir_name: str | None = None,
                 train_ids: list[str] | None = None, val_ids: list[str] | None = None,
                 extra_config: dict | None = None) -> dict:
    """Train one target of one run; returns the metrics dict."""
    if target not in lstm_baseline.BUILDERS:
        raise ValueError(f"unknown target {target!r}")

    run_dir = RUNS_DIR / (run_dir_name or split_payload["run_dir"])
    final_dir = run_dir / target
    partial_dir = run_dir / f"{target}.partial"
    if final_dir.exists():
        if not force:
            raise FileExistsError(f"{final_dir} already exists; pass --force to redo this run")
        shutil.rmtree(final_dir)
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    meta_index = {m["trial_id"]: i for i, m in enumerate(meta)}
    train_ids = list(train_ids if train_ids is not None else split_payload["train"])
    val_ids = list(val_ids if val_ids is not None else split_payload["val"])
    X_tr, Y_tr, _ = _subset(X, Y, meta_index, train_ids)
    X_va, Y_va, val_idx = _subset(X, Y, meta_index, val_ids)

    scaler = fit_scaler(X_tr, Y_tr)
    # Statistics come from train trials only; the same transform is applied to both sets.
    Xtr_n = torch.tensor(scaler.transform_x(X_tr), dtype=torch.float32)
    Ytr_n = torch.tensor(scaler.transform_y(Y_tr), dtype=torch.float32)
    Xva_n = torch.tensor(scaler.transform_x(X_va), dtype=torch.float32)
    Yva_n = torch.tensor(scaler.transform_y(Y_va), dtype=torch.float32)

    set_seeds()
    device = pick_device()
    model = lstm_baseline.BUILDERS[target]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(SEED)

    Xtr_d, Ytr_d = Xtr_n.to(device), Ytr_n.to(device)
    Xva_d, Yva_d = Xva_n.to(device), Yva_n.to(device)
    n_train = Xtr_d.shape[0]

    started = dt.datetime.now().astimezone()
    t0 = time.time()
    log_rows, train_hist, val_hist = [], [], []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        total, counted = 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            xb, yb = Xtr_d[batch], Ytr_d[batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.numel()
            counted += batch.numel()
        train_loss = total / counted

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_d), Yva_d).item()
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"    epoch {epoch:4d}/{epochs}  train {train_loss:.5f}  val {val_loss:.5f}")

    wall = time.time() - t0
    ended = dt.datetime.now().astimezone()

    # Evaluate the FINAL-epoch model.
    model.eval()
    with torch.no_grad():
        pred_n = model(Xva_d).cpu().numpy().astype(np.float64)
    pred = scaler.inverse_y(pred_n)
    true = Y_va
    r, rmse = metrics.per_trial_metrics(pred, true)
    summary = metrics.summarize(r, rmse)
    summary["pooled_r"] = metrics.pooled_r(pred, true)

    val_meta = [meta[i] for i in val_idx]
    by_bin = {}
    for b in range(splits.N_BINS):
        mask = np.array([speed_bins["trial_bin"][m["trial_id"]] == b for m in val_meta])
        if mask.any():
            by_bin[str(b)] = metrics.group_metrics(r, rmse, mask)
    result = {
        "run": run_name, "target": target, "units": lstm_baseline.TARGET_UNITS[target],
        "n_train": len(train_ids), "n_val": len(val_ids),
        "final_epoch": epochs,
        "final_train_loss": train_hist[-1], "final_val_loss": val_hist[-1],
        "initial_train_loss": train_hist[0],
        "best_val_epoch_info_only": int(np.argmin(val_hist) + 1),
        "best_val_loss_info_only": float(np.min(val_hist)),
        "validation": summary,
        "by_speed_bin": by_bin,
    }
    if run_name in ("A", "C"):
        rx_mask = np.array([m["rx_consistent"] for m in val_meta])
        lx_mask = np.array([m["foot"] == "LX" for m in val_meta])
        result["by_side"] = {"rx_consistent": metrics.group_metrics(r, rmse, rx_mask),
                             "foot_LX": metrics.group_metrics(r, rmse, lx_mask)}
    result["per_trial"] = [{"trial_id": m["trial_id"], "r": float(ri), "rmse": float(ei)}
                           for m, ri, ei in zip(val_meta, r, rmse)]

    config = {
        "run": run_name, "run_dir": str(run_dir.relative_to(data_io.PROJECT_ROOT)),
        "target": target, "hidden_size": lstm_baseline.HIDDEN_SIZES[target],
        "input_size": lstm_baseline.INPUT_SIZE, "num_layers": 1, "bidirectional": False,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "loss": "MSELoss(mean)",
        "batch_size": BATCH_SIZE, "epochs": epochs, "scheduler": None, "early_stopping": False,
        "weight_decay": 0.0, "dropout": 0.0, "grad_clipping": None,
        "model_selected": "final epoch",
        "seed": SEED, "device": device,
        "split_file": str(Path(split_file).relative_to(data_io.PROJECT_ROOT)),
        "split_sha256": sha256_file(split_file),
        "dataset_npz_sha256": sha256_file(dataset.NPZ_PATH),
        "n_train": len(train_ids), "n_val": len(val_ids),
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "started": started.isoformat(timespec="seconds"), "ended": ended.isoformat(timespec="seconds"),
        "wall_time_s": round(wall, 2),
        **git_info(),
    }
    if extra_config:
        config.update(extra_config)

    (partial_dir / "config.json").write_text(json.dumps(config, indent=1))
    (partial_dir / "scaler.json").write_text(json.dumps(asdict(scaler), indent=1))
    (partial_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    with open(partial_dir / "loss_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)
    torch.save(model.state_dict(), partial_dir / "model_final.pt")
    unit = lstm_baseline.TARGET_UNITS[target]
    plot_curves(pred, true, target, partial_dir / "curves.png",
                f"run {run_name} — ankle {target}: validation (n={len(val_ids)}), "
                f"r {summary['r_mean']:.3f}±{summary['r_sd']:.3f}, "
                f"RMSE {summary['rmse_mean']:.3f}±{summary['rmse_sd']:.3f} {unit}")
    plot_loss([row["epoch"] for row in log_rows], train_hist, val_hist,
              partial_dir / "loss.png", f"run {run_name} — ankle {target}: loss")

    partial_dir.rename(final_dir)

    append_manifest({
        "run": run_name, "target": target,
        "directory": str(final_dir.relative_to(data_io.PROJECT_ROOT)),
        "finished": ended.isoformat(timespec="seconds"),
        "git_head": config["git_head"], "working_tree_dirty": config["working_tree_dirty"],
        "seed": SEED, "device": device, "epochs": epochs,
        "n_train": len(train_ids), "n_val": len(val_ids),
        "r_mean": summary["r_mean"], "r_sd": summary["r_sd"],
        "rmse_mean": summary["rmse_mean"], "rmse_sd": summary["rmse_sd"],
        "wall_time_s": round(wall, 2),
    })
    print(f"    -> r {summary['r_mean']:.4f} ± {summary['r_sd']:.4f} | "
          f"RMSE {summary['rmse_mean']:.4f} ± {summary['rmse_sd']:.4f} {unit} | {wall:.1f} s")
    return result
