"""Train LSTM / BiLSTM / Transformer with an identical protocol.

Protocol (same for all models): MSELoss on standardised target, Adam,
early stopping on validation loss (best weights restored), fixed seed.

The training DataLoader shuffles the ORDER of training windows between
mini-batches (standard SGD practice). Each window keeps its internal
chronological order, splits are strictly chronological, and validation/test
windows are never shuffled - so no future information reaches training.

Usage:
    python -m src.training.train --model all
    python -m src.training.train --model lstm --epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import MODEL_NAMES, build_model
from src.utils import exp_paths, get_logger, load_config, set_seed

log = get_logger("train", "train.log")


def load_split(name: str, cfg: dict | None = None) -> dict:
    d = np.load(exp_paths(cfg or load_config())["windows"] / f"windows_{name}.npz")
    return {k: d[k] for k in d.files}


def make_loader(split: dict, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(split["X"]), torch.from_numpy(split["y"]))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch_size])).numpy() for i in range(0, len(X), batch_size)]
    return np.concatenate(out) if out else np.empty(0, np.float32)


def train_model(name: str, cfg: dict | None = None, epochs: int | None = None,
                progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Train one model and save weights, config and history. Returns a summary dict."""
    cfg = cfg or load_config()
    tcfg = cfg["training"]
    seed = cfg["project"]["seed"]
    set_seed(seed)
    torch.set_num_threads(tcfg.get("num_threads", 4))
    epochs = epochs or tcfg["epochs"]

    train, val = load_split("train", cfg), load_split("val", cfg)
    n_features = train["X"].shape[-1]
    params = cfg["models"][name]
    model = build_model(name, n_features, params)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Training %s (%d params) on %d windows, validating on %d", MODEL_NAMES[name], n_params,
             len(train["y"]), len(val["y"]))

    train_loader = make_loader(train, tcfg["batch_size"], shuffle=True, seed=seed)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])

    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(yb)
            count += len(yb)
        train_loss = total / count
        val_pred = predict(model, val["X"])
        val_loss = float(np.mean((val_pred - val["y"]) ** 2))
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "best_val_loss": best_val, "elapsed_s": round(time.time() - t0, 1), "improved": improved}
        history.append(rec)
        log.info("%s epoch %3d  train %.5f  val %.5f %s", name, epoch, train_loss, val_loss,
                 "*" if improved else "")
        if progress_callback:
            progress_callback({**rec, "model": name, "epochs": epochs})
        if wait >= tcfg["early_stopping_patience"]:
            log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    paths = exp_paths(cfg)
    model_dir, metrics_dir = paths["models"], paths["metrics"]
    torch.save(model.state_dict(), model_dir / f"{name}.pt")
    summary = {
        "model": name, "display_name": MODEL_NAMES[name], "hyperparameters": params,
        "n_features": n_features, "features": cfg["features"], "target": cfg["target"],
        "input_window": cfg["sequences"]["input_window"], "horizon": cfg["sequences"]["horizon"],
        "target_mode": cfg["sequences"].get("target_mode", "level"),
        "n_trainable_params": n_params, "seed": seed, "optimizer": "Adam", "loss": "MSELoss",
        "learning_rate": tcfg["learning_rate"], "batch_size": tcfg["batch_size"],
        "max_epochs": epochs, "early_stopping_patience": tcfg["early_stopping_patience"],
        "epochs_run": len(history), "best_epoch": best_epoch, "best_val_mse_scaled": best_val,
        "training_time_s": round(time.time() - t0, 1), "torch_version": torch.__version__,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    with open(model_dir / f"{name}_config.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(history).to_csv(metrics_dir / f"{name}_history.csv", index=False)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", *MODEL_NAMES])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    names = list(MODEL_NAMES) if args.model == "all" else [args.model]
    cfg = load_config()
    log.info("Config: target_mode=%s models->%s", cfg["sequences"].get("target_mode"), cfg["training"]["model_dir"])
    for n in names:
        s = train_model(n, cfg, epochs=args.epochs)
        log.info("Done %s: best epoch %d, val MSE (scaled) %.5f, %.1fs", n, s["best_epoch"],
                 s["best_val_mse_scaled"], s["training_time_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
