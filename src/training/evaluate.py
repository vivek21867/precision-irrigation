"""Evaluate trained models on the held-out chronological test split.

All metrics are computed in physical units (m^3/m^3) after inverting the
target scaler. A naive persistence baseline (next hour = last observed hour)
is reported alongside the deep models, because at a 1-hour horizon soil
moisture is highly autocorrelated and any model must be judged against it.

Usage:
    python -m src.training.evaluate
"""
from __future__ import annotations

import json
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models import MODEL_NAMES, build_model
from src.prediction.irrigation import evaluate_decisions, resolve_threshold
from src.training.train import load_split, predict
from src.utils import exp_paths, get_logger, load_config

log = get_logger("evaluate", "evaluate.log")
COLORS = {"Persistence": "#9e9e9e", "LSTM": "#1f77b4", "BiLSTM": "#2a9d8f", "Transformer": "#e76f51"}


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    mse = mean_squared_error(y, p)
    return {"MAE": float(mean_absolute_error(y, p)), "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y, p))}


def load_trained(name: str, cfg: dict):
    mdir = exp_paths(cfg)["models"]
    path = mdir / f"{name}.pt"
    if not path.exists():
        return None
    with open(mdir / f"{name}_config.json", encoding="utf-8") as f:
        mcfg = json.load(f)
    model = build_model(name, mcfg["n_features"], mcfg["hyperparameters"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_scalers(cfg: dict | None = None) -> dict:
    with open(exp_paths(cfg or load_config())["windows"] / "scalers.pkl", "rb") as f:
        return pickle.load(f)


def last_observed_from_X(X: np.ndarray, sc: dict) -> np.ndarray:
    """Unscaled target value at the final input hour t (read from the window itself)."""
    i = sc["target_index"]
    return X[:, -1, i].astype(np.float64) * sc["x_scaler"].scale_[i] + sc["x_scaler"].mean_[i]


def predict_level(model, X: np.ndarray, sc: dict) -> np.ndarray:
    """Predicted next-hour target level in physical units (m^3/m^3).

    level mode: inverse-scaled network output.
    delta mode: last observed value (from the window) + inverse-scaled predicted change.
    """
    out = sc["y_scaler"].inverse_transform(predict(model, X).reshape(-1, 1)).ravel()
    if sc.get("target_mode", "level") == "delta":
        out = last_observed_from_X(X, sc) + out
    return out


def main() -> int:
    cfg = load_config()
    paths = exp_paths(cfg)
    gdir, mdir = paths["graphs"], paths["metrics"]
    torch.set_num_threads(cfg["training"].get("num_threads", 4))

    scalers = load_scalers(cfg)
    log.info("Evaluating target_mode=%s, models in %s", scalers.get("target_mode"), paths["models"])
    train, val, test = (load_split(s, cfg) for s in ("train", "val", "test"))
    y_test = test["y_raw"].astype(float)
    times = pd.to_datetime(test["target_time"])

    preds = {"Persistence": test["last_observed_target"].astype(float)}
    val_preds = {"Persistence": val["last_observed_target"].astype(float)}
    for name, disp in MODEL_NAMES.items():
        m = load_trained(name, cfg)
        if m is None:
            log.warning("No trained weights for %s - skipped", name)
            continue
        preds[disp] = predict_level(m, test["X"], scalers)
        val_preds[disp] = predict_level(m, val["X"], scalers)

    rows = []
    for disp, p in preds.items():
        rows.append({"model": disp, "split": "test", **regression_metrics(y_test, p)})
        rows.append({"model": disp, "split": "val", **regression_metrics(val["y_raw"], val_preds[disp])})
    mt = pd.DataFrame(rows)
    mt.to_csv(mdir / "regression_metrics.csv", index=False)
    test_tbl = mt[mt.split == "test"].drop(columns="split").set_index("model")
    persist_rmse = test_tbl.loc["Persistence", "RMSE"]
    test_tbl["RMSE_skill_vs_persistence"] = 1 - test_tbl["RMSE"] / persist_rmse
    test_tbl.to_csv(mdir / "test_metrics_table.csv")
    (mdir / "test_metrics_table.md").write_text(test_tbl.to_markdown(floatfmt=".6f"), encoding="utf-8")

    pred_df = pd.DataFrame({"target_time": times, "actual": y_test, **{k: v for k, v in preds.items()}})
    pred_df.to_csv(mdir / "test_predictions.csv", index=False)

    # ---- Irrigation decisions ----
    thr = resolve_threshold(cfg["irrigation"], train["y_raw"])
    irr = {"threshold": thr, "models": {d: evaluate_decisions(p, y_test, thr["threshold"]) for d, p in preds.items()}}
    pcts = np.arange(5, 55, 5)
    sweep = []
    for pct in pcts:
        t = float(np.percentile(train["y_raw"], pct))
        for d, p in preds.items():
            r = evaluate_decisions(p, y_test, t)
            sweep.append({"train_percentile": int(pct), "threshold": t, "model": d,
                          **{k: r[k] for k in ("accuracy", "precision_irrigate", "recall_irrigate",
                                               "f1_irrigate", "positive_rate_actual")}})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(mdir / "irrigation_threshold_sweep.csv", index=False)
    with open(mdir / "irrigation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(irr, f, indent=2)
    irr_tbl = pd.DataFrame({d: {k: v for k, v in r.items() if k != "confusion_matrix"}
                            for d, r in irr["models"].items()}).T
    irr_tbl.to_csv(mdir / "irrigation_metrics_table.csv")

    # ---- Graphs ----
    sns.set_theme(style="whitegrid")
    deep = [d for d in preds if d != "Persistence"]

    fig, axes = plt.subplots(len(deep), 1, figsize=(14, 3.2 * len(deep)), sharex=True)
    for ax, d in zip(np.atleast_1d(axes), deep):
        ax.plot(times, y_test, lw=0.8, color="black", label="Actual")
        ax.plot(times, preds[d], lw=0.8, color=COLORS[d], label=f"{d} predicted")
        ax.axhline(thr["threshold"], ls="--", color="red", lw=0.8, label="Irrigation threshold")
        ax.set_ylabel("VWC (m3/m3)"); ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{d}: actual vs predicted next-hour SOIL_MOISTURE_20 (test set)")
    fig.tight_layout(); fig.savefig(gdir / "actual_vs_predicted_timeseries.png", dpi=130); plt.close(fig)

    # zoom: first 14 days of test
    zoom = times < times.min() + pd.Timedelta(days=14)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times[zoom], y_test[zoom], color="black", lw=1.5, label="Actual")
    for d in preds:
        ax.plot(times[zoom], preds[d][zoom], lw=1, color=COLORS[d], label=d,
                ls=":" if d == "Persistence" else "-")
    ax.set_title("Test set, first 14 days (zoom)"); ax.set_ylabel("VWC (m3/m3)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(gdir / "actual_vs_predicted_zoom.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(deep), figsize=(5 * len(deep), 4.6))
    lo, hi = y_test.min(), y_test.max()
    for ax, d in zip(np.atleast_1d(axes), deep):
        ax.scatter(y_test, preds[d], s=3, alpha=0.4, color=COLORS[d])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{d}  R2={test_tbl.loc[d, 'R2']:.4f}"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    fig.tight_layout(); fig.savefig(gdir / "actual_vs_predicted_scatter.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for d in preds:
        sns.kdeplot(preds[d] - y_test, ax=ax, label=d, color=COLORS[d], fill=d != "Persistence", alpha=0.2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Prediction error (predicted - actual), m3/m3"); ax.set_title("Prediction error distribution (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(gdir / "error_distribution.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(deep), figsize=(5 * len(deep), 4))
    for ax, (name, d) in zip(np.atleast_1d(axes), [(n, MODEL_NAMES[n]) for n in MODEL_NAMES if MODEL_NAMES[n] in deep]):
        h = pd.read_csv(mdir / f"{name}_history.csv")
        ax.plot(h.epoch, h.train_loss, label="train", color=COLORS[d])
        ax.plot(h.epoch, h.val_loss, label="validation", color="black", ls="--")
        best = h.loc[h.val_loss.idxmin()]
        ax.scatter([best.epoch], [best.val_loss], color="red", zorder=5, label=f"best (epoch {int(best.epoch)})")
        ax.set_yscale("log"); ax.set_title(f"{d} loss (MSE, scaled)"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(gdir / "training_curves.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, met in zip(axes, ["MAE", "RMSE", "MSE", "R2"]):
        vals = test_tbl[met]
        ax.bar(vals.index, vals.values, color=[COLORS[i] for i in vals.index])
        ax.set_title(met); ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals.values):
            ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=8)
        if met == "R2":
            ax.set_ylim(min(0.9, vals.min() - 0.01), 1.0)
    fig.suptitle("Model comparison on held-out test set"); fig.tight_layout()
    fig.savefig(gdir / "model_comparison.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 3.8))
    for ax, d in zip(axes, preds):
        cm = np.array(irr["models"][d]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["NO", "YES"], yticklabels=["NO", "YES"])
        ax.set_xlabel("Predicted decision"); ax.set_ylabel("Proxy decision from observed VWC")
        ax.set_title(f"{d}\nrecall(YES)={irr['models'][d]['recall_irrigate']:.3f}")
    fig.suptitle(f"Irrigation decisions, threshold {thr['threshold']:.3f} m3/m3"); fig.tight_layout()
    fig.savefig(gdir / "confusion_matrices.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for d in preds:
        s = sweep_df[sweep_df.model == d]
        axes[0].plot(s.threshold, s.recall_irrigate, marker="o", label=d, color=COLORS[d])
        axes[1].plot(s.threshold, s.f1_irrigate, marker="o", label=d, color=COLORS[d])
    axes[0].set_title("Recall (IRRIGATE) vs threshold"); axes[1].set_title("F1 (IRRIGATE) vs threshold")
    for ax in axes:
        ax.set_xlabel("Threshold VWC (m3/m3)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(gdir / "irrigation_threshold_sweep.png", dpi=130); plt.close(fig)

    print("\nTEST regression metrics (m3/m3):\n", test_tbl.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nVAL regression metrics:\n", mt[mt.split == "val"].to_string(index=False))
    print(f"\nIrrigation threshold: {thr}")
    print(irr_tbl[["positive_rate_actual", "accuracy", "precision_irrigate", "recall_irrigate", "f1_irrigate"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
