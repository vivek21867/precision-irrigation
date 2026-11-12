"""Feature-importance analysis for the trained models.

Methods (all computed on the trained weights and the held-out test windows):
  1. Permutation importance: for each input variable, its whole 24-h input
     sequence is swapped between test windows (breaking its link to the
     target while keeping its marginal distribution); importance = increase
     in test RMSE (m^3/m^3), averaged over repeats.
  2. GradientSHAP (shap.GradientExplainer, expected gradients) with a
     training-set background: mean |SHAP| per variable and per (lag, variable).
     SHAP explains the NETWORK output: with target_mode="delta" that is the
     predicted 1-h change in SM20, i.e. what drives wetting/drying beyond
     persistence. Permutation importance and occlusion explain the complete
     forecaster (network + add-back of the last observed SM20).
  3. Local occlusion explanation for a single window (used by the app): each
     variable is replaced by its training mean (0 after standardisation) and
     the change in predicted VWC is reported.

Usage:
    python -m src.explainability.feature_importance
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.models import MODEL_NAMES
from src.training.evaluate import load_scalers, load_trained, predict_level
from src.training.train import load_split
from src.utils import exp_paths, get_logger, load_config, set_seed

log = get_logger("explain", "explain.log")


def permutation_importance(model, X: np.ndarray, y: np.ndarray, sc: dict, features: list[str],
                           repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = predict_level(model, X, sc)
    base_rmse = float(np.sqrt(np.mean((base - y) ** 2)))
    rows = []
    for j, f in enumerate(features):
        scores = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, :, j] = X[rng.permutation(len(X)), :, j]
            p = predict_level(model, Xp, sc)
            scores.append(float(np.sqrt(np.mean((p - y) ** 2))) - base_rmse)
        rows.append({"feature": f, "rmse_increase_mean": float(np.mean(scores)),
                     "rmse_increase_std": float(np.std(scores)), "baseline_rmse": base_rmse})
    return pd.DataFrame(rows).sort_values("rmse_increase_mean", ascending=False)


def gradient_shap(model, X_bg: np.ndarray, X_ex: np.ndarray, y_scale: float) -> np.ndarray:
    """Return SHAP values (n, lags, features) in physical units (m^3/m^3)."""
    import shap

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).unsqueeze(-1)

    # cuDNN-free CPU LSTM backward requires train mode only for RNN dropout; we keep eval mode.
    explainer = shap.GradientExplainer(Wrap(model), torch.from_numpy(X_bg))
    sv = explainer.shap_values(torch.from_numpy(X_ex))
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
    if sv.ndim == 4:
        sv = sv[..., 0]
    return sv * y_scale


def occlusion_explanation(model, x: np.ndarray, sc: dict, features: list[str]) -> pd.DataFrame:
    """Local explanation for one window x (lags, features), standardised.

    contribution = prediction(original) - prediction(feature set to training mean).
    """
    base = predict_level(model, x[None], sc)[0]
    batch = np.repeat(x[None], len(features), axis=0)
    for j in range(len(features)):
        batch[j, :, j] = 0.0
    occl = predict_level(model, batch, sc)
    return pd.DataFrame({"feature": features, "contribution": base - occl}).sort_values(
        "contribution", key=np.abs, ascending=False)


def main() -> int:
    cfg = load_config()
    ecfg = cfg["explainability"]
    seed = cfg["project"]["seed"]
    set_seed(seed)
    torch.set_num_threads(cfg["training"].get("num_threads", 4))
    features = cfg["features"]
    paths = exp_paths(cfg)
    gdir, mdir = paths["graphs"], paths["metrics"]

    scalers = load_scalers(cfg)
    y_scaler = scalers["y_scaler"]
    train, test = load_split("train", cfg), load_split("test", cfg)
    n = min(ecfg["max_samples"], len(test["y"]))
    X, y = test["X"][:n], test["y_raw"][:n].astype(float)

    rng = np.random.default_rng(seed)
    bg = train["X"][rng.choice(len(train["X"]), ecfg["shap_background"], replace=False)]
    ex_idx = np.linspace(0, len(test["X"]) - 1, ecfg["shap_samples"]).astype(int)

    perm_all, shap_feat, shap_lag, summary = [], {}, {}, {}
    for name, disp in MODEL_NAMES.items():
        model = load_trained(name, cfg)
        if model is None:
            log.warning("No trained model %s", name)
            continue
        pi = permutation_importance(model, X, y, scalers, features, ecfg["permutation_repeats"], seed)
        pi.insert(0, "model", disp)
        perm_all.append(pi)
        log.info("%s permutation importance top-3: %s", disp, pi.feature.head(3).tolist())
        try:
            sv = gradient_shap(model, bg, test["X"][ex_idx], float(y_scaler.scale_[0]))
            shap_feat[disp] = np.abs(sv).sum(axis=1).mean(axis=0)   # sum over lags, mean over samples
            shap_lag[disp] = np.abs(sv).mean(axis=0)                 # (lags, features)
            summary[disp] = {"shap": "ok", "n_explained": int(len(ex_idx)), "background": int(len(bg))}
        except Exception as e:  # SHAP is optional; record failure honestly
            log.warning("SHAP failed for %s: %s", disp, e)
            summary[disp] = {"shap": f"failed: {e}"}

    perm = pd.concat(perm_all)
    perm.to_csv(mdir / "permutation_importance.csv", index=False)
    if shap_feat:
        sf = pd.DataFrame(shap_feat, index=features)
        sf.to_csv(mdir / "shap_mean_abs_by_feature.csv")
        for disp, arr in shap_lag.items():
            lags = [f"t-{len(arr) - 1 - i}" for i in range(len(arr))]
            pd.DataFrame(arr, index=lags, columns=features).to_csv(
                mdir / f"shap_lag_feature_{disp.lower()}.csv")
    with open(mdir / "explainability_summary.json", "w", encoding="utf-8") as f:
        json.dump({"permutation": {"samples": int(n), "repeats": ecfg["permutation_repeats"],
                                   "metric": "increase in test RMSE (m3/m3)"},
                   "shap": summary}, f, indent=2)

    # ---- Graphs ----
    sns.set_theme(style="whitegrid")
    models = perm.model.unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.8))
    for ax, d in zip(np.atleast_1d(axes), models):
        s = perm[perm.model == d].sort_values("rmse_increase_mean")
        ax.barh(s.feature, s.rmse_increase_mean, xerr=s.rmse_increase_std, color="#2a9d8f")
        ax.set_title(f"{d}: permutation importance"); ax.set_xlabel("RMSE increase (m3/m3)")
    fig.tight_layout(); fig.savefig(gdir / "permutation_importance.png", dpi=130); plt.close(fig)

    if shap_feat:
        fig, axes = plt.subplots(1, len(shap_feat), figsize=(5.5 * len(shap_feat), 4.8))
        for ax, (d, v) in zip(np.atleast_1d(axes), shap_feat.items()):
            s = pd.Series(v, index=features).sort_values()
            ax.barh(s.index, s.values, color="#e76f51")
            ax.set_title(f"{d}: mean |SHAP|"); ax.set_xlabel("m3/m3 (summed over 24 lags)")
        fig.tight_layout(); fig.savefig(gdir / "shap_feature_importance.png", dpi=130); plt.close(fig)

        fig, axes = plt.subplots(1, len(shap_lag), figsize=(6 * len(shap_lag), 5.5))
        for ax, (d, arr) in zip(np.atleast_1d(axes), shap_lag.items()):
            lags = [f"t-{len(arr) - 1 - i}" for i in range(len(arr))]
            sns.heatmap(pd.DataFrame(arr, index=lags, columns=features), cmap="viridis", ax=ax,
                        cbar_kws={"label": "mean |SHAP|"})
            ax.set_title(f"{d}: importance by lag and variable")
            ax.tick_params(axis="x", rotation=60)
        fig.tight_layout(); fig.savefig(gdir / "shap_lag_heatmap.png", dpi=130); plt.close(fig)

    print(perm.pivot(index="feature", columns="model", values="rmse_increase_mean")
          .sort_values(models[0], ascending=False).to_string(float_format=lambda x: f"{x:.6f}"))
    if shap_feat:
        print("\nMean |SHAP| by feature:\n", sf.to_string(float_format=lambda x: f"{x:.6f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
