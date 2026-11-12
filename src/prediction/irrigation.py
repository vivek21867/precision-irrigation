"""Threshold-based irrigation decision module.

IMPORTANT LIMITATIONS (also in README / methodology):
  * This is a PROXY decision: irrigate if the PREDICTED next-hour volumetric
    water content (VWC, m^3/m^3) at 20 cm falls below a configured threshold.
  * USCRN contains no irrigation-actuation labels, so "ground truth" decisions
    are derived by applying the same threshold to the OBSERVED next-hour VWC.
    The evaluation therefore measures how well the forecast reproduces
    threshold crossings - not whether a farmer would actually irrigate.
  * No single threshold suits every crop/soil; it must be agronomically
    calibrated (e.g. from field capacity, wilting point and management-allowed
    depletion for the specific soil and crop).
  * The module does not estimate irrigation water quantity.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

IRRIGATE, NO_IRRIGATE = "YES", "NO"


def resolve_threshold(irr_cfg: dict, y_train_raw: np.ndarray | None = None) -> dict:
    """Return {'threshold': float, 'source': str} from config.

    mode 'fixed'            -> irr_cfg['threshold']
    mode 'train_percentile' -> percentile of TRAINING-split target values only.
    """
    mode = irr_cfg.get("mode", "fixed")
    if mode == "fixed":
        if irr_cfg.get("threshold") is None:
            raise ValueError("irrigation.mode=fixed requires irrigation.threshold")
        return {"threshold": float(irr_cfg["threshold"]), "source": "fixed (config)"}
    if mode == "train_percentile":
        if y_train_raw is None or len(y_train_raw) == 0:
            raise ValueError("train_percentile mode needs training target values")
        p = float(irr_cfg.get("percentile", 25))
        return {"threshold": float(np.percentile(y_train_raw, p)),
                "source": f"{p:g}th percentile of training-period {len(y_train_raw)} target values"}
    raise ValueError(f"Unknown irrigation mode: {mode}")


def decide(predicted_vwc: np.ndarray | float, threshold: float) -> np.ndarray:
    """True = irrigation required (predicted VWC below threshold)."""
    return np.asarray(predicted_vwc) < threshold


def recommendation(predicted_vwc: float, threshold: float) -> dict:
    """Human-readable recommendation for a single prediction."""
    need = bool(predicted_vwc < threshold)
    return {
        "irrigate": IRRIGATE if need else NO_IRRIGATE,
        "predicted_vwc": float(predicted_vwc),
        "threshold": float(threshold),
        "margin": float(predicted_vwc - threshold),
        "message": ("IRRIGATION REQUIRED - predicted soil moisture is below the configured threshold"
                    if need else
                    "IRRIGATION NOT REQUIRED - predicted soil moisture is at/above the configured threshold"),
    }


def evaluate_decisions(pred_vwc: np.ndarray, actual_vwc: np.ndarray, threshold: float) -> dict:
    """Classification metrics with IRRIGATE (=1) as the positive class."""
    y_true = decide(actual_vwc, threshold).astype(int)
    y_pred = decide(pred_vwc, threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_rate_actual": float(y_true.mean()),
        "positive_rate_predicted": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_irrigate": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_irrigate": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_irrigate": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_no_irrigate": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_no_irrigate": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # rows = actual [NO, YES], cols = predicted [NO, YES]
        "confusion_matrix": cm.tolist(),
    }
