"""Manual (user-entered) inputs -> next-hour soil-moisture forecast.

The trained networks need exactly what they were trained on: a 24-hour window
of the 11 configured hourly features, in the physical units of the NOAA USCRN
hourly02 product. Inputs are checked against the configured physical bounds
(hard error) and the training-period range (warning: the model is extrapolating).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models import MODEL_NAMES
from src.prediction.irrigation import recommendation
from src.training.evaluate import load_scalers, load_trained, predict_level
from src.utils import exp_paths, project_path

FEATURE_META = {
    "T_HR_AVG": ("Air temperature", "°C", "Hourly average air temperature"),
    "P_CALC": ("Precipitation", "mm", "Total precipitation in the hour"),
    "SOLARAD": ("Solar radiation", "W/m²", "Average global solar radiation"),
    "SUR_TEMP": ("Surface temperature", "°C", "Average infrared ground-surface temperature"),
    "RH_HR_AVG": ("Relative humidity", "%", "Hourly average relative humidity"),
    "SOIL_TEMP_20": ("Soil temperature 20 cm", "°C", "Soil temperature at 20 cm depth"),
    "SOIL_MOISTURE_5": ("Soil moisture 5 cm", "m³/m³", "Volumetric water content at 5 cm"),
    "SOIL_MOISTURE_10": ("Soil moisture 10 cm", "m³/m³", "Volumetric water content at 10 cm"),
    "SOIL_MOISTURE_20": ("Soil moisture 20 cm", "m³/m³", "Volumetric water content at 20 cm (target history)"),
    "SOIL_MOISTURE_50": ("Soil moisture 50 cm", "m³/m³", "Volumetric water content at 50 cm"),
    "SOIL_MOISTURE_100": ("Soil moisture 100 cm", "m³/m³", "Volumetric water content at 100 cm"),
}


def _r(v: float) -> float:
    return round(float(v), 6)


class ManualPredictor:
    """Loads data, scalers and every trained model once; predicts arbitrary windows."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.features = cfg["features"]
        self.target = cfg["target"]
        self.window = cfg["sequences"]["input_window"]
        self.bounds = cfg["preprocessing"]["physical_bounds"]
        paths = exp_paths(cfg)
        self.sc = load_scalers(cfg)
        self.models = {MODEL_NAMES[k]: load_trained(k, cfg) for k in MODEL_NAMES
                       if (paths["models"] / f"{k}.pt").exists()}
        if not self.models:
            raise RuntimeError("No trained models found - run src.training.train first")

        df = pd.read_parquet(project_path(cfg["data"]["processed_dir"]) / "cleaned_hourly.parquet")
        seq = json.loads((paths["windows"] / "sequence_metadata.json").read_text("utf-8"))
        train = df.loc[df.index < pd.Timestamp(seq["cuts"]["train_end_exclusive"]), self.features]
        self.train_range = {f: (float(train[f].min()), float(train[f].max())) for f in self.features}
        self.train_mean = {f: float(train[f].mean()) for f in self.features}

        last = df.iloc[-self.window:]
        self.latest_window = last
        irr = json.loads((paths["metrics"] / "irrigation_metrics.json").read_text("utf-8"))
        self.threshold = irr["threshold"]

    def feature_info(self) -> list[dict]:
        return [{"key": f, "label": FEATURE_META[f][0], "unit": FEATURE_META[f][1],
                 "description": FEATURE_META[f][2],
                 "bounds": [float(b) for b in self.bounds[f]],
                 "train_min": _r(self.train_range[f][0]), "train_max": _r(self.train_range[f][1]),
                 "train_mean": _r(self.train_mean[f]),
                 "latest": _r(self.latest_window[f].iloc[-1])} for f in self.features]

    def meta(self) -> dict:
        return {
            "features": self.feature_info(),
            "target": self.target,
            "input_window": self.window,
            "horizon": self.cfg["sequences"]["horizon"],
            "target_mode": self.sc.get("target_mode", "level"),
            "models": list(self.models),
            "threshold": self.threshold,
            "latest_window": {
                "times": [t.strftime("%Y-%m-%dT%H:%M") for t in self.latest_window.index],
                "rows": [[_r(v) for v in row] for row in self.latest_window[self.features].to_numpy()],
            },
        }

    def validate(self, window) -> tuple[np.ndarray, list[str]]:
        """Return the (24, 11) array or raise ValueError; plus extrapolation warnings."""
        try:
            arr = np.asarray(window, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError("All inputs must be numbers") from e
        if arr.shape != (self.window, len(self.features)):
            raise ValueError(f"Expected {self.window} hours x {len(self.features)} features, got {list(arr.shape)}")
        if not np.isfinite(arr).all():
            raise ValueError("Every input must be a finite number (no blanks)")
        errors, warnings = [], []
        for j, f in enumerate(self.features):
            lo, hi = self.bounds[f]
            col = arr[:, j]
            if (col < lo).any() or (col > hi).any():
                errors.append(f"{FEATURE_META[f][0]} must be within {lo}–{hi} {FEATURE_META[f][1]}")
            tlo, thi = self.train_range[f]
            if (col < tlo).any() or (col > thi).any():
                warnings.append(f"{FEATURE_META[f][0]} is outside the training range "
                                f"{tlo:g}–{thi:g} {FEATURE_META[f][1]}; the model is extrapolating")
        if errors:
            raise ValueError("; ".join(errors))
        return arr, warnings

    def predict(self, window) -> dict:
        arr, warnings = self.validate(window)
        X = self.sc["x_scaler"].transform(arr).reshape(1, self.window, len(self.features)).astype(np.float32)
        last = float(arr[-1, self.features.index(self.target)])
        thr = self.threshold["threshold"]
        out = {"Persistence": last}
        out.update({name: float(predict_level(m, X, self.sc)[0]) for name, m in self.models.items()})
        return {
            "last_observed": _r(last),
            "threshold": self.threshold,
            "predictions": {name: {"value": _r(v), "change": _r(v - last), **{
                k: recommendation(v, thr)[k] for k in ("irrigate", "margin")}}
                for name, v in out.items()},
            "warnings": warnings,
        }
