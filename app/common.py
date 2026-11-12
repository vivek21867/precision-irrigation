"""Shared helpers for the Streamlit app: cached loaders, settings, styling.

Everything shown in the app is read from the real pipeline outputs
(data/processed, models/, results/). Nothing is hard-coded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import MODEL_NAMES  # noqa: E402
from src.prediction.irrigation import resolve_threshold  # noqa: E402
from src.utils import load_config  # noqa: E402

SETTINGS_FILE = ROOT / "app_settings.json"
PROC = ROOT / "data" / "processed"
META = ROOT / "data" / "metadata"
RES = ROOT / "results"
MODELS = ROOT / "models"

FEATURE_LABELS = {
    "T_HR_AVG": "Air temperature (°C)",
    "P_CALC": "Precipitation (mm)",
    "SOLARAD": "Solar radiation (W/m²)",
    "SUR_TEMP": "Surface temperature (°C)",
    "RH_HR_AVG": "Relative humidity (%)",
    "SOIL_TEMP_20": "Soil temperature 20 cm (°C)",
    "SOIL_MOISTURE_5": "Soil moisture 5 cm (m³/m³)",
    "SOIL_MOISTURE_10": "Soil moisture 10 cm (m³/m³)",
    "SOIL_MOISTURE_20": "Soil moisture 20 cm (m³/m³)",
    "SOIL_MOISTURE_50": "Soil moisture 50 cm (m³/m³)",
    "SOIL_MOISTURE_100": "Soil moisture 100 cm (m³/m³)",
}
MODEL_COLORS = {"Persistence": "#9e9e9e", "LSTM": "#1f77b4", "BiLSTM": "#2a9d8f", "Transformer": "#e76f51"}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e6ebf0; border-radius: 10px;
  padding: 12px 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04);}
.decision {border-radius: 12px; padding: 18px 22px; font-size: 1.25rem; font-weight: 600;}
.decision.yes {background: #fdecea; color: #b42318; border: 1px solid #f5c2bd;}
.decision.no  {background: #e8f6ef; color: #146c43; border: 1px solid #b7e4cc;}
.note {color: #5b6b7b; font-size: .88rem;}
</style>
"""


def page_setup(title: str, icon: str = "💧") -> None:
    st.set_page_config(page_title=f"{title} · Precision Irrigation", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def config() -> dict:
    return load_config()


def get_settings() -> dict:
    """App settings: persisted JSON, falling back to config.yaml values."""
    cfg = config()
    defaults = {"model": "transformer", "threshold_mode": cfg["irrigation"]["mode"],
                "percentile": cfg["irrigation"]["percentile"],
                "fixed_threshold": cfg["irrigation"].get("threshold") or 0.30}
    if "settings" not in st.session_state:
        saved = {}
        if SETTINGS_FILE.exists():
            try:
                saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                saved = {}
        st.session_state["settings"] = {**defaults, **saved}
    return st.session_state["settings"]


def save_settings(s: dict) -> None:
    st.session_state["settings"] = s
    SETTINGS_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


@st.cache_data
def windows(split: str) -> dict:
    d = np.load(PROC / f"windows_{split}.npz")
    return {k: d[k] for k in d.files}


def current_threshold() -> dict:
    s = get_settings()
    if s["threshold_mode"] == "fixed":
        return resolve_threshold({"mode": "fixed", "threshold": s["fixed_threshold"]})
    return resolve_threshold({"mode": "train_percentile", "percentile": s["percentile"]},
                             windows("train")["y_raw"])


@st.cache_data
def cleaned() -> pd.DataFrame:
    return pd.read_parquet(PROC / "cleaned_hourly.parquet")


@st.cache_data
def read_csv(rel: str, **kw) -> pd.DataFrame | None:
    p = ROOT / rel
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def read_json(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def available_models() -> list[str]:
    return [m for m in MODEL_NAMES if (MODELS / f"{m}.pt").exists()]


@st.cache_resource
def model(name: str, mtime: float):
    """Load a trained model (cache keyed on file mtime so retraining refreshes it)."""
    from src.training.evaluate import load_trained

    return load_trained(name, config())


def get_model(name: str):
    p = MODELS / f"{name}.pt"
    return model(name, p.stat().st_mtime) if p.exists() else None


@st.cache_resource
def scalers() -> dict:
    from src.training.evaluate import load_scalers

    return load_scalers()


def predict_vwc(name: str, X: np.ndarray) -> np.ndarray:
    from src.training.evaluate import predict_level

    return predict_level(get_model(name), X.astype(np.float32), scalers())


def require(paths: list[str], hint: str) -> bool:
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        st.warning(f"Missing pipeline outputs: {', '.join(missing)}. {hint}")
        return False
    return True


def decision_box(rec: dict) -> None:
    cls = "yes" if rec["irrigate"] == "YES" else "no"
    label = "IRRIGATION REQUIRED — YES" if cls == "yes" else "IRRIGATION NOT REQUIRED — NO"
    st.markdown(f'<div class="decision {cls}">{label}<br><span class="note">'
                f'Predicted {rec["predicted_vwc"]:.4f} m³/m³ vs threshold {rec["threshold"]:.4f} m³/m³ '
                f'(margin {rec["margin"]:+.4f})</span></div>', unsafe_allow_html=True)


PROXY_NOTE = ("Decisions are a **threshold-based proxy**: irrigate if predicted 20 cm soil moisture is "
              "below the configured threshold. USCRN has no irrigation-event labels, the threshold needs "
              "agronomic calibration for each crop and soil, and no water quantity is estimated.")
