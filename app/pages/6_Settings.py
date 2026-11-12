"""Settings - irrigation threshold, default model, forecast configuration."""
from __future__ import annotations

import numpy as np
import streamlit as st

from common import MODEL_NAMES, PROXY_NOTE, available_models, config, get_settings, page_setup, save_settings, windows

page_setup("Settings", "⚙️")
st.title("⚙️ Settings")
cfg = config()
s = dict(get_settings())
models = available_models() or list(MODEL_NAMES)

st.subheader("Model")
s["model"] = st.selectbox("Default model for Dashboard and Prediction", models,
                          index=models.index(s["model"]) if s["model"] in models else 0, format_func=MODEL_NAMES.get)

st.subheader("Irrigation threshold")
st.markdown(PROXY_NOTE)
mode = st.radio("Threshold mode", ["train_percentile", "fixed"], index=0 if s["threshold_mode"] == "train_percentile" else 1,
                format_func=lambda m: {"train_percentile": "Percentile of training-period SM 20 cm",
                                       "fixed": "Fixed value (m³/m³)"}[m], horizontal=True)
s["threshold_mode"] = mode
ytr = windows("train")["y_raw"]
if mode == "train_percentile":
    s["percentile"] = st.slider("Percentile", 1, 99, int(s["percentile"]))
    st.write(f"→ threshold = **{np.percentile(ytr, s['percentile']):.4f} m³/m³** "
             f"(training-period distribution only, n={len(ytr):,})")
else:
    s["fixed_threshold"] = st.number_input("Threshold VWC (m³/m³)", 0.05, 0.60, float(s["fixed_threshold"]),
                                           step=0.005, format="%.3f")
    st.caption("Use a value calibrated for your crop and soil, e.g. derived from field capacity, wilting "
               "point and a management-allowed depletion fraction.")
st.caption(f"Training-period SM 20 cm: min {ytr.min():.3f}, 25th pct {np.percentile(ytr, 25):.3f}, "
           f"median {np.median(ytr):.3f}, max {ytr.max():.3f} m³/m³.")

if st.button("Save settings", type="primary"):
    save_settings(s)
    st.success("Saved to app_settings.json")

st.subheader("Forecast configuration (read-only)")
st.markdown(f"- Input window: **{cfg['sequences']['input_window']} h**, horizon: **{cfg['sequences']['horizon']} h**, "
            f"stride {cfg['sequences']['stride']} h\n"
            f"- Features ({len(cfg['features'])}): {', '.join(cfg['features'])}\n"
            f"- Target: `{cfg['target']}`\n"
            f"- Split: {cfg['sequences']['train_frac']:.0%} / {cfg['sequences']['val_frac']:.0%} / "
            f"{cfg['sequences']['test_frac']:.0%} chronological")
st.caption("Trained models are tied to this configuration. To change window, horizon or features, edit "
           "`config.yaml` and re-run the pipeline (`create_sequences` → `train` → `evaluate`).")
