"""Prediction - run a trained model on any real validation/test window."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (FEATURE_LABELS, MODEL_NAMES, PROXY_NOTE, available_models, config, current_threshold,
                    decision_box, get_model, get_settings, page_setup, predict_vwc, scalers, windows)

page_setup("Prediction", "🔮")
st.title("🔮 Prediction")
models = available_models()
if not models:
    st.warning("No trained models found.")
    st.stop()

cfg = config()
features, target = cfg["features"], cfg["target"]
settings = get_settings()
thr = current_threshold()

c1, c2 = st.columns([1, 2])
name = c1.selectbox("Model", models, index=models.index(settings["model"]) if settings["model"] in models else 0,
                    format_func=MODEL_NAMES.get)
split = c1.radio("Window source", ["test", "val"], horizontal=True,
                 help="Only held-out windows are offered so predictions are genuine out-of-sample.")
w = windows(split)
times = pd.to_datetime(w["target_time"])
idx = c2.select_slider("Forecast hour (UTC, end of hour)", options=list(range(len(times))),
                       value=len(times) - 1, format_func=lambda i: f"{times[i]:%Y-%m-%d %H:%M}")

x = w["X"][idx]
pred = float(predict_vwc(name, x[None])[0])
actual = float(w["y_raw"][idx])
from src.prediction.irrigation import recommendation  # noqa: E402

rec = recommendation(pred, thr["threshold"])
actual_dec = "YES" if actual < thr["threshold"] else "NO"

m = st.columns(4)
m[0].metric("Predicted SM 20 cm", f"{pred:.4f} m³/m³")
m[1].metric("Observed SM 20 cm", f"{actual:.4f} m³/m³", delta=f"error {pred - actual:+.4f}", delta_color="off")
m[2].metric("Last input-hour SM 20 cm", f"{w['last_observed_target'][idx]:.4f} m³/m³")
m[3].metric("Proxy decision from observed value", actual_dec)
decision_box(rec)
st.caption(PROXY_NOTE)

raw_x = scalers()["x_scaler"].inverse_transform(x)
in_times = pd.date_range(end=times[idx] - pd.Timedelta(hours=1), periods=len(x), freq="h")
inp = pd.DataFrame(raw_x, index=in_times, columns=features)

st.subheader("24-hour input window")
fig = go.Figure()
fig.add_trace(go.Scatter(x=inp.index, y=inp[target], name="SM 20 cm (input)", line=dict(color="#1d2939")))
fig.add_trace(go.Scatter(x=[times[idx]], y=[actual], mode="markers", name="observed", marker=dict(size=11, color="#1d2939")))
fig.add_trace(go.Scatter(x=[times[idx]], y=[pred], mode="markers", name="predicted",
                         marker=dict(size=12, symbol="diamond", color="#e76f51")))
fig.add_hline(y=thr["threshold"], line_dash="dash", line_color="#b42318", annotation_text="threshold")
fig.update_layout(height=330, template="plotly_white", yaxis_title="VWC (m³/m³)", margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

sel = st.multiselect("Other input variables", features, default=["P_CALC", "T_HR_AVG", "SOLARAD"],
                     format_func=FEATURE_LABELS.get)
if sel:
    st.line_chart(inp[sel], height=260)

st.subheader("Why this prediction? (occlusion)")
from src.explainability.feature_importance import occlusion_explanation  # noqa: E402

occ = occlusion_explanation(get_model(name), x.copy(), scalers(), features)
occ["label"] = occ.feature.map(FEATURE_LABELS)
fig = go.Figure(go.Bar(x=occ.contribution, y=occ.label, orientation="h",
                       marker_color=np.where(occ.contribution >= 0, "#2a9d8f", "#e76f51")))
fig.update_layout(height=380, template="plotly_white", xaxis_title="Δ predicted VWC vs. variable at training mean (m³/m³)",
                  yaxis=dict(autorange="reversed"), margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("Each bar: prediction with the real 24-h history minus prediction with that variable replaced by "
           "its training-period mean. It is a local sensitivity, not a causal effect.")

with st.expander("Raw input values"):
    st.dataframe(inp.style.format("{:.3f}"), use_container_width=True)
