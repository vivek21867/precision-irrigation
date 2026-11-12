"""Dashboard page (registered in main.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (FEATURE_LABELS, MODEL_NAMES, PROXY_NOTE, available_models, cleaned, config,
                    current_threshold, decision_box, get_settings, page_setup, predict_vwc, read_csv,
                    require, scalers)

page_setup("Dashboard")
st.title("💧 Precision Irrigation Dashboard")
st.caption("Deep-learning forecast of next-hour soil moisture (20 cm) from NOAA USCRN station "
           "NE_Lincoln_8_ENE (Lincoln, Nebraska) hourly observations.")

if not require(["data/processed/cleaned_hourly.parquet", "data/processed/scalers.pkl"],
               "Run the data pipeline first (see README)."):
    st.stop()

cfg = config()
features, target = cfg["features"], cfg["target"]
win = cfg["sequences"]["input_window"]
df = cleaned()
settings = get_settings()
models = available_models()
if not models:
    st.warning("No trained models found. Train them on the **Model Training** page or with "
               "`python -m src.training.train --model all`.")
    st.stop()
name = settings["model"] if settings["model"] in models else models[0]
thr = current_threshold()

# Latest contiguous 24-h window in the processed dataset -> forecast the following hour.
last = df.iloc[-win:]
contiguous = last["segment_id"].nunique() == 1
latest_time = df.index[-1]
latest = df.iloc[-1]

st.info(f"Data are historical research observations. The most recent hour with all {len(features)} "
        f"features present is **{latest_time:%Y-%m-%d %H:%M} UTC** (local standard time "
        f"{latest['LST_DATETIME']:%Y-%m-%d %H:%M}); the forecast below is for the hour after it.")

if not contiguous:
    st.error("The last 24 rows are not a contiguous hourly window; cannot forecast.")
    st.stop()

X = scalers()["x_scaler"].transform(last[features].to_numpy()).reshape(1, win, len(features))
pred = float(predict_vwc(name, X)[0])
from src.prediction.irrigation import recommendation  # noqa: E402

rec = recommendation(pred, thr["threshold"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observed SM20 (m³/m³)", f"{latest[target]:.3f}", help="Latest observed soil moisture at 20 cm")
c2.metric(f"Forecast SM20 · {MODEL_NAMES[name]}", f"{pred:.3f}",
          delta=f"{pred - latest[target]:+.4f}", help="Predicted next-hour soil moisture at 20 cm (m³/m³)")
c3.metric("Threshold (m³/m³)", f"{thr['threshold']:.3f}", help=thr["source"])
c4.metric("Recommendation", rec["irrigate"])
decision_box(rec)
st.caption(PROXY_NOTE)

st.subheader("Current environmental conditions")
cols = st.columns(6)
for col, f in zip(cols, ["T_HR_AVG", "RH_HR_AVG", "P_CALC", "SOLARAD", "SUR_TEMP", "SOIL_TEMP_20"]):
    col.metric(FEATURE_LABELS[f].split(" (")[0], f"{latest[f]:.1f}", help=FEATURE_LABELS[f])
cols = st.columns(5)
for col, f in zip(cols, [c for c in features if c.startswith("SOIL_MOISTURE")]):
    col.metric(FEATURE_LABELS[f].replace(" (m³/m³)", ""), f"{latest[f]:.3f}")

st.subheader("Last 7 days and next-hour forecast")
hist = df.loc[df.index > latest_time - pd.Timedelta(days=7)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist[target], name="Observed SM 20 cm", line=dict(color="#1d2939")))
fig.add_trace(go.Scatter(x=[latest_time + pd.Timedelta(hours=1)], y=[pred], mode="markers",
                         name=f"{MODEL_NAMES[name]} forecast", marker=dict(size=12, color="#e76f51")))
fig.add_hline(y=thr["threshold"], line_dash="dash", line_color="#b42318", annotation_text="threshold")
fig.update_layout(height=360, template="plotly_white", yaxis_title="VWC (m³/m³)", margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=hist.index, y=hist["P_CALC"], name="Precipitation (mm)", marker_color="#2a9d8f"))
fig2.add_trace(go.Scatter(x=hist.index, y=hist["T_HR_AVG"], name="Air temp (°C)", yaxis="y2",
                          line=dict(color="#f4a261")))
fig2.update_layout(height=300, template="plotly_white", yaxis=dict(title="mm"),
                   yaxis2=dict(title="°C", overlaying="y", side="right"), margin=dict(t=20, b=20))
st.plotly_chart(fig2, use_container_width=True)

mt = read_csv("results/metrics/test_metrics_table.csv")
if mt is not None:
    st.subheader("Held-out test performance (m³/m³)")
    st.dataframe(mt.set_index("model").style.format("{:.5f}"), use_container_width=True)
