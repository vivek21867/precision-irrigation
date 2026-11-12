"""Explainability - permutation importance, GradientSHAP and local explanations."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import FEATURE_LABELS, MODEL_COLORS, page_setup, read_csv, read_json, require

page_setup("Explainability", "🔍")
st.title("🔍 Explainability")
if not require(["results/metrics/permutation_importance.csv"],
               "Run `python -m src.explainability.feature_importance` first."):
    st.stop()

summ = read_json("results/metrics/explainability_summary.json") or {}
perm = read_csv("results/metrics/permutation_importance.csv")
perm["label"] = perm.feature.map(FEATURE_LABELS)

st.markdown("All importances are computed from the **trained models** on the **held-out test windows**.")
st.subheader("Permutation importance")
st.caption(f"Each variable's full 24-h input sequence is shuffled across {summ.get('permutation', {}).get('samples', '?')} "
           f"test windows ({summ.get('permutation', {}).get('repeats', '?')} repeats); bar = increase in test RMSE (m³/m³).")
fig = px.bar(perm, x="rmse_increase_mean", y="label", color="model", barmode="group", orientation="h",
             error_x="rmse_increase_std", color_discrete_map=MODEL_COLORS, template="plotly_white")
fig.update_layout(height=560, yaxis=dict(categoryorder="total ascending", title=""), xaxis_title="RMSE increase (m³/m³)")
st.plotly_chart(fig, use_container_width=True)

shap_f = read_csv("results/metrics/shap_mean_abs_by_feature.csv", index_col=0)
if shap_f is not None:
    st.subheader("GradientSHAP (expected gradients)")
    s = shap_f.rename(index=FEATURE_LABELS).reset_index(names="feature").melt(
        id_vars="feature", var_name="model", value_name="mean_abs_shap")
    fig = px.bar(s, x="mean_abs_shap", y="feature", color="model", barmode="group", orientation="h",
                 color_discrete_map=MODEL_COLORS, template="plotly_white")
    fig.update_layout(height=520, yaxis=dict(categoryorder="total ascending", title=""),
                      xaxis_title="mean |SHAP| summed over 24 lags (m³/m³)")
    st.plotly_chart(fig, use_container_width=True)

    m = st.selectbox("Lag × variable heatmap for", list(shap_f.columns))
    lag = read_csv(f"results/metrics/shap_lag_feature_{m.lower()}.csv", index_col=0)
    if lag is not None:
        fig = px.imshow(lag.rename(columns=FEATURE_LABELS), aspect="auto", color_continuous_scale="Viridis",
                        labels=dict(color="mean |SHAP|"))
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rows are input lags (t-0 = most recent hour before the forecast).")
else:
    st.info(f"SHAP results not available: {summ.get('shap')}")

st.subheader("Interpretation")
interp = read_json("results/reports/explainability_interpretation.json")
if interp:
    for line in interp.get("points", []):
        st.markdown(f"- {line}")
st.info("Per-prediction explanations (occlusion) are on the **Prediction** page.")
